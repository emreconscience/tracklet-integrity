"""
nis_anomaly_detector.py -- offline Kalman NIS anomaly detection per track.

Idea
    A constant-velocity Kalman filter is run over each finished track in field
    coordinates (metres). For every measurement the Normalized Innovation
    Squared is computed:

        NIS = y^T S^-1 y,   y = z - H x_pred,   S = H P_pred H^T + R

    Under the model NIS follows a chi-square distribution with 2 degrees of
    freedom (5.99 = 95 %, 9.21 = 99 %). A track that suddenly continues on a
    different person produces a burst of very large NIS values.

Method
    1. Take (frame_idx, x_field, y_field) of the track (real detections only
       when an `is_inference` column is present).
    2. State [x, vx, y, vy], measurement [x, y]; forward Kalman pass.
    3. Frames with NIS > 9.21 are anomalies; anomalies closer than
       NIS_CLUSTER_MAX_GAP frames are grouped, clusters shorter than
       NIS_CLUSTER_MIN_SIZE are dropped.
    4. The centre of the strongest cluster is the split candidate.

    This runs on the CSV after tracking, so it takes seconds per match instead
    of re-running the tracker.

Flag rule (used as one vote in chimera_detector.py)
    nis_flag = peak NIS of the strongest cluster >= NIS_STRONG_THRESHOLD (75)
               AND at least NIS_MIN_N_CLUSTERS_FOR_FLAG (2) clusters.
    A single isolated jump therefore never raises the flag by itself.

Public API
    detect_track_nis(df, track_id) -> dict  (see its docstring)
    compute_all_track_anomalies(df) -> {track_id: dict}
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Constants
# =============================================================================

# Nominal frame period (25 fps). Informational: the filter takes dt from
# consecutive frame_idx values and the `fps` argument.
DT_DEFAULT = 1.0 / 25.0

# Process noise: standard deviation of player acceleration (m/s^2).
SIGMA_A = 3.0

# Measurement noise: standard deviation of the field position (m); covers
# homography error and bounding-box jitter (~35 cm).
SIGMA_MEAS = 0.35

# Chi-square quantiles for 2 degrees of freedom (2-D position measurement).
CHI2_DOF2_95 = 5.99
CHI2_DOF2_99 = 9.21
CHI2_DOF2_999 = 13.82

# Flag thresholds. Lower peak thresholds (25, then 50) over-split real tracks:
# many clean tracks with brief detector jitter were flagged. At 75 only strong
# kinematic anomalies remain; the identity swaps seen during development
# typically peaked well above 200. (Tuned on the development match.)
NIS_STRONG_THRESHOLD = 75.0       # peak NIS of a cluster to count as "strong"
NIS_CLUSTER_MIN_SIZE = 5          # min anomalous frames in a cluster
NIS_CLUSTER_MAX_GAP = 15          # max frame gap inside one cluster
MIN_TRACK_FRAMES = 100            # shorter tracks are skipped (filter not settled)
NIS_MIN_N_CLUSTERS_FOR_FLAG = 2   # a single isolated jump does not raise the flag

# Factor for an adaptive median + k*MAD threshold. Currently unused: the
# detector uses the fixed chi-square threshold above.
NIS_MAD_FACTOR = 5.0


# =============================================================================
# Core: Kalman Filter per track
# =============================================================================


def _compute_track_nis(
    frames: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    fps: float = 25.0,
    sigma_a: float = SIGMA_A,
    sigma_meas: float = SIGMA_MEAS,
) -> List[Tuple[int, float]]:
    """
    Constant-velocity Kalman filter, compute NIS per measurement.

    State: x = [px, vx, py, vy]
    Measurement: z = [px, py]
    F(dt) = [[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]]
    H = [[1, 0, 0, 0], [0, 0, 1, 0]]
    Q(dt) = discrete white-noise acceleration (DWNA) block
    R = sigma_meas^2 * I2

    NIS = y^T * S^-1 * y
        y = z - H * x_pred (innovation)
        S = H * P_pred * H^T + R (innovation covariance)

    Args:
        frames: frame index array (sorted)
        xs: x_field array (meters)
        ys: y_field array (meters)
        fps: frame rate

    Returns:
        List of (frame_idx, nis_value). Empty if insufficient data.
    """
    n = len(frames)
    if n < 5:
        return []

    # Sort by frame (ensure monotonic)
    order = np.argsort(frames)
    fr = frames[order]
    xf = xs[order]
    yf = ys[order]

    # Initialize state from first 2 points
    dt0 = (fr[1] - fr[0]) / fps
    if dt0 <= 0 or dt0 > 2.0:
        dt0 = 1.0 / fps

    vx0 = (xf[1] - xf[0]) / dt0 if dt0 > 0 else 0.0
    vy0 = (yf[1] - yf[0]) / dt0 if dt0 > 0 else 0.0
    x = np.array([xf[0], vx0, yf[0], vy0], dtype=np.float64)

    # Initial covariance: high velocity uncertainty
    P = np.diag([sigma_meas ** 2, 25.0, sigma_meas ** 2, 25.0])

    H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    R = np.eye(2) * (sigma_meas ** 2)
    I4 = np.eye(4)

    nis_list: List[Tuple[int, float]] = []

    for k in range(1, n):
        dt = (fr[k] - fr[k - 1]) / fps
        if dt <= 0:
            continue
        if dt > 3.0:
            # Large gap — reinit
            x = np.array([xf[k], 0.0, yf[k], 0.0], dtype=np.float64)
            P = np.diag([sigma_meas ** 2, 25.0, sigma_meas ** 2, 25.0])
            continue

        # State transition
        F = np.array([
            [1.0, dt, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, dt],
            [0.0, 0.0, 0.0, 1.0],
        ])

        # Process noise (DWNA)
        sa2 = sigma_a ** 2
        dt2 = dt ** 2
        dt3 = dt ** 3
        dt4 = dt ** 4
        q_block = sa2 * np.array([[dt4 / 4, dt3 / 2], [dt3 / 2, dt2]])
        Q = np.zeros((4, 4))
        Q[0:2, 0:2] = q_block
        Q[2:4, 2:4] = q_block

        # Predict
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        # Innovation
        z = np.array([xf[k], yf[k]], dtype=np.float64)
        y_innov = z - H @ x_pred
        S = H @ P_pred @ H.T + R

        # NIS
        try:
            S_inv = np.linalg.inv(S)
            nis = float(y_innov @ S_inv @ y_innov)
        except np.linalg.LinAlgError:
            nis = 0.0

        nis_list.append((int(fr[k]), nis))

        # Update (standard Kalman)
        try:
            K = P_pred @ H.T @ S_inv
            x = x_pred + K @ y_innov
            IKH = I4 - K @ H
            P = IKH @ P_pred @ IKH.T + K @ R @ K.T  # Joseph form
        except Exception:
            x = x_pred
            P = P_pred

    return nis_list


# =============================================================================
# Anomaly clustering
# =============================================================================


def _cluster_anomalies(
    anomaly_frames: List[int],
    nis_map: Dict[int, float],
    max_gap: int = NIS_CLUSTER_MAX_GAP,
    min_size: int = NIS_CLUSTER_MIN_SIZE,
) -> List[Tuple[int, int, float]]:
    """
    Group anomalous frames into clusters (gap <= max_gap) and drop clusters
    with fewer than min_size frames.

    Returns:
        List[(start_frame, end_frame, peak_nis)]
    """
    if not anomaly_frames:
        return []

    af = sorted(anomaly_frames)
    clusters: List[List[int]] = []
    current = [af[0]]

    for f in af[1:]:
        if f - current[-1] <= max_gap:
            current.append(f)
        else:
            if len(current) >= min_size:
                clusters.append(current)
            current = [f]

    if len(current) >= min_size:
        clusters.append(current)

    result = []
    for c in clusters:
        peak = max(nis_map.get(f, 0.0) for f in c)
        result.append((c[0], c[-1], peak))

    return result


# =============================================================================
# Public API — per track
# =============================================================================


def detect_track_nis(
    df: pd.DataFrame,
    track_id,
    fps: float = 25.0,
    min_frames: int = MIN_TRACK_FRAMES,
    nis_threshold: float = CHI2_DOF2_99,
    cluster_min_size: int = NIS_CLUSTER_MIN_SIZE,
) -> Dict:
    """
    NIS anomaly analysis for one track.

    Returns dict with keys:
        nis_per_frame: List[(frame, nis)]
        anomaly_frames: List[frame]
        anomaly_clusters: List[(start, end, peak_nis)]
        max_nis: float
        n_anomalies: int
        anomaly_rate: float  (0..1)
        mean_nis: float
        median_nis: float
        split_candidate_frame: Optional[int]
        nis_flag: bool (identity-mix vote, see module docstring)
    """
    info = {
        "track_id": str(track_id),
        "nis_per_frame": [],
        "anomaly_frames": [],
        "anomaly_clusters": [],
        "max_nis": 0.0,
        "mean_nis": 0.0,
        "median_nis": 0.0,
        "n_anomalies": 0,
        "anomaly_rate": 0.0,
        "split_candidate_frame": None,
        "nis_flag": False,
        "n_frames_used": 0,
    }

    sub = df[df["track_id"] == track_id][["frame_idx", "x_field", "y_field"]].dropna()
    if len(sub) < min_frames:
        return info

    # Use real detections only when the column is present:
    #   is_inference == 1 -> the detector ran on this frame (real box)
    #   is_inference == 0 -> the row is a tracker (Kalman) prediction
    # Same convention as chimera_detector.py and build_reid_corpus.py.
    if "is_inference" in df.columns:
        is_real = df["is_inference"].astype(str).isin(["1", "1.0", "True", "true"])
        sub_real = df[(df["track_id"] == track_id) & is_real]
        sub_real = sub_real[["frame_idx", "x_field", "y_field"]].dropna()
        if len(sub_real) >= min_frames:
            sub = sub_real

    sub = sub.sort_values("frame_idx").reset_index(drop=True)

    frames = sub["frame_idx"].values.astype(np.int64)
    xs = sub["x_field"].values.astype(np.float64)
    ys = sub["y_field"].values.astype(np.float64)

    nis_list = _compute_track_nis(frames, xs, ys, fps=fps)
    if not nis_list:
        return info

    info["nis_per_frame"] = nis_list
    info["n_frames_used"] = len(nis_list)

    nis_values = np.array([v for _, v in nis_list])
    info["max_nis"] = float(nis_values.max())
    info["mean_nis"] = float(nis_values.mean())
    info["median_nis"] = float(np.median(nis_values))

    # Anomaly detection (hard threshold)
    anomaly_frames = [f for f, v in nis_list if v > nis_threshold]
    info["anomaly_frames"] = anomaly_frames
    info["n_anomalies"] = len(anomaly_frames)
    info["anomaly_rate"] = len(anomaly_frames) / len(nis_list)

    nis_map = {f: v for f, v in nis_list}
    clusters = _cluster_anomalies(
        anomaly_frames, nis_map,
        max_gap=NIS_CLUSTER_MAX_GAP,
        min_size=cluster_min_size,
    )
    info["anomaly_clusters"] = clusters

    # Strongest cluster = split candidate
    if clusters:
        strongest = max(clusters, key=lambda c: c[2])  # peak_nis
        center = (strongest[0] + strongest[1]) // 2
        info["split_candidate_frame"] = int(center)

        # Flag only when the strongest cluster is very anomalous AND the track
        # has more than one anomalous cluster: a single jump is more often a
        # detector glitch than a change of identity.
        n_clusters = len(clusters)
        if strongest[2] >= NIS_STRONG_THRESHOLD and n_clusters >= NIS_MIN_N_CLUSTERS_FOR_FLAG:
            info["nis_flag"] = True

    return info


def compute_all_track_anomalies(
    df: pd.DataFrame,
    fps: float = 25.0,
    min_frames: int = MIN_TRACK_FRAMES,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """
    Run detect_track_nis for every track; returns {track_id: info}.
    """
    results: Dict[str, Dict] = {}
    tids = sorted(df["track_id"].astype(str).unique())

    for tid in tids:
        info = detect_track_nis(df, tid, fps=fps, min_frames=min_frames)
        results[tid] = info

    if verbose:
        print(f"[NIS] {len(results)} tracks analysed")
        flagged = [t for t, i in results.items() if i["nis_flag"]]
        print(f"[NIS] Tracks with nis_flag=True: {len(flagged)}")
        for tid in sorted(flagged):
            info = results[tid]
            clusters = info["anomaly_clusters"]
            print(f"  {tid}: max_nis={info['max_nis']:.1f}, "
                  f"n_anomaly={info['n_anomalies']} ({info['anomaly_rate']:.1%}), "
                  f"clusters={len(clusters)}, "
                  f"split@{info['split_candidate_frame']}")
            for c in clusters[:3]:
                print(f"    cluster f{c[0]}-f{c[1]}, peak_nis={c[2]:.1f}")

    return results


# =============================================================================
# CLI
# =============================================================================


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Offline Kalman NIS anomaly detector")
    ap.add_argument("--csv", required=True, help="track CSV (see README for columns)")
    ap.add_argument("--fps", type=float, default=25.0,
                    help="frame rate of frame_idx (default: 25)")
    ap.add_argument("--min-frames", type=int, default=MIN_TRACK_FRAMES,
                    help=f"skip tracks with fewer rows (default: {MIN_TRACK_FRAMES})")
    ap.add_argument("--show-all", action="store_true",
                    help="list every analysed track, not only those with "
                         ">= --min-frames NIS values")
    args = ap.parse_args()

    print(f"Loading {args.csv}...")
    df = pd.read_csv(args.csv)
    df["track_id"] = df["track_id"].astype(str)
    print(f"  {len(df)} rows, {df['track_id'].nunique()} tracks")

    print(f"\n[NIS ANOMALY DETECTION]")
    print(f"  Chi2 DOF=2 threshold: {CHI2_DOF2_99:.2f} (99%)")
    print(f"  Strong cluster threshold: peak NIS >= {NIS_STRONG_THRESHOLD}")
    print(f"  Min cluster size: {NIS_CLUSTER_MIN_SIZE} consecutive frames")
    print(f"  Process noise sigma_a: {SIGMA_A} m/s^2")
    print(f"  Measurement noise sigma_meas: {SIGMA_MEAS} m")
    print()

    results = compute_all_track_anomalies(
        df, fps=args.fps, min_frames=args.min_frames, verbose=True
    )

    # All tracks summary (sorted by max_nis)
    print(f"\n=== ALL TRACKS (sorted by max_nis) ===")
    print(f"{'tid':12} {'n_used':>8} {'max_nis':>10} {'mean':>8} {'median':>8} "
          f"{'n_anom':>8} {'rate':>8} {'clusters':>10} {'flag':>6}")
    sorted_results = sorted(
        results.items(),
        key=lambda kv: kv[1]["max_nis"],
        reverse=True,
    )
    for tid, info in sorted_results:
        if info["n_frames_used"] == 0:
            continue
        flag_str = "[!]" if info["nis_flag"] else ""
        if not args.show_all and info["n_frames_used"] < args.min_frames:
            continue
        print(f"{tid:12} {info['n_frames_used']:>8} "
              f"{info['max_nis']:>10.2f} {info['mean_nis']:>8.2f} "
              f"{info['median_nis']:>8.2f} "
              f"{info['n_anomalies']:>8} {info['anomaly_rate']:>8.1%} "
              f"{len(info['anomaly_clusters']):>10} {flag_str:>6}")


if __name__ == "__main__":
    main()

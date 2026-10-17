"""
chimera_detector.py -- find and split identity-mixed ("chimera") tracks.

A chimera is a track whose ID stays the same while the tracker has switched
to a different person. Each track is scored with several independent signals
and is split only when enough of them agree, or when one of them shows a
physically impossible jump.

Per-track signals (see `_analyze_single_track`):
    hsv / reid   first-quarter vs last-quarter cosine distance of the
                 appearance vectors (`appearance`, `reid_embedding` columns)
    pelt         change points in the PCA-reduced appearance sequence
                 (ruptures.Pelt, RBF cost), accepted only if the segments are
                 separated (silhouette), applied recursively
    spatial_bi   positions form two far-apart clusters that are also
                 separated in time
    kin          smoothed position moves > 10 m over 15 detections on one
                 camera (< 100 frames apart)
    xcam         > 3 m jump between consecutive frames on different cameras
    nis          Kalman NIS flag from nis_anomaly_detector.py

CLI stages (`main`):
    1. score every track and collect candidates
    2. decide per candidate (optional VLM vote on before/after crops)
    3. write the CSV with split tracks renamed <id>, <id>_B, <id>_C, ...
       and a JSON report (chimera_report.json in the working directory)

`is_strong_split_candidate` is a stricter decision gate for use when the
module is imported into a larger multi-pass pipeline; the CLI does not call
it (see README).
"""
import cv2
import json
import numpy as np
import pandas as pd
import base64
import os
import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple

# --- Config ---
# All distances below are cosine distances in [0, 1] (higher = more different).
HSV_CHANGE_THRESHOLD = 0.4      # hsv vote: first vs last quarter HSV distance
HSV_STRONG_THRESHOLD = 0.6      # HSV distance that is enough to split on its own
REID_CHANGE_THRESHOLD = 0.5     # reid vote: first vs last quarter ReID distance
MIN_TRACK_FRAMES = 200          # tracks with fewer real detections are skipped
CANDIDATE_MIN_SIGNALS = 2       # CLI: votes needed to split (data signals + VLM)
VLM_MODEL = os.environ.get("VLM_MODEL", "")  # any OpenAI-compatible VLM; empty = VLM step skipped
VLM_MAX_TOKENS = 3000

# --- PELT recursive split config ---
# PELT runs on the filtered appearance sequence (only rows that carry an
# embedding), so sizes below count embeddings, not frames. With roughly one
# embedding every few frames, min_size=30 corresponds to a few seconds.
PELT_MIN_SIZE = 30              # min embeddings per segment
PELT_PEN_FACTOR = 2.0           # penalty = factor * log(N) (BIC-like); 1.0 over-segmented
PELT_MAX_DEPTH = 2              # recursion depth: at most 2 levels of splitting per pass;
                                # deeper recursion over-split, a later pass can continue
PELT_MIN_SILHOUETTE = 0.15      # min silhouette of the segmentation to accept a split
PELT_REID_PCA_DIM = 16          # within-track PCA dimension for ReID
PELT_HSV_PCA_DIM = 8            # within-track PCA dimension for HSV
PELT_MIN_FRAME_SPAN = 100       # change points closer than this (frames) are merged
                                # (100 frames = 4 s at 25 fps)
PELT_MAX_SPLITS_PER_PASS = 3    # cap on split points per track in one pass, keeps the
                                # number of fragments handed to later stages bounded

# --- Strict-gate calibration (used by is_strong_split_candidate) ---
# Without these guards the detector over-split real data: splits with
# silhouette ~0, or driven by a single spatial/NIS signal, produced many
# fragments of clean tracks. Values were tuned on the development match.
# They can be overridden at runtime with apply_chimera_cal_overrides().

# Minimum silhouette for any PELT-based split in the strict gate.
STRONG_SPLIT_MIN_SILHOUETTE = 0.50

# Minimum number of independent votes (appearance, spatial, kinematic, NIS)
# for the multi-signal path of the strict gate. A very large HSV change or a
# physically impossible jump is accepted on its own and bypasses this.
STRONG_SPLIT_MIN_SIGNALS = 3

# NIS and spatial_bi add their split frames to the PELT list, which sets
# pelt_flag. If True, a PELT split whose source is only 'nis' (no appearance
# or spatial evidence) is blocked unless enough other votes agree.
BLOCK_NIS_ONLY_PELT = True


def apply_chimera_cal_overrides(overrides: Dict) -> None:
    """
    Override the strict-gate calibration values at runtime.

    Example:
        from chimera_detector import apply_chimera_cal_overrides
        apply_chimera_cal_overrides({
            "STRONG_SPLIT_MIN_SILHOUETTE": 0.20,
            "STRONG_SPLIT_MIN_SIGNALS": 1,
            "BLOCK_NIS_ONLY_PELT": False,
        })

    Supported keys:
        STRONG_SPLIT_MIN_SILHOUETTE (float): default 0.50
        STRONG_SPLIT_MIN_SIGNALS (int):      default 3
        BLOCK_NIS_ONLY_PELT (bool):          default True
    Unknown keys and values that cannot be cast are skipped with a warning.
    """
    global STRONG_SPLIT_MIN_SILHOUETTE, STRONG_SPLIT_MIN_SIGNALS, BLOCK_NIS_ONLY_PELT
    _valid_keys = {
        "STRONG_SPLIT_MIN_SILHOUETTE": float,
        "STRONG_SPLIT_MIN_SIGNALS": int,
        "BLOCK_NIS_ONLY_PELT": bool,
    }
    for k, v in overrides.items():
        if k not in _valid_keys:
            print(f"  [GATE] WARNING: unknown override key '{k}', skipped")
            continue
        expected_type = _valid_keys[k]
        try:
            cast_v = expected_type(v)
        except (TypeError, ValueError):
            print(f"  [GATE] WARNING: '{k}'={v} has the wrong type, skipped")
            continue
        if k == "STRONG_SPLIT_MIN_SILHOUETTE":
            STRONG_SPLIT_MIN_SILHOUETTE = cast_v
        elif k == "STRONG_SPLIT_MIN_SIGNALS":
            STRONG_SPLIT_MIN_SIGNALS = cast_v
        elif k == "BLOCK_NIS_ONLY_PELT":
            BLOCK_NIS_ONLY_PELT = cast_v
        print(f"  [GATE] Override: {k} = {cast_v}")


# --- Kalman NIS vote (nis_anomaly_detector.py) ---
# Offline constant-velocity Kalman filter per track; NIS = y^T S^-1 y.
# Catches jumps the motion model rejects. If the module cannot be imported the
# vote is disabled; the fallback threshold below is then only used for logging.
try:
    from nis_anomaly_detector import (
        detect_track_nis,
        NIS_STRONG_THRESHOLD as _NIS_STRONG_THRESHOLD,
    )
    _HAS_NIS_DETECTOR = True
except Exception:
    detect_track_nis = None  # type: ignore
    _NIS_STRONG_THRESHOLD = 25.0
    _HAS_NIS_DETECTOR = False

NIS_FPS_DEFAULT = 25.0          # frame rate assumed for the Kalman dt

# Optional videos of the two cameras (also settable with --high / --low).
HIGH_VIDEO = os.environ.get("CHIMERA_HIGH_VIDEO", "")
LOW_VIDEO = os.environ.get("CHIMERA_LOW_VIDEO", "")


# =============================================================================
# Video HSV histogram cache
# =============================================================================
# When the detector is run several times (multi-pass), the same
# (camera, frame, bbox) crops would be decoded and histogrammed again. The
# cache keeps them in memory. Only used when video captures are supplied.
#
# Key:   (cam, frame_idx, x1, y1, x2, y2)  -- bbox in pixels
# Value: normalised HSV histogram (float32)
_HSV_HIST_CACHE: Dict[Tuple[str, int, int, int, int, int], np.ndarray] = {}
_HSV_CACHE_HITS = 0
_HSV_CACHE_MISSES = 0


def reset_hsv_cache() -> None:
    """Clear the HSV cache (call at the start of a new match)."""
    global _HSV_HIST_CACHE, _HSV_CACHE_HITS, _HSV_CACHE_MISSES
    _HSV_HIST_CACHE = {}
    _HSV_CACHE_HITS = 0
    _HSV_CACHE_MISSES = 0


def hsv_cache_stats() -> str:
    """Return cache hit/miss statistics as a string."""
    total = _HSV_CACHE_HITS + _HSV_CACHE_MISSES
    if total == 0:
        return "cache: empty"
    rate = _HSV_CACHE_HITS / total * 100.0
    n_entry = len(_HSV_HIST_CACHE)
    return (
        f"cache: {_HSV_CACHE_HITS}/{total} hit ({rate:.1f}%), "
        f"miss={_HSV_CACHE_MISSES}, {n_entry} entry"
    )


def _hsv_cache_get(cam: str, frame_idx: int, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """Cache lookup; returns the histogram (and counts a hit) or None."""
    global _HSV_CACHE_HITS
    key = (cam, frame_idx, bbox[0], bbox[1], bbox[2], bbox[3])
    cached = _HSV_HIST_CACHE.get(key)
    if cached is not None:
        _HSV_CACHE_HITS += 1
        return cached
    return None


def _hsv_cache_put(cam: str, frame_idx: int, bbox: Tuple[int, int, int, int],
                   hist: np.ndarray) -> None:
    """Store a histogram and count a miss."""
    global _HSV_CACHE_MISSES
    key = (cam, frame_idx, bbox[0], bbox[1], bbox[2], bbox[3])
    _HSV_HIST_CACHE[key] = hist
    _HSV_CACHE_MISSES += 1
# =============================================================================


def decode_embedding(b64_str: str, dtype=np.float16) -> Optional[np.ndarray]:
    """Decode a base64 string into a 1-D array (None on failure)."""
    try:
        return np.frombuffer(base64.b64decode(str(b64_str)), dtype=dtype)
    except Exception:
        return None


def compute_window_embedding(embeddings: List[np.ndarray]) -> Optional[np.ndarray]:
    """Mean of a list of vectors, L2-normalised (None for an empty list)."""
    if not embeddings:
        return None
    avg = np.mean(embeddings, axis=0).astype(np.float32)
    norm = np.linalg.norm(avg)
    if norm > 0:
        avg /= norm
    return avg


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance for unit vectors, clipped at 0 (0 = same direction)."""
    sim = float(np.dot(a, b))
    return max(0.0, 1.0 - sim)


# ============================================================
#  RECURSIVE PELT MULTI-CHANGEPOINT SPLITTER
# ============================================================
# Output: frame indices that cut a track into segments with different
# appearance. Signal: within-track PCA projection of one appearance channel
# (HSV or ReID), z-normalised. Algorithm: ruptures.Pelt(model="rbf"),
# validated by silhouette score, applied recursively to each segment.

def _znorm(arr: np.ndarray) -> np.ndarray:
    """Z-normalize per column (zero mean, unit std)."""
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    std = np.where(std > 1e-8, std, 1.0)
    return ((arr - mean) / std).astype(np.float32)


def _pca_project(features: np.ndarray, n_components: int) -> np.ndarray:
    """Within-track PCA projection. Returns (M, k) where k <= n_components."""
    M = features.shape[0]
    if M < 2:
        return np.zeros((M, n_components), dtype=np.float32)
    centered = features - features.mean(axis=0, keepdims=True)
    try:
        # Use SVD for stable PCA
        U, s, Vt = np.linalg.svd(centered, full_matrices=False)
        k = min(n_components, Vt.shape[0])
        proj = centered @ Vt[:k].T
        # Pad with zeros if k < n_components
        if k < n_components:
            pad = np.zeros((M, n_components - k), dtype=np.float32)
            proj = np.concatenate([proj, pad], axis=1)
        return proj.astype(np.float32)
    except np.linalg.LinAlgError:
        return np.zeros((M, n_components), dtype=np.float32)


def _build_appearance_block(track_rows: pd.DataFrame,
                            embed_col: str,
                            decode_dtype,
                            pca_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generic appearance signal builder. Filters to rows with embeddings only,
    decodes + L2-normalizes, then projects to within-track PCA basis.

    Returns:
        proj: (M, pca_dim) z-normalized PCA projection (M = filtered rows)
        frames: (M,) frame_idx for filtered rows
    """
    if embed_col not in track_rows.columns:
        return np.zeros((0, pca_dim), dtype=np.float32), np.zeros(0, dtype=np.int64)
    rows = track_rows[track_rows[embed_col].notna()].sort_values('frame_idx').reset_index(drop=True)
    M = len(rows)
    if M == 0:
        return np.zeros((0, pca_dim), dtype=np.float32), np.zeros(0, dtype=np.int64)

    embeds = []
    keep_idx = []
    for i in range(M):
        v = rows.iloc[i][embed_col]
        emb = decode_embedding(v, dtype=decode_dtype)
        if emb is None or emb.size == 0:
            continue
        nrm = np.linalg.norm(emb)
        if nrm > 1e-8:
            emb = emb.astype(np.float32) / nrm
        else:
            emb = emb.astype(np.float32)
        embeds.append(emb)
        keep_idx.append(i)

    if len(embeds) < max(2, pca_dim // 2):
        return np.zeros((0, pca_dim), dtype=np.float32), np.zeros(0, dtype=np.int64)

    rows = rows.iloc[keep_idx].reset_index(drop=True)
    mat = np.stack(embeds, axis=0)
    proj = _pca_project(mat, pca_dim)
    proj = _znorm(proj)
    frames = rows['frame_idx'].astype(np.int64).values
    return proj, frames


def build_hsv_signal(track_rows: pd.DataFrame,
                     pca_dim: int = PELT_HSV_PCA_DIM
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """HSV-only PELT signal (the strongest appearance signal on our data).
    The `appearance` column holds float16 bytes, base64-encoded; decoding it as
    float32 would silently produce garbage."""
    return _build_appearance_block(track_rows, 'appearance', np.float16, pca_dim)


def build_reid_signal(track_rows: pd.DataFrame,
                      pca_dim: int = PELT_REID_PCA_DIM
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """ReID-only PELT signal (`reid_embedding` column, float16, base64)."""
    return _build_appearance_block(track_rows, 'reid_embedding', np.float16, pca_dim)


# PCA dimensions tried in order. When the identity change is concentrated in
# the first 2-3 components, the remaining components dilute the silhouette;
# lower dimensions are therefore tried as a fallback, with stricter thresholds
# (see find_best_appearance_split). On the development match this found
# several additional swaps without new false positives on clean tracks.
PELT_HSV_PCA_DIM_CANDIDATES: Tuple[int, ...] = (8, 3, 2)
PELT_REID_PCA_DIM_CANDIDATES: Tuple[int, ...] = (8, 3, 2)


def detect_spatial_bimodality(track_rows: pd.DataFrame,
                              min_cluster_dist_m: float = 8.0,
                              min_minority_frac: float = 0.10,
                              min_temporal_separation: float = 0.30,
                              ) -> Tuple[bool, Optional[int], float]:
    """
    Spatial bimodality test on (x_field, y_field) for ID-swap detection.

    Hypothesis: one player's positions form a single (possibly wide) cloud.
    A track that switched from, say, the goalkeeper to a midfielder forms two
    clusters far apart. A swap is also a change over TIME: two clusters that
    are both visited during the whole track (a winger moving up and down) are
    normal play, not a swap.

    Algorithm:
        1. KMeans(k=2) on (x, y).
        2. Cluster centres at least `min_cluster_dist_m` apart.
        3. Minority cluster holds at least `min_minority_frac` of the rows.
        4. Temporal separation |mean_frame(C0) - mean_frame(C1)| / duration
           >= `min_temporal_separation`.
        5. Split frame: label changes after smoothing the label sequence
           (51-sample moving average); the change point closest to the middle
           of the track, ignoring the first and last 10 %, is returned.

    Example: x = 3.8 m (goalkeeper) before and x = 15.8 m after the swap gives
    d = 12 m and a temporal separation well above 0.3 -> bimodal -> split.

    Note: the caller (`_analyze_single_track`) uses stricter values
    (10 m, 20 %) than the defaults of this function.

    Returns:
        (is_bimodal, split_frame_or_None, cluster_distance_m)
    """
    if not {'x_field', 'y_field', 'frame_idx'}.issubset(track_rows.columns):
        return False, None, 0.0
    rows = track_rows[['frame_idx', 'x_field', 'y_field']].dropna().sort_values('frame_idx').reset_index(drop=True)
    if len(rows) < 100:
        return False, None, 0.0

    coords = rows[['x_field', 'y_field']].values.astype(np.float32)
    frames_arr = rows['frame_idx'].values.astype(np.int64)

    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=2, n_init=3, random_state=42).fit(coords)
        labels = km.labels_
        centers = km.cluster_centers_
    except Exception:
        return False, None, 0.0

    dx = float(centers[1, 0] - centers[0, 0])
    dy = float(centers[1, 1] - centers[0, 1])
    cluster_dist = float(np.sqrt(dx * dx + dy * dy))

    # Balance check: the minority cluster must not be negligible
    n_total = len(labels)
    n0 = int(np.sum(labels == 0))
    n1 = n_total - n0
    minority_frac = min(n0, n1) / max(n_total, 1)

    if cluster_dist < min_cluster_dist_m or minority_frac < min_minority_frac:
        return False, None, cluster_dist

    # Temporal separation: the clusters must occupy different parts of the
    # track's time span, not alternate throughout it
    f0_mean = float(frames_arr[labels == 0].mean())
    f1_mean = float(frames_arr[labels == 1].mean())
    track_duration = float(frames_arr.max() - frames_arr.min())
    if track_duration < 100:
        return False, None, cluster_dist
    temporal_sep = abs(f1_mean - f0_mean) / track_duration

    if temporal_sep < min_temporal_separation:
        # Two areas visited throughout the track: normal play
        # (e.g. a winger), not an identity swap.
        return False, None, cluster_dist

    # Split frame: smooth the label sequence (rolling majority, window 51)
    # and look for label changes
    try:
        from scipy.ndimage import uniform_filter1d
        smooth = uniform_filter1d(labels.astype(np.float32), size=51, mode='nearest')
        smooth_lbl = (smooth >= 0.5).astype(np.int32)
    except Exception:
        smooth_lbl = labels

    changes = np.where(np.diff(smooth_lbl) != 0)[0]
    if len(changes) == 0:
        # No change survives smoothing: the location of a possible swap is
        # undefined, so do not report one
        return False, None, cluster_dist

    # Pick the change closest to the middle; ignore the first and last 10 %
    best_split = None
    best_balance = float('inf')
    for ch in changes:
        if ch < n_total * 0.10 or ch > n_total * 0.90:
            continue
        balance = abs(ch - n_total // 2)
        if balance < best_balance:
            best_balance = balance
            best_split = int(frames_arr[ch])

    if best_split is None:
        return False, None, cluster_dist

    return True, best_split, cluster_dist


# ============================================================
#  VIDEO-BASED UPPER-BODY HSV SIGNAL (optional, needs the videos)
# ============================================================
# The CSV `appearance` vector is a coarse histogram of the whole box, mixed
# with legs, grass and shadow. This builder crops the UPPER half of the box
# (the shirt) from the video and computes a finer 16x8 H-S histogram. On the
# development match it separated several swaps the CSV histogram missed.
# Used only when `video_caps` is passed to detect_chimera_candidates.

def build_video_hsv_signal(track_rows: pd.DataFrame,
                           video_caps: Dict[str, "cv2.VideoCapture"],
                           lens_maps: Optional[Dict[str, Tuple]] = None,
                           fps_ratio: float = 1.0,
                           max_samples: int = 200,
                           pca_dim: int = PELT_HSV_PCA_DIM,
                           upper_ratio: float = 0.5,
                           h_bins: int = 16,
                           s_bins: int = 8,
                           cache_only: bool = False,
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Upper-body (shirt) HSV histograms from the track's video crops -> PCA ->
    PELT signal.

    Args:
        track_rows: all rows of the track (needs bbox_pixel, camera, frame_idx)
        video_caps: {'H': cv2.VideoCapture, 'L': cv2.VideoCapture}
        lens_maps: {'H': (m1, m2), 'L': (m1, m2)}; None = no undistortion
        fps_ratio: low_fps / high_fps (frame_idx is on the H timeline)
        max_samples: at most this many crops per track
        pca_dim: PCA output dimension
        upper_ratio: fraction of the box height kept from the top (0.5 = upper half)
        h_bins, s_bins: histogram bins (16 x 8 = 128)
        cache_only: read only from the cache, never touch the video

    Returns:
        proj: (M, pca_dim) z-normalized PCA projection
        frames: (M,) frame_idx for sampled rows
    """
    if 'bbox_pixel' not in track_rows.columns:
        return np.zeros((0, pca_dim), dtype=np.float32), np.zeros(0, dtype=np.int64)

    rows = track_rows.dropna(subset=['bbox_pixel']).sort_values('frame_idx').reset_index(drop=True)
    n_total = len(rows)
    if n_total < 60:
        return np.zeros((0, pca_dim), dtype=np.float32), np.zeros(0, dtype=np.int64)

    # Sample on the same global frame grid as the prefetch step so that the
    # crops are found in the cache.
    stride_mask = (rows['frame_idx'].astype(int) % GLOBAL_SAMPLE_STRIDE) == 0
    rows_strided = rows[stride_mask].reset_index(drop=True)

    if len(rows_strided) >= 10:
        # Enough samples on the grid (same logic as _collect_video_hsv_samples)
        if len(rows_strided) > max_samples:
            strided_pick = np.linspace(0, len(rows_strided) - 1, max_samples, dtype=int)
            picked_rows = [rows_strided.iloc[i] for i in strided_pick]
        else:
            picked_rows = [rows_strided.iloc[i] for i in range(len(rows_strided))]
    else:
        # Very short track: evenly spaced samples
        n_sample = min(n_total, max_samples)
        sample_idx = np.linspace(0, n_total - 1, n_sample, dtype=int)
        picked_rows = [rows.iloc[i] for i in sample_idx]

    feats: List[np.ndarray] = []
    sampled_frames: List[int] = []

    for row in picked_rows:
        try:
            bbox = [int(float(p)) for p in str(row['bbox_pixel']).split(',')]
        except Exception:
            continue
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            continue

        cam = str(row.get('camera', 'H'))
        fr = int(row['frame_idx'])

        # --- cache lookup ---
        cached = _hsv_cache_get(cam, fr, (x1, y1, x2, y2))
        if cached is not None:
            feats.append(cached)
            sampled_frames.append(fr)
            continue

        # cv2.VideoCapture is not thread-safe. Worker threads run with
        # cache_only=True and skip misses; the prefetch step on the main
        # thread has already filled the cache.
        if cache_only:
            continue

        cap = video_caps.get(cam)
        if cap is None or not cap.isOpened():
            continue

        seek = fr if cam == 'H' else int(round(fr * fps_ratio))

        cap.set(cv2.CAP_PROP_POS_FRAMES, seek)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        if lens_maps is not None:
            maps = lens_maps.get(cam)
            if maps is not None and maps[0] is not None:
                frame = cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR)

        h_f, w_f = frame.shape[:2]
        x1c = max(0, min(x1, w_f - 1))
        y1c = max(0, min(y1, h_f - 1))
        x2c = max(x1c + 1, min(x2, w_f))
        y2c = max(y1c + 1, min(y2, h_f))
        crop = frame[y1c:y2c, x1c:x2c]
        if crop.size == 0:
            continue

        h_box = crop.shape[0]
        upper_h = max(1, int(h_box * upper_ratio))
        upper = crop[:upper_h, :]
        if upper.size == 0:
            continue

        try:
            hsv = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [h_bins, s_bins], [0, 180, 0, 256])
            hist = hist.flatten()
            s = float(hist.sum())
            if s > 0:
                hist = hist / s
            hist = hist.astype(np.float32)
            _hsv_cache_put(cam, fr, (x1, y1, x2, y2), hist)
            feats.append(hist)
            sampled_frames.append(fr)
        except Exception:
            continue

    if len(feats) < max(2, pca_dim):
        return np.zeros((0, pca_dim), dtype=np.float32), np.zeros(0, dtype=np.int64)

    mat = np.stack(feats, axis=0)
    proj = _pca_project(mat, pca_dim)
    proj = _znorm(proj)
    frames_arr = np.array(sampled_frames, dtype=np.int64)
    return proj, frames_arr


# =============================================================================
# Sequential video prefetch
# =============================================================================
# Random seeks are slow (tens to hundreds of ms each); reading forward is much
# cheaper. The prefetch step:
#   1. collects the sample points of all tracks and keeps the cache misses,
#   2. sorts them by (camera, frame),
#   3. reads each camera forward, using grab() for short gaps and a seek for
#      long ones,
#   4. crops and histograms every box on a decoded frame,
#   5. stores the results in the HSV cache used by build_video_hsv_signal.
GRAB_SKIP_THRESHOLD = 30  # gaps up to this many frames are skipped with grab()


# Samples are taken on a global frame grid (frame_idx % stride == 0), so that
# the sub-tracks created by a split reuse the same cached crops.
# 5 frames = 0.2 s at 25 fps.
GLOBAL_SAMPLE_STRIDE = 5


def _collect_video_hsv_samples(
    df: pd.DataFrame,
    max_samples: int = 200,
    min_track_frames: int = 60,
) -> List[Tuple[str, int, Tuple[int, int, int, int]]]:
    """
    Collect (cam, frame_idx, bbox) sample points for every track.

    Frames are chosen on the GLOBAL_SAMPLE_STRIDE grid, so that when a track
    is split into _B/_C/... the sub-tracks pick the same frames and hit the
    HSV cache. Very short tracks fall back to evenly spaced samples.
    """
    if 'bbox_pixel' not in df.columns:
        return []

    out: List[Tuple[str, int, Tuple[int, int, int, int]]] = []
    for tid, sub in df.groupby('track_id'):
        rows = sub.dropna(subset=['bbox_pixel']).sort_values('frame_idx').reset_index(drop=True)
        n_total = len(rows)
        if n_total < min_track_frames:
            continue

        # Keep frames on the global grid
        stride_mask = (rows['frame_idx'].astype(int) % GLOBAL_SAMPLE_STRIDE) == 0
        rows_strided = rows[stride_mask].reset_index(drop=True)

        # Too few grid samples (very short track): evenly spaced samples
        if len(rows_strided) < 10:
            n_sample = min(n_total, max_samples)
            sample_idx = np.linspace(0, n_total - 1, n_sample, dtype=int)
            picked_rows = [rows.iloc[i] for i in sample_idx]
        else:
            # Too many samples: thin out evenly
            if len(rows_strided) > max_samples:
                strided_idx = np.linspace(0, len(rows_strided) - 1, max_samples, dtype=int)
                picked_rows = [rows_strided.iloc[i] for i in strided_idx]
            else:
                picked_rows = [rows_strided.iloc[i] for i in range(len(rows_strided))]

        for row in picked_rows:
            try:
                bbox = tuple(int(float(p)) for p in str(row['bbox_pixel']).split(','))
            except Exception:
                continue
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            if x2 <= x1 or y2 <= y1:
                continue
            cam = str(row.get('camera', 'H'))
            fr = int(row['frame_idx'])
            out.append((cam, fr, (x1, y1, x2, y2)))
    return out


def prefetch_video_hsv_for_tracks(
    df: pd.DataFrame,
    video_caps: Dict[str, "cv2.VideoCapture"],
    lens_maps: Optional[Dict[str, Tuple]] = None,
    fps_ratio: float = 1.0,
    max_samples: int = 200,
    upper_ratio: float = 0.5,
    h_bins: int = 16,
    s_bins: int = 8,
    verbose: bool = False,
) -> int:
    """
    Fill the HSV cache for all tracks with one forward pass per camera.

    Samples already in the cache are skipped. Short gaps between needed
    frames are skipped with grab() instead of seeking.

    Returns:
        number of new cache entries.
    """
    if not video_caps:
        return 0

    samples = _collect_video_hsv_samples(df, max_samples=max_samples)
    if not samples:
        return 0

    # Skip samples that are already cached
    pending: List[Tuple[str, int, Tuple[int, int, int, int]]] = []
    for cam, fr, bbox in samples:
        if _hsv_cache_get(cam, fr, bbox) is None:
            pending.append((cam, fr, bbox))

    if not pending:
        if verbose:
            print(f"  [PREFETCH] all {len(samples)} samples already cached")
        return 0

    # One forward pass per camera
    n_added = 0
    for cam in sorted(set(s[0] for s in pending)):
        cap = video_caps.get(cam)
        if cap is None or not cap.isOpened():
            continue

        cam_samples = [(fr, bbox) for c, fr, bbox in pending if c == cam]
        # (seek_target, fr, bbox); camera L is mapped with fps_ratio
        cam_seeks = []
        for fr, bbox in cam_samples:
            seek = fr if cam == 'H' else int(round(fr * fps_ratio))
            cam_seeks.append((seek, fr, bbox))
        cam_seeks.sort(key=lambda x: x[0])

        # Lens maps
        maps = None
        if lens_maps is not None:
            maps = lens_maps.get(cam)

        # Decode only when the frame changes; several boxes may share a frame
        current_pos = -1  # capture position (index of the next frame)
        current_frame = None  # last decoded frame (BGR)
        current_seek = -2  # frame index that current_frame belongs to

        for seek, fr, bbox in cam_seeks:
            if seek != current_seek:
                gap = seek - current_pos
                if 0 < gap <= GRAB_SKIP_THRESHOLD and current_pos >= 0:
                    # Short gap: advance with grab() (no decoding)
                    for _ in range(gap - 1):
                        if not cap.grab():
                            break
                    ret, frame_data = cap.read()
                else:
                    # Long gap or first frame: seek
                    cap.set(cv2.CAP_PROP_POS_FRAMES, seek)
                    ret, frame_data = cap.read()

                current_pos = seek + 1  # position after read()
                current_seek = seek

                if not ret or frame_data is None:
                    current_frame = None
                    continue

                # Undistort once per frame; all boxes on it share the result
                if maps is not None and maps[0] is not None:
                    frame_data = cv2.remap(frame_data, maps[0], maps[1], cv2.INTER_LINEAR)
                current_frame = frame_data

            if current_frame is None:
                continue

            # Crop the upper body and histogram it
            x1, y1, x2, y2 = bbox
            h_f, w_f = current_frame.shape[:2]
            x1c = max(0, min(x1, w_f - 1))
            y1c = max(0, min(y1, h_f - 1))
            x2c = max(x1c + 1, min(x2, w_f))
            y2c = max(y1c + 1, min(y2, h_f))
            crop = current_frame[y1c:y2c, x1c:x2c]
            if crop.size == 0:
                continue

            h_box = crop.shape[0]
            upper_h = max(1, int(h_box * upper_ratio))
            upper = crop[:upper_h, :]
            if upper.size == 0:
                continue

            try:
                hsv = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [h_bins, s_bins], [0, 180, 0, 256])
                hist = hist.flatten()
                s_sum = float(hist.sum())
                if s_sum > 0:
                    hist = hist / s_sum
                hist = hist.astype(np.float32)
                _hsv_cache_put(cam, fr, bbox, hist)
                n_added += 1
            except Exception:
                continue

    if verbose:
        print(f"  [PREFETCH] {n_added}/{len(pending)} samples loaded")
    return n_added
# =============================================================================


def find_change_points_pelt(signal: np.ndarray,
                            min_size: int = PELT_MIN_SIZE,
                            pen_factor: float = PELT_PEN_FACTOR,
                            jump: int = 2
                            ) -> List[int]:
    """
    Run PELT (Pruned Exact Linear Time) change point detection on a multivariate signal.

    Args:
        signal:     (N, D) feature matrix (FILTERED, no forward-fill)
        min_size:   minimum embedding count per segment
        pen_factor: BIC-like penalty multiplier (calibrated 2.0 = robust)
        jump:       PELT subsampling step (lower = more precise, higher = faster)

    Returns:
        bkps: list of array indices where breakpoints occur (excluding the final length)
    """
    try:
        import ruptures as rpt
    except ImportError:
        return []

    N = len(signal)
    if N < 2 * min_size:
        return []

    # Penalty: BIC-like, pen = pen_factor * log(N). On a manually checked
    # development track, 2.0 recovered the visually verified change points;
    # 1.0 over-segmented (silhouette < 0).
    pen = pen_factor * float(np.log(max(N, 2)))

    try:
        algo = rpt.Pelt(model="rbf", min_size=min_size, jump=jump).fit(signal)
        bkps = algo.predict(pen=pen)
        # ruptures appends N as last bkpt — drop it
        if bkps and bkps[-1] >= N:
            bkps = bkps[:-1]
        return [int(b) for b in bkps if 0 < b < N]
    except Exception:
        return []


def silhouette_score_segments(signal: np.ndarray, bkps: List[int]) -> float:
    """
    Compute silhouette score for the segmentation defined by bkps.

    Args:
        signal: (N, D) feature matrix
        bkps:   list of internal breakpoints (NOT including N)

    Returns:
        score in [-1, 1]; higher = better separation; 0 if invalid.
    """
    if not bkps:
        return 0.0
    N = len(signal)
    labels = np.zeros(N, dtype=np.int32)
    seg_starts = [0] + list(bkps)
    seg_ends = list(bkps) + [N]
    for label, (s, e) in enumerate(zip(seg_starts, seg_ends)):
        labels[s:e] = label
    if len(np.unique(labels)) < 2:
        return 0.0
    try:
        from sklearn.metrics import silhouette_score as _sil
        # Subsample for speed if very long
        if N > 1500:
            rs = np.random.RandomState(42)
            idx = rs.choice(N, 1500, replace=False)
            return float(_sil(signal[idx], labels[idx], metric='euclidean'))
        return float(_sil(signal, labels, metric='euclidean'))
    except Exception:
        # Fallback: between/(between+within) variance ratio (Calinski-Harabasz-ish)
        unique = np.unique(labels)
        within = 0.0
        between = 0.0
        global_mean = signal.mean(axis=0)
        for lbl in unique:
            mask = labels == lbl
            if mask.sum() == 0:
                continue
            seg_mean = signal[mask].mean(axis=0)
            within += float(np.sum((signal[mask] - seg_mean) ** 2))
            between += float(mask.sum() * np.sum((seg_mean - global_mean) ** 2))
        if within + between > 0:
            return float(between / (within + between))
        return 0.0


def recursive_split_track(signal: np.ndarray,
                          frames: np.ndarray,
                          depth: int = 0,
                          max_depth: int = PELT_MAX_DEPTH,
                          min_size: int = PELT_MIN_SIZE,
                          min_silhouette: float = PELT_MIN_SILHOUETTE,
                          pen_factor: float = PELT_PEN_FACTOR,
                          min_frame_span: int = PELT_MIN_FRAME_SPAN
                          ) -> Tuple[List[int], float]:
    """
    Recursively split a track signal until no more change points found.

    Strategy:
        1. Run PELT on the full signal to find first-level breakpoints.
        2. Validate with silhouette score (>= min_silhouette).
        3. For each resulting segment, recurse if size >= 2*min_size.

    Args:
        signal:         (N, D) FILTERED signal (no forward-fill)
        frames:         (N,) corresponding frame_idx values
        min_frame_span: minimum frame_idx distance between consecutive splits (eliminates
                        index-space-close but frame-space-far false neighbors)

    Returns:
        split_frames: sorted list of frame_idx values where the track must be split
        max_silhouette: best silhouette score across all levels (for reporting)
    """
    if depth >= max_depth or len(signal) < 2 * min_size:
        return [], 0.0

    bkps = find_change_points_pelt(signal, min_size=min_size, pen_factor=pen_factor)
    if not bkps:
        return [], 0.0

    sil = silhouette_score_segments(signal, bkps)
    if sil < min_silhouette:
        return [], sil

    split_frames: List[int] = []
    max_sil = sil

    # Convert bkp indices to frame_idx values (these become _B/_C/_D split points)
    for b in bkps:
        if 0 < b < len(frames):
            split_frames.append(int(frames[b]))

    # Recurse into each sub-segment
    seg_starts = [0] + list(bkps)
    seg_ends = list(bkps) + [len(signal)]
    for s, e in zip(seg_starts, seg_ends):
        if (e - s) < 2 * min_size:
            continue
        sub_sig = signal[s:e]
        sub_frm = frames[s:e]
        sub_splits, sub_sil = recursive_split_track(
            sub_sig, sub_frm,
            depth=depth + 1, max_depth=max_depth,
            min_size=min_size, min_silhouette=min_silhouette,
            pen_factor=pen_factor, min_frame_span=min_frame_span,
        )
        if sub_splits:
            split_frames.extend(sub_splits)
        if sub_sil > max_sil:
            max_sil = sub_sil

    # Sort + dedupe + minimum frame span filter (drop bkps that are too close in frame_idx)
    split_frames = sorted(set(split_frames))
    if min_frame_span > 0 and len(split_frames) > 1:
        thinned = [split_frames[0]]
        for sf in split_frames[1:]:
            if sf - thinned[-1] >= min_frame_span:
                thinned.append(sf)
        split_frames = thinned
    return split_frames, max_sil


def find_best_appearance_split(
    track_rows: pd.DataFrame,
    embed_col: str,
    decode_dtype,
    pca_dim_candidates: Tuple[int, ...] = (8,),
    min_size: int = PELT_MIN_SIZE,
    pen_factor: float = PELT_PEN_FACTOR,
    min_silhouette: float = PELT_MIN_SILHOUETTE,
    max_depth: int = PELT_MAX_DEPTH,
) -> Tuple[List[int], float, int]:
    """
    Try recursive_split_track on the primary PCA dim (first candidate).
    Fallback to lower dims ONLY if primary returns nothing, with progressively
    STRICTER thresholds to compensate for the dim-bias inflation of silhouette
    in lower-dim spaces.

    Calibration:
        - 8d: threshold 0.15 (production baseline)
        - 3d: threshold 0.25 (compensate for ~1.5-2x dim-bias inflation)
        - 2d: threshold 0.35 (compensate for ~2-3x dim-bias inflation)

    This catches swaps whose 8-d silhouette is just below the threshold
    without accepting spurious low-dimensional splits.

    Returns:
        (splits, silhouette, pca_dim_used)
    """
    fallback_thresholds = {2: 0.35, 3: 0.25, 4: 0.20, 8: min_silhouette}
    for idx, d in enumerate(pca_dim_candidates):
        sig, frm = _build_appearance_block(track_rows, embed_col, decode_dtype, d)
        if len(sig) < 2 * min_size:
            continue
        thresh = min_silhouette if idx == 0 else fallback_thresholds.get(d, 0.25)
        splits, sil = recursive_split_track(
            sig, frm,
            depth=0, max_depth=max_depth,
            min_size=min_size, min_silhouette=thresh,
            pen_factor=pen_factor,
        )
        if splits:
            return splits, sil, d
    return [], 0.0, 0


def _analyze_single_track(
    tid: str,
    df: pd.DataFrame,
    use_video: bool,
    video_caps: Optional[Dict],
    lens_maps: Optional[Dict],
    fps_ratio: float,
    verbose: bool,
    cache_only_video: bool = False,
    enable_nis: bool = True,
) -> Optional[Dict]:
    """
    Compute all identity-mix signals for one track.

    Does not modify shared state apart from the HSV cache (plain dict get/put),
    so it can run in worker threads. With cache_only_video=True the video is
    never read (cv2.VideoCapture is not thread-safe); only cached crops are used.

    Returns:
        a candidate dict if at least one signal fired (signal_count >= 1 or
        pelt_flag), otherwise None (track looks clean).
    """
    sub = df[df.track_id == tid].sort_values('frame_idx')
    # is_inference == 1 -> the detector ran on this frame (real box);
    # 0 -> tracker (Kalman) prediction. Only real detections are analysed.
    real = sub[sub['is_inference'].astype(str).isin(["1", "1.0", "True", "true"])] \
        if 'is_inference' in sub.columns else sub

    if len(real) < MIN_TRACK_FRAMES:
        return None

    # --- 0) Recursive PELT change points on appearance ---
    # HSV first, then ReID; ReID change points further than PELT_MIN_FRAME_SPAN
    # from existing ones are added. PCA dims are tried as in
    # find_best_appearance_split (8, then 3 and 2 as stricter fallbacks).
    pelt_split_frames: List[int] = []
    pelt_silhouette = 0.0
    pelt_source = None
    try:
        hsv_splits, hsv_sil, hsv_dim = find_best_appearance_split(
            real, 'appearance', np.float16, PELT_HSV_PCA_DIM_CANDIDATES,
            min_size=PELT_MIN_SIZE, pen_factor=PELT_PEN_FACTOR,
            min_silhouette=PELT_MIN_SILHOUETTE, max_depth=PELT_MAX_DEPTH,
        )
        if hsv_splits:
            pelt_split_frames = hsv_splits
            pelt_silhouette = hsv_sil
            pelt_source = f'hsv(d{hsv_dim})'

        reid_splits, reid_sil, reid_dim = find_best_appearance_split(
            real, 'reid_embedding', np.float16, PELT_REID_PCA_DIM_CANDIDATES,
            min_size=PELT_MIN_SIZE, pen_factor=PELT_PEN_FACTOR,
            min_silhouette=PELT_MIN_SILHOUETTE, max_depth=PELT_MAX_DEPTH,
        )
        if reid_splits:
            if not pelt_split_frames:
                pelt_split_frames = reid_splits
                pelt_silhouette = reid_sil
                pelt_source = f'reid(d{reid_dim})'
            else:
                for rs in reid_splits:
                    if not any(abs(rs - hs) <= PELT_MIN_FRAME_SPAN for hs in pelt_split_frames):
                        pelt_split_frames.append(rs)
                pelt_split_frames = sorted(set(pelt_split_frames))
                if reid_sil > pelt_silhouette:
                    pelt_silhouette = reid_sil
                pelt_source = (pelt_source or '') + f'+reid(d{reid_dim})'

        if use_video:
            try:
                vhsv_sig, vhsv_frm = build_video_hsv_signal(
                    real, video_caps, lens_maps, fps_ratio,
                    cache_only=cache_only_video,
                )
                if len(vhsv_sig) >= 2 * PELT_MIN_SIZE:
                    vhsv_splits, vhsv_sil = recursive_split_track(
                        vhsv_sig, vhsv_frm,
                        depth=0, max_depth=PELT_MAX_DEPTH,
                        min_size=PELT_MIN_SIZE,
                        min_silhouette=PELT_MIN_SILHOUETTE,
                        pen_factor=PELT_PEN_FACTOR,
                    )
                    if vhsv_splits:
                        if not pelt_split_frames:
                            pelt_split_frames = vhsv_splits
                            pelt_silhouette = vhsv_sil
                            pelt_source = 'video_hsv'
                        else:
                            for vs in vhsv_splits:
                                if not any(abs(vs - es) <= PELT_MIN_FRAME_SPAN for es in pelt_split_frames):
                                    pelt_split_frames.append(vs)
                            pelt_split_frames = sorted(set(pelt_split_frames))
                            if vhsv_sil > pelt_silhouette:
                                pelt_silhouette = vhsv_sil
                            pelt_source = (pelt_source or '') + '+video_hsv'
            except Exception:
                pass

    except Exception:
        pass

    # --- 0d) Spatial bimodality ---
    # Catches swaps between similarly dressed players (same team) that the
    # appearance signals miss. Stricter than the function defaults: centres
    # >= 10 m apart and minority cluster >= 20 %.
    spatial_bimodal = False
    spatial_split = None
    spatial_dist = 0.0
    try:
        spatial_bimodal, spatial_split, spatial_dist = detect_spatial_bimodality(
            real, min_cluster_dist_m=10.0, min_minority_frac=0.20
        )
        if spatial_bimodal and spatial_split is not None:
            if not any(abs(int(s) - int(spatial_split)) <= PELT_MIN_FRAME_SPAN for s in pelt_split_frames):
                pelt_split_frames.append(int(spatial_split))
                pelt_split_frames = sorted(set(pelt_split_frames))
            pelt_source = (pelt_source or '') + '+spatial_bi' if pelt_source else 'spatial_bi'
    except Exception:
        pass

    # --- 0e) Kalman NIS vote ---
    # See nis_anomaly_detector.py. If the track is flagged, the centres of all
    # clusters with peak NIS >= NIS_STRONG_THRESHOLD become split candidates.
    # Note: detect_track_nis receives the full DataFrame and applies its own
    # real-detection filter.
    nis_flag = False
    nis_max = 0.0
    nis_n_clusters = 0
    nis_anomaly_rate = 0.0
    nis_split_frame = None
    if _HAS_NIS_DETECTOR and enable_nis:
        try:
            nis_info = detect_track_nis(df, tid, fps=NIS_FPS_DEFAULT)
            nis_max = float(nis_info.get("max_nis", 0.0))
            nis_n_clusters = len(nis_info.get("anomaly_clusters", []) or [])
            nis_anomaly_rate = float(nis_info.get("anomaly_rate", 0.0))
            nis_flag = bool(nis_info.get("nis_flag", False))
            nis_split_frame = nis_info.get("split_candidate_frame")

            if nis_flag and nis_split_frame is not None:
                ns = int(nis_split_frame)
                # Merge into pelt_split_frames (same dedup window)
                if not any(abs(int(s) - ns) <= PELT_MIN_FRAME_SPAN for s in pelt_split_frames):
                    pelt_split_frames.append(ns)
                    pelt_split_frames = sorted(set(pelt_split_frames))
                # Also merge all strong cluster centers (not just the strongest)
                for c in (nis_info.get("anomaly_clusters") or []):
                    try:
                        cstart, cend, cpeak = c[0], c[1], c[2]
                    except Exception:
                        continue
                    if cpeak < _NIS_STRONG_THRESHOLD:
                        continue
                    center = int((int(cstart) + int(cend)) // 2)
                    if not any(abs(int(s) - center) <= PELT_MIN_FRAME_SPAN for s in pelt_split_frames):
                        pelt_split_frames.append(center)
                pelt_split_frames = sorted(set(pelt_split_frames))
                pelt_source = (pelt_source or '') + '+nis' if pelt_source else 'nis'
        except Exception:
            pass

    # Defaults
    best_split_frame: Optional[int] = None
    best_split_score: float = 0.0

    # --- 1) HSV vote: first vs last quarter of the track ---
    # Also scans the middle half with a sliding window for the frame of
    # maximum change (used as split frame when PELT found none).
    hsv_scores: List[float] = []
    if 'appearance' in real.columns:
        app_rows = real[real['appearance'].notna()]
        if len(app_rows) >= 10:
            app_embeds = []
            for _, row in app_rows.iterrows():
                emb = decode_embedding(row['appearance'], dtype=np.float16)
                if emb is not None:
                    app_embeds.append((int(row['frame_idx']), emb))
            if len(app_embeds) >= 10:
                n = len(app_embeds)
                q1 = [e for _, e in app_embeds[:n//4]]
                q4 = [e for _, e in app_embeds[3*n//4:]]
                q1_avg = compute_window_embedding(q1)
                q4_avg = compute_window_embedding(q4)
                if q1_avg is not None and q4_avg is not None:
                    hsv_dist = cosine_distance(q1_avg, q4_avg)
                    hsv_scores.append(hsv_dist)
                    step = max(1, len(app_embeds) // 20)
                    for i in range(n//4, 3*n//4, step):
                        left = [e for _, e in app_embeds[max(0,i-n//8):i]]
                        right = [e for _, e in app_embeds[i:min(n,i+n//8)]]
                        l_avg = compute_window_embedding(left)
                        r_avg = compute_window_embedding(right)
                        if l_avg is not None and r_avg is not None:
                            d = cosine_distance(l_avg, r_avg)
                            if d > best_split_score:
                                best_split_score = d
                                best_split_frame = app_embeds[i][0]

    # --- 2) ReID vote: same procedure on the ReID embeddings ---
    reid_scores: List[float] = []
    if 'reid_embedding' in real.columns:
        reid_rows = real[real['reid_embedding'].notna()]
        if len(reid_rows) >= 10:
            reid_embeds = []
            for _, row in reid_rows.iterrows():
                emb = decode_embedding(row['reid_embedding'], dtype=np.float16)
                if emb is not None:
                    reid_embeds.append((int(row['frame_idx']), emb))
            if len(reid_embeds) >= 10:
                n = len(reid_embeds)
                q1 = [e for _, e in reid_embeds[:n//4]]
                q4 = [e for _, e in reid_embeds[3*n//4:]]
                q1_avg = compute_window_embedding(q1)
                q4_avg = compute_window_embedding(q4)
                if q1_avg is not None and q4_avg is not None:
                    reid_dist = cosine_distance(q1_avg, q4_avg)
                    reid_scores.append(reid_dist)
                    if best_split_frame is None:
                        step = max(1, len(reid_embeds) // 20)
                        for i in range(n//4, 3*n//4, step):
                            left = [e for _, e in reid_embeds[max(0,i-n//8):i]]
                            right = [e for _, e in reid_embeds[i:min(n,i+n//8)]]
                            l_avg = compute_window_embedding(left)
                            r_avg = compute_window_embedding(right)
                            if l_avg is not None and r_avg is not None:
                                d = cosine_distance(l_avg, r_avg)
                                if d > best_split_score:
                                    best_split_score = d
                                    best_split_frame = reid_embeds[i][0]

    # --- 3) Same-camera kinematic jump ---
    # Positions are smoothed with a 15-sample rolling mean and compared 15
    # samples apart. At 25 fps 15 frames are 0.6 s; a sprinting player
    # (~25 km/h = 6.9 m/s) covers ~4 m, so > 10 m (within < 100 frames) is
    # treated as a jump to another person.
    kin_flag = False
    kin_jump_m = 0.0
    kin_split_frame = None
    if {'x_field', 'y_field', 'camera'}.issubset(real.columns):
        for cam in ['H', 'L']:
            cam_sub = real[real['camera'] == cam].sort_values('frame_idx')
            if len(cam_sub) < 15:
                continue
            xs = cam_sub['x_field'].rolling(15, min_periods=5).mean().values
            ys = cam_sub['y_field'].rolling(15, min_periods=5).mean().values
            cam_frames = cam_sub['frame_idx'].values
            for i in range(15, len(xs)):
                if np.isnan(xs[i]) or np.isnan(xs[i-15]):
                    continue
                dx = xs[i] - xs[i-15]
                dy = ys[i] - ys[i-15]
                d = np.sqrt(dx*dx + dy*dy)
                fgap = cam_frames[i] - cam_frames[i-15]
                if fgap <= 0:
                    continue
                if d > 10.0 and fgap < 100:
                    if d > kin_jump_m:
                        kin_jump_m = d
                        kin_flag = True
                        kin_split_frame = int(cam_frames[i - 7])

    # --- 3b) Cross-camera jump ---
    # A track can follow one player on camera L and a different player on
    # camera H. The same-camera check above misses this. Here: two
    # consecutive rows (<= 3 frames apart) on different cameras whose
    # positions differ by > 3 m (3 m in 40 ms is ~270 km/h).
    xcam_flag = False
    xcam_jump_m = 0.0
    xcam_split_frame = None
    XCAM_JUMP_THRESHOLD = 3.0  # metres; a genuine camera handover is typically < 2 m
    XCAM_MAX_GAP = 3            # max frame gap between the two rows (0.12 s)
    if {'x_field', 'y_field', 'camera'}.issubset(real.columns):
        sorted_real = real.sort_values('frame_idx').reset_index(drop=True)
        for i in range(1, len(sorted_real)):
            cam_curr = sorted_real.iloc[i]['camera']
            cam_prev = sorted_real.iloc[i-1]['camera']
            if cam_curr == cam_prev:
                continue
            fgap = int(sorted_real.iloc[i]['frame_idx']) - int(sorted_real.iloc[i-1]['frame_idx'])
            if fgap <= 0 or fgap > XCAM_MAX_GAP:
                continue
            dx = sorted_real.iloc[i]['x_field'] - sorted_real.iloc[i-1]['x_field']
            dy = sorted_real.iloc[i]['y_field'] - sorted_real.iloc[i-1]['y_field']
            d = float(np.sqrt(dx*dx + dy*dy))
            if d > XCAM_JUMP_THRESHOLD and d > xcam_jump_m:
                xcam_jump_m = d
                xcam_flag = True
                xcam_split_frame = int(sorted_real.iloc[i]['frame_idx'])

    # --- 4) Combine the votes ---
    hsv_flag = len(hsv_scores) > 0 and max(hsv_scores) > HSV_CHANGE_THRESHOLD
    reid_flag = len(reid_scores) > 0 and max(reid_scores) > REID_CHANGE_THRESHOLD
    pelt_flag = len(pelt_split_frames) > 0
    spatial_flag = bool(spatial_bimodal)
    # spatial_bi and nis also append their split frames to pelt_split_frames,
    # which sets pelt_flag. Counting pelt_flag as a vote would count those
    # signals twice, so PELT only votes when an appearance source
    # (hsv / reid / video_hsv) produced change points.
    _src = (pelt_source or '').lower()
    _pelt_appearance_only = any(s in _src for s in ['hsv', 'reid', 'video_hsv'])
    pelt_independent = pelt_flag and _pelt_appearance_only

    signal_count = (
        int(hsv_flag) + int(reid_flag) + int(kin_flag)
        + int(pelt_independent) + int(spatial_flag) + int(nis_flag)
        + int(xcam_flag)
    )

    merged_splits = list(pelt_split_frames)
    if kin_flag and kin_split_frame is not None:
        if not any(abs(int(s) - int(kin_split_frame)) <= 50 for s in merged_splits):
            merged_splits.append(kin_split_frame)
    if xcam_flag and xcam_split_frame is not None:
        if not any(abs(int(s) - int(xcam_split_frame)) <= 50 for s in merged_splits):
            merged_splits.append(xcam_split_frame)
    if (hsv_flag or reid_flag) and best_split_frame is not None and not pelt_split_frames:
        if not any(abs(int(s) - int(best_split_frame)) <= 50 for s in merged_splits):
            merged_splits.append(best_split_frame)
    merged_splits = sorted(set(int(s) for s in merged_splits))

    # Segment-mean drift guard.
    # An appearance change point with no change in position is usually pose or
    # lighting drift rather than a new person. For each split frame compare the
    # mean position in the 60 frames before and after it:
    #   shift <  MEAN_DRIFT_REJECT_M -> reject the split (drift)
    #   shift >= MEAN_DRIFT_REJECT_M -> keep it
    # (A per-frame delta was tried first; it missed slow swaps where two
    # players were close at the moment of the switch.)
    # Splits within 15 frames of a KIN / NIS / spatial_bi centre are always
    # kept, since those signals already show a physical discontinuity.
    MEAN_WINDOW_FRAMES = 60
    MEAN_DRIFT_REJECT_M = 1.5  # metres
    if merged_splits and {'x_field', 'y_field', 'camera', 'frame_idx'}.issubset(real.columns):
        _kept_splits: List[int] = []
        _rejected_splits: List[int] = []
        # KIN / NIS / spatial_bi centres bypass the guard
        _hard_signal_centers: List[int] = []
        if kin_flag and kin_split_frame is not None:
            _hard_signal_centers.append(int(kin_split_frame))
        if nis_flag and nis_info is not None:
            _ns = nis_info.get('split_candidate_frame')
            if _ns is not None:
                _hard_signal_centers.append(int(_ns))
            # NOTE: the NIS result stores clusters under 'anomaly_clusters',
            # so this loop currently adds nothing; kept as is to preserve
            # the tuned behaviour.
            for c in (nis_info.get('clusters') or []):
                try:
                    _hard_signal_centers.append(int((int(c[0]) + int(c[1])) // 2))
                except Exception:
                    pass
        if spatial_flag and spatial_split is not None:
            _hard_signal_centers.append(int(spatial_split))

        _real_sorted = real[['frame_idx', 'x_field', 'y_field']].sort_values('frame_idx').reset_index(drop=True)
        _fi = _real_sorted.frame_idx.values
        _xv = _real_sorted.x_field.values
        _yv = _real_sorted.y_field.values

        for _sf in merged_splits:
            # Near a KIN / NIS / spatial_bi centre (+-15 frames): keep
            _is_hard = any(abs(_sf - hc) <= 15 for hc in _hard_signal_centers)
            if _is_hard:
                _kept_splits.append(_sf)
                continue

            # Before window: [sf - W, sf), After window: [sf, sf + W]
            _bef = (_fi >= _sf - MEAN_WINDOW_FRAMES) & (_fi < _sf)
            _aft = (_fi >= _sf) & (_fi <= _sf + MEAN_WINDOW_FRAMES)
            if _bef.sum() < 5 or _aft.sum() < 5:
                # too few samples to judge: keep
                _kept_splits.append(_sf)
                continue
            _x_b = float(np.mean(_xv[_bef])); _y_b = float(np.mean(_yv[_bef]))
            _x_a = float(np.mean(_xv[_aft])); _y_a = float(np.mean(_yv[_aft]))
            _mean_drift = ((_x_a - _x_b) ** 2 + (_y_a - _y_b) ** 2) ** 0.5
            if _mean_drift < MEAN_DRIFT_REJECT_M:
                _rejected_splits.append((_sf, _mean_drift))
            else:
                _kept_splits.append(_sf)

        if _rejected_splits:
            _rej_str = ", ".join(f"f{f}({d:.1f}m)" for f, d in _rejected_splits)
            print(f"  [SEG-DRIFT-GUARD] {tid}: {len(_rejected_splits)} drift split(s) rejected: {_rej_str}")
        merged_splits = _kept_splits
        # If every split was rejected, drop the PELT vote and recount.
        # (This recount does not include xcam_flag; kept as tuned.)
        if not merged_splits and (hsv_flag or reid_flag or nis_flag or kin_flag or spatial_flag):
            pelt_flag = False
            pelt_independent = False
            signal_count = (
                int(hsv_flag) + int(reid_flag) + int(kin_flag)
                + int(pelt_independent) + int(spatial_flag) + int(nis_flag)
            )

    # Cap the number of split points per pass; a later pass can split
    # further. This keeps the number of fragments bounded for later stages.
    if len(merged_splits) > PELT_MAX_SPLITS_PER_PASS:
        # Keep K points spread evenly over the list: divide indices 0..N-1
        # into K equal bins and take the middle of each.
        N = len(merged_splits)
        K = PELT_MAX_SPLITS_PER_PASS
        idxs = [int((i + 0.5) * N / K) for i in range(K)]
        merged_splits = sorted(set(merged_splits[i] for i in idxs if 0 <= i < N))

    if not best_split_frame and merged_splits:
        best_split_frame = merged_splits[0]

    # Largest detection gap at any split point. A long gap right before a
    # split (the player left the view and someone else was picked up under the
    # same ID) is strong evidence even when the new segment is short; the
    # strict gate uses it to bypass its small-minority guard.
    # The split frame starts the new segment; the gap is measured between the
    # last present frame before it and the first frame at/after it.
    max_split_gap_frames = 0
    if merged_splits and len(real) > 1:
        import bisect
        present_frames = sorted(set(int(f) for f in real.frame_idx.values))
        gaps_at_splits = []
        for sf in merged_splits:
            sf_int = int(sf)
            # idx_after: insertion index of the split frame
            idx_after = bisect.bisect_left(present_frames, sf_int)
            if 0 < idx_after < len(present_frames):
                gap = present_frames[idx_after] - present_frames[idx_after - 1]
                gaps_at_splits.append(gap)
        if gaps_at_splits:
            max_split_gap_frames = int(max(gaps_at_splits))

    if signal_count >= 1 or pelt_flag:
        return {
            "track_id": tid,
            "hsv_dist": round(max(hsv_scores) if hsv_scores else 0, 4),
            "reid_dist": round(max(reid_scores) if reid_scores else 0, 4),
            "kin_jump_m": round(kin_jump_m, 2),
            "xcam_jump_m": round(xcam_jump_m, 2),
            "xcam_split_frame": xcam_split_frame,
            "spatial_dist_m": round(spatial_dist, 2),
            "nis_max": round(nis_max, 2),
            "nis_n_clusters": int(nis_n_clusters),
            "nis_anomaly_rate": round(nis_anomaly_rate, 4),
            "hsv_flag": hsv_flag,
            "reid_flag": reid_flag,
            "kin_flag": kin_flag,
            "xcam_flag": xcam_flag,
            "pelt_flag": pelt_flag,
            "spatial_flag": spatial_flag,
            "nis_flag": nis_flag,
            "pelt_source": pelt_source,
            "pelt_silhouette": round(float(pelt_silhouette), 4),
            "signal_count": signal_count,
            "split_frame": best_split_frame,
            "split_frames": merged_splits,
            "split_score": round(best_split_score, 4),
            "frame_range": (int(real.frame_idx.min()), int(real.frame_idx.max())),
            "total_frames": len(real),
            "max_split_gap_frames": max_split_gap_frames,
        }
    return None


def detect_chimera_candidates(df: pd.DataFrame,
                              verbose: bool = True,
                              video_caps: Optional[Dict] = None,
                              lens_maps: Optional[Dict] = None,
                              fps_ratio: float = 1.0,
                              max_workers: Optional[int] = None,
                              enable_nis: bool = True,
                              ) -> List[Dict]:
    """
    Score every track and return the list of candidate dicts.

    A track can contain more than two people, so all split points are
    reported in `split_frames`; `split_frame` holds a single representative
    frame (used by the CLI log and report).

    Args:
        df: track DataFrame (see README for columns)
        verbose: print per-candidate details
        video_caps: optional {'H': cap_h, 'L': cap_l}; enables the upper-body
            video HSV signal (stronger than the CSV histogram)
        lens_maps: optional {'H': (m1, m2), 'L': (m1, m2)} undistortion maps
        fps_ratio: low_fps / high_fps
        max_workers: worker threads (None -> os.cpu_count()); threads are used
            only for more than 4 tracks
        enable_nis: include the Kalman NIS vote
    """
    candidates: List[Dict] = []
    track_ids = df['track_id'].unique()
    use_video = video_caps is not None and len(video_caps) > 0
    if verbose and use_video:
        print(f"  [PELT] video HSV signal enabled (caps: {list(video_caps.keys())})")
    if verbose:
        if _HAS_NIS_DETECTOR and enable_nis:
            print(f"  [NIS] Kalman NIS vote enabled (strong>={_NIS_STRONG_THRESHOLD})")
        elif _HAS_NIS_DETECTOR and not enable_nis:
            print(f"  [NIS] Kalman NIS vote disabled (enable_nis=False)")
        else:
            print(f"  [NIS] Kalman NIS vote unavailable (nis_anomaly_detector not importable)")
        # Strict-gate settings (used by is_strong_split_candidate, not by the CLI)
        print(f"  [GATE] min_silhouette={STRONG_SPLIT_MIN_SILHOUETTE:.2f}, "
              f"min_signals={STRONG_SPLIT_MIN_SIGNALS}, "
              f"block_nis_only={BLOCK_NIS_ONLY_PELT}")

    # === Sequential video prefetch ========================================
    # Load all needed crops into the HSV cache with one forward pass per
    # camera instead of random seeks per track.
    if use_video:
        try:
            t_pf = time.time()
            n_prefetch = prefetch_video_hsv_for_tracks(
                df, video_caps, lens_maps=lens_maps,
                fps_ratio=fps_ratio, verbose=verbose,
            )
            if verbose and n_prefetch > 0:
                print(f"  [PREFETCH] {n_prefetch} samples in {time.time()-t_pf:.1f}s")
        except Exception as _pe:
            if verbose:
                print(f"  [PREFETCH-ERROR] {_pe}")
    # ======================================================================

    # === Per-track analysis, optionally multi-threaded ====================
    # After the prefetch, the analysis only reads the cache and the DataFrame;
    # the heavy parts (numpy SVD, ruptures, OpenCV) release the GIL, so threads
    # help. Threads rather than processes: no spawn cost on Windows and the
    # module-level cache is shared.
    sorted_ids = sorted(track_ids)
    if max_workers is None:
        try:
            max_workers = max(1, (os.cpu_count() or 4))
        except Exception:
            max_workers = 4

    use_threads = max_workers > 1 and len(sorted_ids) > 4
    t_pelt = time.time()

    if use_threads:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        if verbose:
            print(f"  [PELT-MT] analysing {len(sorted_ids)} tracks with {max_workers} threads")
        # cv2.VideoCapture is not thread-safe: workers only read the cache
        # (cache_only=True), which the prefetch filled on the main thread.
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_to_tid = {
                ex.submit(
                    _analyze_single_track, tid, df, use_video,
                    video_caps, lens_maps, fps_ratio, verbose,
                    True,  # cache_only_video=True (thread-safety)
                    enable_nis,
                ): tid for tid in sorted_ids
            }
            for fut in as_completed(future_to_tid):
                tid = future_to_tid[fut]
                try:
                    cand = fut.result()
                except Exception as _e:
                    if verbose:
                        print(f"  [PELT-ERROR] {tid}: {_e}")
                    continue
                if cand is not None:
                    candidates.append(cand)
                    if verbose:
                        flags = []
                        if cand['hsv_flag']: flags.append(f"HSV={cand['hsv_dist']:.3f}")
                        if cand['reid_flag']: flags.append(f"ReID={cand['reid_dist']:.3f}")
                        if cand['kin_flag']: flags.append(f"KIN={cand['kin_jump_m']:.1f}m")
                        if cand.get('xcam_flag'):
                            flags.append(f"XCAM={cand.get('xcam_jump_m', 0):.1f}m")
                        if cand.get('nis_flag'):
                            flags.append(
                                f"NIS={cand.get('nis_max', 0):.0f}"
                                f"({cand.get('nis_n_clusters', 0)}c)"
                            )
                        if cand['pelt_flag']:
                            src = cand.get('pelt_source') or 'pelt'
                            flags.append(
                                f"PELT[{src}]={len(cand['split_frames'])}cp"
                                f"(sil={cand['pelt_silhouette']:.2f})"
                            )
                        gap_f = int(cand.get('max_split_gap_frames', 0) or 0)
                        gap_str = f" gap={gap_f}f" if gap_f >= 30 else ""
                        pretty = ",".join(f"f{s}" for s in cand['split_frames'][:5])
                        more = "" if len(cand['split_frames']) <= 5 else f"+{len(cand['split_frames'])-5}"
                        print(f"  [CANDIDATE] {tid}: {' + '.join(flags)}, splits=[{pretty}{more}]{gap_str}")
        # Sort for deterministic output
        candidates.sort(key=lambda c: c['track_id'])
    else:
        # Single-threaded path (few tracks / debugging)
        for tid in sorted_ids:
            try:
                cand = _analyze_single_track(
                    tid, df, use_video, video_caps, lens_maps, fps_ratio, verbose,
                    cache_only_video=False, enable_nis=enable_nis,
                )
            except Exception as _e:
                if verbose:
                    print(f"  [PELT-ERROR] {tid}: {_e}")
                continue
            if cand is not None:
                candidates.append(cand)
                if verbose:
                    flags = []
                    if cand['hsv_flag']: flags.append(f"HSV={cand['hsv_dist']:.3f}")
                    if cand['reid_flag']: flags.append(f"ReID={cand['reid_dist']:.3f}")
                    if cand['kin_flag']: flags.append(f"KIN={cand['kin_jump_m']:.1f}m")
                    if cand.get('xcam_flag'):
                        flags.append(f"XCAM={cand.get('xcam_jump_m', 0):.1f}m")
                    if cand.get('nis_flag'):
                        flags.append(
                            f"NIS={cand.get('nis_max', 0):.0f}"
                            f"({cand.get('nis_n_clusters', 0)}c)"
                        )
                    if cand['pelt_flag']:
                        src = cand.get('pelt_source') or 'pelt'
                        flags.append(
                            f"PELT[{src}]={len(cand['split_frames'])}cp"
                            f"(sil={cand['pelt_silhouette']:.2f})"
                        )
                    pretty = ",".join(f"f{s}" for s in cand['split_frames'][:5])
                    more = "" if len(cand['split_frames']) <= 5 else f"+{len(cand['split_frames'])-5}"
                    print(f"  [CANDIDATE] {tid}: {' + '.join(flags)}, splits=[{pretty}{more}]")

    if verbose:
        print(f"  [PELT] {len(candidates)} candidates found in {time.time()-t_pelt:.1f}s")

    return candidates


def vlm_confirm_split(client, track_id: str, split_frame: int,
                      df: pd.DataFrame, caps: Dict, undistort_maps: Dict,
                      fps_ratio: float) -> Optional[bool]:
    """
    Ask a vision-language model whether crops from before and after the split
    frame show the same person.

    Returns True (different people -> split confirmed), False (same person)
    or None (no usable answer). The prompt and the expected JSON keys
    ("ayni_kisi" = same person, "guven" = confidence, "aciklama" =
    explanation) are in Turkish; the parser below depends on these keys.
    """
    sub = df[df.track_id == track_id].sort_values('frame_idx')
    sub_with_bbox = sub[sub.bbox_pixel.notna()]

    # Rows with a box before / after the split frame
    before = sub_with_bbox[sub_with_bbox.frame_idx < split_frame]
    after = sub_with_bbox[sub_with_bbox.frame_idx >= split_frame]

    if len(before) < 3 or len(after) < 3:
        return None

    # One crop from the middle of each side
    before_row = before.iloc[len(before)//2]
    after_row = after.iloc[len(after)//2]

    crop_before = extract_crop(before_row, caps, undistort_maps, fps_ratio)
    crop_after = extract_crop(after_row, caps, undistort_maps, fps_ratio)

    if crop_before is None or crop_after is None:
        return None

    b64_before = base64.b64encode(cv2.imencode('.jpg', crop_before)[1]).decode()
    b64_after = base64.b64encode(cv2.imencode('.jpg', crop_after)[1]).decode()

    # Prompt (Turkish): "Two crops of track <id> from different times. Are
    # these the same person? Look carefully at clothing colour, build and hair
    # colour. Answer ONLY with JSON: {same_person, confidence, explanation}."
    content = [
        {"type": "text", "text": (
            f"Track {track_id} icin iki farkli zamandan crop:\n"
            f"Foto 1: frame {int(before_row.frame_idx)} (oncesi)\n"
            f"Foto 2: frame {int(after_row.frame_idx)} (sonrasi)\n\n"
            "Bu iki foto AYNI kisi mi? Kiyafet rengi, vucut yapisi, sac rengine dikkatlice bak.\n"
            "SADECE JSON cevap ver:\n"
            '{"ayni_kisi": true/false, "guven": 0.0-1.0, "aciklama": "..."}'
        )},
        {"type": "text", "text": "\nFoto 1 (oncesi):"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_before}"}},
        {"type": "text", "text": "\nFoto 2 (sonrasi):"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_after}"}},
    ]

    try:
        resp = client.chat.completions.create(
            model=VLM_MODEL,
            messages=[{"role": "user", "content": content}],
            max_tokens=VLM_MAX_TOKENS,
            temperature=0.1,
        )
        text = resp.choices[0].message.content
        if text:
            parsed = _parse_json(text)
            if parsed:
                is_same = parsed.get("ayni_kisi", True)
                conf = parsed.get("guven", 0.5)
                desc = parsed.get("aciklama", "")
                print(f"    VLM: {'SAME' if is_same else 'DIFFERENT'} (confidence={conf}): {desc}")
                return not is_same  # True = different people = split confirmed
            else:
                # No parsable JSON: fall back to keywords (Turkish / English)
                lower = text.lower()
                if "farkli" in lower or "different" in lower or '"ayni_kisi": false' in lower:
                    print(f"    VLM (text): DIFFERENT")
                    return True
                elif "ayni" in lower or "same" in lower or '"ayni_kisi": true' in lower:
                    print(f"    VLM (text): SAME")
                    return False
        print(f"    VLM: no answer")
        return None
    except Exception as e:
        print(f"    VLM ERROR: {e}")
        return None


def extract_crop(row, caps, undistort_maps, fps_ratio):
    """Read the frame of one row from the video and return a padded crop
    resized to 150 px height (None on failure)."""
    cam = str(row.get('camera', 'H'))
    bbox_str = str(row.get('bbox_pixel', ''))

    try:
        x1, y1, x2, y2 = [int(float(p)) for p in bbox_str.split(',')]
    except:
        return None

    cap = caps.get(cam)
    if cap is None:
        return None

    frame_idx = int(row['frame_idx'])
    seek_frame = frame_idx
    if cam == 'L':
        seek_frame = int(round(frame_idx * fps_ratio))

    cap.set(cv2.CAP_PROP_POS_FRAMES, seek_frame)
    ret, frame = cap.read()
    if not ret or frame is None:
        return None

    # Undistort (bbox_pixel is in undistorted coordinates)
    maps = undistort_maps.get(cam)
    if maps is not None and maps[0] is not None:
        frame = cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR)

    # Crop
    h, w = frame.shape[:2]
    pad_x = int((x2 - x1) * 0.1)
    pad_y = int((y2 - y1) * 0.15)
    cy1 = max(0, y1 - pad_y)
    cx1 = max(0, x1 - pad_x)
    cx2 = min(w, x2 + pad_x)
    cy2 = min(h, y2 + int((y2-y1)*0.05))

    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None

    # Resize
    scale = 150 / max(crop.shape[0], 1)
    return cv2.resize(crop, None, fx=scale, fy=scale)


def _parse_json(text):
    """Extract the first JSON object from a VLM answer (None if there is none)."""
    if not text:
        return None
    try:
        return json.loads(text)
    except:
        pass
    import re
    m = re.search(r'\{[^{}]*"ayni_kisi"[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except:
            pass
    m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except:
            pass
    return None


def compute_undistort_maps(cap, k1):
    """Undistortion maps for a single radial coefficient k1.

    Simple pinhole model with focal length = image width and the principal
    point at the image centre; alpha=0.8 for the new camera matrix. This must
    match the model used when bbox_pixel was produced. Returns (None, None)
    when |k1| < 0.001 (no undistortion)."""
    if abs(k1) < 0.001:
        return None, None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    focal = w
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array([k1, 0, 0, 0, 0], dtype=np.float64)
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0.8, (w, h))
    m1, m2 = cv2.initUndistortRectifyMap(K, dist, None, new_K, (w, h), cv2.CV_16SC2)
    return m1, m2


def _split_suffix(idx: int) -> str:
    """0->_B, 1->_C, ..., 24->_Z, 25->_AA, 26->_AB ..."""
    if idx < 25:
        return "_" + chr(ord('B') + idx)
    # Beyond Z: _AA, _AB ...
    first = idx - 25
    return f"_{chr(ord('A') + first // 25)}{chr(ord('B') + first % 25)}"


# --- Strict-gate PELT thresholds (is_strong_split_candidate, paths P4/P5) ---
# Silhouette needed for a PELT split, by the lowest PCA dimension involved.
# Silhouette is inflated in fewer dimensions (on synthetic data roughly 1.5x
# at 3-d and 2x at 2-d compared with 8-d), so the bar is raised accordingly.
# A tighter 8-d value (0.25) left more mixed tracks unsplit on the
# development match, so the looser value is combined with the minority guard.
PELT_STRONG_SILHOUETTE = 0.20      # 8-d
PELT_STRONG_SILHOUETTE_D3 = 0.30   # 3-d (0.20 * 1.5)
PELT_STRONG_SILHOUETTE_D2 = 0.40   # 2-d (0.20 * 2)
# Minority guard: if the smallest segment is < 15 % of the track and the
# silhouette is < 0.30, the "change" is more likely a lighting/pose episode
# than a second person -> reject. A balanced swap passes even with a
# borderline silhouette.
PELT_MIN_MINORITY_FRAC = 0.15
PELT_MINORITY_OVERRIDE_SIL = 0.30
# ...unless there is a detection gap of >= 50 frames at the split: the player
# left the view and someone else was picked up under the same ID, so a short
# segment is a genuine new identity.
PELT_GAP_BYPASS_FRAMES = 50
# Threshold for splits whose PELT source includes the video upper-body HSV
# signal (it targets the shirt colour directly).
PELT_STRONG_SILHOUETTE_VIDEO = 0.20


def _dim_aware_strong_threshold(pelt_source: str) -> float:
    """Silhouette threshold for a PELT source string, by its lowest PCA dim.

    Example source: 'hsv(d8)+reid(d3)+spatial_bi+nis' -> lowest dim 3 -> 0.30.
        d >= 8 -> 0.20,  3 <= d < 8 -> 0.30,  d < 3 -> 0.40
        no dimension in the string (nis / spatial_bi / video_hsv only) -> 0.20
    """
    if not pelt_source:
        return PELT_STRONG_SILHOUETTE
    src = pelt_source.lower()
    import re
    dims = [int(m) for m in re.findall(r'd(\d+)', src)]
    if not dims:
        return PELT_STRONG_SILHOUETTE
    min_dim = min(dims)
    if min_dim >= 8:
        return PELT_STRONG_SILHOUETTE       # 0.20
    elif min_dim >= 3:
        return PELT_STRONG_SILHOUETTE_D3    # 0.30
    else:  # d2
        return PELT_STRONG_SILHOUETTE_D2    # 0.40


def _min_segment_fraction(c: Dict) -> float:
    """Length of the shortest segment as a fraction of the track's frame span.

    split_frames=[600, 1500, 3000], frame_range=(0, 5000):
        segments (0,600) (600,1500) (1500,3000) (3000,5000)
        sizes 600, 900, 1500, 2000 -> 600 / 5000 = 0.12

    Returns 1.0 (guard passes) when there are no split frames or no range.
    """
    splits = c.get('split_frames') or []
    fr = c.get('frame_range')
    if not splits or not fr or len(fr) != 2:
        return 1.0
    start, end = int(fr[0]), int(fr[1])
    total = max(end - start, 1)
    bps = sorted(int(b) for b in splits if start < int(b) < end)
    if not bps:
        return 1.0
    boundaries = [start] + bps + [end]
    sizes = [boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)]
    return min(sizes) / total


def is_strong_split_candidate(c: Dict) -> bool:
    """
    Strict decision gate: should this candidate be split without further review?

    Intended for use when the module is imported into a larger, multi-pass
    pipeline. The CLI in this file uses a simpler rule (see `main`).
    Thresholds were tuned on one development match.

    Guards (evaluated first; any of them can reject):
      G1  Silhouette floor. If PELT contributed (pelt_flag) and
          pelt_silhouette < STRONG_SPLIT_MIN_SILHOUETTE (0.50): reject, unless
          one of these bypasses holds: HSV distance >= 0.60, same-camera jump
          >= 15 m, cross-camera jump >= 3 m, >= 3 votes, or nis_max >= 2000
          with nis_flag.
      G2  NIS-only PELT. If BLOCK_NIS_ONLY_PELT and the PELT source has
          neither appearance nor spatial_bi: reject when
          signal_count < STRONG_SPLIT_MIN_SIGNALS (bypass: nis_max >= 2000).
      G4  Many fragments. With >= 2 split frames and silhouette < 0.65:
          reject, unless HSV / same-camera / cross-camera evidence is strong;
          with nis_max >= 2000 the candidate is reduced to its middle split
          frame instead (this mutates c['split_frames']).

    Accept paths (first one that holds):
      P1  signal_count >= STRONG_SPLIT_MIN_SIGNALS (3)
      P2  HSV first-vs-last-quarter distance >= HSV_STRONG_THRESHOLD (0.60)
      P3  same-camera jump >= 15 m (kin_flag)
      P3b cross-camera jump >= 3 m (xcam_flag)
      P4  PELT silhouette >= dimension-aware threshold
          (_dim_aware_strong_threshold: 0.20 / 0.30 / 0.40)
      P5  PELT source includes video_hsv and silhouette >=
          PELT_STRONG_SILHOUETTE_VIDEO
    P4 and P5 apply the minority guard: shortest segment < 15 % of the track
    and silhouette < 0.30 -> reject, unless the detection gap at the split is
    >= PELT_GAP_BYPASS_FRAMES. Because of G1, P4/P5 in practice need a
    silhouette >= 0.50 unless a G1 bypass applies.
    """
    tid = c.get('track_id', '?')
    sil = c.get('pelt_silhouette', 0.0)
    src = c.get('pelt_source') or ''
    sig_count = c.get('signal_count', 0)

    # --- G1: silhouette floor for every candidate with PELT involvement ---
    # A silhouette near 0 means the PELT segments are not separated.
    # Strong physical evidence, >= 3 votes or an extreme NIS bypass the floor.
    if c.get('pelt_flag', False) and sil < STRONG_SPLIT_MIN_SILHOUETTE:
        _bypass_hsv = c.get('hsv_dist', 0.0) >= HSV_STRONG_THRESHOLD
        _bypass_kin = c.get('kin_flag', False) and c.get('kin_jump_m', 0.0) >= 15.0
        _bypass_xcam = c.get('xcam_flag', False) and c.get('xcam_jump_m', 0.0) >= 3.0
        _bypass_multi = sig_count >= 3
        # NIS >= 2000: no single person moves like that
        _nis_extreme = c.get('nis_flag', False) and c.get('nis_max', 0) >= 2000
        if not _bypass_hsv and not _bypass_kin and not _bypass_xcam and not _bypass_multi and not _nis_extreme:
            print(f"  [GATE] {tid}: REJECTED — pelt_silhouette={sil:.2f} < "
                  f"STRONG_SPLIT_MIN_SILHOUETTE={STRONG_SPLIT_MIN_SILHOUETTE:.2f} "
                  f"(src={src})")
            return False
        if _bypass_multi:
            print(f"  [GATE] {tid}: G1 BYPASS — multi-signal override "
                  f"(sig_count={sig_count}, sil={sil:.2f})")
        if _nis_extreme:
            print(f"  [GATE] {tid}: G1 BYPASS — extreme NIS "
                  f"(nis_max={c.get('nis_max', 0)}, sil={sil:.2f})")

    # --- G2: NIS-only PELT ---
    # NIS is purely kinematic; without appearance or spatial support it
    # over-split clean tracks during development.
    if BLOCK_NIS_ONLY_PELT and c.get('pelt_flag', False):
        _src_lower = src.lower()
        _has_appearance = any(s in _src_lower for s in ['hsv', 'reid', 'video_hsv'])
        _has_spatial = 'spatial_bi' in _src_lower
        if not _has_appearance and not _has_spatial:
            # Extreme NIS bypass: NIS>2000 is physically impossible for one person
            _nis_extreme_g2 = c.get('nis_flag', False) and c.get('nis_max', 0) >= 2000
            if _nis_extreme_g2:
                print(f"  [GATE] {tid}: G2 BYPASS — extreme NIS "
                      f"(nis_max={c.get('nis_max', 0)}, overrides NIS-only block)")
            elif sig_count < STRONG_SPLIT_MIN_SIGNALS:
                print(f"  [GATE] {tid}: REJECTED — NIS-only PELT, "
                      f"signal_count={sig_count} < {STRONG_SPLIT_MIN_SIGNALS} "
                      f"(src={src}, sil={sil:.2f})")
                return False

    # --- G4: multi-fragment guard ---
    # Two or more split frames create three or more fragments. With a
    # borderline silhouette this mostly produced short fragments that later
    # filtering stages discarded, losing data. Require silhouette >= 0.65,
    # unless the HSV / same-camera / cross-camera evidence is strong.
    _n_splits = len(c.get('split_frames', []) or [])
    if _n_splits >= 2 and c.get('pelt_flag', False):
        _multi_frag_min_sil = 0.65
        if sil < _multi_frag_min_sil:
            _bypass_hsv = c.get('hsv_dist', 0.0) >= HSV_STRONG_THRESHOLD
            _bypass_kin = c.get('kin_flag', False) and c.get('kin_jump_m', 0.0) >= 15.0
            _bypass_xcam = c.get('xcam_flag', False) and c.get('xcam_jump_m', 0.0) >= 3.0
            # Extreme NIS bypass: reduce to single best split instead of blocking entirely
            _nis_extreme_g4 = c.get('nis_flag', False) and c.get('nis_max', 0) >= 2000
            if _nis_extreme_g4:
                # Allow, but keep a single split
                _split_frames = c.get('split_frames', [])
                if len(_split_frames) > 1:
                    # Keep only the middle split frame
                    c['split_frames'] = [_split_frames[len(_split_frames) // 2]]
                    print(f"  [GATE] {tid}: G4 REDUCED — extreme NIS, "
                          f"{_n_splits} splits -> 1 (sil={sil:.2f})")
            elif not _bypass_hsv and not _bypass_kin and not _bypass_xcam:
                print(f"  [GATE] {tid}: REJECTED — multi-fragment guard: "
                      f"{_n_splits} split -> {_n_splits+1} fragment, "
                      f"sil={sil:.2f} < {_multi_frag_min_sil:.2f}")
                return False

    # --- P1) enough independent votes ---
    if sig_count >= STRONG_SPLIT_MIN_SIGNALS:
        return True

    # --- P2) very large HSV change (sufficient on its own) ---
    if c.get('hsv_dist', 0.0) >= HSV_STRONG_THRESHOLD:
        return True

    # --- P3) same-camera jump >= 15 m (physically impossible) ---
    if c.get('kin_flag', False) and c.get('kin_jump_m', 0.0) >= 15.0:
        return True

    # --- P3b) cross-camera jump >= 3 m between consecutive frames ---
    # 3 m in one frame (40 ms) is ~270 km/h; players reach ~30 km/h.
    if c.get('xcam_flag', False) and c.get('xcam_jump_m', 0.0) >= 3.0:
        return True

    # --- P4) PELT dim-aware silhouette ---
    if c.get('pelt_flag', False):
        threshold = _dim_aware_strong_threshold(src)
        if sil >= threshold:
            # Minority guard: tiny segment + borderline silhouette -> reject
            min_frac = _min_segment_fraction(c)
            gap = int(c.get('max_split_gap_frames', 0) or 0)
            if min_frac < PELT_MIN_MINORITY_FRAC and sil < PELT_MINORITY_OVERRIDE_SIL:
                # ...unless a long detection gap precedes the split
                if gap < PELT_GAP_BYPASS_FRAMES:
                    return False
            return True

    # --- P5) video upper-body HSV source ---
    if c.get('pelt_flag', False) and 'video_hsv' in src and \
       sil >= PELT_STRONG_SILHOUETTE_VIDEO:
        # Same minority guard
        min_frac = _min_segment_fraction(c)
        gap = int(c.get('max_split_gap_frames', 0) or 0)
        if min_frac < PELT_MIN_MINORITY_FRAC and sil < PELT_MINORITY_OVERRIDE_SIL:
            if gap < PELT_GAP_BYPASS_FRAMES:
                return False
        return True

    return False


def _suffix_to_index(suffix: str) -> Optional[int]:
    """`_B`, `_C`, ..., `_Z`, `_AA`, `_AB` -> int idx (B=0, Z=24, AA=25 ...).

    Returns None if suffix is not a valid chimera suffix.
    """
    if not isinstance(suffix, str) or not suffix.startswith("_"):
        return None
    s = suffix[1:]
    if len(s) == 1 and 'B' <= s <= 'Z':
        return ord(s) - ord('B')  # 0..24
    if len(s) == 2 and s.isalpha() and s.isupper():
        first = ord(s[0]) - ord('A')  # A=0
        second = ord(s[1]) - ord('B')  # B=0
        return 25 + first * 25 + second
    return None


def _max_existing_suffix_idx(df: pd.DataFrame, base_tid: str) -> int:
    """Highest suffix index among IDs of the form `<base_tid>_X` /
    `<base_tid>_XX` in df; -1 if there are none (next index 0 = `_B`).
    """
    prefix = f"{base_tid}_"
    max_idx = -1
    try:
        all_ids = df['track_id'].astype(str).unique()
    except Exception:
        return -1
    for tid in all_ids:
        if not tid.startswith(prefix):
            continue
        rest = tid[len(prefix):]
        # IDs from repeated passes can carry several suffixes (e.g. 'B_C');
        # only the first level is counted here ('B_C' -> 'B')
        if "_" in rest:
            rest = rest.split("_")[0]
        idx = _suffix_to_index("_" + rest)
        if idx is not None and idx > max_idx:
            max_idx = idx
    return max_idx


def apply_splits(df: pd.DataFrame, confirmed_splits: List[Dict]) -> pd.DataFrame:
    """
    Apply confirmed splits and return a new DataFrame.

    Uses `split_frames` of each candidate (falls back to `split_frame`).
    N split frames give N+1 segments: the first keeps <tid>, the others become
    <tid>_B, <tid>_C, ... (rows with frame_idx >= split frame move to the new ID).

    Suffix collisions: if df already contains <tid>_B, <tid>_C (from an
    earlier pass), new suffixes continue after the highest existing one, so
    a second pass never reuses an ID created by the first.
    """
    df = df.copy()
    for split in confirmed_splits:
        tid = split['track_id']

        sfs = split.get('split_frames')
        if not sfs:
            sf = split.get('split_frame')
            sfs = [sf] if sf is not None else []
        sfs = sorted({int(s) for s in sfs if s is not None})
        if not sfs:
            continue

        mask_track = (df['track_id'] == tid)
        if not mask_track.any():
            continue
        track_frames = df.loc[mask_track, 'frame_idx'].values

        # Continue after the highest suffix already used for this track
        _start_idx = _max_existing_suffix_idx(df, tid) + 1

        # Assign each row the ID of its segment
        new_ids = np.full(len(track_frames), tid, dtype=object)
        for i, sf in enumerate(sfs):
            suffix = _split_suffix(_start_idx + i)
            new_ids[track_frames >= sf] = f"{tid}{suffix}"

        df.loc[mask_track, 'track_id'] = new_ids

        # Log
        for i, sf in enumerate(sfs):
            suffix = _split_suffix(_start_idx + i)
            print(f"  SPLIT: {tid} -> {tid}{suffix} (f>={sf})")
    return df


def main():
    global HIGH_VIDEO, LOW_VIDEO, VLM_MODEL
    import argparse
    ap = argparse.ArgumentParser(
        description="Find and split identity-mixed (chimera) tracks in a track CSV.",
        epilog="Writes the split CSV to --out (only if something was split) and "
               "chimera_report.json (if there are candidates) to the working "
               "directory. Videos are only "
               "needed for the optional VLM step (VLM_BASE_URL, VLM_API_KEY, "
               "VLM_MODEL).")
    ap.add_argument("--csv", required=True, help="track CSV (see README for columns)")
    ap.add_argument("--high", default=HIGH_VIDEO,
                    help="optional video of camera H (crops for the VLM step)")
    ap.add_argument("--low", default=LOW_VIDEO,
                    help="optional video of camera L (crops for the VLM step)")
    ap.add_argument("--out", default="tracks_cleaned.csv",
                    help="output CSV with split tracks (default: tracks_cleaned.csv)")
    args = ap.parse_args()
    HIGH_VIDEO, LOW_VIDEO = args.high, args.low

    print("=" * 60)
    print("CHIMERA DETECTOR")
    print("=" * 60)

    csv_path = args.csv
    df = pd.read_csv(csv_path)
    print(f"\nCSV: {len(df)} rows, {df.track_id.nunique()} tracks")

    # --- Stage 1: data signals ---
    # (Videos are not passed here, so the video HSV signal is not used by
    # the CLI; see detect_chimera_candidates(video_caps=...).)
    print("\n--- STAGE 1: data signals ---")
    candidates = detect_chimera_candidates(df, verbose=True)
    print(f"\n{len(candidates)} candidates")

    if not candidates:
        print("No candidates found. Exiting.")
        return

    # --- Stage 2: decision (optional VLM vote) ---
    print("\n--- STAGE 2: decision (optional VLM vote) ---")

    # Optional .env next to this file (KEY=VALUE lines); does not override
    # variables that are already set
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

    # Any OpenAI-compatible VLM: VLM_BASE_URL + VLM_API_KEY + VLM_MODEL
    api_key = os.environ.get("VLM_API_KEY", "")
    base_url = os.environ.get("VLM_BASE_URL", "")
    VLM_MODEL = os.environ.get("VLM_MODEL", VLM_MODEL)
    if not (api_key and base_url and VLM_MODEL):
        print("NOTE: VLM not configured; deciding on data signals only.")
        client = None
    else:
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key=api_key)

    # Videos and lens parameters, only used to cut crops for the VLM.
    # lens_params.json (optional, working directory):
    #   {"high": {"k1": <float>}, "low": {"k1": <float>}}; missing -> k1 = 0.
    # Missing videos are fine: the captures simply stay closed.
    lens_params = {}
    if Path("lens_params.json").exists():
        lens_params = json.loads(Path("lens_params.json").read_text())

    # An empty path gives an unopened capture (no OpenCV warning); its
    # properties read as -1, exactly as for a capture that failed to open.
    cap_h = cv2.VideoCapture(HIGH_VIDEO) if HIGH_VIDEO else cv2.VideoCapture()
    cap_l = cv2.VideoCapture(LOW_VIDEO) if LOW_VIDEO else cv2.VideoCapture()
    caps = {"H": cap_h, "L": cap_l}

    fps_high = cap_h.get(cv2.CAP_PROP_FPS)
    fps_low = cap_l.get(cv2.CAP_PROP_FPS)
    fps_ratio = fps_low / fps_high

    k1_h = lens_params.get("high", {}).get("k1", 0.0)
    k1_l = lens_params.get("low", {}).get("k1", 0.0)
    undistort_maps = {
        "H": compute_undistort_maps(cap_h, k1_h),
        "L": compute_undistort_maps(cap_l, k1_l),
    }

    confirmed_splits = []

    for cand in candidates:
        tid = cand['track_id']
        split_frame = cand['split_frame']
        signal_count = cand['signal_count']

        if split_frame is None:
            print(f"\n  {tid}: no split frame, skipped")
            continue

        print(f"\n  {tid}: split=f{split_frame}, signals={signal_count} "
              f"(HSV={'!' if cand['hsv_flag'] else '.'} ReID={'!' if cand['reid_flag'] else '.'})")

        vlm_confirmed = None
        if client:
            vlm_confirmed = vlm_confirm_split(
                client, tid, split_frame, df, caps, undistort_maps, fps_ratio
            )
            time.sleep(1)  # rate limit

        # CLI decision rule:
        #   split if one strong signal fires (HSV >= 0.60, same-camera jump
        #   >= 15 m, cross-camera jump >= 3 m) and the VLM did not say
        #   "same person";
        #   otherwise split if data votes + VLM "different" vote >= 2.
        # total_checks is only used in the log ("x/y").
        total_signals = signal_count + (1 if vlm_confirmed else 0)
        total_checks = 2 + (1 if vlm_confirmed is not None else 0)

        # A very large HSV change is sufficient on its own (ReID was weak on
        # our footage because of the domain gap)
        strong_hsv = cand["hsv_dist"] >= HSV_STRONG_THRESHOLD
        # Same-camera jump >= 15 m: physically impossible
        strong_kin = cand.get("kin_flag", False) and cand.get("kin_jump_m", 0) >= 15.0
        # Cross-camera jump > 3 m between consecutive frames: physically impossible
        strong_xcam = cand.get("xcam_flag", False) and cand.get("xcam_jump_m", 0) >= 3.0

        if (strong_hsv or strong_kin or strong_xcam) and vlm_confirmed is not False:

            confirmed_splits.append(cand)
            reasons = []
            if strong_hsv:
                reasons.append("STRONG HSV ({:.2f})".format(cand["hsv_dist"]))
            if strong_kin:
                reasons.append("KIN TELEPORT ({:.1f}m)".format(cand["kin_jump_m"]))
            if strong_xcam:
                reasons.append("XCAM TELEPORT ({:.1f}m)".format(cand["xcam_jump_m"]))
            if vlm_confirmed:
                reasons.append("VLM: DIFFERENT")
            print(f"    -> CONFIRMED ({' + '.join(reasons)})")
        elif total_signals >= CANDIDATE_MIN_SIGNALS:
            confirmed_splits.append(cand)
            print(f"    -> CONFIRMED ({total_signals}/{total_checks} signals)")
        elif vlm_confirmed is False:
            print(f"    -> REJECTED (VLM: same person)")
        else:
            # Defensive: unreachable with the rule above (signal_count >= 2
            # already implies total_signals >= 2)
            if signal_count >= 2:
                confirmed_splits.append(cand)
                print(f"    -> CONFIRMED BY DATA ({signal_count} data signals)")
            else:
                print(f"    -> UNCERTAIN ({total_signals}/{total_checks}), skipped")

    cap_h.release()
    cap_l.release()

    # --- Stage 3: apply splits ---
    print(f"\n--- STAGE 3: applying {len(confirmed_splits)} split(s) ---")
    if confirmed_splits:
        df_new = apply_splits(df, confirmed_splits)
        output_path = args.out
        df_new.to_csv(output_path, index=False)
        print(f"\nSaved: {output_path}")
        print(f"Tracks: {df_new.track_id.nunique()} "
              f"(before: {df.track_id.nunique()})")
    else:
        print("Nothing to split; no output CSV written.")

    # --- Report ---
    report = {
        "candidates": candidates,
        "confirmed_splits": [
            {"track_id": s["track_id"], "split_frame": s["split_frame"],
             "hsv_dist": s["hsv_dist"], "reid_dist": s["reid_dist"]}
            for s in confirmed_splits
        ],
    }
    with open("chimera_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report: chimera_report.json")


if __name__ == "__main__":
    main()

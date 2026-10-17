"""
build_reid_corpus.py -- build a labelled ReID crop corpus from tracker output.

Inputs
    - tracker CSV (one row per track per frame, see README)
    - the two camera videos (H and L); frame_idx is on the H timeline
    - optional swap_events.json: known identity swaps / relabels
    - optional lens_params.json in the working directory (see _load_lens_params)

Output
    reid_corpus/
        player_01/ ...    one folder per long, clean track (sorted by numeric ID)
        review_queue/     short tracks, close-contact and post-swap crops
        trash/            empty; for crops removed during manual review
    Each non-empty folder also gets an _overview.jpg contact sheet.
    The folders are meant to be checked by hand before training
    (reid_finetune_v2.py only reads player_*/).

Filter chain
    1. is_inference == 1 (real detector boxes; tracker predictions dropped)
    2. box quality: height >= MIN_HEIGHT_H / MIN_HEIGHT_L px, aspect 1.5-3.5
    3. per-track box-centre outliers (tracker drift) dropped
    4. swap_events.json: rows from REVIEW_TAIL_FRAMES before a track's first
       known swap onward go to review_queue
    5. close contact: frames where another track is within 1.8 m
       (+-5 frames) go to review_queue
    6. evenly spaced sampling, at most TARGET_CROPS_PER_ID crops per track
    7. crops that are mostly grass (box missed the player) are dropped

swap_events.json format (all fields except the frame/track keys optional):
    {"events":   [{"track_a": "P-3", "track_b": "P-7", "swap_frame": 1200, "skip": false}],
     "relabels": [{"track_id": "P-5", "from_frame": 900, "skip": false}]}
"""
from __future__ import annotations
import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd


# --- Lens undistortion ------------------------------------------------------
# bbox_pixel is in undistorted image coordinates, so the video frame must be
# undistorted with the same model before cropping; otherwise boxes are off by
# tens of pixels and crops show grass.

def _load_lens_params(path="lens_params.json"):
    """Read {"high": {"k1": ..}, "low": {"k1": ..}}. If the file is missing,
    k1 = -0.5 is used for both cameras (the wide-angle lenses of the rig this
    was developed on). Provide a file with k1 = 0 to disable undistortion."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {"high": {"k1": -0.5}, "low": {"k1": -0.5}}


def _compute_undistort_maps(k1, width, height):
    """Same model as chimera_detector.compute_undistort_maps: pinhole with
    focal length = width, principal point at the centre, alpha = 0.8."""
    if abs(k1) < 0.001:
        return None, None
    focal = width
    cx, cy = width / 2.0, height / 2.0
    K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array([k1, 0, 0, 0, 0], dtype=np.float64)
    new_K, _ = cv2.getOptimalNewCameraMatrix(
        K, dist, (width, height), 0.8, (width, height)
    )
    m1, m2 = cv2.initUndistortRectifyMap(
        K, dist, None, new_K, (width, height), cv2.CV_16SC2
    )
    return m1, m2


def _apply_undistort(frame, map1, map2):
    if map1 is None or map2 is None:
        return frame
    return cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)


# --- Parameters -------------------------------------------------------------
# Small, distant boxes were often displaced by 100-200 px (tracker
# extrapolation), giving crops of empty grass; lower minimum heights (50/40 px)
# let too many of them through.
MIN_HEIGHT_H = 80        # min box height on camera H (px)
MIN_HEIGHT_L = 60        # min box height on camera L (px)
ASPECT_RANGE = (1.5, 3.5)        # allowed height / width
CLOSE_CONTACT_RADIUS_M = 1.8     # another track closer than this = close contact
CLOSE_CONTACT_WINDOW   = 5       # +- frames around a close contact
TARGET_CROPS_PER_ID    = 60      # crops sampled per player
MIN_CROPS_PER_ID       = 15      # shorter tracks go to review_queue
REVIEW_TAIL_FRAMES     = 20      # also review N frames before a known swap
PAD_RATIO              = 0.10    # default argument of clamp_pad_bbox (not used
                                 # there; it applies fixed asymmetric padding)
OVERVIEW_COLS          = 8
OVERVIEW_THUMB_H       = 140
# Grass filter, two levels: drop a crop if more than MAX_GREEN_RATIO_PCT of
# it is green, or if more than MAX_CENTER_GREEN_PCT of its central region is
# green (the torso should be at the centre of a correct box).
MAX_GREEN_RATIO_PCT    = 65
MAX_CENTER_GREEN_PCT   = 50


# --- Helpers ----------------------------------------------------------------

def parse_bbox(s):
    try:
        p = [int(float(x)) for x in str(s).split(",")]
        if len(p) != 4:
            return None
        return p
    except Exception:
        return None


def maybe_center_crop_bbox(bbox):
    """If width / height >= 0.7 the box probably covers two players merged by
    the detector (a standing person is ~0.4). Keep a central strip of width
    0.4 * height at full height, which reduces "two players in one crop".
    """
    if bbox is None:
        return bbox
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = max(1, y2 - y1)
    ar = w / h
    if ar < 0.7:
        return bbox  # normal bbox
    target_w = int(h * 0.4)
    cx = (x1 + x2) / 2.0
    new_x1 = int(cx - target_w / 2.0)
    new_x2 = int(cx + target_w / 2.0)
    return [new_x1, y1, new_x2, y2]


def clamp_pad_bbox(bbox, frame_w, frame_h, pad_ratio=PAD_RATIO):
    """Asymmetric padding with absolute minimums, so that small boxes still
    give a full-body crop (detector boxes sometimes cut the head or feet, and
    relative padding alone adds only a few pixels to a 30-40 px wide box):
      top:    max(0.40 * h, 30 px)
      bottom: max(0.30 * h, 20 px)
      side:   max(0.25 * w, 15 px)
    The result is clamped to the frame; None if it degenerates.
    `pad_ratio` is accepted but not used.
    """
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    pad_top    = max(int(bh * 0.40), 30)
    pad_bottom = max(int(bh * 0.30), 20)
    pad_side   = max(int(bw * 0.25), 15)
    x1p = max(0, x1 - pad_side)
    y1p = max(0, y1 - pad_top)
    x2p = min(frame_w, x2 + pad_side)
    y2p = min(frame_h, y2 + pad_bottom)
    if x2p <= x1p + 5 or y2p <= y1p + 5:
        return None
    return [x1p, y1p, x2p, y2p]


def load_swap_events(path):
    """Return {track_id: first frame of a known swap or relabel}.
    A missing file is not an error (no swap filter)."""
    contam = {}
    if not os.path.exists(path):
        print(f"  [INFO] {path} not found, no swap filter applied.")
        return contam
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for ev in data.get("events", []) or []:
            if ev.get("skip"):
                continue
            f_ = int(ev["swap_frame"])
            for key in ("track_a", "track_b"):
                tid = ev.get(key)
                if tid and (tid not in contam or contam[tid] > f_):
                    contam[tid] = f_
        for rl in data.get("relabels", []) or []:
            if rl.get("skip"):
                continue
            f_ = int(rl["from_frame"])
            tid = rl.get("track_id")
            if tid and (tid not in contam or contam[tid] > f_):
                contam[tid] = f_
        print(f"  [SWAP-EVENTS] first swap frame loaded for {len(contam)} tracks.")
    except Exception as e:
        print(f"  [WARNING] could not parse swap events: {e}")
    return contam


def compute_close_contact_mask(df, radius_m=CLOSE_CONTACT_RADIUS_M,
                               window=CLOSE_CONTACT_WINDOW):
    """Set of (track_id, frame_idx) where another track is within radius_m,
    widened by +-window frames.
    """
    print(f"  [CLOSE-CONTACT] radius={radius_m}m window=+-{window}f ...")
    bad = set()
    # Pairwise distances per frame
    by_frame = df.groupby("frame_idx")
    frames_contact = defaultdict(set)  # frame -> track_ids in close contact
    for fidx, grp in by_frame:
        if len(grp) < 2:
            continue
        coords = grp[["x_field", "y_field"]].values
        tids = grp["track_id"].values
        for i in range(len(grp)):
            for j in range(i + 1, len(grp)):
                dx = coords[i, 0] - coords[j, 0]
                dy = coords[i, 1] - coords[j, 1]
                if dx * dx + dy * dy < radius_m * radius_m:
                    frames_contact[int(fidx)].add(str(tids[i]))
                    frames_contact[int(fidx)].add(str(tids[j]))

    # Widen by +-window frames
    if window > 0:
        all_frames = sorted(frames_contact.keys())
        for f_ in all_frames:
            for df_ in range(-window, window + 1):
                for tid in frames_contact[f_]:
                    bad.add((tid, f_ + df_))
    else:
        for f_, tids in frames_contact.items():
            for tid in tids:
                bad.add((tid, f_))
    print(f"  [CLOSE-CONTACT] {len(bad)} (track, frame) pairs marked.")
    return bad


def save_overview(folder: Path, cols=OVERVIEW_COLS, thumb_h=OVERVIEW_THUMB_H):
    """Write _overview.jpg, a contact sheet of the crops in a folder
    (at most 8 rows)."""
    imgs = sorted([p for p in folder.glob("*.jpg")])
    if not imgs:
        return
    thumbs = []
    for p in imgs[:min(len(imgs), cols * 8)]:
        im = cv2.imread(str(p))
        if im is None:
            continue
        h, w = im.shape[:2]
        nh = thumb_h
        nw = int(w * nh / h)
        thumbs.append(cv2.resize(im, (nw, nh)))
    if not thumbs:
        return
    # Pad to a common width
    max_w = max(t.shape[1] for t in thumbs)
    padded = []
    for t in thumbs:
        if t.shape[1] < max_w:
            pad = np.zeros((thumb_h, max_w - t.shape[1], 3), dtype=np.uint8)
            t = np.hstack([t, pad])
        padded.append(t)
    rows = []
    for i in range(0, len(padded), cols):
        row = padded[i:i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(padded[0]))
        rows.append(np.hstack(row))
    grid = np.vstack(rows)
    cv2.imwrite(str(folder / "_overview.jpg"), grid,
                [cv2.IMWRITE_JPEG_QUALITY, 85])


# --- Main -------------------------------------------------------------------

def build_corpus(
    csv_path: str,
    high_video: str,
    low_video: str,
    swap_events_path: str,
    out_dir: str,
):
    print("=" * 72)
    print("REID CORPUS BUILDER")
    print("=" * 72)
    out = Path(out_dir)
    if out.exists():
        print(f"  [WARNING] {out} already exists; delete it by hand for a clean run.")
        print(f"  Continuing (existing crops may be overwritten).")
    out.mkdir(parents=True, exist_ok=True)
    review = out / "review_queue"
    trash = out / "trash"
    review.mkdir(exist_ok=True)
    trash.mkdir(exist_ok=True)

    # -- load CSV
    print(f"\n[1/6] Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    df["track_id"] = df.track_id.astype(str)
    print(f"  Rows: {len(df):,}")

    # Keep real detections only:
    #   is_inference == 1 -> the detector ran on this frame (real box)
    #   is_inference == 0 -> tracker (Kalman) prediction, position may be off
    # (An earlier version had this inverted and produced crops of grass.)
    if "is_inference" in df.columns:
        mask_real = df.is_inference.astype(str).isin(["1", "1.0", "True", "true"])
        n_pred = (~mask_real).sum()
        df = df[mask_real].copy()
        print(f"  Dropped tracker predictions (is_inference=0): {n_pred:,} -> {len(df):,} detections left")

    # Parse boxes
    bbs = df.bbox_pixel.apply(parse_bbox)
    ok_bbox = bbs.notna()
    df = df[ok_bbox].copy()
    bbs = bbs[ok_bbox].values
    df["x1"] = [b[0] for b in bbs]
    df["y1"] = [b[1] for b in bbs]
    df["x2"] = [b[2] for b in bbs]
    df["y2"] = [b[3] for b in bbs]
    df["bw"] = df.x2 - df.x1
    df["bh"] = df.y2 - df.y1
    df["aspect"] = df.bh / df.bw.clip(lower=1)

    # Size / aspect filter
    h_mask = (
        ((df.camera == "H") & (df.bh >= MIN_HEIGHT_H)) |
        ((df.camera == "L") & (df.bh >= MIN_HEIGHT_L))
    )
    asp_mask = (df.aspect >= ASPECT_RANGE[0]) & (df.aspect <= ASPECT_RANGE[1])
    df = df[h_mask & asp_mask].copy()
    print(f"  After box size/aspect filter: {len(df):,}")

    # -- per-track box-centre outliers (tracker drift)
    # For each (track, camera): drop boxes whose centre is further than
    # max(1.5 * median height, 60 px) from the median centre.
    print(f"\n[1b/6] Per-track bbox drift filter")
    df["cx"] = (df.x1 + df.x2) / 2.0
    df["cy"] = (df.y1 + df.y2) / 2.0
    n_before = len(df)
    keep = np.ones(len(df), dtype=bool)
    df_idx = df.index.to_numpy()
    pos_iloc = {k: i for i, k in enumerate(df_idx)}
    for (tid, cam), grp in df.groupby(["track_id", "camera"]):
        if len(grp) < 8:
            continue  # too few rows for robust statistics
        med_cx = grp.cx.median()
        med_cy = grp.cy.median()
        med_h = grp.bh.median()
        thresh = max(1.5 * med_h, 60.0)
        d = np.hypot(grp.cx - med_cx, grp.cy - med_cy)
        bad_idx = grp.index[d > thresh]
        for bi in bad_idx:
            keep[pos_iloc[bi]] = False
    df = df[keep].copy()
    print(f"  Drift filter dropped: {n_before - len(df):,} -> {len(df):,} left")

    # -- swap events
    print(f"\n[2/6] Loading swap events: {swap_events_path}")
    contam = load_swap_events(swap_events_path)

    # -- close contact
    print(f"\n[3/6] Close-contact mask")
    contact_set = compute_close_contact_mask(df)

    # -- assign labels
    print(f"\n[4/6] Labelling tracks (P-X -> player_XX)")
    # Row count per track (shown for the 20 longest)
    track_counts = df.groupby("track_id").size().sort_values(ascending=False)
    print(f"  Tracks: {len(track_counts)}")
    for tid, n in track_counts.head(20).items():
        con = contam.get(tid, "-")
        print(f"    {tid}: n={n}  contam_at={con}")

    # Tracks below MIN_CROPS_PER_ID go to review_queue
    valid_tracks = [t for t, n in track_counts.items() if n >= MIN_CROPS_PER_ID]
    small_tracks = [t for t, n in track_counts.items() if n < MIN_CROPS_PER_ID]
    print(f"  Tracks with >= {MIN_CROPS_PER_ID} rows: {len(valid_tracks)}")
    print(f"  Short tracks (to review_queue): {len(small_tracks)}")

    # track_id -> player_XX, in numeric order of "P-<n>" IDs
    def _numeric_sort_key(t):
        s = str(t).replace("P-", "").replace("p-", "")
        try:
            return int(s)
        except Exception:
            return 99999
    valid_tracks_sorted = sorted(valid_tracks, key=_numeric_sort_key)
    tid_to_label = {}
    for idx, tid in enumerate(valid_tracks_sorted, 1):
        tid_to_label[tid] = f"player_{idx:02d}"
        label_dir = out / tid_to_label[tid]
        label_dir.mkdir(exist_ok=True)
    print(f"  Label mapping: {tid_to_label}")

    # -- extract crops from the videos
    print(f"\n[5/6] Extracting crops")
    cap_h = cv2.VideoCapture(high_video)
    cap_l = cv2.VideoCapture(low_video)
    if not cap_h.isOpened():
        print(f"  ERROR: cannot open video H: {high_video}")
        return
    if not cap_l.isOpened():
        print(f"  ERROR: cannot open video L: {low_video}")
        return
    fw_h = int(cap_h.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh_h = int(cap_h.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fw_l = int(cap_l.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh_l = int(cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # frame_idx is on the H timeline; camera L may run at a different fps
    fps_h = cap_h.get(cv2.CAP_PROP_FPS)
    fps_l = cap_l.get(cv2.CAP_PROP_FPS)
    fps_ratio = fps_l / max(fps_h, 1.0)
    print(f"  HIGH fps={fps_h:.2f}, LOW fps={fps_l:.2f}, ratio={fps_ratio:.4f}")

    # -- undistortion maps (boxes are in undistorted coordinates)
    lens = _load_lens_params()
    k1_h = float(lens.get("high", {}).get("k1", 0.0))
    k1_l = float(lens.get("low", {}).get("k1", 0.0))
    print(f"  Lens k1: H={k1_h}, L={k1_l}")
    map_h1, map_h2 = _compute_undistort_maps(k1_h, fw_h, fh_h)
    map_l1, map_l2 = _compute_undistort_maps(k1_l, fw_l, fh_l)
    if map_h1 is None and map_l1 is None:
        print(f"  [WARNING] k1 ~ 0, no undistortion; boxes may not line up.")

    # Plan crops per track: safe rows (before any known swap, no close
    # contact) are sampled; the rest go to review_queue
    task_plan = []  # (track_id, label, frame_idx, cam, bbox, status)

    for tid, grp in df.groupby("track_id"):
        grp_sorted = grp.sort_values("frame_idx")
        swap_at = contam.get(tid, None)
        safe_rows = []
        review_rows = []
        for _, row in grp_sorted.iterrows():
            f_ = int(row.frame_idx)
            is_close = (tid, f_) in contact_set
            is_post_swap = (swap_at is not None and f_ >= swap_at - REVIEW_TAIL_FRAMES)
            bb = [int(row.x1), int(row.y1), int(row.x2), int(row.y2)]
            cam = str(row.camera)
            item = (f_, cam, bb)
            if is_close or is_post_swap:
                review_rows.append(item)
            else:
                safe_rows.append(item)

        label = tid_to_label.get(tid, None)

        if label is None:
            # Short track: every crop goes to review_queue
            for f_, cam, bb in (safe_rows + review_rows):
                task_plan.append((tid, f"review_queue", f_, cam, bb,
                                  f"small_track_n{len(grp)}"))
        else:
            # Long track: evenly spaced sample of the safe rows
            if len(safe_rows) > 0:
                n_target = min(TARGET_CROPS_PER_ID, len(safe_rows))
                if len(safe_rows) <= n_target:
                    sample = safe_rows
                else:
                    idxs = np.linspace(0, len(safe_rows) - 1, n_target).astype(int)
                    sample = [safe_rows[i] for i in idxs]
                for f_, cam, bb in sample:
                    task_plan.append((tid, label, f_, cam, bb, "safe"))
            # Review rows: at most 30 per track
            rvlim = min(30, len(review_rows))
            if rvlim > 0:
                rv_idxs = np.linspace(0, len(review_rows) - 1, rvlim).astype(int)
                for i in rv_idxs:
                    f_, cam, bb = review_rows[int(i)]
                    task_plan.append((tid, "review_queue", f_, cam, bb,
                                      "post_swap_or_close_contact"))

    print(f"  Planned crops: {len(task_plan)}")
    # Sort by (camera, frame) so each frame is decoded once
    task_plan.sort(key=lambda t: (t[3], t[2]))  # (cam, frame_idx)

    # Extract
    written = defaultdict(int)
    last_cam = None
    last_frame_idx = -1
    cur_frame = None
    for (tid, label, f_, cam, bb, status) in task_plan:
        cap = cap_h if cam == "H" else cap_l
        fw = fw_h if cam == "H" else fw_l
        fh = fh_h if cam == "H" else fh_l

        if (cam != last_cam) or (f_ != last_frame_idx):
            # Camera L: map the H frame index with fps_ratio
            seek_f = f_ if cam == "H" else int(round(f_ * fps_ratio))
            cap.set(cv2.CAP_PROP_POS_FRAMES, seek_f)
            ok, raw_frame = cap.read()
            last_cam = cam
            last_frame_idx = f_
            if not ok or raw_frame is None:
                cur_frame = None
                continue
            # Undistort (boxes are in undistorted coordinates)
            if cam == "H":
                cur_frame = _apply_undistort(raw_frame, map_h1, map_h2)
            else:
                cur_frame = _apply_undistort(raw_frame, map_l1, map_l2)

        # Wide box (two merged players): keep the central strip
        bb_centered = maybe_center_crop_bbox(bb)
        padded = clamp_pad_bbox(bb_centered, fw, fh)
        if padded is None:
            continue
        x1, y1, x2, y2 = padded
        crop = cur_frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Grass filter (pitch green in OpenCV HSV: H 35-85, S 40-255, V 40-200):
        #   1) green share of the whole crop > MAX_GREEN_RATIO_PCT -> drop
        #   2) green share of the central region > MAX_CENTER_GREEN_PCT -> drop
        #      (a correct box has the player, not grass, at its centre)
        try:
            _hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            _gmask = cv2.inRange(_hsv, (35, 40, 40), (85, 255, 200))
            _gratio = _gmask.sum() / 255 / (crop.shape[0] * crop.shape[1]) * 100
            if _gratio > MAX_GREEN_RATIO_PCT:
                continue
            # Central region: middle half in each direction
            _ch, _cw = crop.shape[:2]
            _mid = _gmask[_ch//4:3*_ch//4, _cw//4:3*_cw//4]
            if _mid.size > 0:
                _mid_green = _mid.sum() / 255 / _mid.size * 100
                if _mid_green > MAX_CENTER_GREEN_PCT:
                    continue  # grass at the centre: box missed the player
        except Exception:
            pass  # on an OpenCV error keep the crop

        out_dir_ = out / label
        out_dir_.mkdir(exist_ok=True)
        safe_tid = str(tid).replace("/", "_")
        fn = f"{safe_tid}_f{f_:05d}_{cam}_{status}.jpg"
        cv2.imwrite(str(out_dir_ / fn), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        written[label] += 1

    cap_h.release()
    cap_l.release()
    print(f"\n  Crops written:")
    for lbl, n in sorted(written.items()):
        print(f"    {lbl}: {n}")

    # -- contact sheets
    print(f"\n[6/6] Writing overview images")
    for sub in sorted(out.iterdir()):
        if sub.is_dir():
            save_overview(sub)
            n = len(list(sub.glob("*.jpg"))) - (1 if (sub / "_overview.jpg").exists() else 0)
            print(f"    {sub.name}: {n} crop + overview")

    print()
    print("=" * 72)
    print("DONE. Review the folders by hand, then run reid_finetune_v2.py")
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser(
        description="Build a labelled ReID crop corpus from tracker output and videos.")
    ap.add_argument("--csv", required=True, help="raw tracker CSV (see README for columns)")
    ap.add_argument("--high", required=True, help="video of camera H")
    ap.add_argument("--low", required=True, help="video of camera L")
    ap.add_argument("--swap-events", default="swap_events.json",
                    help="optional JSON of known swaps (default: swap_events.json; "
                         "skipped if missing)")
    ap.add_argument("--out", default="reid_corpus",
                    help="output folder (default: reid_corpus)")
    args = ap.parse_args()
    build_corpus(args.csv, args.high, args.low, args.swap_events, args.out)


if __name__ == "__main__":
    main()

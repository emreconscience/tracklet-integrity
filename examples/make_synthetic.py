"""Generate a small synthetic two-player match CSV for trying the detectors.

Players A and B move on separate paths. With --swap (default) the two track IDs
exchange identities at frame 600: each track jumps to the other player's position
and appearance. This is the kind of "chimera" track that the detectors should split.

    python examples/make_synthetic.py demo.csv            # with an identity swap
    python examples/make_synthetic.py clean.csv --no-swap # control: nothing to split
"""
import argparse
import base64

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser(description="Write a synthetic two-player track CSV.")
ap.add_argument("out", help="output CSV path")
ap.add_argument("--no-swap", action="store_true", help="control data without an identity swap")
ap.add_argument("--frames", type=int, default=1200, help="number of frames at 25 fps (default: 1200)")
ap.add_argument("--swap-at", type=int, default=600, help="frame of the swap (default: 600)")
args = ap.parse_args()

rng = np.random.default_rng(0)
emb_a, emb_b = rng.normal(size=512), rng.normal(size=512)


def b64(v):
    v = v + rng.normal(scale=0.15, size=v.size)
    v = v / np.linalg.norm(v)
    return base64.b64encode(v.astype(np.float16).tobytes()).decode()


rows = []
for f in range(args.frames):
    t = f / 25.0
    pa = (10 + 1.5 * t, 20 + np.sin(t / 3) * 3)
    pb = (60 - 1.0 * t, 45 + np.cos(t / 4) * 4)
    swapped = (not args.no_swap) and f >= args.swap_at
    tracks = {
        "P-1": (pb, emb_b) if swapped else (pa, emb_a),
        "P-2": (pa, emb_a) if swapped else (pb, emb_b),
    }
    for tid, (pos, emb) in tracks.items():
        rows.append(dict(
            frame_idx=f, track_id=tid,
            x_field=round(pos[0] + rng.normal(scale=0.05), 3),
            y_field=round(pos[1] + rng.normal(scale=0.05), 3),
            speed_kmh=0.0, is_inference=1, appearance="",
            reid_embedding=b64(emb) if f % 3 == 0 else "",
            reid_quality=0.8 if f % 3 == 0 else 0.0,
            bbox_pixel="100,100,150,200", camera="H",
        ))

pd.DataFrame(rows).to_csv(args.out, index=False)
print(f"wrote {len(rows)} rows to {args.out}")

"""Generate a small synthetic match CSV for trying the detectors.

Two scenarios:

* ``teleport`` (default): players A and B move on separate paths. At frame 600
  the two track IDs exchange identities, so each track jumps to the other
  player's position and appearance.
* ``crossing``: a 40 x 20 m pitch with six players. A and B run past each other
  and, at the moment they meet (frame 600), the tracker follows the wrong
  person: positions stay continuous, only the identity changes. The other four
  players are clean tracks and should be left alone.

With --no-swap the same paths are written without the identity swap (control).

    python examples/make_synthetic.py demo.csv                      # teleport swap
    python examples/make_synthetic.py cross.csv --scenario crossing # swap at a crossing
    python examples/make_synthetic.py clean.csv --no-swap           # control: nothing to split
"""
import argparse
import base64

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser(description="Write a synthetic track CSV.")
ap.add_argument("out", help="output CSV path")
ap.add_argument("--scenario", choices=["teleport", "crossing"], default="teleport",
                help="kind of identity swap (default: teleport)")
ap.add_argument("--no-swap", action="store_true", help="control data without an identity swap")
ap.add_argument("--frames", type=int, default=1200, help="number of frames at 25 fps (default: 1200)")
ap.add_argument("--swap-at", type=int, default=600, help="frame of the swap (default: 600)")
args = ap.parse_args()

FPS = 25.0
rng = np.random.default_rng(0)


def b64(v):
    v = v + rng.normal(scale=0.15, size=v.size)
    v = v / np.linalg.norm(v)
    return base64.b64encode(v.astype(np.float16).tobytes()).decode()


def teleport_paths():
    def pa(t):
        return 10 + 1.5 * t, 20 + np.sin(t / 3) * 3

    def pb(t):
        return 60 - 1.0 * t, 45 + np.cos(t / 4) * 4

    return {"A": pa, "B": pb}


def crossing_paths():
    t_meet = args.swap_at / FPS
    meet = np.array([20.0, 10.0])

    def runner(start, wobble, phase):
        start = np.array(start, dtype=float)
        vel = (meet - start) / t_meet           # both runners reach `meet` at t_meet

        def p(t):
            x, y = start + vel * t
            return x, y + wobble * np.sin(t / 2.5 + phase) * min(1.0, abs(t - t_meet) / 4)
        return p

    def wanderer(cx, cy, rx, ry, w, phase):
        def p(t):
            return cx + rx * np.sin(w * t + phase), cy + ry * np.cos(0.7 * w * t + phase)
        return p

    return {
        "A": runner((4.0, 4.0), 1.2, 0.0),       # bottom-left to top-right
        "B": runner((36.0, 5.0), 1.0, 1.3),      # bottom-right to top-left
        "C": wanderer(8, 14, 3.0, 2.5, 0.25, 0.4),
        "D": wanderer(31, 15, 3.5, 2.0, 0.22, 1.9),
        "E": wanderer(15, 4, 4.0, 1.5, 0.30, 2.7),
        "F": wanderer(27, 9, 2.5, 3.0, 0.27, 0.9),
    }


paths = teleport_paths() if args.scenario == "teleport" else crossing_paths()
embs = {name: rng.normal(size=512) for name in paths}
# Shirt-colour histograms (HSV, 48 bins) only in the crossing scenario, so the
# teleport demo stays exactly as before.
hists = {name: rng.gamma(0.6, size=48) for name in paths} if args.scenario == "crossing" else {}


def b64_hist(h):
    h = np.clip(h + rng.normal(scale=0.05 * h.mean(), size=h.size), 0, None)
    h = h / h.sum()
    return base64.b64encode(h.astype(np.float16).tobytes()).decode()


rows = []
for f in range(args.frames):
    t = f / FPS
    swapped = (not args.no_swap) and f >= args.swap_at
    # track ID -> player actually followed in this frame
    follow = {f"P-{i + 1}": name for i, name in enumerate(paths)}
    if swapped:
        follow["P-1"], follow["P-2"] = "B", "A"
    for tid, name in follow.items():
        x, y = paths[name](t)
        rows.append(dict(
            frame_idx=f, track_id=tid,
            x_field=round(x + rng.normal(scale=0.05), 3),
            y_field=round(y + rng.normal(scale=0.05), 3),
            speed_kmh=0.0, is_inference=1,
            appearance=b64_hist(hists[name]) if hists and f % 3 == 0 else "",
            reid_embedding=b64(embs[name]) if f % 3 == 0 else "",
            reid_quality=0.8 if f % 3 == 0 else 0.0,
            bbox_pixel="100,100,150,200", camera="H",
        ))

pd.DataFrame(rows).to_csv(args.out, index=False)
print(f"wrote {len(rows)} rows to {args.out}")

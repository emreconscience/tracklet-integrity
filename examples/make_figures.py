"""Reproduce the figures in assets/ from synthetic data.

    python examples/make_figures.py

Generates the "crossing" scenario (examples/make_synthetic.py), runs the real
command-line detector on it, and plots what it did. Nothing here is drawn by
hand: the split tracks in the figure are read back from the detector's output.
"""
import os
import subprocess
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from chimera_detector import decode_embedding  # noqa: E402
from nis_anomaly_detector import detect_track_nis  # noqa: E402

OUT = os.path.join(ROOT, "assets")
SWAP = 600
FPS = 25.0
L, W = 40.0, 20.0

INK = "#1f2328"
MUTED = "#8c959f"
GRID = "#d0d7de"
PITCH = "#f3f7f2"
C1, C1B = "#d1495b", "#f4a259"    # P-1 and the part split off from it
C2, C2B = "#2e86ab", "#7fc8a9"    # P-2 and the part split off from it

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlecolor": INK,
})


def run(cmd):
    subprocess.run([sys.executable] + cmd, cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def draw_pitch(ax):
    ax.add_patch(plt.Rectangle((0, 0), L, W, fc=PITCH, ec=MUTED, lw=1.2, zorder=0))
    ax.plot([L / 2, L / 2], [0, W], color=MUTED, lw=1, zorder=0)
    ax.add_patch(plt.Circle((L / 2, W / 2), 3, fill=False, ec=MUTED, lw=1, zorder=0))
    for x0 in (0, L - 6):
        ax.add_patch(plt.Rectangle((x0, W / 2 - 5), 6, 10, fill=False, ec=MUTED, lw=1, zorder=0))
    ax.set_xlim(-1, L + 1)
    ax.set_ylim(-1, W + 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def path(ax, d, color, label, at="end"):
    """Draw a track; the label sits at its first ("start") or last ("end") point."""
    d = d.sort_values("frame_idx")
    ax.plot(d.x_field, d.y_field, color=color, lw=2.4, solid_capstyle="round", zorder=3)
    i = 0 if at == "start" else -1
    x, y = d.x_field.iloc[i], d.y_field.iloc[i]
    ax.scatter([x], [y], color=color, s=28, zorder=4)
    ax.annotate(label, (x, y), xytext=(-6, -13) if at == "start" else (-16, 7), textcoords="offset points",
                color=color, fontsize=9, fontweight="bold")


def figure_demo(raw, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    meet = raw[(raw.track_id == "P-1") & (raw.frame_idx == SWAP)][["x_field", "y_field"]].iloc[0]
    for ax in axes:
        draw_pitch(ax)
        for tid in ["P-3", "P-4", "P-5", "P-6"]:
            d = out[out.track_id == tid].sort_values("frame_idx")
            ax.plot(d.x_field, d.y_field, color=MUTED, lw=1, alpha=0.55, zorder=1)

    ax = axes[0]
    path(ax, raw[raw.track_id == "P-1"], C1, "P-1")
    path(ax, raw[raw.track_id == "P-2"], C2, "P-2")
    ax.add_patch(plt.Circle((meet.x_field, meet.y_field), 1.4, fill=False, ec=INK, lw=1.2, ls="--", zorder=5))
    ax.annotate("players cross, tracker follows the wrong one", (meet.x_field, meet.y_field - 1.4),
                xytext=(12.5, 0.6), fontsize=8.5, color=INK, zorder=6,
                bbox=dict(boxstyle="round,pad=0.25", fc=PITCH, ec="none"),
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
    ax.set_title("Tracker output: each track holds two people", loc="left")

    ax = axes[1]
    for tid, color, at in [("P-1", C1, "start"), ("P-1_B", C1B, "end"),
                           ("P-2", C2, "start"), ("P-2_B", C2B, "end")]:
        path(ax, out[out.track_id == tid], color, tid, at)
    ax.set_title("After tracklet-integrity: split at the switch", loc="left")

    fig.text(0.01, 0.01, "Synthetic 40 × 20 m pitch, 6 players, 48 s at 25 fps. Grey: tracks left untouched. "
             "Reproduce with examples/make_figures.py", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(OUT, "demo_crossing.png"), dpi=160)
    plt.close(fig)


def drift(d, col):
    """Cosine distance of each embedding to the mean of the track's first 100 embeddings."""
    d = d[d[col].notna() & (d[col] != "")].sort_values("frame_idx")
    v = np.stack([decode_embedding(s).astype(np.float32) for s in d[col]])
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    ref = v[:100].mean(axis=0)
    ref /= np.linalg.norm(ref)
    dist = pd.Series(1 - v @ ref).rolling(9, center=True, min_periods=1).median()  # light smoothing
    return d.frame_idx.to_numpy() / FPS, dist.to_numpy()


def figure_signals(raw, out):
    p1 = raw[raw.track_id == "P-1"]
    nis = detect_track_nis(raw, "P-1", fps=FPS)
    f_nis = np.array([f for f, _ in nis["nis_per_frame"]]) / FPS
    v_nis = np.array([n for _, n in nis["nis_per_frame"]])
    split_t = out[out.track_id == "P-1_B"].frame_idx.min() / FPS

    fig, axes = plt.subplots(3, 1, figsize=(11, 5.2), sharex=True)
    rows = [
        (*drift(p1, "reid_embedding"), "ReID distance", C1),
        (*drift(p1, "appearance"), "Shirt-colour distance", C1),
        (f_nis, v_nis, "Kalman NIS", INK),
    ]
    for ax, (t, y, name, color) in zip(axes, rows):
        ax.plot(t, y, color=color, lw=1.3)
        ax.axvline(split_t, color=INK, lw=1, ls="--")
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9.5)
        ax.grid(axis="y", color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[2].set_ylim(0, max(30, float(v_nis.max()) * 1.1))
    axes[2].axhline(9.21, color=MUTED, lw=0.8, ls=":")
    axes[2].text(0.3, 10.5, "χ² 99% threshold", color=MUTED, fontsize=8)
    axes[0].set_title("Signals along track P-1 (dashed line: split frame chosen by the detector)", loc="left")
    axes[0].annotate(f"split at {split_t:.0f} s (frame {int(split_t * FPS)})", (split_t, 0.6),
                     xytext=(8, 0), textcoords="offset points", fontsize=9, color=INK, va="center")
    axes[2].set_xlabel("time (s)")
    fig.text(0.01, 0.01, "At a crossing the position barely changes, so kinematics alone cannot see the switch; "
             "appearance carries the decision.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(OUT, "signals.png"), dpi=160)
    plt.close(fig)


def figure_pipeline():
    fig, ax = plt.subplots(figsize=(11, 3.3))
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 33)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#ffffff", ec=GRID, bold=False, color=INK, size=9.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=color,
                fontweight="bold" if bold else "normal")

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))

    box(1, 13, 14, 7, "finished\ntracks (CSV)", bold=True)
    signals = [
        ("Kalman NIS", "impossible motion"),
        ("position jumps", "teleports"),
        ("appearance change points", "PELT + silhouette"),
        ("appearance drift", "start vs end of track"),
        ("spatial bimodality", "two areas, apart in time"),
    ]
    for i, (name, sub) in enumerate(signals):
        y = 27 - i * 6.2
        box(26, y, 30, 4.6, f"{name}  ·  {sub}", size=8.8)
        arrow(15.5, 16.5, 25.5, y + 2.3)
        arrow(56.5, y + 2.3, 66.5, 16.5)
    box(67, 11, 18, 11, "vote\n≥ 2 signals, or one\nphysically certain", fc="#fff7e6", ec="#f4a259", size=9)
    box(67, 1, 18, 6, "optional VLM vote", size=8.5, color=MUTED)
    arrow(76, 7.4, 76, 10.6)
    box(93, 13, 16, 7, "split tracks\n<id>, <id>_B", bold=True)
    arrow(85.5, 16.5, 92.5, 16.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pipeline.png"), dpi=160)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        raw_csv = os.path.join(tmp, "crossing.csv")
        out_csv = os.path.join(tmp, "crossing_split.csv")
        run(["examples/make_synthetic.py", raw_csv, "--scenario", "crossing"])
        run(["chimera_detector.py", "--csv", raw_csv, "--out", out_csv])
        report = os.path.join(ROOT, "chimera_report.json")
        if os.path.exists(report):
            os.remove(report)
        raw, out = pd.read_csv(raw_csv), pd.read_csv(out_csv)
    figure_demo(raw, out)
    figure_signals(raw, out)
    figure_pipeline()
    print("wrote", ", ".join(sorted(os.listdir(OUT))))


if __name__ == "__main__":
    main()

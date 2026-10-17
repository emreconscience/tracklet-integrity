# Details

Technical reference for `tracklet-integrity`: signals, decision rule, input format, options and known limitations. For the overview see the [README](../README.md).

## How it works

```
track CSV --> per-track signals ------------------------------> decision ----------------> split CSV
              hsv / reid  first vs last quarter of the track    >= 2 votes, or one         <id>, <id>_B, ...
              pelt        change points in appearance           physically impossible      + chimera_report.json
              spatial_bi  two position clusters, apart in time  jump; optional VLM vote
              kin / xcam  same-camera / cross-camera jumps
              nis         Kalman NIS (nis_anomaly_detector.py)
```

Each track is checked with several independent signals:

| Signal | What it catches | File |
|---|---|---|
| **Kinematic consistency** – a constant-velocity Kalman filter is run over the track in field coordinates; the Normalized Innovation Squared (NIS = yᵀS⁻¹y, χ² with 2 DOF) measures how surprising each new position is. A track is flagged only if it has a strong cluster (peak NIS ≥ 75) and at least two anomalous clusters | teleports and physically impossible motion | `nis_anomaly_detector.py` |
| **Position jumps** – smoothed position moves > 10 m within 15 detections on one camera (`kin`), or > 3 m between consecutive frames on different cameras (`xcam`) | a track that jumps to another player | `chimera_detector.py` |
| **Appearance change points** – HSV colour and ReID embeddings along the track, PCA-reduced within the track, segmented with PELT (`ruptures`, RBF cost); a split is accepted only if the segments separate (silhouette) | the track "changes shirt" at some frame | `chimera_detector.py` |
| **Appearance drift** – cosine distance between the mean HSV / ReID vector of the first and last quarter of the track | slower or less clear-cut identity changes | `chimera_detector.py` |
| **Spatial bimodality** – do the track's positions fall into two clusters (≥ 10 m apart) that are also separated in time? | a track that moves between two players' areas of the pitch | `chimera_detector.py` |
| *(optional)* **VLM vote** – crops from before/after the candidate frame are sent to any OpenAI-compatible vision model | an extra vote on borderline cases | `chimera_detector.py` |

**Decision rule of the command-line tool** (`main` in `chimera_detector.py`): a track is split
if at least two signals fire (a VLM answer "different person" counts as one more vote), or if a
single signal is physically unambiguous: an HSV distance ≥ 0.60, a same-camera jump ≥ 15 m or a
cross-camera jump ≥ 3 m (these are vetoed if the VLM says "same person"). Split frames come from
the change-point, spatial, NIS and jump signals (or from the appearance-drift scan if none of
those gives one). An appearance-only split point is discarded when the mean position in the
60 frames before and after it differs by less than 1.5 m, since that is usually pose or lighting
drift rather than a new person.

`is_strong_split_candidate()` in the same file is a **stricter gate** (higher silhouette floor,
three votes for the multi-signal path, guards against NIS-only and many-fragment splits) for
use when the detector is imported into a larger multi-pass pipeline. The CLI does not call it.

Design choices that came from failures on real data:

- **A single kinematic spike never splits a track.** One NIS spike is often detector jitter or a
  bad frame. The NIS flag needs two anomalous clusters, and it is one vote among several; only a
  jump that no player could make counts on its own.
- **Splits are recursive** (up to depth 2 per pass), because one track can contain three people.
- **Weak candidates are dropped, not guessed.** If the signals do not agree, the track is left alone.
- **All thresholds were tuned on the development match.** They are documented next to their
  definitions in the code, together with the reason for each value.

The ReID part (`build_reid_corpus.py`, `reid_finetune_v2.py`) builds a labelled crop corpus
from tracker output (keeping only real detections, dropping close-contact frames, crops that
show mostly grass, and anything after a known swap) and fine-tunes OSNet x1_0 with a 4-stripe
part-based head (PCB) + global head, SupCon + cross-entropy, and a P×K identity sampler.
The corpus folders are meant to be checked by hand before training.

## Input format

One row per track per frame:

| column | meaning | needed by |
|---|---|---|
| `frame_idx`, `track_id` | frame number and tracker ID | all |
| `x_field`, `y_field` | position on the pitch in metres | all |
| `is_inference` | `1` = the detector ran on this frame (real box), `0` = tracker (Kalman) prediction; if the column is missing, all rows are used | all |
| `camera` | `H` or `L` (two-camera rig; `frame_idx` is on the H timeline) | jump signals, ReID corpus |
| `appearance` | base64-encoded float16 HSV histogram of the box (may be empty) | HSV signals |
| `reid_embedding` | base64-encoded float16 ReID vector (may be empty on some frames) | ReID signals |
| `bbox_pixel` | `"x1,y1,x2,y2"` in undistorted pixel coordinates of the source video | VLM crops, ReID corpus |

Missing optional columns switch the corresponding signal off.

### Optional inputs

| input | used by | effect |
|---|---|---|
| `--high`, `--low` videos | `chimera_detector.py` | only used to cut crops for the VLM vote; the CLI's data signals come from the CSV alone. (An upper-body video HSV signal is available to library users via `detect_chimera_candidates(video_caps=...)`.) |
| `VLM_BASE_URL`, `VLM_API_KEY`, `VLM_MODEL` (environment or a `.env` file next to the script) | `chimera_detector.py` | enables the VLM vote; without them it is skipped. Needs `pip install openai`. |
| `lens_params.json` in the working directory: `{"high": {"k1": …}, "low": {"k1": …}}` | both tools that read video | single-coefficient radial undistortion so that crops line up with `bbox_pixel`. If missing, `chimera_detector.py` assumes no distortion and `build_reid_corpus.py` assumes `k1 = -0.5` (the lenses of the original rig). |
| `swap_events.json` (`--swap-events`) | `build_reid_corpus.py` | known swaps; crops from shortly before a track's first swap onward go to `review_queue/`. Format in the script's docstring. Skipped if missing. |

## Repository layout

```
nis_anomaly_detector.py      Kalman NIS per track (CLI + importable)
chimera_detector.py          multi-signal identity-mix detector and splitter (CLI + importable)
build_reid_corpus.py         crop corpus from tracker CSV + videos
reid_finetune_v2.py          OSNet + PCB fine-tuning on the corpus
examples/make_synthetic.py   synthetic data: "teleport" or "crossing" swap, or a clean control
examples/make_figures.py     regenerates assets/*.png by running the detector on synthetic data
assets/                      figures used in the README
docs/DETAILS.md              this file
```

## Status and honest results

- **Synthetic data:** the detector finds the planted swaps at the exact frame and does not split clean tracks.
- **Real data:** developed and tuned on a single amateur match (two cameras, ~25 fps) as one stage
  of a longer post-processing chain. Its thresholds were tuned on that same match, so no accuracy
  number from it would be a fair estimate, and none is claimed here.
- **ReID:** the margin metric printed during training (`eval_margin`) is computed on the
  **training** corpus. It shows the model can separate the players it was trained on; it says
  nothing about new matches. The corpus was a few thousand crops of ~40 identities from 1–3 matches.
- **Swaps at a crossing need two appearance cues.** When two players pass through each other the
  position before and after the switch is the same, so the guard that rejects appearance-only split
  points with less than 1.5 m of movement also rejects this one. In the synthetic crossing scenario
  the switch is found at the right frame either way, but it is only split when both ReID and shirt
  colour disagree (two votes through the appearance-drift scan); with ReID alone the track is left
  as it is. Making the guard aware of nearby players is an open item.
- **Not done yet:** evaluation on a public benchmark (e.g. SportsMOT / SoccerNet-Tracking) and a
  comparison against existing tracklet splitters (e.g. GTA-Link). That is the obvious next step.
- **Known rough edges,** left as they are so that the code matches what was tuned: a few
  branches in `chimera_detector.py` have no effect or cannot be reached (marked with comments),
  and the VLM prompt and its JSON keys are in Turkish (the parser depends on them). Everything
  else (comments, docstrings, log output, CLI help) is in English.

## Lessons learned

- **Validate on data the component did not see.** An early ReID score looked excellent (~0.96
  self-similarity) and turned out to be measured on the training set.
- **Check what your flags mean.** `is_inference` was read with opposite meanings in different
  scripts, so some of them were filtering *for* Kalman predictions instead of real detections.
  Found while preparing this repo; all scripts now use the same convention.
- **One signal is never enough** for an irreversible operation like splitting an identity.

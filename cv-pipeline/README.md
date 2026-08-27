# cv-pipeline/

Computer-vision pipeline: pose estimation, ball tracking, and shot-outcome
classification from camera footage.

## Scope (Phase 1 — MVP)

- Pose estimation on the batter's stance, backlift, and footwork.
- Ball tracking: release point, trajectory, pitch location, deviation.
- Bat-ball contact analysis and shot-outcome classification (middled,
  edged, missed, defended, attacked).
- Runs offline or with a short processing delay on a laptop/cloud GPU —
  **not** on embedded hardware yet (see `docs/product/MVP_RD_Plan_Software_First.docx`).

## What's real vs. still placeholder

- **Real, working**: `pose/` — body-pose landmark extraction from video
  using a pretrained MediaPipe Pose Landmarker model (no training, no
  scraped video — see `pose/download_model.py`), plus derived stance,
  knee-flex, and hand-height features (`pose/features.py`). Run
  `pytest cv-pipeline/tests/` to check it.
- **Not built yet**: `ball_tracking/` and `shot_outcome/` — no code.
- **Not built yet**: any link between pose output and the mastery
  scorer in `adaptation-engine/` — outcomes there are still typed in by
  a human operator (see `app/README.md`).

## Setup

```
pip install -r requirements.txt
python cv-pipeline/pose/download_model.py   # fetches the pretrained model (~5.5MB, gitignored)
```

## Try it on real footage

```
python cv-pipeline/pose/pose_estimation.py path/to/a/clip.mp4
```

Prints per-frame stance/knee-flex/hand-height features. Two honest
caveats, detailed in `pose/features.py`'s docstring: "hand height" is a
proxy for backlift, not real bat tracking, and landmarks are left/right,
not front/back leg (that needs the batter's handedness, which pose
estimation alone doesn't give). This has never been run against real
cricket footage — only synthetic frames with no person in them, to check
the pipeline itself doesn't crash (`cv-pipeline/tests/`). Real accuracy
is unverified until it's pointed at an actual net session.

## Structure

```
cv-pipeline/
  pose/
    pose_estimation.py   video -> per-frame landmarks + features (real)
    features.py           pure landmark -> stance/footwork feature math (real)
    download_model.py     fetches the pretrained model file
    models/                gitignored; holds the downloaded model
  ball_tracking/          trajectory and pitch-location detection (not started)
  shot_outcome/            contact-quality and outcome classification (not started)
  tests/
    fixtures/              tiny synthetic video (no real footage — see above)
```

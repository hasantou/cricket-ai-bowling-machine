# cv-pipeline/

Computer-vision pipeline: pose estimation and footwork/timing feature
extraction from a batter's video, so `app/app.py` can eventually be
filled in automatically instead of a human ticking checkboxes each ball.

## What's real here

- **`pose_estimation.py`** — a genuine pretrained neural network doing
  real inference: Google's BlazePose model via MediaPipe's Tasks API,
  wrapped as `PoseEstimator`. Turns video frames into 33-point body
  landmarks per frame. This is not a placeholder or a mock — it's the
  actual model, and it will work correctly on real batting footage the
  moment it's pointed at some. It needs its ~5.5MB pretrained weights
  file, which MediaPipe's pip package doesn't bundle; `download_model()`
  fetches Google's official weights explicitly, once, only when you call
  it — nothing here downloads anything on its own.
- **`feature_extraction.py`** — pure, deterministic math over a
  landmark sequence: how far the front foot moved, front-knee bend angle
  at the peak-movement frame, leading-wrist swing speed, and which frame
  the swing started on. No model involved, fully unit-tested with
  synthetic coordinate sequences (`tests/test_feature_extraction.py`) —
  the same "test the logic honestly, independent of real footage" idea
  `adaptation-engine/simulator.py` uses for the decision engine.
- **`outcome_bridge.py`** — turns those features into the same
  `(on_time, footwork_correct)` booleans `adaptation-engine`'s scorers
  already accept from a human's checkboxes. Right now this is a
  documented, adjustable heuristic (named thresholds, not hidden magic
  numbers) — **not** a trained classifier, because there's no labelled
  real footage yet to train one on honestly. It does **not** attempt shot
  outcome (middled/edged/missed/...) — that needs ball tracking against
  the bat, which isn't built.
- **`video_pipeline.py`** — wires the three above into one call,
  `estimate_outcome_from_video(path)`: video file in, `(on_time,
  footwork_correct)` out.

## What isn't built yet

- Ball tracking (release point, trajectory, pitch location, deviation).
- Bat-ball contact analysis / shot-outcome classification — this still
  needs a human, same as today's app.
- Any wiring of this pipeline into `app/app.py` itself — right now it's
  a standalone module you can run against a video file from the command
  line or a script; hooking up an "upload a clip" mode in the app is the
  next step once there's real footage to test it against.

## Once real data exists

- `outcome_bridge.py`'s heuristic is the piece designed to be replaced
  first: once there are labelled clips (a delivery video paired with a
  coach's real timing/footwork verdict), train a small classifier over
  the same `DeliveryFeatures` fields and swap it in behind
  `VisionOutcomeEstimator.estimate()` — the interface doesn't change,
  the same migration pattern `adaptation-engine/neural_scorer.py`
  documents for its own simulated-to-real move.
- Camera-angle and left-handed-batter handling are real limitations of
  `feature_extraction.py` today (it assumes a side-on view of a
  right-handed batter) — worth fixing once real footage shows how much
  it matters in practice, rather than guessing now.

## Run it

```
python cv-pipeline/pose_estimation.py          # one-time: downloads the pretrained model
python -c "from video_pipeline import estimate_outcome_from_video; \
           print(estimate_outcome_from_video('path/to/delivery_clip.mp4'))"
```

## Tests

```
python3 -m pytest cv-pipeline/tests/ -v
```

11 tests: feature-extraction math against synthetic landmark sequences,
the outcome-bridge heuristic's branches, and a check that `PoseEstimator`
fails loudly and helpfully (not silently) when the model hasn't been
downloaded yet.

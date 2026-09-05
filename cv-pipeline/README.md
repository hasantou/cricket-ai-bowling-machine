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
- **`delivery_segmentation.py`** — finds the single delivery inside a
  longer clip. Added after running real WhatsApp footage (6-71 second
  clips of a nets session) through this pipeline for the first time:
  `feature_extraction.py`'s math assumes a clip *is* one delivery, and on
  real footage that's false — front-foot displacement over 71 seconds of
  walking between balls produced numbers that didn't mean anything. This
  module reuses the same leading-wrist swing-speed signal to find where
  the clearest swing happens and windows a fixed span around it before
  the existing feature math runs. Picks one delivery per clip (the
  clearest swing) — a clip with several deliveries back-to-back still
  only scores one of them.
- **`video_pipeline.py`** — wires the above into one call,
  `estimate_outcome_from_video(path)`: video file in, `(on_time,
  footwork_correct)` out. **Wired into `app/app.py`** — the "Estimate
  from a video clip" mode there calls this directly.

### `on_time`'s first version was broken — found and fixed against real footage

The original heuristic compared `swing_start_frame` against a fraction of
the *clip's own length*. That happened to somewhat work for a
hand-trimmed single-delivery clip, but once `delivery_segmentation.py`
started building windows centred *on the swing itself*, the check became
circular — the window's construction already guaranteed an early-looking
swing start relative to its own boundaries. Concretely: all 5 real clips
tested read `on_time: true`, regardless of the actual delivery, which was
the tell that something was structurally wrong, not evidence it worked.

Fixed by replacing it with `footwork_lead_seconds`
(`feature_extraction.py`): the real-seconds gap between two independently
measured events — when the front foot starts moving vs. when the bat
swing peaks — instead of measuring swing timing against whatever window
it happens to be inside. Re-run against the same 5 real clips, it now
discriminates (2 read `on_time: true`, 3 read `false`) instead of being
mechanically stuck on one answer. That's confirmation the structural bug
is fixed — **not** confirmation the threshold (`MIN_FOOTWORK_LEAD_SECONDS`
in `outcome_bridge.py`) is correct; there's still no ball-arrival time or
coach's verdict to check it against, only genuine variation instead of a
constant.

## What isn't built yet

- Ball tracking (release point, trajectory, pitch location, deviation).
- Bat-ball contact analysis / shot-outcome classification — this still
  needs a human, same as today's app.
- Any validation of `on_time`/`footwork_correct` against a real coach's
  independent verdict — everything so far confirms the pipeline measures
  *something* real and non-tautological, not that the something is right.
- Splitting a multi-delivery clip into every delivery in it, rather than
  just the single clearest one.

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

21 tests: feature-extraction math (including `footwork_lead_seconds`) and
delivery-segmentation windowing against synthetic landmark sequences, the
outcome-bridge heuristic's branches, and checks that `PoseEstimator` fails
loudly and helpfully (not silently) when the model hasn't been downloaded,
and that one instance can process multiple clips in sequence without
MediaPipe's video-timestamp error (a real bug found and fixed by running
actual WhatsApp footage through this pipeline — see git history).

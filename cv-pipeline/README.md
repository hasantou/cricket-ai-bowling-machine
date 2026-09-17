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
- **`delivery_segmentation.py`** — finds every delivery inside a longer
  clip. Added after running real WhatsApp footage (6-71 second clips of a
  nets session) through this pipeline for the first time:
  `feature_extraction.py`'s math assumes a clip *is* one delivery, and on
  real footage that's false — front-foot displacement over 71 seconds of
  walking between balls produced numbers that didn't mean anything. This
  module reuses the same leading-wrist swing-speed signal, greedily
  picking the strongest remaining peak and suppressing a window around it
  (so one swing's rise-and-fall isn't double-counted) until nothing above
  threshold remains. Run against the same 5 real clips: **26 deliveries
  detected in total** (2 to 8 per clip) — mechanically working and
  plausible for nets-session rotation speed, but nobody has watched the
  footage to confirm each detection is a genuine batting swing and not,
  say, a stray arm movement. `find_delivery_window()` (singular) is kept
  for callers that only want the single clearest one.
- **`video_pipeline.py`** — `estimate_outcomes_from_video(path)` (plural)
  wires the above into one call: video file in, one `(on_time,
  footwork_correct)` result per detected delivery out.
  `estimate_outcome_from_video()` (singular) wraps it for the clearest
  delivery only. **Wired into `app/app.py`** — the "Estimate from a video
  clip" mode calls the plural version and lets you pick which detected
  delivery the current log entry applies to, rather than silently
  discarding everything but the single clearest one from an uploaded
  session.

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

- **`ball_tracking.py`** — takes sparse, noisy per-frame ball detections
  (whatever a detector produces — real or synthetic) and fits one
  continuous trajectory across the whole clip via RANSAC over a
  line+parabola image-space motion model, so a detector only has to fire
  on *some* frames, not every frame, for every frame to end up with a
  usable ball position. There is no ball detector committed to this repo
  to feed it real detections from — an earlier YOLOv8 experiment (trained
  on Roboflow's CC BY 4.0 cricket-ball dataset) reached ~12% recall on
  real nets footage, weak enough, and never checked in (trained weights
  are gitignored) — so this is tested entirely against synthetic
  detections sampled at that same 12% recall rate (`tests/test_ball_tracking.py`,
  `demo_ball_tracking.py`). A real calibration bug was found and fixed
  building this: an early version tried to reject "dangerously local"
  fits (all 3 points bunched close together in time, extrapolated far
  outside where they actually looked) by requiring inliers to span a
  *fraction of the detections' own range* — which is circular exactly
  like the original `on_time` bug below, since 3 detections trivially
  span 100% of their own range no matter how clustered they are. Fixed
  by requiring a fixed amount of real time (`min_span_seconds`, default
  0.45s) instead, independent of how many detections exist — caps the
  worst-case error at 67px instead of 822px, measured across a 250-trial
  synthetic sweep at the real 12% recall rate.

## What isn't built yet

- A real ball detector — `ball_tracking.py` above assumes one exists;
  it doesn't yet (see that section for the honest history).
- Bat-ball contact analysis / shot-outcome classification — this still
  needs a human, same as today's app.
- Any validation of `on_time`/`footwork_correct` against a real coach's
  independent verdict — everything so far confirms the pipeline measures
  *something* real and non-tautological, not that the something is right.
- Confirmation that multi-delivery detection's 26-out-of-5-clips count is
  actually correct — nobody has watched the source footage ball-by-ball
  to check against it.

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
python -c "from video_pipeline import estimate_outcomes_from_video; \
           print(estimate_outcomes_from_video('path/to/clip.mp4'))"  # a whole session works, not just one ball
```

## Tests

```
python3 -m pytest cv-pipeline/tests/ -v
```

33 tests: feature-extraction math (including `footwork_lead_seconds`) and
delivery-segmentation windowing — single and multi-delivery, including
that close-together swings merge into one delivery rather than
double-counting — against synthetic landmark sequences, the
outcome-bridge heuristic's branches, checks that `PoseEstimator` fails
loudly and helpfully (not silently) when the model hasn't been downloaded,
that one instance can process multiple clips in sequence without
MediaPipe's video-timestamp error (a real bug found and fixed by running
actual WhatsApp footage through this pipeline — see git history), and
`ball_tracking.py`'s RANSAC trajectory fit against synthetic detections at
the real ~12% recall rate — dense/clean recovery, sparse+noisy recovery,
outlier rejection, refusing to fit on too little data, and a regression
test for the temporal-clustering bug described above.

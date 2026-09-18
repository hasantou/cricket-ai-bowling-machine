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
- **`motion_ball_detector.py`** — an untrained alternative to a YOLO-style
  detector: background subtraction (OpenCV's MOG2) + filtering blobs by
  size and circularity, on the theory that a cricket ball is one of the
  few small, round, fast-moving things in a nets clip. No training data
  needed, so — unlike the detector above — this could actually be run
  against real footage instead of only synthetic data, and was: **it
  didn't work.** Camera shake was ruled out first (0.2px mean frame-to-frame
  shift, measured directly). What actually happened: on a wide/distant
  clip, most flagged "candidates" traced back to the net mesh flickering
  against the sky (wind + compression noise on fine repeating texture),
  confirmed by drawing every candidate on real frames; turning sensitivity
  down 8x cut the count but a manually re-checked survivor was a player's
  hand, not the ball. On a second, closer-camera clip, directly hunting for
  the ball — zoomed crops of the flight corridor, a frame-diff peak trace
  outside the bowler's own body — never found it either; the one trace that
  looked like real continuous motion turned out to be the batter's glove
  shifting. **The honest conclusion this testing supports: on this
  footage, the ball is at or below the visibility floor for a human
  reviewer, not just an algorithm** — likely small size + motion blur +
  video compression stacking together. That points at the camera setup
  (distance, zoom, shutter speed) as the actual blocker, not detector
  choice — worth fixing before spending more effort on any ball-detection
  approach, trained or classical. Kept in the repo, tested against
  synthetic data (5 tests, `tests/test_motion_ball_detector.py`) exactly
  like `ball_tracking.py`, in case footage from a closer/better-positioned
  camera makes the underlying idea viable later. Now also accepts an
  `roi` (region of interest) parameter — a direct, no-new-hardware fix
  drawn from the real-footage testing above: most false positives sat
  well outside where the ball could plausibly be, so restricting
  detection to a region around the batting crease removes a whole class
  of them for free.
- **Extracted from a proposed stereo-camera exit-trajectory design**
  (post-shot ball tracking via two synchronised cameras, triangulation,
  and a kinematic fit) — the parts of that design that are genuinely
  useful in software today, independent of whether that hardware ever
  exists: `ball_tracking.py` gained `find_impact_and_split()` (detects
  bat-ball contact from a velocity discontinuity between consecutive
  detections, implementing an Incoming → Impact → Outgoing model — a
  ball's flight before being struck tells you nothing about where it's
  going after) and `trim_before_deceleration_spike()` (cuts a post-impact
  sequence at the first sign of net-collision corruption, via a spike in
  the second derivative of position). Both work in this module's existing
  2D image-space domain — no camera calibration or real 3D needed — and
  are tested against synthetic detection sequences
  (`tests/test_ball_tracking.py`).

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

45 tests: feature-extraction math (including `footwork_lead_seconds`) and
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
test for the temporal-clustering bug described above — plus
`motion_ball_detector.py`'s blob filtering against synthetic frames
(detects a small moving circle, rejects a large rectangle and a thin
elongated sliver, per-frame best-candidate selection, an ROI correctly
excluding an out-of-region candidate while keeping coordinates in the
original frame's space, a clear error on a missing file), and
`find_impact_and_split()`/`trim_before_deceleration_spike()` against
synthetic velocity-discontinuity and deceleration-spike sequences. Its
real-footage evaluation (see above) was run by hand, not as part of this
suite, since there's no ground truth to assert against — only visual
inspection.

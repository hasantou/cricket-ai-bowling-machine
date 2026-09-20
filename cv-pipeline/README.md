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

## Real-time readiness: a camera mounted on the machine, not an uploaded file

Everything above this point processes a whole pre-recorded clip in one
batch call. That's fine for "upload a session afterward," but the actual
target is a camera physically mounted on the machine, running
continuously — so this was built and, critically, **validated against
real footage played back frame by frame as if it were live**, not just
synthetic data, because a file and a live camera look identical to
`cv2.VideoCapture.read()` in a loop.

- **`live_video_source.py`** — `LiveVideoSource` wraps a camera index,
  stream URL, or file (all identical to `cv2.VideoCapture`) behind one
  `frames()` generator, or accepts an injected frame source for testing
  without hardware.
- **`live_delivery_detector.py`** — `LiveDeliveryDetector` wraps
  `delivery_segmentation.py`'s existing batch peak-finding logic (not a
  second copy of it) for a continuously-arriving stream, returning
  `DetectedDelivery` objects with both buffer-relative coordinates (to
  slice the current buffer right away) and absolute coordinates (stable
  across the whole session, safe to log or compare).
- **`pose_estimation.py`** gained `extract_landmarks_from_one_live_frame()`
  — the same real MediaPipe model, called once per live frame with a
  correctly-incrementing internal clock (see below for why that needed
  to be a separate method).
- **`demo_live_delivery_detection.py`** — plays a real WhatsApp clip back
  frame by frame through the live pipeline and cross-checks the result
  against the existing batch pipeline on the same clip. Run against 3 of
  the real clips already used elsewhere in this project's testing; all 3
  now match the batch pipeline exactly on every delivery a genuinely
  continuous camera would also have completed.

**Three real bugs were found this way, not assumed up front** — each one
only showed up on real footage, not synthetic tests, and each is now a
regression test:

1. Calling the batch pose-extraction method once per single live frame
   compressed MediaPipe's internal clock to 1ms between frames instead of
   a real ~33ms frame interval, since that method's "advance by 1ms"
   convention is meant for starting a new, unrelated clip, not "the next
   frame arrived shortly after." This measurably changed which deliveries
   got detected. Fixed with a dedicated live method that advances the
   clock by a real frame interval every call.
2. `find_delivery_windows()` is a *global* greedy algorithm — the single
   biggest wrist-speed peak anywhere in the buffer wins, suppressing a
   2-second radius around it. Re-running it on a growing live buffer
   means an early, smaller peak can look "complete" (enough trailing
   context exists for *it*) before the real, bigger swing nearby has even
   happened — and once it does, the batch algorithm would have suppressed
   the smaller one entirely. Fixed by not trusting a peak until the
   buffer extends the full 2-second suppression radius past it, not just
   the ~0.8s window needed to draw its box.
3. On a longer real session, the buffer-relative "already reported"
   marker silently drained toward zero during quiet gaps between
   deliveries as old frames got trimmed off the front — so an old,
   unchanged delivery already sitting in the buffer could look unreported
   again and fire a second (on one real clip: sixth) time. Fixed by
   tracking "reported up to" as an absolute, monotonically-increasing
   position that trimming can never erode — which is also why
   `DetectedDelivery` exposes both coordinate systems explicitly instead
   of one bare tuple: this project's own validation script mixed the two
   up on its first attempt, which is exactly the mistake a real caller
   could make too.

**What this does and doesn't prove**: the pipeline correctly detects
deliveries and computes matching footwork/timing results in real time
from real footage, sustained around 28-30fps on ordinary CPU inference —
comfortably fast enough for this project's actual timing budget (a
delivery roughly every 7-11 seconds, per BOLA's own spec). It does *not*
prove a real camera mounted on a real machine works, since no such camera
exists yet — `LiveVideoSource` opening a real device is unexercised code,
identical in shape to opening a file, but unexercised all the same.

## Is this clip good enough? And what about the bowler?

Two additions for "how the ball was bowled and how the batter did" from
video — both deliberately limited to what a phone clip can honestly show.

**`footage_check.py`** turns a silent failure into a visible warning. Pose
analysis on a poor clip still produces plausible-looking footwork/timing
verdicts; nothing in the output says the input was bad. It measures how big
the tracked person is, how sure the model is of the joints, how often a
person is found, and whether the file decoded fully (frames read vs.
frames the container promises), and gives good / marginal / poor with
reasons. Thresholds are judgment calls from a handful of clips, not
calibrated. (Checking it on real footage also corrected one of my own
guesses: I had assumed the arena clip's batter would be tiny; measured, they
are 37% of the frame.)

**`bowler_analysis.py`** reads the bowler's *body*: arm side, projected arm
angle (high / three-quarter / round-arm), release height, run-up pace and
arm speed (in torso-lengths per second — one camera has no real-world
scale), and release-to-swing time. It runs a second multi-person pose pass
and follows people frame to frame. It does **not** and cannot say ball
speed, line, length, swing or spin from video — the machine's
`DeliveryReport` (`trajectory-engine/`) and its release/impact sensors
provide those.

What testing against real footage found, honestly:

- The first version returned nonsense on a real clip (it measured against
  nose-to-ankle height, which collapses when a bowler bends double after
  release). Rebuilt on torso length with an upright-torso requirement.
- A scan of the other real clips reported "bowlers" with the wrist 11–15
  torso-lengths above the shoulder — physically impossible (a near-zero
  torso length from a glitchy skeleton). Such frames are now discarded.
- The remaining "bowler" hits, when I looked at the actual frame, were the
  **batter** stepping out and lifting the bat. Pose alone cannot tell a
  backlift from a delivery, so the known batter (the person the batter
  analysis followed) is now excluded from bowler candidates.
- After that, none of the real clips I have shows a bowler close enough to
  analyse, and the app correctly says "no overhead bowling action found".
  **The positive case — correctly reading a real bowler's action — is
  tested on synthetic skeletons only, not yet validated on real footage.**
  It needs a clip with the bowler in shot and large enough.

## Body vectors and shot type from video

**`body_vectors.py`** turns pose landmarks into movement vectors: for every
joint, which way it is heading and how fast (torso-lengths per second, so
camera distance doesn't matter), plus the hands' path through the swing, hip
(weight) shift, foot movement and shoulder/hip-line rotation.
**`body_vector_render.py`** draws them on the frame (an arrow per joint,
showing where it is heading over the next 0.15 s; grey and arrow-less for
joints the model is unsure of). The app shows this per delivery.

What real footage taught it, in order:

- **Raw landmark differences are noise.** Unsmoothed, a batter merely taking
  guard scored as fast as a real swing. (An earlier version of this README
  said pose *cannot* separate a swing from taking guard. That was wrong: it
  was an artifact of using unsmoothed jitter.) Tracks are now confidence-gated
  and smoothed; on the arena clip, smoothed peak hand speeds were ~7-9
  torso-lengths/s for the two real swings and ~1-3 for the two stances,
  which I checked by eye. That is four windows on one clip, not a validation.
- **The whole body moving is not a swing.** A batter drifting out of frame
  moved every joint together and was first called a swing. Hands are now
  measured relative to the hip centre, so walking or a panning camera cancels.
- **Some speeds are impossible.** A second clip gave "hand speeds" of 36 and
  60 torso-lengths/s (pose glitches on a small, distant batter). Above 25 is
  flagged as a glitch and the verdict is "unclear".
- **Hands are often not visible.** From behind the batter the model was
  confident in the wrists in only 13-49% of frames, and both legs project onto
  one column. Every joint carries good / partial / poor confidence, and if
  neither wrist is usable the trunk stands in and the report says so.

**`shot_from_video.py`** is the shot-type algorithm for good footage: swing or
not, then defensive push / drive-type / horizontal-bat, then off/leg side from
the hands' path and the camera position, then a name from the shot vocabulary
(straight/off/cover drive, on drive, flick, cut, pull, hook, leave). Read its
docstring before trusting it:

- Its thresholds are **rules of thumb, not fitted to labelled shots**. There is
  no accuracy figure. `shot_labels.py:evaluate()` gives one once labelled shots
  exist; the thresholds are what to tune.
- It reads the **hands, not the ball**. The bat face at contact decides where
  the ball goes. The machine's sensor path is the stronger source.
- It **refuses** ("cannot tell", with the reason) unless the footage is good,
  the batter is tracked in nearly every frame, and the fast hand itself was
  tracked with high confidence. On the arena clip it refuses every delivery,
  because from behind the hands are only partly visible. It has therefore
  **never named a shot on real footage**; the naming rules are tested on
  synthetic data only.
- The caller must give the camera position and batting hand; from side-on it
  names only the shot family, never a side.

**Built so real-world tests make it stronger** (protocol:
`docs/real_world_test_protocol.md`):

- **Camera suitability** (`footage_check.py`): besides "is the clip usable", it
  now reports `hand_visibility` (share of frames with both wrists confidently
  seen), leg separation and a `shot_reading_ok` verdict with reasons. Measured
  on the two real clips: hands confidently seen in only **7%** (behind the
  batter) and **30%** (small, side-on) of frames, so neither camera position can
  read a swing. That number is what to check first at any new camera position.
- **Tunable rules** (`shot_from_video.ShotRules`): every threshold is a field
  of one value; `classify_features()` is a pure function of the four measured
  numbers, so it can be tuned and scored offline.
- **Trial log + calibrator** (`shot_calibration.py`): labelled deliveries
  (`TrialRecord`) are stored as JSON lines; `evaluate_rules()` scores the
  algorithm against them with a confusion table; `fit_rules()` tunes the
  thresholds by coordinate descent. Guardrails: learns only from deliveries the
  algorithm trusted; refuses under 30 of them; reports **held-out** k-fold
  accuracy next to the current rules' held-out accuracy; adopts the fit only if
  it wins by 3 points; never overwrites the defaults itself; saved rules carry
  their evidence. The tests show the fitting mechanism recovers known
  thresholds from synthetic data (including with 15% wrong labels, and refusing
  to "adopt" a fit to random labels). **They do not show that real shots separate
  cleanly by hand path** - only real labelled trials can.
- **In the app**: "Real-world trial" lets a person pick each delivery's true
  shot, log it, see the agreement score, tune once 30 trusted examples exist,
  and download the log (the hosted app's storage is temporary).

Also in `trajectory-engine`: `shot_fusion.fuse_shot()` combines the sensor's
shot (the ball) with the video's (the hands): agreement corroborates, a
disagreement is reported and the sensor's name used.

`shot_labels.py` finds which frame of a clip an annotated screenshot is
(`locate_frame_in_video`), `labels/shot_labels.json` records the annotations
(first entry: arena clip, frame 111, shot name still blank), and `evaluate()`
scores predictions against labels and reports how few it rests on.

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
python cv-pipeline/demo_live_delivery_detection.py path/to/clip.mp4   # the live-shaped pipeline, validated against the same clip's batch result
```

## Tests

```
python3 -m pytest cv-pipeline/tests/ -v
```

147 tests: feature-extraction math (including `footwork_lead_seconds`) and
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
synthetic velocity-discontinuity and deceleration-spike sequences.
`motion_ball_detector.py`'s real-footage evaluation (see above) was run
by hand, not as part of this suite, since there's no ground truth to
assert against — only visual inspection. `live_video_source.py` and
`live_delivery_detector.py` add 12 more against synthetic streams,
including three regression tests locking in the real bugs described
above — but the real proof for the live pipeline is
`demo_live_delivery_detection.py` run against actual footage (see that
section), which needs a real clip and the downloaded pose model, so it
isn't part of the automated `pytest` run.

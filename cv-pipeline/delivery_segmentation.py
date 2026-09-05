"""
Finds the delivery inside a longer clip — real net-session footage runs
anywhere from a few seconds to over a minute (confirmed against actual
WhatsApp footage: 6s to 71s), not the tightly trimmed single-delivery
clips feature_extraction.py's math otherwise assumes. Feeding a whole
71-second session into that math produces meaningless numbers: front-foot
displacement over a minute of walking/resetting between balls, or an
"on time" check comparing swing timing against a session's length instead
of one delivery's — this is exactly what running real footage through the
pipeline exposed (see cv-pipeline/README.md).

Approach: reuse the same signal extract_delivery_features already computes
internally — frame-to-frame leading-wrist speed — to find where the single
clearest swing happens in the whole clip, then window a fixed span of time
around that peak. This picks ONE delivery per clip (the most prominent
swing). A clip showing several deliveries back-to-back still only scores
the clearest one; splitting a clip into every individual delivery is real
future work, not solved here — call this out again anywhere this module's
output gets summarised, so nobody mistakes "found one delivery" for "found
every delivery in this clip".

Known limitation, not hidden: PoseEstimator.extract_landmarks_from_frames()
skips frames where no person was detected, so the landmark list's frame
indices don't perfectly line up with wall-clock time if detection ever
drops out mid-clip. Windowing here treats them as evenly spaced at `fps`
regardless — a reasonable approximation for clips with mostly-continuous
detection (true of every real clip tested so far), but a real source of
timing drift on a clip with significant detection gaps.
"""

from typing import List, Tuple

from feature_extraction import LEFT_WRIST, FrameLandmarks, _dist, _point

# Named, adjustable — not hidden magic numbers. A delivery's visible
# action (batter loads/strides, then swings) fits comfortably inside this
# span; tune against real footage as more of it comes in.
WINDOW_BEFORE_PEAK_SEC = 0.6
WINDOW_AFTER_PEAK_SEC = 0.8

# Below this frame-to-frame wrist speed, there's no genuine swing in the
# clip to anchor a window on — fail loudly rather than guessing a window
# around noise, the same "no silent fake success" pattern used throughout
# this pipeline (see pose_estimation.py, neural_scorer.py).
MIN_PEAK_SPEED = 0.01


def find_delivery_window(frames: List[FrameLandmarks], fps: float) -> Tuple[int, int]:
    """Returns (start, end) indices into `frames` bounding the single most
    prominent delivery detected, using leading-wrist swing speed as the
    signal. `frames[start:end]` is what should be handed to
    extract_delivery_features(), not the full clip.

    Raises ValueError if fewer than 2 frames are given, or if no swing
    clears MIN_PEAK_SPEED (e.g. a clip with no batting action in it).
    """
    if len(frames) < 2:
        raise ValueError("find_delivery_window requires at least 2 frames")

    wrist_positions = [_point(f, LEFT_WRIST) for f in frames]
    frame_speeds = [_dist(wrist_positions[i - 1], wrist_positions[i]) for i in range(1, len(wrist_positions))]

    peak_speed = max(frame_speeds)
    if peak_speed < MIN_PEAK_SPEED:
        raise ValueError(
            f"No clear delivery detected in this clip (peak wrist speed "
            f"{peak_speed:.4f} is below the {MIN_PEAK_SPEED} minimum) — "
            "check the clip actually shows a batting action."
        )

    peak_idx = frame_speeds.index(peak_speed) + 1  # +1: frame_speeds[i] is between frames[i] and frames[i+1]

    before = int(WINDOW_BEFORE_PEAK_SEC * fps)
    after = int(WINDOW_AFTER_PEAK_SEC * fps)
    start = max(0, peak_idx - before)
    end = min(len(frames), peak_idx + after + 1)
    return start, end

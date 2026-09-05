"""
Finds deliveries inside a longer clip — real net-session footage runs
anywhere from a few seconds to over a minute (confirmed against actual
WhatsApp footage: 6s to 71s), not the tightly trimmed single-delivery
clips feature_extraction.py's math otherwise assumes. Feeding a whole
71-second session into that math produces meaningless numbers: front-foot
displacement over a minute of walking/resetting between balls — this is
exactly what running real footage through the pipeline exposed (see
cv-pipeline/README.md).

Approach: reuse the same signal extract_delivery_features already computes
internally — frame-to-frame leading-wrist speed — to find every clear
swing in the whole clip (a greedy peak-pick: take the strongest remaining
peak, suppress a window around it so the same swing's rise-and-fall
doesn't get counted twice, repeat), then window a fixed span of time
around each one. A nets-session clip showing several deliveries
back-to-back gets scored as several deliveries, not just the loudest one.

Known limitation, not hidden: PoseEstimator.extract_landmarks_from_frames()
skips frames where no person was detected, so the landmark list's frame
indices don't perfectly line up with wall-clock time if detection ever
drops out mid-clip. Windowing here treats them as evenly spaced at `fps`
regardless — a reasonable approximation for clips with mostly-continuous
detection (true of every real clip tested so far), but a real source of
timing drift on a clip with significant detection gaps. Also untested:
two deliveries genuinely closer together than MIN_DELIVERY_SEPARATION_SEC
would still be merged into one — real footage so far hasn't had balls
bowled that close together, but nothing here proves it can't happen.
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

# Two swing peaks closer together than this are treated as the same
# delivery (its natural rise-and-fall can produce more than one
# above-threshold frame) rather than two separate balls. A real bowler
# needs meaningfully longer than this between deliveries in practice;
# this is a conservative first guess, not calibrated against real footage
# with genuinely back-to-back balls yet.
MIN_DELIVERY_SEPARATION_SEC = 2.0


def find_delivery_windows(
    frames: List[FrameLandmarks], fps: float, max_deliveries: int = None
) -> List[Tuple[int, int]]:
    """Returns a list of (start, end) index pairs into `frames`, one per
    detected delivery, ordered by when they occur in the clip.
    `frames[start:end]` for each pair is what should be handed to
    extract_delivery_features().

    Raises ValueError if fewer than 2 frames are given, or if no swing
    anywhere in the clip clears MIN_PEAK_SPEED (e.g. a clip with no
    batting action in it).
    """
    if len(frames) < 2:
        raise ValueError("find_delivery_windows requires at least 2 frames")

    wrist_positions = [_point(f, LEFT_WRIST) for f in frames]
    frame_speeds = [_dist(wrist_positions[i - 1], wrist_positions[i]) for i in range(1, len(wrist_positions))]

    remaining = list(frame_speeds)
    separation_frames = int(MIN_DELIVERY_SEPARATION_SEC * fps)
    peak_indices = []
    while max_deliveries is None or len(peak_indices) < max_deliveries:
        peak_speed = max(remaining)
        if peak_speed < MIN_PEAK_SPEED:
            break
        peak_i = remaining.index(peak_speed)
        peak_indices.append(peak_i)
        lo = max(0, peak_i - separation_frames)
        hi = min(len(remaining), peak_i + separation_frames + 1)
        for j in range(lo, hi):
            remaining[j] = -1.0  # suppressed — already claimed by this peak

    if not peak_indices:
        raise ValueError(
            f"No clear delivery detected in this clip (peak wrist speed "
            f"{max(frame_speeds):.4f} is below the {MIN_PEAK_SPEED} minimum) — "
            "check the clip actually shows a batting action."
        )

    peak_indices.sort()
    before = int(WINDOW_BEFORE_PEAK_SEC * fps)
    after = int(WINDOW_AFTER_PEAK_SEC * fps)
    windows = []
    for peak_i in peak_indices:
        peak_idx = peak_i + 1  # +1: frame_speeds[i] is between frames[i] and frames[i+1]
        start = max(0, peak_idx - before)
        end = min(len(frames), peak_idx + after + 1)
        windows.append((start, end))
    return windows


def find_delivery_window(frames: List[FrameLandmarks], fps: float) -> Tuple[int, int]:
    """Backwards-compatible single-delivery version: the span around the
    single clearest swing (highest peak speed) in the clip. Prefer
    find_delivery_windows() for anything that might contain more than one
    delivery — real nets-session footage usually does."""
    return find_delivery_windows(frames, fps, max_deliveries=1)[0]

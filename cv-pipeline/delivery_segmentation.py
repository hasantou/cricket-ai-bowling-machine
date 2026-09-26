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

A real, found-on-real-footage bug this module does NOT fully fix on its
own: a broadcast highlight clip often shows the SAME delivery more than
once — a normal-speed shot, then a slow-motion replay, sometimes from a
different camera angle — and each showing has its own genuine-looking
swing peak. Confirmed on a real wicket clip: multiple windows were found
for what was, on inspection, one ball shown more than once. Nothing about
the wrist-speed signal alone can tell "a second real delivery" apart from
"a replay of the first one" — they look identical from the batter's wrist
alone. `check_windows_for_a_bowler()` was built as a fix — a genuine
delivery is always preceded by a bowler's run-up, a replay never shows
one — but tested on the real clip this was built for, it flagged a
genuine (not replayed) delivery for the same reason a real replay would
be flagged: the highlight edit itself started right at release, with the
run-up trimmed out of the footage entirely, not by anything happening in
the clip. Both cases look identical to this check. It therefore only
FLAGS a window as worth a human glance — see its own docstring — it does
not, and must not, silently decide anything on its own.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

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


# Named, adjustable: how far back from a candidate window to look for a bowler's
# run-up. A run-up plus delivery stride comfortably fits inside 3 seconds; much
# longer risks picking up an unrelated PREVIOUS delivery's run-up instead.
BOWLER_LOOKBACK_SEC = 3.0


@dataclass(frozen=True)
class BowlerCheckedWindow:
    window: Tuple[int, int]
    bowler_seen: Optional[bool]  # True: a real run-up preceded it. False: no bowler found in real lookback
                                  # footage. None: no lookback footage existed at all to check (genuinely unknown).
    note: str


def check_windows_for_a_bowler(
    windows: Sequence[Tuple[int, int]],
    frames_multi: Sequence[Sequence[FrameLandmarks]],
    fps: float,
    lookback_sec: float = BOWLER_LOOKBACK_SEC,
    batter_hip: Optional[Tuple[float, float]] = None,
) -> List[BowlerCheckedWindow]:
    """FLAGS windows from find_delivery_windows() that aren't preceded by a
    real bowler's run-up — it never drops one itself. The idea a broadcast
    highlight clip replaying the SAME delivery (slow-motion, a different
    camera angle, a reaction shot) produces its own convincing wrist-speed
    peak, indistinguishable from a second real ball by that signal alone,
    while a replay never re-shows someone running in to bowl, is real and
    the check for it is real. What real-clip testing found is that this
    can't be trusted to decide on its own:

    Tested on the real wicket clip this was built for, the ONE window found
    was flagged `bowler_seen=False` — but on inspection that almost
    certainly wasn't a replay at all. It's a highlight edit that starts
    right at the bowler's release, with the run-up itself trimmed away by
    whoever cut the clip — completely normal editing, nothing to do with
    being a duplicate. That looks IDENTICAL to this check as a genuine
    replay: no bowler visible before the swing, either way. Given that
    real, confirmed failure mode, this function only flags for a human to
    glance at — it must never be wired up to silently discard a window.

    `frames_multi[i]` is every skeleton found in frame i, across the WHOLE
    clip (bowler_analysis.track_people()'s input shape) — not the single
    already-isolated batter track find_delivery_windows() itself consumes,
    since telling a bowler and a batter apart needs seeing both people.

    `bowler_seen=None` (rather than False) when a window starts at or near
    frame 0 — there is no "before" footage at all to judge, which is a
    different, more honest state than "checked and found nothing."
    """
    from bowler_analysis import analyse_bowler_action  # deferred: keeps this module usable without the pose-heavy bowler_analysis import unless this specific function is actually called

    results: List[BowlerCheckedWindow] = []
    lookback = int(lookback_sec * fps)
    for start, end in windows:
        lo = max(0, start - lookback)
        if lo >= start:
            results.append(BowlerCheckedWindow(
                (start, end), None, "No footage exists before this window to check at all.",
            ))
            continue
        segment = frames_multi[lo:start]
        if analyse_bowler_action(segment, fps, batter_hip=batter_hip) is not None:
            results.append(BowlerCheckedWindow((start, end), True, "A bowler's run-up was found beforehand."))
        else:
            results.append(BowlerCheckedWindow(
                (start, end), False,
                f"No bowler run-up found in the {lookback_sec:.1f}s before this window — worth a quick look: "
                "could be a replay of an earlier delivery, or just footage trimmed before the run-up.",
            ))
    return results

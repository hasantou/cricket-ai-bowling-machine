import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from delivery_segmentation import find_delivery_windows, WINDOW_AFTER_PEAK_SEC
from feature_extraction import LEFT_WRIST
from live_delivery_detector import LiveDeliveryDetector

N_LANDMARKS = 33
FPS = 30.0
TRAILING_FRAMES = int(WINDOW_AFTER_PEAK_SEC * FPS)


def _blank_frame(x=0.5, y=0.5):
    return [(x, y, 1.0) for _ in range(N_LANDMARKS)]


def _still_frames(n):
    return [_blank_frame() for _ in range(n)]


def _with_swing_at(frames, peak_idx, step=0.1):
    frames = [list(f) for f in frames]
    for offset, mult in ((-1, 0.3), (0, 1.0), (1, 0.3)):
        idx = peak_idx + offset
        if 0 <= idx < len(frames):
            frames[idx][LEFT_WRIST] = (0.5 + step * mult, 0.5, 1.0)
    return frames


def test_reports_nothing_until_enough_trailing_context_has_arrived():
    frames = _with_swing_at(_still_frames(300), peak_idx=150)
    detector = LiveDeliveryDetector(fps=FPS, buffer_seconds=20.0)

    reported = []
    for i, landmarks in enumerate(frames):
        windows = detector.add_frame(landmarks)
        if windows:
            reported.append((i, windows))
        # Before enough trailing frames past the peak have arrived, nothing
        # should be reported yet.
        if i < 150 + TRAILING_FRAMES:
            assert windows == [], f"reported a window too early, at live frame {i}"

    assert len(reported) == 1, f"expected exactly one report event, got {reported}"


def test_live_reported_window_matches_the_batch_result_on_the_same_data():
    frames = _with_swing_at(_still_frames(300), peak_idx=150)
    detector = LiveDeliveryDetector(fps=FPS, buffer_seconds=20.0)

    live_windows = []
    for landmarks in frames:
        live_windows.extend(detector.add_frame(landmarks))

    batch_windows = find_delivery_windows(frames, fps=FPS)
    assert [(d.start, d.end) for d in live_windows] == batch_windows
    # With no trimming in this short a clip, buffer-relative and absolute
    # positions should coincide exactly too.
    assert [(d.absolute_start, d.absolute_end) for d in live_windows] == batch_windows


def test_a_window_is_never_reported_more_than_once():
    frames = _with_swing_at(_still_frames(400), peak_idx=150)
    detector = LiveDeliveryDetector(fps=FPS, buffer_seconds=20.0)

    all_reported = []
    for landmarks in frames:
        all_reported.extend(detector.add_frame(landmarks))

    assert len(all_reported) == len(set(all_reported)) == 1


def test_two_separate_deliveries_are_both_eventually_reported_once_each():
    frames = _still_frames(600)
    frames = [list(f) for f in frames]
    for offset, mult in ((-1, 0.3), (0, 1.0), (1, 0.3)):
        frames[100 + offset][LEFT_WRIST] = (0.5 + 0.1 * mult, 0.5, 1.0)
        frames[400 + offset][LEFT_WRIST] = (0.5 + 0.1 * mult, 0.5, 1.0)

    detector = LiveDeliveryDetector(fps=FPS, buffer_seconds=20.0)
    all_reported = []
    for landmarks in frames:
        all_reported.extend(detector.add_frame(landmarks))

    assert len(all_reported) == 2
    assert len(set(all_reported)) == 2


def test_buffer_trimming_does_not_corrupt_or_duplicate_reports():
    # A short buffer relative to the clip forces mid-session trimming -
    # confirms _reported_up_to bookkeeping survives the buffer shrinking
    # out from under already-reported windows.
    frames = _still_frames(600)
    frames = [list(f) for f in frames]
    for offset, mult in ((-1, 0.3), (0, 1.0), (1, 0.3)):
        frames[100 + offset][LEFT_WRIST] = (0.5 + 0.1 * mult, 0.5, 1.0)
        frames[400 + offset][LEFT_WRIST] = (0.5 + 0.1 * mult, 0.5, 1.0)

    detector = LiveDeliveryDetector(fps=FPS, buffer_seconds=5.0)  # 150 frames - much shorter than the clip
    all_reported = []
    for landmarks in frames:
        all_reported.extend(detector.add_frame(landmarks))

    # Both swings are still far enough apart, and each is followed by
    # plenty of still frames before the buffer would trim past it, so
    # both should still be found exactly once despite trimming.
    assert len(all_reported) == 2


def test_regression_a_long_quiet_gap_does_not_cause_an_old_delivery_to_be_reported_twice():
    """Regression test for a real bug found on a ~75s, 9-delivery real
    clip: one delivery got reported six times over. Cause: with a buffer
    much shorter than the gap between deliveries, continuous trimming
    during the quiet gap silently drained a buffer-relative "already
    reported" marker back toward zero, so once it fell below the OLD
    delivery's (unchanged, still-buffered) start position, that old
    window looked unreported again. Reproduced here with a short buffer
    and a long quiet gap between two deliveries."""
    frames = _still_frames(700)
    frames = [list(f) for f in frames]
    for offset, mult in ((-1, 0.3), (0, 1.0), (1, 0.3)):
        frames[60 + offset][LEFT_WRIST] = (0.5 + 0.1 * mult, 0.5, 1.0)
        frames[500 + offset][LEFT_WRIST] = (0.5 + 0.1 * mult, 0.5, 1.0)

    # 3-second buffer (90 frames) against a ~440-frame (14.6s) quiet gap
    # between the two deliveries - exactly the shape that reproduced the
    # real bug. Note: with a short sliding buffer, two DIFFERENT real
    # deliveries can legitimately land on identical buffer-relative
    # (start, end) coordinates - that's expected, not a duplicate - so
    # this checks the number and timing of report *events*, not raw
    # tuple uniqueness.
    detector = LiveDeliveryDetector(fps=FPS, buffer_seconds=3.0)
    report_events = []  # (frame_index, windows) for every call that reported something
    for i, landmarks in enumerate(frames):
        windows = detector.add_frame(landmarks)
        if windows:
            report_events.append((i, windows))

    assert len(report_events) == 2, f"expected exactly 2 report events, got: {report_events}"
    first_frame, second_frame = report_events[0][0], report_events[1][0]
    # The two real swings are ~440 frames apart - two genuine reports
    # should be similarly far apart in when they fire, not clustered
    # together (which is what the original bug looked like: six reports
    # firing back-to-back for what should have been one).
    assert second_frame - first_frame > 300, (
        f"the two reports fired suspiciously close together ({first_frame} vs {second_frame}) - "
        "looks like stale re-reporting, not two distinct deliveries"
    )

    # The real motivation for absolute_start/absolute_end existing at all:
    # with this short a buffer, the two deliveries' buffer-relative (start,
    # end) can legitimately be identical (confirmed on real footage), so
    # only the absolute positions reliably tell them apart.
    first_delivery = report_events[0][1][0]
    second_delivery = report_events[1][1][0]
    assert first_delivery.absolute_start != second_delivery.absolute_start
    assert second_delivery.absolute_start - first_delivery.absolute_start > 300


def test_regression_an_early_minor_peak_is_not_reported_before_a_bigger_nearby_peak_arrives():
    """Regression test for a real bug found running this against real
    footage: an early, smaller wrist-speed peak looked "complete" (enough
    WINDOW_AFTER_PEAK_SEC trailing context existed for it) well before a
    bigger, genuine swing nearby had even happened yet. find_delivery_windows()
    is a global greedy algorithm - it would have picked the bigger peak
    first and suppressed the smaller one entirely (they're within
    MIN_DELIVERY_SEPARATION_SEC of each other), so reporting the smaller
    one early is reporting something the batch algorithm itself never
    would have selected. Mirrors the real clip's shape: a minor peak at
    frame 60, a much later - but still within the 2s suppression radius -
    genuine peak at frame 100."""
    frames = _still_frames(200)
    frames = [list(f) for f in frames]
    # A small early wobble - big enough to clear MIN_PEAK_SPEED, small
    # enough that a real swing nearby would dominate it.
    for offset, mult in ((-1, 0.15), (0, 0.5), (1, 0.15)):
        frames[60 + offset][LEFT_WRIST] = (0.5 + 0.1 * mult, 0.5, 1.0)
    # The genuine, bigger swing - close enough (40 frames, ~1.3s at 30fps)
    # to fall within MIN_DELIVERY_SEPARATION_SEC (2s = 60 frames) of the
    # wobble above, so the batch algorithm would suppress the wobble.
    for offset, mult in ((-1, 0.3), (0, 1.0), (1, 0.3)):
        frames[100 + offset][LEFT_WRIST] = (0.5 + 0.1 * mult, 0.5, 1.0)

    detector = LiveDeliveryDetector(fps=FPS, buffer_seconds=20.0)
    reported_before_the_real_swing_is_even_visible = []
    for i, landmarks in enumerate(frames):
        windows = detector.add_frame(landmarks)
        if i < 100 and windows:
            reported_before_the_real_swing_is_even_visible.append((i, windows))

    assert reported_before_the_real_swing_is_even_visible == [], (
        f"reported a window the batch algorithm would have suppressed: "
        f"{reported_before_the_real_swing_is_even_visible}"
    )

    # And it should agree with the batch result on the full sequence -
    # the real swing at frame 100 only, not the earlier wobble. Run a
    # fresh detector rather than reusing the one above, which already
    # consumed the stream.
    all_reported = []
    detector2 = LiveDeliveryDetector(fps=FPS, buffer_seconds=20.0)
    for landmarks in frames:
        all_reported.extend(detector2.add_frame(landmarks))
    batch_windows = find_delivery_windows(frames, fps=FPS)
    assert [(d.start, d.end) for d in all_reported] == batch_windows
    assert len(all_reported) == 1  # the wobble was correctly suppressed, matching batch


def test_skips_none_frames_without_treating_them_as_landmarks():
    frames = _with_swing_at(_still_frames(300), peak_idx=150)
    detector = LiveDeliveryDetector(fps=FPS, buffer_seconds=20.0)

    reported = []
    for i, landmarks in enumerate(frames):
        # Simulate a detection dropout every 10th frame.
        reported.extend(detector.add_frame(None if i % 10 == 0 else landmarks))

    assert len(reported) == 1

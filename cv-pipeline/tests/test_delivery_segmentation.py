import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from delivery_segmentation import find_delivery_window, find_delivery_windows
from feature_extraction import LEFT_WRIST

N_LANDMARKS = 33
FPS = 30.0


def _blank_frame(x=0.5, y=0.5):
    return [(x, y, 1.0) for _ in range(N_LANDMARKS)]


def _still_frames(n):
    return [_blank_frame() for _ in range(n)]


def _with_swings_at(frames, peak_indices, step=0.1):
    """Injects a sharp wrist movement centred on each index in
    peak_indices into an otherwise-still sequence, simulating one or more
    clear swings inside a longer, mostly-static clip (a batter standing
    between deliveries)."""
    frames = [list(f) for f in frames]
    for peak_idx in peak_indices:
        for offset, mult in ((-1, 0.3), (0, 1.0), (1, 0.3)):
            idx = peak_idx + offset
            if 0 <= idx < len(frames):
                frames[idx][LEFT_WRIST] = (0.5 + step * mult, 0.5, 1.0)
    return frames


def _with_swing_at(frames, peak_idx, step=0.1):
    return _with_swings_at(frames, [peak_idx], step=step)


def test_finds_window_centred_on_the_swing_in_a_long_mostly_still_clip():
    # 300 still frames (10s at 30fps) representing session dead time, with
    # one clear swing spike in the middle — the realistic shape of the
    # real WhatsApp footage this was built to handle.
    frames = _with_swing_at(_still_frames(300), peak_idx=150)
    start, end = find_delivery_window(frames, fps=FPS)
    assert start < 150 < end
    assert 0 <= start < end <= len(frames)


def test_window_is_clamped_when_swing_is_near_the_start():
    frames = _with_swing_at(_still_frames(100), peak_idx=2)
    start, end = find_delivery_window(frames, fps=FPS)
    assert start == 0  # can't extend before frame 0
    assert end <= len(frames)


def test_window_is_clamped_when_swing_is_near_the_end():
    frames = _with_swing_at(_still_frames(100), peak_idx=98)
    start, end = find_delivery_window(frames, fps=FPS)
    assert end == len(frames)  # can't extend past the last frame
    assert start >= 0


def test_raises_when_no_clear_swing_exists():
    frames = _still_frames(150)  # no motion anywhere in the clip
    with pytest.raises(ValueError, match="No clear delivery detected"):
        find_delivery_window(frames, fps=FPS)


def test_requires_at_least_two_frames():
    with pytest.raises(ValueError):
        find_delivery_window(_still_frames(1), fps=FPS)


def test_finds_multiple_well_separated_deliveries_in_order():
    # A nets-session clip with three balls, each 4s apart (well past
    # MIN_DELIVERY_SEPARATION_SEC) — the realistic shape find_delivery_window
    # (singular) would silently under-report by only returning one of these.
    frames = _with_swings_at(_still_frames(600), peak_indices=[100, 300, 500])
    windows = find_delivery_windows(frames, fps=FPS)
    assert len(windows) == 3
    # returned in clip order, and each window brackets its own peak
    assert windows[0][0] < 100 < windows[0][1]
    assert windows[1][0] < 300 < windows[1][1]
    assert windows[2][0] < 500 < windows[2][1]
    # no two windows overlap
    assert windows[0][1] <= windows[1][0]
    assert windows[1][1] <= windows[2][0]


def test_two_close_swings_are_treated_as_one_delivery():
    # Peaks 10 frames apart (0.33s at 30fps) are well inside
    # MIN_DELIVERY_SEPARATION_SEC — this is one delivery's natural
    # rise-and-fall producing two above-threshold frames, not two balls.
    frames = _with_swings_at(_still_frames(200), peak_indices=[100, 110])
    windows = find_delivery_windows(frames, fps=FPS)
    assert len(windows) == 1


def test_max_deliveries_caps_the_count():
    frames = _with_swings_at(_still_frames(600), peak_indices=[100, 300, 500])
    windows = find_delivery_windows(frames, fps=FPS, max_deliveries=2)
    assert len(windows) == 2


def test_find_delivery_window_singular_matches_the_strongest_peak_from_plural():
    # The single-delivery function should agree with the plural one
    # capped at 1 — same underlying peak-picking, just one call site.
    frames = _with_swings_at(_still_frames(400), peak_indices=[100, 300])
    single_start, single_end = find_delivery_window(frames, fps=FPS)
    plural_windows = find_delivery_windows(frames, fps=FPS, max_deliveries=1)
    assert (single_start, single_end) == plural_windows[0]


def test_plural_raises_when_no_clear_swing_exists():
    frames = _still_frames(150)
    with pytest.raises(ValueError, match="No clear delivery detected"):
        find_delivery_windows(frames, fps=FPS)

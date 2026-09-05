import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from delivery_segmentation import find_delivery_window
from feature_extraction import LEFT_WRIST

N_LANDMARKS = 33
FPS = 30.0


def _blank_frame(x=0.5, y=0.5):
    return [(x, y, 1.0) for _ in range(N_LANDMARKS)]


def _still_frames(n):
    return [_blank_frame() for _ in range(n)]


def _with_swing_at(frames, peak_idx, step=0.1):
    """Injects a sharp wrist movement centred on peak_idx into an
    otherwise-still sequence, simulating one clear swing inside a longer,
    mostly-static clip (a batter standing between deliveries)."""
    frames = [list(f) for f in frames]
    for offset, mult in ((-1, 0.3), (0, 1.0), (1, 0.3)):
        idx = peak_idx + offset
        if 0 <= idx < len(frames):
            frames[idx][LEFT_WRIST] = (0.5 + step * mult, 0.5, 1.0)
    return frames


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

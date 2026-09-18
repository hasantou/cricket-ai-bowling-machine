import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from live_video_source import LiveVideoSource


def test_injected_frame_factory_yields_frames_in_order():
    frames = [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(5)]
    source = LiveVideoSource(frame_factory=lambda: iter(frames), fps_hint=60.0)
    assert source.fps == 60.0
    collected = list(source.frames())
    assert len(collected) == 5
    for i, frame in enumerate(collected):
        assert (frame == i).all()


def test_frames_can_be_iterated_more_than_once_with_a_factory():
    # A real generator would be exhausted after one pass - frame_factory
    # is a callable that produces a FRESH iterable each time, so this
    # matters for anything that might call .frames() more than once.
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return iter([np.zeros((2, 2, 3), dtype=np.uint8)])

    source = LiveVideoSource(frame_factory=factory)
    list(source.frames())
    list(source.frames())
    assert calls["n"] == 2


def test_context_manager_closes_without_error_when_no_real_capture_opened():
    with LiveVideoSource(frame_factory=lambda: iter([])) as source:
        assert list(source.frames()) == []
    # __exit__ must not raise even though no cv2.VideoCapture was ever opened.


def test_invalid_device_raises_a_clear_error():
    with pytest.raises(RuntimeError):
        LiveVideoSource(device="this/path/does/not/exist.mp4")

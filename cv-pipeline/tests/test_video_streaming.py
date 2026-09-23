import os
import sys
import tempfile
import tracemalloc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
from video_pipeline import analyse_video

W, H, FPS = 640, 480, 30.0


def scene(seed):
    """A high-contrast blocky scene, so two different seeds are visibly different shots."""
    rng = np.random.default_rng(seed)
    blocks = (rng.integers(0, 2, (H // 16, W // 16, 3)) * 200 + 25).astype(np.uint8)
    return cv2.resize(blocks, (W, H), interpolation=cv2.INTER_NEAREST)


def write_clip(path, n=150, cut_at=None, doubled=False):
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    a, b = scene(1), scene(2)
    for i in range(n):
        base = b if (cut_at is not None and i >= cut_at) else a
        step = i // 2 if doubled else i
        frame = base.copy()
        cv2.circle(frame, (60 + (step * 40) % 500, 240), 60, (255, 255, 255), -1)    # a big object moving a lot each new frame
        writer.write(frame)
    writer.release()


def person(wrist_xy):
    lm = [(0.5, 0.5, 1.0)] * 33
    lm = list(lm)
    lm[0] = (0.5, 0.30, 1.0)
    lm[11], lm[12] = (0.45, 0.40, 1.0), (0.55, 0.40, 1.0)
    lm[23], lm[24] = (0.47, 0.60, 1.0), (0.53, 0.60, 1.0)
    lm[27], lm[28] = (0.46, 0.85, 1.0), (0.54, 0.85, 1.0)
    lm[15] = (wrist_xy[0], wrist_xy[1], 1.0)
    lm[16] = (0.56, 0.62, 1.0)
    return lm


class FakePose:
    """Stands in for the pose model: consumes the frame stream (as the real one does) and
    returns one landmark set per frame, with a fast left-wrist swing at each given frame."""

    def __init__(self, swings=(), teleport_at=None):
        self.swings, self.teleport_at = set(swings), teleport_at
        self.frames_seen = 0

    def extract_aligned_landmarks_from_frames(self, frames, fps):
        out = []
        for i, _ in enumerate(frames):
            self.frames_seen += 1
            x = 0.30
            for s in self.swings:
                if s - 3 <= i <= s + 3:
                    x = 0.30 + 0.09 * (i - (s - 3))                 # a fast burst of hand movement
            if self.teleport_at is not None and i >= self.teleport_at:
                x = 0.8                                             # the person 'jumps' at an edit cut
            out.append(person((x, 0.55)))
        return out

    def close(self):
        pass


def test_a_clip_is_processed_without_holding_its_frames_in_memory():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "c.mp4")
        write_clip(path, n=150)
        all_frames_bytes = 150 * W * H * 3                           # ~138 MB if every frame were kept
        tracemalloc.start()
        a = analyse_video(path, estimator=FakePose(swings=[60]))
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert peak < all_frames_bytes / 4
        assert len(a.estimates) >= 1 and a.container_fps == FPS


def test_an_edit_cut_is_reported_and_a_fake_swing_on_the_cut_is_dropped():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cut.mp4")
        write_clip(path, n=150, cut_at=100)
        # a real swing at frame 40, and a 'person teleports' jump exactly at the cut (frame 100)
        a = analyse_video(path, estimator=FakePose(swings=[40], teleport_at=100))
        assert 100 in a.cut_events or any(abs(c - 100) <= 2 for c in a.cut_events)
        assert any("edit cut" in n for n in a.clip_notes)
        assert len(a.estimates) == 1                                  # only the real swing survives


def test_a_doubled_frame_rate_is_reported_as_the_real_frame_rate():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "dup.mp4")
        write_clip(path, n=120, doubled=True)
        a = analyse_video(path, estimator=FakePose(swings=[60]))
        assert a.container_fps == FPS and a.effective_fps < 0.7 * FPS
        assert any("genuinely different" in n for n in a.clip_notes)


def test_a_plain_clip_has_no_cut_or_frame_rate_notes_and_a_still_camera_is_left_alone():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "plain.mp4")
        write_clip(path, n=120)
        a = analyse_video(path, estimator=FakePose(swings=[60]))
        assert a.cut_events == [] and a.clip_notes == [] and not a.camera_moved
        assert a.delivery_notes == [[] for _ in a.estimates]


def test_swing_frames_are_absolute_clip_indices_not_window_relative():
    """swing_frames must be usable directly as `frame / fps` real-clip time -- a consumer aligning
    this against another clock (e.g. commentary_labels.py) cannot re-derive the offset itself."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "c.mp4")
        write_clip(path, n=150)
        a = analyse_video(path, estimator=FakePose(swings=[60]))
        assert len(a.swing_frames) == len(a.estimates) == 1
        assert abs(a.swing_frames[0] - 60) <= 5     # within the detector's own smoothing/peak-picking slack

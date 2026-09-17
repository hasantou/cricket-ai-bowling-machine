import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import pytest
from motion_ball_detector import detect_ball_candidates, to_ball_detections

WIDTH, HEIGHT = 240, 160
N_FRAMES = 30


def _write_synthetic_video(path, draw_frame):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 30.0, (WIDTH, HEIGHT))
    for i in range(N_FRAMES):
        frame = np.full((HEIGHT, WIDTH, 3), 90, dtype=np.uint8)  # flat grey background
        draw_frame(frame, i)
        writer.write(frame)
    writer.release()


def test_detects_a_small_moving_circle_close_to_its_true_path():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ball.mp4")

        def draw(frame, i):
            cv2.circle(frame, (20 + i * 6, 80), radius=4, color=(230, 230, 230), thickness=-1)

        _write_synthetic_video(path, draw)
        candidates = detect_ball_candidates(path, min_radius_px=1.5, max_radius_px=10.0)

    detected_frames = {c.frame_index for c in candidates}
    # MOG2 needs a few frames to build its background model before it can
    # flag anything as foreground - later frames should be reliably caught.
    assert len(detected_frames & set(range(10, N_FRAMES))) >= 10

    by_frame = {c.frame_index: c for c in candidates}
    for i in range(15, N_FRAMES):
        if i in by_frame:
            expected_x = 20 + i * 6
            assert abs(by_frame[i].x_px - expected_x) < 5.0
            assert abs(by_frame[i].y_px - 80) < 5.0


def test_rejects_a_large_moving_rectangle_as_too_big_for_a_ball():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "person.mp4")

        def draw(frame, i):
            cv2.rectangle(frame, (30 + i * 4, 20), (70 + i * 4, 140), color=(230, 230, 230), thickness=-1)

        _write_synthetic_video(path, draw)
        candidates = detect_ball_candidates(path, min_radius_px=1.5, max_radius_px=10.0)

    assert candidates == []


def test_rejects_a_thin_elongated_blob_on_circularity_even_if_small_enough():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sliver.mp4")

        def draw(frame, i):
            # A thin line ~1px thick can fit inside a small enclosing
            # circle by radius alone, but has very low circularity -
            # this is the shape check catching what the size check can't.
            cv2.line(frame, (10 + i * 5, 60), (10 + i * 5 + 18, 60), color=(230, 230, 230), thickness=1)

        _write_synthetic_video(path, draw)
        candidates = detect_ball_candidates(
            path, min_radius_px=1.0, max_radius_px=15.0, min_circularity=0.55,
        )

    assert candidates == []


def test_to_ball_detections_keeps_only_the_most_circular_candidate_per_frame():
    from motion_ball_detector import MotionCandidate

    candidates = [
        MotionCandidate(frame_index=5, x_px=10, y_px=10, radius_px=3, circularity=0.6),
        MotionCandidate(frame_index=5, x_px=50, y_px=50, radius_px=4, circularity=0.9),
        MotionCandidate(frame_index=6, x_px=20, y_px=20, radius_px=3, circularity=0.8),
    ]
    detections = to_ball_detections(candidates)
    assert len(detections) == 2
    frame_5 = next(d for d in detections if d.frame_index == 5)
    assert frame_5.x_px == 50 and frame_5.confidence == 0.9


def test_missing_file_raises_a_clear_error_not_a_silent_empty_result():
    with pytest.raises(FileNotFoundError):
        detect_ball_candidates("this/path/does/not/exist.mp4")

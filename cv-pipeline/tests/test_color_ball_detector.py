import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
from color_ball_detector import detect_ball_candidates_by_color

W, H = 200, 150


def write_clip(path, ball_positions, radius=6, bg_color=(80, 140, 60)):
    """A plain green-ish background with a small red-ish circle at each frame's position — the
    controlled case this module should solve cleanly, distinct from the honest real-footage
    negative result found testing against an actual broadcast clip (see the module's docstring)."""
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    for pos in ball_positions:
        frame = np.full((H, W, 3), bg_color, np.uint8)
        if pos is not None:
            cv2.circle(frame, pos, radius, (40, 30, 150), -1)   # BGR: a dark, saturated red
        writer.write(frame)
    writer.release()


def test_a_clearly_red_ball_on_a_plain_background_is_found_close_to_its_true_position():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "clip.mp4")
        write_clip(path, [(100, 75)] * 5)
        cands = detect_ball_candidates_by_color(path)
        assert len(cands) >= 5
        for c in cands:
            assert abs(c.x_px - 100) < 4 and abs(c.y_px - 75) < 4


def test_the_ball_is_tracked_as_it_moves_frame_to_frame():
    positions = [(30 + 10 * i, 75) for i in range(10)]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "clip.mp4")
        write_clip(path, positions)
        cands = detect_ball_candidates_by_color(path)
        by_frame = {c.frame_index: c for c in cands}
        assert len(by_frame) == 10
        for i, (px, py) in enumerate(positions):
            assert abs(by_frame[i].x_px - px) < 4 and abs(by_frame[i].y_px - py) < 4


def test_a_frame_with_no_ball_produces_no_candidate():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "clip.mp4")
        write_clip(path, [None, None, (100, 75), None])
        cands = detect_ball_candidates_by_color(path)
        assert {c.frame_index for c in cands} == {2}


def test_a_green_only_background_produces_no_false_candidates():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "clip.mp4")
        write_clip(path, [None] * 5)
        assert detect_ball_candidates_by_color(path) == []


def test_something_red_but_too_big_to_be_a_ball_is_rejected():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "clip.mp4")
        write_clip(path, [(100, 75)], radius=60)     # far larger than any plausible ball radius
        assert detect_ball_candidates_by_color(path) == []


def test_roi_restricts_the_search_but_coordinates_stay_in_the_original_frame():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "clip.mp4")
        write_clip(path, [(100, 75)])
        cands = detect_ball_candidates_by_color(path, roi=(50, 25, 150, 125))
        assert len(cands) == 1
        assert abs(cands[0].x_px - 100) < 4 and abs(cands[0].y_px - 75) < 4


def test_a_ball_outside_the_roi_is_not_found():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "clip.mp4")
        write_clip(path, [(10, 10)])
        assert detect_ball_candidates_by_color(path, roi=(50, 25, 150, 125)) == []


def test_missing_file_raises_a_clear_error():
    import pytest
    with pytest.raises(FileNotFoundError):
        detect_ball_candidates_by_color("does_not_exist.mp4")

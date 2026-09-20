import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from body_vector_render import GOOD, arrow_tip, draw_body_vectors, to_jpeg
from body_vectors import analyse_body_vectors
from test_body_vectors import FPS, moving_wrist, skeleton

W, H = 640, 480


def _report_and_frame(dx, dy, **kw):
    frames = moving_wrist(dx, dy, **kw)
    r = analyse_body_vectors(frames, FPS, aspect=W / H)
    return r, frames[r.peak_frame]


def test_drawing_changes_the_image_and_leaves_the_input_untouched():
    r, lm = _report_and_frame(0.0105 / (W / H), 0.0)
    blank = np.zeros((H, W, 3), np.uint8)
    out = draw_body_vectors(blank, lm, r)
    assert blank.sum() == 0 and out.sum() > 0


def test_arrows_point_the_way_the_joint_moves_on_screen():
    """Right is +x, up on screen is -y in pixels, and length scales with speed."""
    assert arrow_tip((100, 100), 2.0, 0.0, 100.0) == (130, 100)       # right
    assert arrow_tip((100, 100), 0.0, 2.0, 100.0) == (100, 70)        # up
    assert arrow_tip((100, 100), -2.0, -2.0, 100.0) == (70, 130)      # down-left
    far = arrow_tip((0, 0), 4.0, 0.0, 100.0)[0]
    near = arrow_tip((0, 0), 2.0, 0.0, 100.0)[0]
    assert far == 2 * near


def test_a_joint_the_model_could_not_see_gets_no_confident_arrow():
    frames = [skeleton(right_wrist=(0.5 + 0.002 * i, 0.55, 0.2), right_vis=0.2) for i in range(30)]
    r = analyse_body_vectors(frames, FPS, aspect=W / H)
    assert r is not None
    v = r.vectors_at_peak.get("right wrist")
    assert v is None or v.confidence == "poor"
    lm = frames[r.peak_frame]
    out = draw_body_vectors(np.zeros((H, W, 3), np.uint8), lm, r)
    wx, wy = int(lm[16][0] * W), int(lm[16][1] * H)
    near_wrist = out[wy - 5: wy + 6, wx - 5: wx + 6]
    assert not np.all(near_wrist == GOOD, axis=2).any()               # its dot is grey, never 'good' green


def test_jpeg_encoding_roundtrips_to_bytes():
    data = to_jpeg(np.full((20, 20, 3), 100, np.uint8))
    assert data[:2] == b"\xff\xd8"

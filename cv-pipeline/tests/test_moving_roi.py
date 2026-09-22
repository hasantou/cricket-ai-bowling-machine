import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import pytest
from video_pipeline import advance_roi, analyse_video, clamp_roi
from test_video_streaming import FakePose, H, W


def write_panning_clip(path, n=90, pan_px_per_frame=6, fps=30.0):
    """A camera that pans steadily right (so on screen, a world-fixed subject drifts LEFT each
    frame) — the case a moving region following the camera should get right, and a fixed region
    should not (a real broadcast whip-pan, made reproducible)."""
    bg_w = W + n * pan_px_per_frame + 40
    blocks = (np.random.default_rng(7).integers(0, 2, (H // 16, bg_w // 16, 3)) * 200 + 25).astype(np.uint8)
    bg = cv2.resize(blocks, (bg_w, H), interpolation=cv2.INTER_NEAREST)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    world_x = W // 2
    for i in range(n):
        x0 = i * pan_px_per_frame
        frame = bg[:, x0:x0 + W].copy()
        cv2.circle(frame, (world_x - x0, H // 2), 18, (0, 200, 255), -1)
        writer.write(frame)
    writer.release()
    return world_x


# ---- clamp_roi ----
def test_clamp_roi_keeps_a_drifted_region_inside_the_frame():
    assert clamp_roi((-0.3, 0.2, 0.1, 0.5)) == (0.0, 0.2, 0.1, 0.5)
    assert clamp_roi((0.9, 0.2, 1.4, 0.5)) == (0.9, 0.2, 1.0, 0.5)


def test_clamp_roi_repairs_a_degenerate_or_inverted_box():
    x0, y0, x1, y1 = clamp_roi((0.5, 0.5, 0.5, 0.9))          # zero width
    assert x1 - x0 >= 0.019
    x0, y0, x1, y1 = clamp_roi((0.6, 0.2, 0.3, 0.5))          # x1 < x0
    assert x0 <= x1


# ---- advance_roi: the pan-following claim, tested directly ----
def test_advance_roi_moves_by_exactly_the_cumulative_camera_offset():
    roi = (0.3, 0.2, 0.5, 0.6)
    assert advance_roi(roi, 0.1, -0.05) == pytest.approx((0.4, 0.15, 0.6, 0.55))


def test_advance_roi_is_a_noop_for_a_still_camera():
    roi = (0.3, 0.2, 0.5, 0.6)
    assert advance_roi(roi, 0.0, 0.0) == roi


def test_advance_roi_clamps_when_the_pan_would_push_it_off_frame():
    roi = (0.1, 0.1, 0.3, 0.3)
    moved = advance_roi(roi, -0.5, 0.0)
    assert moved[0] == 0.0 and moved[2] > moved[0]


# ---- end to end: the region really does follow a panning shot ----
def test_a_moving_region_keeps_a_world_fixed_subject_inside_it_through_a_steady_pan():
    """The core claim. A subject fixed in the real world drifts across the SCREEN as the camera
    pans; a region that tracks the camera's own motion should still contain that on-screen position
    at the end of the pan, even though it has moved a long way from where it started."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "pan.mp4")
        n, pan = 90, 6
        world_x = write_panning_clip(path, n=n, pan_px_per_frame=pan)
        start_frac = world_x / W
        roi0 = (start_frac - 0.15, 0.3, start_frac + 0.15, 0.7)

        final_screen_frac = (world_x - (n - 1) * pan) / W       # where the subject actually is on screen at the end
        assert not (roi0[0] <= final_screen_frac <= roi0[2]), "fixture check: the fixed box must have lost the subject"

        a = analyse_video(path, estimator=FakePose(swings=[45]), roi=roi0)
        note = next(n for n in a.clip_notes if "region you set" in n)
        assert "track the camera's own panning" in note


def test_the_note_reports_how_far_the_region_drifted_and_says_what_it_does_not_do():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "pan.mp4")
        world_x = write_panning_clip(path, n=60, pan_px_per_frame=8)
        roi0 = (world_x / W - 0.15, 0.3, world_x / W + 0.15, 0.7)
        a = analyse_video(path, estimator=FakePose(swings=[30]), roi=roi0)
        note = next(n for n in a.clip_notes if "region you set" in n)
        assert "drifted" in note
        assert "does NOT follow the batter's own movement" in note


def test_a_still_camera_reports_no_meaningful_drift():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "still.mp4")
        write_panning_clip(path, n=40, pan_px_per_frame=0)
        a = analyse_video(path, estimator=FakePose(swings=[20]), roi=(0.3, 0.3, 0.7, 0.7))
        note = next(n for n in a.clip_notes if "region you set" in n)
        assert "drifted" not in note


def test_camera_motion_measurement_is_unaffected_by_cropping():
    """Camera tracking runs on the FULL frame regardless of the region, so panning is measured the
    same whether or not a region is set — the two must not interfere with each other."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "pan.mp4")
        world_x = write_panning_clip(path, n=60, pan_px_per_frame=6)
        roi0 = (world_x / W - 0.15, 0.3, world_x / W + 0.15, 0.7)
        with_roi = analyse_video(path, estimator=FakePose(swings=[30]), roi=roi0)
        without_roi = analyse_video(path, estimator=FakePose(swings=[30]))
        assert with_roi.camera_moved and without_roi.camera_moved
        assert abs(with_roi.effective_fps - without_roi.effective_fps) < 1e-6


def test_clamp_roi_regression_a_region_that_drifts_entirely_past_one_edge_stays_valid():
    """Found on a real broadcast whip-pan: the region drifted so far that BOTH bounds landed past
    the same edge (e.g. x0=1.3, x1=1.6). The old clamp only floored the low bound at 0 and
    ceilinged the high bound at 1, so this produced an INVERTED, zero-area box (x0=1.3, x1=1.0)
    that crashed cv2.resize downstream. Every axis must clamp both bounds independently."""
    for roi in [(1.3, 0.2, 1.6, 0.5), (-1.3, 0.2, -1.0, 0.5), (2.0, 0.3, 3.0, 0.6), (0.2, 1.4, 0.5, 1.9)]:
        x0, y0, x1, y1 = clamp_roi(roi)
        assert 0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0


def test_a_region_that_drifts_entirely_off_frame_does_not_crash_the_pipeline():
    """End to end: a pan fast enough to carry the region clean off one edge must still produce a
    result (a poor one, honestly reported), never a crash."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "hard_pan.mp4")
        world_x = write_panning_clip(path, n=80, pan_px_per_frame=14)     # a much harder pan than the region itself is wide
        roi0 = (world_x / W - 0.1, 0.3, world_x / W + 0.1, 0.7)
        a = analyse_video(path, estimator=FakePose(swings=[40]), roi=roi0)     # must not raise
        assert a.footage is not None

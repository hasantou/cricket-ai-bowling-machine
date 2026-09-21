import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import pytest
from video_pipeline import ROI_TARGET_HEIGHT, MAX_POSE_SIDE, analyse_video, crop_to_roi, map_roi_landmarks
from test_video_streaming import FakePose, W, H, write_clip


def test_a_small_region_is_upscaled_so_the_model_gets_more_pixels():
    frame = np.zeros((1080, 1920, 3), np.uint8)
    crop = crop_to_roi(frame, (0.4, 0.1, 0.6, 0.3))              # 384 x 216 px region
    assert crop.shape[0] == ROI_TARGET_HEIGHT and crop.shape[1] == round(384 * ROI_TARGET_HEIGHT / 216)


def test_a_region_already_large_enough_is_not_resized_and_a_huge_one_is_capped():
    frame = np.zeros((2160, 3840, 3), np.uint8)
    big = crop_to_roi(frame, (0.0, 0.0, 1.0, 1.0))
    assert max(big.shape[:2]) <= MAX_POSE_SIDE
    mid = crop_to_roi(np.zeros((1000, 1000, 3), np.uint8), (0.0, 0.0, 1.0, 0.8))     # 800 px tall: already > target
    assert mid.shape[0] == 800


def test_the_crop_contains_the_region_it_was_asked_for():
    frame = np.zeros((400, 800, 3), np.uint8)
    frame[100:200, 400:600] = 255                                # a bright block
    crop = crop_to_roi(frame, (0.5, 0.25, 0.75, 0.5))            # exactly that block
    assert crop.mean() > 250


def test_landmarks_in_the_crop_map_back_to_the_whole_frame():
    roi = (0.5, 0.2, 0.9, 0.6)
    lm = [(0.0, 0.0, 1.0), (1.0, 1.0, 0.5), (0.5, 0.5, 0.9)]
    out = map_roi_landmarks(lm, roi)
    assert out[0] == pytest.approx((0.5, 0.2, 1.0)) and out[1] == pytest.approx((0.9, 0.6, 0.5))
    assert out[2] == pytest.approx((0.7, 0.4, 0.9))              # the centre of the crop is the centre of the region
    assert map_roi_landmarks(None, roi) is None


def test_the_pipeline_maps_a_region_analysis_back_to_whole_frame_coordinates_and_says_it_used_one():
    roi = (0.4, 0.1, 0.9, 0.7)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "c.mp4")
        write_clip(path, n=120)
        seen = {}

        class Spy(FakePose):
            def extract_aligned_landmarks_from_frames(self, frames, fps):
                shapes = []
                frames = list(frames)
                seen["heights"] = {f.shape[0] for f in frames}
                return super().extract_aligned_landmarks_from_frames(iter(frames), fps)

        a = analyse_video(path, estimator=Spy(swings=[60]), roi=roi)
        assert seen["heights"] == {ROI_TARGET_HEIGHT}                     # the model only ever saw the upscaled region
        assert any("region you set" in n for n in a.clip_notes)
        # the fake model reports the person at x=0.5 of the CROP; in the frame that is 0.4 + 0.5*0.5 = 0.65 of the width.
        first = next(w for w in a.body_windows if w)
        assert first[0][0][0] == pytest.approx(0.4 + 0.5 * 0.5, abs=0.02)


def test_without_a_region_nothing_changes():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "c.mp4")
        write_clip(path, n=120)
        a = analyse_video(path, estimator=FakePose(swings=[60]))
        assert not any("region you set" in n for n in a.clip_notes)

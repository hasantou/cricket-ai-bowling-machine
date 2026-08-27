import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pose"))

import pytest
from features import (
    Point, compute_frame_features, has_required_landmarks,
    stance_width_ratio, knee_flex_deg, hand_height_ratio,
)


def _standing_pose(ankle_half_width=0.1, wrist_y=0.25):
    """A roughly upright, straight-legged pose facing the camera. Image
    coordinates: y grows downward, so smaller y = higher up the frame."""
    return {
        "left_shoulder": Point(-0.15, 0.30),
        "right_shoulder": Point(0.15, 0.30),
        "left_hip": Point(-0.10, 0.60),
        "right_hip": Point(0.10, 0.60),
        "left_knee": Point(-0.10, 0.80),
        "right_knee": Point(0.10, 0.80),
        "left_ankle": Point(-ankle_half_width, 1.00),
        "right_ankle": Point(ankle_half_width, 1.00),
        "left_wrist": Point(-0.15, wrist_y),
        "right_wrist": Point(0.15, wrist_y),
    }


def test_stance_width_ratio_wider_stance_gives_bigger_ratio():
    narrow = stance_width_ratio(_standing_pose(ankle_half_width=0.05))
    wide = stance_width_ratio(_standing_pose(ankle_half_width=0.30))
    assert wide > narrow


def test_stance_width_ratio_raises_on_zero_shoulder_width():
    pose = _standing_pose()
    pose["left_shoulder"] = pose["right_shoulder"]
    with pytest.raises(ValueError):
        stance_width_ratio(pose)


def test_knee_flex_straight_leg_is_near_180():
    pose = _standing_pose()
    assert knee_flex_deg(pose, "left") == pytest.approx(180.0, abs=0.1)
    assert knee_flex_deg(pose, "right") == pytest.approx(180.0, abs=0.1)


def test_knee_flex_bent_leg_is_near_90():
    pose = _standing_pose()
    # Bend the left knee: ankle moves out to the side instead of straight
    # down, at the same height as the knee.
    pose["left_ankle"] = Point(-0.30, 0.80)
    assert knee_flex_deg(pose, "left") == pytest.approx(90.0, abs=0.1)


def test_knee_flex_rejects_invalid_side():
    with pytest.raises(ValueError):
        knee_flex_deg(_standing_pose(), "middle")


def test_hand_height_ratio_positive_when_hands_above_shoulders():
    pose = _standing_pose(wrist_y=0.10)  # above shoulders (y=0.30)
    assert hand_height_ratio(pose) > 0


def test_hand_height_ratio_negative_when_hands_below_shoulders():
    pose = _standing_pose(wrist_y=0.50)  # below shoulders, above hips
    assert hand_height_ratio(pose) < 0


def test_has_required_landmarks_false_when_missing_one():
    pose = _standing_pose()
    del pose["right_wrist"]
    assert has_required_landmarks(pose) is False


def test_compute_frame_features_returns_none_when_no_person_detected():
    assert compute_frame_features(None) is None


def test_compute_frame_features_returns_none_on_incomplete_landmarks():
    pose = _standing_pose()
    del pose["left_knee"]
    assert compute_frame_features(pose) is None


def test_compute_frame_features_returns_all_four_metrics_for_full_pose():
    features = compute_frame_features(_standing_pose())
    assert set(features.keys()) == {
        "stance_width_ratio", "left_knee_flex_deg",
        "right_knee_flex_deg", "hand_height_ratio",
    }

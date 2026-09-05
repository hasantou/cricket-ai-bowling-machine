import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from feature_extraction import (
    LEFT_ANKLE, LEFT_FOOT_INDEX, LEFT_HIP, LEFT_KNEE, LEFT_WRIST,
    extract_delivery_features,
)

FPS = 30.0

N_LANDMARKS = 33


def _blank_frame(x=0.5, y=0.5):
    """A frame with every landmark parked at the same point, so tests
    only need to move the handful of landmarks they care about."""
    return [(x, y, 1.0) for _ in range(N_LANDMARKS)]


def _set(frame, idx, x, y):
    frame = list(frame)
    frame[idx] = (x, y, 1.0)
    return frame


def test_stationary_front_foot_gives_near_zero_displacement():
    frames = [_blank_frame() for _ in range(10)]
    features = extract_delivery_features(frames)
    assert features.front_foot_displacement == pytest.approx(0.0)


def test_front_foot_moving_forward_is_measured():
    frames = []
    for i in range(10):
        f = _blank_frame()
        f = _set(f, LEFT_FOOT_INDEX, 0.5 + i * 0.02, 0.5)  # steps forward each frame
        frames.append(f)
    features = extract_delivery_features(frames)
    assert features.front_foot_displacement == pytest.approx(0.18, abs=1e-6)


def test_wrist_swing_is_detected_as_peak_speed():
    frames = [_blank_frame(x=0.5) for _ in range(5)]
    fast_frames = []
    for i in range(5):
        f = _blank_frame()
        f = _set(f, LEFT_WRIST, 0.5 + i * 0.1, 0.5)  # fast wrist movement
        fast_frames.append(f)
    features = extract_delivery_features(frames + fast_frames)
    assert features.swing_peak_speed == pytest.approx(0.1, abs=1e-6)


def test_swing_start_frame_is_where_speed_crosses_threshold():
    frames = [_blank_frame() for _ in range(3)]  # wrist still for 3 frames
    for i in range(3):
        f = _blank_frame()
        f = _set(f, LEFT_WRIST, 0.5 + i * 0.05, 0.5)
        frames.append(f)
    features = extract_delivery_features(frames, swing_speed_threshold=0.02)
    # movement only starts once the wrist actually moves — expect it detected
    # somewhere in the second half of the sequence, not frame 0
    assert features.swing_start_frame > 0


def test_bent_knee_reads_as_smaller_angle_than_straight_knee():
    bent = _blank_frame()
    bent = _set(bent, LEFT_HIP, 0.5, 0.3)
    bent = _set(bent, LEFT_KNEE, 0.55, 0.5)
    bent = _set(bent, LEFT_ANKLE, 0.5, 0.6)  # knee bent forward relative to hip-ankle line

    straight = _blank_frame()
    straight = _set(straight, LEFT_HIP, 0.5, 0.3)
    straight = _set(straight, LEFT_KNEE, 0.5, 0.5)
    straight = _set(straight, LEFT_ANKLE, 0.5, 0.7)  # hip-knee-ankle in a straight line

    bent_features = extract_delivery_features([bent, bent])
    straight_features = extract_delivery_features([straight, straight])
    assert bent_features.front_knee_bend_deg < straight_features.front_knee_bend_deg
    assert straight_features.front_knee_bend_deg == pytest.approx(180.0, abs=1.0)


def test_requires_at_least_one_frame():
    with pytest.raises(ValueError):
        extract_delivery_features([])


def test_footwork_lead_is_positive_when_feet_move_before_the_swing():
    frames = [_blank_frame() for _ in range(5)]  # everything still
    for i in range(5):  # feet start moving here
        f = _blank_frame()
        f = _set(f, LEFT_FOOT_INDEX, 0.5 + i * 0.05, 0.5)
        frames.append(f)
    for i in range(5):  # swing starts later
        f = _blank_frame()
        f = _set(f, LEFT_FOOT_INDEX, 0.75, 0.5)
        f = _set(f, LEFT_WRIST, 0.5 + i * 0.1, 0.5)
        frames.append(f)

    features = extract_delivery_features(frames, fps=FPS)
    assert features.footwork_start_frame < features.swing_start_frame
    assert features.footwork_lead_seconds > 0


def test_footwork_lead_is_negative_when_feet_move_after_the_swing():
    frames = [_blank_frame() for _ in range(5)]
    for i in range(5):  # swing starts first
        f = _blank_frame()
        f = _set(f, LEFT_WRIST, 0.5 + i * 0.1, 0.5)
        frames.append(f)
    for i in range(5):  # feet only start moving after the swing is already underway
        f = _blank_frame()
        f = _set(f, LEFT_WRIST, 0.9, 0.5)
        f = _set(f, LEFT_FOOT_INDEX, 0.5 + i * 0.05, 0.5)
        frames.append(f)

    features = extract_delivery_features(frames, fps=FPS)
    assert features.swing_start_frame < features.footwork_start_frame
    assert features.footwork_lead_seconds < 0


def test_footwork_lead_seconds_scales_with_fps():
    frames = [_blank_frame() for _ in range(3)]
    for i in range(3):
        f = _blank_frame()
        f = _set(f, LEFT_FOOT_INDEX, 0.5 + i * 0.05, 0.5)
        frames.append(f)
    for i in range(3):
        f = _blank_frame()
        f = _set(f, LEFT_FOOT_INDEX, 0.65, 0.5)
        f = _set(f, LEFT_WRIST, 0.5 + i * 0.1, 0.5)
        frames.append(f)

    slow_fps = extract_delivery_features(frames, fps=15.0)
    fast_fps = extract_delivery_features(frames, fps=60.0)
    # Same frame gap, but a lower fps means each frame spans more real
    # time — the lead in seconds should be proportionally larger.
    assert slow_fps.footwork_lead_seconds > fast_fps.footwork_lead_seconds > 0

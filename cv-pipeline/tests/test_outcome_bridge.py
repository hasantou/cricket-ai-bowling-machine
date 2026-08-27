import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from feature_extraction import DeliveryFeatures
from outcome_bridge import VisionOutcomeEstimator, heuristic_bridge


def test_good_footwork_and_early_swing_reads_as_on_time_and_correct():
    features = DeliveryFeatures(
        front_foot_displacement=0.10,
        front_knee_bend_deg=140,
        swing_peak_speed=0.15,
        swing_start_frame=2,
        n_frames=10,
    )
    on_time, footwork_correct = heuristic_bridge(features)
    assert on_time is True
    assert footwork_correct is True


def test_no_foot_movement_reads_as_incorrect_footwork():
    features = DeliveryFeatures(
        front_foot_displacement=0.0,
        front_knee_bend_deg=178,
        swing_peak_speed=0.15,
        swing_start_frame=2,
        n_frames=10,
    )
    _, footwork_correct = heuristic_bridge(features)
    assert footwork_correct is False


def test_late_swing_start_reads_as_not_on_time():
    features = DeliveryFeatures(
        front_foot_displacement=0.10,
        front_knee_bend_deg=140,
        swing_peak_speed=0.15,
        swing_start_frame=9,  # in the last quarter of a 10-frame clip
        n_frames=10,
    )
    on_time, _ = heuristic_bridge(features)
    assert on_time is False


def test_estimator_end_to_end_from_synthetic_landmarks():
    n_landmarks = 33
    frames = []
    for i in range(10):
        frame = [(0.5, 0.5, 1.0) for _ in range(n_landmarks)]
        frame[27] = (0.5, 0.5, 1.0)          # LEFT_ANKLE fixed
        frame[25] = (0.5, 0.4, 1.0)          # LEFT_KNEE bent forward
        frame[23] = (0.55, 0.2, 1.0)         # LEFT_HIP
        frame[31] = (0.5 + i * 0.02, 0.5, 1.0)  # LEFT_FOOT_INDEX steps forward
        frame[15] = (0.5 + i * 0.03, 0.5, 1.0)  # LEFT_WRIST swings early
        frames.append(frame)

    estimate = VisionOutcomeEstimator().estimate(frames)
    assert estimate.features.n_frames == 10
    assert isinstance(estimate.on_time, bool)
    assert isinstance(estimate.footwork_correct, bool)

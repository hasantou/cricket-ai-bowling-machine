import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from feature_extraction import DeliveryFeatures
from outcome_bridge import VisionOutcomeEstimator, heuristic_bridge


def test_good_footwork_and_early_footwork_lead_reads_as_on_time_and_correct():
    features = DeliveryFeatures(
        front_foot_displacement=0.10,
        front_knee_bend_deg=140,
        swing_peak_speed=0.15,
        swing_start_frame=5,
        footwork_start_frame=2,
        footwork_lead_seconds=0.10,  # feet moved well before the swing
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
        footwork_start_frame=9,
        footwork_lead_seconds=-0.23,
        n_frames=10,
    )
    _, footwork_correct = heuristic_bridge(features)
    assert footwork_correct is False


def test_feet_moving_after_the_swing_reads_as_not_on_time():
    # Real finding, not a hypothetical: the previous version of this
    # heuristic compared swing timing against the clip's own length,
    # which delivery_segmentation.py's peak-centred windowing made
    # circular — every real clip tested read on_time=True regardless of
    # the delivery. footwork_lead_seconds compares two independently
    # measured events instead, so a genuinely late reaction can fail.
    features = DeliveryFeatures(
        front_foot_displacement=0.10,
        front_knee_bend_deg=140,
        swing_peak_speed=0.15,
        swing_start_frame=2,
        footwork_start_frame=6,
        footwork_lead_seconds=-0.13,  # feet moved after the swing already started
        n_frames=10,
    )
    on_time, _ = heuristic_bridge(features)
    assert on_time is False


def test_footwork_lead_below_the_minimum_reads_as_not_on_time():
    features = DeliveryFeatures(
        front_foot_displacement=0.10,
        front_knee_bend_deg=140,
        swing_peak_speed=0.15,
        swing_start_frame=5,
        footwork_start_frame=4,
        footwork_lead_seconds=0.01,  # feet moved first, but barely — below MIN_FOOTWORK_LEAD_SECONDS
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

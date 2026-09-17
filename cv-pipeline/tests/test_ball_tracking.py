import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from ball_tracking import (
    BallDetection, BallTrajectoryFitError, fit_ball_trajectory, recover_missing_frames,
)

FPS = 30.0

# A synthetic ground-truth flight, in pixel space: constant horizontal
# velocity, a gravity-style parabola vertically. Not real physics (that's
# trajectory-engine's job) - just a curve to test the fitting math against.
TRUE_X = (12.0, 40.0)          # x(f) = 12*f + 40
TRUE_Y = (-0.35, 9.0, 380.0)   # y(f) = -0.35*f^2 + 9*f + 380
N_FRAMES = 40


def true_position(frame: int):
    x = TRUE_X[0] * frame + TRUE_X[1]
    y = TRUE_Y[0] * frame ** 2 + TRUE_Y[1] * frame + TRUE_Y[2]
    return x, y


def test_dense_clean_detections_recover_the_curve_almost_exactly():
    detections = [BallDetection(f, *true_position(f)) for f in range(N_FRAMES)]
    traj = fit_ball_trajectory(detections, fps=FPS, rng=random.Random(0))
    for f in (0, 10, 25, 39):
        px, py = traj.position_at(f)
        tx, ty = true_position(f)
        assert abs(px - tx) < 0.5
        assert abs(py - ty) < 0.5
    assert traj.outlier_frames == []


def test_sparse_detections_at_real_world_recall_still_recover_the_curve():
    """~12% recall is the measured number from real footage - this is the
    exact scenario this module exists for: most frames have no detection
    at all, plus realistic pixel jitter on the ones that do."""
    rng = random.Random(1)
    detections = [
        BallDetection(f, *true_position(f))
        for f in range(N_FRAMES)
        if rng.random() < 0.12
    ]
    noisy = [
        BallDetection(d.frame_index, d.x_px + rng.uniform(-3, 3), d.y_px + rng.uniform(-3, 3))
        for d in detections
    ]
    assert 3 <= len(noisy) < N_FRAMES * 0.3, "test setup should land near the real 12% recall regime"

    traj = fit_ball_trajectory(noisy, fps=FPS, inlier_threshold_px=10.0, rng=random.Random(2))

    for f in (5, 20, 35):
        px, py = traj.position_at(f)
        tx, ty = true_position(f)
        assert abs(px - tx) < 6.0
        assert abs(py - ty) < 6.0


def test_false_positive_outliers_are_rejected_not_averaged_in():
    rng = random.Random(3)
    real = [BallDetection(f, *true_position(f)) for f in range(N_FRAMES) if rng.random() < 0.3]
    outliers = [
        BallDetection(rng.randint(0, N_FRAMES - 1), rng.uniform(0, 2000), rng.uniform(0, 2000))
        for _ in range(len(real))  # as many false positives as real detections
    ]
    combined = real + outliers

    traj = fit_ball_trajectory(combined, fps=FPS, inlier_threshold_px=10.0, rng=random.Random(4))

    real_frames = {d.frame_index for d in real}
    # Most of the accepted inliers should be genuine detections, not noise.
    inlier_set = set(traj.inlier_frames)
    genuine_inliers = inlier_set & real_frames
    assert len(genuine_inliers) >= len(inlier_set) * 0.8

    for f in (5, 20, 35):
        px, py = traj.position_at(f)
        tx, ty = true_position(f)
        assert abs(px - tx) < 8.0
        assert abs(py - ty) < 8.0


def test_fewer_than_three_detections_refuses_to_fit():
    with pytest.raises(BallTrajectoryFitError):
        fit_ball_trajectory(
            [BallDetection(0, 10, 10), BallDetection(1, 12, 11)], fps=FPS,
        )


def test_pure_noise_with_no_shared_trajectory_refuses_rather_than_guessing():
    rng = random.Random(5)
    noise = [BallDetection(f, rng.uniform(0, 2000), rng.uniform(0, 2000)) for f in range(20)]
    with pytest.raises(BallTrajectoryFitError):
        fit_ball_trajectory(noise, fps=FPS, inlier_threshold_px=5.0, min_inliers=5, rng=random.Random(6))


def test_refuses_a_temporally_clustered_fit_instead_of_extrapolating_wildly():
    """Regression test for a real bug found while calibrating this module:
    3 detections at frames 27/28/32 (a 500-trial synthetic sweep at real
    12% recall) fit a "valid" 3-point trajectory with zero error on those
    3 points, but extrapolating it to frame 0 was off by 822px - the
    quadratic had no information about curvature outside that 5-frame
    window. An earlier version of this check compared the winning
    inliers' span against the *detections' own* observed span, which is
    circular when there are only 3 detections total (they trivially
    "span" 100% of their own range) - min_span_seconds fixes that by
    requiring a fixed amount of real time instead, independent of how
    many detections exist."""
    detections = [
        BallDetection(f, *true_position(f)) for f in (27, 28, 32)
    ]
    with pytest.raises(BallTrajectoryFitError, match="clustered"):
        fit_ball_trajectory(detections, fps=30.0, inlier_threshold_px=10.0, rng=random.Random(0))


def test_recover_missing_frames_fills_every_frame_in_range():
    rng = random.Random(7)
    detections = [BallDetection(f, *true_position(f)) for f in range(N_FRAMES) if rng.random() < 0.2]
    positions = recover_missing_frames(detections, fps=FPS, frame_range=(0, N_FRAMES - 1), rng=random.Random(8))
    assert set(positions.keys()) == set(range(N_FRAMES))
    px, py = positions[15]
    tx, ty = true_position(15)
    assert abs(px - tx) < 10.0
    assert abs(py - ty) < 10.0

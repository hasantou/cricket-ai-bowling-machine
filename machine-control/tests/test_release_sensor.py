import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "trajectory-engine"))

import pytest
from cricket_trajectory import BallProperties
from cricket_trajectory.machine import WheelMachine
from machine_control.release_sensor import SimulatedReleaseSensor, SpeedCalibrator


def test_simulated_sensor_reports_the_true_machines_speed_with_no_noise():
    true_machine = WheelMachine(speed_efficiency=0.75)
    sensor = SimulatedReleaseSensor(true_machine, noise_std_mps=0.0)
    sensor.arm(2000.0, 2000.0)
    measured = sensor.measure_speed_mps()
    expected, _ = true_machine.ball_release_state(2000.0, 2000.0, BallProperties())
    assert abs(measured - expected) < 1e-9


def test_sensor_raises_if_measured_before_arming():
    sensor = SimulatedReleaseSensor(WheelMachine())
    with pytest.raises(RuntimeError):
        sensor.measure_speed_mps()


def test_calibrator_returns_none_before_enough_samples():
    assumed_machine = WheelMachine(speed_efficiency=0.90)
    calibrator = SpeedCalibrator(assumed_machine, min_samples=5)
    calibrator.record(2000.0, 2000.0, measured_speed_mps=20.0)
    assert calibrator.suggested_speed_efficiency() is None
    assert calibrator.apply_calibration() is False
    assert assumed_machine.speed_efficiency == 0.90  # unchanged


def test_calibrator_recovers_the_true_efficiency_from_noisy_measurements():
    """The core claim: if the real machine's true speed_efficiency (0.75)
    differs from what the software assumes (0.90), enough real
    measurements should let SpeedCalibrator recover something close to
    the truth - closing exactly the gap machine.py's own docstring flags
    ('calibrate ... with a radar gun ... then treat it as fixed')."""
    true_machine = WheelMachine(speed_efficiency=0.75)
    assumed_machine = WheelMachine(speed_efficiency=0.90)  # what the software starts out believing
    sensor = SimulatedReleaseSensor(true_machine, noise_std_mps=0.3, rng=random.Random(0))
    calibrator = SpeedCalibrator(assumed_machine, min_samples=10)

    rng = random.Random(1)
    for _ in range(40):
        rpm1 = rng.uniform(1000.0, 4000.0)
        rpm2 = rng.uniform(1000.0, 4000.0)
        sensor.arm(rpm1, rpm2)
        measured = sensor.measure_speed_mps()
        calibrator.record(rpm1, rpm2, measured)

    suggested = calibrator.suggested_speed_efficiency()
    assert suggested is not None
    assert abs(suggested - 0.75) < 0.02  # recovered within 2% of the true value

    applied = calibrator.apply_calibration()
    assert applied is True
    assert abs(assumed_machine.speed_efficiency - 0.75) < 0.02


def test_calibration_actually_improves_future_delivery_accuracy():
    """Not just 'the number looks right' - confirms the corrected machine
    now computes RPMs that produce the ball speed actually requested,
    when driven against the true (different) machine."""
    from cricket_trajectory.ball import BallProperties

    true_machine = WheelMachine(speed_efficiency=0.75)
    assumed_machine = WheelMachine(speed_efficiency=0.90)
    ball = BallProperties()
    sensor = SimulatedReleaseSensor(true_machine, noise_std_mps=0.0)
    calibrator = SpeedCalibrator(assumed_machine, min_samples=10)

    rng = random.Random(2)
    for _ in range(20):
        rpm1 = rng.uniform(1500.0, 3500.0)
        rpm2 = rng.uniform(1500.0, 3500.0)
        sensor.arm(rpm1, rpm2)
        calibrator.record(rpm1, rpm2, sensor.measure_speed_mps())
    calibrator.apply_calibration()

    target_speed = 30.0
    # RPMs computed from the NOW-CORRECTED assumed_machine...
    rpm1, rpm2 = assumed_machine.wheel_rpms_for_delivery(target_speed, 0.0, ball)
    # ...should make the TRUE machine actually deliver close to the target.
    actual_speed_on_real_machine, _ = true_machine.ball_release_state(rpm1, rpm2, ball)
    assert abs(actual_speed_on_real_machine - target_speed) < 1.0

    # Without calibration, the same target computed from the WRONG (0.90)
    # assumption misses badly on the real (0.75) machine - showing the
    # calibration step above is what closed the gap, not coincidence.
    uncalibrated_rpm1, uncalibrated_rpm2 = WheelMachine(speed_efficiency=0.90).wheel_rpms_for_delivery(
        target_speed, 0.0, ball
    )
    actual_speed_uncalibrated, _ = true_machine.ball_release_state(uncalibrated_rpm1, uncalibrated_rpm2, ball)
    assert abs(actual_speed_uncalibrated - target_speed) > 4.0

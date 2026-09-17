import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from cricket_trajectory.ball import BallProperties, Delivery
from machine_control.impact_sensor import (
    ImpactSensorOutcomeObserver, NoContactDetected, SimulatedImpactSensor,
)

BALL = BallProperties()
DELIVERY = Delivery(speed_mps=30.0, seam_angle_deg=10.0, label="test")


def make_observer():
    sensor = SimulatedImpactSensor()
    return ImpactSensorOutcomeObserver(sensor), sensor


def test_raises_if_measured_before_arming():
    sensor = SimulatedImpactSensor()
    with pytest.raises(RuntimeError):
        sensor.measure_impact()


def test_no_contact_reading_raises_the_documented_exception():
    sensor = SimulatedImpactSensor()
    sensor.arm(None)
    with pytest.raises(NoContactDetected):
        sensor.measure_impact()


def test_no_contact_classifies_as_missed_via_the_real_outcome_observer():
    observer, sensor = make_observer()
    sensor.arm(None)
    outcome = observer.observe(DELIVERY, BALL)
    assert outcome == "missed"


def test_dead_bat_distance_classifies_correctly():
    observer, sensor = make_observer()
    sensor.arm((1.0, 0.5))  # short distance, low height
    assert observer.observe(DELIVERY, BALL) == "defended"


def test_controlled_placement_distance_classifies_correctly():
    observer, sensor = make_observer()
    sensor.arm((6.0, 1.0))
    assert observer.observe(DELIVERY, BALL) == "controlled"


def test_well_struck_distance_classifies_correctly():
    observer, sensor = make_observer()
    sensor.arm((15.0, 1.5))
    assert observer.observe(DELIVERY, BALL) == "boundary"


def test_high_ball_classifies_as_mistimed_regardless_of_distance():
    observer, sensor = make_observer()
    sensor.arm((15.0, 5.0))  # would otherwise be well_struck on distance alone
    assert observer.observe(DELIVERY, BALL) == "edged"


def test_arms_only_once_then_requires_rearming():
    observer, sensor = make_observer()
    sensor.arm((1.0, 0.5))
    observer.observe(DELIVERY, BALL)
    with pytest.raises(RuntimeError):
        sensor.measure_impact()

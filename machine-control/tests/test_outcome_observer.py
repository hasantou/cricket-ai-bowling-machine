import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "trajectory-engine"))

import pytest
from cricket_trajectory import BallProperties, Delivery
from machine_control.outcome_observer import ScriptedOutcomeObserver


def test_returns_outcomes_in_order():
    observer = ScriptedOutcomeObserver(["defended", "boundary", "missed"])
    ball = BallProperties()
    delivery = Delivery(label="x")
    assert observer.observe(delivery, ball) == "defended"
    assert observer.observe(delivery, ball) == "boundary"
    assert observer.observe(delivery, ball) == "missed"


def test_raises_once_the_script_runs_out():
    observer = ScriptedOutcomeObserver(["defended"])
    ball = BallProperties()
    delivery = Delivery(label="x")
    observer.observe(delivery, ball)
    with pytest.raises(IndexError):
        observer.observe(delivery, ball)

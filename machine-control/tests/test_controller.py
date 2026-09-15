import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from machine_control.controller import SimulatedMachineController


def test_ready_immediately_before_any_command():
    controller = SimulatedMachineController(cycle_time_s=0.05)
    assert controller.is_ready() is True


def test_not_ready_immediately_after_a_command():
    controller = SimulatedMachineController(cycle_time_s=0.2)
    controller.set_delivery(3000.0, 3000.0)
    assert controller.is_ready() is False


def test_becomes_ready_after_cycle_time_elapses():
    import time
    controller = SimulatedMachineController(cycle_time_s=0.05)
    controller.set_delivery(3000.0, 3000.0)
    time.sleep(0.08)
    assert controller.is_ready() is True


def test_raises_if_commanded_while_not_ready():
    controller = SimulatedMachineController(cycle_time_s=1.0)
    controller.set_delivery(3000.0, 3000.0)
    with pytest.raises(RuntimeError):
        controller.set_delivery(2000.0, 2000.0)


def test_history_records_each_accepted_command():
    controller = SimulatedMachineController(cycle_time_s=0.0)
    controller.set_delivery(1000.0, 1200.0)
    controller.set_delivery(1500.0, 1500.0)
    assert len(controller.history) == 2
    assert controller.history[0].wheel1_rpm == 1000.0
    assert controller.history[1].wheel2_rpm == 1500.0


def test_emergency_stop_makes_it_not_ready_and_is_recorded():
    controller = SimulatedMachineController(cycle_time_s=0.0)
    controller.emergency_stop()
    assert controller.is_ready() is False
    assert controller.history[-1].stopped is True


def test_emergency_stop_does_not_auto_clear_with_time():
    import time
    controller = SimulatedMachineController(cycle_time_s=0.01)
    controller.emergency_stop()
    time.sleep(0.05)
    assert controller.is_ready() is False  # still stopped - time alone must not clear it


def test_reset_after_stop_restores_readiness():
    controller = SimulatedMachineController(cycle_time_s=0.0)
    controller.emergency_stop()
    controller.reset_after_stop()
    assert controller.is_ready() is True

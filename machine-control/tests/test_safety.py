import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from machine_control.controller import SimulatedMachineController
from machine_control.safety import SafeMachineController, SafetyLimits, SafetyViolation


def make_safe(limits=None):
    inner = SimulatedMachineController(cycle_time_s=0.0)
    return SafeMachineController(inner, limits), inner


def test_valid_command_passes_through_to_the_inner_controller():
    safe, inner = make_safe()
    safe.set_delivery(2000.0, 2100.0)
    assert len(inner.history) == 1
    assert inner.history[0].wheel1_rpm == 2000.0


def test_rejects_rpm_above_max():
    safe, inner = make_safe(SafetyLimits(max_wheel_rpm=4000.0))
    with pytest.raises(SafetyViolation):
        safe.set_delivery(4500.0, 3000.0)
    assert len(inner.history) == 0  # never reached the underlying controller


def test_rejects_rpm_below_min():
    safe, inner = make_safe(SafetyLimits(min_wheel_rpm=500.0))
    with pytest.raises(SafetyViolation):
        safe.set_delivery(100.0, 600.0)
    assert len(inner.history) == 0


def test_rejects_excessive_rpm_differential():
    safe, inner = make_safe(SafetyLimits(max_wheel_rpm=6000.0, max_rpm_differential=1000.0))
    with pytest.raises(SafetyViolation):
        safe.set_delivery(5000.0, 2000.0)  # differential 3000 > limit 1000
    assert len(inner.history) == 0


def test_within_all_limits_succeeds():
    safe, inner = make_safe(SafetyLimits(max_wheel_rpm=6000.0, max_rpm_differential=2000.0))
    safe.set_delivery(4000.0, 3000.0)  # differential 1000, within both limits
    assert len(inner.history) == 1


def test_emergency_stop_disables_further_commands():
    safe, inner = make_safe()
    safe.emergency_stop()
    with pytest.raises(SafetyViolation):
        safe.set_delivery(2000.0, 2000.0)


def test_disabled_state_does_not_clear_on_its_own():
    safe, inner = make_safe()
    safe.emergency_stop()
    assert safe.enabled is False
    assert safe.is_ready() is False


def test_reset_re_enables_after_a_stop():
    safe, inner = make_safe()
    safe.emergency_stop()
    safe.reset()
    assert safe.enabled is True
    safe.set_delivery(2000.0, 2000.0)  # should succeed now
    # history[0] is the stop record itself; history[1] is this delivery -
    # confirms reset() actually cleared the INNER controller's stopped
    # state too, not just this layer's own enabled flag.
    assert len(inner.history) == 2
    assert inner.history[-1].wheel1_rpm == 2000.0
    assert inner.history[-1].stopped is False

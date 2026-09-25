"""
Confirms SafetyLimits' default ceiling actually bounds what the adaptive
decision engine can realistically ask a machine to do -- not just that it
"sounds like a big enough number." If trajectory-engine's realistic
envelope (DEFAULT_SPEED_RANGE_KMH, SPIN_RPM_RANGE) ever widens without
someone checking this, that is exactly the kind of drift a real machine
would find out about the hard way; this test finds it at commit time
instead.
"""

import itertools
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "trajectory-engine"))

from cricket_trajectory import BallProperties, DEFAULT_SPEED_RANGE_KMH, SPIN_RPM_RANGE
from cricket_trajectory.machine import WheelMachine
import cricket_trajectory.constants as c
from machine_control.safety import SafetyLimits

BALL = BallProperties()
MACHINE = WheelMachine()


def _worst_case_rpms():
    """Every (speed, spin) combination at the CORNERS of the realistic
    envelope -- the worst case for a monotonic forward model is always at
    an extreme, not in the middle, so a coarse grid over the actual range
    (not just the four corners) is used to avoid assuming that without
    checking it."""
    speeds_kmh = [DEFAULT_SPEED_RANGE_KMH[0] + i * (DEFAULT_SPEED_RANGE_KMH[1] - DEFAULT_SPEED_RANGE_KMH[0]) / 20
                  for i in range(21)]
    spin_rpms = [SPIN_RPM_RANGE[0] + i * (SPIN_RPM_RANGE[1] - SPIN_RPM_RANGE[0]) / 20 for i in range(21)] + [0.0]
    for speed_kmh, spin_rpm in itertools.product(speeds_kmh, spin_rpms):
        speed_mps = c.kmh_to_ms(speed_kmh)
        omega = spin_rpm * 2 * math.pi / 60.0
        rpm1, rpm2 = MACHINE.wheel_rpms_for_delivery(speed_mps, omega, BALL)
        yield rpm1, rpm2


def test_default_safety_limits_comfortably_bound_the_realistic_delivery_envelope():
    limits = SafetyLimits()
    max_single_wheel = 0.0
    max_differential = 0.0
    for rpm1, rpm2 in _worst_case_rpms():
        max_single_wheel = max(max_single_wheel, abs(rpm1), abs(rpm2))
        max_differential = max(max_differential, abs(rpm1 - rpm2))

    # Not just "doesn't exceed" -- real headroom, so calibration error (a real wheel that
    # needs a bit more RPM than this model predicts) doesn't immediately trip a hard limit.
    assert max_single_wheel < limits.max_wheel_rpm
    assert max_single_wheel < 0.9 * limits.max_wheel_rpm, (
        f"worst-case single-wheel RPM ({max_single_wheel:.0f}) is within 10% of the safety "
        f"ceiling ({limits.max_wheel_rpm:.0f}) -- too little headroom for calibration error."
    )
    assert max_differential < limits.max_rpm_differential


def test_commissioning_limits_are_genuinely_tighter_than_the_full_defaults():
    default = SafetyLimits()
    commissioning = SafetyLimits.commissioning()
    assert commissioning.max_wheel_rpm < default.max_wheel_rpm
    assert commissioning.max_rpm_differential < default.max_rpm_differential


def test_commissioning_limits_would_reject_a_full_pace_delivery():
    """The whole point of commissioning() -- confirmed directly, not assumed:
    a real, realistic fast delivery must NOT fit inside it, or it isn't
    actually a reduced-power mode."""
    from machine_control.safety import SafeMachineController, SafetyViolation
    from machine_control.controller import SimulatedMachineController

    speed_mps = c.kmh_to_ms(DEFAULT_SPEED_RANGE_KMH[1])  # fastest realistic delivery
    rpm1, rpm2 = MACHINE.wheel_rpms_for_delivery(speed_mps, 0.0, BALL)

    safe = SafeMachineController(SimulatedMachineController(), limits=SafetyLimits.commissioning())
    try:
        safe.set_delivery(rpm1, rpm2)
        assert False, "a full-pace delivery should have been rejected under commissioning limits"
    except SafetyViolation:
        pass

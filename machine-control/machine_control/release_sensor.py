"""
Closes a gap `trajectory-engine/cricket_trajectory/machine.py` already
names in its own docstring: "calibrate [speed_efficiency] against your
own machine with a radar gun ... then treat it as fixed." That
calibration step never existed as code — this is it.

The insight this exploits: unlike a human bowler, this system commands
the delivery it's about to make (`set_delivery(rpm1, rpm2)`), so it
already knows what speed it *asked* for. It doesn't need to visually
track the ball in flight to check the release was right — it only needs
one number, the actual exit speed, from one cheap sensor at the release
point (a photogate/IR beam-break pair, the same kind of tech radar guns
and gym timing gates already use), which is a far easier sensing problem
than resolving a small fast-moving ball in flight (see
`cv-pipeline/README.md` and `cv-pipeline/motion_ball_detector.py` for how
hard that direct approach turned out to be).

No real sensor exists yet — same honesty pattern as the rest of
machine-control/: SimulatedReleaseSensor stands in for one so
SpeedCalibrator's actual math is real, tested software today.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from cricket_trajectory.machine import WheelMachine


class ReleaseSensor(ABC):
    """The only thing this software knows about a real release-speed
    sensor. A real implementation (photogate, radar gun with a serial/USB
    output) implements this; nothing else changes when the sensor does."""

    @abstractmethod
    def measure_speed_mps(self) -> float:
        """Blocks until the delivery just commanded has passed the
        sensor, then returns its measured exit speed in m/s."""


class SimulatedReleaseSensor(ReleaseSensor):
    """Stands in for real hardware: reports what a machine with some
    (possibly different from the software's assumed) true efficiency
    would actually produce, plus optional measurement noise — so
    SpeedCalibrator can be tested against a known ground truth."""

    def __init__(self, true_machine: WheelMachine, noise_std_mps: float = 0.0,
                 rng=None):
        import random
        self._true_machine = true_machine
        self._noise_std = noise_std_mps
        self._rng = rng or random.Random()
        self._next_rpms: Optional[tuple] = None

    def arm(self, wheel1_rpm: float, wheel2_rpm: float) -> None:
        """Test/demo helper: tells this stand-in what the next
        measure_speed_mps() call should be based on, standing in for the
        sensor physically observing that same commanded delivery."""
        self._next_rpms = (wheel1_rpm, wheel2_rpm)

    def measure_speed_mps(self) -> float:
        if self._next_rpms is None:
            raise RuntimeError("SimulatedReleaseSensor.arm() must be called before measuring.")
        rpm1, rpm2 = self._next_rpms
        self._next_rpms = None
        from cricket_trajectory.ball import BallProperties
        true_speed, _ = self._true_machine.ball_release_state(rpm1, rpm2, BallProperties())
        return true_speed + self._rng.gauss(0.0, self._noise_std)


@dataclass
class ReleaseMeasurement:
    wheel1_rpm: float
    wheel2_rpm: float
    commanded_speed_mps: float
    measured_speed_mps: float


@dataclass
class SpeedCalibrator:
    """Accumulates (commanded, measured) speed pairs across real
    deliveries and fits a corrected `speed_efficiency` for a
    `WheelMachine` — replacing the assumed default with one measured
    against the actual machine, exactly what that class's own docstring
    asks for.

    The fit is a single scale factor through the origin (not a general
    linear regression with an intercept): at zero wheel speed the ball
    speed must physically be zero, so forcing the fit through the origin
    uses that known constraint instead of wasting degrees of freedom
    estimating an intercept that has to be zero anyway.
    """
    machine: WheelMachine
    min_samples: int = 5
    measurements: List[ReleaseMeasurement] = field(default_factory=list)

    def record(self, wheel1_rpm: float, wheel2_rpm: float, measured_speed_mps: float) -> None:
        v1 = self.machine.wheel_rpm_to_surface_speed(wheel1_rpm)
        v2 = self.machine.wheel_rpm_to_surface_speed(wheel2_rpm)
        # Commanded speed under the machine's CURRENT (possibly still
        # wrong) speed_efficiency - what the software believed it asked for.
        commanded_speed = self.machine.speed_efficiency * (v1 + v2) / 2.0
        self.measurements.append(
            ReleaseMeasurement(wheel1_rpm, wheel2_rpm, commanded_speed, measured_speed_mps)
        )

    def suggested_speed_efficiency(self) -> Optional[float]:
        """Least-squares fit of measured_speed = k * avg_surface_speed,
        forced through the origin: k = sum(x*y) / sum(x*x). Returns None
        rather than a guess when there isn't enough data yet — this is an
        expected early-session state, not an error."""
        if len(self.measurements) < self.min_samples:
            return None
        num = 0.0
        den = 0.0
        for m in self.measurements:
            v1 = self.machine.wheel_rpm_to_surface_speed(m.wheel1_rpm)
            v2 = self.machine.wheel_rpm_to_surface_speed(m.wheel2_rpm)
            avg_surface_speed = (v1 + v2) / 2.0
            num += avg_surface_speed * m.measured_speed_mps
            den += avg_surface_speed * avg_surface_speed
        if den <= 0.0:
            return None
        return num / den

    def apply_calibration(self) -> bool:
        """Updates `self.machine.speed_efficiency` in place if enough
        measurements exist. Returns whether it did, so a caller can tell
        'not calibrated yet' apart from 'calibration made no difference'."""
        suggested = self.suggested_speed_efficiency()
        if suggested is None:
            return False
        self.machine.speed_efficiency = suggested
        return True

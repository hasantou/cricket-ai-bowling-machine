"""
Post-contact half of the sensing story — the half the machine genuinely
can't know in advance, since it doesn't control what the batter does.
`net_outcome.py` (trajectory-engine) was already built sensor-agnostic on
purpose: it classifies a shot from two numbers, "distance travelled" and
"max height reached," from wherever those numbers come from. This module
is the first real (if simulated) source for those numbers that isn't a
full trajectory simulation — a simple impact/landing sensor (a couple of
trip-beams at known distances, or a pressure-sensitive strip in the net)
is a far easier sensing problem than resolving a small fast ball
mid-flight against a batter's shot, which is exactly the problem
`motion_ball_detector.py` and `ball_tracking.py` found genuinely hard on
real footage.

No real sensor exists yet; SimulatedImpactSensor stands in for one, same
role SimulatedMachineController and SimulatedReleaseSensor play
elsewhere in this package.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

from cricket_trajectory.ball import BallProperties, Delivery
from cricket_trajectory.net_outcome import NET_OUTCOMES, classify_exit_velocity, classify_net_outcome

from .outcome_observer import OutcomeObserver


class NoContactDetected(Exception):
    """Raised by an ImpactSensor when no bat/ball contact was detected
    for the delivery just bowled — a real, expected outcome (a missed
    ball), not a sensor failure."""


class ImpactSensor(ABC):
    """The only thing this software knows about a real post-contact
    sensor. A real implementation (trip-beams, a pressure-sensitive net
    panel, or eventually vision) implements this; the outcome-observer
    logic below doesn't change when the sensing method does."""

    @abstractmethod
    def measure_impact(self) -> Tuple[float, float]:
        """Returns (distance_m, max_height_m) for the delivery just
        bowled, measured from the point of contact. Raises
        NoContactDetected if the batter didn't hit the ball."""


class SimulatedImpactSensor(ImpactSensor):
    """Stands in for real hardware: reports a pre-armed (distance,
    height) reading, or raises NoContactDetected if armed with `None` —
    so ImpactSensorOutcomeObserver's classification logic is real,
    tested software without needing a real sensor to exist."""

    def __init__(self):
        self._next_reading: Optional[Tuple[float, float]] = "unset"

    def arm(self, reading: Optional[Tuple[float, float]]) -> None:
        """Test/demo helper: `reading` is (distance_m, height_m), or None
        to simulate no contact on the next measure_impact() call."""
        self._next_reading = reading

    def measure_impact(self) -> Tuple[float, float]:
        if self._next_reading == "unset":
            raise RuntimeError("SimulatedImpactSensor.arm() must be called before measuring.")
        reading = self._next_reading
        self._next_reading = "unset"
        if reading is None:
            raise NoContactDetected("No bat/ball contact detected for this delivery.")
        return reading


class ImpactSensorOutcomeObserver(OutcomeObserver):
    """A real OutcomeObserver implementation (the first one that isn't
    ScriptedOutcomeObserver): wraps any ImpactSensor and runs its reading
    through net_outcome.py's existing, already-tested classification
    logic, rather than introducing a second decision path."""

    def __init__(self, sensor: ImpactSensor):
        self._sensor = sensor

    def observe(self, delivery: Delivery, ball: BallProperties) -> str:
        try:
            distance_m, height_m = self._sensor.measure_impact()
        except NoContactDetected:
            return NET_OUTCOMES["no_contact"].adaptive_outcome
        return classify_net_outcome(distance_m, height_m).adaptive_outcome


class VelocitySensor(ABC):
    """A second, independent sensing modality for the same post-contact
    problem: instead of distance/height (ImpactSensor above), this
    measures the ball's exit velocity vector directly — exactly what the
    stereo-camera exit-trajectory design discussed for this project would
    produce (triangulated 3D position over the first 50-200ms after
    contact, fit to a kinematic curve, differentiated for velocity),
    without needing a full post-contact flight simulated or measured."""

    @abstractmethod
    def measure_exit_velocity(self) -> Tuple[float, float, float]:
        """Returns (vx, vy, vz) in m/s, in the same (x=down-pitch,
        y=lateral, z=up) frame as ball.py's Delivery. Raises
        NoContactDetected if the batter didn't hit the ball."""


class SimulatedVelocitySensor(VelocitySensor):
    """Stands in for real hardware (a stereo rig, or any sensor that can
    report an exit velocity vector): reports a pre-armed (vx, vy, vz)
    reading, or raises NoContactDetected if armed with `None`."""

    def __init__(self):
        self._next_reading = "unset"

    def arm(self, reading: Optional[Tuple[float, float, float]]) -> None:
        """Test/demo helper: `reading` is (vx, vy, vz) in m/s, or None to
        simulate no contact on the next measure_exit_velocity() call."""
        self._next_reading = reading

    def measure_exit_velocity(self) -> Tuple[float, float, float]:
        if self._next_reading == "unset":
            raise RuntimeError("SimulatedVelocitySensor.arm() must be called before measuring.")
        reading = self._next_reading
        self._next_reading = "unset"
        if reading is None:
            raise NoContactDetected("No bat/ball contact detected for this delivery.")
        return reading


class VelocitySensorOutcomeObserver(OutcomeObserver):
    """A real OutcomeObserver implementation built on exit velocity
    instead of distance/height — wraps any VelocitySensor and runs its
    reading through net_outcome.classify_exit_velocity(), the same
    already-tested classification logic ImpactSensorOutcomeObserver
    uses, just entered from the other of the two independent paths that
    module documents."""

    def __init__(self, sensor: VelocitySensor):
        self._sensor = sensor

    def observe(self, delivery: Delivery, ball: BallProperties) -> str:
        try:
            vx, vy, vz = self._sensor.measure_exit_velocity()
        except NoContactDetected:
            return NET_OUTCOMES["no_contact"].adaptive_outcome
        outcome, _exit_velocity = classify_exit_velocity(vx, vy, vz)
        return outcome.adaptive_outcome

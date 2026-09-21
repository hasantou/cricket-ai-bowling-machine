"""
The sensor that turns "the model says this would be a wide" into "the ball was
measured passing 0.92 m wide" — and, as a by-product, the data to check and
calibrate the physics model itself.

Why a plane sensor: a wide is judged where the ball PASSES the striker's wicket
(Law 22.2), and the ICC head-height rule is judged at the popping crease. Both
are a question about one plane, not about tracking a small fast ball through
the air (which real footage showed to be hard — see cv-pipeline/README.md).
A light-curtain / laser-scanner / stereo-pair that reports "the ball crossed
this plane at lateral position y and height z at time t" answers it directly.
What it must deliver: lateral position to about ±2 cm and height to about ±3 cm,
against a 3 cm "borderline" band in the rules (laws.WideRules.tolerance_m).

No real sensor exists yet. SimulatedCreaseSensor stands in, in the same role
SimulatedReleaseSensor and SimulatedImpactSensor play; the parsing, the
decision logic and the model-error log are real, tested software.

What the observer does:
  * takes the sensor's reading where there is one, and the physics model's
    prediction where there is not (and says which);
  * re-applies the same wide rule to the MEASURED position;
  * reports whether the sensor and the model agree on the call;
  * records the model's error (measured minus predicted) so a session builds
    the calibration data for the bounce model — the model is otherwise unchecked.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from cricket_trajectory.crease_crossing import Crossing
from cricket_trajectory.laws import (
    POPPING_CREASE_X_M, WICKET_X_M, DeliveryLegality, WideRules, judge_wide, no_ball_flags, stumps_check,
)

WICKET, POPPING = "wicket", "popping"


class NoCrossingDetected(Exception):
    """The sensor saw nothing cross its plane for the delivery just bowled — an expected outcome
    (the ball was hit before the wicket, or died short), not necessarily a sensor failure."""


@dataclass(frozen=True)
class CreaseReading:
    plane: str            # "wicket" (the striker's stumps) or "popping" (the popping crease)
    t_s: float            # on the sensor's clock, seconds
    y_m: float            # lateral, + = off side for a right-hander
    z_m: float            # height above the ground
    source: str = "crease-plane sensor"


class CreaseSensor(ABC):
    """The only thing this software knows about a real crease-plane sensor."""

    @abstractmethod
    def measure_crossing(self, plane: str) -> CreaseReading:
        """The last delivery's crossing of `plane`. Raises NoCrossingDetected if nothing crossed."""


class SimulatedCreaseSensor(CreaseSensor):
    """Stands in for hardware: reports a pre-armed crossing with Gaussian noise, or misses."""

    def __init__(self, noise_lateral_m: float = 0.01, noise_height_m: float = 0.015, miss_rate: float = 0.0, seed: Optional[int] = None):
        self.noise_lateral_m, self.noise_height_m, self.miss_rate = noise_lateral_m, noise_height_m, miss_rate
        self._rng = random.Random(seed)
        self._armed = {}

    def arm(self, plane: str, crossing: Optional[Crossing]) -> None:
        self._armed[plane] = crossing

    def measure_crossing(self, plane: str) -> CreaseReading:
        crossing = self._armed.get(plane)
        if crossing is None or self._rng.random() < self.miss_rate:
            raise NoCrossingDetected(f"nothing crossed the {plane} plane")
        return CreaseReading(
            plane, crossing.t_s, crossing.y_m + self._rng.gauss(0, self.noise_lateral_m),
            max(0.0, crossing.z_m + self._rng.gauss(0, self.noise_height_m)), "simulated crease-plane sensor",
        )


def parse_crease_line(line: str) -> CreaseReading:
    """Parse one line from a sensor board: `CREASE <plane> <t_us> <y_mm> <z_mm>` (signed integers;
    y positive toward the off side). Raises ValueError on anything else — a garbled line is
    rejected, never guessed at."""
    parts = line.strip().split()
    if len(parts) != 5 or parts[0] != "CREASE" or parts[1] not in (WICKET, POPPING):
        raise ValueError(f"not a CREASE line: {line!r}")
    try:
        t_us, y_mm, z_mm = int(parts[2]), int(parts[3]), int(parts[4])
    except ValueError:
        raise ValueError(f"CREASE fields must be integers: {line!r}") from None
    if z_mm < -50 or z_mm > 6000 or abs(y_mm) > 6000:
        raise ValueError(f"CREASE reading outside any physical range: {line!r}")
    return CreaseReading(parts[1], t_us / 1e6, y_mm / 1000.0, max(0.0, z_mm / 1000.0))


@dataclass
class ModelResidual:
    plane: str
    dy_m: float           # measured - predicted, lateral
    dz_m: float           # measured - predicted, height


@dataclass(frozen=True)
class LegalityDecision:
    call: Optional[str]                    # "wide" or None
    basis: str                             # "sensor" / "physics model (no sensor reading)"
    wide_reasons: List[str]
    wide_margin_m: float
    borderline: bool
    hits_stumps: bool
    no_ball_flags: List[str]
    model_call: Optional[str]              # what the model predicted before the ball was bowled
    agrees_with_model: bool
    notes: List[str] = field(default_factory=list)


class LegalityObserver:
    """Decides the call from the sensor when there is a reading, else from the model, and logs how
    wrong the model was."""

    def __init__(self, rules: WideRules = WideRules(), hand: str = "right-handed"):
        self.rules, self.hand = rules, hand
        self.residuals: List[ModelResidual] = []

    def decide(self, predicted: DeliveryLegality, wicket: Optional[CreaseReading] = None,
               popping: Optional[CreaseReading] = None) -> LegalityDecision:
        notes: List[str] = []
        at_wkt = self._as_crossing(wicket, predicted.at_wicket, WICKET_X_M)
        at_pop = self._as_crossing(popping, predicted.at_popping_crease, POPPING_CREASE_X_M)
        for reading, model in ((wicket, predicted.at_wicket), (popping, predicted.at_popping_crease)):
            if reading is not None and model is not None:
                self.residuals.append(ModelResidual(reading.plane, reading.y_m - model.y_m, reading.z_m - model.z_m))
        wide, reasons, margin = judge_wide(at_wkt, at_pop, self.rules, self.hand)
        measured = wicket is not None or popping is not None
        if not measured:
            notes.append("No sensor reading for this delivery: the call is the physics model's prediction.")
        elif (wicket is None) != (popping is None):
            notes.append("Only one of the two planes was measured; the other is the model's prediction.")
        ref = at_wkt or at_pop
        flags = no_ball_flags(
            predicted.at_wicket.pitch_x_m if predicted.at_wicket else None,
            predicted.at_wicket.pitch_y_m if predicted.at_wicket else None,
            predicted.at_popping_crease.bounces if predicted.at_popping_crease else 0, at_pop, self.rules,
        )
        call = "wide" if wide else None
        return LegalityDecision(
            call=call, basis="sensor" if measured else "physics model (no sensor reading)", wide_reasons=reasons,
            wide_margin_m=margin, borderline=abs(margin) < self.rules.tolerance_m,
            hits_stumps=stumps_check(at_wkt).hits, no_ball_flags=flags, model_call=predicted.call,
            agrees_with_model=(call == predicted.call), notes=notes,
        )

    @staticmethod
    def _as_crossing(reading: Optional[CreaseReading], model: Optional[Crossing], x_m: float) -> Optional[Crossing]:
        """The sensor's numbers dressed as a Crossing, keeping the model's bounce history (a plane
        sensor cannot tell whether the ball has pitched)."""
        if reading is None:
            return model
        return Crossing(
            reading.t_s, x_m, reading.y_m, reading.z_m, model.vx if model else 0.0, model.vy if model else 0.0,
            model.vz if model else 0.0, model.bounces if model else 1, model.pitch_x_m if model else None,
            model.pitch_y_m if model else None, source=reading.source,
        )

    def model_error_summary(self) -> dict:
        """Mean and spread of (measured - predicted), per plane, over the session so far: the evidence
        for calibrating the bounce model. Empty until a sensor has reported."""
        out = {}
        for plane in (WICKET, POPPING):
            rs = [r for r in self.residuals if r.plane == plane]
            if not rs:
                continue
            dy, dz = [r.dy_m for r in rs], [r.dz_m for r in rs]
            mean = lambda v: sum(v) / len(v)
            std = lambda v: (sum((x - mean(v)) ** 2 for x in v) / len(v)) ** 0.5
            out[plane] = {"n": len(rs), "mean_dy_m": mean(dy), "std_dy_m": std(dy), "mean_dz_m": mean(dz), "std_dz_m": std(dz)}
        return out

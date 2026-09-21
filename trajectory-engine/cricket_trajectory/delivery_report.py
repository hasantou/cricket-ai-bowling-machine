"""
"How was that ball bowled?" — as a plain, readable report, built only
from data this project genuinely has for a machine delivery: what the
machine was commanded to do, what the physics simulation says the flight
did, and (when a release-point sensor exists) what the machine actually
measured leaving it.

This is the machine-side answer to "how the ball was bowled". It is
deliberately NOT something to be inferred from video: a phone clip cannot
reliably show the ball (see cv-pipeline/README.md), but a machine that
commands its own deliveries already knows their parameters, and the
simulation already computes where the ball pitches and how much it swings.

Length and line labels use conventional coaching bands measured at the
point the ball first hits the ground (the simulation stops there — see
simulate.py), so "line" here is the *pitching* line, not the line at the
stumps. Coaches disagree on exact bands and real "good length" shifts with
bowler pace and batter height; the thresholds below are named, adjustable,
and uncalibrated — the same honesty as net_outcome.py's thresholds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from . import constants as c
from .ball import Delivery
from .scorecard import classify_delivery_legality
from .laws import DeliveryLegality
from .simulate import SimulationResult

# --- length: distance from the BATTER's stumps to where the ball pitches ---
YORKER_MAX_FROM_STUMPS_M = 2.0
FULL_MAX_FROM_STUMPS_M = 5.0
GOOD_MAX_FROM_STUMPS_M = 7.5
SHORT_OF_LENGTH_MAX_FROM_STUMPS_M = 9.5

# --- line: lateral pitching position (y>0 = off side, right-handed batter) ---
STUMP_HALF_WIDTH_M = 0.114      # a stump line is ~22.9cm wide
OFF_CORRIDOR_MAX_M = 0.5        # "corridor of uncertainty" outside off
LEG_SIDE_LIMIT_M = -0.5

# --- swing: sideways drift in the air before pitching ---
STRAIGHT_SWING_MAX_CM = 3.0


def classify_length(pitch_x_m: float) -> Tuple[float, str]:
    """Returns (metres from the batter's stumps, label). A ball that hasn't
    bounced by the batting crease is a full toss."""
    from_stumps = c.PITCH_LENGTH_M - pitch_x_m
    if from_stumps < 0:
        return from_stumps, "full toss"
    if from_stumps < YORKER_MAX_FROM_STUMPS_M:
        return from_stumps, "yorker"
    if from_stumps < FULL_MAX_FROM_STUMPS_M:
        return from_stumps, "full"
    if from_stumps < GOOD_MAX_FROM_STUMPS_M:
        return from_stumps, "good length"
    if from_stumps < SHORT_OF_LENGTH_MAX_FROM_STUMPS_M:
        return from_stumps, "short of a length"
    return from_stumps, "short (bouncer territory)"


def classify_line(pitch_y_m: float) -> str:
    if abs(pitch_y_m) <= STUMP_HALF_WIDTH_M:
        return "on the stumps"
    if pitch_y_m > 0:
        return "outside off (corridor)" if pitch_y_m <= OFF_CORRIDOR_MAX_M else "wide outside off"
    return "leg side" if pitch_y_m >= LEG_SIDE_LIMIT_M else "well down the leg side"


def classify_swing(deviation_m: float) -> Tuple[float, str]:
    """Signed sideways drift in cm and a label, for a right-handed batter:
    drift toward the off side is swinging away, toward leg is swinging in."""
    cm = deviation_m * 100.0
    if abs(cm) < STRAIGHT_SWING_MAX_CM:
        return cm, "straight (no meaningful swing)"
    return cm, "swings away (toward off)" if cm > 0 else "swings in (toward leg)"


def describe_spin(spin_rad_s) -> Tuple[float, str]:
    """(rpm, description) from an angular-velocity vector, using the same
    axis convention as ball.py's Delivery.spin_vector: +y backspin, -y
    topspin, +z offspin, -z legspin."""
    sx, sy, sz = spin_rad_s
    magnitude = math.sqrt(sx * sx + sy * sy + sz * sz)
    rpm = magnitude * 60.0 / (2 * math.pi)
    if rpm < 50.0:
        return rpm, "no spin"
    axes = {"backspin": sy, "topspin": -sy, "off-spin": sz, "leg-spin": -sz}
    dominant = max(axes, key=axes.get)
    return rpm, dominant


@dataclass(frozen=True)
class DeliveryReport:
    label: str
    commanded_speed_kmh: float
    measured_speed_kmh: Optional[float]
    seam_angle_deg: float
    reverse_swing: bool
    spin_rpm: float
    spin_description: str
    pitch_from_stumps_m: float
    length_label: str
    pitch_y_m: float
    line_label: str
    swing_cm: float
    swing_label: str
    flight_time_s: float
    speed_at_pitch_kmh: float
    legality: Optional[str]
    assessment: Optional[DeliveryLegality] = None   # the bounce-aware, Laws-based call at the batter (laws.py), if computed

    @property
    def speed_error_kmh(self) -> Optional[float]:
        """Measured minus commanded: how far the real machine was from what
        it was told to do. None until a release sensor has reported."""
        if self.measured_speed_kmh is None:
            return None
        return self.measured_speed_kmh - self.commanded_speed_kmh

    def rows(self) -> List[Tuple[str, str]]:
        """(label, value) pairs, in the order a coach would want to read them."""
        rows = [("Pace (commanded)", f"{self.commanded_speed_kmh:.0f} km/h")]
        if self.measured_speed_kmh is not None:
            rows.append((
                "Pace (sensor-confirmed)",
                f"{self.measured_speed_kmh:.0f} km/h ({self.speed_error_kmh:+.1f} vs commanded)",
            ))
        if self.length_label == "full toss":
            length_text = "full toss — reaches the batter without bouncing"
        else:
            length_text = f"{self.length_label} — pitches {self.pitch_from_stumps_m:.1f} m from the stumps"
        rows += [
            ("Length", length_text),
            ("Line", f"{self.line_label} ({self.pitch_y_m * 100:+.0f} cm from centre)"),
            ("Swing", f"{self.swing_label} ({self.swing_cm:+.1f} cm)"),
            ("Spin", f"{self.spin_description}" + (f" at {self.spin_rpm:.0f} rpm" if self.spin_rpm >= 50 else "")),
            ("Seam", f"{self.seam_angle_deg:.0f}° to the flight" + (" — reverse swing regime" if self.reverse_swing else "")),
            ("Flight", f"{self.flight_time_s:.2f} s to pitch, {self.speed_at_pitch_kmh:.0f} km/h at pitch"),
        ]
        rows += self._legality_rows()
        return rows

    def _legality_rows(self) -> List[Tuple[str, str]]:
        a = self.assessment
        if a is None:
            return [("Legality", (self.legality or "fair delivery").replace("_", "-"))]
        if a.is_wide:
            call = "WIDE — " + "; ".join(a.wide_reasons)
        else:
            call = f"fair — {a.wide_margin_m:.2f} m inside the nearest wide limit" + (" (borderline)" if a.borderline else "")
        if a.at_wicket is None:
            stumps = "does not reach the stumps in this model"
        elif a.hits_stumps:
            stumps = f"yes — on target ({a.stumps.lateral_margin_m * 100:.0f} cm inside the stumps' width)"
        else:
            stumps = f"no — misses by {abs(min(a.stumps.lateral_margin_m, a.stumps.height_margin_m)) * 100:.0f} cm"
        rows = [("Legality (at the batter)", call), ("Would hit the stumps", stumps)]
        rows.append(("No-ball conditions (counted, not scored)", "; ".join(a.no_ball_flags) if a.no_ball_flags else "none"))
        return rows


def build_delivery_report(
    delivery: Delivery,
    result: SimulationResult,
    measured_release_speed_mps: Optional[float] = None,
    assessment: Optional[DeliveryLegality] = None,
) -> DeliveryReport:
    pitch_x, pitch_y = result.landing_point_m
    from_stumps, length_label = classify_length(pitch_x)
    swing_cm, swing_label = classify_swing(result.lateral_deviation_m)
    spin_rpm, spin_description = describe_spin(delivery.spin_rad_s)
    return DeliveryReport(
        label=delivery.label,
        commanded_speed_kmh=c.ms_to_kmh(delivery.speed_mps),
        measured_speed_kmh=(
            c.ms_to_kmh(measured_release_speed_mps) if measured_release_speed_mps is not None else None
        ),
        seam_angle_deg=delivery.seam_angle_deg,
        reverse_swing=delivery.reverse_swing,
        spin_rpm=spin_rpm,
        spin_description=spin_description,
        pitch_from_stumps_m=from_stumps,
        length_label=length_label,
        pitch_y_m=pitch_y,
        line_label=classify_line(pitch_y),
        swing_cm=swing_cm,
        swing_label=swing_label,
        flight_time_s=result.flight_time_s,
        speed_at_pitch_kmh=c.ms_to_kmh(float(result.trajectory["speed"].iloc[-1])),
        legality=(assessment.call if assessment is not None else classify_delivery_legality(result)),
        assessment=assessment,
    )

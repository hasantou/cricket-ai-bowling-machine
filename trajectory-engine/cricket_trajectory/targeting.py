"""
Aim a delivery: choose WHERE the ball should pitch, then solve for the launch angles that put it there.

Why this exists: the adaptive generator (adaptive.suggest_next_delivery) picks pace, seam and spin
from a difficulty target, but never the launch angles, so every ball left the machine on the same
line and at the same angle. Measured on 120 generated deliveries against the Laws-based checks in
laws.py: 35% were wides, 8% would have hit the stumps, and the length mix at the batter was 61%
full tosses, 32% yorkers, 8% full and NO good-length or short balls. A machine that cannot bowl a
length or hold a line is not a coaching tool.

The fix keeps the adaptive logic untouched: difficulty depends only on pace, seam and spin (never on
launch angle), so a candidate is chosen for its difficulty exactly as before and is then AIMED at a
sampled target — a length band and a line inside a legal corridor. Vertical launch angle sets how far
the ball travels before it pitches (found by bisection, since landing distance falls monotonically as
the angle steepens downward); horizontal launch angle sets the line (a secant correction, since swing
and spin bend the flight after release). Both use the same physics as everything else here.

Limits, stated: the aim is solved on the simulated physics, so its accuracy is only as good as that
model (uncalibrated); the length mix and the line corridor are choices, named below, not measured
coaching norms; and a slow ball cannot always reach every length, in which case the solver says so
and another candidate is drawn.
"""

from __future__ import annotations

import dataclasses
import math
import random
from typing import Dict, Optional, Tuple

from dataclasses import dataclass

from . import constants as c
from .adaptive import DeliverySuggestion, PlayerProfile, suggest_next_delivery_with_reason
from .ball import BallProperties, Delivery, Environment
from .laws import WICKET_X_M, WideRules, assess_delivery
from .simulate import run_simulation

# Metres from the batter's stumps at which the ball pitches (delivery_report's length bands).
LENGTH_BANDS: Dict[str, Tuple[float, float]] = {
    "yorker": (0.4, 1.9),
    "full": (2.1, 4.9),
    "good length": (5.1, 7.4),
    "short of a length": (7.6, 9.4),
    "short": (9.6, 11.5),
}
DEFAULT_LENGTH_MIX: Dict[str, float] = {
    "yorker": 0.10, "full": 0.20, "good length": 0.40, "short of a length": 0.20, "short": 0.10,
}
LINE_CORRIDOR_M = (-0.30, 0.45)        # where the ball pitches laterally; + = off side. Well inside the wide limits.
ANGLE_RANGE_DEG = (-14.0, 8.0)         # the vertical launch angles the solver will consider
LENGTH_TOLERANCE_M = 0.10
LINE_TOLERANCE_M = 0.05
MAX_LINE_ITERATIONS = 4


def _landing(ball: BallProperties, env: Environment, d: Delivery) -> Tuple[float, float]:
    return run_simulation(ball, env, d).landing_point_m


def aim_delivery(ball: BallProperties, env: Environment, delivery: Delivery, pitch_x_m: float, pitch_y_m: float,
                 length_tol_m: float = LENGTH_TOLERANCE_M, line_tol_m: float = LINE_TOLERANCE_M) -> Optional[Delivery]:
    """A copy of `delivery` with its launch angles solved so the ball first lands at (pitch_x_m, pitch_y_m)
    to within the tolerances; None if it cannot (a ball too slow to reach that length, say)."""
    lo, hi = ANGLE_RANGE_DEG
    phi = math.degrees(math.atan2(pitch_y_m, pitch_x_m))        # first guess for the line
    theta = 0.5 * (lo + hi)
    for _ in range(MAX_LINE_ITERATIONS):
        a, b = lo, hi
        # landing x DEcreases as the ball is thrown more steeply down, i.e. increases with theta
        for _ in range(12):
            theta = 0.5 * (a + b)
            x, _ = _landing(ball, env, dataclasses.replace(delivery, vertical_launch_deg=theta, horizontal_launch_deg=phi))
            if abs(x - pitch_x_m) <= length_tol_m:
                break
            if x < pitch_x_m:
                a = theta
            else:
                b = theta
        cand = dataclasses.replace(delivery, vertical_launch_deg=theta, horizontal_launch_deg=phi)
        x, y = _landing(ball, env, cand)
        if abs(x - pitch_x_m) > length_tol_m + 0.15:
            return None                                         # the length is not reachable at this pace
        if abs(y - pitch_y_m) <= line_tol_m:
            return dataclasses.replace(cand, label=f"{delivery.label} aimed {pitch_x_m:.1f}m/{pitch_y_m:+.2f}m")
        phi += math.degrees(math.atan2(pitch_y_m - y, max(x, 1.0)))       # correct the line, then re-solve the length
    return None


def sample_target(rng: random.Random, length_mix: Dict[str, float] = DEFAULT_LENGTH_MIX,
                  corridor: Tuple[float, float] = LINE_CORRIDOR_M) -> Tuple[str, float, float]:
    """(length band name, pitch_x_m, pitch_y_m)."""
    names = list(length_mix)
    band = rng.choices(names, weights=[length_mix[n] for n in names], k=1)[0]
    lo, hi = LENGTH_BANDS[band]
    return band, WICKET_X_M - rng.uniform(lo, hi), rng.uniform(*corridor)


@dataclass(frozen=True)
class AimedDeliverySuggestion:
    """Same as adaptive.DeliverySuggestion, but for the AIMED delivery
    (line and length solved in) that suggest_aimed_delivery() actually
    returns -- the targeted_skill/reason describe the adaptive engine's
    pick BEFORE aiming, since aiming a ball at a length never changes which
    skill it was chosen to test."""
    delivery: Delivery
    targeted_skill: Optional[str]
    reason: str


def suggest_aimed_delivery_with_reason(
    profile: PlayerProfile, ball: BallProperties, env: Optional[Environment] = None,
    challenge_margin: float = 60.0, rng: Optional[random.Random] = None,
    length_mix: Dict[str, float] = DEFAULT_LENGTH_MIX, corridor: Tuple[float, float] = LINE_CORRIDOR_M,
    max_attempts: int = 6, legal_only: bool = True, wide_rules: WideRules = WideRules(), **suggest_kwargs,
) -> AimedDeliverySuggestion:
    """The adaptive suggestion (unchanged difficulty logic, now skill-targeted -- see
    adaptive.suggest_next_delivery_with_reason()), aimed at a sampled line and length, and —
    if `legal_only` — verified not to be a wide by the Laws-based check. Falls back to the unaimed
    suggestion, labelled as such, only if nothing could be aimed after `max_attempts` tries."""
    rng = rng or random.Random()
    env = env or Environment()
    last_suggestion: Optional[DeliverySuggestion] = None
    for _ in range(max_attempts):
        suggestion = suggest_next_delivery_with_reason(profile, ball, challenge_margin=challenge_margin, rng=rng, **suggest_kwargs)
        last_suggestion = suggestion
        candidate = suggestion.delivery
        band, px, py = sample_target(rng, length_mix, corridor)
        aimed = aim_delivery(ball, env, candidate, px, py)
        if aimed is None:
            continue
        if legal_only and assess_delivery(ball, env, aimed, wide_rules).is_wide:
            continue
        return AimedDeliverySuggestion(
            dataclasses.replace(aimed, label=f"{aimed.label} [{band}]"), suggestion.targeted_skill, suggestion.reason,
        )
    last = last_suggestion.delivery
    return AimedDeliverySuggestion(
        dataclasses.replace(last, label=f"{last.label} [unaimed: could not solve a legal line and length]"),
        last_suggestion.targeted_skill, last_suggestion.reason,
    )


def suggest_aimed_delivery(
    profile: PlayerProfile, ball: BallProperties, env: Optional[Environment] = None,
    challenge_margin: float = 60.0, rng: Optional[random.Random] = None,
    length_mix: Dict[str, float] = DEFAULT_LENGTH_MIX, corridor: Tuple[float, float] = LINE_CORRIDOR_M,
    max_attempts: int = 6, legal_only: bool = True, wide_rules: WideRules = WideRules(), **suggest_kwargs,
) -> Delivery:
    """Unchanged public behaviour: just the aimed Delivery, no reason attached.
    See suggest_aimed_delivery_with_reason() for the same pick plus WHY it was
    made -- this is now a thin wrapper around that, not a separate
    implementation, so the two can never quietly drift apart."""
    return suggest_aimed_delivery_with_reason(
        profile, ball, env, challenge_margin, rng, length_mix, corridor, max_attempts, legal_only, wide_rules,
        **suggest_kwargs,
    ).delivery

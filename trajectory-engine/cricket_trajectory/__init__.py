"""
cricket_trajectory
===================

A physics-based simulator for cricket ball flight, combining gravity,
speed-dependent aerodynamic drag, spin-induced Magnus force, and seam-induced
swing force into one numerically-integrated trajectory model.

See README.md for the physics background and usage examples, or demo.py for
runnable scenarios.
"""

from .ball import BallProperties, Environment, Delivery
from .simulate import run_simulation, SimulationResult
from .adaptive import (
    PlayerProfile,
    FacedRecord,
    delivery_difficulty_rating,
    expected_success,
    suggest_next_delivery,
    next_delivery_after,
    OUTCOME_SCORES,
)
from .scorecard import (
    Scorecard,
    BallRecord,
    ScoringInfo,
    DEFAULT_SCORING,
    score_outcome,
    classify_delivery_legality,
)

__all__ = [
    "BallProperties",
    "Environment",
    "Delivery",
    "run_simulation",
    "SimulationResult",
    "PlayerProfile",
    "FacedRecord",
    "delivery_difficulty_rating",
    "expected_success",
    "suggest_next_delivery",
    "next_delivery_after",
    "OUTCOME_SCORES",
    "Scorecard",
    "BallRecord",
    "ScoringInfo",
    "DEFAULT_SCORING",
    "score_outcome",
    "classify_delivery_legality",
]

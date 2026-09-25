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
    DeliverySuggestion,
    delivery_difficulty_rating,
    dimension_delivery_rating,
    expected_success,
    suggest_next_delivery,
    suggest_next_delivery_with_reason,
    next_delivery_after,
    next_delivery_after_with_reason,
    OUTCOME_SCORES,
    SKILL_DIMENSIONS,
    DEFAULT_SPEED_RANGE_KMH,
    SPIN_RPM_RANGE,
)
from .scorecard import (
    Scorecard,
    BallRecord,
    ScoringInfo,
    DEFAULT_SCORING,
    score_outcome,
    style_delivery_scoring,
    classify_delivery_legality,
)
from .delivery_report import DeliveryReport, build_delivery_report
from .shot_analysis import ShotAnalysis, analyse_shot
from .shot_fusion import FusedShot, fuse_shot
from .laws import DeliveryLegality, WideRules, assess_delivery
from .crease_crossing import BounceModel, Crossing, predict_crossing
from .targeting import aim_delivery, suggest_aimed_delivery, suggest_aimed_delivery_with_reason, AimedDeliverySuggestion
from .delivery_story import DeliveryStory, ImpactInfo, SyncInfo, VideoDeliveryInfo, build_story
from .net_outcome import (
    NetOutcome,
    NET_OUTCOMES,
    classify_net_outcome,
    classify_from_simulation,
)

__all__ = [
    "BallProperties",
    "Environment",
    "Delivery",
    "run_simulation",
    "SimulationResult",
    "PlayerProfile",
    "FacedRecord",
    "DeliverySuggestion",
    "delivery_difficulty_rating",
    "dimension_delivery_rating",
    "expected_success",
    "suggest_next_delivery",
    "suggest_next_delivery_with_reason",
    "next_delivery_after",
    "next_delivery_after_with_reason",
    "OUTCOME_SCORES",
    "SKILL_DIMENSIONS",
    "DEFAULT_SPEED_RANGE_KMH",
    "SPIN_RPM_RANGE",
    "Scorecard",
    "BallRecord",
    "ScoringInfo",
    "DEFAULT_SCORING",
    "score_outcome",
    "style_delivery_scoring",
    "classify_delivery_legality",
    "DeliveryReport",
    "build_delivery_report",
    "ShotAnalysis",
    "analyse_shot",
    "FusedShot",
    "fuse_shot",
    "DeliveryLegality",
    "WideRules",
    "assess_delivery",
    "BounceModel",
    "Crossing",
    "predict_crossing",
    "aim_delivery",
    "suggest_aimed_delivery",
    "suggest_aimed_delivery_with_reason",
    "AimedDeliverySuggestion",
    "DeliveryStory",
    "ImpactInfo",
    "SyncInfo",
    "VideoDeliveryInfo",
    "build_story",
    "NetOutcome",
    "NET_OUTCOMES",
    "classify_net_outcome",
    "classify_from_simulation",
]

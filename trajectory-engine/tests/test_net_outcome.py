import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cricket_trajectory.adaptive import OUTCOME_SCORES
from cricket_trajectory.scorecard import DEFAULT_SCORING
from cricket_trajectory.net_outcome import (
    NET_OUTCOMES,
    classify_net_outcome,
    classify_from_simulation,
    DEAD_BAT_DISTANCE_M,
    WELL_STRUCK_DISTANCE_M,
    SKIED_HEIGHT_M,
)
from cricket_trajectory.ball import BallProperties, Environment, Delivery
from cricket_trajectory.simulate import run_simulation


def test_every_net_outcome_maps_to_a_real_adaptive_outcome_key():
    # Regression guard: this module's whole design relies on reusing
    # adaptive.OUTCOME_SCORES and scorecard.DEFAULT_SCORING unchanged - a
    # typo here would silently KeyError deep inside scorecard.py instead
    # of failing where the mistake actually is.
    for outcome in NET_OUTCOMES.values():
        assert outcome.adaptive_outcome in OUTCOME_SCORES, (
            f"{outcome.label} maps to '{outcome.adaptive_outcome}', not a real OUTCOME_SCORES key"
        )
        assert outcome.adaptive_outcome in DEFAULT_SCORING, (
            f"{outcome.label} maps to '{outcome.adaptive_outcome}', not a real DEFAULT_SCORING key"
        )


def test_dead_bat_below_threshold_distance():
    result = classify_net_outcome(post_contact_distance_m=1.0, post_contact_max_height_m=0.5)
    assert result.label == "dead_bat"


def test_controlled_placement_mid_distance():
    result = classify_net_outcome(post_contact_distance_m=6.0, post_contact_max_height_m=0.8)
    assert result.label == "controlled_placement"


def test_well_struck_beyond_threshold_distance():
    result = classify_net_outcome(post_contact_distance_m=WELL_STRUCK_DISTANCE_M + 1, post_contact_max_height_m=1.0)
    assert result.label == "well_struck"


def test_skied_takes_priority_over_distance():
    # A shot that travels far but also balloons up is still a mistimed
    # chance, not a well-struck shot - height is checked first on purpose.
    result = classify_net_outcome(
        post_contact_distance_m=WELL_STRUCK_DISTANCE_M + 5,
        post_contact_max_height_m=SKIED_HEIGHT_M + 1,
    )
    assert result.label == "mistimed_skied"


def test_boundary_values_are_exclusive_on_the_low_side():
    # Exactly at DEAD_BAT_DISTANCE_M should already read as the next tier
    # up, not dead_bat - confirms the "<" (not "<=") comparison is real.
    result = classify_net_outcome(post_contact_distance_m=DEAD_BAT_DISTANCE_M, post_contact_max_height_m=0.5)
    assert result.label == "controlled_placement"


def test_classify_from_simulation_matches_direct_classification():
    ball = BallProperties()
    env = Environment()
    # A firmly struck, flat shot off the bat - release position and speed
    # chosen to land well past WELL_STRUCK_DISTANCE_M.
    delivery = Delivery(release_pos_m=(20.0, 0.0, 0.9), speed_mps=25.0,
                         vertical_launch_deg=2.0, label="test shot")
    result = run_simulation(ball, env, delivery)

    from_sim = classify_from_simulation(result)
    distance = abs(result.landing_point_m[0] - result.trajectory["x"].iloc[0])
    height = float(result.trajectory["z"].max())
    direct = classify_net_outcome(distance, height)

    assert from_sim.label == direct.label
    assert from_sim.label == "well_struck"

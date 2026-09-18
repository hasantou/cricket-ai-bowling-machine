import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cricket_trajectory.adaptive import OUTCOME_SCORES
from cricket_trajectory.scorecard import DEFAULT_SCORING
from cricket_trajectory.net_outcome import (
    NET_OUTCOMES,
    classify_net_outcome,
    classify_from_simulation,
    classify_exit_velocity,
    ExitVelocity,
    DEAD_BAT_DISTANCE_M,
    WELL_STRUCK_DISTANCE_M,
    SKIED_HEIGHT_M,
    DEAD_BAT_SPEED_MPS,
    WELL_STRUCK_SPEED_MPS,
    SKIED_ELEVATION_DEG,
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


def test_exit_velocity_speed_elevation_and_azimuth_are_correct():
    ev = ExitVelocity(vx=10.0, vy=10.0, vz=0.0)
    assert abs(ev.speed_mps - (200.0 ** 0.5)) < 1e-9
    assert abs(ev.elevation_deg - 0.0) < 1e-9
    assert abs(ev.azimuth_deg - 45.0) < 1e-9
    assert ev.direction_label() == "off side"


def test_direction_label_leg_side_and_straight():
    assert ExitVelocity(vx=10.0, vy=-10.0, vz=0.0).direction_label() == "leg side"
    assert ExitVelocity(vx=10.0, vy=0.5, vz=0.0).direction_label() == "straight"


def test_classify_exit_velocity_dead_bat_on_low_speed():
    outcome, ev = classify_exit_velocity(vx=3.0, vy=0.0, vz=0.0)
    assert outcome.label == "dead_bat"
    assert abs(ev.speed_mps - 3.0) < 1e-9


def test_classify_exit_velocity_controlled_placement_mid_speed():
    outcome, _ = classify_exit_velocity(vx=10.0, vy=0.0, vz=0.0)
    assert outcome.label == "controlled_placement"


def test_classify_exit_velocity_well_struck_high_speed():
    outcome, _ = classify_exit_velocity(vx=25.0, vy=0.0, vz=0.0)
    assert outcome.label == "well_struck"


def test_classify_exit_velocity_skied_takes_priority_over_speed():
    # Fast off the bat (would otherwise read as well_struck) but launched
    # steeply upward - elevation is checked first, same priority as
    # classify_net_outcome()'s height-before-distance check.
    outcome, ev = classify_exit_velocity(vx=20.0, vy=0.0, vz=15.0)
    assert ev.speed_mps > WELL_STRUCK_SPEED_MPS  # would qualify as well_struck by speed alone
    assert outcome.label == "mistimed_skied"


def test_classify_exit_velocity_boundary_is_exclusive_on_the_low_side():
    outcome, _ = classify_exit_velocity(vx=DEAD_BAT_SPEED_MPS, vy=0.0, vz=0.0)
    assert outcome.label == "controlled_placement"


def test_classify_exit_velocity_agrees_with_distance_based_classification_in_spirit():
    # Not the same function, but both paths should agree on the clear
    # cases: a gentle defensive-speed shot with no elevation shouldn't
    # read as well_struck under either classifier.
    outcome_by_velocity, _ = classify_exit_velocity(vx=2.0, vy=0.0, vz=0.0)
    outcome_by_distance = classify_net_outcome(post_contact_distance_m=1.0, post_contact_max_height_m=0.3)
    assert outcome_by_velocity.label == outcome_by_distance.label == "dead_bat"

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from cricket_trajectory import BallProperties, Delivery, Environment
from cricket_trajectory import constants as c
from cricket_trajectory.crease_crossing import Crossing
from cricket_trajectory.laws import (
    APPLIES, BALL_RADIUS_M, LAWS, NA, NEEDS_SENSOR, PITCH_HALF_WIDTH_M, PITCH_LENGTH_M, POPPING_CREASE_X_M,
    STUMP_HEIGHT_M, STUMP_SET_WIDTH_M, WICKET_X_M, WideRules, assess_delivery, coverage, judge_wide,
    no_ball_flags, stumps_check,
)


def crossing(y=0.0, z=0.5, bounces=1, pitch_x=15.0, pitch_y=0.0, x=WICKET_X_M):
    return Crossing(0.6, x, y, z, 30.0, 0.0, -1.0, bounces, pitch_x if bounces else None, pitch_y if bounces else None)


# ---- dimensions, as read from the Laws ----
def test_dimensions_match_the_laws():
    assert PITCH_LENGTH_M == 20.12 == c.PITCH_LENGTH_M                    # MCC 6.1: 22 yards
    assert PITCH_HALF_WIDTH_M == pytest.approx(3.05 / 2, abs=0.005)        # MCC 6.1: 10 ft wide
    assert STUMP_SET_WIDTH_M == pytest.approx(9 * 0.0254, abs=1e-4)         # MCC 8.1: 9 in
    assert STUMP_HEIGHT_M == pytest.approx(28 * 0.0254, abs=1e-4)           # MCC 8.2: 28 in
    assert POPPING_CREASE_X_M == pytest.approx(20.12 - 1.22)                # MCC 7.3: 4 ft in front of the bowling crease
    assert 0.0356 < BALL_RADIUS_M < 0.0366                                  # MCC 4.1: 22.4-22.9 cm circumference


# ---- Law 22: wide ----
def test_a_ball_inside_the_reach_limits_is_fair_and_the_margin_is_the_room_to_spare():
    wide, reasons, margin = judge_wide(crossing(y=0.5), crossing(x=POPPING_CREASE_X_M))
    assert not wide and reasons == [] and margin == pytest.approx(0.89 - 0.5)


def test_beyond_the_off_limit_is_a_wide_and_says_by_how_much():
    wide, reasons, margin = judge_wide(crossing(y=1.10), None)
    assert wide and margin == pytest.approx(-0.21) and "off side" in reasons[0] and "Law 22.1" in reasons[0]


def test_the_leg_side_uses_its_own_tighter_limit():
    assert judge_wide(crossing(y=-0.40), None)[0] is False
    assert judge_wide(crossing(y=-0.65), None)[0] is True


def test_a_left_handers_sides_are_mirrored():
    assert judge_wide(crossing(y=0.7), None, hand="right-handed")[0] is False          # off side for a righty
    assert judge_wide(crossing(y=0.7), None, hand="left-handed")[0] is True            # that is the LEG side for a lefty (limit 0.5)


def test_where_the_striker_stands_moves_the_limits_with_him():
    """Law 22.1: wide of WHERE THE STRIKER IS STANDING. A batter who has taken guard outside leg stump
    (striker_y negative) makes an off-side ball wider."""
    ball = crossing(y=0.7)
    assert judge_wide(ball, None, WideRules(striker_y_m=0.0))[0] is False
    assert judge_wide(ball, None, WideRules(striker_y_m=-0.3))[0] is True


def test_ICC_a_ball_above_head_height_at_the_popping_crease_is_a_wide_even_if_on_line():
    high = crossing(y=0.0, z=2.0, bounces=0, x=POPPING_CREASE_X_M)
    wide, reasons, margin = judge_wide(crossing(y=0.0), high)
    assert wide and "head height" in reasons[0] and margin < 0


def test_no_ball_and_borderline_margins():
    wide, _, margin = judge_wide(crossing(y=0.88), None)
    assert not wide and margin == pytest.approx(0.01)                        # 1 cm inside: legal but within tolerance


def test_no_batter_position_no_call():
    wide, reasons, _ = judge_wide(None, None)
    assert not wide and "never reached" in reasons[0]


# ---- Law 21.7 / 41.7.1: no-ball flags (informational) ----
def test_pitching_off_the_pitch_is_flagged():
    flags = no_ball_flags(15.0, 1.6, 1, None)
    assert len(flags) == 1 and "off the pitch" in flags[0] and "21.7" in flags[0]
    assert no_ball_flags(15.0, 1.0, 1, None) == []


def test_a_ball_bouncing_twice_is_flagged():
    assert any("more than once" in f for f in no_ball_flags(10.0, 0.0, 2, None))


def test_a_full_toss_above_waist_at_the_popping_crease_is_flagged_but_a_low_one_is_not():
    assert any("waist" in f for f in no_ball_flags(None, None, 0, crossing(z=1.3, bounces=0, x=POPPING_CREASE_X_M)))
    assert no_ball_flags(None, None, 0, crossing(z=0.6, bounces=0, x=POPPING_CREASE_X_M)) == []
    assert no_ball_flags(None, None, 1, crossing(z=1.3, bounces=1, x=POPPING_CREASE_X_M)) == []       # pitched: not a full toss


# ---- Law 32: bowled (would hit the stumps) ----
def test_a_ball_on_the_stumps_would_hit_them():
    r = stumps_check(crossing(y=0.0, z=0.4))
    assert r.hits and r.lateral_margin_m > 0.1 and r.height_margin_m > 0.3


def test_just_wide_of_off_stump_misses_and_over_the_top_misses():
    half = STUMP_SET_WIDTH_M / 2
    assert stumps_check(crossing(y=half + BALL_RADIUS_M - 0.005)).hits             # the ball's edge clips the stump
    r = stumps_check(crossing(y=half + BALL_RADIUS_M + 0.02))
    assert not r.hits and r.lateral_margin_m < 0
    over = stumps_check(crossing(y=0.0, z=STUMP_HEIGHT_M + BALL_RADIUS_M + 0.05))
    assert not over.hits and over.height_margin_m < 0


def test_a_ball_that_never_arrives_does_not_hit_anything():
    assert stumps_check(None).hits is False


# ---- end to end through the physics (deterministic) ----
def _assess(**kw):
    return assess_delivery(BallProperties(), Environment(), Delivery(**kw))


def test_end_to_end_a_straight_ball_is_fair_and_hits_the_stumps():
    r = _assess(speed_mps=140 / 3.6, vertical_launch_deg=-3.0)
    assert r.call is None and r.hits_stumps and r.no_ball_flags == []
    assert r.at_wicket.bounced and 0.0 <= r.at_wicket.z_m < 0.6


def test_end_to_end_aimed_well_outside_off_is_a_wide_that_misses_the_stumps():
    r = _assess(speed_mps=110 / 3.6, vertical_launch_deg=-3.0, horizontal_launch_deg=3.0)
    assert r.is_wide and not r.hits_stumps and r.wide_margin_m < 0 and r.at_wicket.y_m > 0.89


def test_end_to_end_down_the_leg_side_is_a_wide():
    r = _assess(speed_mps=120 / 3.6, vertical_launch_deg=-3.0, horizontal_launch_deg=-2.5)
    assert r.is_wide and "leg side" in r.wide_reasons[0]


def test_end_to_end_a_ball_aimed_a_little_off_is_fair_and_misses():
    r = _assess(speed_mps=140 / 3.6, vertical_launch_deg=-3.0, horizontal_launch_deg=1.6)
    assert r.call is None and not r.hits_stumps and 0.2 < r.wide_margin_m < 0.5


# ---- the rulebook map ----
def test_all_forty_two_laws_are_accounted_for_once():
    assert [law.number for law in LAWS] == list(range(1, 43))
    assert all(law.title and law.note for law in LAWS)


def test_the_machine_relevant_laws_are_marked_implemented_and_the_rest_honestly_not():
    cov = coverage()
    assert {3, 6, 7, 8, 17, 18, 22, 29, 32} <= set(cov[APPLIES])
    assert {19, 23, 33, 34, 35, 36} <= set(cov[NEEDS_SENSOR])
    assert {24, 27, 28, 37, 38, 39} <= set(cov[NA])


# ---- Law 32 in the app's no-contact path ----
def test_no_contact_is_bowled_only_if_the_ball_was_on_target():
    from cricket_trajectory.laws import resolve_no_contact
    outcome, why = resolve_no_contact(True)
    assert outcome == "missed" and "bowled" in why
    outcome, why = resolve_no_contact(False)
    assert outcome == "beaten" and "beaten, not bowled" in why


def test_the_two_no_contact_outcomes_score_as_a_wicket_and_a_dot_ball_respectively():
    from cricket_trajectory.scorecard import score_outcome
    assert score_outcome("missed").is_wicket and score_outcome("missed").dismissal == "bowled"
    assert not score_outcome("beaten").is_wicket and score_outcome("beaten").runs == 0

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cricket_trajectory.ball import BallProperties, Delivery
from cricket_trajectory.adaptive import (
    SKILL_DIMENSIONS,
    PlayerProfile,
    _dimension_factors,
    _dimension_relevance,
    delivery_difficulty_rating,
    dimension_delivery_rating,
    expected_success,
)

BALL = BallProperties()


def pure_pace_delivery():
    """As close to a pace-only test as real physics allows: fast, no seam
    angle (no swing/seam-drag), no spin."""
    return Delivery(speed_mps=45.0, seam_angle_deg=0.0, spin_rad_s=(0.0, 0.0, 0.0), label="pace")


def leg_break_delivery():
    """Slow enough that pace barely registers, heavy spin."""
    return Delivery(speed_mps=18.0, seam_angle_deg=0.0, spin_rad_s=(0.0, 0.0, 120.0), label="legbreak")


# ---- dimension_delivery_rating: same Elo-scale mapping as the overall rating, for one factor ----

def test_dimension_rating_at_the_three_reference_points():
    assert dimension_delivery_rating("pace", {"pace": 0.0}) == 1200.0 - 600.0
    assert dimension_delivery_rating("pace", {"pace": 0.5}) == 1200.0
    assert dimension_delivery_rating("pace", {"pace": 1.0}) == 1200.0 + 600.0


def test_dimension_rating_clips_factors_above_one():
    # _dimension_factors() itself caps pace at 1.3 / others at 1.5 -- confirm the Elo mapping
    # doesn't blow past the intended scale if a factor above 1.0 ever reaches it.
    assert dimension_delivery_rating("pace", {"pace": 1.3}) == dimension_delivery_rating("pace", {"pace": 1.0})


# ---- _dimension_relevance: shares must sum to 1, and reflect what actually made the ball hard ----

def test_relevance_shares_always_sum_to_one():
    for delivery in (pure_pace_delivery(), leg_break_delivery(), Delivery()):
        factors = _dimension_factors(delivery, BALL, 1.225, 1.8e-5)
        relevance = _dimension_relevance(factors)
        assert abs(sum(relevance.values()) - 1.0) < 1e-9


def test_all_zero_factors_fall_back_to_equal_relevance_not_a_crash():
    relevance = _dimension_relevance({"pace": 0.0, "swing": 0.0, "seam": 0.0, "spin": 0.0})
    assert relevance == {d: 0.25 for d in SKILL_DIMENSIONS}


def test_a_fast_seamless_spinless_ball_is_overwhelmingly_a_pace_test():
    factors = _dimension_factors(pure_pace_delivery(), BALL, 1.225, 1.8e-5)
    relevance = _dimension_relevance(factors)
    assert relevance["pace"] > 0.9
    assert relevance["spin"] < 0.05


def test_a_leg_break_has_more_spin_relevance_than_a_spinless_ball_of_the_same_pace():
    """NOT "spin dominates a leg-break's relevance" -- it honestly doesn't,
    because magnus_lift_coefficient() saturates at cl_max=0.35 and even that
    ceiling is small next to a cricket ball's weight (see
    _dimension_relevance()'s docstring). What IS true, and worth testing:
    adding real spin measurably raises spin's relevance share over an
    otherwise-identical ball with none."""
    spinning = _dimension_factors(leg_break_delivery(), BALL, 1.225, 1.8e-5)
    same_pace_no_spin = _dimension_factors(
        Delivery(speed_mps=18.0, seam_angle_deg=0.0, spin_rad_s=(0.0, 0.0, 0.0)), BALL, 1.225, 1.8e-5,
    )
    assert _dimension_relevance(spinning)["spin"] > _dimension_relevance(same_pace_no_spin)["spin"]


# ---- PlayerProfile: defaults, weakest_skill, and the actual per-dimension update ----

def test_a_fresh_profile_starts_equal_on_every_dimension():
    p = PlayerProfile(name="P")
    assert p.skill_ratings == {d: 1000.0 for d in SKILL_DIMENSIONS}


def test_weakest_skill_is_deterministic_on_a_tie():
    p = PlayerProfile(name="P")
    assert p.weakest_skill() == SKILL_DIMENSIONS[0]


def test_weakest_skill_picks_the_actual_lowest_rated_dimension():
    p = PlayerProfile(name="P")
    p.skill_ratings["spin"] = 800.0
    assert p.weakest_skill() == "spin"


def test_record_outcome_still_updates_the_overall_rating_exactly_as_before():
    """Regression: the overall-rating math must not change just because
    per-dimension tracking was added alongside it."""
    p = PlayerProfile(name="P", rating=1000.0, k_factor=24.0)
    delivery = pure_pace_delivery()
    d_rating = delivery_difficulty_rating(delivery, BALL)
    expected = expected_success(1000.0, d_rating)
    record = p.record_outcome(delivery, BALL, "missed")
    assert record.rating_after == 1000.0 + 24.0 * (0.0 - expected)
    assert p.rating == record.rating_after


def test_a_pure_pace_ball_moves_only_the_pace_rating_not_the_others_at_all():
    """A fast, seam-angle-0, spin-0 delivery has swing/seam/spin factors of
    EXACTLY 0.0 (confirmed directly, not assumed) -- so their relevance
    share is exactly 0.0 too, and record_outcome() should leave those three
    ratings completely untouched, not just "mostly" untouched. A surprising
    outcome ("six" against a ball rated far harder than the player) is used
    so the pace move itself is large and unambiguous, not lost in Elo noise."""
    p = PlayerProfile(name="P")
    p.record_outcome(pure_pace_delivery(), BALL, "six")
    pace_move = abs(p.skill_ratings["pace"] - 1000.0)
    other_moves = [abs(p.skill_ratings[d] - 1000.0) for d in SKILL_DIMENSIONS if d != "pace"]
    assert pace_move > 15.0  # a real, large, unmissable move
    assert other_moves == [0.0, 0.0, 0.0]


def test_a_leg_break_moves_the_spin_rating_more_than_a_spinless_ball_would():
    """See test_a_leg_break_has_more_spin_relevance... above for why this
    doesn't claim spin outweighs pace overall -- it honestly doesn't. What's
    tested here is the comparative claim record_outcome() can actually back:
    real spin present moves the spin rating more than an otherwise-identical
    ball with none."""
    with_spin = PlayerProfile(name="P")
    with_spin.record_outcome(leg_break_delivery(), BALL, "missed")

    without_spin = PlayerProfile(name="P")
    without_spin.record_outcome(
        Delivery(speed_mps=18.0, seam_angle_deg=0.0, spin_rad_s=(0.0, 0.0, 0.0), label="nospin"), BALL, "missed",
    )
    spin_move_with = abs(with_spin.skill_ratings["spin"] - 1000.0)
    spin_move_without = abs(without_spin.skill_ratings["spin"] - 1000.0)
    assert spin_move_with > spin_move_without
    assert spin_move_without == 0.0


def test_faced_record_carries_the_relevance_and_resulting_skill_ratings():
    p = PlayerProfile(name="P")
    record = p.record_outcome(pure_pace_delivery(), BALL, "controlled")
    assert abs(sum(record.dimension_relevance.values()) - 1.0) < 1e-9
    assert record.skill_ratings_after == p.skill_ratings  # a snapshot taken AFTER this ball's update


# ---- mutation check: relevance-weighting is load-bearing, not decorative ----

def test_without_relevance_weighting_a_pure_pace_ball_would_also_move_spin_a_lot():
    """Directly exercises what record_outcome() would do WITHOUT the
    relevance multiplier, to prove that multiplier is the thing keeping an
    irrelevant dimension from moving on a delivery that never tested it."""
    p = PlayerProfile(name="P")
    delivery = pure_pace_delivery()
    factors = _dimension_factors(delivery, BALL, 1.225, 1.8e-5)
    spin_rating_dim = dimension_delivery_rating("spin", factors)
    spin_expected = expected_success(p.skill_ratings["spin"], spin_rating_dim)
    unweighted_move = abs(p.k_factor * (0.0 - spin_expected))  # outcome_score=0.0 for "missed"

    p.record_outcome(delivery, BALL, "missed")
    actual_spin_move = abs(p.skill_ratings["spin"] - 1000.0)
    assert actual_spin_move < 0.1 * unweighted_move


# ---- backward compatibility: old-style construction (no new fields given) still works ----

def test_player_profile_still_constructs_with_only_name_and_rating():
    p = PlayerProfile(name="P", rating=1100.0)
    assert p.rating == 1100.0
    assert p.skill_ratings == {d: 1000.0 for d in SKILL_DIMENSIONS}

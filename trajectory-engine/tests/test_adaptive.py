import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cricket_trajectory.ball import BallProperties, Delivery
from cricket_trajectory.adaptive import (
    NEUTRAL_SEAM_ANGLE_DEG,
    NEUTRAL_SPEED_RANGE_KMH,
    SEAM_DRAG_PEAK_ANGLE_DEG,
    SKILL_DIMENSIONS,
    SKILL_TIE_SPREAD,
    SWING_PEAK_SEAM_ANGLE_DEG,
    PlayerProfile,
    _dimension_factors,
    _dimension_relevance,
    _isolating_candidate,
    _random_candidate,
    _shortlist_weights,
    delivery_difficulty_rating,
    dimension_delivery_rating,
    expected_success,
    next_delivery_after_with_reason,
    suggest_next_delivery,
    suggest_next_delivery_with_reason,
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


# ---- _shortlist_weights: the actual rule, tested directly against known candidates ----

def test_no_targeted_dimension_gives_the_plain_pre_existing_weighting():
    shortlist = [(0.0, 1000.0, "a"), (10.0, 1010.0, "b")]
    assert _shortlist_weights(shortlist, BALL, targeted=None) == [1.0 / 1.0, 1.0 / 11.0]


def test_a_candidate_leaning_more_on_the_targeted_dimension_gets_more_weight_at_equal_gap():
    pace_only = pure_pace_delivery()
    with_spin = leg_break_delivery()
    shortlist = [(5.0, 1000.0, pace_only), (5.0, 1000.0, with_spin)]  # identical gap, isolates the bias term
    weights = _shortlist_weights(shortlist, BALL, targeted="spin")
    assert weights[1] > weights[0]  # the spinning delivery, which has real spin relevance, outweighs the pace-only one


def test_the_weak_skill_bias_is_load_bearing_not_a_no_op():
    """Mutation-style: with WEAK_SKILL_BIAS effectively zeroed (relevance term
    dropped), two candidates at the same gap MUST tie -- confirming the bias
    term above is genuinely what breaks that tie, not something else."""
    shortlist = [(5.0, 1000.0, pure_pace_delivery()), (5.0, 1000.0, leg_break_delivery())]
    plain = [1.0 / (1.0 + gap) for gap, _, _ in shortlist]
    assert plain[0] == plain[1]  # equal gap -> equal weight without the bias term
    biased = _shortlist_weights(shortlist, BALL, targeted="spin")
    assert biased[0] != biased[1]  # the bias term is what breaks the tie


# ---- suggest_next_delivery_with_reason: the actual "why" ----

def test_a_fresh_profile_targets_no_dimension_and_says_so():
    p = PlayerProfile(name="P")
    suggestion = suggest_next_delivery_with_reason(p, BALL, rng=random.Random(1))
    assert suggestion.targeted_skill is None
    assert "no dimension is clearly weaker" in suggestion.reason.lower()


def test_a_profile_with_a_clear_weakness_targets_it_by_name():
    p = PlayerProfile(name="P")
    p.skill_ratings["seam"] = 1000.0 - SKILL_TIE_SPREAD - 1.0  # just over the tie threshold, unambiguous
    suggestion = suggest_next_delivery_with_reason(p, BALL, rng=random.Random(1))
    assert suggestion.targeted_skill == "seam"
    assert "seam" in suggestion.reason.lower()
    assert "targeting" in suggestion.reason.lower()


def test_a_tie_within_the_threshold_is_not_treated_as_a_real_weakness():
    p = PlayerProfile(name="P")
    p.skill_ratings["seam"] = 1000.0 - SKILL_TIE_SPREAD + 1.0  # just UNDER the threshold
    suggestion = suggest_next_delivery_with_reason(p, BALL, rng=random.Random(1))
    assert suggestion.targeted_skill is None


# ---- suggest_next_delivery (old, public) is still exactly the .delivery half of the new call ----

def test_suggest_next_delivery_matches_the_delivery_half_of_with_reason_for_the_same_seed():
    p1, p2 = PlayerProfile(name="P"), PlayerProfile(name="P")
    plain = suggest_next_delivery(p1, BALL, rng=random.Random(42))
    full = suggest_next_delivery_with_reason(p2, BALL, rng=random.Random(42))
    assert plain == full.delivery


# ---- next_delivery_after_with_reason: the record+reason convenience call ----

def test_next_delivery_after_with_reason_returns_both_a_record_and_a_suggestion():
    p = PlayerProfile(name="P")
    record, suggestion = next_delivery_after_with_reason(
        p, BALL, pure_pace_delivery(), "controlled", rng=random.Random(3),
    )
    assert record.outcome_score == 0.75
    assert suggestion.delivery is not None
    assert suggestion.reason


# ---- _isolating_candidate: vary ONE dimension, hold the others neutral ----

def _kmh(delivery):
    return delivery.speed_mps / 1000.0 * 3600.0


def test_isolating_pace_varies_speed_and_holds_seam_and_spin_neutral():
    rng = random.Random(7)
    speeds, seams = [], []
    for _ in range(100):
        d = _isolating_candidate(rng, (70.0, 150.0), "pace")
        speeds.append(_kmh(d))
        seams.append(d.seam_angle_deg)
        assert all(w == 0.0 for w in d.spin_rad_s)
    assert min(speeds) < 90.0 and max(speeds) > 130.0  # genuinely spans the full range, not a narrow slice
    assert all(NEUTRAL_SEAM_ANGLE_DEG[0] <= s <= NEUTRAL_SEAM_ANGLE_DEG[1] for s in seams)


def test_isolating_spin_varies_spin_and_holds_speed_and_seam_neutral():
    rng = random.Random(7)
    for _ in range(100):
        d = _isolating_candidate(rng, (70.0, 150.0), "spin")
        assert NEUTRAL_SPEED_RANGE_KMH[0] <= _kmh(d) <= NEUTRAL_SPEED_RANGE_KMH[1]
        assert NEUTRAL_SEAM_ANGLE_DEG[0] <= d.seam_angle_deg <= NEUTRAL_SEAM_ANGLE_DEG[1]
        assert any(w != 0.0 for w in d.spin_rad_s)  # never "no spin" when spin IS the target


def test_isolating_swing_uses_the_swing_peak_seam_angle_band():
    rng = random.Random(7)
    for _ in range(100):
        d = _isolating_candidate(rng, (70.0, 150.0), "swing")
        assert NEUTRAL_SPEED_RANGE_KMH[0] <= _kmh(d) <= NEUTRAL_SPEED_RANGE_KMH[1]
        assert SWING_PEAK_SEAM_ANGLE_DEG[0] <= d.seam_angle_deg <= SWING_PEAK_SEAM_ANGLE_DEG[1]
        assert all(w == 0.0 for w in d.spin_rad_s)


def test_isolating_seam_uses_the_cross_seam_peak_angle_band():
    rng = random.Random(7)
    for _ in range(100):
        d = _isolating_candidate(rng, (70.0, 150.0), "seam")
        assert SEAM_DRAG_PEAK_ANGLE_DEG[0] <= d.seam_angle_deg <= SEAM_DRAG_PEAK_ANGLE_DEG[1]
        assert all(w == 0.0 for w in d.spin_rad_s)


def _mean_relevance(candidates, dimension):
    total = 0.0
    for d in candidates:
        factors = _dimension_factors(d, BALL, 1.225, 1.8e-5)
        total += _dimension_relevance(factors)[dimension]
    return total / len(candidates)


def test_isolating_beats_broad_random_for_every_dimension_even_the_structurally_weak_ones():
    """The honest, bounded claim: isolation measurably helps ALL four dimensions over
    _random_candidate()'s "vary everything" approach -- even seam and spin, whose
    absolute relevance share stays small regardless (see _dimension_relevance()'s
    docstring: real physics caps how much either can ever contribute), isolation at
    least concentrates what little signal exists instead of diluting it further."""
    n = 400
    for dim in SKILL_DIMENSIONS:
        isolating = [_isolating_candidate(random.Random(1000 + i), (70.0, 150.0), dim) for i in range(n)]
        broad = [_random_candidate(random.Random(2000 + i), (70.0, 150.0)) for i in range(n)]
        isolating_mean = _mean_relevance(isolating, dim)
        broad_mean = _mean_relevance(broad, dim)
        assert isolating_mean > broad_mean, f"{dim}: isolating ({isolating_mean:.3f}) should beat broad-random ({broad_mean:.3f})"


# ---- integration: suggest_next_delivery_with_reason actually uses the isolating generator ----

def test_a_clear_pace_weakness_produces_deliveries_with_neutral_seam_angle():
    p = PlayerProfile(name="P")
    p.skill_ratings["pace"] = 1000.0 - SKILL_TIE_SPREAD - 1.0
    suggestion = suggest_next_delivery_with_reason(p, BALL, rng=random.Random(5))
    assert suggestion.targeted_skill == "pace"
    assert NEUTRAL_SEAM_ANGLE_DEG[0] <= suggestion.delivery.seam_angle_deg <= NEUTRAL_SEAM_ANGLE_DEG[1]

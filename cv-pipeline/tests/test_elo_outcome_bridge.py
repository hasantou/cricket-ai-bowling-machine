import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "trajectory-engine"))

from body_vectors import BodyVectorReport, JointVector, MAX_PLAUSIBLE_SPEED
from commentary_labels import DOT_BALL, FOUR, SIX, WICKET
from cricket_trajectory.adaptive import OUTCOME_SCORES
from elo_outcome_bridge import key_from_commentary, key_from_video_estimate
from outcome_from_video import estimate_net_outcome
from post_shot import LOST_TRACK, PostShotReport


def body(speed, up=0.0, across=0.0):
    v = JointVector(1.0, 0.0, speed, 0.0, "right", "good")
    return BodyVectorReport(
        peak_frame=10, peak_hand="right wrist", peak_hand_speed=speed, vectors_at_peak={"right wrist": v},
        hand_path_start=5, hand_path_end=15, hand_path_length=max(abs(up), abs(across), 0.1),
        hand_path_net=(across, up), hand_path_net_direction="right", hand_path_points=[],
        hip_shift=(0, 0), ankle_shift={"left ankle": (0, 0), "right ankle": (0, 0)},
        shoulder_line_change_deg=0.0, hip_line_change_deg=0.0, body_speed_at_peak=0.5,
        tracked_fraction=1.0, caveats=[],
    )


# ---- commentary side: real ground truth, always confident, and must produce real adaptive.py keys ----

def test_commentary_keys_are_all_real_entries_in_adaptive_engines_own_scale():
    """This bridge hard-codes copies of adaptive.OUTCOME_SCORES' own key strings so
    cv-pipeline doesn't need trajectory-engine installed just to compute a key — this
    test is what keeps that copy honest if adaptive.py's own keys ever change."""
    for commentary_outcome in (SIX, FOUR, WICKET, DOT_BALL):
        assert key_from_commentary(commentary_outcome) in OUTCOME_SCORES


def test_commentary_keys_map_to_the_expected_score_direction():
    assert key_from_commentary(SIX) == "six"
    assert key_from_commentary(FOUR) == "boundary"
    assert key_from_commentary(WICKET) == "missed"
    assert key_from_commentary(DOT_BALL) == "defended"
    assert OUTCOME_SCORES[key_from_commentary(SIX)] > OUTCOME_SCORES[key_from_commentary(FOUR)]
    assert OUTCOME_SCORES[key_from_commentary(FOUR)] > OUTCOME_SCORES[key_from_commentary(DOT_BALL)]
    assert OUTCOME_SCORES[key_from_commentary(DOT_BALL)] > OUTCOME_SCORES[key_from_commentary(WICKET)]


# ---- video side: honest confidence flags, and every returned key must be real ----

def test_no_swing_measured_gives_no_key_at_all():
    r = key_from_video_estimate(estimate_net_outcome(None))
    assert r.outcome_key is None
    assert r.is_confident is False


def test_implausible_speed_gives_no_key_at_all():
    r = key_from_video_estimate(estimate_net_outcome(body(MAX_PLAUSIBLE_SPEED + 5)))
    assert r.outcome_key is None
    assert r.is_confident is False


def test_a_wicket_type_read_is_never_confident_even_though_it_has_a_key():
    r = key_from_video_estimate(estimate_net_outcome(body(5.0, up=0.8)))
    assert r.outcome_key == "missed"
    assert r.is_confident is False
    assert "cannot confirm a wicket" in r.reason


def test_a_clean_measured_dot_is_confident():
    r = key_from_video_estimate(estimate_net_outcome(body(1.0)))
    assert r.outcome_key == "defended"
    assert r.is_confident is True


def test_a_clean_measured_four_type_swing_is_confident():
    r = key_from_video_estimate(estimate_net_outcome(body(9.0, up=0.1)))
    assert r.outcome_key == "boundary"
    assert r.is_confident is True


def test_a_clean_measured_six_type_swing_is_confident():
    r = key_from_video_estimate(estimate_net_outcome(body(13.0, up=1.0)))
    assert r.outcome_key == "six"
    assert r.is_confident is True


def test_a_dot_leaning_on_a_lost_track_secondary_signal_is_not_confident():
    post = PostShotReport(LOST_TRACK, None, frames_available=50, frames_tracked=40, gap_started_at_s=None)
    estimate = estimate_net_outcome(body(1.0), post)
    assert len(estimate.caveats) > 1   # sanity: this really did take the secondary-signal path
    r = key_from_video_estimate(estimate)
    assert r.outcome_key == "defended"
    assert r.is_confident is False
    assert "secondary signal" in r.reason


# ---- every non-None key returned by either function must plug straight into scorecard.score_outcome() ----

def test_every_possible_returned_key_is_accepted_by_the_real_scoring_table():
    from cricket_trajectory.scorecard import DEFAULT_SCORING, score_outcome
    for commentary_outcome in (SIX, FOUR, WICKET, DOT_BALL):
        score_outcome(key_from_commentary(commentary_outcome), DEFAULT_SCORING)  # must not raise
    for speed, up in [(1.0, 0.0), (9.0, 0.1), (13.0, 1.0), (5.0, 0.8)]:
        guess = key_from_video_estimate(estimate_net_outcome(body(speed, up=up)))
        score_outcome(guess.outcome_key, DEFAULT_SCORING)  # must not raise


# ---- mutation check: dropping the caveats-length check must break the secondary-signal test ----

def test_the_secondary_signal_check_is_load_bearing_not_a_tautology():
    """A confident and an unconfident case must reach the SAME outcome key (DOT /
    "defended") through DIFFERENT code paths in key_from_video_estimate, so the
    is_confident split provably comes from the caveats check and not just from which
    outcome string was produced."""
    confident = key_from_video_estimate(estimate_net_outcome(body(1.0)))
    post = PostShotReport(LOST_TRACK, None, frames_available=50, frames_tracked=40, gap_started_at_s=None)
    unconfident = key_from_video_estimate(estimate_net_outcome(body(1.0), post))
    assert confident.outcome_key == unconfident.outcome_key  # same label (DOT / "defended")
    assert confident.is_confident is True
    assert unconfident.is_confident is False

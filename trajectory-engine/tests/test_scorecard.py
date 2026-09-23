

def test_an_explicit_legality_call_overrides_the_single_trajectory_proxy():
    """The Laws-based, bounce-aware call (or a crease sensor's) can be passed in and is what gets scored."""
    from cricket_trajectory import BallProperties, Delivery, Environment, PlayerProfile, run_simulation
    from cricket_trajectory.scorecard import Scorecard
    ball, env = BallProperties(), Environment()
    d = Delivery(speed_mps=140 / 3.6, vertical_launch_deg=-3.0)
    result = run_simulation(ball, env, d)
    card = Scorecard(batter_name="P")
    wide = card.record_ball(PlayerProfile(name="P"), ball, d, result, legality="wide")      # forced wide
    assert wide.legality == "wide" and wide.runs == 1 and card.legal_balls == 0 and card.extras == 1
    fair = card.record_ball(PlayerProfile(name="P"), ball, d, result, outcome="defended", legality=None)
    assert fair.legality is None and card.legal_balls == 1


# ---- style_delivery_scoring: turning the style-library engine's OUTCOME_QUALITY into runs/wickets ----
def test_each_outcome_quality_scores_the_documented_runs_when_not_out():
    from cricket_trajectory.scorecard import style_delivery_scoring
    assert style_delivery_scoring("missed").runs == 0
    assert style_delivery_scoring("defended").runs == 0
    assert style_delivery_scoring("edged").runs == 0
    assert style_delivery_scoring("attacked").runs == 1
    assert style_delivery_scoring("middled").runs == 4
    for key in ("missed", "defended", "edged", "attacked", "middled"):
        assert style_delivery_scoring(key).is_wicket is False
        assert style_delivery_scoring(key).dismissal is None


def test_a_dismissal_zeroes_the_runs_regardless_of_outcome_quality():
    from cricket_trajectory.scorecard import style_delivery_scoring
    info = style_delivery_scoring("middled", dismissal="caught")
    assert info.runs == 0 and info.is_wicket is True and info.dismissal == "caught"
    # even a "middled" (normally 4-run) shot scores 0 if the batter was actually dismissed on it
    assert style_delivery_scoring("attacked", dismissal="run out").runs == 0


def test_every_dismissal_type_is_accepted_and_recorded_verbatim():
    from cricket_trajectory.scorecard import DISMISSAL_TYPES, style_delivery_scoring
    for dismissal in DISMISSAL_TYPES:
        if dismissal == "not out":
            continue
        info = style_delivery_scoring("missed", dismissal=dismissal)
        assert info.is_wicket is True and info.dismissal == dismissal


def test_an_unknown_outcome_quality_or_dismissal_raises_rather_than_guessing():
    import pytest
    from cricket_trajectory.scorecard import style_delivery_scoring
    with pytest.raises(KeyError):
        style_delivery_scoring("six")               # a net_outcome/adaptive key, not an OUTCOME_QUALITY key
    with pytest.raises(KeyError):
        style_delivery_scoring("missed", dismissal="hit the ball twice")


def test_style_delivery_scoring_returns_the_same_scoring_info_type_as_the_sensor_path():
    from cricket_trajectory.scorecard import ScoringInfo, score_outcome, style_delivery_scoring
    assert type(style_delivery_scoring("missed")) is type(score_outcome("missed")) is ScoringInfo

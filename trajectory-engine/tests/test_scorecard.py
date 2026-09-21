

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

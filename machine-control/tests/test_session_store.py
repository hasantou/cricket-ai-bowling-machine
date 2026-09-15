import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "trajectory-engine"))

from cricket_trajectory import PlayerProfile, Scorecard, BallProperties, Environment, Delivery, run_simulation
from machine_control.session_store import save_profile, load_profile, save_scorecard, load_scorecard


def test_profile_round_trips_with_history(tmp_path):
    profile = PlayerProfile(name="Test Player", rating=1000.0)
    profile.record_outcome(
        Delivery(speed_mps=30.0, seam_angle_deg=15.0, label="test ball"),
        BallProperties(), "defended",
    )
    path = tmp_path / "profile.json"
    save_profile(profile, path)
    loaded = load_profile(path)

    assert loaded.name == profile.name
    assert loaded.rating == profile.rating
    assert len(loaded.history) == 1
    assert loaded.history[0].label == "test ball"
    assert loaded.history[0].rating_after == profile.history[0].rating_after


def test_profile_round_trips_with_empty_history(tmp_path):
    profile = PlayerProfile(name="Fresh Player", rating=1000.0)
    path = tmp_path / "fresh.json"
    save_profile(profile, path)
    loaded = load_profile(path)
    assert loaded.name == "Fresh Player"
    assert loaded.history == []


def test_scorecard_round_trips_with_balls(tmp_path):
    ball = BallProperties()
    env = Environment()
    profile = PlayerProfile(name="Test Player", rating=1000.0)
    card = Scorecard(batter_name=profile.name)
    delivery = Delivery(speed_mps=28.0, seam_angle_deg=10.0, label="a ball")
    result = run_simulation(ball, env, delivery)
    card.record_ball(profile, ball, delivery, result, outcome="boundary")

    path = tmp_path / "card.json"
    save_scorecard(card, path)
    loaded = load_scorecard(path)

    assert loaded.batter_name == card.batter_name
    assert loaded.total_runs == card.total_runs == 4
    assert len(loaded.balls) == 1
    assert loaded.balls[0].delivery_label == "a ball"
    assert loaded.balls[0].runs == 4

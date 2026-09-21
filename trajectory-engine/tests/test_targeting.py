import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from cricket_trajectory import BallProperties, Delivery, Environment, PlayerProfile, run_simulation
from cricket_trajectory.laws import WICKET_X_M, assess_delivery
from cricket_trajectory.targeting import (
    DEFAULT_LENGTH_MIX, LENGTH_BANDS, LINE_CORRIDOR_M, aim_delivery, sample_target, suggest_aimed_delivery,
)

BALL, ENV = BallProperties(), Environment()


@pytest.mark.parametrize("kmh,from_stumps,y", [(140, 6.2, 0.10), (110, 1.0, -0.10), (130, 10.0, 0.30)])
def test_the_solver_lands_the_ball_where_it_was_asked(kmh, from_stumps, y):
    target_x = WICKET_X_M - from_stumps
    d = aim_delivery(BALL, ENV, Delivery(speed_mps=kmh / 3.6, seam_angle_deg=25), target_x, y)
    assert d is not None
    x, ly = run_simulation(BALL, ENV, d).landing_point_m
    assert abs(x - target_x) < 0.25 and abs(ly - y) < 0.08
    assert "aimed" in d.label


def test_a_length_the_pace_cannot_reach_is_reported_not_faked():
    """Nothing this slow reaches the far end at any angle in range: the solver must say so."""
    assert aim_delivery(BALL, ENV, Delivery(speed_mps=8.0), WICKET_X_M - 1.0, 0.0) is None


def test_targets_follow_the_length_mix_and_stay_inside_the_line_corridor():
    rng = random.Random(3)
    picks = [sample_target(rng) for _ in range(600)]
    counts = {name: sum(1 for b, _, _ in picks if b == name) for name in DEFAULT_LENGTH_MIX}
    for name, share in DEFAULT_LENGTH_MIX.items():
        assert abs(counts[name] / 600 - share) < 0.06
    for band, px, py in picks:
        lo, hi = LENGTH_BANDS[band]
        assert lo <= WICKET_X_M - px <= hi and LINE_CORRIDOR_M[0] <= py <= LINE_CORRIDOR_M[1]


def test_an_aimed_suggestion_is_legal_and_carries_its_length_band():
    d = suggest_aimed_delivery(PlayerProfile(name="p", rating=1100), BALL, ENV, rng=random.Random(5))
    assert "unaimed" not in d.label and "[" in d.label
    assert not assess_delivery(BALL, ENV, d).is_wide


def test_the_same_seed_gives_the_same_delivery():
    a = suggest_aimed_delivery(PlayerProfile(name="p", rating=1000), BALL, ENV, rng=random.Random(9))
    b = suggest_aimed_delivery(PlayerProfile(name="p", rating=1000), BALL, ENV, rng=random.Random(9))
    assert (a.speed_mps, a.vertical_launch_deg, a.horizontal_launch_deg) == (b.speed_mps, b.vertical_launch_deg, b.horizontal_launch_deg)


def test_across_many_suggestions_no_wides_and_more_than_one_length():
    rng = random.Random(21)
    lengths, wides = set(), 0
    for _ in range(12):
        d = suggest_aimed_delivery(PlayerProfile(name="p", rating=1050), BALL, ENV, rng=rng)
        a = assess_delivery(BALL, ENV, d)
        wides += a.is_wide
        lengths.add(d.label.rsplit("[", 1)[-1])
    assert wides == 0 and len(lengths) >= 3

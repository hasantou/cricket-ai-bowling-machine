import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator import VirtualBatter, simulate_session, generate_dataset, MASTERY_SKILL_THRESHOLD
from styles import STYLE_LIBRARY
import random


def test_skill_increases_monotonically_within_a_session():
    batter = VirtualBatter.random_batter(random.Random(1))
    style_key = STYLE_LIBRARY[0].key
    records = simulate_session(batter, style_key, n_balls=10)
    skills = [r["true_skill_before"] for r in records]
    assert all(later >= earlier for earlier, later in zip(skills, skills[1:])), (
        "skill should never decrease within a session — batters don't un-learn mid-session"
    )


def test_fast_learner_reaches_mastery_faster_than_slow_learner():
    style_key = STYLE_LIBRARY[0].key

    fast = VirtualBatter(style_skill={style_key: 0.2}, style_learning_rate={style_key: 0.30},
                          rng=random.Random(2))
    slow = VirtualBatter(style_skill={style_key: 0.2}, style_learning_rate={style_key: 0.08},
                          rng=random.Random(2))

    fast_records = simulate_session(fast, style_key, n_balls=20)
    slow_records = simulate_session(slow, style_key, n_balls=20)

    fast_master_ball = next((r["ball_index"] for r in fast_records if r["true_mastered_after"]), None)
    slow_master_ball = next((r["ball_index"] for r in slow_records if r["true_mastered_after"]), None)

    assert fast_master_ball is not None, "fast learner should master the style within 20 balls"
    if slow_master_ball is not None:
        assert fast_master_ball < slow_master_ball


def test_dataset_has_a_genuine_mix_of_labels():
    # Regression guard for the bug this caught earlier: a learning curve
    # too slow (or too fast) collapses the dataset to a single class,
    # which silently makes the classifier untrainable (sklearn warns but
    # doesn't error). Both extremes are checked.
    records = generate_dataset(n_batters=40, balls_per_style=16, seed=7)
    positive_rate = sum(1 for r in records if r["true_mastered_after"]) / len(records)
    assert 0.15 < positive_rate < 0.85, (
        f"positive label rate {positive_rate:.2f} is too skewed for a trainable dataset"
    )


def test_true_mastery_matches_threshold():
    batter = VirtualBatter(style_skill={"x": 0.80}, style_learning_rate={"x": 0.0}, rng=random.Random(3))
    assert batter.true_mastery("x") is True
    batter2 = VirtualBatter(style_skill={"x": 0.50}, style_learning_rate={"x": 0.0}, rng=random.Random(3))
    assert batter2.true_mastery("x") is False
    assert MASTERY_SKILL_THRESHOLD == 0.75  # documents the constant the two assertions above rely on

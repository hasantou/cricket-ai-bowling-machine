"""
A virtual-batter simulator — the honest way to train and evaluate a
neural mastery model before any real session data exists.

This is NOT a substitute for real players. It exists to answer one
question responsibly: "if a neural network replaced the rule-based
mastery threshold, would it behave sensibly?" — measured against a
simulated ground truth we control, before ever pointing it at a real
person. See adaptation-engine/README.md and
docs/product/AI_Adaptive_Bowling_Machine_Program_Plan.docx (Section 4.5)
for the same caveat in the planning docs: simulated data is a prior,
not a replacement for data fed in from real sessions later.

Model: each virtual batter has a hidden, continuous "true skill" per
style in [0, 1] that rises with repeated exposure (a diminishing-returns
learning curve), and never directly visible to the scorer under test —
only noisy, skill-correlated delivery outcomes are. That hidden skill is
the ground truth used to check whether a scorer's "mastered" call is
actually well-timed.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from styles import STYLE_LIBRARY
from scoring import OUTCOME_QUALITY, MasteryScorer

MASTERY_SKILL_THRESHOLD = 0.75  # ground-truth "actually mastered" cutoff

# Outcome distributions at skill 0.0 and 1.0 — linearly interpolated by
# skill in between, then sampled. Keys must match OUTCOME_QUALITY.
DIST_AT_ZERO = {"missed": 0.45, "edged": 0.30, "defended": 0.20, "attacked": 0.03, "middled": 0.02}
DIST_AT_ONE = {"missed": 0.02, "edged": 0.05, "defended": 0.13, "attacked": 0.30, "middled": 0.50}

assert set(DIST_AT_ZERO) == set(OUTCOME_QUALITY)
assert set(DIST_AT_ONE) == set(OUTCOME_QUALITY)


@dataclass
class VirtualBatter:
    """One simulated player. `initial_skill` and `learning_rate` are drawn
    per style so different virtual batters have different natural
    strengths/weaknesses and different learning speeds — the same way
    real batters do."""

    style_skill: Dict[str, float] = field(default_factory=dict)
    style_learning_rate: Dict[str, float] = field(default_factory=dict)
    rng: random.Random = field(default_factory=random.Random)

    @classmethod
    def random_batter(cls, rng: random.Random) -> "VirtualBatter":
        skill = {s.key: rng.uniform(0.05, 0.35) for s in STYLE_LIBRARY}
        rate = {s.key: rng.uniform(0.08, 0.30) for s in STYLE_LIBRARY}
        return cls(style_skill=skill, style_learning_rate=rate, rng=rng)

    def face_delivery(self, style_key: str) -> Tuple[str, bool, bool, float]:
        """Simulate facing one delivery of `style_key`. Returns
        (outcome, on_time, footwork_correct, true_skill_before_this_ball).
        Skill updates (the batter learns a little) *after* the outcome is
        drawn, matching reality: you learn from the ball you just faced."""
        skill = self.style_skill.get(style_key, 0.15)

        dist = {
            k: DIST_AT_ZERO[k] + (DIST_AT_ONE[k] - DIST_AT_ZERO[k]) * skill
            for k in DIST_AT_ZERO
        }
        total = sum(dist.values())
        dist = {k: v / total for k, v in dist.items()}

        outcome = self.rng.choices(list(dist.keys()), weights=list(dist.values()), k=1)[0]
        on_time = self.rng.random() < (0.30 + 0.65 * skill)
        footwork_correct = self.rng.random() < (0.35 + 0.60 * skill)

        # Learn from this ball — diminishing returns as skill -> 1. With
        # lr in [0.08, 0.30], a fast learner crosses the mastery threshold
        # in roughly 5-6 balls and a slow learner in roughly 12-15 —
        # deliberately tuned so simulated sessions produce a genuine mix
        # of "mastered" and "not yet" labels instead of everyone plateauing
        # below threshold (which silently produces a single-class,
        # untrainable dataset — caught by test_simulator.py).
        lr = self.style_learning_rate.get(style_key, 0.15)
        self.style_skill[style_key] = skill + lr * (1 - skill)

        return outcome, on_time, footwork_correct, skill

    def true_mastery(self, style_key: str) -> bool:
        return self.style_skill.get(style_key, 0.0) >= MASTERY_SKILL_THRESHOLD


def simulate_session(batter: VirtualBatter, style_key: str, n_balls: int) -> List[dict]:
    """Face `n_balls` deliveries of one style against one virtual batter.

    Each record carries two kinds of information, kept clearly separate:
    - OBSERVED features (outcome history, rolling EMA score from the same
      rule-based scorer the live app uses, running rates) — this is all a
      real deployed system would ever see.
    - The HIDDEN ground truth (true_skill / true_mastered) — only the
      simulator has access to this, and it exists purely so we can check,
      after the fact, whether a scorer's "mastered" call was well-timed.
      No trained model ever sees this as an input feature.
    """
    records = []
    scorer = MasteryScorer()
    outcome_counts = {k: 0 for k in OUTCOME_QUALITY}
    on_time_count = 0
    footwork_count = 0

    for i in range(n_balls):
        outcome, on_time, footwork, skill_before = batter.face_delivery(style_key)
        rule_record = scorer.record_delivery(style_key, outcome, on_time, footwork)

        outcome_counts[outcome] += 1
        on_time_count += int(on_time)
        footwork_count += int(footwork)
        seen = i + 1

        records.append({
            "style_key": style_key,
            "ball_index": i,
            "outcome": outcome,
            "on_time": on_time,
            "footwork_correct": footwork,
            # observed / engineered features (what a real system has):
            "ema_score": rule_record.score,
            "deliveries_seen": seen,
            "on_time_rate": on_time_count / seen,
            "footwork_rate": footwork_count / seen,
            "good_outcome_rate": (outcome_counts["middled"] + outcome_counts["attacked"]) / seen,
            "miss_rate": outcome_counts["missed"] / seen,
            # hidden ground truth (label only — never a model input):
            "true_skill_before": skill_before,
            "true_mastered_after": batter.true_mastery(style_key),
        })
    return records


def generate_dataset(n_batters: int = 250, balls_per_style: int = 16, seed: int = 42) -> List[dict]:
    """Generate a full simulated dataset: many virtual batters, each
    facing every style in the library for `balls_per_style` deliveries.
    This is what neural_scorer.py trains and evaluates on."""
    rng = random.Random(seed)
    all_records = []
    for _ in range(n_batters):
        batter = VirtualBatter.random_batter(random.Random(rng.randrange(1_000_000)))
        for style in STYLE_LIBRARY:
            all_records.extend(simulate_session(batter, style.key, balls_per_style))
    return all_records

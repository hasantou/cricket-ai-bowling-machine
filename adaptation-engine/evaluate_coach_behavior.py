"""
Does this behave like an attentive coach — raising difficulty right when
a batter is ready, not before and not long after? This script is the
honest answer to that question, run against simulated batters (see
simulator.py's docstring for why simulation, not real players, is the
right first check).

What "like a coach" means here, made measurable:
  - Responsiveness: how many balls after a virtual batter *actually*
    crosses the mastery threshold does the scorer notice and call it?
    An attentive coach doesn't need 15 more balls of proof once a player
    has clearly got it — but also shouldn't jump the gun on 2 good balls.
  - False triggers: how often does the scorer call "mastered" on a batter
    who, by the simulator's ground truth, hasn't actually crossed the
    threshold yet? A coach who keeps moving on before a player is ready
    isn't training them, just cycling deliveries.

Compares the original rule-based scorer (scoring.py) against the trained
neural scorer (neural_scorer.py) on withheld simulated batters, so the
comparison means something.

Run: python evaluate_coach_behavior.py
"""

import random
import statistics

from simulator import VirtualBatter, MASTERY_SKILL_THRESHOLD
from styles import STYLE_LIBRARY
from scoring import MasteryScorer, MIN_SAMPLE
from neural_scorer import NeuralMasteryScorer, load_model


def _true_mastery_ball(batter, style_key, n_balls, rng_face):
    """Replay isn't used — instead we track, ball by ball, the first index
    at which ground-truth skill crosses threshold, alongside feeding the
    same deliveries to a scorer under test. Returns the list of
    (outcome, on_time, footwork, true_mastered_after) tuples so both
    scorers can be evaluated on the *identical* delivery sequence."""
    sequence = []
    for _ in range(n_balls):
        outcome, on_time, footwork, _ = batter.face_delivery(style_key)
        sequence.append((outcome, on_time, footwork, batter.true_mastery(style_key)))
    return sequence


def evaluate_scorer(scorer_factory, n_batters=80, balls_per_style=16, seed=99):
    """scorer_factory() must return a fresh scorer with .record_delivery()
    and produce an object whose .mastered attribute exists."""
    rng = random.Random(seed)
    delays = []            # balls between true mastery and the scorer noticing, when it does notice
    false_trigger_count = 0
    never_triggered_count = 0
    total_style_sessions = 0

    for _ in range(n_batters):
        batter = VirtualBatter.random_batter(random.Random(rng.randrange(1_000_000)))
        for style in STYLE_LIBRARY:
            total_style_sessions += 1
            scorer = scorer_factory()
            sequence = _true_mastery_ball(batter, style.key, balls_per_style, rng)

            true_master_index = next((i for i, rec in enumerate(sequence) if rec[3]), None)
            triggered_index = None

            for i, (outcome, on_time, footwork, _true_mastered) in enumerate(sequence):
                record = scorer.record_delivery(style.key, outcome, on_time, footwork)
                if record.mastered and triggered_index is None:
                    triggered_index = i

            if triggered_index is None:
                if true_master_index is not None:
                    never_triggered_count += 1  # missed a real mastery event entirely
                continue

            if true_master_index is None or triggered_index < true_master_index:
                false_trigger_count += 1
            else:
                delays.append(triggered_index - true_master_index)

    return {
        "total_style_sessions": total_style_sessions,
        "mean_delay_balls": statistics.mean(delays) if delays else None,
        "median_delay_balls": statistics.median(delays) if delays else None,
        "false_trigger_rate": false_trigger_count / total_style_sessions,
        "missed_mastery_rate": never_triggered_count / total_style_sessions,
        "n_correct_triggers": len(delays),
    }


def print_report(name, metrics):
    print(f"\n{name}")
    print("-" * len(name))
    print(f"  style-sessions evaluated:      {metrics['total_style_sessions']}")
    print(f"  correctly-timed triggers:      {metrics['n_correct_triggers']}")
    if metrics["mean_delay_balls"] is not None:
        print(f"  avg delay after true mastery:  {metrics['mean_delay_balls']:.2f} balls "
              f"(median {metrics['median_delay_balls']:.1f})")
    else:
        print("  avg delay after true mastery:  n/a (no correctly-timed triggers)")
    print(f"  false-trigger rate:            {metrics['false_trigger_rate']*100:.1f}% "
          "(called 'mastered' before the batter truly had it)")
    print(f"  missed-mastery rate:           {metrics['missed_mastery_rate']*100:.1f}% "
          "(batter truly mastered it, scorer never noticed within the session)")


if __name__ == "__main__":
    print(f"Evaluating on withheld simulated batters (mastery threshold = {MASTERY_SKILL_THRESHOLD}, "
          f"rule-based min sample = {MIN_SAMPLE})...")

    rule_metrics = evaluate_scorer(lambda: MasteryScorer(), seed=99)
    neural_model = load_model()
    neural_metrics = evaluate_scorer(lambda: NeuralMasteryScorer(model=neural_model), seed=99)

    print_report("Rule-based scorer (scoring.py)", rule_metrics)
    print_report("Neural scorer (neural_scorer.py)", neural_metrics)

    print("\nRead this as: lower delay = notices mastery sooner, like an attentive coach.")
    print("Lower false-trigger rate = doesn't move a batter on before they're actually ready.")

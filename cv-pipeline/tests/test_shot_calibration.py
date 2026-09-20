import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shot_calibration import (
    ADOPT_MARGIN, MIN_EXAMPLES, TrialRecord, append_records, evaluate_rules, fit_rules, load_records,
    load_rules, prediction_for, records_from_lines, save_rules, target_for,
)
from shot_from_video import (
    BEHIND_BATTER, DEFAULT_RULES, RIGHT_HANDED, SIDE_ON, ShotRules, classify_features,
)

# A "true" way the world works that differs from the shipped rules of thumb.
TRUE_RULES = ShotRules(swing_min_speed=5.5, defence_max_path=1.8, straight_max_across=0.7,
                       wide_min_across=1.7, horizontal_ratio=0.8, high_finish_up=1.0)


def synthetic(n, rules=TRUE_RULES, flip=0.0, seed=1, camera=BEHIND_BATTER, gates=True):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        across, up = rng.uniform(-2.6, 2.6), rng.uniform(-0.4, 2.0)
        length, speed = rng.uniform(0.3, 5.0), rng.uniform(1.0, 12.0)
        label = prediction_for(classify_features(across, up, length, speed, camera, RIGHT_HANDED, rules))
        if rng.random() < flip:
            label = rng.choice(["cover drive", "pull", "leave", "cut", "on drive"])
        out.append(TrialRecord(f"s{i}", label, camera, RIGHT_HANDED, across, up, length, speed, gates))
    return out


def test_it_refuses_to_fit_on_too_few_examples_and_leaves_the_rules_alone():
    r = fit_rules(synthetic(MIN_EXAMPLES - 1))
    assert not r.adopt and r.rules == DEFAULT_RULES and r.heldout_accuracy_fitted is None
    assert "need at least" in r.reason


def test_records_that_failed_the_trust_gates_are_not_learned_from():
    r = fit_rules(synthetic(200, gates=False))
    assert r.n_trusted == 0 and not r.adopt


def test_when_the_world_differs_from_the_rules_of_thumb_the_fit_recovers_and_beats_them_held_out():
    r = fit_rules(synthetic(240))
    assert r.adopt
    assert r.heldout_accuracy_fitted >= r.heldout_accuracy_current + ADOPT_MARGIN
    assert r.heldout_accuracy_fitted > 0.85
    # the thresholds moved toward the truth, not just anywhere
    assert abs(r.rules.wide_min_across - TRUE_RULES.wide_min_across) < abs(DEFAULT_RULES.wide_min_across - TRUE_RULES.wide_min_across)
    assert abs(r.rules.swing_min_speed - TRUE_RULES.swing_min_speed) < abs(DEFAULT_RULES.swing_min_speed - TRUE_RULES.swing_min_speed)


def test_it_still_improves_with_noisy_labels_and_does_not_claim_perfection():
    r = fit_rules(synthetic(300, flip=0.15))
    assert r.adopt and r.heldout_accuracy_fitted < 0.95


def test_when_the_rules_of_thumb_are_already_right_it_changes_nothing():
    r = fit_rules(synthetic(200, rules=DEFAULT_RULES))
    assert not r.adopt and r.rules == DEFAULT_RULES


def test_labels_carry_no_information_so_it_does_not_adopt_a_fit_that_memorised_noise():
    """Random labels: any fit looks good on its own training data but is no better held out."""
    rng = random.Random(5)
    recs = synthetic(120)
    for r in recs:
        r.label = rng.choice(["cover drive", "pull", "cut", "on drive", "leave", "hook"])
    assert not fit_rules(recs).adopt


def test_from_side_on_a_specific_label_is_compared_at_family_level():
    assert target_for("Cover drive", SIDE_ON) == "drive-type swing"
    assert target_for("Pull", SIDE_ON) == "horizontal-bat swing"
    assert target_for("Cover drive", BEHIND_BATTER) == "cover drive"
    assert target_for("Forward defence", BEHIND_BATTER) == "defensive push"


def test_evaluate_reports_a_confusion_and_how_few_it_rests_on():
    recs = synthetic(10, rules=DEFAULT_RULES)
    recs[0].label = "hook" if recs[0].label != "hook" else "cut"
    ev = evaluate_rules(DEFAULT_RULES, recs)
    assert ev["trusted_records"] == 10 and ev["accuracy"] == 0.9 and "only 10" in ev["note"]
    assert len(ev["disagreements"]) == 1 and ev["confusion"]


def test_records_and_rules_round_trip_through_disk_with_their_evidence():
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "trials.jsonl")
        recs = synthetic(5)
        append_records(log, recs[:2])
        append_records(log, recs[2:])
        assert [r.id for r in load_records(log)] == [r.id for r in recs]
        assert records_from_lines(open(log, encoding="utf-8").read())[0].label == recs[0].label
        result = fit_rules(synthetic(240))
        path = os.path.join(d, "rules.json")
        save_rules(path, result)
        rules, meta = load_rules(path)
        assert rules == result.rules and meta["n_trusted"] == 240 and meta["heldout_accuracy_fitted"] is not None

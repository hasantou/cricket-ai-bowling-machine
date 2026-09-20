"""
Turn real-world trials into a stronger shot algorithm: record what the shot
really was, score the algorithm against it, and tune its thresholds.

A "trial record" is one delivery: the four measured numbers the classifier
sees (hand path across / up / length, peak hand speed), the camera position
and batting hand, whether the tracking passed the trust gates, and the label
a person gave the delivery. Records are appended to a JSON-lines file, so a
session of real deliveries becomes data.

What the calibrator will and will not do — the guardrails matter more than
the fitting:

  * It only learns from records that PASSED the trust gates. A delivery the
    algorithm would refuse can't teach it anything about naming.
  * It will not fit fewer than MIN_EXAMPLES trusted records (default 30).
    With fewer, thresholds would just memorise the examples.
  * Its accuracy figure is HELD-OUT: k-fold cross-validation, each fold fitted
    on the others and scored on the one it never saw, next to the same held-out
    score for the current rules. A figure measured on the training data is not
    reported as the result.
  * It recommends the fitted rules only if they beat the current rules on the
    held-out folds by ADOPT_MARGIN. Otherwise `adopt` is False and nothing
    changes. It never overwrites the defaults itself; saving/loading is explicit.
  * Fitting can only move the thresholds, not fix the underlying limit: the
    algorithm reads the hands, not the ball. If real trials show the shapes of
    different shots overlap, the honest result is low accuracy at any thresholds.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from shot_from_video import (
    DEFAULT_RULES, DEFENCE_FAMILY, DRIVE_FAMILY, HORIZONTAL_FAMILY, SIDE_ON, ShotRules, VideoShotEstimate,
    classify_features,
)

MIN_EXAMPLES = 30
ADOPT_MARGIN = 0.03           # fitted rules must beat the current rules by this much, held-out
FOLDS = 5
PASSES = 3

DEFENCE_LABELS = {"forward defence", "back-foot defence", "dead-bat", "defensive push", "block"}
DRIVE_LABELS = {"straight drive", "off drive", "cover drive", "on drive", "square drive", "flick", "leg glance", "lofted drive"}
HORIZONTAL_LABELS = {"cut", "square cut", "late cut", "pull", "hook", "back-foot punch", "upper cut", "slog", "sweep"}

# The grid each threshold is searched over.
GRIDS: Dict[str, List[float]] = {
    "swing_min_speed": [x / 2 for x in range(4, 17)],        # 2.0 .. 8.0
    "defence_max_path": [x / 10 for x in range(4, 26)],      # 0.4 .. 2.5
    "straight_max_across": [x / 20 for x in range(2, 21)],   # 0.1 .. 1.0
    "wide_min_across": [x / 10 for x in range(5, 26)],       # 0.5 .. 2.5
    "horizontal_ratio": [x / 10 for x in range(6, 26)],      # 0.6 .. 2.5
    "high_finish_up": [x / 10 for x in range(1, 16)],        # 0.1 .. 1.5
}


@dataclass
class TrialRecord:
    id: str
    label: str                     # what the shot really was, as a person judged it
    camera: str
    hand: str
    across: float
    up: float
    length: float
    speed: float
    gates_passed: bool             # would the live algorithm have trusted these numbers?
    clip: str = ""
    delivery_index: int = 0
    labeller: str = ""
    notes: str = ""


def target_for(label: str, camera: str) -> str:
    """The thing the algorithm's output is compared with. From side-on only the
    shot FAMILY is readable, so a specific label is reduced to its family."""
    key = label.strip().lower()
    if key in ("leave", "left alone"):
        return "leave"
    if key in DEFENCE_LABELS:
        return DEFENCE_FAMILY
    if camera == SIDE_ON:
        if key in DRIVE_LABELS:
            return DRIVE_FAMILY
        if key in HORIZONTAL_LABELS:
            return HORIZONTAL_FAMILY
    return key


def prediction_for(estimate: VideoShotEstimate) -> str:
    if estimate.shot:
        return estimate.shot.lower()
    return estimate.family or "cannot tell"


def _score(rules: ShotRules, records: List[TrialRecord]) -> float:
    if not records:
        return 0.0
    hits = 0
    for r in records:
        est = classify_features(r.across, r.up, r.length, r.speed, r.camera, r.hand, rules)
        hits += prediction_for(est) == target_for(r.label, r.camera)
    return hits / len(records)


def evaluate_rules(rules: ShotRules, records: Iterable[TrialRecord]) -> Dict:
    """Accuracy of `rules` on the trusted records, with the disagreements. This is
    a plain (not held-out) score of rules that were not fitted on these records —
    fine for judging the shipped defaults on new real trials."""
    records = list(records)
    trusted = [r for r in records if r.gates_passed]
    misses, confusion = [], {}
    correct = 0
    for r in trusted:
        predicted = prediction_for(classify_features(r.across, r.up, r.length, r.speed, r.camera, r.hand, rules))
        truth = target_for(r.label, r.camera)
        confusion[(truth, predicted)] = confusion.get((truth, predicted), 0) + 1
        if predicted == truth:
            correct += 1
        else:
            misses.append({"id": r.id, "truth": truth, "predicted": predicted})
    n = len(trusted)
    return {
        "trusted_records": n,
        "correct": correct,
        "accuracy": (correct / n) if n else None,
        "disagreements": misses,
        "confusion": {f"{t} -> {p}": c for (t, p), c in sorted(confusion.items())},
        "excluded_by_gates": sum(1 for r in records if not r.gates_passed),
        "note": f"Rests on only {n} trusted delivery(ies)." if n < MIN_EXAMPLES else "",
    }


def _fit_once(records: List[TrialRecord], start: ShotRules) -> ShotRules:
    """Coordinate descent: for each threshold in turn, pick the grid value that
    scores best with the others held; ties go to the value closest to the
    starting one, so it doesn't wander where the data is indifferent."""
    current = start.as_dict()
    for _ in range(PASSES):
        changed = False
        for name, grid in GRIDS.items():
            best_value, best_key = current[name], None
            for value in grid:
                trial = ShotRules.from_dict({**current, name: value})
                key = (-_score(trial, records), abs(value - start.as_dict()[name]))
                if best_key is None or key < best_key:
                    best_key, best_value = key, value
            if best_value != current[name]:
                current[name] = best_value
                changed = True
        if not changed:
            break
    return ShotRules.from_dict(current)


@dataclass
class CalibrationResult:
    rules: ShotRules
    n_trusted: int
    heldout_accuracy_fitted: Optional[float]
    heldout_accuracy_current: Optional[float]
    adopt: bool
    reason: str
    fitted_on_all: Optional[ShotRules] = None


def fit_rules(
    records: List[TrialRecord], current: ShotRules = DEFAULT_RULES, min_examples: int = MIN_EXAMPLES,
    folds: int = FOLDS, seed: int = 0,
) -> CalibrationResult:
    """Fit thresholds from trusted labelled records, judged on held-out folds.
    Returns the rules to use (the fitted ones only if `adopt`, otherwise the
    current ones unchanged)."""
    trusted = [r for r in records if r.gates_passed]
    n = len(trusted)
    if n < min_examples:
        return CalibrationResult(
            current, n, None, None, False,
            f"Only {n} trusted labelled deliveries; need at least {min_examples} before fitting means anything. "
            "Rules left unchanged.",
        )
    order = list(range(n))
    random.Random(seed).shuffle(order)
    folds = min(folds, n)
    fitted_scores, current_scores = [], []
    for k in range(folds):
        held = [trusted[i] for j, i in enumerate(order) if j % folds == k]
        train = [trusted[i] for j, i in enumerate(order) if j % folds != k]
        fitted_scores.append((_score(_fit_once(train, current), held), len(held)))
        current_scores.append((_score(current, held), len(held)))
    total = sum(w for _, w in fitted_scores)
    fitted_acc = sum(a * w for a, w in fitted_scores) / total
    current_acc = sum(a * w for a, w in current_scores) / total
    final = _fit_once(trusted, current)
    adopt = fitted_acc >= current_acc + ADOPT_MARGIN
    if adopt:
        reason = (f"Fitted rules scored {fitted_acc:.0%} on held-out deliveries vs {current_acc:.0%} for the "
                  f"current rules (n={n}, {folds}-fold).")
    else:
        reason = (f"Fitted rules scored {fitted_acc:.0%} vs {current_acc:.0%} for the current rules on held-out "
                  f"deliveries (n={n}) — not better by the required {ADOPT_MARGIN:.0%}, so nothing changes.")
    return CalibrationResult(final if adopt else current, n, fitted_acc, current_acc, adopt, reason, fitted_on_all=final)


# ---- trial log on disk ---------------------------------------------------------
def append_records(path: str, records: Iterable[TrialRecord]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")


def load_records(path: str) -> List[TrialRecord]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [TrialRecord(**json.loads(line)) for line in f if line.strip()]


def records_from_lines(text: str) -> List[TrialRecord]:
    return [TrialRecord(**json.loads(line)) for line in text.splitlines() if line.strip()]


def save_rules(path: str, result: CalibrationResult) -> None:
    """Writes the rules AND the evidence they were judged on; a rules file with
    no record of how it was validated is exactly what this module tries to avoid."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "rules": result.rules.as_dict(), "adopted": result.adopt, "n_trusted": result.n_trusted,
            "heldout_accuracy_fitted": result.heldout_accuracy_fitted,
            "heldout_accuracy_current": result.heldout_accuracy_current, "reason": result.reason,
        }, f, indent=2)


def load_rules(path: str) -> Tuple[ShotRules, Dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return ShotRules.from_dict(data["rules"]), data

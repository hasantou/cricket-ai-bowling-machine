"""
Neural mastery scorer — a real, trained multi-layer perceptron (a neural
network — sklearn's MLPClassifier, chosen deliberately over a deep
learning framework: the input is a handful of engineered features, not
raw video, so a small MLP is the right-sized model, not an underpowered
one) that predicts whether a batter has mastered a bowling style from the
same observed features a live session produces.

Honesty check, up front: this is trained entirely on simulator.py's
virtual batters (see that file's docstring). It has never seen a real
player. Training on real session data once it exists (see the program
plan, Section 4.5, and the MVP plan) means re-running train() against a
dataset built the same way — same feature columns — from real logged
sessions instead of generate_dataset(). Swapping the data source is the
whole migration; the feature schema and the NeuralMasteryScorer interface
don't need to change.

Interface deliberately mirrors scoring.MasteryScorer so app.py can switch
between "rule-based" and "neural" with no other code changes.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier

from simulator import generate_dataset
from scoring import MasteryScorer, MIN_SAMPLE

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "neural_scorer.joblib")

FEATURE_COLUMNS = [
    "ema_score", "deliveries_seen", "on_time_rate",
    "footwork_rate", "good_outcome_rate", "miss_rate",
]


def _to_feature_vector(rec: dict) -> List[float]:
    return [
        rec["ema_score"],
        min(rec["deliveries_seen"], 15) / 15.0,  # cap + normalise
        rec["on_time_rate"],
        rec["footwork_rate"],
        rec["good_outcome_rate"],
        rec["miss_rate"],
    ]


def build_training_matrix(records: List[dict]):
    X = np.array([_to_feature_vector(r) for r in records], dtype=float)
    y = np.array([1 if r["true_mastered_after"] else 0 for r in records], dtype=int)
    return X, y


def train(n_batters: int = 250, balls_per_style: int = 16, seed: int = 42, save: bool = True) -> dict:
    """Train the neural scorer on simulated data and return honest
    held-out metrics — not training-set numbers dressed up as
    performance. Called from tests and from evaluate_coach_behavior.py."""
    records = generate_dataset(n_batters=n_batters, balls_per_style=balls_per_style, seed=seed)
    X, y = build_training_matrix(records)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    clf = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        max_iter=2000,
        random_state=seed,
    )
    clf.fit(X_train, y_train)

    proba_test = clf.predict_proba(X_test)[:, 1]
    pred_test = clf.predict(X_test)
    metrics = {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_auc": roc_auc_score(y_test, proba_test),
        "test_precision": precision_score(y_test, pred_test, zero_division=0),
        "test_recall": recall_score(y_test, pred_test, zero_division=0),
    }

    if save:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(clf, MODEL_PATH)

    return {"model": clf, "metrics": metrics}


def load_model() -> MLPClassifier:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model at {MODEL_PATH}. Run `python neural_scorer.py` "
            "or call train() first."
        )
    return joblib.load(MODEL_PATH)


@dataclass
class NeuralStyleRecord:
    style_key: str
    score: float = 0.0            # MLP's predicted mastery probability
    deliveries_seen: int = 0
    mastered: bool = False


class NeuralMasteryScorer:
    """Drop-in alternative to scoring.MasteryScorer. Same public methods
    (record_delivery, get, all_records, mastered_styles), but "mastered"
    is a trained model's prediction instead of a fixed EMA threshold.

    Internally still runs a rule-based MasteryScorer per style to produce
    the engineered features (ema_score, rates, etc.) the trained model
    expects — the neural network sits on top of those features rather
    than replacing feature engineering entirely, which keeps it small,
    fast, and trainable on the little data simulation can realistically
    provide.
    """

    def __init__(self, model: MLPClassifier = None, mastery_threshold: float = 0.85):
        # 0.85 was chosen empirically, not guessed: evaluate_coach_behavior.py
        # sweeps this threshold against simulated ground truth. At 0.85 the
        # neural scorer matches the rule-based scorer's false-trigger rate
        # (~12% vs ~11%) while responding faster (~3.0 vs ~3.5 balls delay)
        # and missing far fewer real mastery events (~0.4% vs ~7%). See that
        # script's output for the full sweep and how to re-run it.
        self._model = model or load_model()
        self._threshold = mastery_threshold
        self._feature_scorer = MasteryScorer()
        self._outcome_counts: Dict[str, Dict[str, int]] = {}
        self._on_time_counts: Dict[str, int] = {}
        self._footwork_counts: Dict[str, int] = {}
        self._results: Dict[str, NeuralStyleRecord] = {}

    def record_delivery(self, style_key: str, outcome: str,
                         on_time: bool = True, footwork_correct: bool = True) -> NeuralStyleRecord:
        rule_record = self._feature_scorer.record_delivery(style_key, outcome, on_time, footwork_correct)

        counts = self._outcome_counts.setdefault(style_key, {})
        counts[outcome] = counts.get(outcome, 0) + 1
        self._on_time_counts[style_key] = self._on_time_counts.get(style_key, 0) + int(on_time)
        self._footwork_counts[style_key] = self._footwork_counts.get(style_key, 0) + int(footwork_correct)

        seen = rule_record.deliveries_seen
        good = counts.get("middled", 0) + counts.get("attacked", 0)
        missed = counts.get("missed", 0)

        features = np.array([[
            rule_record.score,
            min(seen, 15) / 15.0,
            self._on_time_counts[style_key] / seen,
            self._footwork_counts[style_key] / seen,
            good / seen,
            missed / seen,
        ]])

        proba = float(self._model.predict_proba(features)[0, 1])
        # Same min-sample gate as the rule-based scorer — a confident
        # prediction off one ball is not something to trust regardless of
        # what the model says.
        mastered = seen >= MIN_SAMPLE and proba >= self._threshold

        record = NeuralStyleRecord(style_key=style_key, score=proba, deliveries_seen=seen, mastered=mastered)
        self._results[style_key] = record
        return record

    def get(self, style_key: str) -> NeuralStyleRecord:
        return self._results.get(style_key, NeuralStyleRecord(style_key=style_key))

    def all_records(self) -> Dict[str, NeuralStyleRecord]:
        return dict(self._results)

    def mastered_styles(self) -> List[str]:
        return [k for k, r in self._results.items() if r.mastered]


if __name__ == "__main__":
    result = train()
    m = result["metrics"]
    print(f"Trained on {m['n_train']} simulated deliveries, held out {m['n_test']}.")
    print(f"Held-out AUC: {m['test_auc']:.3f}  precision: {m['test_precision']:.3f}  recall: {m['test_recall']:.3f}")
    print(f"Model saved to {MODEL_PATH}")

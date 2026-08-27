import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from neural_scorer import train, NeuralMasteryScorer, MODEL_PATH


@pytest.fixture(scope="module")
def trained():
    # Smaller than the production training run (neural_scorer.py's
    # __main__) so the test suite stays fast, but big enough that a real
    # regression in the learning setup shows up as a real metric drop,
    # not noise.
    return train(n_batters=60, balls_per_style=16, seed=1, save=False)


def test_held_out_auc_is_meaningfully_better_than_chance(trained):
    auc = trained["metrics"]["test_auc"]
    assert auc > 0.85, f"AUC {auc:.3f} — model is barely better than a coin flip, something regressed"


def test_precision_and_recall_are_reasonable(trained):
    m = trained["metrics"]
    assert m["test_precision"] > 0.7
    assert m["test_recall"] > 0.7


def test_neural_scorer_not_mastered_before_min_sample(trained):
    scorer = NeuralMasteryScorer(model=trained["model"])
    record = None
    for _ in range(3):  # below MIN_SAMPLE (4)
        record = scorer.record_delivery("fast-good-off-none", "middled")
    assert record.mastered is False


def test_neural_scorer_flags_consistent_good_outcomes(trained):
    scorer = NeuralMasteryScorer(model=trained["model"])
    record = None
    for _ in range(10):
        record = scorer.record_delivery("fast-good-off-none", "middled", on_time=True, footwork_correct=True)
    assert record.mastered is True, (
        f"10 straight 'middled' deliveries should read as mastered (score={record.score:.2f})"
    )


def test_neural_scorer_does_not_flag_consistent_misses(trained):
    scorer = NeuralMasteryScorer(model=trained["model"])
    record = None
    for _ in range(10):
        record = scorer.record_delivery("slow-good-off-offspin", "missed", on_time=False, footwork_correct=False)
    assert record.mastered is False


def test_neural_scorer_mastered_styles_lists_only_mastered(trained):
    scorer = NeuralMasteryScorer(model=trained["model"])
    for _ in range(8):
        scorer.record_delivery("fast-good-off-none", "middled")
    for _ in range(8):
        scorer.record_delivery("slow-good-off-offspin", "missed")
    assert scorer.mastered_styles() == ["fast-good-off-none"]


def test_model_file_was_saved_by_the_production_training_run():
    # neural_scorer.py's __main__ (run separately, see README) saves here;
    # this just checks the artifact exists for app.py to load, without
    # re-running the full training job as part of the test suite.
    assert os.path.exists(MODEL_PATH), (
        "No trained model found — run `python neural_scorer.py` before running the app or this check"
    )

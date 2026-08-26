import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scoring import MasteryScorer, MASTERY_THRESHOLD, MIN_SAMPLE


def test_not_mastered_before_min_sample():
    scorer = MasteryScorer()
    for _ in range(MIN_SAMPLE - 1):
        record = scorer.record_delivery("fast-good-off-none", "middled")
    assert record.mastered is False


def test_mastered_after_consistent_good_outcomes():
    scorer = MasteryScorer()
    record = None
    for _ in range(MIN_SAMPLE + 2):
        record = scorer.record_delivery("fast-good-off-none", "middled")
    assert record.mastered is True
    assert record.score >= MASTERY_THRESHOLD


def test_poor_outcomes_keep_score_low():
    scorer = MasteryScorer()
    record = None
    for _ in range(MIN_SAMPLE + 2):
        record = scorer.record_delivery("slow-good-off-offspin", "missed")
    assert record.mastered is False
    assert record.score < MASTERY_THRESHOLD


def test_recent_struggle_pulls_score_back_down():
    scorer = MasteryScorer()
    for _ in range(6):
        scorer.record_delivery("fast-good-off-none", "middled")
    mastered_record = scorer.get("fast-good-off-none")
    assert mastered_record.mastered is True
    score_at_mastery = mastered_record.score  # snapshot the float, not the live record

    # A run of misses afterwards should erode the score even though the
    # batter "mastered" it earlier — the EMA naturally forgets stale form.
    for _ in range(4):
        record = scorer.record_delivery("fast-good-off-none", "missed")
    assert record.score < score_at_mastery


def test_unknown_outcome_rejected():
    scorer = MasteryScorer()
    try:
        scorer.record_delivery("fast-good-off-none", "six")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_mastered_styles_lists_only_mastered():
    scorer = MasteryScorer()
    for _ in range(6):
        scorer.record_delivery("fast-good-off-none", "middled")
    for _ in range(6):
        scorer.record_delivery("slow-good-off-offspin", "missed")
    assert scorer.mastered_styles() == ["fast-good-off-none"]

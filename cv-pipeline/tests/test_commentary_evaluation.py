import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import commentary_evaluation as ce
from commentary_labels import TranscriptSegment
from outcome_from_video import OutcomeEstimate


def estimate(outcome):
    return OutcomeEstimate(outcome, "estimate from arm motion — not validated against labelled outcomes", "reason")


def analysis(swing_frames, outcomes, fps=30.0):
    return SimpleNamespace(
        container_fps=fps, swing_frames=swing_frames, outcome_estimates=[estimate(o) for o in outcomes],
    )


def fake_transcribe(segments):
    def _fn(video_path, model_size="base"):
        return segments
    return _fn


def seg(start, text):
    return TranscriptSegment(start, start + 3.0, text)


def test_agreement_is_reported_when_the_estimate_matches_the_commentary(monkeypatch):
    a = analysis(swing_frames=[300], outcomes=["four-type shot"])          # 300/30 = 10.0s
    monkeypatch.setattr(ce, "transcribe_commentary", fake_transcribe([seg(12.0, "that's timed well for four")]))
    result = ce.compare_predictions_to_commentary(a, "clip.mp4")
    assert result.compared == 1 and result.correct == 1 and result.accuracy == 1.0
    assert result.rows[0].agrees is True and result.rows[0].actual == "four"


def test_a_disagreement_is_reported_not_hidden(monkeypatch):
    a = analysis(swing_frames=[300], outcomes=["dot ball"])
    monkeypatch.setattr(ce, "transcribe_commentary", fake_transcribe([seg(12.0, "and that's launched for six!")]))
    result = ce.compare_predictions_to_commentary(a, "clip.mp4")
    assert result.compared == 1 and result.correct == 0 and result.accuracy == 0.0
    assert result.rows[0].agrees is False
    assert result.confusion == {"dot ball -> six": 1}


def test_a_delivery_with_no_matching_commentary_is_not_compared_either_way(monkeypatch):
    a = analysis(swing_frames=[300, 600], outcomes=["four-type shot", "dot ball"])
    monkeypatch.setattr(ce, "transcribe_commentary", fake_transcribe([seg(12.0, "four runs there")]))
    result = ce.compare_predictions_to_commentary(a, "clip.mp4")
    assert result.compared == 1                            # only the first delivery had commentary
    assert result.rows[1].actual is None and result.rows[1].agrees is None


def test_a_delivery_the_video_pipeline_could_not_read_at_all_is_skipped_cleanly(monkeypatch):
    """swing_frames[i] is None when body_vectors[i] was None -- must not crash trying to divide by fps."""
    a = analysis(swing_frames=[None, 300], outcomes=["cannot estimate", "four-type shot"])
    monkeypatch.setattr(ce, "transcribe_commentary", fake_transcribe([seg(12.0, "four runs there")]))
    result = ce.compare_predictions_to_commentary(a, "clip.mp4")
    assert result.rows[0].actual is None
    assert result.rows[1].agrees is True


def test_wicket_type_delivery_matches_the_plain_word_wicket(monkeypatch):
    a = analysis(swing_frames=[300], outcomes=["wicket-type delivery"])
    monkeypatch.setattr(ce, "transcribe_commentary", fake_transcribe([seg(11.0, "bowled him, that's out!")]))
    result = ce.compare_predictions_to_commentary(a, "clip.mp4")
    assert result.correct == 1


def test_cannot_estimate_never_silently_counts_as_a_match():
    """Two nonsense strings must not accidentally produce True by string coincidence."""
    assert not ce._outcomes_match("cannot estimate", "dot ball")
    assert not ce._outcomes_match("cannot estimate", "wicket")


def test_commentary_that_matches_no_delivery_is_reported_as_unmatched_not_dropped(monkeypatch):
    a = analysis(swing_frames=[300], outcomes=["four-type shot"])
    monkeypatch.setattr(ce, "transcribe_commentary", fake_transcribe(
        [seg(12.0, "four runs there"), seg(90.0, "and that is out, bowled!")],
    ))
    result = ce.compare_predictions_to_commentary(a, "clip.mp4")
    assert len(result.unmatched_events) == 1 and result.unmatched_events[0].outcome == "wicket"


def test_the_note_states_how_few_examples_a_small_run_rests_on(monkeypatch):
    a = analysis(swing_frames=[300], outcomes=["four-type shot"])
    monkeypatch.setattr(ce, "transcribe_commentary", fake_transcribe([seg(12.0, "four runs there")]))
    result = ce.compare_predictions_to_commentary(a, "clip.mp4")
    assert "only 1" in result.note


def test_no_commentary_at_all_gives_no_accuracy_rather_than_a_fake_one(monkeypatch):
    a = analysis(swing_frames=[300], outcomes=["four-type shot"])
    monkeypatch.setattr(ce, "transcribe_commentary", fake_transcribe([]))
    result = ce.compare_predictions_to_commentary(a, "clip.mp4")
    assert result.accuracy is None and result.compared == 0

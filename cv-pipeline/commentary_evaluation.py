"""
The connecting piece: run `outcome_from_video.py`'s six/four/dot/wicket ESTIMATE against a clip's
own broadcast commentary as free ground truth, and report how often they agree — a real accuracy
number instead of "it runs without crashing".

This exists so that the moment a labelled (i.e. commentated) clip arrives, evaluating it is one
call, not a fresh scaffolding exercise. It follows the same guardrails
`shot_calibration.py` already established for exactly this kind of self-graded accuracy claim:
compare only where BOTH sides had something to say (an unmatched delivery or an event with nothing
to compare teaches nothing), report a confusion breakdown, and say plainly how few examples the
number rests on rather than let a big-looking percentage speak for a handful of deliveries.

What this genuinely checks and doesn't:
  - It checks whether the SWING-MOTION estimate (outcome_from_video.py) agrees with what a real
    commentator said happened. That is real evidence about the estimate's accuracy.
  - It does NOT validate the commentary transcription itself — Whisper can mishear or miss an
    announcement (see commentary_labels.py's docstring for the mishearings found on a real clip),
    so a wrong "match" here could be either side being wrong. A human should spot-check the
    `commentary_text` shown per row before trusting a run of these as calibration-grade data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from commentary_labels import (
    CommentaryEvent, DeliveryLabel, align_events_to_deliveries, find_outcome_events, transcribe_commentary,
)
from outcome_from_video import OutcomeEstimate
from video_pipeline import VideoAnalysis

MIN_EXAMPLES_FOR_A_HEADLINE_NUMBER = 15


@dataclass(frozen=True)
class ComparisonRow:
    delivery_index: int
    predicted: str
    actual: Optional[str]           # None if no commentary matched this delivery
    agrees: Optional[bool]          # None if `actual` is None
    commentary_text: str = ""
    lag_s: Optional[float] = None


@dataclass(frozen=True)
class CommentaryComparison:
    rows: List[ComparisonRow]
    compared: int                   # deliveries with both a prediction and a real commentary label
    correct: int
    accuracy: Optional[float]       # None if `compared` is 0
    confusion: dict                 # {"predicted -> actual": count}
    unmatched_events: List[CommentaryEvent]     # commentary that didn't line up with any detected delivery
    note: str


def compare_predictions_to_commentary(
    analysis: VideoAnalysis, video_path: str, model_size: str = "base", max_lag_s: float = 8.0,
) -> CommentaryComparison:
    """`analysis` is an already-run `video_pipeline.analyse_video()` result for `video_path` (kept
    separate rather than run internally, since transcription is slow and a caller may already have
    the video analysis from elsewhere — e.g. the app's cached Batch-mode result)."""
    segments = transcribe_commentary(video_path, model_size)
    events = find_outcome_events(segments)
    swing_times = [None if f is None else f / analysis.container_fps for f in analysis.swing_frames]
    known_times = [t for t in swing_times if t is not None]
    labels, unmatched = align_events_to_deliveries(known_times, events, max_lag_s=max_lag_s) if known_times else ([], events)

    # `labels` is indexed against `known_times`; map back to the original (possibly sparser) delivery list.
    known_positions = [i for i, t in enumerate(swing_times) if t is not None]
    label_by_delivery: dict = {known_positions[j]: lbl for j, lbl in enumerate(labels) if lbl is not None}

    rows: List[ComparisonRow] = []
    confusion: dict = {}
    correct = 0
    for i, estimate in enumerate(analysis.outcome_estimates):
        label: Optional[DeliveryLabel] = label_by_delivery.get(i)
        if label is None:
            rows.append(ComparisonRow(i, estimate.outcome, None, None))
            continue
        agrees = _outcomes_match(estimate.outcome, label.outcome)
        key = f"{estimate.outcome} -> {label.outcome}"
        confusion[key] = confusion.get(key, 0) + 1
        correct += agrees
        rows.append(ComparisonRow(i, estimate.outcome, label.outcome, agrees, label.event.text, label.lag_s))

    compared = sum(1 for r in rows if r.actual is not None)
    accuracy = (correct / compared) if compared else None
    note = (
        f"Rests on only {compared} delivery(ies) with real commentary to compare against."
        if compared < MIN_EXAMPLES_FOR_A_HEADLINE_NUMBER else ""
    )
    return CommentaryComparison(rows, compared, correct, accuracy, confusion, unmatched, note)


def _outcomes_match(predicted: str, actual: str) -> bool:
    """outcome_from_video's categories (dot ball / four-type shot / six-type shot / wicket-type
    delivery / cannot estimate) vs commentary_labels' plain outcome words (dot ball / four / six /
    wicket) — same four ideas, different spelling on each side."""
    predicted_key = predicted.replace("-type shot", "").replace("-type delivery", "")
    return predicted_key == actual

"""
Free ground-truth labels from broadcast commentary: a commentator saying "SIX!" or "That's out,
bowled!" is a real human judgement of the outcome, already recorded on the clip's own audio track —
no separate labelling session needed. This turns that audio into timestamped outcome events, and
lines them up with the deliveries the video pipeline already detected, so
`shot_calibration.py`/`outcome_from_video.py`'s guesses can be checked against a real answer instead
of only synthetic data.

Two independently testable steps:
  1. `transcribe_commentary()` — real speech-to-text (OpenAI's Whisper, run locally; see below for
     what "real" means here) turns the clip's audio into timestamped text segments.
  2. `find_outcome_events()` — pure text matching over those segments, tolerant of the specific
     mishearings found running this on a real broadcast clip (an ICC T20 World Cup qualifier):
     Whisper's "base" model heard "wicket" as "wicked" and, in one segment, apparently "four" as
     "fall" — cricket jargon in an accented commentary voice is exactly the kind of thing generic
     speech-to-text training data under-represents. This is a genuine, measured limitation, not a
     guess: the same clip's base-model transcript is what surfaced it.
  3. `align_events_to_deliveries()` — a commentator reacts AFTER seeing the outcome, so an event's
     timestamp lags the delivery it describes by a few seconds, not the near-simultaneous alignment
     delivery_sync.py assumes for two sensor-like clocks; matching here is deliberately asymmetric
     (an event may only follow a delivery, within a bounded window) for that reason.

What this is NOT: automatic ground truth. Whisper can mishear or miss an announcement entirely
(silence, crowd noise, an unclear call), and even a correct transcript still needs a human to check
the resulting label file before it is trusted as calibration data — this fills
`shot_calibration.TrialRecord`s FASTER, it does not remove the "labelled by a person" step
shot_calibration.py's guardrails already insist on. Treat its output as a strong suggestion to
confirm, not a finished label.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

SIX, FOUR, WICKET, DOT_BALL = "six", "four", "wicket", "dot ball"

# Cricket-specific keyword patterns, INCLUDING near-homophones Whisper was found to produce on a
# real broadcast clip ("wicket" -> "wicked"). \b keeps these as whole words, so "fourteen" or
# "wickedly" don't false-match; run in lowercase, punctuation-stripped text.
OUTCOME_PATTERNS: Dict[str, List[str]] = {
    SIX: [r"\bsix\b", r"\bsixes\b", r"\bmaximum\b"],
    FOUR: [r"\bfour\b", r"\bfours\b", r"\bboundary\b"],
    WICKET: [r"\bwicket\b", r"\bwicked\b", r"\bbowled\b", r"\bcaught\b", r"\bstumped\b",
              r"\brun\s*out\b", r"\bgiven\s*out\b", r"\bhe.?s\s+out\b", r"\bthat.?s\s+out\b", r"\blbw\b"],
    DOT_BALL: [r"\bdot\s*ball\b", r"\bno\s*run\b", r"\bnothing\s*there\b", r"\bdefended\b"],
}
# A bare "\bout\b" was tried and dropped: real commentary produces it constantly in ways that are
# NOT a dismissal ("watch out", "without", "as it turns out" — found running this on a real clip,
# where "Watch out, delivery, this is." was misread as a wicket). Every genuine dismissal call this
# project has actually seen also says "wicket", "bowled", "caught", "stumped", "lbw" or "given/run
# out" somewhere in the same breath, so those specific words carry the WICKET signal instead.
# Checked in this order: WICKET and FOUR/SIX are more specific claims — DOT_BALL last, since its
# patterns are the most generic and likeliest to false-match ordinary commentary filler.
_ORDER = [WICKET, SIX, FOUR, DOT_BALL]


@dataclass(frozen=True)
class TranscriptSegment:
    start_s: float
    end_s: float
    text: str


@dataclass(frozen=True)
class CommentaryEvent:
    time_s: float               # the segment's start — when the commentator began saying this
    outcome: str                # SIX / FOUR / WICKET / DOT_BALL
    matched_phrase: str
    text: str                   # the full segment, for a human to read and confirm


def transcribe_commentary(video_path: str, model_size: str = "base") -> List[TranscriptSegment]:
    """Extracts the clip's audio (via ffmpeg — must be on PATH; raises a clear error naming the
    fix, the same "no silent failure" pattern as pose_estimation.download_model()) and transcribes
    it locally with OpenAI's Whisper. Real speech-to-text, real audio, real model weights (Whisper
    downloads its own checkpoint the first time a given `model_size` is used — the same one-time,
    explicit, on-demand pattern as cv-pipeline's pose model). "base" is fast and was enough to
    surface real commentary on a genuine broadcast clip; a larger model may transcribe cricket
    jargon more accurately at the cost of speed — neither has been checked against a labelled
    transcript, so treat either's mishearings as expected, not a bug to chase blindly."""
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError(
            "ffmpeg is not on PATH — install it (e.g. https://ffmpeg.org/download.html) before "
            "transcribing a clip's commentary; nothing here can extract audio without it."
        )
    import whisper  # deferred: importing pulls in torch, only needed if this function is actually called

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path],
            check=True, capture_output=True,
        )
        model = whisper.load_model(model_size)
        result = model.transcribe(audio_path, fp16=False)
        return [
            TranscriptSegment(float(seg["start"]), float(seg["end"]), seg["text"].strip())
            for seg in result["segments"]
        ]
    finally:
        import os
        os.unlink(audio_path)


def find_outcome_events(segments: Sequence[TranscriptSegment]) -> List[CommentaryEvent]:
    """One event per segment at most — the FIRST outcome keyword found, in `_ORDER`'s priority, so
    a segment mentioning both "wicket" and an incidental "out" isn't double-counted, and a segment
    that just says "...he's out of his crease..." isn't reported twice for the same word appearing
    in two patterns."""
    events = []
    for seg in segments:
        clean = re.sub(r"[^\w\s]", " ", seg.text.lower())
        # "not out" is one of the most common phrases in cricket commentary and means the OPPOSITE
        # of a wicket, but contains the standalone word "out" — found by a test, not by inspection.
        # "round/around/over the wicket" describes the bowler's angle of approach and has nothing to
        # do with a dismissal — found running this on the REAL transcript below ("goes around the
        # wicket..."). Both are stripped before matching so they can never be read as a dismissal,
        # while any OTHER, unrelated occurrence of "out" or "wicket" in the same segment still can.
        clean = re.sub(r"\bnot\s+out\b", " ", clean)
        clean = re.sub(r"\b(around|round|over)\s+the\s+wicke[dt]\b", " ", clean)   # tolerate the same "wicket"->"wicked" mishearing
        for outcome in _ORDER:
            match = next((p for p in OUTCOME_PATTERNS[outcome] if re.search(p, clean)), None)
            if match:
                events.append(CommentaryEvent(seg.start_s, outcome, re.search(match, clean).group(0), seg.text))
                break
    return events


@dataclass(frozen=True)
class DeliveryLabel:
    delivery_index: int
    outcome: str
    event: CommentaryEvent
    lag_s: float                # how long after the delivery's swing the commentary event began


def align_events_to_deliveries(
    swing_times_s: Sequence[float], events: Sequence[CommentaryEvent],
    min_lag_s: float = 0.0, max_lag_s: float = 8.0,
) -> Tuple[List[Optional[DeliveryLabel]], List[CommentaryEvent]]:
    """Match each commentary event to the delivery it most likely describes. Deliberately
    ASYMMETRIC, unlike delivery_sync.align()'s two-sensor-clock matching: a commentator reacts
    AFTER the ball, so an event may only match a swing at or before it, within `max_lag_s` — an
    event can never explain a delivery that hasn't happened yet. Each delivery gets at most the
    NEAREST preceding unclaimed event within the window; each event claims at most one delivery.

    Returns (one label-or-None per delivery, in the given order; events that matched nothing).
    """
    swings = sorted(enumerate(swing_times_s), key=lambda p: p[1])
    labels: List[Optional[DeliveryLabel]] = [None] * len(swing_times_s)
    unmatched: List[CommentaryEvent] = []
    used_deliveries = set()
    for event in sorted(events, key=lambda e: e.time_s):
        best = None
        for idx, t in swings:
            if idx in used_deliveries:
                continue
            lag = event.time_s - t
            if min_lag_s <= lag <= max_lag_s and (best is None or lag < best[1]):
                best = (idx, lag)
        if best is None:
            unmatched.append(event)
        else:
            idx, lag = best
            labels[idx] = DeliveryLabel(idx, event.outcome, event, lag)
            used_deliveries.add(idx)
    return labels, unmatched

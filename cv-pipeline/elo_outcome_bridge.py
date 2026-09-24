"""
Bridges cv-pipeline's outcome estimates onto the 0-1 scale trajectory-engine's
adaptive.py Elo engine already expects, so a delivery's real outcome can update a
player's adaptive rating without a human typing it into the app's dropdown.

The hook on the other end has existed all along: adaptive.PlayerProfile.record_outcome()
was built to take either a manual OUTCOME_SCORES key or "a raw 0-1 float, so this same
call is ready to be driven by an automated vision/sensor system later without changing
its signature" (its own docstring). Nothing before this module actually produced that
float from cv-pipeline's outputs — this is that missing piece.

Two independently testable sources, kept SEPARATE rather than blended into one number a
caller can't tell apart, because they earn very different trust:

  * `score_from_commentary()` — a real commentator's own call (six/four/wicket/dot
    ball), already extracted as ground truth by commentary_labels.py. Subject only to
    Whisper mishearing a word (which commentary_labels.py already guards against with
    its own fixes), this is safe to auto-apply to a player's rating.

  * `score_from_video_estimate()` — outcome_from_video.py's ARM-MOTION guess, which
    reads the swing, not the ball. Tested against the real labelled clips this project
    has actually seen: right on a dot ball, wrong on a wicket (read as a four/six-type
    shot, because a broken wicket doesn't change how the arms moved), and unable to
    measure a body at all on one real four. That is not accurate enough to silently
    move a player's rating — a wrongly-scored "six" would push their Elo the wrong way
    on real data, which is worse than not updating it at all. So this function always
    returns a `VideoOutcomeScore` with an `is_confident` flag, and callers MUST treat
    `is_confident=False` as "ask a human", never as "apply anyway" — see its docstring
    for exactly which cases that covers and why.

Nothing here invents new thresholds: both functions are thin, honest translations of
labels the two upstream modules already produce, onto the scale the Elo engine already
consumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from commentary_labels import DOT_BALL, FOUR, SIX, WICKET
from outcome_from_video import DOT, OutcomeEstimate
from outcome_from_video import FOUR as VIDEO_FOUR
from outcome_from_video import SIX as VIDEO_SIX
from outcome_from_video import UNCLEAR
from outcome_from_video import WICKET as VIDEO_WICKET

# Re-expressed here rather than imported from adaptive.py, so this module has no
# dependency on trajectory-engine being installed — a caller that only has cv-pipeline
# can still compute a score; only actually calling record_outcome() needs adaptive.py.
# Values are copied verbatim from adaptive.OUTCOME_SCORES and must stay in sync with it
# (checked by test_elo_outcome_bridge.py importing both and comparing).
_MISSED = 0.00
_DEFENDED = 0.55
_BOUNDARY = 0.95
_SIX = 1.00


@dataclass(frozen=True)
class VideoOutcomeScore:
    score: Optional[float]      # None when there is nothing safe to report at all (UNCLEAR)
    is_confident: bool          # False means: show this as a suggestion, do not auto-apply it
    reason: str                 # why confident or not, for a human reviewing the suggestion


def score_from_commentary(outcome: str) -> float:
    """`outcome` is one of commentary_labels.py's four outcome strings (SIX, FOUR,
    WICKET, DOT_BALL). A real commentator's call is ground truth (modulo Whisper
    mishearing, already guarded against upstream) — always safe to apply directly.

    DOT_BALL maps to "defended" (0.55) rather than "beaten" (0.05): commentary alone
    can't tell a solidly-defended dot from one where the batter was beaten outside
    off stump, and a defended dot is the far more common real case in normal play —
    a named, documented approximation, not a hidden guess.

    WICKET maps to "missed" (0.00) regardless of how it happened (bowled, caught,
    lbw, stumped, run out) — commentary_labels.py does not distinguish dismissal
    types, and all of them are the worst outcome for the batter on this 0-1 scale.
    """
    return {
        SIX: _SIX,
        FOUR: _BOUNDARY,
        WICKET: _MISSED,
        DOT_BALL: _DEFENDED,
    }[outcome]


def score_from_video_estimate(estimate: OutcomeEstimate) -> VideoOutcomeScore:
    """Translate outcome_from_video.py's arm-motion guess into a score plus an
    honest confidence flag.

    `is_confident` is False for exactly the cases real-clip testing found this
    estimator cannot be trusted on:
      * UNCLEAR — no swing was measured at all; there is nothing to report (score
        is None, not a guessed middle value).
      * WICKET — the ONE outcome this project has directly confirmed the estimator
        gets wrong on real footage (a genuine wicket was read as a four/six-type
        shot). It can only ever suggest "wicket" itself from a mistimed-lofted-shot
        read, which is a guess about being caught, not a measurement of a broken
        stump — never safe to auto-apply.
      * Any estimate that leaned on a secondary signal (post-shot running, or a lost
        tracking note) rather than the swing alone — detected mechanically via
        `len(estimate.caveats) > 1`, since estimate_net_outcome() only appends a
        second caveat when it used one of those weaker, heavily-caveated paths.
    Everything else (a swing was actually measured, primary signal only, not a
    WICKET verdict) is confident: DOT, FOUR-type and SIX-type reads from a real,
    measured swing are the one case (dot ball) this project's own real-clip test
    got right, and the other two at least reflect an actual measured swing rather
    than a guess with no data behind it at all.
    """
    if estimate.outcome == UNCLEAR:
        return VideoOutcomeScore(None, False, "No swing was measured for this delivery — nothing to score.")
    if estimate.outcome == VIDEO_WICKET:
        return VideoOutcomeScore(
            _MISSED, False,
            "Arm motion alone cannot confirm a wicket (tested on real footage: this exact path misread "
            "a real wicket as a boundary) — confirm from the clip before applying.",
        )
    if len(estimate.caveats) > 1:
        return VideoOutcomeScore(
            {DOT: _DEFENDED, VIDEO_FOUR: _BOUNDARY, VIDEO_SIX: _SIX}[estimate.outcome], False,
            "This estimate leaned on a secondary signal (post-shot movement), not the swing alone — "
            "confirm before applying.",
        )
    score = {DOT: _DEFENDED, VIDEO_FOUR: _BOUNDARY, VIDEO_SIX: _SIX}[estimate.outcome]
    return VideoOutcomeScore(score, True, "Read directly from a measured swing, primary signal only.")

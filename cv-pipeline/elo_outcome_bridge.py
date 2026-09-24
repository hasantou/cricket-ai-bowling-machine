"""
Bridges cv-pipeline's outcome estimates onto adaptive.py's OUTCOME_SCORES keys, so a
delivery's real outcome can drive both the Elo rating AND the scorecard (runs, wickets)
without a human typing it into the app's dropdown.

Returns the KEY ("six", "boundary", "missed", "defended", ...), not a bare float. That
match matters: `scorecard.ScoreCard.record_ball()` — the actual call the app makes per
ball — needs the key for two different reasons at once, not one. It passes it straight
to `profile.record_outcome()` for the Elo update (which does also accept a raw float,
per that function's own docstring), but it ALSO passes it to `score_outcome()` to work
out runs scored and whether it's a wicket — and that lookup only understands named keys,
it has no notion of a bare 0-1 number. A float would silently satisfy the Elo half of
record_ball() and then blow up (or worse, not blow up) on the scoring half, so the key is
the only thing that actually plugs into how this app is wired, not a design preference.

Two independently testable sources, kept SEPARATE rather than blended into one call a
caller can't tell apart, because they earn very different trust:

  * `key_from_commentary()` — a real commentator's own call (six/four/wicket/dot
    ball), already extracted as ground truth by commentary_labels.py. Subject only to
    Whisper mishearing a word (which commentary_labels.py already guards against with
    its own fixes), this is safe to auto-apply to a player's rating and scorecard.

  * `key_from_video_estimate()` — outcome_from_video.py's ARM-MOTION guess, which
    reads the swing, not the ball. Tested against the real labelled clips this project
    has actually seen: right on a dot ball, wrong on a wicket (read as a four/six-type
    shot, because a broken wicket doesn't change how the arms moved), and unable to
    measure a body at all on one real four. That is not accurate enough to silently
    move a player's rating or scorecard — a wrongly-scored "six" would push their Elo
    the wrong way AND wrongly credit six real runs, which is worse than not logging the
    ball automatically at all. So this function always returns a `VideoOutcomeGuess`
    with an `is_confident` flag, and callers MUST treat `is_confident=False` as "ask a
    human", never as "apply anyway" — see its docstring for exactly which cases that
    covers and why.

Nothing here invents new thresholds: both functions are thin, honest translations of
labels the two upstream modules already produce, onto keys the scoring engine already
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

# adaptive.OUTCOME_SCORES' own keys, re-typed here rather than imported so this module
# has no hard dependency on trajectory-engine being installed — a caller that only has
# cv-pipeline can still compute a key; only actually calling record_outcome()/
# record_ball() needs adaptive.py itself. Kept in sync by
# test_elo_outcome_bridge.py, which imports both and checks every key used below is a
# real member of adaptive.OUTCOME_SCORES.
_MISSED = "missed"
_DEFENDED = "defended"
_BOUNDARY = "boundary"
_SIX = "six"

_VIDEO_OUTCOME_TO_KEY = {DOT: _DEFENDED, VIDEO_FOUR: _BOUNDARY, VIDEO_SIX: _SIX}


@dataclass(frozen=True)
class VideoOutcomeGuess:
    outcome_key: Optional[str]  # an adaptive.OUTCOME_SCORES key, or None (nothing safe to report — UNCLEAR)
    is_confident: bool          # False means: show this as a suggestion, do not auto-apply it
    reason: str                 # why confident or not, for a human reviewing the suggestion


def key_from_commentary(outcome: str) -> str:
    """`outcome` is one of commentary_labels.py's four outcome strings (SIX, FOUR,
    WICKET, DOT_BALL). A real commentator's call is ground truth (modulo Whisper
    mishearing, already guarded against upstream) — always safe to apply directly.

    DOT_BALL maps to "defended" rather than "beaten": commentary alone can't tell a
    solidly-defended dot from one where the batter was beaten outside off stump, and a
    defended dot is the far more common real case in normal play — a named, documented
    approximation, not a hidden guess.

    WICKET maps to "missed" regardless of how it happened (bowled, caught, lbw,
    stumped, run out) — commentary_labels.py does not distinguish dismissal types, and
    "missed" ("played and missed entirely / bowled") is the closest existing bucket for
    "the batter is out", the worst outcome on this scale either way.
    """
    return {
        SIX: _SIX,
        FOUR: _BOUNDARY,
        WICKET: _MISSED,
        DOT_BALL: _DEFENDED,
    }[outcome]


def key_from_video_estimate(estimate: OutcomeEstimate) -> VideoOutcomeGuess:
    """Translate outcome_from_video.py's arm-motion guess into an outcome key plus an
    honest confidence flag.

    `is_confident` is False for exactly the cases real-clip testing found this
    estimator cannot be trusted on:
      * UNCLEAR — no swing was measured at all; there is nothing to report
        (`outcome_key` is None, not a guessed middle value).
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
        return VideoOutcomeGuess(None, False, "No swing was measured for this delivery — nothing to score.")
    if estimate.outcome == VIDEO_WICKET:
        return VideoOutcomeGuess(
            _MISSED, False,
            "Arm motion alone cannot confirm a wicket (tested on real footage: this exact path misread "
            "a real wicket as a boundary) — confirm from the clip before applying.",
        )
    if len(estimate.caveats) > 1:
        return VideoOutcomeGuess(
            _VIDEO_OUTCOME_TO_KEY[estimate.outcome], False,
            "This estimate leaned on a secondary signal (post-shot movement), not the swing alone — "
            "confirm before applying.",
        )
    return VideoOutcomeGuess(
        _VIDEO_OUTCOME_TO_KEY[estimate.outcome], True,
        "Read directly from a measured swing, primary signal only.",
    )

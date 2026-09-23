"""
"Was that a six, a four, a dot ball, or a wicket?" — estimated from video, for a net session with
no boundary rope and no fielders to actually produce those outcomes.

This is the video-side counterpart to trajectory-engine's net_outcome.py, which classifies the same
four-way outcome from a SENSOR's measured exit speed and launch angle. That module cannot run here —
there is no post-contact sensor on a phone clip — so this reads the same idea off the BATTER'S ARM
MOTION instead: how fast the hands moved through the swing, and how much they rose (a lofted swing).

READ THIS BEFORE TRUSTING IT — the single biggest gap, stated plainly:
  This cannot see the BALL, and cannot confirm bat-ball CONTACT happened at all. It reads the swing
  the arms made and assumes the ball came off it roughly the way the swing looks. A hard, clean
  swing that missed the ball completely (a genuine bowled/missed delivery) looks, to this code,
  identical to a hard swing that connected — because both are just "the arms swung fast". This is
  why every label below says "the SWING looked like..." rather than "the ball was...", and why this
  is offered as a coaching prompt to check, not a scorecard entry to trust.

Thresholds (torso-lengths/s for power, torso-lengths for the swing's net upward travel) are
illustrative round numbers set from watching a handful of real swings measured elsewhere in this
project (7-12 torso-lengths/s for a real, connected-looking swing) — NOT fitted to labelled
outcomes. `cv-pipeline/shot_calibration.py`'s guardrails (30+ trusted labelled examples, held-out
scoring, adopt only if clearly better) are the template for calibrating these once real labelled
deliveries exist; nothing here is calibrated yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from body_vectors import BodyVectorReport
from post_shot import LOST_TRACK, RAN, PostShotReport

DOT = "dot ball"
FOUR = "four-type shot"
SIX = "six-type shot"
WICKET = "wicket-type delivery"
UNCLEAR = "cannot estimate"

# --- swing power, torso-lengths/s (relative to the body — see body_vectors.py) ---
FOUR_MIN_SPEED = 7.0          # a swing at least this fast looks like a real, connected shot
SIX_MIN_SPEED = 11.0          # ...and at least this fast looks like a big, deliberate hit
# --- how much the hands rose net through the swing, torso-lengths — a lofted-looking swing ---
LOFT_MIN_UP = 0.5


@dataclass(frozen=True)
class OutcomeEstimate:
    outcome: str                    # DOT / FOUR / SIX / WICKET / UNCLEAR
    confidence: str                 # always says it's an estimate from arm motion, not a measurement
    reasoning: str
    caveats: List[str] = field(default_factory=list)


_BASE_CAVEAT = (
    "This reads the batter's ARM MOTION, not the ball — it cannot confirm bat-ball contact happened "
    "at all, so a hard swing that missed the ball completely would look the same as one that connected. "
    "Treat it as a prompt to check the clip, not a scorecard entry."
)


def _estimate(outcome: str, reasoning: str, extra: Optional[str] = None) -> OutcomeEstimate:
    caveats = [_BASE_CAVEAT] + ([extra] if extra else [])
    return OutcomeEstimate(outcome, "estimate from arm motion — not validated against labelled outcomes", reasoning, caveats)


def estimate_net_outcome(
    body: Optional[BodyVectorReport], post_shot: Optional[PostShotReport] = None,
) -> OutcomeEstimate:
    """`body` is the delivery's BodyVectorReport (body_vectors.py); `post_shot`, if available, is
    what the batter did in the couple of seconds after the swing (post_shot.py) — used only as a
    weak, heavily-caveated secondary signal, never to call a wicket on its own."""
    if body is None:
        return _estimate(UNCLEAR, "No body movement could be measured for this delivery.")
    if not body.speed_plausible:
        return _estimate(UNCLEAR, "The measured swing speed is physically implausible — a pose-tracking glitch, not a real reading.")
    if body.swing_verdict == "no swing":
        reasoning = "The hands barely moved relative to the body — no swing was played (defended or left alone)."
        if post_shot is not None and post_shot.classification == LOST_TRACK:
            return _estimate(
                DOT, reasoning,
                "The batter also dropped out of tracking shortly after — that alone could mean anything "
                "(walked off, camera cut, occlusion) and is not evidence of a dismissal by itself.",
            )
        return _estimate(DOT, reasoning)

    speed, up = body.peak_hand_speed, body.hand_path_net[1]
    lofted = up >= LOFT_MIN_UP
    shape = f"peak hand speed {speed:.1f} torso-lengths/s, net {up:+.1f} torso-lengths of upward hand travel"

    if lofted and speed < FOUR_MIN_SPEED:
        return _estimate(
            WICKET, f"A swing that rose a lot ({shape}) without much pace behind it looks like a mistimed, "
                    "lofted shot — the kind that often ends up caught.",
            "A powerful, deliberate lofted shot can look similar; speed alone can't fully separate the two.",
        )
    if lofted and speed >= SIX_MIN_SPEED:
        extra = None
        if post_shot is not None and post_shot.classification != RAN:
            extra = "No running was measured after the shot, which would be unusual for a genuine six — worth checking the clip."
        return _estimate(SIX, f"A fast, lofted swing ({shape}) looks like a big, deliberate hit.", extra)
    if speed >= FOUR_MIN_SPEED:
        flat_or_lofted = "a lofted, well-struck" if lofted else "a fast, flat"
        return _estimate(FOUR, f"{flat_or_lofted.capitalize()} swing ({shape}) looks like a well-struck shot.")

    reasoning = f"A swing was played, but neither fast nor lofted enough to look like a boundary attempt ({shape})."
    if post_shot is not None and post_shot.classification == RAN:
        return _estimate(
            FOUR, reasoning + " The batter was measured running hard afterwards, though, which points the other way.",
            "Running can also mean a risky single, not necessarily a boundary — a weak signal on its own.",
        )
    return _estimate(DOT, reasoning)

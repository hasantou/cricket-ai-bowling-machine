"""
"What shot was it?" from a video of the batter — an algorithm for GOOD footage.

It reads the batter's body vectors (body_vectors.py) and turns them into a
shot: first whether there was a swing at all, then how big and which shape it
was (a defensive push, a drive-type swing that finishes high, or a
horizontal-bat swing), then which side of the batter the hands travelled to,
and from those a shot name from the project's shot vocabulary.

HOW MUCH TO TRUST IT — read this first
  - The rules are a rule of thumb: named, adjustable thresholds set from
    cricket knowledge, NOT fitted to labelled shots. No labelled good-quality
    clip exists yet, so there is no accuracy figure and none is claimed.
    Expect errors. cv-pipeline/shot_labels.py:evaluate() is the way to get a
    real number once labelled shots exist, and the thresholds are the things
    to tune against it.
  - It reads the BATTER'S HANDS, not the ball. A hand path is correlated with
    where the ball goes, not the same thing: the bat face at contact decides
    that. The machine path (trajectory-engine/shot_analysis.py) reads the
    ball itself and is the stronger source wherever sensors exist.
  - So it refuses rather than guesses. It names a shot only if ALL of these
    hold, and otherwise says "cannot tell" and why:
        * the footage-quality verdict is "good";
        * the person was tracked in nearly every frame;
        * the fast hand itself (not the trunk) was tracked with high model
          confidence at the swing's peak;
        * the peak speed is physically plausible.
    On the one real clip available (batter filmed from behind, hands blurred
    and hidden) it correctly refuses.
  - The caller must say where the camera stood and which hand the batter
    bats with: "off side" is not a property of the picture. From behind a
    right-hander, the off side is screen-right; from the bowler's end it is
    screen-left; left-handers mirror it. From side-on, screen-left/right
    means toward the bowler / wicketkeeper, so the algorithm names the FAMILY
    of shot but no side.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from body_vectors import BodyVectorReport
from footage_check import FootageReport

# --- camera positions and hands -------------------------------------------
BEHIND_BATTER = "behind the batter"
BOWLERS_END = "bowler's end (facing the batter)"
SIDE_ON = "side-on"
CAMERAS = (BEHIND_BATTER, BOWLERS_END, SIDE_ON)
RIGHT_HANDED, LEFT_HANDED = "right-handed", "left-handed"

# --- gates: when to refuse ---------------------------------------------------
MIN_TRACKED_FRACTION = 0.90      # person found in at least this share of the delivery window

# --- swing shape rules (torso-lengths; uncalibrated rules of thumb) ---------
DEFENCE_MAX_PATH = 1.2           # hand path through the swing shorter than this = a push/block, not a stroke
STRAIGHT_MAX_ACROSS = 0.4        # net sideways travel within this = down the ground
WIDE_MIN_ACROSS = 1.0            # at least this far across = the wider shot of its kind (cover / flick)
HORIZONTAL_RATIO = 1.2           # |across| >= this * |up|  => a horizontal-bat swing; otherwise drive-type
HIGH_FINISH_UP = 0.5             # a horizontal swing whose hands also rise this far = a hook, not a pull

DRIVE_FAMILY, HORIZONTAL_FAMILY, DEFENCE_FAMILY = "drive-type swing", "horizontal-bat swing", "defensive push"
OFF, LEG, STRAIGHT = "off side", "leg side", "straight"


@dataclass(frozen=True)
class VideoShotEstimate:
    verdict: str                     # "shot named" | "family only" | "no shot" | "cannot tell"
    shot: Optional[str]              # a shot_vocabulary name, when one is justified
    family: Optional[str]
    side: Optional[str]              # "off side" / "leg side" / "straight" / None (not readable from this camera)
    confidence: str                  # always "rule of thumb — not validated" when a shot is named
    reasons: List[str]


def _refuse(reasons: List[str]) -> VideoShotEstimate:
    return VideoShotEstimate("cannot tell", None, None, None, "n/a", reasons)


def _screen_direction_to_side(across: float, camera: str, hand: str) -> Optional[str]:
    """Map net screen-horizontal hand travel (+ = screen-right, torso-lengths)
    to the batter's off / leg side. None from side-on: there, sideways on
    screen is toward the bowler or keeper, not off/leg."""
    if camera == SIDE_ON:
        return None
    off_is_screen_right = camera == BEHIND_BATTER
    if hand == LEFT_HANDED:
        off_is_screen_right = not off_is_screen_right
    if abs(across) < STRAIGHT_MAX_ACROSS:
        return STRAIGHT
    goes_right = across > 0
    return OFF if goes_right == off_is_screen_right else LEG


def estimate_shot_from_video(
    report: Optional[BodyVectorReport], footage: FootageReport, camera: str, hand: str = RIGHT_HANDED,
) -> VideoShotEstimate:
    if camera not in CAMERAS:
        raise ValueError(f"camera must be one of {CAMERAS}")
    if report is None:
        return _refuse(["No body movement could be measured for this delivery."])

    # --- gates: refuse rather than guess -------------------------------------
    reasons: List[str] = []
    if footage.verdict != "good":
        reasons.append(f"Footage quality is {footage.verdict}, not good ({' '.join(footage.reasons)})")
    if report.tracked_fraction < MIN_TRACKED_FRACTION:
        reasons.append(f"The batter was found in only {report.tracked_fraction * 100:.0f}% of this delivery's frames.")
    if not report.speed_plausible:
        reasons.append("The measured hand speed is physically implausible — a pose-tracking glitch.")
    if report.peak_hand == "trunk":
        reasons.append("Neither hand could be tracked (hidden or blurred from this camera angle).")
    else:
        v = report.vectors_at_peak.get(report.peak_hand)
        if v is None or v.confidence != "good":
            reasons.append(
                f"The fast hand ({report.peak_hand}) was only "
                f"{'partly' if v is not None and v.confidence == 'partial' else 'poorly'} visible to the pose model."
            )
    if reasons:
        return _refuse(reasons)

    # --- was there a swing? --------------------------------------------------
    if report.swing_verdict == "no swing":
        return VideoShotEstimate(
            "no shot", "Leave", None, None, "rule of thumb — not validated",
            ["The hands barely moved relative to the body. A leave and a soft block can't be told apart "
             "without seeing the ball touch the bat."],
        )

    across, up = report.hand_path_net
    length = report.hand_path_length

    if length < DEFENCE_MAX_PATH:
        return VideoShotEstimate(
            "family only", None, DEFENCE_FAMILY, None, "rule of thumb — not validated",
            [f"A short hand path ({length:.1f} torso-lengths) — a push or block. Whether it was forward or "
             "back-foot defence needs the feet, which this does not read."],
        )

    horizontal = abs(across) >= HORIZONTAL_RATIO * abs(up)
    family = HORIZONTAL_FAMILY if horizontal else DRIVE_FAMILY
    side = _screen_direction_to_side(across, camera, hand)
    shape = (
        f"hand path {length:.1f} torso-lengths, net {across:+.1f} across / {up:+.1f} up, "
        f"{report.peak_hand_speed:.0f} torso-lengths/s at the peak"
    )
    if side is None:
        return VideoShotEstimate(
            "family only", None, family, None, "rule of thumb — not validated",
            [f"{family.capitalize()} ({shape}). Filmed side-on, sideways on screen is toward the bowler or "
             "keeper, so which side of the wicket it went cannot be read."],
        )

    wide = abs(across) >= WIDE_MIN_ACROSS
    if family == DRIVE_FAMILY:
        shot = {STRAIGHT: "Straight drive", OFF: "Cover drive" if wide else "Off drive",
                LEG: "Flick" if wide else "On drive"}[side]
    else:
        if side == OFF:
            shot = "Cut"
        elif side == LEG:
            shot = "Hook" if up >= HIGH_FINISH_UP else "Pull"
        else:
            shot = "Back-foot punch"
    return VideoShotEstimate(
        "shot named", shot, family, side, "rule of thumb — not validated",
        [f"{family.capitalize()} toward the {side} ({shape}). Read from the hands' path, not the ball: "
         "the bat face at contact decides where the ball actually goes."],
    )

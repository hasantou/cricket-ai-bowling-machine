"""
Is this clip good enough for pose-based analysis to mean anything?

Pose analysis fails quietly. A tiny person in shot, a model that isn't sure
of the joints, frequent dropouts, or a video the server can only partly
decode all still yield plausible-looking deliveries with footwork and
timing verdicts — nothing in the output says the input was poor. This
module is that missing warning: a few measurable facts about the footage,
and a plain verdict with reasons.

(An early guess that a particular arena clip fell into the "tiny person"
case was wrong — checking showed the tracked batter was 37% of the frame,
because the camera stood behind them. The check exists so that kind of
guess gets replaced by a measurement, not the other way round.)

The thresholds are judgment calls from a handful of clips, named and
adjustable, not calibrated against any labelled ground truth — they exist so
an operator is told "the person is tiny in this shot," not to be trusted as
a precise quality score.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import List, Sequence

from feature_extraction import FrameLandmarks

NOSE = 0
KEY_JOINTS = (11, 12, 15, 16, 23, 24)   # shoulders, wrists, hips
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

POOR_PERSON_FRACTION = 0.15      # person shorter than this fraction of the frame -> poor
MARGINAL_PERSON_FRACTION = 0.30
POOR_VISIBILITY = 0.50
MARGINAL_VISIBILITY = 0.70
POOR_DETECTION_RATE = 0.60
MARGINAL_DETECTION_RATE = 0.85
MIN_HAND_VISIBILITY = 0.60       # share of frames with BOTH wrists confidently seen, needed to read a swing's shape
MIN_LEG_SEPARATION = 0.06        # median ankle-to-ankle gap as a fraction of the person's height; below = legs overlap
UNREADABLE_FRACTION = 0.95     # decoded fewer than this share of the promised frames -> unreadable


@dataclass(frozen=True)
class FootageReport:
    frames_total: int
    frames_with_person: int
    detection_rate: float
    person_height_fraction: float    # median nose-to-ankle height as a fraction of the frame height
    median_visibility: float         # median model confidence over the key joints
    frames_expected: int             # what the video container said it holds (0 = unknown)
    verdict: str                     # "good" | "marginal" | "poor"
    reasons: List[str]
    # Camera-position suitability for reading SHOTS (a stricter question than "is the clip usable"):
    hand_visibility: float = 0.0     # share of frames with both wrists confidently seen
    leg_separation: float = 0.0      # median ankle gap / person height (near 0 = legs overlap, e.g. filmed from behind)
    shot_reading_ok: bool = False
    shot_reading_notes: List[str] = field(default_factory=list)


def _heights_by_frame(landmarks):
    return [((f[LEFT_ANKLE][1] + f[RIGHT_ANKLE][1]) / 2.0) - f[NOSE][1] for f in landmarks]


def assess_footage(
    landmarks: Sequence[FrameLandmarks], frames_total: int, frames_expected: int = 0,
) -> FootageReport:
    """`landmarks` is one skeleton per frame that had a person (the same
    list the rest of the pipeline uses); `frames_total` is how many frames
    the clip had, so dropouts are visible. `frames_expected`, if known, is
    what the video container claims to hold: if far fewer frames could be
    decoded than that, the file is partly unreadable on this machine (a
    codec problem) and every count downstream is silently truncated — which
    is worth saying out loud, because nothing else would."""
    if frames_total <= 0 or not landmarks:
        return FootageReport(
            frames_total=max(frames_total, 0), frames_with_person=0, detection_rate=0.0,
            person_height_fraction=0.0, median_visibility=0.0, frames_expected=frames_expected,
            verdict="poor",
            reasons=["No person was detected in any frame."],
        )

    heights = []
    visibilities = []
    for frame in landmarks:
        ankle_y = (frame[LEFT_ANKLE][1] + frame[RIGHT_ANKLE][1]) / 2.0
        height = ankle_y - frame[NOSE][1]
        if height > 0:
            heights.append(height)
        visibilities.append(statistics.median(frame[j][2] for j in KEY_JOINTS))

    detection_rate = len(landmarks) / frames_total
    person_fraction = statistics.median(heights) if heights else 0.0
    visibility = statistics.median(visibilities)

    reasons: List[str] = []
    levels = []

    if frames_expected and frames_total < UNREADABLE_FRACTION * frames_expected:
        levels.append(2)
        reasons.append(
            f"Only {frames_total} of the {frames_expected} frames this video holds could be read on "
            "this machine — the file may be partly unreadable here (a codec issue), so anything "
            "after the point it stopped was never analysed and delivery counts will be too low."
        )

    if person_fraction < POOR_PERSON_FRACTION:
        levels.append(2)
        reasons.append(
            f"The tracked person is only {person_fraction * 100:.0f}% of the frame's height — "
            "too small for pose estimates to be reliable. Film closer or zoom in."
        )
    elif person_fraction < MARGINAL_PERSON_FRACTION:
        levels.append(1)
        reasons.append(f"The tracked person is small ({person_fraction * 100:.0f}% of the frame's height).")

    if visibility < POOR_VISIBILITY:
        levels.append(2)
        reasons.append(f"The pose model is unsure of the key joints (median confidence {visibility:.2f}).")
    elif visibility < MARGINAL_VISIBILITY:
        levels.append(1)
        reasons.append(f"The pose model's confidence in the key joints is middling ({visibility:.2f}).")

    if detection_rate < POOR_DETECTION_RATE:
        levels.append(2)
        reasons.append(f"A person was found in only {detection_rate * 100:.0f}% of frames.")
    elif detection_rate < MARGINAL_DETECTION_RATE:
        levels.append(1)
        reasons.append(f"A person was found in {detection_rate * 100:.0f}% of frames.")

    worst = max(levels) if levels else 0

    wrist_ok = sum(1 for f in landmarks if f[15][2] >= POOR_VISIBILITY and f[16][2] >= POOR_VISIBILITY)
    hand_visibility = wrist_ok / len(landmarks)
    gaps = [abs(f[LEFT_ANKLE][0] - f[RIGHT_ANKLE][0]) / h for f, h in zip(landmarks, _heights_by_frame(landmarks)) if h > 0]
    leg_separation = statistics.median(gaps) if gaps else 0.0
    notes: List[str] = []
    if worst == 2:
        notes.append("The footage itself is poor (see above).")
    if hand_visibility < MIN_HAND_VISIBILITY:
        notes.append(
            f"Both hands were confidently seen in only {hand_visibility * 100:.0f}% of frames "
            f"(need {MIN_HAND_VISIBILITY * 100:.0f}%+) — the swing's shape cannot be read from this camera position."
        )
    if person_fraction < MARGINAL_PERSON_FRACTION:
        notes.append("The batter is too small in frame to read the swing; move the camera closer or zoom.")
    if leg_separation < MIN_LEG_SEPARATION:
        notes.append(
            "The two legs overlap from this angle (typical when filmed from behind), so front/back foot "
            "cannot be read."
        )
    shot_ok = worst < 2 and hand_visibility >= MIN_HAND_VISIBILITY and person_fraction >= MARGINAL_PERSON_FRACTION
    return FootageReport(
        frames_total=frames_total,
        frames_with_person=len(landmarks),
        detection_rate=detection_rate,
        person_height_fraction=person_fraction,
        median_visibility=visibility,
        frames_expected=frames_expected,
        verdict=("good", "marginal", "poor")[worst],
        reasons=reasons or ["Person size, pose confidence and detection rate all look fine."],
        hand_visibility=hand_visibility, leg_separation=leg_separation,
        shot_reading_ok=shot_ok, shot_reading_notes=notes,
    )

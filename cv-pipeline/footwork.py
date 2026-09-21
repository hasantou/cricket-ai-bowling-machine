"""
Footwork from pose: which foot is the front foot, where it stepped, how far,
when — and whether the weight went forward or back.

Everything is measured relative to the batter's own STANCE (where each ankle
sat before the movement began) and in TORSO-LENGTHS, so camera distance doesn't
matter, and on camera-stabilised coordinates so a panning camera doesn't fake a
step. "Toward the bowler" is a direction on screen that depends on where the
camera stood, so the caller states it; from the front or behind, that is mostly
DEPTH, which a single camera sees only as a small vertical shift and a change
of size — so the thresholds for those views are lower, and the result says the
distance is foreshortened rather than pretending it is exact.

What is reported:
  * front foot (the left foot of a right-hander, the right foot of a left-hander)
    and back foot, with their stance positions and their positions at the swing
    peak, so the step can be shown on the picture;
  * front-foot stride toward the bowler, the back foot's shift, and the hips'
    shift (weight transfer), all in torso-lengths;
  * lead time: how long before the swing peak the stride began;
  * a class: "front-foot", "back-foot", "minimal" or "unclear (feet not visible)".

Limits, stated:
  * The class is a rule of thumb (named thresholds below), not fitted to labelled
    footwork; expect errors near the thresholds.
  * The ankle must be confidently seen: a joint the model could not see well
    makes the class "unclear" rather than a guess. Filmed from behind, legs often
    overlap and this is the usual outcome.
  * It does not know where the stumps or crease are, so it says how the feet
    moved relative to the batter's own stance, not "outside off stump".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from body_vectors import (
    JOINTS, MAX_BRIDGE_SECONDS, MIN_VISIBILITY, PARTIAL_FRACTION, RELIABLE_FRACTION, SMOOTH_SECONDS,
    _bridge, _series, _smooth, _torso_scale,
)

BEHIND_BATTER = "behind the batter"
BOWLERS_END = "bowler's end (facing the batter)"
SIDE_ON = "side-on"
RIGHT_HANDED, LEFT_HANDED = "right-handed", "left-handed"
BOWLER_ON_LEFT, BOWLER_ON_RIGHT = "bowler on the left of the frame", "bowler on the right of the frame"

STANCE_SECONDS = 0.30         # the ankle's stance position = its median over this much of the window's start
STRIDE_START = 0.12           # a foot has started to step once it is this far (torso-lengths) from its stance
# Thresholds per view (torso-lengths). Depth is foreshortened front-on/behind, so those are lower.
FRONT_STRIDE_MIN = {BEHIND_BATTER: 0.12, BOWLERS_END: 0.12, SIDE_ON: 0.30}
BACK_SHIFT_MIN = {BEHIND_BATTER: 0.10, BOWLERS_END: 0.10, SIDE_ON: 0.25}
HIP_BACK_MIN = {BEHIND_BATTER: 0.08, BOWLERS_END: 0.08, SIDE_ON: 0.20}

FRONT, BACK, MINIMAL, UNCLEAR = "front-foot", "back-foot", "minimal", "unclear (feet not visible)"


@dataclass(frozen=True)
class FootworkReport:
    footwork_class: str
    front_ankle: str                                  # joint name of the front foot
    front_stride: float                               # torso-lengths toward the bowler at the swing peak (+ = toward)
    front_across: float                               # sideways travel of the front foot (screen +right), torso-lengths
    back_shift: float                                 # back foot's shift toward the bowler (- = moved back)
    hip_shift: float                                  # hip centre toward the bowler (- = weight went back)
    lead_time_s: Optional[float]                      # seconds the front-foot stride began before the swing peak
    front_start: Optional[Tuple[float, float]]        # stance and peak positions, normalised (stabilised) image coords
    front_at_peak: Optional[Tuple[float, float]]
    back_start: Optional[Tuple[float, float]]
    back_at_peak: Optional[Tuple[float, float]]
    ankle_confidence: Dict[str, str]
    foreshortened: bool                               # front-on/behind: forward distance is under-measured
    notes: List[str] = field(default_factory=list)

    @property
    def observed_foot(self) -> Optional[str]:
        """"front foot" / "back foot" for the shot naming, or None if not established."""
        return {FRONT: "front foot", BACK: "back foot"}.get(self.footwork_class)


def toward_bowler(camera: str, bowler_side: Optional[str] = None) -> Optional[Tuple[float, float]]:
    """Unit vector on screen (x right, y DOWN as in image coordinates) pointing toward the bowler."""
    if camera == BEHIND_BATTER:
        return (0.0, -1.0)            # the bowler is further away = higher in the picture
    if camera == BOWLERS_END:
        return (0.0, 1.0)             # the batter faces the camera; toward the bowler = toward the camera = lower
    if camera == SIDE_ON:
        if bowler_side == BOWLER_ON_LEFT:
            return (-1.0, 0.0)
        if bowler_side == BOWLER_ON_RIGHT:
            return (1.0, 0.0)
    return None


def _confidence(trusted_share: float) -> str:
    return "good" if trusted_share >= RELIABLE_FRACTION else ("partial" if trusted_share >= PARTIAL_FRACTION else "poor")


def analyse_footwork(
    window: Sequence[Optional[list]], fps: float, aspect: float, peak_frame: int, camera: str,
    hand: str = RIGHT_HANDED, bowler_side: Optional[str] = None,
) -> Optional[FootworkReport]:
    """`window[i]` is frame i's (stabilised) landmarks or None; `peak_frame` is the index of the
    swing peak within the window (BodyVectorReport.peak_frame). Returns None if the direction
    toward the bowler is unknown (side-on with no bowler side given) or there is too little data."""
    u = toward_bowler(camera, bowler_side)
    n = len(window)
    if u is None or n < 5 or _torso_scale(window, aspect) is None:
        return None
    scale = _torso_scale(window, aspect)
    front_name, back_name = ("left ankle", "right ankle") if hand == RIGHT_HANDED else ("right ankle", "left ankle")
    half = max(1, int(round(SMOOTH_SECONDS * fps / 2)))
    gap = max(1, int(round(MAX_BRIDGE_SECONDS * fps)))

    tracks, conf = {}, {}
    for name in (front_name, back_name):
        pos, trusted = _series(window, JOINTS[name], aspect)
        conf[name] = _confidence(float(trusted.sum()) / n)
        tracks[name] = _smooth(_bridge(pos, gap), half)
    hip_l, _ = _series(window, JOINTS["left hip"], aspect)
    hip_r, _ = _series(window, JOINTS["right hip"], aspect)
    hips = _smooth(_bridge((hip_l + hip_r) / 2.0, gap), half)

    notes: List[str] = []
    foreshortened = camera != SIDE_ON
    if foreshortened:
        notes.append("Filmed front-on or from behind, a step toward the bowler is mostly depth, which one camera sees "
                     "only as a small shift; distances here are under-measured and the thresholds are set lower.")
    if conf[front_name] == "poor" or conf[back_name] == "poor":
        notes.append("A foot could not be seen well by the pose model (often the legs overlap from this angle), so "
                     "the footwork is not judged.")
        return FootworkReport(UNCLEAR, front_name, 0.0, 0.0, 0.0, 0.0, None, None, None, None, None, conf, foreshortened, notes)

    stance_n = max(2, int(round(STANCE_SECONDS * fps)))

    def stance(track):
        seg = track[:stance_n]
        seg = seg[~np.isnan(seg[:, 0])]
        return np.median(seg, axis=0) if len(seg) else None

    peak = min(max(peak_frame, 0), n - 1)
    ud = np.array(u)

    def along(track, s0):
        """Per-frame displacement from the stance, torso-lengths: (toward the bowler, across screen-x)."""
        d = (track - s0) / scale
        return d @ ud, d[:, 0]

    s_front, s_back, s_hip = stance(tracks[front_name]), stance(tracks[back_name]), stance(hips)
    if s_front is None or s_back is None or s_hip is None:
        notes.append("Not enough of the batter's stance was visible to measure movement from it.")
        return FootworkReport(UNCLEAR, front_name, 0.0, 0.0, 0.0, 0.0, None, None, None, None, None, conf, foreshortened, notes)

    f_fwd, f_across = along(tracks[front_name], s_front)
    b_fwd, _ = along(tracks[back_name], s_back)
    h_fwd, _ = along(hips, s_hip)

    def at(series, i):
        v = series[i]
        return float(v) if not np.isnan(v) else 0.0

    front_stride, front_across, back_shift, hip_shift = at(f_fwd, peak), at(f_across, peak), at(b_fwd, peak), at(h_fwd, peak)

    lead = None
    moved = np.where(np.abs(np.nan_to_num(f_fwd[: peak + 1])) >= STRIDE_START)[0]
    if len(moved):
        lead = (peak - int(moved[0])) / fps

    def norm(track, i):
        p = track[i]
        return None if np.isnan(p).any() else (float(p[0] / aspect), float(p[1]))

    front_min, back_min, hip_min = FRONT_STRIDE_MIN[camera], BACK_SHIFT_MIN[camera], HIP_BACK_MIN[camera]
    if front_stride >= front_min and front_stride >= back_shift:
        cls = FRONT
    elif (back_shift <= -back_min or hip_shift <= -hip_min) and front_stride < front_min:
        cls = BACK
    else:
        cls = MINIMAL
    if "partial" in conf.values():
        notes.append("A foot was only partly visible; treat the class as a rough reading.")
    return FootworkReport(
        cls, front_name, front_stride, front_across, back_shift, hip_shift, lead,
        (float(s_front[0] / aspect), float(s_front[1])),
        norm(tracks[front_name], peak),
        (float(s_back[0] / aspect), float(s_back[1])), norm(tracks[back_name], peak),
        conf, foreshortened, notes,
    )

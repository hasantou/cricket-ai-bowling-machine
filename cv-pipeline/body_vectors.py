"""
Body-movement vectors: where each joint is moving, and how fast, through a
delivery — the numbers behind an arrow drawn on a paused frame.

For every tracked joint this produces a velocity VECTOR (direction + speed),
and from those a small report on the delivery: the peak-swing moment, the
vector of every joint at that moment, the hands' path through the swing, how
the hips shifted (weight transfer) and how the feet moved.

Three things were learned the hard way on real footage and shape the design:

  1. Raw pose landmarks jitter. Frame-to-frame differences of raw wrist
     positions were dominated by noise (a batter merely taking guard scored
     as high as a real swing). So each joint's track is smoothed, and
     low-confidence points are dropped and (briefly) bridged, before any
     velocity is taken.
  2. Distance from the camera must not matter, so everything is measured in
     TORSO-LENGTHS (the person's own scale) and per second, with the frame's
     aspect ratio corrected so a "horizontal" move isn't stretched.
  3. Some joints simply aren't visible from some angles (filmed from behind,
     both legs land on one column; a hidden arm comes back at 0.2-0.4
     confidence). Every vector carries a `reliable` flag from the model's own
     confidence, and callers should not read meaning into the unreliable ones.

What this is not: the vectors are 2-D, in the image plane of ONE camera — a
"rightward" hand is rightward on screen, and what that means for the cricket
(off side or leg side, a drive or a cut) depends on where the camera stood
and which hand the batter uses. This module reports the geometry; it does not
turn it into a shot name (see cv-pipeline/README.md for why that was not
reliable on real footage).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

JOINTS: Dict[str, int] = {
    "left shoulder": 11, "right shoulder": 12,
    "left elbow": 13, "right elbow": 14,
    "left wrist": 15, "right wrist": 16,
    "left hip": 23, "right hip": 24,
    "left knee": 25, "right knee": 26,
    "left ankle": 27, "right ankle": 28,
}
LEFT_WRIST, RIGHT_WRIST = "left wrist", "right wrist"

MIN_VISIBILITY = 0.5          # a joint point below this model confidence is not trusted
RELIABLE_FRACTION = 0.6       # 'good': trusted in at least this share of the window
PARTIAL_FRACTION = 0.3        # 'partial': trusted often enough to draw, not to lean on; below this, 'poor'
SMOOTH_SECONDS = 0.10         # centred moving-average window
MAX_BRIDGE_SECONDS = 0.20     # gaps up to this long are interpolated across; longer stay missing
MIN_PERSON_SECONDS = 0.30     # need at least this much tracked person to say anything
SWING_MIN_SPEED = 4.0         # torso-lengths/s: below this the batter did not swing (judgment call from 4 real windows: swings 7.5-9.4, non-swings 1-2)
MAX_PLAUSIBLE_SPEED = 25.0    # torso-lengths/s: a hand faster than this is a pose glitch, not a swing (found on real footage: 36 and 60)
SWING_EDGE_FRACTION = 0.25    # the swing phase runs while the hand is above this share of its peak speed

Landmarks = Sequence[Tuple[float, float, float]]


@dataclass(frozen=True)
class JointVector:
    vx: float                    # torso-lengths/s, +right on screen
    vy: float                    # torso-lengths/s, +up on screen
    speed: float                 # torso-lengths/s
    angle_deg: float             # 0 = right, 90 = up, 180 = left, 270 = down (on screen)
    direction: str               # eight-way compass on screen, e.g. "up-right"
    confidence: str              # "good" | "partial" | "poor" — from the model's own confidence over the window

    @property
    def reliable(self) -> bool:
        return self.confidence == "good"


@dataclass(frozen=True)
class BodyVectorReport:
    peak_frame: int                              # index into the window given
    peak_hand: str                               # which wrist was fastest, or "trunk" if no wrist could be trusted
    peak_hand_speed: float                       # torso-lengths/s, relative to the body (hip centre)
    vectors_at_peak: Dict[str, JointVector]
    hand_path_start: int
    hand_path_end: int
    hand_path_length: float                      # torso-lengths travelled by the fast hand in the swing phase
    hand_path_net: Tuple[float, float]           # (dx, dy up) net displacement start->end, torso-lengths
    hand_path_net_direction: str
    hand_path_points: List[Tuple[float, float]]  # smoothed path, normalised image coords, for drawing
    hip_shift: Tuple[float, float]               # (dx, dy up) of hip centre over the window, torso-lengths
    ankle_shift: Dict[str, Tuple[float, float]]  # (dx, dy up) per ankle over the window
    shoulder_line_change_deg: float              # rotation of the shoulder line, start -> peak (image plane)
    hip_line_change_deg: float
    body_speed_at_peak: Optional[float]          # hip-centre speed at the peak, torso-lengths/s (whole-body motion)
    tracked_fraction: float                      # share of the window's frames with a person
    caveats: List[str] = field(default_factory=list)

    @property
    def speed_plausible(self) -> bool:
        return self.peak_hand_speed <= MAX_PLAUSIBLE_SPEED

    @property
    def swing_verdict(self) -> str:
        """"swing" / "no swing" / "unclear" — from the peak hand (or trunk) speed
        against two uncalibrated thresholds (SWING_MIN_SPEED, MAX_PLAUSIBLE_SPEED),
        set from a handful of real windows. Not a validated classifier."""
        if not self.speed_plausible:
            return "unclear"
        return "swing" if self.peak_hand_speed >= SWING_MIN_SPEED else "no swing"


_COMPASS = ["right", "up-right", "up", "up-left", "left", "down-left", "down", "down-right"]


def compass(angle_deg: float) -> str:
    return _COMPASS[int(((angle_deg % 360) + 22.5) // 45) % 8]


def _angle(dx: float, dy_up: float) -> float:
    return math.degrees(math.atan2(dy_up, dx)) % 360.0


def _series(window: Sequence[Optional[Landmarks]], idx: int, aspect: float) -> Tuple[np.ndarray, np.ndarray]:
    """(N,2) isotropic positions and (N,) trusted mask for one joint; untrusted -> NaN."""
    n = len(window)
    pos = np.full((n, 2), np.nan)
    trusted = np.zeros(n, dtype=bool)
    for i, frame in enumerate(window):
        if frame is None:
            continue
        x, y, vis = frame[idx]
        if vis >= MIN_VISIBILITY:
            pos[i] = (x * aspect, y)
            trusted[i] = True
    return pos, trusted


def _bridge(pos: np.ndarray, max_gap: int) -> np.ndarray:
    """Linear interpolation across NaN gaps no longer than max_gap frames."""
    out = pos.copy()
    n = len(out)
    valid = np.where(~np.isnan(out[:, 0]))[0]
    for a, b in zip(valid[:-1], valid[1:]):
        gap = b - a - 1
        if 0 < gap <= max_gap:
            for k in range(2):
                out[a + 1:b, k] = np.linspace(out[a, k], out[b, k], gap + 2)[1:-1]
    return out


def _smooth(pos: np.ndarray, half: int) -> np.ndarray:
    """Centred moving average that ignores NaNs (a point with no neighbours stays NaN)."""
    n = len(pos)
    out = np.full_like(pos, np.nan)
    for i in range(n):
        seg = pos[max(0, i - half): i + half + 1]
        ok = ~np.isnan(seg[:, 0])
        if ok.any():
            out[i] = seg[ok].mean(axis=0)
    return out


def _velocity(pos: np.ndarray, fps: float) -> np.ndarray:
    """Central-difference velocity (units/s), NaN where either neighbour is missing."""
    n = len(pos)
    vel = np.full_like(pos, np.nan)
    for i in range(n):
        a, b = max(0, i - 1), min(n - 1, i + 1)
        if b > a and not (np.isnan(pos[a]).any() or np.isnan(pos[b]).any()):
            vel[i] = (pos[b] - pos[a]) * fps / (b - a)
    return vel


def _torso_scale(window: Sequence[Optional[Landmarks]], aspect: float) -> Optional[float]:
    lengths = []
    for frame in window:
        if frame is None:
            continue
        need = (frame[11], frame[12], frame[23], frame[24])
        if min(p[2] for p in need) < MIN_VISIBILITY:
            continue
        sx = (frame[11][0] + frame[12][0]) / 2 * aspect
        sy = (frame[11][1] + frame[12][1]) / 2
        hx = (frame[23][0] + frame[24][0]) / 2 * aspect
        hy = (frame[23][1] + frame[24][1]) / 2
        lengths.append(math.hypot(sx - hx, sy - hy))
    return float(np.median(lengths)) if lengths else None


def analyse_body_vectors(
    window: Sequence[Optional[Landmarks]], fps: float, aspect: float = 1.0,
) -> Optional[BodyVectorReport]:
    """`window[i]` is the landmarks of frame i (None where no person was found —
    frame positions are kept so time stays true). `aspect` is frame width /
    height. Returns None if there isn't enough tracked person to say anything."""
    n = len(window)
    tracked = sum(1 for f in window if f is not None)
    if n < 3 or tracked < MIN_PERSON_SECONDS * fps:
        return None
    scale = _torso_scale(window, aspect)
    if scale is None or scale < 1e-6:
        return None

    half = max(1, int(round(SMOOTH_SECONDS * fps / 2)))
    max_gap = max(1, int(round(MAX_BRIDGE_SECONDS * fps)))
    smooth: Dict[str, np.ndarray] = {}
    vel: Dict[str, np.ndarray] = {}
    level: Dict[str, str] = {}
    for name, idx in JOINTS.items():
        pos, trusted = _series(window, idx, aspect)
        share = trusted.sum() / n
        level[name] = "good" if share >= RELIABLE_FRACTION else ("partial" if share >= PARTIAL_FRACTION else "poor")
        sm = _smooth(_bridge(pos, max_gap), half)
        smooth[name] = sm
        vel[name] = _velocity(sm, fps) / scale            # torso-lengths per second

    # The whole body can move without any swing (walking in, stepping out of
    # frame, a panning camera) — every joint then shares one velocity. A swing is
    # the hands moving RELATIVE TO THE BODY, so measure against the hip centre.
    hips_ok = level["left hip"] != "poor" and level["right hip"] != "poor"
    if hips_ok:
        hip_c = (smooth["left hip"] + smooth["right hip"]) / 2.0
        hip_vel = _velocity(hip_c, fps) / scale
        body_speed = np.linalg.norm(hip_vel, axis=1)
    else:
        hip_vel, body_speed = None, None

    def relative(v: np.ndarray) -> np.ndarray:
        return v - hip_vel if hip_vel is not None else v

    # Peak swing: the fastest smoothed wrist (relative to the body) among those
    # the model can see at least partly. If neither wrist can be trusted (hidden
    # behind the body, blurred), fall back to the trunk — shoulder midpoint,
    # relative to the hips — and say so.
    candidates = [w for w in (LEFT_WRIST, RIGHT_WRIST) if level[w] != "poor"]
    best = (-1.0, -1, "")
    for w in candidates:
        speeds = np.linalg.norm(relative(vel[w]), axis=1)
        if np.isnan(speeds).all():
            continue
        i = int(np.nanargmax(speeds))
        if speeds[i] > best[0]:
            best = (float(speeds[i]), i, w)
    used_trunk = best[1] < 0
    if used_trunk:
        if level["left shoulder"] == "poor" or level["right shoulder"] == "poor":
            return None
        smooth["trunk"] = (smooth["left shoulder"] + smooth["right shoulder"]) / 2.0
        vel["trunk"] = _velocity(smooth["trunk"], fps) / scale
        speeds = np.linalg.norm(relative(vel["trunk"]), axis=1)
        if np.isnan(speeds).all():
            return None
        best = (float(np.nanmax(speeds)), int(np.nanargmax(speeds)), "trunk")
    peak_speed, peak, hand = best

    vectors: Dict[str, JointVector] = {}
    for name in JOINTS:
        v = vel[name][peak]
        if np.isnan(v).any():
            continue
        vx, vy_up = float(v[0]), float(-v[1])
        vectors[name] = JointVector(
            vx=vx, vy=vy_up, speed=math.hypot(vx, vy_up), angle_deg=_angle(vx, vy_up),
            direction=compass(_angle(vx, vy_up)), confidence=level[name],
        )

    # Hand path through the swing phase.
    speeds = np.linalg.norm(relative(vel[hand]), axis=1)
    edge = SWING_EDGE_FRACTION * peak_speed
    start = peak
    while start > 0 and not np.isnan(speeds[start - 1]) and speeds[start - 1] >= edge:
        start -= 1
    end = peak
    while end < n - 1 and not np.isnan(speeds[end + 1]) and speeds[end + 1] >= edge:
        end += 1
    path = smooth[hand][start:end + 1]
    rel_path = path - hip_c[start:end + 1] if hips_ok else path          # the hands' path relative to the body
    keep = ~np.isnan(rel_path[:, 0])
    path, rel_path = path[keep], rel_path[keep]
    length = float(np.sum(np.linalg.norm(np.diff(rel_path, axis=0), axis=1)) / scale) if len(rel_path) > 1 else 0.0
    net = (rel_path[-1] - rel_path[0]) / scale if len(rel_path) > 1 else np.zeros(2)
    net_dx, net_dy_up = float(net[0]), float(-net[1])

    def shift(names: Tuple[str, ...]) -> Tuple[float, float]:
        pts = np.mean([smooth[nm] for nm in names], axis=0)
        ok = pts[~np.isnan(pts[:, 0])]
        if len(ok) < 2:
            return (0.0, 0.0)
        d = (ok[-1] - ok[0]) / scale
        return (float(d[0]), float(-d[1]))

    def line_change(a: str, b: str) -> float:
        p_a, p_b = smooth[a], smooth[b]
        def ang(i):
            d = p_b[i] - p_a[i]
            return math.degrees(math.atan2(-d[1], d[0])) if not np.isnan(d).any() else None
        first = next((ang(i) for i in range(n) if ang(i) is not None), None)
        at_peak = ang(peak)
        if first is None or at_peak is None:
            return 0.0
        return (at_peak - first + 180.0) % 360.0 - 180.0

    caveats = [
        "Vectors are 2-D in one camera's image plane: 'right' means right on screen, not off/leg side.",
        "Speeds are torso-lengths per second, smoothed; not km/h.",
    ]
    weak = sorted(nm for nm, lv in level.items() if lv != "good")
    if weak:
        caveats.append(
            "Joints the model could not see well (treat their arrows as rough or ignore them): "
            + ", ".join(f"{nm} ({level[nm]})" for nm in weak) + "."
        )
    if used_trunk:
        caveats.append(
            "Neither wrist could be tracked (hidden or blurred from this camera angle), so the swing "
            "moment and 'hand path' below are the TRUNK's movement, not the hands'."
        )
    if body_speed is not None and not np.isnan(body_speed[peak]) and body_speed[peak] > SWING_MIN_SPEED:
        caveats.append(
            f"The whole body was moving at {body_speed[peak]:.1f} torso-lengths/s at the peak (walking, stepping "
            "out of frame or a moving camera). The swing figure above is the hands' speed relative to the body."
        )
    if peak_speed > MAX_PLAUSIBLE_SPEED:
        caveats.append(
            f"Peak speed {peak_speed:.0f} torso-lengths/s is faster than a human hand can move — "
            "almost certainly a pose-tracking glitch, so this delivery's swing verdict is 'unclear'."
        )
    if tracked < 0.8 * n:
        caveats.append(f"A person was found in only {tracked * 100 // n}% of this window's frames.")

    return BodyVectorReport(
        peak_frame=peak, peak_hand=hand, peak_hand_speed=peak_speed, vectors_at_peak=vectors,
        hand_path_start=start, hand_path_end=end, hand_path_length=length,
        hand_path_points=[(float(x / aspect), float(y)) for x, y in path],
        hand_path_net=(net_dx, net_dy_up), hand_path_net_direction=compass(_angle(net_dx, net_dy_up)),
        hip_shift=shift(("left hip", "right hip")),
        ankle_shift={"left ankle": shift(("left ankle",)), "right ankle": shift(("right ankle",))},
        shoulder_line_change_deg=line_change("left shoulder", "right shoulder"),
        hip_line_change_deg=line_change("left hip", "right hip"),
        body_speed_at_peak=(float(body_speed[peak]) if body_speed is not None and not np.isnan(body_speed[peak]) else None),
        tracked_fraction=tracked / n, caveats=caveats,
    )

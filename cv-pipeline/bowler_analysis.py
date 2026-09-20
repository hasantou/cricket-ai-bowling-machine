"""
Reads a bowler's ACTION from body pose — arm side, release height, arm
angle, arm speed and run-up pace — for footage where the bowler is
actually in shot.

What this is and is not:
  - It measures the bowler's BODY (large, trackable), never the ball. It
    cannot say how fast the ball went, where it pitched, or how much it
    swung — a phone clip can't show that reliably (see README.md's
    ball-tracking findings). For a machine, that's what the sensors and the
    physics-based delivery report are for.
  - Every number is an estimate from 2D pose landmarks in ONE camera view.
    Arm angle in particular is a projection: it reads very differently
    from behind the bowler than from square-on, so treat the label as a
    coarse hint, not a biomechanics measurement.
  - Speeds are in torso-lengths per second, not km/h: a single uncalibrated
    camera has no scale, but a person's own torso is a scale they carry
    with them. It still shrinks/grows with distance to the camera.

Finding who the bowler is: with several people in shot the pose model
returns several skeletons per frame, in no reliable order. People are
followed frame to frame by hip position, then the bowler is the one who
both travels (a run-up) and raises an arm above their head (a delivery
swing), scored below with named, uncalibrated thresholds. If nobody clears
the bar, this returns None rather than guessing — a missing bowler is a
correct answer for footage from behind the bowler's end.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from feature_extraction import FrameLandmarks

NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

# --- tracking ---
MAX_TRACK_JUMP = 0.12            # normalised image units a person may move between frames
# --- who is the bowler ---
MIN_FRAMES_TRACKED = 12          # need at least this many frames to judge a track
MIN_ARM_RAISE = 0.5              # wrist must get this far above the shoulder (in torso-lengths)
MIN_TRAVEL_TORSOS = 1.5          # hips must travel at least this far (in torso-lengths)
MAX_RELEASE_TORSO_TILT_DEG = 50.0  # a release happens with the torso roughly upright, not bent double
MAX_PLAUSIBLE_RAISE = 1.6        # a wrist can't sit further than ~an arm's length + margin above its shoulder;
                                 # beyond this the skeleton is a glitch (found on real footage: 11-15 torso-lengths,
                                 # from a degenerate near-zero torso length), so the frame is ignored
# --- action labels, from the projected arm angle to vertical ---
HIGH_ARM_MAX_DEG = 25.0
THREE_QUARTER_MAX_DEG = 55.0
ROUND_ARM_MAX_DEG = 80.0
BATTER_EXCLUSION_RADIUS = 0.10   # a track whose typical hip position is this close to the known batter IS the batter
# --- measurement windows ---
RUN_UP_SECONDS = 1.0             # how far back from release to measure run-up pace
ARM_SPEED_WINDOW_SECONDS = 0.15  # either side of release


@dataclass(frozen=True)
class BowlerActionEstimate:
    arm_side: str                     # "right-arm" / "left-arm" (the bowler's own sides)
    arm_angle_deg: float              # projected angle of the bowling arm from vertical at release
    arm_action_label: str
    release_frame: int                # index into the frames given
    release_height_torsos: float       # wrist height above the shoulder at release, in torso-lengths
    run_up_speed_torsos_per_s: float      # hip travel over the last second, torso-lengths per second
    arm_speed_torsos_per_s: float         # peak bowling-wrist speed around release, torso-lengths per second
    release_to_swing_s: Optional[float]   # release -> the batter's swing peak, if the batter's frame is known
    frames_tracked: int
    caveats: Tuple[str, ...]


def _xy(frame: FrameLandmarks, idx: int) -> Tuple[float, float]:
    return frame[idx][0], frame[idx][1]


def hip_center(frame: FrameLandmarks) -> Tuple[float, float]:
    (lx, ly), (rx, ry) = _xy(frame, LEFT_HIP), _xy(frame, RIGHT_HIP)
    return (lx + rx) / 2.0, (ly + ry) / 2.0


def _mid(frame: FrameLandmarks, a: int, b: int) -> Tuple[float, float]:
    return (frame[a][0] + frame[b][0]) / 2.0, (frame[a][1] + frame[b][1]) / 2.0


def torso_length(frame: FrameLandmarks) -> float:
    """Shoulder-midpoint to hip-midpoint distance in the image plane — the
    person's own scale. Deliberately NOT nose-to-ankle height: that shrinks
    to almost nothing when a bowler bends double in the follow-through
    (found the hard way on real footage), whereas an in-plane torso length
    survives bending."""
    (sx, sy), (hx, hy) = _mid(frame, LEFT_SHOULDER, RIGHT_SHOULDER), hip_center(frame)
    return math.hypot(sx - hx, sy - hy)


def torso_tilt_deg(frame: FrameLandmarks) -> float:
    """How far the hip->shoulder axis leans from straight up (0 = upright,
    90 = horizontal, >90 = upside down) in the image plane."""
    (sx, sy), (hx, hy) = _mid(frame, LEFT_SHOULDER, RIGHT_SHOULDER), hip_center(frame)
    upward = hy - sy                      # image y grows downward
    return math.degrees(math.atan2(abs(sx - hx), upward)) if upward > 0 else 90.0 + math.degrees(
        math.atan2(-upward, max(abs(sx - hx), 1e-9))
    )


def arm_raise(frame: FrameLandmarks, wrist_idx: int) -> float:
    """How far a wrist is above its own shoulder, in torso-lengths. A hand
    straight overhead is roughly 1; hands at the side are negative."""
    torso = torso_length(frame)
    if torso < 1e-6:
        return -1.0
    shoulder_idx = LEFT_SHOULDER if wrist_idx == LEFT_WRIST else RIGHT_SHOULDER
    return (frame[shoulder_idx][1] - frame[wrist_idx][1]) / torso


@dataclass
class PersonTrack:
    frames: Dict[int, FrameLandmarks]   # frame index -> that person's skeleton

    def indices(self) -> List[int]:
        return sorted(self.frames)


def track_people(frames_poses: Sequence[Sequence[FrameLandmarks]]) -> List[PersonTrack]:
    """Greedy nearest-hip tracking of every skeleton across frames. A skeleton
    joins the closest existing track that was seen within the last few
    frames and is within MAX_TRACK_JUMP; otherwise it starts a new track."""
    tracks: List[PersonTrack] = []
    last_seen: List[Tuple[int, Tuple[float, float]]] = []
    for frame_index, poses in enumerate(frames_poses):
        claimed = set()
        for pose in poses:
            center = hip_center(pose)
            best, best_dist = None, MAX_TRACK_JUMP
            for t, (seen_at, seen_center) in enumerate(last_seen):
                if t in claimed or frame_index - seen_at > 5:
                    continue
                dist = math.hypot(center[0] - seen_center[0], center[1] - seen_center[1])
                if dist < best_dist:
                    best, best_dist = t, dist
            if best is None:
                tracks.append(PersonTrack(frames={}))
                last_seen.append((frame_index, center))
                best = len(tracks) - 1
            claimed.add(best)
            tracks[best].frames[frame_index] = pose
            last_seen[best] = (frame_index, center)
    return tracks


def _median_hip(track: PersonTrack) -> Tuple[float, float]:
    centers = [hip_center(p) for p in track.frames.values()]
    xs, ys = sorted(c[0] for c in centers), sorted(c[1] for c in centers)
    return xs[len(xs) // 2], ys[len(ys) // 2]


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _travel_torsos(track: PersonTrack) -> float:
    idx = track.indices()
    total = 0.0
    for a, b in zip(idx, idx[1:]):
        ca, cb = hip_center(track.frames[a]), hip_center(track.frames[b])
        total += math.hypot(cb[0] - ca[0], cb[1] - ca[1])
    torsos = [torso_length(track.frames[i]) for i in idx]
    mean_torso = sum(torsos) / len(torsos)
    return total / mean_torso if mean_torso > 1e-6 else 0.0


def _peak_raise(track: PersonTrack) -> Tuple[float, int, int]:
    """(best raise, frame index, wrist index) over the track — considering
    only frames where the torso is roughly upright, so a hand swept up
    behind a bowler bent double in the follow-through can't pass for a
    release."""
    best = (-1e9, -1, RIGHT_WRIST)
    for i, pose in track.frames.items():
        if torso_tilt_deg(pose) > MAX_RELEASE_TORSO_TILT_DEG:
            continue
        for wrist in (LEFT_WRIST, RIGHT_WRIST):
            raise_ = arm_raise(pose, wrist)
            if raise_ > MAX_PLAUSIBLE_RAISE:
                continue
            if raise_ > best[0]:
                best = (raise_, i, wrist)
    return best


def identify_bowler(tracks: Sequence[PersonTrack]) -> Optional[PersonTrack]:
    """The track that both runs and delivers: hips travel AND a hand goes
    above the head. None if nobody clears both bars."""
    best_track, best_score = None, 0.0
    for track in tracks:
        if len(track.frames) < MIN_FRAMES_TRACKED:
            continue
        raise_, _, _ = _peak_raise(track)
        travel = _travel_torsos(track)
        if raise_ < MIN_ARM_RAISE or travel < MIN_TRAVEL_TORSOS:
            continue
        score = travel + 2.0 * raise_
        if score > best_score:
            best_track, best_score = track, score
    return best_track


def _label_for_angle(angle_deg: float) -> str:
    if angle_deg <= HIGH_ARM_MAX_DEG:
        return "high, near-vertical arm"
    if angle_deg <= THREE_QUARTER_MAX_DEG:
        return "three-quarter arm"
    if angle_deg <= ROUND_ARM_MAX_DEG:
        return "round-arm"
    return "low / sidearm"


def analyse_bowler_action(
    frames_poses: Sequence[Sequence[FrameLandmarks]],
    fps: float,
    batter_swing_frame: Optional[int] = None,
    batter_hip: Optional[Tuple[float, float]] = None,
) -> Optional[BowlerActionEstimate]:
    """`frames_poses[i]` is every skeleton found in frame i (the window of a
    single delivery). `batter_hip`, if known, is the batter's typical hip
    position (normalised image coords): whoever is tracked there is the
    batter and is never a bowler candidate — found on real footage, where a
    batter stepping out and lifting the bat overhead passed for a bowler
    (pose alone can't tell a backlift from a delivery). Returns None if no
    track looks like a bowler."""
    tracks = track_people(frames_poses)
    if batter_hip is not None:
        tracks = [t for t in tracks if _dist(_median_hip(t), batter_hip) > BATTER_EXCLUSION_RADIUS]
    bowler = identify_bowler(tracks)
    if bowler is None:
        return None

    _, release_frame, wrist_idx = _peak_raise(bowler)
    pose = bowler.frames[release_frame]
    shoulder_idx = LEFT_SHOULDER if wrist_idx == LEFT_WRIST else RIGHT_SHOULDER
    sx, sy = _xy(pose, shoulder_idx)
    wx, wy = _xy(pose, wrist_idx)
    # Angle between the arm (shoulder->wrist) and the torso's own up axis
    # (hips->shoulders), so a bowler leaning sideways at release isn't
    # misread as having a low arm.
    mid_sx, mid_sy = _mid(pose, LEFT_SHOULDER, RIGHT_SHOULDER)
    hx, hy = hip_center(pose)
    arm_vec, torso_vec = (wx - sx, wy - sy), (mid_sx - hx, mid_sy - hy)
    dot = arm_vec[0] * torso_vec[0] + arm_vec[1] * torso_vec[1]
    norm = math.hypot(*arm_vec) * math.hypot(*torso_vec)
    angle = math.degrees(math.acos(max(-1.0, min(1.0, dot / norm)))) if norm > 1e-9 else 90.0

    indices = bowler.indices()
    torsos = [torso_length(bowler.frames[i]) for i in indices]
    mean_height = sum(torsos) / len(torsos)      # (name kept: it's the scale unit)

    run_start = release_frame - int(RUN_UP_SECONDS * fps)
    run_indices = [i for i in indices if run_start <= i <= release_frame]
    run_up = 0.0
    if len(run_indices) >= 2 and mean_height > 1e-6:
        path = sum(
            math.hypot(
                hip_center(bowler.frames[b])[0] - hip_center(bowler.frames[a])[0],
                hip_center(bowler.frames[b])[1] - hip_center(bowler.frames[a])[1],
            )
            for a, b in zip(run_indices, run_indices[1:])
        )
        elapsed = (run_indices[-1] - run_indices[0]) / fps
        run_up = (path / mean_height) / elapsed if elapsed > 0 else 0.0

    half = max(1, int(ARM_SPEED_WINDOW_SECONDS * fps))
    arm_speed = 0.0
    arm_indices = [i for i in indices if release_frame - half <= i <= release_frame + half]
    for a, b in zip(arm_indices, arm_indices[1:]):
        dt = (b - a) / fps
        if dt <= 0 or mean_height <= 1e-6:
            continue
        pa, pb = _xy(bowler.frames[a], wrist_idx), _xy(bowler.frames[b], wrist_idx)
        arm_speed = max(arm_speed, math.hypot(pb[0] - pa[0], pb[1] - pa[1]) / mean_height / dt)

    caveats = [
        "Arm angle is a 2D projection from one camera and shifts with camera angle.",
        "Speeds are in torso-lengths per second (no real-world scale from one camera).",
    ]
    if len(bowler.frames) < 0.6 * len(frames_poses):
        caveats.append("The bowler was tracked for well under the whole window — some values rest on few frames.")

    return BowlerActionEstimate(
        arm_side="left-arm" if wrist_idx == LEFT_WRIST else "right-arm",
        arm_angle_deg=angle,
        arm_action_label=_label_for_angle(angle),
        release_frame=release_frame,
        release_height_torsos=arm_raise(pose, wrist_idx),
        run_up_speed_torsos_per_s=run_up,
        arm_speed_torsos_per_s=arm_speed,
        release_to_swing_s=(
            (batter_swing_frame - release_frame) / fps if batter_swing_frame is not None else None
        ),
        frames_tracked=len(bowler.frames),
        caveats=tuple(caveats),
    )

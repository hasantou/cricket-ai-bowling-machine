"""
Pure, deterministic feature-extraction math over a sequence of MediaPipe
pose landmarks for one batting delivery. Fully unit-testable without a
camera, video file, or the pose-detection model itself — feed it
synthetic landmark sequences and check the numbers, exactly the same
"test the logic honestly before real data exists" pattern used by
adaptation-engine/simulator.py for the decision engine.

Landmark indexing follows MediaPipe's 33-point BlazePose topology.
Assumes a side-on camera view of a right-handed batter (front foot =
left foot, leading/top hand = left wrist) — the common setup for
batting-analysis footage. Left-handed batters and other camera angles
are real future work, not solved here, and are called out again in
README.md so nobody mistakes this for finished.

`footwork_lead_seconds` — how long before the bat-swing peak the front
foot started moving — exists to give outcome_bridge.py's `on_time` check
something to measure that isn't defined relative to a clip/window's own
length. An earlier version compared swing timing against the whole
clip's length, which real footage exposed as circular once
delivery_segmentation.py started building windows centred on the swing
itself (see that module's docstring). This is still an unvalidated
heuristic, not a ground truth — there's no actual ball-arrival time to
compare against without ball tracking, which isn't built.
"""

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

# MediaPipe BlazePose landmark indices we use.
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 31, 32

Landmark = Tuple[float, float, float]  # (x, y, visibility) in normalised image coords
FrameLandmarks = Sequence[Landmark]     # 33 landmarks for one frame


@dataclass
class DeliveryFeatures:
    front_foot_displacement: float  # how far the front foot moved during the clip (normalised units)
    front_knee_bend_deg: float      # front knee angle at the peak-movement frame; smaller = more bent
    swing_peak_speed: float         # fastest frame-to-frame leading-wrist movement in the clip
    swing_start_frame: int          # first frame where wrist speed crosses the "swing began" threshold
    footwork_start_frame: int       # first frame where front-foot speed crosses the "feet began moving" threshold
    footwork_lead_seconds: float    # (swing_start_frame - footwork_start_frame) / fps; positive = feet moved before the swing
    n_frames: int


def _point(frame: FrameLandmarks, idx: int) -> Tuple[float, float]:
    return frame[idx][0], frame[idx][1]


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle_deg(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """Angle at point b, formed by segments b->a and b->c, in degrees."""
    ab = (a[0] - b[0], a[1] - b[1])
    cb = (c[0] - b[0], c[1] - b[1])
    mag_ab, mag_cb = math.hypot(*ab), math.hypot(*cb)
    if mag_ab == 0 or mag_cb == 0:
        return 180.0
    cos_angle = max(-1.0, min(1.0, (ab[0] * cb[0] + ab[1] * cb[1]) / (mag_ab * mag_cb)))
    return math.degrees(math.acos(cos_angle))


def extract_delivery_features(
    frames: List[FrameLandmarks],
    fps: float = 30.0,
    swing_speed_threshold: float = 0.02,
    foot_speed_threshold: float = 0.015,
) -> DeliveryFeatures:
    if not frames:
        raise ValueError("extract_delivery_features requires at least one frame")

    foot_positions = [_point(f, LEFT_FOOT_INDEX) for f in frames]
    front_foot_displacement = max((_dist(foot_positions[0], p) for p in foot_positions), default=0.0)
    foot_speeds = [_dist(foot_positions[i - 1], foot_positions[i]) for i in range(1, len(foot_positions))]
    footwork_start_frame = next(
        (i + 1 for i, s in enumerate(foot_speeds) if s >= foot_speed_threshold),
        len(frames) - 1,
    )

    wrist_positions = [_point(f, LEFT_WRIST) for f in frames]
    frame_speeds = [_dist(wrist_positions[i - 1], wrist_positions[i]) for i in range(1, len(wrist_positions))]
    swing_peak_speed = max(frame_speeds, default=0.0)
    swing_start_frame = next(
        (i + 1 for i, s in enumerate(frame_speeds) if s >= swing_speed_threshold),
        len(frames) - 1,
    )
    footwork_lead_seconds = (swing_start_frame - footwork_start_frame) / fps

    peak_idx = (frame_speeds.index(swing_peak_speed) + 1) if frame_speeds else 0
    peak_frame = frames[peak_idx]
    knee_angle = _angle_deg(
        _point(peak_frame, LEFT_HIP), _point(peak_frame, LEFT_KNEE), _point(peak_frame, LEFT_ANKLE)
    )

    return DeliveryFeatures(
        front_foot_displacement=front_foot_displacement,
        front_knee_bend_deg=knee_angle,
        swing_peak_speed=swing_peak_speed,
        swing_start_frame=swing_start_frame,
        footwork_start_frame=footwork_start_frame,
        footwork_lead_seconds=footwork_lead_seconds,
        n_frames=len(frames),
    )

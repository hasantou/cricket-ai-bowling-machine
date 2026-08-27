"""
Pure feature-computation functions over body-pose landmarks — derived
batting-stance/footwork signals used to feed the mastery scorer.

Deliberately decoupled from mediapipe: these functions take plain
(x, y, z) landmark coordinates, not mediapipe's own types, so they're
unit-testable with synthetic coordinates and don't need a model, a video,
or a network connection to test.

Two honest caveats, stated up front:
- "Backlift" here is a hand-height proxy (wrist height relative to
  shoulders, normalised by torso length), not real bat tracking — body
  pose alone can't see the bat. Real backlift measurement needs bat/object
  tracking, which is separate, later work.
- Landmarks are reported as left/right, not front/back leg — knowing
  which leg is "front" needs the batter's handedness and stance
  orientation, which pose estimation alone doesn't give us.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional

LEFT_SHOULDER, RIGHT_SHOULDER = "left_shoulder", "right_shoulder"
LEFT_HIP, RIGHT_HIP = "left_hip", "right_hip"
LEFT_KNEE, RIGHT_KNEE = "left_knee", "right_knee"
LEFT_ANKLE, RIGHT_ANKLE = "left_ankle", "right_ankle"
LEFT_WRIST, RIGHT_WRIST = "left_wrist", "right_wrist"

REQUIRED_LANDMARKS = [
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_WRIST, RIGHT_WRIST,
]


@dataclass(frozen=True)
class Point:
    x: float
    y: float
    z: float = 0.0


def _dist(a: Point, b: Point) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _midpoint(a: Point, b: Point) -> Point:
    return Point((a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2)


def _angle_deg(a: Point, vertex: Point, b: Point) -> float:
    """Angle in degrees at `vertex`, between rays toward `a` and `b`."""
    v1 = (a.x - vertex.x, a.y - vertex.y, a.z - vertex.z)
    v2 = (b.x - vertex.x, b.y - vertex.y, b.z - vertex.z)
    dot = sum(p * q for p, q in zip(v1, v2))
    mag1 = math.sqrt(sum(p * p for p in v1))
    mag2 = math.sqrt(sum(p * p for p in v2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_angle))


def has_required_landmarks(landmarks: Dict[str, Point]) -> bool:
    return all(name in landmarks for name in REQUIRED_LANDMARKS)


def stance_width_ratio(landmarks: Dict[str, Point]) -> float:
    """Ankle-to-ankle distance / shoulder-to-shoulder distance. Scale-
    invariant (a batter closer to or further from the camera reads the
    same), higher = wider stance."""
    ankle_width = _dist(landmarks[LEFT_ANKLE], landmarks[RIGHT_ANKLE])
    shoulder_width = _dist(landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER])
    if shoulder_width == 0:
        raise ValueError("Degenerate pose: zero shoulder width")
    return ankle_width / shoulder_width


def knee_flex_deg(landmarks: Dict[str, Point], side: str) -> float:
    """Knee bend angle (hip-knee-ankle) in degrees for `side` ('left' or
    'right'). 180 = fully straight leg, smaller = more bent."""
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    hip = landmarks[f"{side}_hip"]
    knee = landmarks[f"{side}_knee"]
    ankle = landmarks[f"{side}_ankle"]
    return _angle_deg(hip, knee, ankle)


def hand_height_ratio(landmarks: Dict[str, Point]) -> float:
    """Proxy for backlift: how far above the shoulders the hands sit,
    normalised by torso length (shoulder-midpoint to hip-midpoint
    distance). Positive = hands above shoulders, negative = hands below.
    NOT real bat tracking — see module docstring."""
    shoulder_mid = _midpoint(landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER])
    hip_mid = _midpoint(landmarks[LEFT_HIP], landmarks[RIGHT_HIP])
    wrist_mid = _midpoint(landmarks[LEFT_WRIST], landmarks[RIGHT_WRIST])

    torso_length = _dist(shoulder_mid, hip_mid)
    if torso_length == 0:
        raise ValueError("Degenerate pose: zero torso length")

    # Image y grows downward, so "above" means a smaller y value.
    return (shoulder_mid.y - wrist_mid.y) / torso_length


def compute_frame_features(landmarks: Optional[Dict[str, Point]]) -> Optional[dict]:
    """All derived features for one frame's landmarks, or None if no
    person was detected or required landmarks are missing."""
    if not landmarks or not has_required_landmarks(landmarks):
        return None
    return {
        "stance_width_ratio": stance_width_ratio(landmarks),
        "left_knee_flex_deg": knee_flex_deg(landmarks, "left"),
        "right_knee_flex_deg": knee_flex_deg(landmarks, "right"),
        "hand_height_ratio": hand_height_ratio(landmarks),
    }

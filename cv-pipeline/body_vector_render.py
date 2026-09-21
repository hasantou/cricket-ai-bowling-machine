"""
Draw a delivery's body vectors on a video frame: the picture behind the
numbers in body_vectors.py.

Each joint gets an arrow showing where it is heading over the next
LOOKAHEAD_SECONDS at its current velocity (so a fast hand has a long arrow),
the fast hand's path through the swing is drawn as a trail, and a joint the
model could not see well is drawn grey with no arrow — a confident-looking
arrow on a joint the model was guessing at would be a small lie.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from body_vectors import JOINTS, BodyVectorReport

LOOKAHEAD_SECONDS = 0.15
GOOD, PARTIAL, POOR = (60, 200, 60), (0, 165, 255), (150, 150, 150)   # BGR
TRAIL = (255, 120, 0)
BONES = [
    ("left shoulder", "right shoulder"), ("left shoulder", "left elbow"), ("left elbow", "left wrist"),
    ("right shoulder", "right elbow"), ("right elbow", "right wrist"), ("left shoulder", "left hip"),
    ("right shoulder", "right hip"), ("left hip", "right hip"), ("left hip", "left knee"),
    ("left knee", "left ankle"), ("right hip", "right knee"), ("right knee", "right ankle"),
]


def _px(lm, idx, w, h) -> Tuple[int, int]:
    return int(lm[idx][0] * w), int(lm[idx][1] * h)


def arrow_tip(origin: Tuple[int, int], vx: float, vy_up: float, torso_px: float) -> Tuple[int, int]:
    """Where a joint at `origin` (pixels) is heading after LOOKAHEAD_SECONDS at
    velocity (vx, vy_up) torso-lengths/s. Screen y grows downward, so 'up' subtracts."""
    return (
        int(round(origin[0] + vx * torso_px * LOOKAHEAD_SECONDS)),
        int(round(origin[1] - vy_up * torso_px * LOOKAHEAD_SECONDS)),
    )


def draw_body_vectors(
    frame_bgr: np.ndarray, landmarks, report: BodyVectorReport, torso_px: Optional[float] = None,
    offset: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Returns a copy of `frame_bgr` with the skeleton, per-joint vector arrows
    (as at the report's peak moment) and the hand-path trail drawn on it.
    `landmarks` are the landmarks of that peak frame, in image coordinates. If the
    report was measured on camera-stabilised coordinates, `offset` is the camera path at
    that frame (normalised), added back so the hand-path trail sits on the picture."""
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    if torso_px is None:
        sx = (landmarks[11][0] + landmarks[12][0]) / 2 * w
        sy = (landmarks[11][1] + landmarks[12][1]) / 2 * h
        hx = (landmarks[23][0] + landmarks[24][0]) / 2 * w
        hy = (landmarks[23][1] + landmarks[24][1]) / 2 * h
        torso_px = math.hypot(sx - hx, sy - hy)
    thickness = max(2, int(round(w / 320)))

    colour = {}
    for name in JOINTS:
        v = report.vectors_at_peak.get(name)
        colour[name] = POOR if v is None else {"good": GOOD, "partial": PARTIAL}.get(v.confidence, POOR)

    for a, b in BONES:
        cv2.line(img, _px(landmarks, JOINTS[a], w, h), _px(landmarks, JOINTS[b], w, h), (230, 230, 230), 1, cv2.LINE_AA)

    if len(report.hand_path_points) > 1:
        pts = np.array([(int((x + offset[0]) * w), int((y + offset[1]) * h)) for x, y in report.hand_path_points], np.int32)
        cv2.polylines(img, [pts], False, TRAIL, thickness, cv2.LINE_AA)

    for name, idx in JOINTS.items():
        origin = _px(landmarks, idx, w, h)
        cv2.circle(img, origin, thickness + 1, colour[name], -1, cv2.LINE_AA)
        v = report.vectors_at_peak.get(name)
        if v is None or v.confidence == "poor" or v.speed < 0.3:
            continue
        tip = arrow_tip(origin, v.vx, v.vy, torso_px)
        cv2.arrowedLine(img, origin, tip, colour[name], thickness, cv2.LINE_AA, tipLength=0.3)
    return img


def to_jpeg(img: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("could not encode image")
    return buf.tobytes()

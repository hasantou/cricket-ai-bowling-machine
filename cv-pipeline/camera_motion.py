"""
Camera-motion compensation: subtract the camera's own movement from where
people appear to be.

Broadcast and phone footage pans, tilts and shakes. Pose landmarks are
positions in the IMAGE, so when the camera pans, a person standing still
appears to run across the frame. Anything that measures travel or speed from
image positions — a bowler's run-up pace, weight shift, "who is moving" — is
then wrong by the camera's speed. (On real broadcast clips this produced bowler
arm speeds of 60-120 body-lengths/s, which are impossible.)

Method: between consecutive frames, follow up to a couple of hundred textured
points (Lucas-Kanade optical flow) and take the MEDIAN motion. People are a
minority of the picture, so the median is the background, i.e. the camera; a
median-absolute-deviation filter drops points that moved differently (a
running fielder). The per-frame shifts are summed into a camera path, and
`stabilize` subtracts it from landmarks so a person who is stationary in the
world has constant coordinates.

Limits, stated:
  * Translation only (pan/tilt/shake). A zoom changes scale, which this does
    not model; a zoom in or out leaves a residual error that grows with the
    person's distance from the image centre.
  * At an edit cut the camera path is meaningless, so the path is reset there
    (positions before and after a cut are not comparable and are not compared).
  * With too little background texture (a plain sky, a blurred frame) the shift
    for that frame is unknown and is treated as zero; the count of such frames
    is reported so it is visible rather than silent.
  * Nothing here is applied when the camera is essentially still, so tripod
    footage is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

FLOW_WIDTH = 240                 # frames are shrunk to this width for flow (fast; landmarks are normalised anyway)
MAX_POINTS = 200
MIN_POINTS = 12                  # fewer tracked background points than this = shift unknown for that frame
MAD_K = 3.0                      # points further than this many MADs from the median flow are not background
MOVING_RANGE = 0.03              # the camera "moves" if its path spans more than this fraction of the frame
MIN_MAD = 0.0005                 # floor for the outlier filter (in normalised units)


class CameraMotionTracker:
    """Feed it frames in order; it returns each frame's shift relative to the
    previous one as (dx, dy) in NORMALISED image units (fractions of width and
    height), or None if it could not be measured."""

    def __init__(self):
        self._prev = None
        self.unknown_frames = 0

    def push(self, frame_bgr: np.ndarray, reset: bool = False) -> Optional[Tuple[float, float]]:
        import cv2
        h, w = frame_bgr.shape[:2]
        scale = FLOW_WIDTH / w if w > FLOW_WIDTH else 1.0
        small = cv2.resize(frame_bgr, (max(2, int(w * scale)), max(2, int(h * scale))), interpolation=cv2.INTER_AREA) \
            if scale != 1.0 else frame_bgr
        grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        prev, self._prev = self._prev, grey
        if prev is None or reset or prev.shape != grey.shape:
            return (0.0, 0.0) if prev is None or reset else None
        pts = cv2.goodFeaturesToTrack(prev, maxCorners=MAX_POINTS, qualityLevel=0.01, minDistance=6)
        if pts is None or len(pts) < MIN_POINTS:
            self.unknown_frames += 1
            return None
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev, grey, pts, None, winSize=(21, 21), maxLevel=3)
        ok = status.reshape(-1) == 1
        if ok.sum() < MIN_POINTS:
            self.unknown_frames += 1
            return None
        flow = (nxt.reshape(-1, 2)[ok] - pts.reshape(-1, 2)[ok])
        flow[:, 0] /= grey.shape[1]
        flow[:, 1] /= grey.shape[0]
        med = np.median(flow, axis=0)
        mad = np.maximum(np.median(np.abs(flow - med), axis=0), MIN_MAD)
        keep = np.all(np.abs(flow - med) <= MAD_K * mad, axis=1)
        if keep.sum() < MIN_POINTS:
            self.unknown_frames += 1
            return None
        dx, dy = np.median(flow[keep], axis=0)
        return float(dx), float(dy)


@dataclass
class CameraPath:
    """Cumulative background displacement per frame (normalised units, frame 0 = 0,0)."""
    x: List[float]
    y: List[float]
    unknown_frames: int = 0
    moving: bool = False
    range_x: float = 0.0
    range_y: float = 0.0
    notes: List[str] = field(default_factory=list)

    def at(self, i: int) -> Tuple[float, float]:
        return self.x[min(max(i, 0), len(self.x) - 1)], self.y[min(max(i, 0), len(self.y) - 1)]


def build_path(shifts: Sequence[Optional[Tuple[float, float]]], cut_frames: Sequence[int] = ()) -> CameraPath:
    """Sum per-frame shifts into a path. The path restarts at each cut (new shot)."""
    cuts = set(cut_frames)
    xs, ys = [0.0], [0.0]
    unknown = 0
    for i, shift in enumerate(shifts):
        if i == 0:
            continue
        if i in cuts:
            xs.append(xs[-1]); ys.append(ys[-1])       # no continuity across a cut
            continue
        if shift is None:
            unknown += 1
            shift = (0.0, 0.0)
        xs.append(xs[-1] + shift[0]); ys.append(ys[-1] + shift[1])
    rx, ry = (max(xs) - min(xs)), (max(ys) - min(ys))
    moving = max(rx, ry) > MOVING_RANGE
    notes = []
    if moving:
        notes.append(
            f"The camera moved (path spans {rx * 100:.0f}% of the frame width, {ry * 100:.0f}% of the height); "
            "positions were corrected for it."
        )
    if unknown and unknown > 0.1 * max(1, len(shifts)):
        notes.append(f"Camera motion could not be measured on {unknown} of {len(shifts)} frames (too little background texture).")
    return CameraPath(xs, ys, unknown, moving, rx, ry, notes)


def stabilize_pose(pose, cam_x: float, cam_y: float):
    """A pose's landmarks with the camera's accumulated displacement removed."""
    return [(x - cam_x, y - cam_y, v) for (x, y, v) in pose]


def stabilize(aligned: Sequence[Optional[list]], path: CameraPath) -> List[Optional[list]]:
    """`aligned[i]` is frame i's landmarks (or None). Returns the same list with
    the camera path removed — unchanged (the same objects) if the camera did not move."""
    if not path.moving:
        return list(aligned)
    return [None if pose is None else stabilize_pose(pose, *path.at(i)) for i, pose in enumerate(aligned)]

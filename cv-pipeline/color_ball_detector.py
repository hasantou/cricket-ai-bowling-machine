"""
A colour-based ball-candidate detector: HSV thresholding for the ball's dark red, plus size and
circularity filtering — no motion, no background model, no training data.

Why this is a SEPARATE module from motion_ball_detector.py, not a tweak to it: that module's honest
negative finding (cv-pipeline/README.md) was on nets footage where the ball is small, distant, and
similar in tone to the surroundings — colour alone was never going to separate it there. This exists
because different real footage (close broadcast crops, found 2026-09-23) showed a ball with strong,
distinct hue and saturation against green outfield or a tan pitch — a genuinely different regime,
worth a genuinely different detector, not a parameter tweak to the motion-based one.

Measured directly off a real frame (a probe crop centred on a visually confirmed ball position):
HSV hue 1-32 (red wraps through 0/180 in OpenCV's 0-179 hue range), saturation 87-171, value 66-130.
The default range below is that measurement with margin, NOT tuned against a labelled dataset — a
real red cricket ball can shift with lighting, shine, and camera white balance, so treat the defaults
as a starting point to check on new footage, the same honesty as motion_ball_detector.py's own
untrained thresholds.

What this does NOT solve: white-ball cricket footage (different colour entirely — this red-hued
default would find approximately nothing, correctly), a ball in shadow, or a ball against a
similarly-hued background (a red kit, an advertising board, skin tone can all land in this hue
range — size and circularity filtering catches some but not all of that, exactly the same
precision caveat motion_ball_detector.py states for its own false positives).

TESTED AGAINST REAL FOOTAGE, HONEST RESULT: on a real frame with the ball's position confirmed by
eye (a close broadcast crop, bat-pad height), a first hue-only threshold covering the ball's own
measured range (0-32) came back with the ball's entire 40x40-pixel surroundings masked "red" —
the batter's hand and the bat's grip sit in the same hue band at that closeness, so hue alone
merges them into one blob and no ball-sized candidate survives the size/circularity filter.
Tightening on SATURATION (the ball's own pixels measured 137-158, clearly above the hand/grip in
the same frame) narrowed the merged region but still did not isolate a clean, correctly-positioned
candidate — the closest surviving blob was still visually large and 16px from the true position.
**Colour alone does not reliably separate this ball from nearby skin and equipment at this
distance**, on the one real frame this was checked against. The synthetic tests below confirm the
method works correctly on the case it CAN solve (a clearly isolated red circle on a plain
background) — the gap is real footage's proximity and colour overlap, not a bug in the matching
logic itself. Worth revisiting with a stricter saturation floor per-clip (lighting varies) or a
small, targeted classifier over color-mask candidates, not a larger hand-tuned threshold sweep.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

# OpenCV hue is 0-179. Cricket-ball red wraps around 0, so two bands are combined.
HUE_LOW_MAX = 25          # 0.. this
HUE_HIGH_MIN = 165        # this ..179
SAT_MIN = 110             # the ball's own leather measured 137-158 on a real frame; skin and the bat's
                          # wooden/rubber grip sit lower — a first, WIDER hue-only threshold (up to 35,
                          # saturation 70+) merged the ball, the batter's hand and the grip into one
                          # blob (a 40x40 probe centred on the ball came back 100% "red"), so a real
                          # frame's core-pixel saturation is what actually separates them, not hue.
VAL_MIN = 50
VAL_MAX = 150             # measured 66-132 on the same real ball; excludes bright highlights/kit

MIN_RADIUS_PX = 2.0
MAX_RADIUS_PX = 22.0
MIN_CIRCULARITY = 0.55


@dataclass(frozen=True)
class ColorCandidate:
    frame_index: int
    x_px: float
    y_px: float
    radius_px: float
    circularity: float
    mean_hue: float           # for a human to sanity-check a candidate against the measured real range


def _red_mask(frame_bgr: np.ndarray, hue_low_max: int, hue_high_min: int, sat_min: int, val_min: int, val_max: int) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower = cv2.inRange(hsv, (0, sat_min, val_min), (hue_low_max, 255, val_max))
    upper = cv2.inRange(hsv, (hue_high_min, sat_min, val_min), (179, 255, val_max))
    return cv2.bitwise_or(lower, upper)


def detect_ball_candidates_by_color(
    video_path: str,
    hue_low_max: int = HUE_LOW_MAX, hue_high_min: int = HUE_HIGH_MIN,
    sat_min: int = SAT_MIN, val_min: int = VAL_MIN, val_max: int = VAL_MAX,
    min_radius_px: float = MIN_RADIUS_PX, max_radius_px: float = MAX_RADIUS_PX,
    min_circularity: float = MIN_CIRCULARITY,
    roi: Optional[Tuple[int, int, int, int]] = None,
    max_frames: Optional[int] = None,
) -> List[ColorCandidate]:
    """Same candidate-per-frame contract as motion_ball_detector.detect_ball_candidates(): every
    frame's surviving blobs, unfiltered further across time — ball_tracking.fit_ball_trajectory()
    (RANSAC over the whole clip) is what turns a noisy candidate stream like this into one
    trajectory, exactly as it already does for the motion-based candidates."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    candidates: List[ColorCandidate] = []
    frame_index = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    try:
        while max_frames is None or frame_index < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if roi is not None:
                x_min, y_min, x_max, y_max = roi
                frame = frame[y_min:y_max, x_min:x_max]
            else:
                x_min = y_min = 0

            mask = _red_mask(frame, hue_low_max, hue_high_min, sat_min, val_min, val_max)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, closed=True)
                if area <= 0 or perimeter <= 0:
                    continue
                (x, y), radius = cv2.minEnclosingCircle(contour)
                if not (min_radius_px <= radius <= max_radius_px):
                    continue
                circularity = 4 * 3.141592653589793 * area / (perimeter ** 2)
                if circularity < min_circularity:
                    continue
                blob_mask = np.zeros(mask.shape, np.uint8)
                cv2.drawContours(blob_mask, [contour], -1, 255, -1)
                mean_hue = float(cv2.mean(hsv[:, :, 0], mask=blob_mask)[0])
                candidates.append(ColorCandidate(frame_index, x + x_min, y + y_min, radius, circularity, mean_hue))
            frame_index += 1
    finally:
        cap.release()
    return candidates

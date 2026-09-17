"""
A classical, untrained ball-candidate detector: background subtraction +
blob filtering, not a neural network. Exists because the earlier YOLOv8
detector needed labelled training data to reach even ~12% recall, and
that data-hungry path has a real cost (hours of training, a dataset that
doesn't match nets footage well). This needs no training data at all —
just a plausible size/shape for the moving object — so it can be run
directly against real footage today and honestly evaluated, rather than
only against synthetic data like ball_tracking.py's own tests are.

The trade continuous background subtraction makes: it flags *anything*
small and moving, which includes real ball candidates but also insects,
rustling net material, or noise — precision is unvalidated and likely
not great. That's fine for this repo's actual use: ball_tracking.py's
RANSAC fit already exists specifically to reject exactly this kind of
outlier-contaminated input (see its own tests). This module's job is
only to get *some* real candidates in front of that fit; it is not
claimed to be a clean detector on its own.

Tested against two real WhatsApp nets clips — an honest negative
result, not a working detector:

- Camera shake was ruled out first (measured via cv2.phaseCorrelate:
  mean frame-to-frame shift 0.2px, max 0.66px — negligible), so it isn't
  the cause of anything below.
- At default sensitivity, most flagged candidates traced back to the
  net mesh flickering against the sky (wind + video-compression noise on
  fine repeating texture), not the ball — confirmed by drawing every
  candidate on real frames and looking at them.
- Raising `var_threshold` well past default (up to 200) cut candidate
  count roughly 6x but never reached zero, and a manually re-inspected
  survivor at that setting was a player's hand, not the ball — false
  positives, not a cleanly isolated true one.
- Directly hunting for the ball itself — zoomed crops of the flight
  corridor during the release window on both clips, plus a frame-diff
  peak trace outside the bowler's own body region on the closer-camera
  clip — never conclusively found it. One trace that looked like smooth,
  continuous real motion turned out, on closer zoom, to be the batter's
  own glove/bat shifting, not the ball.

Conclusion this module's own real-footage testing supports: on this
footage, the ball is at or below the visibility floor — small, motion
blurred, and further degraded by video compression — for a human
reviewer as much as for this algorithm. That points at the capture
setup (camera distance/zoom, shutter speed, ball-background contrast)
as the actual blocker, not detector choice. See cv-pipeline/README.md
for the full writeup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2

from ball_tracking import BallDetection


@dataclass
class MotionCandidate:
    """One blob flagged by background subtraction as plausibly ball-sized
    and ball-shaped. `circularity` is 1.0 for a perfect circle (4*pi*area /
    perimeter^2); real players' limbs and torsos score much lower, which
    is the main signal this module uses to tell a ball from a person."""
    frame_index: int
    x_px: float
    y_px: float
    radius_px: float
    circularity: float


def detect_ball_candidates(
    video_path: str,
    min_radius_px: float = 2.0,
    max_radius_px: float = 20.0,
    min_circularity: float = 0.55,
    history: int = 50,
    var_threshold: float = 24.0,
) -> List[MotionCandidate]:
    """Runs MOG2 background subtraction over the whole clip and keeps
    only foreground blobs that are both small (a person or their limbs
    should be much larger than `max_radius_px` at any normal filming
    distance) and round (`min_circularity`) — the two properties that
    plausibly distinguish a cricket ball from everything else that moves
    in a nets session: players, their shadows, netting in the wind.

    Returns every frame's surviving candidates, unfiltered further —
    there may be 0, 1, or several per frame. Downstream code (e.g.
    ball_tracking.fit_ball_trajectory) is what actually decides which
    candidates, across the whole clip, form one consistent trajectory.
    """
    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=history, varThreshold=var_threshold, detectShadows=False,
    )
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    candidates: List[MotionCandidate] = []
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            mask = subtractor.apply(frame)
            # A closed morphology pass merges a ball's often-fragmented
            # foreground pixels (motion blur, compression) into one blob
            # without also merging in unrelated nearby noise.
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

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
                candidates.append(MotionCandidate(frame_index, x, y, radius, circularity))
            frame_index += 1
    finally:
        cap.release()
    return candidates


def to_ball_detections(candidates: List[MotionCandidate]) -> List[BallDetection]:
    """Bridges to ball_tracking.py's input type. When a frame has more
    than one surviving candidate, keeps only the most circular one —
    fit_ball_trajectory takes one detection per frame, and circularity is
    this module's only real confidence signal."""
    best_per_frame = {}
    for c in candidates:
        current = best_per_frame.get(c.frame_index)
        if current is None or c.circularity > current.circularity:
            best_per_frame[c.frame_index] = c
    return [
        BallDetection(c.frame_index, c.x_px, c.y_px, confidence=c.circularity)
        for c in sorted(best_per_frame.values(), key=lambda c: c.frame_index)
    ]

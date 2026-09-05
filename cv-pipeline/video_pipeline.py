"""
End to end: a video file in, (on_time, footwork_correct) out per delivery
detected — the same shape scoring.py and neural_scorer.py already accept
from a human's checkboxes in app.py. estimate_outcomes_from_video() is
the function app.py's "upload a clip" mode calls.

The video does not need to be pre-trimmed to one delivery, and doesn't
need to contain only one — real footage tested against this pipeline ran
6 to 71 seconds long (chunks of a nets session, sometimes with more than
one ball in frame), so delivery_segmentation.py finds every delivery it
can inside whatever's uploaded before the feature math runs on each one.
See that module's docstring for how "a delivery" is detected and its
real limits.

estimate_outcome_from_video() (singular) is kept for callers that only
want the single clearest delivery in a clip.

Requires pose_estimation.download_model() to have been run once (see
that module's docstring for what it fetches and why it's a deliberate,
explicit step rather than automatic).
"""

from typing import List

import cv2

from delivery_segmentation import find_delivery_windows
from outcome_bridge import VisionOutcomeEstimate, VisionOutcomeEstimator
from pose_estimation import PoseEstimator


def _read_frames(video_path: str):
    cap = cv2.VideoCapture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        return frames, fps
    finally:
        cap.release()


def estimate_outcomes_from_video(
    video_path: str, estimator: PoseEstimator = None, max_deliveries: int = None
) -> List[VisionOutcomeEstimate]:
    """Returns one VisionOutcomeEstimate per delivery detected in the
    clip, in the order they occur (not ranked by how clear the swing
    was). Raises ValueError if no person is detected anywhere, or if no
    delivery is detected at all (see
    delivery_segmentation.find_delivery_windows)."""
    owns_estimator = estimator is None
    estimator = estimator or PoseEstimator()
    try:
        frames, fps = _read_frames(video_path)
        landmarks = estimator.extract_landmarks_from_frames(frames, fps)
        if not landmarks:
            raise ValueError(
                f"No person detected in any frame of {video_path} — check the clip actually "
                "shows a batter in frame, or that the model downloaded correctly."
            )
        windows = find_delivery_windows(landmarks, fps, max_deliveries=max_deliveries)
        vision_estimator = VisionOutcomeEstimator()
        return [vision_estimator.estimate(landmarks[start:end], fps=fps) for start, end in windows]
    finally:
        if owns_estimator:
            estimator.close()


def estimate_outcome_from_video(video_path: str, estimator: PoseEstimator = None) -> VisionOutcomeEstimate:
    """Single-delivery convenience wrapper: the single clearest delivery
    (strongest swing) in the clip — matches delivery_segmentation.
    find_delivery_window()'s selection, not necessarily the first one in
    time. Prefer estimate_outcomes_from_video() for clips that might
    contain more than one — real nets-session footage usually does."""
    return estimate_outcomes_from_video(video_path, estimator=estimator, max_deliveries=1)[0]

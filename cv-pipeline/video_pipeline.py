"""
End to end: a video file in, (on_time, footwork_correct) out — the same
shape scoring.py and neural_scorer.py already accept from a human's
checkboxes in app.py. This is the function app.py's "upload a clip" mode
calls.

The video does not need to be pre-trimmed to one delivery — real footage
tested against this pipeline ran 6 to 71 seconds long (a chunk of a nets
session, not an isolated ball), so delivery_segmentation.py finds the
single clearest delivery inside whatever's uploaded before the feature
math runs on it. See that module's docstring for what "clearest" means
and its real limits (one delivery per clip, not every delivery in a
multi-ball clip).

Requires pose_estimation.download_model() to have been run once (see
that module's docstring for what it fetches and why it's a deliberate,
explicit step rather than automatic).
"""

import cv2

from delivery_segmentation import find_delivery_window
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


def estimate_outcome_from_video(video_path: str, estimator: PoseEstimator = None) -> VisionOutcomeEstimate:
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
        start, end = find_delivery_window(landmarks, fps)
        return VisionOutcomeEstimator().estimate(landmarks[start:end])
    finally:
        if owns_estimator:
            estimator.close()

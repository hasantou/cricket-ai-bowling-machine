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

from dataclasses import dataclass
from typing import List, Optional

import cv2

from bowler_analysis import hip_center, BowlerActionEstimate, analyse_bowler_action
from delivery_segmentation import WINDOW_BEFORE_PEAK_SEC, find_delivery_windows
from footage_check import FootageReport, assess_footage
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


def _expected_frame_count(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
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


# The bowler runs in and releases BEFORE the batter's swing; the ball's
# flight (~0.4-0.7s) sits between the two. Look this far back from the
# batter's swing peak for the bowler's run-up and release, and this far
# past it in case the flight was short.
BOWLER_LOOKBACK_SECONDS = 3.0
BOWLER_LOOKAHEAD_SECONDS = 0.3
BOWLER_FRAME_STEP = 2   # analyse every 2nd frame - the bowler's motion is smooth, and it halves the cost


@dataclass
class VideoAnalysis:
    estimates: List[VisionOutcomeEstimate]                 # the batter, one per detected delivery
    footage: FootageReport                                 # is this clip good enough to trust?
    bowler_actions: List[Optional[BowlerActionEstimate]]   # aligned with `estimates`; None = no bowler found


def _typical_hip(window_landmarks):
    """Median hip position of the person the batter analysis followed in this
    delivery's window — the batter, by construction."""
    if not window_landmarks:
        return None
    centers = [hip_center(frame) for frame in window_landmarks]
    xs, ys = sorted(c[0] for c in centers), sorted(c[1] for c in centers)
    return xs[len(xs) // 2], ys[len(ys) // 2]


def analyse_video(
    video_path: str, estimator: PoseEstimator = None, analyse_bowler: bool = False,
    max_deliveries: int = None,
) -> VideoAnalysis:
    """Everything estimate_outcomes_from_video() does (the batter's
    footwork and timing per delivery — unchanged code path), plus a verdict
    on whether the footage is good enough to trust, plus — only if asked,
    since it runs a second, multi-person pose pass — the bowler's action
    for each delivery. A bowler who isn't in shot (or doesn't raise a hand
    overhead) comes back as None for that delivery, not a guess."""
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
        footage = assess_footage(landmarks, len(frames), frames_expected=_expected_frame_count(video_path))
        windows = find_delivery_windows(landmarks, fps, max_deliveries=max_deliveries)
        vision_estimator = VisionOutcomeEstimator()
        estimates = [vision_estimator.estimate(landmarks[start:end], fps=fps) for start, end in windows]

        bowler_actions: List[Optional[BowlerActionEstimate]] = [None] * len(windows)
        if analyse_bowler:
            eff_fps = fps / BOWLER_FRAME_STEP
            with PoseEstimator(num_poses=4) as multi:
                for k, (start, end) in enumerate(windows):
                    swing_frame = start + int(WINDOW_BEFORE_PEAK_SEC * fps)
                    lo = max(0, swing_frame - int(BOWLER_LOOKBACK_SECONDS * fps))
                    hi = min(len(frames), swing_frame + int(BOWLER_LOOKAHEAD_SECONDS * fps))
                    window_poses = [
                        multi.extract_all_poses_from_one_live_frame(frames[i], eff_fps)
                        for i in range(lo, hi, BOWLER_FRAME_STEP)
                    ]
                    bowler_actions[k] = analyse_bowler_action(
                        window_poses, eff_fps,
                        batter_swing_frame=(swing_frame - lo) // BOWLER_FRAME_STEP,
                        batter_hip=_typical_hip(landmarks[start:end]),
                    )
        return VideoAnalysis(estimates=estimates, footage=footage, bowler_actions=bowler_actions)
    finally:
        if owns_estimator:
            estimator.close()

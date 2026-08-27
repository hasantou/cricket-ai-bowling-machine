"""
Pose-estimation adapter: turns a video file into per-frame body-pose
landmarks and derived batting features (features.py), using a pretrained
MediaPipe Pose Landmarker model — no training involved, and nothing here
has ever seen a real player or scraped video (see download_model.py and
adaptation-engine/simulator.py for why that matters).

This is the real, working slice of "pose estimation on the batter's
stance, backlift, and footwork" from this folder's README (Phase-1
scope). What it does NOT yet do: know which frames matter (delivery
boundaries), tell front leg from back leg, or track the bat — see
features.py's docstring for the same caveats on the derived features.
Those are natural next slices once real footage exists to test against.

Run against a real clip: python cv-pipeline/pose/pose_estimation.py <video_path>
(requires the model file — run download_model.py first)
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from features import Point, compute_frame_features

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "pose_landmarker_lite.task")

# Index -> name for the 10 landmarks features.py needs, in MediaPipe's
# standard 33-point BlazePose topology.
LANDMARK_INDEX_TO_NAME = {
    11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist",
    23: "left_hip", 24: "right_hip",
    25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle",
}


@dataclass
class FrameResult:
    frame_index: int
    timestamp_ms: int
    landmarks: Optional[Dict[str, Point]]   # None if no person detected
    features: Optional[dict]                # None if landmarks is None


def require_model(model_path: str = None) -> str:
    path = model_path or MODEL_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No pose model at {path}. Run "
            "`python cv-pipeline/pose/download_model.py` first."
        )
    return path


def _extract_named_landmarks(detection_result) -> Optional[Dict[str, Point]]:
    if not detection_result.pose_landmarks:
        return None
    raw = detection_result.pose_landmarks[0]  # first detected person
    return {
        name: Point(raw[idx].x, raw[idx].y, raw[idx].z)
        for idx, name in LANDMARK_INDEX_TO_NAME.items()
    }


def process_video(video_path: str, model_path: str = None) -> List[FrameResult]:
    """Run pose estimation over every frame of `video_path` and return
    per-frame landmarks + derived features. Frames with no detected
    person get landmarks=None, features=None rather than being dropped —
    callers need to know a frame was seen but empty, not just missing."""
    resolved_model_path = require_model(model_path)

    base_options = mp_python.BaseOptions(model_asset_path=resolved_model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    results: List[FrameResult] = []

    try:
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            frame_index = 0
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                timestamp_ms = int((frame_index / fps) * 1000)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                detection = landmarker.detect_for_video(mp_image, timestamp_ms)
                landmarks = _extract_named_landmarks(detection)
                frame_features = compute_frame_features(landmarks)

                results.append(FrameResult(
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                    landmarks=landmarks,
                    features=frame_features,
                ))
                frame_index += 1
    finally:
        cap.release()

    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python pose_estimation.py <video_path>")
        sys.exit(1)

    frames = process_video(sys.argv[1])
    detected = sum(1 for f in frames if f.landmarks is not None)
    print(f"Processed {len(frames)} frames, pose detected in {detected}.")
    for f in frames[:5]:
        print(f.frame_index, f.timestamp_ms, "ms", f.features)

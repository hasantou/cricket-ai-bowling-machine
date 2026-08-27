"""
Wraps MediaPipe's real pretrained BlazePose model (Google's Tasks API,
mediapipe>=0.10) to turn a video clip into a sequence of per-frame body
landmarks — the input feature_extraction.py and outcome_bridge.py need.

This is the one piece of the CV pipeline that is a genuine pretrained
neural network doing real inference on real footage, not a placeholder.
It does need its model weights, though: MediaPipe's pip package does not
bundle them, and instead expects a separate ~5.5MB `.task` file from
Google's official model storage. download_model() fetches that file
explicitly, once, and only when called — PoseEstimator never triggers a
network download on its own; it fails loudly with FileNotFoundError and
tells you to call download_model() first, the same "no silent fake
success" pattern neural_scorer.load_model() uses.
"""

import os
import urllib.request
from typing import List

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from feature_extraction import FrameLandmarks

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "pose_landmarker_lite.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)


def download_model() -> str:
    """Fetches Google's official pretrained pose-landmarker weights
    (~5.5MB, from MediaPipe's own model storage). Call this deliberately,
    once — nothing in this module calls it automatically."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


class PoseEstimator:
    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No pose-landmarker model at {model_path}. Call "
                "pose_estimation.download_model() once (downloads Google's "
                "official ~5.5MB pretrained weights), then retry."
            )
        options = mp_vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    def extract_landmarks_from_frames(self, frames_bgr, fps: float) -> List[FrameLandmarks]:
        """frames_bgr: list of HxWx3 uint8 numpy arrays, as read by
        OpenCV (BGR channel order). Returns one landmark list per frame
        that had a detected person; frames with nobody detected are
        skipped outright rather than padded with fabricated zeros, so
        downstream feature math never sees a fake detection."""
        results = []
        for i, frame in enumerate(frames_bgr):
            rgb = np.ascontiguousarray(frame[:, :, ::-1])
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(i * (1000.0 / fps))
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
            if result.pose_landmarks:
                landmarks = [(lm.x, lm.y, lm.visibility) for lm in result.pose_landmarks[0]]
                results.append(landmarks)
        return results

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


if __name__ == "__main__":
    print("Downloading pose landmarker model...")
    path = download_model()
    print(f"Saved to {path}")
    PoseEstimator(path).close()
    print("Model loads and initialises correctly.")

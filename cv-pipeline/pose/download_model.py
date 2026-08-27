"""
Fetches the pretrained MediaPipe Pose Landmarker model file. This is a
one-time setup step, not something the app or tests do implicitly on
every run — a model download shouldn't happen silently mid-pipeline.

The model itself is Google's pretrained BlazePose-based landmarker,
licensed for this kind of use; nothing here trains anything or touches
YouTube data (see adaptation-engine's simulator.py docstring and the
project discussion of why: no delivery-level labels, and copyright risk
on scraped video). It's gitignored rather than committed because it's a
~5.5MB third-party binary, reproducible from this URL — re-run this
script after a fresh clone.

Run: python cv-pipeline/pose/download_model.py
"""

import os
import urllib.request

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "pose_landmarker_lite.task")


def ensure_model(force: bool = False) -> str:
    if force or not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        print(f"Downloading pose landmarker model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")
    else:
        print(f"Model already present at {MODEL_PATH}")
    return MODEL_PATH


if __name__ == "__main__":
    ensure_model()

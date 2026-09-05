import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from pose_estimation import MODEL_PATH, PoseEstimator


def test_fails_loudly_without_a_downloaded_model(tmp_path):
    # No model file has been downloaded in this environment (that's a
    # deliberate, explicit step — see pose_estimation.py's docstring),
    # so this must fail with a clear, actionable error, not a silent
    # fake success or a cryptic stack trace.
    missing_path = str(tmp_path / "does_not_exist.task")
    with pytest.raises(FileNotFoundError, match="download_model"):
        PoseEstimator(model_path=missing_path)


@pytest.mark.skipif(
    not os.path.exists(MODEL_PATH),
    reason="requires the downloaded pose-landmarker model — run download_model() first",
)
def test_one_estimator_can_process_multiple_clips_in_sequence():
    # Regression test: MediaPipe's VIDEO mode enforces strictly increasing
    # timestamps for the lifetime of one landmarker, not just within a
    # single clip. Reusing one PoseEstimator across two separate "videos"
    # (the realistic case for a coaching session logging several
    # deliveries without reloading the model each time) used to raise
    # "Input timestamp must be monotonically increasing" on the second
    # call — found by running real footage through the pipeline.
    blank_frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)]
    estimator = PoseEstimator()
    try:
        estimator.extract_landmarks_from_frames(blank_frames, fps=30.0)
        estimator.extract_landmarks_from_frames(blank_frames, fps=30.0)  # must not raise
    finally:
        estimator.close()

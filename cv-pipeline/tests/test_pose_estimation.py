import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from pose_estimation import PoseEstimator


def test_fails_loudly_without_a_downloaded_model(tmp_path):
    # No model file has been downloaded in this environment (that's a
    # deliberate, explicit step — see pose_estimation.py's docstring),
    # so this must fail with a clear, actionable error, not a silent
    # fake success or a cryptic stack trace.
    missing_path = str(tmp_path / "does_not_exist.task")
    with pytest.raises(FileNotFoundError, match="download_model"):
        PoseEstimator(model_path=missing_path)

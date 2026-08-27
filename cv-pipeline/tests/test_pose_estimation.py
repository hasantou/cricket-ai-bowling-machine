import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pose"))

import pytest
from pose_estimation import process_video, require_model, MODEL_PATH

FIXTURE_VIDEO = os.path.join(os.path.dirname(__file__), "fixtures", "synthetic_no_person.mp4")

# The pretrained model is a ~5.5MB third-party download (download_model.py),
# gitignored rather than committed — these tests need it on disk and skip
# cleanly if it isn't there yet, rather than failing or hitting the network.
requires_model = pytest.mark.skipif(
    not os.path.exists(MODEL_PATH),
    reason="Pose model not downloaded — run `python cv-pipeline/pose/download_model.py` first",
)


def test_require_model_raises_clear_error_when_missing(tmp_path):
    missing_path = str(tmp_path / "does_not_exist.task")
    with pytest.raises(FileNotFoundError, match="download_model.py"):
        require_model(missing_path)


def test_process_video_raises_on_missing_video_file():
    with pytest.raises(FileNotFoundError):
        process_video("no_such_video.mp4")


@requires_model
def test_process_video_processes_every_frame_of_the_fixture():
    frames = process_video(FIXTURE_VIDEO)
    assert len(frames) == 8  # matches make_fixture.py's N_FRAMES


@requires_model
def test_process_video_reports_no_detection_on_a_video_with_no_person():
    # The fixture has no human shape in it at all, so this is an honest
    # check of the "seen but empty" path, not a claim about accuracy on
    # real footage — that needs real footage (see cv-pipeline/README.md).
    frames = process_video(FIXTURE_VIDEO)
    assert all(f.landmarks is None for f in frames)
    assert all(f.features is None for f in frames)

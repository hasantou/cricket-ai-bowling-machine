import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
from shot_labels import crop_video_band, evaluate, load_labels, locate_frame_in_video


def _make_clip(path, n=30, w=96, h=140):
    rng = np.random.default_rng(0)
    frames = []
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30, (w, h))
    for i in range(n):
        f = np.full((h, w, 3), 90, np.uint8)
        cv2.circle(f, (10 + i * 2, 20 + (i * 7) % 90), 12, (30, 200, 240), -1)   # a moving blob
        f = np.clip(f.astype(int) + rng.integers(0, 12, f.shape), 0, 255).astype(np.uint8)
        frames.append(f)
        writer.write(f)
    writer.release()
    return frames


def test_the_band_is_cropped_out_of_a_screenshot_with_black_bars():
    band = np.full((140, 96, 3), 120, np.uint8)
    shot = np.vstack([np.zeros((60, 96, 3), np.uint8), band, np.zeros((50, 96, 3), np.uint8)])
    assert crop_video_band(shot).shape[0] == 140


def test_a_screenshot_of_frame_k_is_located_at_frame_k():
    with tempfile.TemporaryDirectory() as d:
        clip = os.path.join(d, "c.avi")
        frames = _make_clip(clip)
        k = 17
        shot = np.vstack([np.zeros((60, 96, 3), np.uint8), frames[k], np.zeros((50, 96, 3), np.uint8)])
        cv2.rectangle(shot, (10, 100), (30, 120), (0, 255, 0), 4)            # a hand-drawn mark on top
        img = os.path.join(d, "s.png")
        cv2.imwrite(img, shot)
        best = locate_frame_in_video(img, clip)
        assert abs(best[0][0] - k) <= 1


def test_evaluate_skips_unlabelled_shots_and_reports_how_little_it_rests_on():
    labels = [{"id": "a", "shot": None}, {"id": "b", "shot": "Pull"}, {"id": "c", "shot": "Cover drive"}]
    r = evaluate({"a": "Pull", "b": "pull", "c": "Off drive"}, labels)
    assert r["labelled_and_predicted"] == 2 and r["correct"] == 1
    assert r["accuracy"] == 0.5 and r["disagreements"][0]["id"] == "c"
    assert "only 2" in r["note"]


def test_evaluate_with_nothing_to_compare_does_not_invent_an_accuracy():
    assert evaluate({}, [{"id": "a", "shot": None}])["accuracy"] is None


def test_the_shipped_label_file_loads_and_its_frame_is_recorded():
    labels = load_labels()
    assert labels and labels[0]["frame"] == 111 and labels[0]["shot"] is None

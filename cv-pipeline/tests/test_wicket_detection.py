import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
from wicket_detection import find_disruption

W, H = 200, 150
REGION = (70, 40, 130, 100)


def write_clip(path, region_states, bg=(80, 140, 60), disrupted_color=(30, 30, 200), fps=60):
    """`region_states` is one bool per frame: True draws a bright, different-colour patch filling
    the region (a "disrupted" look), False leaves it as plain background (the calm, undisrupted
    look) — the controlled case the algorithm should get right."""
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    x0, y0, x1, y1 = REGION
    for disrupted in region_states:
        frame = np.full((H, W, 3), bg, np.uint8)
        if disrupted:
            frame[y0:y1, x0:x1] = disrupted_color
        writer.write(frame)
    writer.release()


def test_a_sustained_change_in_the_region_is_reported_as_an_event():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "clip.mp4")
        states = [False] * 20 + [True] * 15 + [False] * 20
        write_clip(path, states)
        report = find_disruption(path, REGION)
        assert report.wicket_detected
        assert report.events[0].start_frame == 20


def test_a_brief_blip_that_recovers_is_not_reported():
    """A ball flying past the stumps, or a moment's noise, must not be mistaken for a break."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "clip.mp4")
        states = [False] * 20 + [True] * 2 + [False] * 30       # far shorter than SUSTAIN_FRAMES
        write_clip(path, states)
        report = find_disruption(path, REGION)
        assert not report.wicket_detected


def test_a_calm_clip_with_no_disruption_reports_nothing():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "clip.mp4")
        write_clip(path, [False] * 40)
        report = find_disruption(path, REGION)
        assert not report.wicket_detected
        assert any("No sustained disruption" in n for n in report.notes)


def test_disruption_starting_right_at_the_edge_of_the_baseline_is_still_found():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "clip.mp4")
        states = [False] * 10 + [True] * 20        # disruption starts exactly at the baseline boundary
        write_clip(path, states)
        report = find_disruption(path, REGION)
        assert report.wicket_detected


def test_two_separate_disruptions_are_reported_once_each_not_repeatedly():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "clip.mp4")
        states = [False] * 15 + [True] * 10 + [False] * 15 + [True] * 10 + [False] * 15
        write_clip(path, states)
        report = find_disruption(path, REGION)
        assert len(report.events) == 2
        assert report.events[0].start_frame == 15
        assert report.events[1].start_frame == 40


def test_a_clip_shorter_than_the_baseline_window_is_handled_without_crashing():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "clip.mp4")
        write_clip(path, [False] * 5)
        report = find_disruption(path, REGION)
        assert not report.wicket_detected
        assert any("too short" in n.lower() for n in report.notes)


def test_change_outside_the_region_does_not_trigger_a_false_event():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "clip.mp4")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 60, (W, H))
        for i in range(40):
            frame = np.full((H, W, 3), (80, 140, 60), np.uint8)
            cv2.circle(frame, (20, 20), 10, (255, 255, 255), -1)   # something moving, well outside REGION
            writer.write(frame)
        writer.release()
        report = find_disruption(path, REGION)
        assert not report.wicket_detected


def test_missing_file_raises_a_clear_error():
    import pytest
    with pytest.raises(FileNotFoundError):
        find_disruption("does_not_exist.mp4", REGION)

import os
import sys
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from cricket_trajectory.delivery_sync import align


def clocks(n=8, offset=12.34, jitter=0.03, seed=0, start=5.0, gap=9.0):
    rng = random.Random(seed)
    sensor = [start + gap * i + rng.uniform(0, 2.5) for i in range(n)]
    video = [t + offset + rng.uniform(-jitter, jitter) for t in sensor]
    return video, sensor


def test_the_clock_offset_is_recovered_and_every_delivery_paired_correctly():
    video, sensor = clocks()
    r = align(video, sensor)
    assert r.ok and r.offset_s == pytest.approx(12.34, abs=0.03)
    assert r.matches == [(i, i) for i in range(8)] and not r.unmatched_video and not r.unmatched_sensor


def test_a_negative_offset_and_shuffled_input_still_align():
    video, sensor = clocks(offset=-3.7, seed=3)
    random.Random(1).shuffle(video)
    r = align(video, sensor)
    assert r.ok and r.offset_s == pytest.approx(-3.7, abs=0.03)
    assert len(r.matches) == 8


def test_a_delivery_the_camera_missed_is_reported_as_unmatched_not_forced():
    video, sensor = clocks()
    missing = video.pop(3)
    r = align(video, sensor)
    assert r.ok and len(r.matches) == 7 and r.unmatched_sensor == [3] and r.unmatched_video == []
    assert any("no video partner" in n for n in r.notes)
    # and the neighbours are still paired with the right partners
    assert r.sensor_for_video(3) == 4 and r.video_for_sensor(3) is None


def test_a_swing_with_no_contact_leaves_an_unmatched_video_moment():
    video, sensor = clocks()
    sensor.pop(5)
    r = align(video, sensor)
    assert r.ok and len(r.unmatched_video) == 1 and any("no sensor partner" in n for n in r.notes)


def test_extra_spurious_camera_detections_do_not_break_the_alignment():
    video, sensor = clocks()
    video += [1.0, 40.0, 77.7]
    r = align(video, sensor)
    assert r.ok and len(r.matches) == 8 and len(r.unmatched_video) == 3


def test_a_single_delivery_can_align_but_two_unrelated_lists_cannot_be_trusted():
    assert align([20.0], [7.0]).ok
    unrelated = align([1.0, 30.0, 61.0], [5.0, 22.0, 50.0], tolerance_s=0.05)
    assert not unrelated.ok or len(unrelated.matches) < 3


def test_clock_drift_is_estimated_and_flagged():
    sensor = [10.0 * i for i in range(1, 10)]
    video = [t + 5.0 + 0.004 * t for t in sensor]          # video clock runs 4 ms/s fast
    r = align(video, sensor, tolerance_s=0.5)
    assert r.drift_s_per_s == pytest.approx(0.004, abs=0.0006)
    assert any("drift" in n for n in r.notes)


def test_empty_inputs_are_handled_honestly():
    r = align([], [1.0, 2.0])
    assert not r.ok and r.offset_s is None and r.unmatched_sensor == [0, 1]
    assert not align([1.0], []).ok

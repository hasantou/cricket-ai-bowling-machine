import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from footage_check import assess_footage


def skeleton(height_fraction, visibility=0.95):
    """Nose at y=0.3, ankles `height_fraction` lower; the rest at hip height."""
    lm = [(0.5, 0.5, visibility)] * 33
    lm = list(lm)
    lm[0] = (0.5, 0.30, visibility)
    lm[27] = (0.48, 0.30 + height_fraction, visibility)
    lm[28] = (0.52, 0.30 + height_fraction, visibility)
    return lm


def test_a_large_well_detected_person_is_good():
    report = assess_footage([skeleton(0.55)] * 100, frames_total=100)
    assert report.verdict == "good"
    assert report.person_height_fraction == 0.55 or abs(report.person_height_fraction - 0.55) < 1e-9


def test_a_tiny_person_is_poor_and_the_reason_says_so():
    """A person only 13% of the frame's height is still 'detected' every frame — the check has to say it's too small anyway."""
    report = assess_footage([skeleton(0.13)] * 100, frames_total=100)
    assert report.verdict == "poor"
    assert any("too small" in r for r in report.reasons)


def test_a_middling_size_is_marginal():
    assert assess_footage([skeleton(0.22)] * 100, frames_total=100).verdict == "marginal"


def test_low_pose_confidence_downgrades_an_otherwise_large_person():
    report = assess_footage([skeleton(0.55, visibility=0.3)] * 100, frames_total=100)
    assert report.verdict == "poor"
    assert any("unsure" in r for r in report.reasons)


def test_frequent_dropouts_are_reported():
    report = assess_footage([skeleton(0.55)] * 40, frames_total=100)
    assert report.detection_rate == 0.4
    assert report.verdict == "poor"


def test_the_worst_single_problem_decides_the_verdict():
    # big person, fine confidence, but found in only 70% of frames -> marginal
    assert assess_footage([skeleton(0.55)] * 70, frames_total=100).verdict == "marginal"


def test_no_person_anywhere_is_poor_not_a_crash():
    report = assess_footage([], frames_total=100)
    assert report.verdict == "poor"
    assert report.frames_with_person == 0


def test_a_video_that_could_only_be_partly_read_is_flagged_as_unreadable():
    """The suspected cause of 'there were 4 deliveries, it found 1': a
    server whose video codec stops decoding partway leaves the pipeline
    quietly analysing only the start of the clip."""
    report = assess_footage([skeleton(0.55)] * 200, frames_total=200, frames_expected=789)
    assert report.verdict == "poor"
    assert any("partly unreadable" in r for r in report.reasons)


def test_a_fully_read_video_is_not_flagged():
    report = assess_footage([skeleton(0.55)] * 789, frames_total=789, frames_expected=789)
    assert report.verdict == "good"


def test_an_unknown_expected_count_is_ignored():
    assert assess_footage([skeleton(0.55)] * 50, frames_total=50, frames_expected=0).verdict == "good"


def _with(lm, index, x=None, vis=None):
    lm = list(lm)
    px, py, pv = lm[index]
    lm[index] = (px if x is None else x, py, pv if vis is None else vis)
    return lm


def test_visible_hands_and_a_large_batter_are_suitable_for_reading_shots():
    r = assess_footage([skeleton(0.55)] * 100, frames_total=100)
    assert r.hand_visibility == 1.0 and r.shot_reading_ok and r.shot_reading_notes == []


def test_hidden_hands_make_the_camera_position_unsuitable_and_say_why():
    hidden = _with(_with(skeleton(0.55), 15, vis=0.2), 16, vis=0.2)
    r = assess_footage([hidden] * 100, frames_total=100)
    assert r.hand_visibility == 0.0 and not r.shot_reading_ok
    assert any("hands" in n for n in r.shot_reading_notes)
    assert r.verdict == "good"          # the clip is fine as a clip; it is the camera position that can't read shots


def test_hands_seen_in_only_some_frames_is_scored_as_that_share():
    good, hidden = skeleton(0.55), _with(skeleton(0.55), 16, vis=0.1)
    r = assess_footage([good] * 40 + [hidden] * 60, frames_total=100)
    assert abs(r.hand_visibility - 0.4) < 1e-9 and not r.shot_reading_ok


def test_overlapping_legs_are_reported_as_such():
    overlapped = _with(_with(skeleton(0.55), 27, x=0.50), 28, x=0.505)
    r = assess_footage([overlapped] * 100, frames_total=100)
    assert r.leg_separation < 0.06
    assert any("legs overlap" in n for n in r.shot_reading_notes)


def test_a_small_batter_is_told_to_move_closer():
    r = assess_footage([skeleton(0.22)] * 100, frames_total=100)
    assert any("closer" in n for n in r.shot_reading_notes) and not r.shot_reading_ok

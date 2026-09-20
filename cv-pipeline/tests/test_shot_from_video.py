import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "trajectory-engine"))

import pytest
from body_vectors import BodyVectorReport, JointVector, analyse_body_vectors
from cricket_trajectory.shot_vocabulary import lookup
from footage_check import FootageReport
from shot_from_video import (
    BEHIND_BATTER, BOWLERS_END, LEFT_HANDED, RIGHT_HANDED, SIDE_ON, estimate_shot_from_video,
)
from test_body_vectors import FPS, moving_wrist


def good_footage(verdict="good", shot_reading_ok=True):
    return FootageReport(300, 300, 1.0, 0.4, 0.95, 300, verdict, ["fine"],
                         hand_visibility=1.0 if shot_reading_ok else 0.1, leg_separation=0.1,
                         shot_reading_ok=shot_reading_ok,
                         shot_reading_notes=[] if shot_reading_ok else ["Both hands were confidently seen in only 10% of frames"])


def report(across=0.0, up=0.0, length=3.0, speed=8.0, hand="right wrist", confidence="good",
           tracked=1.0, verdict_speed=None):
    v = JointVector(1.0, 0.0, speed, 0.0, "right", confidence)
    return BodyVectorReport(
        peak_frame=10, peak_hand=hand, peak_hand_speed=speed, vectors_at_peak={hand: v} if hand != "trunk" else {},
        hand_path_start=5, hand_path_end=15, hand_path_length=length, hand_path_net=(across, up),
        hand_path_net_direction="right", hand_path_points=[], hip_shift=(0, 0),
        ankle_shift={"left ankle": (0, 0), "right ankle": (0, 0)}, shoulder_line_change_deg=0.0,
        hip_line_change_deg=0.0, body_speed_at_peak=0.5, tracked_fraction=tracked, caveats=[],
    )


def est(rep, camera=BEHIND_BATTER, hand=RIGHT_HANDED, footage=None):
    return estimate_shot_from_video(rep, footage or good_footage(), camera, hand)


# ---- refusing ----
@pytest.mark.parametrize("kwargs,footage_verdict,fragment", [
    ({}, "marginal", "Footage quality"),
    ({"confidence": "partial"}, "good", "partly visible"),
    ({"confidence": "poor"}, "good", "poorly visible"),
    ({"hand": "trunk"}, "good", "Neither hand"),
    ({"tracked": 0.7}, "good", "only 70%"),
    ({"speed": 40.0}, "good", "implausible"),
])
def test_it_refuses_rather_than_guesses_when_the_footage_or_tracking_is_not_good(kwargs, footage_verdict, fragment):
    r = est(report(across=1.5, up=0.5, **kwargs), footage=good_footage(footage_verdict))
    assert r.verdict == "cannot tell" and r.shot is None
    assert any(fragment in why for why in r.reasons)


def test_no_body_movement_at_all_is_cannot_tell():
    assert estimate_shot_from_video(None, good_footage(), BEHIND_BATTER).verdict == "cannot tell"


# ---- swing or not ----
def test_hands_that_barely_move_are_a_leave_not_a_shot():
    r = est(report(speed=1.0, length=0.3))
    assert r.verdict == "no shot" and r.shot == "Leave"


def test_a_short_hand_path_is_a_defensive_push_family_only():
    r = est(report(across=0.3, up=0.2, length=0.8, speed=6.0))
    assert r.verdict == "family only" and r.family == "defensive push" and r.shot is None


# ---- naming, camera behind a right-hander: screen-right is the off side ----
@pytest.mark.parametrize("across,up,expected", [
    (0.1, 1.6, "Straight drive"),
    (0.7, 1.6, "Off drive"),
    (1.4, 1.2, "Cover drive"),
    (-0.7, 1.6, "On drive"),
    (-1.4, 1.2, "Flick"),
    (1.8, 0.3, "Cut"),
    (-1.8, 0.2, "Pull"),
    (-2.0, 0.9, "Hook"),
])
def test_shots_from_the_hand_path_behind_a_right_hander(across, up, expected):
    r = est(report(across=across, up=up))
    assert r.verdict == "shot named" and r.shot == expected


def test_the_same_swing_flips_sides_from_the_bowlers_end_and_for_a_left_hander():
    rep = report(across=1.4, up=1.2)
    assert est(rep, BEHIND_BATTER, RIGHT_HANDED).side == "off side"
    assert est(rep, BOWLERS_END, RIGHT_HANDED).side == "leg side"
    assert est(rep, BEHIND_BATTER, LEFT_HANDED).side == "leg side"
    assert est(rep, BOWLERS_END, LEFT_HANDED).side == "off side"


def test_side_on_names_the_family_but_never_a_side():
    r = est(report(across=1.4, up=1.2), camera=SIDE_ON)
    assert r.verdict == "family only" and r.side is None and r.shot is None
    assert r.family == "drive-type swing"


def test_a_shot_named_always_carries_the_unvalidated_label():
    assert "not validated" in est(report(across=1.4, up=1.2)).confidence


def test_every_name_it_can_emit_is_in_the_shot_vocabulary():
    seen = set()
    for camera in (BEHIND_BATTER, BOWLERS_END):
        for hand in (RIGHT_HANDED, LEFT_HANDED):
            for across in (-2.0, -1.4, -0.7, 0.0, 0.7, 1.4, 2.0):
                for up in (0.0, 0.3, 0.9, 1.6):
                    for speed, length in ((1.0, 0.3), (6.0, 0.8), (8.0, 3.0)):
                        r = est(report(across=across, up=up, speed=speed, length=length), camera, hand)
                        if r.shot:
                            seen.add(r.shot)
    assert seen and all(lookup(name) is not None for name in seen), seen


def test_bad_camera_is_an_error_not_a_silent_default():
    with pytest.raises(ValueError):
        estimate_shot_from_video(report(), good_footage(), "from a drone")


# ---- end to end from body motion ----
def test_end_to_end_a_rightward_hand_swing_from_behind_reads_as_toward_the_off_side():
    rep = analyse_body_vectors(moving_wrist(0.0105 * 2, 0.0), FPS)
    r = estimate_shot_from_video(rep, good_footage(), BEHIND_BATTER, RIGHT_HANDED)
    assert (r.verdict, r.shot, r.side) == ("shot named", "Cut", "off side")     # a wide rightward hand path
    flipped = estimate_shot_from_video(rep, good_footage(), BOWLERS_END, RIGHT_HANDED)
    assert (flipped.shot, flipped.side) == ("Pull", "leg side")


def test_a_camera_position_that_cannot_show_the_hands_is_itself_a_reason_to_refuse():
    r = est(report(across=1.4, up=1.2), footage=good_footage(shot_reading_ok=False))
    assert r.verdict == "cannot tell" and any("hands" in why for why in r.reasons)

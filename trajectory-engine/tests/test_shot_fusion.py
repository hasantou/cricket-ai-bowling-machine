import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cricket_trajectory import fuse_shot


def sensor(shot, confidence="approximate"):
    return NS(shot=shot, confidence=confidence)


def video(shot=None, verdict="shot named", confidence="rule of thumb — not validated"):
    return NS(shot=shot, verdict=verdict, confidence=confidence)


def test_two_witnesses_agreeing_is_corroborated():
    f = fuse_shot(sensor("Cover drive"), video("Cover drive"))
    assert (f.shot, f.agreement) == ("Cover drive", "agree") and "two independent" in f.confidence


def test_the_videos_generic_cut_agrees_with_the_sensors_specific_cut():
    assert fuse_shot(sensor("Square cut"), video("Cut")).agreement == "agree"
    assert fuse_shot(sensor("Late cut"), video("Cut")).agreement == "agree"


def test_same_kind_of_shot_but_different_name_is_family_level_only_and_uses_the_sensor():
    f = fuse_shot(sensor("Off drive"), video("Cover drive"))
    assert f.agreement == "family only" and f.shot == "Off drive"


def test_disagreement_uses_the_sensor_and_says_so_out_loud():
    f = fuse_shot(sensor("Straight drive"), video("Pull"))
    assert f.agreement == "disagree" and f.shot == "Straight drive"
    assert "Pull" in f.notes[0] and "Straight drive" in f.notes[0]


def test_ball_off_the_bat_but_no_hand_movement_is_flagged_not_swallowed():
    f = fuse_shot(sensor("Forward defence"), video(verdict="no shot", shot="Leave"))
    assert f.agreement == "disagree" and f.shot == "Forward defence"


def test_sensor_alone_says_it_is_sensor_alone():
    f = fuse_shot(sensor("Pull"), video(verdict="cannot tell"))
    assert (f.source, f.agreement) == ("sensor only", "single source") and "no usable video" in f.notes[0]
    assert fuse_shot(sensor("Pull"), None).source == "sensor only"


def test_video_alone_keeps_its_unvalidated_label():
    f = fuse_shot(None, video("Pull"))
    assert f.source == "video only" and "not validated" in f.confidence


def test_a_swing_with_no_contact_is_a_swing_and_a_miss_but_only_with_both_witnesses():
    assert fuse_shot(sensor("no shot / missed"), video("Cover drive")).shot == "swing and a miss"
    assert fuse_shot(sensor("no shot / missed"), video(verdict="family only")).shot == "swing and a miss"
    only = fuse_shot(sensor("no shot / missed"), None)
    assert only.shot == "no shot / missed" and "needs the video" in only.notes[0]


def test_no_contact_and_no_swing_is_a_leave():
    assert fuse_shot(sensor("no shot / missed"), video(verdict="no shot", shot="Leave")).shot == "leave / no shot"


def test_nothing_is_invented_when_neither_source_names_a_shot():
    f = fuse_shot(None, video(verdict="cannot tell"))
    assert f.shot is None and f.agreement == "none"
    assert fuse_shot(None, None).shot is None

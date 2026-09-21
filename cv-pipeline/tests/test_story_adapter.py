import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "trajectory-engine"))

from bowler_analysis import BowlerActionEstimate
from cricket_trajectory.delivery_story import build_story
from footage_check import FootageReport
from footwork import FootworkReport
from shot_from_video import VideoShotEstimate
from story_adapter import video_info_from
from test_body_vectors import FPS, moving_wrist
from body_vectors import analyse_body_vectors


def fw(cls="front-foot"):
    return FootworkReport(cls, "left ankle", 0.31, 0.0, -0.02, 0.10, 0.55, (0.5, 0.8), (0.5, 0.75), (0.5, 0.8), (0.5, 0.8),
                          {"left ankle": "good", "right ankle": "good"}, True, [])


def test_everything_the_pipeline_produced_reaches_the_story_input():
    body = analyse_body_vectors(moving_wrist(0.0105 * 2, 0.0), FPS)
    bowler = BowlerActionEstimate("right-arm", 6.0, "high, near-vertical arm", 40, 1.1, 4.0, 20.0, None, 90, ())
    footage = FootageReport(300, 300, 1.0, 0.4, 0.95, 300, "good", ["fine"])
    shot = VideoShotEstimate("shot named", "Cut", "horizontal-bat swing", "off side", "rule of thumb", ["x"])
    v = video_info_from(body, fw(), shot, bowler, footage, swing_time_s=3.2)
    assert v.swing_verdict == body.swing_verdict and v.peak_hand_speed == body.peak_hand_speed
    assert (v.footwork_class, v.front_stride, v.lead_time_s, v.hip_shift) == ("front-foot", 0.31, 0.55, 0.10)
    assert (v.shot, v.shot_verdict) == ("Cut", "shot named") and v.bowler_action == "high, near-vertical arm"
    assert v.footage_verdict == "good" and v.swing_time_s == 3.2


def test_missing_pieces_stay_none_so_the_story_says_not_measured():
    v = video_info_from()
    assert v.swing_verdict is None and v.footwork_class is None and v.shot is None and v.bowler_action is None
    s = build_story(None, video=v)
    rows = {r.label: r for r in s.all_rows()}
    assert rows["Batter's movement"].source == "not measured"
    assert "the camera pipeline produced no reading" in rows["Batter's movement"].value
    assert not any(label in rows for label in ("Footwork", "Batter's swing", "Shot (from the hands)"))


def test_unclear_feet_give_no_stride_numbers():
    v = video_info_from(footwork=fw("unclear (feet not visible)"))
    assert v.footwork_class.startswith("unclear") and v.front_stride is None and v.lead_time_s is None

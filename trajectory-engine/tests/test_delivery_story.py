import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from cricket_trajectory import BallProperties, Delivery, Environment, assess_delivery, run_simulation, aim_delivery
from cricket_trajectory.delivery_report import build_delivery_report
from cricket_trajectory.delivery_story import (
    CREASE_SENSOR, NOT_MEASURED, PHYSICS, ImpactInfo, SyncInfo, VideoDeliveryInfo, build_story,
)
from cricket_trajectory.laws import WICKET_X_M
from cricket_trajectory.shot_fusion import FusedShot

BALL, ENV = BallProperties(), Environment()


def report_for(from_stumps, kmh=135.0, measured_kmh=None):
    d = aim_delivery(BALL, ENV, Delivery(speed_mps=kmh / 3.6, seam_angle_deg=20), WICKET_X_M - from_stumps, 0.1)
    assert d is not None
    a = assess_delivery(BALL, ENV, d)
    return build_delivery_report(
        d, run_simulation(BALL, ENV, d),
        measured_release_speed_mps=None if measured_kmh is None else measured_kmh / 3.6, assessment=a,
    )


@pytest.fixture(scope="module")
def short_report():
    return report_for(8.6)               # short of a length


@pytest.fixture(scope="module")
def full_report():
    return report_for(3.5)               # full


@pytest.fixture(scope="module")
def good_report():
    return report_for(6.2)               # good length


def video(**kw):
    base = dict(swing_verdict="swing", peak_hand_speed=8.2, footwork_class="front-foot", front_stride=0.31, lead_time_s=0.55,
                shot="Cover drive", shot_verdict="shot named", footage_verdict="good")
    base.update(kw)
    return VideoDeliveryInfo(**base)


def impact(**kw):
    base = dict(exit_speed_kmh=88.0, elevation_deg=6.0, azimuth_deg=48.0, region="cover", shot="Cover drive",
                shot_confidence="approximate", outcome="boundary", simulated=True)
    base.update(kw)
    return ImpactInfo(**base)


def labels(story):
    return {r.label: r for r in story.all_rows()}


def test_a_machine_only_story_says_what_it_does_not_know(good_report):
    s = build_story(good_report)
    rows = labels(s)
    assert rows["Batter's movement"].source == NOT_MEASURED and rows["Contact"].source == NOT_MEASURED
    assert rows["Length"].source == PHYSICS and "Footwork" not in rows and "Batter's swing" not in rows
    assert "No sensor recorded" in s.narrative and "swung" not in s.narrative
    assert not any("video" in k for k in s.provenance)


def test_simulated_sensors_are_labelled_simulated_everywhere():
    rep = report_for(6.2, measured_kmh=135.0)
    s = build_story(rep, impact=impact(), video=video())
    rows = labels(s)
    assert rows["Off the bat"].source_text() == "simulated impact sensor"
    assert rows["Pace (measured at release)"].source_text() == "simulated release sensor"
    assert "(simulated)" in s.narrative


def test_a_crease_sensor_call_is_attributed_to_the_sensor_and_a_model_call_to_the_model(good_report):
    model = NS(basis="physics model (no sensor reading)", call=None, wide_reasons=[], wide_margin_m=0.4, borderline=False,
               hits_stumps=True, no_ball_flags=[], agrees_with_model=True)
    sensor = NS(basis="sensor", call=None, wide_reasons=[], wide_margin_m=0.4, borderline=False,
                hits_stumps=True, no_ball_flags=[], agrees_with_model=True)
    assert labels(build_story(good_report, decision=model))["Call at the batter"].source == PHYSICS
    assert labels(build_story(good_report, decision=sensor, simulated_sensors=False))["Call at the batter"].source == CREASE_SENSOR


def test_footwork_against_length_is_a_coaching_observation_only_when_both_are_known(short_report, full_report):
    forward_to_short = build_story(short_report, video=video(footwork_class="front-foot"))
    assert any("Went forward to a short of a length ball" in o for o in forward_to_short.observations)
    back_to_full = build_story(full_report, video=video(footwork_class="back-foot"))
    assert any("Stayed back to a full ball" in o for o in back_to_full.observations)
    suited = build_story(full_report, video=video(footwork_class="front-foot", lead_time_s=0.5))
    assert any("suited the length" in o for o in suited.observations)
    unclear = build_story(full_report, video=video(footwork_class="unclear (feet not visible)", lead_time_s=None))
    assert not any("length" in o for o in unclear.observations)
    assert build_story(full_report).observations == []


def test_late_and_early_footwork_are_called_out(good_report):
    assert any("late footwork" in o for o in build_story(good_report, video=video(lead_time_s=0.08)).observations)
    assert any("very early" in o for o in build_story(good_report, video=video(lead_time_s=1.2)).observations)
    normal = build_story(good_report, video=video(lead_time_s=0.5)).observations
    assert not any("late footwork" in o or "very early" in o for o in normal)


def test_disagreements_between_sources_are_flagged_not_hidden(full_report):
    contact_no_swing = build_story(full_report, impact=impact(), video=video(swing_verdict="no swing"))
    assert any("felt bat" in f and "no swing" in f for f in contact_no_swing.flags)
    rep = report_for(6.2, kmh=135.0, measured_kmh=127.0)
    assert any("not delivering what it is told" in f for f in build_story(rep).flags)
    disagree = NS(basis="sensor", call="wide", wide_reasons=["x"], wide_margin_m=-0.1, borderline=False,
                  hits_stumps=False, no_ball_flags=[], agrees_with_model=False)
    assert any("physics model" in f for f in build_story(full_report, decision=disagree).flags)


def test_clock_alignment_problems_are_flagged(good_report):
    assert any("could not be matched" in f for f in build_story(good_report, sync=SyncInfo(matched=False)).flags)
    assert any("apart after aligning" in f for f in build_story(good_report, sync=SyncInfo(matched=True, residual_s=0.4)).flags)
    assert build_story(good_report, sync=SyncInfo(matched=True, residual_s=0.03)).flags == []


def test_swing_with_no_contact_is_a_swing_and_a_miss_and_no_swing_is_a_leave(full_report):
    missed = build_story(
        full_report, impact=impact(exit_speed_kmh=None, elevation_deg=None, shot="no shot / missed", outcome="missed"),
        video=video(swing_verdict="swing"),
    )
    assert any("swung and missed" in o for o in missed.observations)
    left = build_story(full_report, impact=impact(exit_speed_kmh=None, elevation_deg=None, outcome="beaten"),
                       video=video(swing_verdict="no swing"))
    assert any("left alone" in o for o in left.observations)


def test_result_uses_the_scorecard_and_a_boundary_notes_that_a_six_is_not_measured(full_report):
    s = build_story(full_report, impact=impact())
    assert labels(s)["Result"].value.startswith("4 runs")
    assert "Four or six?" in labels(s) and labels(s)["Four or six?"].source == NOT_MEASURED
    out = build_story(full_report, impact=impact(
        outcome="missed", exit_speed_kmh=None, elevation_deg=None,
        outcome_note="no contact, and the ball was on target - bowled (Law 32)"))
    assert labels(out)["Result"].value.startswith("OUT") and "Law 32" in labels(out)["Result"].value


def test_a_shot_the_video_could_not_name_is_shown_as_not_measured_with_the_reason(good_report):
    v = video(shot=None, shot_verdict="cannot tell", shot_reasons=["Both hands were confidently seen in only 7% of frames"])
    row = labels(build_story(good_report, video=v))["Shot (from the hands)"]
    assert row.source == NOT_MEASURED and "7%" in row.value


def test_both_sources_agreeing_appears_in_the_story_and_the_narrative(short_report):
    fused = FusedShot("Cover drive", "sensor + video", "agree", "corroborated by two independent sources", [])
    s = build_story(short_report, impact=impact(), video=video(), fused=fused)
    assert "Cover drive" in labels(s)["Shot (both sources)"].value
    assert "corroborated" in s.narrative and "swung" in s.narrative and "front foot" in s.narrative


def test_markdown_carries_the_source_column_and_provenance_counts(good_report):
    s = build_story(good_report, impact=impact(), video=video())
    md = s.markdown()
    assert "| Source |" in md and "simulated impact sensor" in md and "video (rule of thumb)" in md
    assert s.provenance["simulated impact sensor"] >= 1 and sum(s.provenance.values()) == len(s.all_rows())


def test_a_video_only_story_needs_no_machine_report_and_says_so():
    s = build_story(None, video=video(bowler_arm="right-arm", bowler_action="high, near-vertical arm", bowler_release_height=1.1))
    rows = labels(s)
    assert rows["Machine data"].source == NOT_MEASURED and "not measured from video" in rows["Machine data"].value
    assert rows["Bowler's action"].source == "video (pose)"
    assert "Length" not in rows and "Pace (commanded)" not in rows
    assert s.narrative.startswith("A delivery seen only on video")
    assert not any(o for o in s.observations if "length" in o)


def test_a_story_with_nothing_at_all_still_does_not_invent_anything():
    s = build_story(None)
    assert s.observations == [] and s.flags == []
    assert all(r.source == NOT_MEASURED for r in s.all_rows())


def test_poor_footage_is_flagged_so_precise_looking_numbers_are_not_over_trusted(good_report):
    poor = build_story(good_report, video=video(footage_verdict="poor"))
    assert any("indicative only" in f for f in poor.flags)
    assert not any("indicative only" in f for f in build_story(good_report, video=video(footage_verdict="good")).flags)

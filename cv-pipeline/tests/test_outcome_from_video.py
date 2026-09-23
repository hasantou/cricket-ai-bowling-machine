import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from body_vectors import BodyVectorReport, JointVector, MAX_PLAUSIBLE_SPEED
from outcome_from_video import DOT, FOUR, SIX, UNCLEAR, WICKET, estimate_net_outcome
from post_shot import LOST_TRACK, RAN, STAYED, PostShotReport


def report(speed, up=0.0, across=0.0):
    v = JointVector(1.0, 0.0, speed, 0.0, "right", "good")
    return BodyVectorReport(
        peak_frame=10, peak_hand="right wrist", peak_hand_speed=speed, vectors_at_peak={"right wrist": v},
        hand_path_start=5, hand_path_end=15, hand_path_length=max(abs(up), abs(across), 0.1),
        hand_path_net=(across, up), hand_path_net_direction="right", hand_path_points=[],
        hip_shift=(0, 0), ankle_shift={"left ankle": (0, 0), "right ankle": (0, 0)},
        shoulder_line_change_deg=0.0, hip_line_change_deg=0.0, body_speed_at_peak=0.5,
        tracked_fraction=1.0, caveats=[],
    )


def after(classification, net=None):
    return PostShotReport(classification, net, frames_available=50, frames_tracked=40, gap_started_at_s=None)


# ---- refusing / no data ----
def test_no_body_report_is_cannot_estimate():
    assert estimate_net_outcome(None).outcome == UNCLEAR


def test_an_implausible_speed_is_cannot_estimate_not_a_guess():
    assert estimate_net_outcome(report(MAX_PLAUSIBLE_SPEED + 5)).outcome == UNCLEAR


# ---- no swing ----
def test_no_swing_is_a_dot_ball():
    assert estimate_net_outcome(report(1.0)).outcome == DOT


def test_no_swing_plus_lost_track_is_still_only_a_dot_with_a_heavy_caveat():
    r = estimate_net_outcome(report(1.0), after(LOST_TRACK))
    assert r.outcome == DOT
    assert any("not evidence of a dismissal" in c for c in r.caveats)


# ---- the four-way split ----
def test_a_weak_lofted_swing_is_a_wicket_type_estimate():
    r = estimate_net_outcome(report(5.0, up=0.8))
    assert r.outcome == WICKET
    assert "mistimed" in r.reasoning


def test_a_fast_lofted_swing_is_a_six_type_estimate():
    r = estimate_net_outcome(report(13.0, up=1.0))
    assert r.outcome == SIX


def test_a_fast_flat_swing_is_a_four_type_estimate():
    r = estimate_net_outcome(report(9.0, up=0.1))
    assert r.outcome == FOUR
    assert "flat" in r.reasoning


def test_a_moderate_swing_neither_fast_nor_lofted_defaults_to_dot():
    r = estimate_net_outcome(report(5.5, up=0.1))
    assert r.outcome == DOT


# ---- post-shot as a weak secondary signal only ----
def test_running_after_a_moderate_swing_nudges_toward_four_but_stays_caveated():
    r = estimate_net_outcome(report(5.5, up=0.1), after(RAN, net=5.0))
    assert r.outcome == FOUR
    assert any("risky single" in c for c in r.caveats)


def test_no_running_after_a_big_shot_is_flagged_but_does_not_change_the_call():
    r = estimate_net_outcome(report(13.0, up=1.0), after(STAYED, net=0.5))
    assert r.outcome == SIX
    assert any("unusual for a genuine six" in c for c in r.caveats)


def test_running_after_a_clear_six_swing_adds_no_extra_flag():
    r = estimate_net_outcome(report(13.0, up=1.0), after(RAN, net=6.0))
    assert r.outcome == SIX
    assert not any("unusual for a genuine six" in c for c in r.caveats)


# ---- every result carries the core honesty caveat ----
@pytest.mark.parametrize("body,post", [
    (None, None), (report(1.0), None), (report(5.0, up=0.8), None), (report(13.0, up=1.0), None),
    (report(9.0, up=0.1), None), (report(5.5), None),
])
def test_every_estimate_states_it_cannot_see_the_ball_or_confirm_contact(body, post):
    r = estimate_net_outcome(body, post)
    assert any("cannot confirm bat-ball contact" in c for c in r.caveats)
    assert "estimate" in r.confidence and "not validated" in r.confidence

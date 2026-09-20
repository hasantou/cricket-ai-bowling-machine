import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from bowler_analysis import (
    LEFT_WRIST, RIGHT_WRIST, analyse_bowler_action, identify_bowler, track_people,
)

FPS = 30.0
N_LANDMARKS = 33


def person(cx, cy, h, right_wrist=None, left_wrist=None):
    """A crude standing skeleton: `h` is the nose-to-ankle height, `cx, cy`
    the hip centre. Wrists default to hip height, hands down."""
    lm = [(cx, cy, 1.0)] * N_LANDMARKS
    lm = list(lm)
    lm[0] = (cx, cy - 0.5 * h, 1.0)                       # nose
    lm[11] = (cx - 0.10 * h, cy - 0.35 * h, 1.0)          # left shoulder
    lm[12] = (cx + 0.10 * h, cy - 0.35 * h, 1.0)          # right shoulder
    lm[15] = left_wrist or (cx - 0.15 * h, cy, 1.0)
    lm[16] = right_wrist or (cx + 0.15 * h, cy, 1.0)
    lm[23] = (cx - 0.05 * h, cy, 1.0)                     # left hip
    lm[24] = (cx + 0.05 * h, cy, 1.0)                     # right hip
    lm[27] = (cx - 0.05 * h, cy + 0.5 * h, 1.0)           # left ankle
    lm[28] = (cx + 0.05 * h, cy + 0.5 * h, 1.0)           # right ankle
    return lm


def make_delivery(n=45, release=30, h=0.30, arm="right", arm_dx=0.02, scale=1.0):
    """Frame list where a bowler runs left-to-right and raises one arm at
    `release`, beside a stationary batter. Pose order is shuffled per frame
    so identity has to come from tracking, not list position."""
    rng = random.Random(1)
    frames = []
    for i in range(n):
        cx = (0.20 + 0.40 * i / (n - 1) * scale)
        cy = 0.55
        raised = None
        if i == release:
            nose_y = cy - 0.5 * h
            sx = cx + (0.10 if arm == "right" else -0.10) * h
            raised = (sx + arm_dx * h * 10, nose_y - 0.12 * h, 1.0)
        kwargs = {"right_wrist": raised} if arm == "right" else {"left_wrist": raised}
        bowler = person(cx, cy, h, **(kwargs if raised else {}))
        batter = person(0.85, 0.50, h, right_wrist=(0.85 + 0.05 * (i % 5), 0.5, 1.0))
        poses = [bowler, batter]
        rng.shuffle(poses)
        frames.append(poses)
    return frames


def test_tracking_keeps_the_two_people_apart_despite_shuffled_order():
    tracks = track_people(make_delivery())
    long_tracks = [t for t in tracks if len(t.frames) > 30]
    assert len(long_tracks) == 2


def test_the_runner_who_raises_an_arm_is_the_bowler_not_the_stationary_batter():
    tracks = track_people(make_delivery())
    bowler = identify_bowler(tracks)
    assert bowler is not None
    xs = [p[24][0] for p in bowler.frames.values()]
    assert max(xs) - min(xs) > 0.3            # it's the one that travelled


def test_right_arm_bowler_and_release_frame_are_read_correctly():
    est = analyse_bowler_action(make_delivery(release=30, arm="right"), FPS)
    assert est is not None
    assert est.arm_side == "right-arm"
    assert est.release_frame == 30


def test_left_arm_bowler_is_read_as_left_arm():
    est = analyse_bowler_action(make_delivery(release=30, arm="left"), FPS)
    assert est.arm_side == "left-arm"


def test_a_straight_up_arm_reads_as_a_high_action():
    est = analyse_bowler_action(make_delivery(arm_dx=0.0), FPS)
    assert est.arm_angle_deg < 25
    assert est.arm_action_label == "high, near-vertical arm"


def test_a_more_sideways_arm_reads_as_a_lower_action():
    high = analyse_bowler_action(make_delivery(arm_dx=0.0), FPS)
    low = analyse_bowler_action(make_delivery(arm_dx=0.06), FPS)
    assert low.arm_angle_deg > high.arm_angle_deg + 20
    assert low.arm_action_label != "high, near-vertical arm"


def test_release_height_is_above_the_head():
    est = analyse_bowler_action(make_delivery(), FPS)
    assert est.release_height_torsos == pytest.approx(0.77, abs=0.05)


def test_run_up_speed_is_positive_and_scale_free():
    """The same action filmed from twice as far away (half the body size,
    half the on-screen travel) must give the same torso-lengths-per-second."""
    near = analyse_bowler_action(make_delivery(h=0.30, scale=1.0), FPS)
    far = analyse_bowler_action(make_delivery(h=0.15, scale=0.5), FPS)
    assert near.run_up_speed_torsos_per_s > 0
    assert far.run_up_speed_torsos_per_s == pytest.approx(near.run_up_speed_torsos_per_s, rel=0.05)


def test_release_to_swing_time_is_computed_when_the_batters_frame_is_known():
    est = analyse_bowler_action(make_delivery(release=20), FPS, batter_swing_frame=35)
    assert est.release_to_swing_s == pytest.approx(0.5, abs=1e-6)
    assert analyse_bowler_action(make_delivery(), FPS).release_to_swing_s is None


def test_no_bowler_when_only_a_stationary_batter_is_in_shot():
    """The correct answer for footage from behind the bowler's end."""
    frames = [[person(0.5, 0.5, 0.3, right_wrist=(0.5 + 0.05 * (i % 5), 0.5, 1.0))] for i in range(45)]
    assert analyse_bowler_action(frames, FPS) is None


def test_no_bowler_when_someone_runs_but_never_raises_an_arm():
    frames = [[person(0.2 + 0.01 * i, 0.55, 0.3)] for i in range(45)]
    assert analyse_bowler_action(frames, FPS) is None


def test_no_bowler_when_a_hand_goes_up_but_nobody_travels():
    frames = []
    for i in range(45):
        raised = (0.53, 0.5 - 0.15 - 0.036, 1.0) if i == 20 else None
        frames.append([person(0.5, 0.5, 0.3, right_wrist=raised) if raised else person(0.5, 0.5, 0.3)])
    assert analyse_bowler_action(frames, FPS) is None


def test_too_few_frames_is_not_enough_to_call_anyone_a_bowler():
    assert analyse_bowler_action(make_delivery(n=8, release=5), FPS) is None


def test_empty_frames_are_tolerated():
    frames = make_delivery()
    frames[10] = []
    frames[11] = []
    assert analyse_bowler_action(frames, FPS) is not None


def test_regression_a_bowler_bent_double_in_the_follow_through_is_not_a_release():
    """Found on a real clip: after delivering, the bowler bends forward, so
    the nose drops to hip level and a trailing hand ends up 'above the
    head' — measured against nose-to-ankle height that read as a release
    with an absurd 1.0 body-heights of raise. Only upright-torso frames may
    count as a release, so a track whose only raised hand is in a bent-over
    frame must not be identified as a bowler at all."""
    frames = []
    for i in range(45):
        cx = 0.20 + 0.40 * i / 44
        p = person(cx, 0.55, 0.30)
        if i == 30:
            # torso horizontal: shoulders level with (and beside) the hips,
            # a wrist swung high behind.
            p[11] = (cx - 0.12, 0.55, 1.0)
            p[12] = (cx - 0.12, 0.54, 1.0)
            p[16] = (cx - 0.20, 0.30, 1.0)
        frames.append([p])
    assert analyse_bowler_action(frames, FPS) is None


def test_regression_a_degenerate_skeleton_with_absurd_raise_is_ignored():
    """Found on a real clip: a skeleton whose shoulders and hips nearly
    coincided made the torso length ~0, so a wrist read as 11-15 torso-
    lengths above the shoulder and the run was reported as a delivery.
    A physically impossible raise is a glitch, not a release."""
    frames = []
    for i in range(45):
        cx = 0.20 + 0.40 * i / 44
        p = person(cx, 0.55, 0.30)
        if i == 30:
            p[11] = (cx, 0.549, 1.0)      # shoulders squashed almost onto the hips (upright, but ~0 torso)
            p[12] = (cx + 0.001, 0.549, 1.0)
            p[16] = (cx, 0.40, 1.0)
        frames.append([p])
    assert analyse_bowler_action(frames, FPS) is None


def test_regression_the_known_batter_is_never_the_bowler():
    """Found on a real clip (verified by looking at the frame): the only
    person in shot was the batter, who stepped out and lifted the bat
    overhead — enough travel plus a raised hand to pass for a bowler.
    With the batter's position known, that track is excluded."""
    frames = []
    for i in range(45):
        cx = 0.30 + 0.10 * i / 44                  # a modest step, > 1.5 torsos
        raised = (cx + 0.01, 0.55 - 0.035 - 0.8 * 0.035, 1.0) if i == 30 else None   # ~0.8 torso-lengths above the shoulder
        frames.append([person(cx, 0.55, 0.10, right_wrist=raised) if raised else person(cx, 0.55, 0.10)])
    assert analyse_bowler_action(frames, FPS) is not None            # without the hint it would be misread
    assert analyse_bowler_action(frames, FPS, batter_hip=(0.35, 0.55)) is None


def test_a_bowler_is_still_found_when_the_batter_is_known_and_elsewhere():
    est = analyse_bowler_action(make_delivery(), FPS, batter_hip=(0.85, 0.50))
    assert est is not None and est.arm_side == "right-arm"

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from post_shot import LOST_TRACK, RAN, STAYED, UNCLEAR, analyse_post_shot_movement
from test_body_vectors import FPS, skeleton

H = 0.30    # torso = 0.35 * H = 0.105


def clip(pre=10, post_frames=None, torso=H):
    """`post_frames` is a list of skeletons (or None) placed after the shot frame; `pre` frames of a
    stationary figure come before it, giving the scale-estimation window something to measure."""
    return [skeleton(h=torso) for _ in range(pre)] + [skeleton(h=torso)] + (post_frames or [])


def stayed_frames(n, torso=H, jitter=0.01):
    import random
    rng = random.Random(3)
    return [skeleton(cx=0.5 + rng.uniform(-jitter, jitter), h=torso) for _ in range(n)]


def ran_frames(n, torso=H, per_frame=0.02):
    return [skeleton(cx=0.5 + per_frame * i, h=torso) for i in range(1, n + 1)]


def test_a_batter_who_barely_moves_is_stayed_at_the_crease():
    frames = clip(post_frames=stayed_frames(50))
    r = analyse_post_shot_movement(frames, FPS, 1.0, shot_frame=10)
    assert r.classification == STAYED
    assert r.net_travel_torsos < 1.5


def test_a_batter_who_moves_a_long_way_in_one_direction_is_ran():
    # 0.02/frame * 50 frames = 1.0 normalised units net = ~9.5 torso-lengths (torso=0.105)
    frames = clip(post_frames=ran_frames(50, per_frame=0.02))
    r = analyse_post_shot_movement(frames, FPS, 1.0, shot_frame=10)
    assert r.classification == RAN
    assert r.net_travel_torsos > 3.0


def test_movement_between_the_two_thresholds_is_unclear_not_forced_either_way():
    # aim for ~2 torso-lengths net (0.105*2 = 0.21 over the window)
    frames = clip(post_frames=ran_frames(50, per_frame=0.0042))
    r = analyse_post_shot_movement(frames, FPS, 1.0, shot_frame=10)
    assert r.classification == UNCLEAR
    assert 1.5 < r.net_travel_torsos < 3.0


def test_the_person_vanishing_soon_after_the_shot_is_lost_track_not_a_wicket():
    missing = [None] * 30
    frames = clip(post_frames=missing)
    r = analyse_post_shot_movement(frames, FPS, 1.0, shot_frame=10)
    assert r.classification == LOST_TRACK
    assert r.net_travel_torsos is None
    assert r.gap_started_at_s is not None and r.gap_started_at_s < 0.5
    assert any("can't tell those apart" in c for c in r.caveats)


def test_a_short_dropout_that_recovers_is_not_lost_track():
    """A few missing frames (a blink of tracking) followed by a real reading must not be mistaken
    for the batter having left."""
    frames = clip(post_frames=[None, None] + stayed_frames(48))
    r = analyse_post_shot_movement(frames, FPS, 1.0, shot_frame=10)
    assert r.classification != LOST_TRACK


def test_a_gap_appearing_well_after_the_shot_is_not_attributed_to_this_delivery():
    """A dropout starting 1.5s after the shot is the next ball's business, not this one's -- treated
    as ordinary movement measurement over what WAS tracked, not a 'lost track' verdict."""
    frames = clip(post_frames=stayed_frames(45) + [None] * 15)
    r = analyse_post_shot_movement(frames, FPS, 1.0, shot_frame=10)
    assert r.classification != LOST_TRACK


def test_every_result_says_plainly_it_cannot_detect_a_wicket_or_count_runs():
    for post in (stayed_frames(50), ran_frames(50, per_frame=0.02), [None] * 30):
        r = analyse_post_shot_movement(clip(post_frames=post), FPS, 1.0, shot_frame=10)
        assert any("cannot detect a wicket" in c for c in r.caveats)


def test_nothing_after_the_shot_returns_none():
    frames = [skeleton() for _ in range(10)]
    assert analyse_post_shot_movement(frames, FPS, 1.0, shot_frame=9) is None


def test_no_established_scale_near_the_shot_returns_none():
    frames = [None] * 40      # never a person at all, including around the shot
    assert analyse_post_shot_movement(frames, FPS, 1.0, shot_frame=20) is None


def test_frame_aspect_ratio_does_not_change_the_classification():
    a = analyse_post_shot_movement(clip(post_frames=ran_frames(50, per_frame=0.02)), FPS, 1.0, shot_frame=10)
    b = analyse_post_shot_movement(clip(post_frames=ran_frames(50, per_frame=0.01)), FPS, 2.0, shot_frame=10)
    assert a.classification == b.classification == RAN

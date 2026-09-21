import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from footwork import (
    BACK, BEHIND_BATTER, BOWLER_ON_LEFT, BOWLER_ON_RIGHT, BOWLERS_END, FRONT, LEFT_HANDED, MINIMAL, RIGHT_HANDED,
    SIDE_ON, UNCLEAR, analyse_footwork, toward_bowler,
)

FPS = 30.0
H = 0.30                      # figure height (normalised); torso = 0.35 * H = 0.105


def frame(cx=0.5, cy=0.55, left_ankle=None, right_ankle=None, hip_dy=0.0, hip_dx=0.0, vis_l=1.0, vis_r=1.0):
    lm = [(cx, cy, 1.0)] * 33
    lm = list(lm)
    lm[0] = (cx, cy - 0.5 * H, 1.0)
    lm[11] = (cx - 0.10 * H, cy - 0.35 * H, 1.0)
    lm[12] = (cx + 0.10 * H, cy - 0.35 * H, 1.0)
    lm[23] = (cx - 0.05 * H + hip_dx, cy + hip_dy, 1.0)
    lm[24] = (cx + 0.05 * H + hip_dx, cy + hip_dy, 1.0)
    la = left_ankle or (cx - 0.08 * H, cy + 0.5 * H)
    ra = right_ankle or (cx + 0.08 * H, cy + 0.5 * H)
    lm[27] = (la[0], la[1], vis_l)
    lm[28] = (ra[0], ra[1], vis_r)
    return lm


def stride_clip(dy_total=-0.05, dx_total=0.0, n=40, start=8, end=25, who="left", hips=0.0, back_dy=0.0):
    """A batter whose `who` ankle steps by (dx_total, dy_total) between frames start..end. Image y grows downward."""
    out = []
    for i in range(n):
        t = min(max((i - start) / (end - start), 0.0), 1.0)
        la = (0.5 - 0.08 * H, 0.55 + 0.5 * H)
        ra = (0.5 + 0.08 * H, 0.55 + 0.5 * H)
        if who == "left":
            la = (la[0] + dx_total * t, la[1] + dy_total * t)
            ra = (ra[0], ra[1] + back_dy * t)
        else:
            ra = (ra[0] + dx_total * t, ra[1] + dy_total * t)
            la = (la[0], la[1] + back_dy * t)
        out.append(frame(left_ankle=la, right_ankle=ra, hip_dy=hips * t))
    return out


def test_toward_bowler_directions():
    assert toward_bowler(BEHIND_BATTER) == (0.0, -1.0)
    assert toward_bowler(BOWLERS_END) == (0.0, 1.0)
    assert toward_bowler(SIDE_ON) is None
    assert toward_bowler(SIDE_ON, BOWLER_ON_LEFT) == (-1.0, 0.0)
    assert toward_bowler(SIDE_ON, BOWLER_ON_RIGHT) == (1.0, 0.0)


def test_a_forward_stride_of_the_front_foot_is_front_foot_and_measured_in_torso_lengths():
    # behind the batter, the bowler is up the picture: stepping toward him = y decreasing. 0.05 / 0.105 = ~0.48 torso-lengths
    r = analyse_footwork(stride_clip(dy_total=-0.05), FPS, 1.0, peak_frame=30, camera=BEHIND_BATTER)
    assert r.footwork_class == FRONT and r.front_stride == pytest.approx(0.05 / (0.35 * H), rel=0.1)
    assert r.observed_foot == "front foot" and r.foreshortened


def test_the_front_foot_is_the_left_for_a_righthander_and_the_right_for_a_lefthander():
    clip = stride_clip(dy_total=-0.05, who="left")
    assert analyse_footwork(clip, FPS, 1.0, 30, BEHIND_BATTER, RIGHT_HANDED).footwork_class == FRONT
    lh = analyse_footwork(clip, FPS, 1.0, 30, BEHIND_BATTER, LEFT_HANDED)
    assert lh.front_ankle == "right ankle" and lh.footwork_class != FRONT       # the LEFT foot stepping is the back foot of a lefty
    assert analyse_footwork(stride_clip(dy_total=-0.05, who="right"), FPS, 1.0, 30, BEHIND_BATTER, LEFT_HANDED).footwork_class == FRONT


def test_the_same_step_reads_the_opposite_way_from_the_bowlers_end():
    """Toward the bowler is down the picture when the batter faces the camera."""
    away = analyse_footwork(stride_clip(dy_total=-0.05), FPS, 1.0, 30, BOWLERS_END)
    toward = analyse_footwork(stride_clip(dy_total=+0.05), FPS, 1.0, 30, BOWLERS_END)
    assert toward.footwork_class == FRONT and away.footwork_class != FRONT


def test_weight_going_back_with_the_back_foot_is_back_foot():
    clip = stride_clip(dy_total=0.0, back_dy=+0.04, hips=+0.03)      # behind: +y = away from the bowler
    r = analyse_footwork(clip, FPS, 1.0, 30, BEHIND_BATTER)
    assert r.footwork_class == BACK and r.back_shift < 0 and r.hip_shift < 0 and r.observed_foot == "back foot"


def test_a_batter_who_stays_put_has_minimal_footwork():
    r = analyse_footwork([frame() for _ in range(40)], FPS, 1.0, 30, BEHIND_BATTER)
    assert r.footwork_class == MINIMAL and r.lead_time_s is None


def test_lead_time_is_how_long_before_the_swing_peak_the_stride_began():
    r = analyse_footwork(stride_clip(dy_total=-0.05, start=8, end=25), FPS, 1.0, peak_frame=30, camera=BEHIND_BATTER)
    # the step runs 0.476 torso-lengths over frames 8-25; it 'has started' once 0.12 of that is covered,
    # i.e. after 0.12/0.476 * 17 = ~4.3 frames, so at about frame 12-13
    assert r.lead_time_s == pytest.approx((30 - 12.3) / FPS, abs=0.07)


def test_side_on_measures_forward_distance_directly_and_uses_higher_thresholds():
    small = stride_clip(dy_total=0.0, dx_total=-0.02)                    # bowler on the left: a 0.19-torso step, below the side-on threshold
    big = stride_clip(dy_total=0.0, dx_total=-0.06)                       # 0.57 torso-lengths
    assert analyse_footwork(small, FPS, 1.0, 30, SIDE_ON, bowler_side=BOWLER_ON_LEFT).footwork_class == MINIMAL
    r = analyse_footwork(big, FPS, 1.0, 30, SIDE_ON, bowler_side=BOWLER_ON_LEFT)
    assert r.footwork_class == FRONT and not r.foreshortened


def test_side_on_without_saying_which_way_the_bowler_is_gives_no_answer_not_a_guess():
    assert analyse_footwork(stride_clip(dx_total=-0.06), FPS, 1.0, 30, SIDE_ON) is None


def test_feet_the_model_cannot_see_are_unclear_not_guessed():
    clip = [frame(vis_l=0.1, vis_r=0.1) for _ in range(40)]
    r = analyse_footwork(clip, FPS, 1.0, 30, BEHIND_BATTER)
    assert r.footwork_class == UNCLEAR and r.observed_foot is None and any("legs overlap" in n for n in r.notes)


def test_too_little_data_returns_none():
    assert analyse_footwork([frame()] * 3, FPS, 1.0, 1, BEHIND_BATTER) is None
    assert analyse_footwork([None] * 40, FPS, 1.0, 20, BEHIND_BATTER) is None


def test_the_step_is_reported_in_normalised_positions_so_it_can_be_drawn():
    r = analyse_footwork(stride_clip(dy_total=-0.05), FPS, 1.0, 30, BEHIND_BATTER)
    assert r.front_at_peak[1] < r.front_start[1]                          # moved up the picture (toward the bowler)
    assert abs(r.front_start[1] - r.front_at_peak[1] - 0.05) < 0.01

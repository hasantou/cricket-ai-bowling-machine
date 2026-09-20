import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from body_vectors import analyse_body_vectors, compass

FPS = 30.0
N = 33


def skeleton(cx=0.5, cy=0.55, h=0.30, right_wrist=None, right_vis=1.0, hip_dx=0.0):
    """Standing figure. Torso length is 0.35*h (shoulders above hips)."""
    lm = [(cx, cy, 1.0)] * N
    lm = list(lm)
    lm[11] = (cx - 0.10 * h, cy - 0.35 * h, 1.0)
    lm[12] = (cx + 0.10 * h, cy - 0.35 * h, 1.0)
    lm[13] = (cx - 0.15 * h, cy - 0.15 * h, 1.0)
    lm[14] = (cx + 0.15 * h, cy - 0.15 * h, 1.0)
    lm[15] = (cx - 0.15 * h, cy, 1.0)
    rw = right_wrist or (cx + 0.15 * h, cy, 1.0)
    lm[16] = (rw[0], rw[1], right_vis)
    lm[23] = (cx - 0.05 * h + hip_dx, cy, 1.0)
    lm[24] = (cx + 0.05 * h + hip_dx, cy, 1.0)
    for i, off in ((25, -0.05), (26, 0.05)):
        lm[i] = (cx + off * h, cy + 0.25 * h, 1.0)
    for i, off in ((27, -0.05), (28, 0.05)):
        lm[i] = (cx + off * h, cy + 0.5 * h, 1.0)
    return lm


def moving_wrist(dx_per_frame, dy_per_frame, n=30, h=0.30, noise=0.0, seed=0):
    rng = random.Random(seed)
    frames = []
    for i in range(n):
        wx = 0.5 + 0.15 * h + dx_per_frame * i + rng.uniform(-noise, noise)
        wy = 0.55 + dy_per_frame * i + rng.uniform(-noise, noise)
        frames.append(skeleton(h=h, right_wrist=(wx, wy, 1.0)))
    return frames


def test_a_wrist_moving_right_reads_as_a_rightward_vector_of_the_right_size():
    # torso = 0.105; 0.0105/frame at 30fps = 0.315/s ... = 3 torso-lengths/s
    r = analyse_body_vectors(moving_wrist(0.0105, 0.0), FPS)
    v = r.vectors_at_peak["right wrist"]
    assert v.direction == "right"
    assert v.speed == pytest.approx(3.0, rel=0.05)
    assert r.peak_hand == "right wrist"


def test_up_on_screen_is_up_not_down():
    """Image y grows downward; the report must not flip it."""
    r = analyse_body_vectors(moving_wrist(0.0, -0.0105), FPS)
    v = r.vectors_at_peak["right wrist"]
    assert v.direction == "up" and v.vy > 0
    r2 = analyse_body_vectors(moving_wrist(0.0, +0.0105), FPS)
    assert r2.vectors_at_peak["right wrist"].direction == "down"


def test_speed_is_in_torso_lengths_so_distance_from_the_camera_does_not_matter():
    near = analyse_body_vectors(moving_wrist(0.0105, 0.0, h=0.30), FPS)
    far = analyse_body_vectors(moving_wrist(0.00525, 0.0, h=0.15), FPS)
    assert far.peak_hand_speed == pytest.approx(near.peak_hand_speed, rel=0.05)


def test_frame_aspect_ratio_is_corrected():
    """The same physical move filmed in a frame twice as wide is half the
    normalised x distance; it must read as the same speed."""
    plain = analyse_body_vectors(moving_wrist(0.0105, 0.0), FPS, aspect=1.0)
    wide = analyse_body_vectors(moving_wrist(0.0105 / 2, 0.0), FPS, aspect=2.0)
    assert wide.peak_hand_speed == pytest.approx(plain.peak_hand_speed, rel=0.05)


def test_pose_jitter_on_a_still_hand_is_well_below_a_real_swing():
    """The failure this module exists to fix: unsmoothed differences of a
    jittering, motionless wrist read as fast as a real swing. The noise here is
    large enough that, raw, a still hand would look as fast as the swing —
    it only separates because of the smoothing (checked by disabling it)."""
    for seed in range(3):
        still = analyse_body_vectors(moving_wrist(0.0, 0.0, noise=0.012, seed=seed), FPS)
        swing = analyse_body_vectors(moving_wrist(0.0105, 0.0, noise=0.012, seed=seed), FPS)
        assert swing.peak_hand_speed > 2.5 * still.peak_hand_speed


def test_a_low_confidence_joint_is_flagged_and_cannot_be_the_peak_hand():
    frames = [skeleton(right_wrist=(0.5 + 0.02 * i, 0.55, 0.2), right_vis=0.2) for i in range(30)]
    r = analyse_body_vectors(frames, FPS)
    assert r is None or r.peak_hand != "right wrist"
    if r is not None:
        assert not r.vectors_at_peak["right wrist"].reliable if "right wrist" in r.vectors_at_peak else True


def test_short_dropouts_are_bridged_but_time_stays_true():
    frames = moving_wrist(0.0105, 0.0)
    frames[10] = None
    frames[11] = None
    r = analyse_body_vectors(frames, FPS)
    assert r is not None
    assert r.vectors_at_peak["right wrist"].speed == pytest.approx(3.0, rel=0.1)


def test_too_little_tracked_person_says_nothing():
    frames = [None] * 25 + moving_wrist(0.0105, 0.0, n=5)
    assert analyse_body_vectors(frames, FPS) is None
    assert analyse_body_vectors([], FPS) is None


def test_hand_path_net_displacement_and_length_over_the_swing():
    """A hand that accelerates from rest, sweeps right and up, then stops."""
    frames = []
    for i in range(40):
        t = min(max(i - 10, 0), 15)             # moves during frames 10-25
        frames.append(skeleton(right_wrist=(0.545 + 0.02 * t, 0.55 - 0.01 * t, 1.0)))
    r = analyse_body_vectors(frames, FPS)
    dx, dy = r.hand_path_net
    assert dx > 0 and dy > 0
    assert r.hand_path_net_direction in ("right", "up-right")
    assert r.hand_path_length > 1.5             # torso-lengths travelled
    assert r.hand_path_start < r.peak_frame < r.hand_path_end


def test_hip_shift_reports_weight_transfer():
    frames = [skeleton(hip_dx=0.004 * i, right_wrist=(0.545, 0.55, 1.0)) for i in range(30)]
    r = analyse_body_vectors(frames, FPS)
    assert r.hip_shift[0] > 0.5 and abs(r.hip_shift[1]) < 0.2


def test_compass_points():
    assert [compass(a) for a in (0, 45, 90, 135, 180, 225, 270, 315)] == [
        "right", "up-right", "up", "up-left", "left", "down-left", "down", "down-right"]


def test_swing_verdict_separates_a_swing_from_taking_guard_and_flags_glitches():
    assert analyse_body_vectors(moving_wrist(0.0105 * 2, 0.0), FPS).swing_verdict == "swing"        # ~6 torso/s
    assert analyse_body_vectors(moving_wrist(0.0, 0.0, noise=0.002), FPS).swing_verdict == "no swing"
    glitch = analyse_body_vectors(moving_wrist(0.0105 * 10, 0.0), FPS)                              # ~30 torso/s
    assert glitch.swing_verdict == "unclear" and not glitch.speed_plausible
    assert any("glitch" in c for c in glitch.caveats)


def test_when_no_wrist_can_be_trusted_the_trunk_stands_in_and_says_so():
    frames = []
    for i in range(30):
        f = skeleton(cx=0.4 + 0.006 * i, right_vis=0.1)
        f[15] = (f[15][0], f[15][1], 0.1)                     # left wrist hidden too
        frames.append(f)
    r = analyse_body_vectors(frames, FPS)
    assert r is not None and r.peak_hand == "trunk"
    assert any("TRUNK" in c for c in r.caveats)


def _walking_body(dx_per_frame, n=30):
    """The whole figure translates together (walking in / stepping out / camera pan);
    the hands stay fixed relative to the body — nobody swings."""
    return [skeleton(cx=0.3 + dx_per_frame * i) for i in range(n)]


def test_regression_the_whole_body_moving_is_not_a_swing():
    """Found on a real clip: a batter drifting out of frame moved every joint at
    ~10 torso-lengths/s and was called a swing. Hands measured against the body
    do not move at all here."""
    r = analyse_body_vectors(_walking_body(0.0105 * 3), FPS)
    assert r.swing_verdict == "no swing"
    assert r.peak_hand_speed < 1.0
    assert r.body_speed_at_peak > 5.0
    assert any("whole body was moving" in c for c in r.caveats)


def test_a_real_swing_is_still_a_swing_while_the_body_walks():
    frames = []
    for i in range(30):
        f = skeleton(cx=0.3 + 0.0105 * i, right_wrist=(0.345 + 0.0105 * i + 0.0105 * 2 * i, 0.55, 1.0))
        frames.append(f)
    r = analyse_body_vectors(frames, FPS)
    assert r.swing_verdict == "swing"

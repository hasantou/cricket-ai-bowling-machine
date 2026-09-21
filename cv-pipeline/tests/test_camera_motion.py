import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import pytest
from camera_motion import CameraMotionTracker, build_path, stabilize, stabilize_pose

W, H = 320, 240


def world(seed=0):
    rng = np.random.default_rng(seed)
    bg = rng.integers(0, 255, (H + 40, 1200), dtype=np.uint8)
    bg = cv2.GaussianBlur(bg, (0, 0), 2.0)
    return cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)


def view(bg, x0, y0=20, blob=None):
    f = bg[y0:y0 + H, x0:x0 + W].copy()
    if blob is not None:
        bx, by = blob
        cv2.rectangle(f, (bx, by), (bx + 40, by + 90), (255, 255, 255), -1)      # a "person" moving on its own
    return f


def run(tracker, frames):
    return [tracker.push(f) for f in frames]


def test_a_known_pan_is_recovered_in_size_and_sign():
    bg = world()
    speed = 4                                   # camera moves right 4 px/frame -> scene moves left
    shifts = run(CameraMotionTracker(), [view(bg, 100 + speed * i) for i in range(20)])
    measured = [s[0] for s in shifts[1:]]
    assert all(s is not None for s in shifts)
    assert np.median(measured) == pytest.approx(-speed / W, abs=0.003)


def test_a_tilt_is_recovered_too():
    bg = world(1)
    shifts = run(CameraMotionTracker(), [view(bg, 300, 5 + 2 * i) for i in range(15)])
    assert np.median([s[1] for s in shifts[1:]]) == pytest.approx(-2 / H, abs=0.003)


def test_a_still_camera_reads_as_no_motion():
    bg = world(2)
    shifts = run(CameraMotionTracker(), [view(bg, 200)] * 10)
    assert max(abs(s[0]) + abs(s[1]) for s in shifts) < 0.002


def test_a_moving_foreground_person_does_not_bias_the_camera_estimate():
    """The point of the median: a person running across the frame is not the camera."""
    bg = world(3)
    frames = [view(bg, 200, blob=(60 + 12 * i, 60)) for i in range(15)]      # still camera, fast runner
    shifts = run(CameraMotionTracker(), frames)
    assert max(abs(s[0]) for s in shifts) < 0.004


def test_a_featureless_frame_is_reported_unknown_not_guessed():
    t = CameraMotionTracker()
    flat = np.full((H, W, 3), 120, np.uint8)
    out = run(t, [flat, flat, flat])
    assert out[0] == (0.0, 0.0) and out[1] is None and t.unknown_frames >= 1


def test_the_path_sums_shifts_and_flags_a_moving_camera():
    path = build_path([None, (-0.01, 0.0)] + [(-0.01, 0.0)] * 9)
    assert path.moving and path.x[-1] == pytest.approx(-0.10) and any("camera moved" in n for n in path.notes)


def test_a_tripod_is_not_treated_as_moving():
    path = build_path([None] + [(0.0005, -0.0003)] * 30)
    assert not path.moving and path.notes == []


def test_the_path_restarts_at_a_cut_so_two_shots_are_never_compared():
    shifts = [None] + [(-0.01, 0.0)] * 5 + [(0.4, 0.3)] + [(-0.01, 0.0)] * 5      # frame 6 is a cut: its "shift" is garbage
    path = build_path(shifts, cut_frames=[6])
    assert path.x[6] == path.x[5]
    assert path.x[-1] == pytest.approx(path.x[5] - 0.05)


def test_stabilising_makes_a_world_stationary_person_stationary():
    """A person fixed in the world appears to slide across the frame as the camera pans; after
    compensation their coordinates must be constant."""
    world_x = 0.6
    cam = [-0.01 * i for i in range(30)]                       # camera path (background displacement)
    path = build_path([None] + [(-0.01, 0.0)] * 29)
    pose = lambda x: [(x, 0.5, 1.0)] * 33
    apparent = [pose(world_x + c) for c in cam]                # what the pose model reports in the image
    fixed = stabilize(apparent, path)
    assert all(p[0][0] == pytest.approx(world_x, abs=1e-9) for p in fixed)


def test_stabilising_leaves_tripod_footage_untouched():
    path = build_path([None] + [(0.0002, 0.0)] * 20)
    poses = [[(0.5 + 0.001 * i, 0.5, 1.0)] * 33 for i in range(21)]
    assert stabilize(poses, path) == poses


def test_missing_frames_stay_missing_and_visibility_is_preserved():
    path = build_path([None] + [(-0.01, 0.0)] * 5)
    out = stabilize([[(0.5, 0.5, 0.7)] * 33, None, [(0.5, 0.5, 0.2)] * 33], path)
    assert out[1] is None and out[0][0][2] == 0.7 and out[2][0][2] == 0.2
    assert stabilize_pose([(0.5, 0.5, 0.9)], 0.1, -0.2) == [(0.4, 0.7, 0.9)]

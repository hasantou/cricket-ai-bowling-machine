import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from cut_detection import (
    DUPLICATE_LEVEL, duplicate_fraction, effective_fps, find_cuts_from_changes, frame_changes, thumbnail,
)


def test_a_single_isolated_jump_after_calm_is_a_cut():
    changes = [0.0] + [0.5] * 20 + [40.0] + [0.6] * 20
    assert find_cuts_from_changes(changes) == [21]


def test_ordinary_motion_is_not_a_cut():
    rng = np.random.default_rng(0)
    changes = [0.0] + list(rng.uniform(0.5, 6.0, 200))
    assert find_cuts_from_changes(changes) == []


def test_a_sustained_whip_pan_is_not_a_cut_even_though_every_frame_changes_a_lot():
    """Real TikTok clip: every new frame of a fast pan differs by 18-32 grey levels."""
    changes = [0.0] + [0.5] * 10 + [25.0, 21.0, 18.0, 28.0, 27.0, 19.0, 31.0, 20.0, 32.0, 21.0] * 3
    assert find_cuts_from_changes(changes) == []


def test_a_cut_is_still_found_when_half_the_frames_are_duplicates():
    """A '60 fps' file that is really 30 fps: alternate frames are repeats (change ~0.1). The
    repeats must not be mistaken for a calm shot or hide the cut."""
    calm = [0.1, 0.9] * 12
    changes = [0.0] + calm + [38.0] + [0.1, 0.8] * 12
    assert find_cuts_from_changes(changes) == [len(calm) + 1]


def test_two_cuts_a_frame_apart_are_one_event():
    changes = [0.0] + [0.5] * 20 + [40.0, 39.0] + [0.6] * 20
    assert len(find_cuts_from_changes(changes)) == 1


def test_duplicate_fraction_and_effective_fps_for_a_doubled_frame_rate():
    changes = [0.0] + [0.1, 3.0] * 50
    assert abs(duplicate_fraction(changes) - 0.5) < 0.02
    assert abs(effective_fps(60.0, changes) - 30.0) < 1.5


def test_a_few_chance_repeats_do_not_change_the_frame_rate():
    changes = [0.0] + [2.0] * 45 + [0.1] * 5
    assert effective_fps(30.0, changes) == 30.0


def test_frame_changes_and_thumbnails_on_real_images():
    a = np.full((120, 200, 3), 60, np.uint8)
    b = np.full((120, 200, 3), 160, np.uint8)
    ch = frame_changes([thumbnail(a), thumbnail(a), thumbnail(b)])
    assert ch[0] == 0.0 and ch[1] < DUPLICATE_LEVEL and 90 < ch[2] < 110
    assert find_cuts_from_changes([0.0] + [0.3] * 10 + [ch[2]] + [0.3] * 10) == [11]


def test_a_still_scene_with_tiny_changes_is_not_mistaken_for_repeated_frames():
    """Regression: a tripod shot with small movement changes ~0.1-0.3 every frame; that is a quiet
    picture, not a doubled frame rate."""
    changes = [0.0] + [0.15, 0.2, 0.1, 0.25, 0.12] * 20
    assert duplicate_fraction(changes) == 0.0
    assert effective_fps(60.0, changes) == 60.0

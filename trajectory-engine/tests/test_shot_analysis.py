import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from cricket_trajectory.net_outcome import ExitVelocity
from cricket_trajectory.shot_analysis import analyse_shot, region_for


def ev(speed, azimuth_deg, elevation_deg):
    th, ph = math.radians(elevation_deg), math.radians(azimuth_deg)
    return ExitVelocity(
        speed * math.cos(th) * math.cos(ph), speed * math.cos(th) * math.sin(ph), speed * math.sin(th),
    )


@pytest.mark.parametrize("az,expected", [
    (0, "straight"), (14, "straight"), (-14, "straight"),
    (25, "mid-off"), (-25, "mid-on"),
    (50, "cover"), (-50, "midwicket"),
    (85, "point"), (-85, "square leg"),
    (130, "third man / gully (behind square)"), (-130, "fine leg (behind square)"),
])
def test_regions(az, expected):
    assert region_for(az) == expected


@pytest.mark.parametrize("az,length,expected", [
    (0, "full", "Straight drive"),
    (25, "good length", "Off drive"),
    (-25, "full", "On drive"),
    (50, "full", "Cover drive"),
    (-50, "short of a length", "Pull"),
    (-50, "short (bouncer territory)", "Hook"),
    (85, "short of a length", "Square cut"),
    (-85, "short (bouncer territory)", "Hook"),
    (130, "short of a length", "Late cut"),
])
def test_attacking_shots_named_from_direction_and_length(az, length, expected):
    assert analyse_shot(ev(25.0, az, 5.0), length).shot == expected


def test_a_lofted_drive_is_called_lofted():
    assert analyse_shot(ev(30.0, 0, 16.0), "full").shot == "Lofted drive"


def test_a_slow_ball_off_the_bat_is_a_defensive_block_whatever_the_direction():
    assert analyse_shot(ev(2.0, 50, 0.0), "good length").shot == "Forward defence"
    assert analyse_shot(ev(2.0, 50, 0.0), "short of a length").shot == "Back-foot defence"


def test_a_steep_launch_is_a_mistimed_hit_not_a_named_drive():
    a = analyse_shot(ev(25.0, 0, 35.0), "full")
    assert a.shot == "mistimed / skied hit"
    assert a.trajectory == "skied"


def test_no_contact_is_no_shot():
    a = analyse_shot(None, "full")
    assert a.shot == "no shot / missed" and a.confidence == "n/a"


def test_footwork_is_inferred_from_length_and_says_so():
    a = analyse_shot(ev(25.0, 50, 5.0), "full")
    assert a.footwork == "front foot"
    assert "inferred" in a.footwork_source and a.confidence == "approximate"


def test_observed_footwork_overrides_the_inference_and_raises_confidence():
    """A full ball played off the back foot through cover is not a cover drive."""
    a = analyse_shot(ev(25.0, 50, 5.0), "full", footwork="back foot")
    assert a.shot == "Back-foot punch"
    assert a.footwork_source.startswith("observed") and a.confidence == "firm"


def test_rows_are_present_and_the_off_the_bat_speed_is_in_kmh():
    rows = dict(analyse_shot(ev(25.0, 0, 5.0), "full").rows())
    assert rows["Off the bat"] == "90 km/h"
    assert {"Shot", "Where it went", "Trajectory", "Footwork", "Confidence"} <= set(rows)


# ---- the shot vocabulary (transcribed from the owner's saved shot list) ----
import itertools
from cricket_trajectory.shot_vocabulary import (
    NON_SHOT_OUTCOMES, SENSOR, SHOT_VOCABULARY, by_detectability, lookup,
)

_LENGTHS = ["full toss", "yorker", "full", "good length", "short of a length", "short (bouncer territory)"]


def _every_output():
    seen = set()
    for az, elev, speed, length, foot in itertools.product(
        range(-170, 171, 10), (0.0, 5.0, 15.0, 30.0), (2.0, 12.0, 30.0), _LENGTHS, (None, "front foot", "back foot"),
    ):
        seen.add(analyse_shot(ev(speed, az, elev), length, footwork=foot).shot)
    seen.add(analyse_shot(None, "full").shot)
    return seen


def test_every_name_the_classifier_can_output_is_in_the_vocabulary_or_a_declared_non_shot():
    unknown = [n for n in _every_output() if lookup(n) is None and n not in NON_SHOT_OUTCOMES]
    assert unknown == []


def test_every_shot_the_vocabulary_says_is_sensor_detectable_really_can_be_produced():
    """The other direction: don't claim a shot is detectable if no input can
    ever produce it."""
    produced = {n.lower() for n in _every_output()}
    claimed = {n.lower() for n in by_detectability()[SENSOR]}
    assert claimed - produced == set()


def test_shots_that_need_information_the_sensor_lacks_are_never_claimed():
    """Sweeps, scoops, switch hits etc. must not be output — nothing here can tell them apart."""
    produced = {n.lower() for n in _every_output()}
    unsupported = {e.name.lower() for e in SHOT_VOCABULARY if e.detectable != SENSOR}
    assert produced & unsupported == set()

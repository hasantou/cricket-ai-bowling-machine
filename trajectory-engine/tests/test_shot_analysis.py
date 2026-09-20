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
    (0, "full", "straight drive"),
    (25, "good length", "off drive"),
    (-25, "full", "on drive"),
    (50, "full", "cover drive"),
    (-50, "short of a length", "pull"),
    (-50, "short (bouncer territory)", "hook"),
    (85, "short of a length", "square cut"),
    (-85, "short (bouncer territory)", "hook"),
    (130, "short of a length", "late cut"),
])
def test_attacking_shots_named_from_direction_and_length(az, length, expected):
    assert analyse_shot(ev(25.0, az, 5.0), length).shot == expected


def test_a_lofted_drive_is_called_lofted():
    assert analyse_shot(ev(30.0, 0, 16.0), "full").shot == "lofted straight drive"


def test_a_slow_ball_off_the_bat_is_a_defensive_block_whatever_the_direction():
    assert analyse_shot(ev(2.0, 50, 0.0), "good length").shot == "forward defence (block)"
    assert analyse_shot(ev(2.0, 50, 0.0), "short of a length").shot == "back-foot defence (block)"


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
    assert a.shot == "back-foot punch through cover"
    assert a.footwork_source.startswith("observed") and a.confidence == "firm"


def test_rows_are_present_and_the_off_the_bat_speed_is_in_kmh():
    rows = dict(analyse_shot(ev(25.0, 0, 5.0), "full").rows())
    assert rows["Off the bat"] == "90 km/h"
    assert {"Shot", "Where it went", "Trajectory", "Footwork", "Confidence"} <= set(rows)

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from cricket_trajectory import constants as c
from cricket_trajectory.ball import BallProperties, Delivery, Environment
from cricket_trajectory.delivery_report import (
    build_delivery_report, classify_length, classify_line, classify_swing, describe_spin,
    GOOD_MAX_FROM_STUMPS_M, YORKER_MAX_FROM_STUMPS_M,
)
from cricket_trajectory.simulate import run_simulation

BALL = BallProperties()
ENV = Environment()


def test_length_bands_in_order_from_the_batters_stumps():
    crease = c.PITCH_LENGTH_M
    assert classify_length(crease - 1.0)[1] == "yorker"
    assert classify_length(crease - 3.0)[1] == "full"
    assert classify_length(crease - 6.0)[1] == "good length"
    assert classify_length(crease - 8.5)[1] == "short of a length"
    assert classify_length(crease - 11.0)[1] == "short (bouncer territory)"


def test_a_ball_still_airborne_at_the_crease_is_a_full_toss():
    assert classify_length(c.PITCH_LENGTH_M + 0.5)[1] == "full toss"


def test_length_reports_distance_from_the_stumps_not_from_the_bowler():
    from_stumps, _ = classify_length(c.PITCH_LENGTH_M - 6.0)
    assert from_stumps == pytest.approx(6.0)


def test_line_bands():
    assert classify_line(0.0) == "on the stumps"
    assert classify_line(0.3) == "outside off (corridor)"
    assert classify_line(1.0) == "wide outside off"
    assert classify_line(-0.3) == "leg side"
    assert classify_line(-1.0) == "well down the leg side"


def test_swing_direction_matches_the_project_axis_convention():
    # y > 0 is the off side for a right-handed batter (ball.py), so drift
    # toward +y is swinging AWAY from that batter.
    assert classify_swing(0.10)[1].startswith("swings away")
    assert classify_swing(-0.10)[1].startswith("swings in")
    assert classify_swing(0.01)[1].startswith("straight")


def test_spin_axes_match_delivery_spin_vector_convention():
    for spin_type, expected in (
        ("backspin", "backspin"), ("topspin", "topspin"),
        ("offspin", "off-spin"), ("legspin", "leg-spin"),
    ):
        rpm, description = describe_spin(Delivery.spin_vector(spin_type, 600.0))
        assert description == expected
        assert rpm == pytest.approx(600.0, rel=1e-6)


def test_no_spin_is_reported_as_no_spin():
    assert describe_spin((0.0, 0.0, 0.0)) == (0.0, "no spin")


def test_a_real_simulated_delivery_produces_a_coherent_report():
    delivery = Delivery(speed_mps=c.kmh_to_ms(130.0), seam_angle_deg=20.0,
                         spin_rad_s=Delivery.spin_vector("none", 0.0), label="test fast ball")
    result = run_simulation(BALL, ENV, delivery)
    report = build_delivery_report(delivery, result)

    assert report.commanded_speed_kmh == pytest.approx(130.0, rel=1e-6)
    assert report.measured_speed_kmh is None      # no sensor has reported yet
    assert report.speed_error_kmh is None
    assert report.speed_at_pitch_kmh < report.commanded_speed_kmh   # air drag slowed it
    assert report.flight_time_s > 0
    assert report.pitch_from_stumps_m == pytest.approx(
        c.PITCH_LENGTH_M - result.landing_point_m[0], rel=1e-9
    )


def test_sensor_measurement_adds_a_confirmed_speed_and_error():
    delivery = Delivery(speed_mps=c.kmh_to_ms(120.0), seam_angle_deg=10.0, label="x")
    result = run_simulation(BALL, ENV, delivery)
    report = build_delivery_report(delivery, result, measured_release_speed_mps=c.kmh_to_ms(115.0))

    assert report.measured_speed_kmh == pytest.approx(115.0, rel=1e-6)
    assert report.speed_error_kmh == pytest.approx(-5.0, rel=1e-6)
    labels = [label for label, _ in report.rows()]
    assert "Pace (sensor-confirmed)" in labels


def test_rows_omit_the_sensor_line_until_a_sensor_has_reported():
    delivery = Delivery(speed_mps=c.kmh_to_ms(120.0), seam_angle_deg=10.0, label="x")
    result = run_simulation(BALL, ENV, delivery)
    labels = [label for label, _ in build_delivery_report(delivery, result).rows()]
    assert "Pace (sensor-confirmed)" not in labels
    assert labels[0] == "Pace (commanded)"


def test_a_much_faster_ball_pitches_further_up_the_pitch_than_a_slow_one():
    """Sanity check that the report tracks real physics, not just labels."""
    slow = Delivery(speed_mps=c.kmh_to_ms(75.0), seam_angle_deg=0.0, label="slow")
    fast = Delivery(speed_mps=c.kmh_to_ms(145.0), seam_angle_deg=0.0, label="fast")
    slow_report = build_delivery_report(slow, run_simulation(BALL, ENV, slow))
    fast_report = build_delivery_report(fast, run_simulation(BALL, ENV, fast))
    # Same release angle: the faster ball travels further before it drops.
    assert fast_report.pitch_from_stumps_m < slow_report.pitch_from_stumps_m
    assert fast_report.flight_time_s < slow_report.flight_time_s


def test_named_thresholds_are_ordered():
    assert YORKER_MAX_FROM_STUMPS_M < GOOD_MAX_FROM_STUMPS_M


def test_a_full_toss_never_reports_a_negative_distance_from_the_stumps():
    """Regression: an early version printed 'pitches -0.4 m from the
    stumps' for a ball that never bounced before the batter."""
    fast = Delivery(speed_mps=c.kmh_to_ms(145.0), seam_angle_deg=0.0, label="fast")
    report = build_delivery_report(fast, run_simulation(BALL, ENV, fast))
    assert report.length_label == "full toss"
    length_row = dict(report.rows())["Length"]
    assert "without bouncing" in length_row
    assert "-" not in length_row.split("—")[1]


def test_the_report_shows_the_bounce_aware_call_and_would_hit_the_stumps_when_given_an_assessment():
    from cricket_trajectory import BallProperties, Environment, assess_delivery, run_simulation
    ball, env = BallProperties(), Environment()
    d = Delivery(speed_mps=110 / 3.6, vertical_launch_deg=-3.0, horizontal_launch_deg=3.0, label="wide one")
    rep = build_delivery_report(d, run_simulation(ball, env, d), assessment=assess_delivery(ball, env, d))
    rows = dict(rep.rows())
    assert rows["Legality (at the batter)"].startswith("WIDE") and "off side" in rows["Legality (at the batter)"]
    assert rows["Would hit the stumps"].startswith("no")
    assert rows["No-ball conditions (counted, not scored)"] == "none"
    assert rep.legality == "wide"


def test_without_an_assessment_the_old_single_legality_row_is_unchanged():
    from cricket_trajectory import BallProperties, Environment, run_simulation
    d = Delivery(speed_mps=140 / 3.6, vertical_launch_deg=-3.0)
    rows = dict(build_delivery_report(d, run_simulation(BallProperties(), Environment(), d)).rows())
    assert rows["Legality"] == "fair delivery" and "Would hit the stumps" not in rows

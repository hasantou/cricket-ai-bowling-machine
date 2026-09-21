import os
import sys

HERE = os.path.dirname(__file__)
for rel in ("..", os.path.join("..", "..", "trajectory-engine")):
    sys.path.insert(0, os.path.join(HERE, rel))

import pytest
from cricket_trajectory import BallProperties, Delivery, Environment, assess_delivery
from machine_control.crease_sensor import (
    LegalityObserver, NoCrossingDetected, SimulatedCreaseSensor, CreaseReading, parse_crease_line, POPPING, WICKET,
)


def predicted(**kw):
    return assess_delivery(BallProperties(), Environment(), Delivery(**kw))


FAIR = dict(speed_mps=140 / 3.6, vertical_launch_deg=-3.0)
WIDE = dict(speed_mps=110 / 3.6, vertical_launch_deg=-3.0, horizontal_launch_deg=3.0)


# ---- the serial line ----
def test_a_well_formed_crease_line_is_parsed_into_metres_and_seconds():
    r = parse_crease_line("CREASE wicket 1234567 -120 950\n")
    assert (r.plane, r.t_s, r.y_m, r.z_m) == (WICKET, 1.234567, -0.12, 0.95)


@pytest.mark.parametrize("bad", [
    "", "OK", "CREASE wicket 1 2", "CREASE wicket 1 2 3 4", "CREASE middle 1 2 3", "CREASE wicket a b c",
    "CREASE wicket 1 99999 0", "CREASE wicket 1 0 -900",
])
def test_garbled_or_impossible_lines_are_rejected_not_guessed(bad):
    with pytest.raises(ValueError):
        parse_crease_line(bad)


# ---- the simulated sensor ----
def test_the_simulated_sensor_reports_the_armed_crossing_with_noise():
    p = predicted(**FAIR)
    s = SimulatedCreaseSensor(noise_lateral_m=0.01, seed=1)
    s.arm(WICKET, p.at_wicket)
    r = s.measure_crossing(WICKET)
    assert abs(r.y_m - p.at_wicket.y_m) < 0.05 and r.plane == WICKET


def test_nothing_armed_or_a_miss_raises_no_crossing():
    s = SimulatedCreaseSensor(seed=1)
    with pytest.raises(NoCrossingDetected):
        s.measure_crossing(WICKET)
    s.arm(WICKET, predicted(**FAIR).at_wicket)
    with pytest.raises(NoCrossingDetected):
        SimulatedCreaseSensor(miss_rate=1.0, seed=1).measure_crossing(WICKET)


# ---- deciding the call ----
def test_without_a_sensor_reading_the_call_is_the_models_and_says_so():
    d = LegalityObserver().decide(predicted(**WIDE))
    assert d.call == "wide" and d.basis.startswith("physics model") and d.agrees_with_model
    assert any("No sensor reading" in n for n in d.notes)


def test_the_measured_position_overrides_the_model_when_they_disagree():
    """The model says fair (0.56 m out); the sensor measures the ball 1.0 m out: wide, and the disagreement is flagged."""
    p = predicted(speed_mps=140 / 3.6, vertical_launch_deg=-3.0, horizontal_launch_deg=1.6)
    assert p.call is None
    d = LegalityObserver().decide(p, wicket=CreaseReading(WICKET, 0.6, 1.0, 0.3))
    assert d.call == "wide" and d.basis == "sensor" and not d.agrees_with_model and d.model_call is None
    assert "off side" in d.wide_reasons[0]


def test_the_sensor_can_also_clear_a_ball_the_model_thought_was_wide():
    p = predicted(**WIDE)
    d = LegalityObserver().decide(p, wicket=CreaseReading(WICKET, 0.7, 0.60, 0.3))
    assert d.call is None and not d.agrees_with_model and d.model_call == "wide"


def test_a_head_high_ball_measured_at_the_popping_crease_is_a_wide():
    p = predicted(**FAIR)
    d = LegalityObserver().decide(p, wicket=CreaseReading(WICKET, 0.7, 0.0, 1.0), popping=CreaseReading(POPPING, 0.68, 0.0, 2.1))
    assert d.call == "wide" and any("head height" in r for r in d.wide_reasons)


def test_borderline_calls_are_marked_borderline():
    p = predicted(**FAIR)
    d = LegalityObserver().decide(p, wicket=CreaseReading(WICKET, 0.6, 0.88, 0.3))
    assert d.call is None and d.borderline


def test_would_hit_the_stumps_follows_the_measured_position():
    p = predicted(**FAIR)
    assert LegalityObserver().decide(p, wicket=CreaseReading(WICKET, 0.6, 0.02, 0.3)).hits_stumps
    assert not LegalityObserver().decide(p, wicket=CreaseReading(WICKET, 0.6, 0.30, 0.3)).hits_stumps


# ---- the calibration by-product ----
def test_model_errors_accumulate_into_calibration_data():
    obs = LegalityObserver()
    p = predicted(**FAIR)
    for dy in (0.02, 0.03, 0.01):
        obs.decide(p, wicket=CreaseReading(WICKET, 0.6, p.at_wicket.y_m + dy, p.at_wicket.z_m - 0.05))
    s = obs.model_error_summary()[WICKET]
    assert s["n"] == 3 and s["mean_dy_m"] == pytest.approx(0.02, abs=1e-9) and s["mean_dz_m"] == pytest.approx(-0.05)
    assert POPPING not in obs.model_error_summary()


def test_no_readings_no_calibration_claims():
    obs = LegalityObserver()
    obs.decide(predicted(**FAIR))
    assert obs.model_error_summary() == {}

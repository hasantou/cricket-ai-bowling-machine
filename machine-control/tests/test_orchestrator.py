import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "trajectory-engine"))

from cricket_trajectory import BallProperties, PlayerProfile
from machine_control.controller import SimulatedMachineController
from machine_control.safety import SafeMachineController, SafetyLimits
from machine_control.outcome_observer import ScriptedOutcomeObserver
from machine_control.orchestrator import MachineOrchestrator

N_DELIVERIES = 8


def make_orchestrator(limits=None, seed=1):
    inner = SimulatedMachineController(cycle_time_s=0.0)
    safe = SafeMachineController(inner, limits)
    profile = PlayerProfile(name="Test Player", rating=1000.0)
    ball = BallProperties()
    # N_DELIVERIES is a safe upper bound on how many times observe() can
    # be called - illegal (wide/no-ball) attempts never consume one, so
    # this never runs out regardless of the legal/illegal mix.
    observer = ScriptedOutcomeObserver(["defended"] * N_DELIVERIES)
    orchestrator = MachineOrchestrator(safe, observer, profile, ball, rng=random.Random(seed))
    return orchestrator, safe, inner, profile


def test_runs_the_requested_number_of_deliveries():
    orchestrator, safe, inner, profile = make_orchestrator()
    events = orchestrator.run(N_DELIVERIES)
    assert len(events) == N_DELIVERIES
    assert all(e.stopped_reason is None for e in events)


def test_every_event_has_a_real_wheel_rpm_command():
    orchestrator, safe, inner, profile = make_orchestrator()
    events = orchestrator.run(N_DELIVERIES)
    for e in events:
        assert e.wheel1_rpm > 0
        assert e.wheel2_rpm > 0
    assert len(inner.history) == N_DELIVERIES


def test_legal_deliveries_get_an_outcome_illegal_ones_dont():
    orchestrator, safe, inner, profile = make_orchestrator()
    events = orchestrator.run(N_DELIVERIES)
    for e in events:
        if e.legality is None:
            assert e.outcome is not None
        else:
            assert e.outcome is None


def test_scorecard_accumulates_across_the_run():
    orchestrator, safe, inner, profile = make_orchestrator()
    orchestrator.run(N_DELIVERIES)
    # every legal ball was scripted as "defended" (0 runs, no wicket) -
    # the scorecard's legal-ball count should match how many events had
    # no legality flag (fair balls), and total runs should stay 0.
    legal_events = [e for e in orchestrator.events if e.legality is None]
    assert orchestrator.card.legal_balls == len(legal_events)
    assert orchestrator.card.total_runs == orchestrator.card.extras  # only wide/no-ball extras, no runs off the bat


def test_profile_rating_changes_after_legal_deliveries():
    orchestrator, safe, inner, profile = make_orchestrator()
    start_rating = profile.rating
    orchestrator.run(N_DELIVERIES)
    legal_events = [e for e in orchestrator.events if e.legality is None]
    if legal_events:  # extremely unlikely to be empty at this seed, but don't assume
        assert profile.rating != start_rating


def test_stops_immediately_on_a_safety_violation():
    # max_wheel_rpm=0 guarantees every single command gets rejected before
    # ever reaching the simulated machine.
    orchestrator, safe, inner, profile = make_orchestrator(limits=SafetyLimits(max_wheel_rpm=0.0))
    events = orchestrator.run(N_DELIVERIES)
    assert len(events) == 1
    assert events[0].stopped_reason is not None
    assert len(inner.history) == 0  # never reached the underlying controller


def test_run_one_delivery_raises_directly_for_a_caller_that_wants_to_handle_it():
    orchestrator, safe, inner, profile = make_orchestrator(limits=SafetyLimits(max_wheel_rpm=0.0))
    from machine_control.safety import SafetyViolation
    import pytest
    with pytest.raises(SafetyViolation):
        orchestrator.run_one_delivery()

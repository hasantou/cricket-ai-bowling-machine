"""
The autonomous control loop: decide a delivery, command the machine
through the safety layer, wait for it to actually happen, sense the
outcome, update the player's rating and scorecard, repeat.

This is the genuinely new piece — app.py today is a human clicking
through a form once per ball; nothing before this ties decide -> command
-> sense -> score into one running cycle. It's built and tested entirely
against SimulatedMachineController and ScriptedOutcomeObserver, so it's
real, runnable software, ready for a real controller and a real observer
to be substituted in later without this loop itself changing at all.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "trajectory-engine"))

from cricket_trajectory import (
    BallProperties, Environment, PlayerProfile, Scorecard, run_simulation, classify_delivery_legality,
)
from cricket_trajectory.adaptive import suggest_next_delivery
from cricket_trajectory.machine import WheelMachine

from .safety import SafeMachineController, SafetyViolation
from .outcome_observer import OutcomeObserver


@dataclass
class OrchestratorEvent:
    """One iteration's result — what a real coach-facing dashboard would
    subscribe to for live monitoring."""
    delivery_label: str
    wheel1_rpm: float
    wheel2_rpm: float
    legality: Optional[str] = None
    outcome: Optional[str] = None
    rating_after: float = 0.0
    stopped_reason: Optional[str] = None


class MachineOrchestrator:
    """Runs the decide -> command -> sense -> score cycle. Stops
    immediately and records why on any SafetyViolation or a machine that
    never reports ready — it does not retry, back off, or attempt to
    recover a bad command on its own; a real deployment surfaces that to
    a human, it doesn't paper over it."""

    def __init__(self,
                 controller: SafeMachineController,
                 observer: OutcomeObserver,
                 profile: PlayerProfile,
                 ball: BallProperties,
                 card: Optional[Scorecard] = None,
                 env: Optional[Environment] = None,
                 machine: Optional[WheelMachine] = None,
                 poll_interval_s: float = 0.01,
                 rng=None):
        self.controller = controller
        self.observer = observer
        self.profile = profile
        self.ball = ball
        self.card = card or Scorecard(batter_name=profile.name)
        self.env = env or Environment()
        self.machine = machine or WheelMachine()
        self.poll_interval_s = poll_interval_s
        self.rng = rng  # optional, for deterministic tests - see suggest_next_delivery
        self.events: List[OrchestratorEvent] = []

    def _wait_until_ready(self, timeout_s: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.controller.is_ready():
                return True
            time.sleep(self.poll_interval_s)
        return False

    def run_one_delivery(self) -> OrchestratorEvent:
        """One full cycle. Raises SafetyViolation if the machine refuses
        the command outright — callers (run(), or a real deployment)
        decide whether that ends the session."""
        kwargs = {"rng": self.rng} if self.rng is not None else {}
        next_ball = suggest_next_delivery(self.profile, self.ball, **kwargs)
        spin_mag = sum(w * w for w in next_ball.spin_rad_s) ** 0.5
        rpm1, rpm2 = self.machine.wheel_rpms_for_delivery(next_ball.speed_mps, spin_mag, self.ball)

        self.controller.set_delivery(rpm1, rpm2)  # SafetyViolation propagates to the caller

        if not self._wait_until_ready():
            event = OrchestratorEvent(
                delivery_label=next_ball.label, wheel1_rpm=rpm1, wheel2_rpm=rpm2,
                rating_after=self.profile.rating,
                stopped_reason="machine never reported ready — timed out waiting for the delivery cycle",
            )
            self.events.append(event)
            return event

        sim_result = run_simulation(self.ball, self.env, next_ball)
        legality = classify_delivery_legality(sim_result)

        if legality is not None:
            self.card.record_ball(self.profile, self.ball, next_ball, sim_result)
            outcome = None
        else:
            outcome = self.observer.observe(next_ball, self.ball)
            self.card.record_ball(self.profile, self.ball, next_ball, sim_result, outcome=outcome)

        event = OrchestratorEvent(
            delivery_label=next_ball.label, wheel1_rpm=rpm1, wheel2_rpm=rpm2,
            legality=legality, outcome=outcome, rating_after=self.profile.rating,
        )
        self.events.append(event)
        return event

    def run(self, n_deliveries: int) -> List[OrchestratorEvent]:
        """Runs up to n_deliveries, stopping early (without raising) on
        the first SafetyViolation — the event log records why."""
        for _ in range(n_deliveries):
            try:
                self.run_one_delivery()
            except SafetyViolation as e:
                self.events.append(OrchestratorEvent(
                    delivery_label="(rejected before reaching the machine)",
                    wheel1_rpm=0.0, wheel2_rpm=0.0,
                    rating_after=self.profile.rating, stopped_reason=str(e),
                ))
                break
        return self.events

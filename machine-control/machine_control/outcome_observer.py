"""
The sensing boundary, mirroring controller.py on the other side of the
loop: one abstract interface for "what happened on that ball", so the
orchestrator doesn't care whether the answer came from a human, a camera,
or a sensor.

No real implementation exists yet — ball tracking is real but weak
(~12% recall on real footage), bat tracking isn't built, and the
sparse-detection physics fit is only reliable for slower/shorter shots
(see project notes). ScriptedOutcomeObserver stands in for all of that
during testing, the same role SimulatedMachineController plays for
hardware.
"""

from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "trajectory-engine"))

from cricket_trajectory.ball import BallProperties, Delivery


class OutcomeObserver(ABC):
    """The only thing the orchestrator knows about how a ball's outcome
    gets determined. A real implementation (vision, a bat/ball sensor, a
    human via some UI) implements this; the orchestrator doesn't change
    when the sensing method does."""

    @abstractmethod
    def observe(self, delivery: Delivery, ball: BallProperties) -> str:
        """Returns an outcome key valid in adaptive.OUTCOME_SCORES for the
        delivery just bowled. May block (e.g. waiting on a human, or on a
        detection pipeline finishing) — the orchestrator awaits this
        before moving on."""


class ScriptedOutcomeObserver(OutcomeObserver):
    """Replays a fixed, pre-decided sequence of outcomes — for tests and
    demos, not a real sensing method. Raises once the script runs out
    rather than looping or guessing, so a test can't silently run longer
    than it meant to."""

    def __init__(self, outcomes: List[str]):
        self._outcomes = list(outcomes)
        self._index = 0

    def observe(self, delivery: Delivery, ball: BallProperties) -> str:
        if self._index >= len(self._outcomes):
            raise IndexError(
                f"ScriptedOutcomeObserver ran out of scripted outcomes after {self._index} calls."
            )
        outcome = self._outcomes[self._index]
        self._index += 1
        return outcome

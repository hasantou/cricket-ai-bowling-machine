"""
machine_control
================

The software architecture connecting the existing decision engines
(trajectory-engine's physics-based adaptive layer) to real bowling-
machine hardware and real outcome sensing — built and tested entirely
against simulated stand-ins for both, so it's ready for real hardware
and real sensing to be substituted in without this package's control
loop changing at all.

See README.md for what's real here versus what's still simulated.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "trajectory-engine"))

from .controller import MachineController, SimulatedMachineController, CommandedDelivery
from .safety import SafeMachineController, SafetyLimits, SafetyViolation
from .outcome_observer import OutcomeObserver, ScriptedOutcomeObserver
from .session_store import save_profile, load_profile, save_scorecard, load_scorecard
from .orchestrator import MachineOrchestrator, OrchestratorEvent

__all__ = [
    "MachineController", "SimulatedMachineController", "CommandedDelivery",
    "SafeMachineController", "SafetyLimits", "SafetyViolation",
    "OutcomeObserver", "ScriptedOutcomeObserver",
    "save_profile", "load_profile", "save_scorecard", "load_scorecard",
    "MachineOrchestrator", "OrchestratorEvent",
]

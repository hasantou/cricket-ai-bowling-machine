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
from .profile_store import (
    save_named_profile, load_named_profile, list_known_batters, profile_summary, ProfileNameCollision,
)
from .orchestrator import MachineOrchestrator, OrchestratorEvent
from .release_sensor import ReleaseSensor, SimulatedReleaseSensor, SpeedCalibrator, ReleaseMeasurement
from .impact_sensor import (
    ImpactSensor, SimulatedImpactSensor, ImpactSensorOutcomeObserver, NoContactDetected,
    VelocitySensor, SimulatedVelocitySensor, VelocitySensorOutcomeObserver,
)

__all__ = [
    "MachineController", "SimulatedMachineController", "CommandedDelivery",
    "SafeMachineController", "SafetyLimits", "SafetyViolation",
    "OutcomeObserver", "ScriptedOutcomeObserver",
    "save_profile", "load_profile", "save_scorecard", "load_scorecard",
    "save_named_profile", "load_named_profile", "list_known_batters", "profile_summary", "ProfileNameCollision",
    "MachineOrchestrator", "OrchestratorEvent",
    "ReleaseSensor", "SimulatedReleaseSensor", "SpeedCalibrator", "ReleaseMeasurement",
    "ImpactSensor", "SimulatedImpactSensor", "ImpactSensorOutcomeObserver", "NoContactDetected",
    "VelocitySensor", "SimulatedVelocitySensor", "VelocitySensorOutcomeObserver",
]

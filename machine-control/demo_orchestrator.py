"""
The full autonomous loop, end to end, against simulated hardware and a
scripted outcome sequence — proving the software architecture works
before any real machine or real sensing exists to plug into it.

Run with: python3 demo_orchestrator.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "trajectory-engine"))

from cricket_trajectory import BallProperties, PlayerProfile

from machine_control.controller import SimulatedMachineController
from machine_control.safety import SafeMachineController, SafetyLimits
from machine_control.outcome_observer import ScriptedOutcomeObserver
from machine_control.orchestrator import MachineOrchestrator
from machine_control.session_store import save_profile, save_scorecard

N_DELIVERIES = 12

inner_controller = SimulatedMachineController(cycle_time_s=0.02)
safe_controller = SafeMachineController(inner_controller, SafetyLimits(max_wheel_rpm=6000.0))
profile = PlayerProfile(name="Nets Session - Player 1", rating=1000.0)
ball = BallProperties()
observer = ScriptedOutcomeObserver(
    ["defended", "controlled", "boundary", "missed", "defended", "controlled",
     "beaten", "six", "defended", "controlled", "edged", "controlled"]
)

orchestrator = MachineOrchestrator(safe_controller, observer, profile, ball, rng=random.Random(42))

print(f"Running {N_DELIVERIES} deliveries through the full loop "
      f"(simulated machine + scripted outcomes)...\n")

for i, event in enumerate(orchestrator.run(N_DELIVERIES), 1):
    if event.stopped_reason:
        print(f"  STOPPED after {i} attempt(s): {event.stopped_reason}")
        break
    if event.legality:
        print(f"  {i:>2}. {event.delivery_label:<28} wheels={event.wheel1_rpm:.0f}/{event.wheel2_rpm:.0f}rpm "
              f"-> {event.legality.upper()} (+1 extra)")
    else:
        print(f"  {i:>2}. {event.delivery_label:<28} wheels={event.wheel1_rpm:.0f}/{event.wheel2_rpm:.0f}rpm "
              f"-> {event.outcome:<10} -> rating {event.rating_after:.0f}")

print()
print(orchestrator.card.render_text())
print()
print(f"Final rating: {profile.rating:.0f} ({profile.skill_tier()})")
print(f"Machine commands actually sent: {len(inner_controller.history)}")

os.makedirs("output", exist_ok=True)
save_profile(profile, "output/demo_profile.json")
save_scorecard(orchestrator.card, "output/demo_scorecard.json")
print("\nSaved session to output/demo_profile.json and output/demo_scorecard.json "
      "- restart this script's state from those files to prove resumability, "
      "instead of losing everything when the process ends.")

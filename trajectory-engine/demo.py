"""
Quickstart: the full pipeline end to end, in one script.

    1. Define the ball, the air it's flying through, and a delivery.
    2. Run the trajectory simulation (gravity + drag + Magnus + swing + seam-drag).
    3. Convert that same delivery into wheel RPMs for a twin-wheel machine.
    4. Feed a few outcomes into the adaptive-difficulty layer and watch it pick
       progressively better-matched deliveries for the player.

Run with:  python3 demo.py
"""

from cricket_trajectory import (
    BallProperties, Environment, Delivery, run_simulation,
    PlayerProfile, next_delivery_after,
)
from cricket_trajectory.machine import WheelMachine
from cricket_trajectory.constants import kmh_to_ms, ms_to_kmh

# ---------------------------------------------------------------------------
# 1-2. One delivery, in flight
# ---------------------------------------------------------------------------
ball = BallProperties()
env = Environment()

delivery = Delivery(
    speed_mps=kmh_to_ms(135),      # a brisk fast-medium delivery
    seam_angle_deg=22,             # classic "ideal" seam angle for swing
    swing_sign=1.0,
    label="outswinger @135km/h",
)

result = run_simulation(ball, env, delivery)
print("=== 1. Trajectory ===")
print(result.summary())

# ---------------------------------------------------------------------------
# 3. The same delivery, translated into wheel RPMs
# ---------------------------------------------------------------------------
machine = WheelMachine()
rpm1, rpm2 = machine.wheel_rpms_for_delivery(
    ball_speed_mps=delivery.speed_mps,
    spin_rate_rad_s=0.0,           # this delivery has no Magnus spin, only seam swing
    ball=ball,
)
print("\n=== 2. Machine control ===")
print(f"Wheel 1: {rpm1:.0f} rpm   Wheel 2: {rpm2:.0f} rpm")

# ---------------------------------------------------------------------------
# 4. Adaptive difficulty: a handful of balls, with a player improving
# ---------------------------------------------------------------------------
print("\n=== 3. Adaptive difficulty layer ===")
profile = PlayerProfile(name="Demo Player", rating=1000.0)
next_ball = delivery  # start with the delivery above, then let the layer take over

# a hand-scripted set of outcomes to show the loop reacting to good and bad form
scripted_outcomes = ["missed", "beaten", "defended", "controlled", "boundary", "controlled"]

for i, outcome in enumerate(scripted_outcomes, start=1):
    record, next_ball = next_delivery_after(profile, ball, next_ball, outcome)
    print(f"Ball {i}: faced '{record.label}' (difficulty {record.delivery_rating:.0f}) "
          f"-> outcome '{outcome}' -> rating {record.rating_before:.0f} -> {record.rating_after:.0f}")

print(f"\nFinal rating: {profile.rating:.0f} ({profile.skill_tier()})")
print(f"Next suggested delivery: {next_ball.label}  "
      f"({ms_to_kmh(next_ball.speed_mps):.0f} km/h, seam {next_ball.seam_angle_deg:.0f} deg)")

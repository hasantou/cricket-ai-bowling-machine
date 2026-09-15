"""
The full bridge, end to end: machine bowls -> physics simulates the flight
-> legality is checked automatically from that flight -> a reported outcome
scores the ball AND updates the player's adaptive rating -> the scorecard
accumulates -> the adaptive layer picks the next delivery -> repeat.

Run with: python3 demo_scorecard.py
"""

import random

from cricket_trajectory import (
    BallProperties, Environment, Delivery, run_simulation,
    PlayerProfile, suggest_next_delivery, Scorecard,
)
from cricket_trajectory.constants import kmh_to_ms

rng = random.Random(3)
ball = BallProperties()
env = Environment()
profile = PlayerProfile(name="Nets Session - Player 1", rating=1000.0)
card = Scorecard(batter_name=profile.name)

# A scripted set of outcomes so the demo is reproducible; in a real session
# this would come from a coach tapping a button, or eventually a sensor.
scripted_outcomes = [
    "beaten", "defended", "controlled", "missed", "defended",
    "controlled", "boundary", "defended", "beaten", "six",
    "defended", "controlled",
]

# Start with a reasonable opening delivery, then let the adaptive layer take over
next_ball = Delivery(speed_mps=kmh_to_ms(128), seam_angle_deg=20, label="opening delivery")

from cricket_trajectory import classify_delivery_legality

i = 0
illegal_streak = 0
MAX_ILLEGAL_RETRIES = 5  # defensive cap: re-bowling on a wide/no-ball should be rare, never unbounded

while i < len(scripted_outcomes):
    # occasionally simulate a machine-timing wide, just to exercise that path
    forced_wide = (i == 4 and illegal_streak == 0)
    delivery_for_this_ball = next_ball
    if forced_wide:
        delivery_for_this_ball = Delivery(
            speed_mps=next_ball.speed_mps, seam_angle_deg=next_ball.seam_angle_deg,
            horizontal_launch_deg=25.0, spin_rad_s=next_ball.spin_rad_s,
            label=next_ball.label + " (drifted wide)",
        )
    result = run_simulation(ball, env, delivery_for_this_ball)
    legality = classify_delivery_legality(result)

    if legality is not None:
        illegal_streak += 1
        record = card.record_ball(profile, ball, delivery_for_this_ball, result)
        print(f"  [{legality.upper()}] {delivery_for_this_ball.label} -> +1 extra, re-bowl")
        if illegal_streak >= MAX_ILLEGAL_RETRIES:
            # extremely unlikely with realistic wide/no-ball tolerances, but
            # never loop forever on it -- fall back to a safe, central delivery
            next_ball = Delivery(speed_mps=kmh_to_ms(120), seam_angle_deg=15, label="safe re-set delivery")
            illegal_streak = 0
        else:
            next_ball = suggest_next_delivery(profile, ball, rng=rng)
        continue  # illegal ball doesn't consume one of the scripted outcomes

    illegal_streak = 0

    outcome = scripted_outcomes[i]
    record = card.record_ball(profile, ball, delivery_for_this_ball, result, outcome=outcome)
    print(f"  {card.overs_str:>5}  {delivery_for_this_ball.label:<28} "
          f"outcome={outcome:<10} -> {record.runs} run(s)"
          f"{'  OUT (' + record.dismissal + ')' if record.is_wicket else ''}  "
          f"rating {record.player_rating_before:.0f}->{record.player_rating_after:.0f}")

    next_ball = suggest_next_delivery(profile, ball, rng=rng)
    i += 1

print()
print(card.render_text())
print()
print(f"Final player rating: {profile.rating:.0f} ({profile.skill_tier()})")

"""
Shows net_outcome.py driving the scorecard automatically from a shot's
post-contact trajectory, instead of a human picking "controlled" or
"boundary" by name.

The post-contact trajectories here are simulated (same physics engine
used everywhere else in this package) as a stand-in for wherever they'll
eventually come from for real — the sparse-detection physics fit, or a
bat/ball impact sensor. net_outcome.py doesn't care which; it only
classifies the distance/height it's given. That upstream piece (actually
detecting a real shot's trajectory) isn't built yet — this demonstrates
the classification and scoring logic that's ready for it.

Run with: python3 demo_net_outcome.py
"""
from cricket_trajectory import (
    BallProperties, Environment, Delivery, PlayerProfile, Scorecard,
    run_simulation, classify_from_simulation,
)

ball = BallProperties()
env = Environment()
profile = PlayerProfile(name="Nets Session - Player 1", rating=1000.0)
card = Scorecard(batter_name=profile.name)

# Five simulated shots off the bat, each a different quality of contact -
# release position/speed/launch angle chosen to exercise every category.
SHOTS = [
    Delivery(release_pos_m=(20.0, 0.0, 0.9), speed_mps=2.0, vertical_launch_deg=-20.0, label="smothered"),
    Delivery(release_pos_m=(20.0, 0.0, 0.9), speed_mps=8.0, vertical_launch_deg=0.0, label="pushed for one"),
    Delivery(release_pos_m=(20.0, 0.0, 0.9), speed_mps=25.0, vertical_launch_deg=2.0, label="firmly struck"),
    Delivery(release_pos_m=(20.0, 0.0, 0.9), speed_mps=16.0, vertical_launch_deg=60.0, label="mistimed, high"),
    Delivery(release_pos_m=(20.0, 0.0, 0.9), speed_mps=1.0, vertical_launch_deg=-30.0, label="dead-batted"),
]

print("Bowling delivery -> simulated shot -> auto-classified outcome -> scorecard\n")

for shot in SHOTS:
    # In a real session, this bowled delivery would come from
    # adaptive.suggest_next_delivery() and the resulting shot's trajectory
    # from wherever it's actually detected. Here, both are just simulated.
    bowled = Delivery(speed_mps=30.0, seam_angle_deg=15.0, label="delivery")
    bowled_result = run_simulation(ball, env, bowled)

    shot_result = run_simulation(ball, env, shot)
    net_outcome = classify_from_simulation(shot_result)

    record = card.record_ball(profile, ball, bowled, bowled_result, outcome=net_outcome.adaptive_outcome)
    distance = abs(shot_result.landing_point_m[0] - shot_result.trajectory["x"].iloc[0])
    height = float(shot_result.trajectory["z"].max())

    print(f"  {shot.label:<16} (dist {distance:5.1f}m, height {height:4.1f}m) "
          f"-> classified '{net_outcome.label}' -> scored as '{net_outcome.adaptive_outcome}' "
          f"-> {record.runs} run(s)"
          f"{'  CHANCE (would likely be caught)' if net_outcome.label == 'mistimed_skied' else ''}")

print()
print(card.render_text())
print()
print(f"Final player rating: {profile.rating:.0f} ({profile.skill_tier()})")

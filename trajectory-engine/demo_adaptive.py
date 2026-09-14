"""
Runnable demo of the adaptive-difficulty layer (cricket_trajectory.adaptive).

Simulates a synthetic training session: a virtual batter with some hidden
"true skill" faces 70 balls from the adaptive machine, where the outcome of
each ball is drawn probabilistically from their true skill vs. the physics-
derived difficulty of the delivery bowled -- and the virtual batter's true
skill also creeps up slightly over the session, the way a real player
actually improves with practice.

This is here to (a) prove the module runs end-to-end, and (b) generate the
chart used in the companion write-up showing the rating tracking, and the
chosen delivery difficulty rising to follow it.
"""

import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cricket_trajectory import (
    BallProperties, PlayerProfile, next_delivery_after, expected_success,
    OUTCOME_SCORES,
)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "text.color": INK, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "grid.color": GRID, "axes.grid": True, "grid.linewidth": 0.8,
})

rng = random.Random(7)
ball = BallProperties()
profile = PlayerProfile(name="Demo Batter", rating=900.0, k_factor=24.0)

# a "true skill" that improves slowly over the session -- representing real
# learning happening as they face progressively better-matched deliveries
true_skill = 900.0
true_skill_growth_per_ball = 3.0

n_balls = 70
outcome_labels = list(OUTCOME_SCORES.keys())
outcome_values = np.array([OUTCOME_SCORES[k] for k in outcome_labels])

player_ratings = [profile.rating]
delivery_ratings = []
expected_scores = []

# first ball: no history yet, so just ask for something near the player's
# starting rating directly
next_ball = None
from cricket_trajectory.adaptive import suggest_next_delivery, delivery_difficulty_rating
next_ball = suggest_next_delivery(profile, ball, rng=rng)

for i in range(n_balls):
    d_rating = delivery_difficulty_rating(next_ball, ball)
    true_success_prob = expected_success(true_skill, d_rating)

    # sample an outcome score correlated with true_success_prob: pick the
    # OUTCOME_SCORES bucket whose value is closest to a noisy draw around
    # true_success_prob, so results look like discrete real-world outcomes
    noisy_draw = min(max(rng.gauss(true_success_prob, 0.22), 0.0), 1.0)
    outcome_key = outcome_labels[int(np.argmin(np.abs(outcome_values - noisy_draw)))]

    record, next_ball = next_delivery_after(profile, ball, next_ball, outcome_key, rng=rng)

    player_ratings.append(profile.rating)
    delivery_ratings.append(d_rating)
    expected_scores.append(record.expected_score)

    true_skill += true_skill_growth_per_ball  # the player is genuinely improving too

balls_axis = np.arange(1, n_balls + 1)

fig, axes = plt.subplots(2, 1, figsize=(9, 7.2), sharex=True,
                          gridspec_kw={"height_ratios": [2, 1]})

ax = axes[0]
ax.plot(balls_axis, player_ratings[1:], color=BLUE, linewidth=2.2, label="Player rating (tracked)")
ax.plot(balls_axis, delivery_ratings, color=ORANGE, linewidth=1.6, alpha=0.85,
        label="Difficulty of delivery bowled")
ax.set_ylabel("Elo-style rating")
ax.set_title("Adaptive layer tracking a player's rating over a 70-ball session",
              fontsize=13, fontweight="bold", color=INK, loc="left")
ax.legend(frameon=False, loc="upper left", fontsize=9.5)

ax2 = axes[1]
ax2.plot(balls_axis, expected_scores, color=AQUA, linewidth=1.8)
ax2.axhline(0.5, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
ax2.set_ylim(0, 1)
ax2.set_ylabel("Expected success\n(pre-ball)")
ax2.set_xlabel("Ball number")
ax2.text(n_balls * 0.985, 0.52, "50% line", ha="right", fontsize=8.5, color=MUTED)

for a in axes:
    for spine in ("top", "right"):
        a.spines[spine].set_visible(False)

fig.tight_layout()
import os
os.makedirs("output", exist_ok=True)
fig.savefig("output/adaptive_session.png", dpi=200, bbox_inches="tight")
print("saved output/adaptive_session.png")

print()
print(f"Final player rating: {profile.rating:.0f} ({profile.skill_tier()})")
print(f"Final true skill (hidden): {true_skill:.0f}")
print(f"Mean expected success across session: {np.mean(expected_scores):.2f}")
print(f"Delivery rating range offered: {min(delivery_ratings):.0f}-{max(delivery_ratings):.0f}")

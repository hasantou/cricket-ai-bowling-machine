"""
Where is the ball, and how high, when it gets to the batter?

The simulation in simulate.py stops at the first ground contact, which is
the right place to describe a delivery's length but not to judge whether it is
a wide or a no-ball: a wide is decided by where the ball PASSES the batter,
and a ball that swings or turns after pitching passes somewhere different from
where it landed. This module carries the ball through its bounce and on to the
batting crease and reports its position and velocity there.

The bounce is a deliberately simple model:
  * the vertical speed reverses and is scaled by `normal_restitution`;
  * the two horizontal speeds are scaled by `tangential_retention` (the pitch's
    grip and friction take some pace off the ball);
  * spin's effect on the bounce is ignored (it keeps acting in flight through
    the existing aerodynamics).

The defaults are grounded in published measurement, not a guess, but they are
still NOT measured for this machine, this ball or this surface — read the gap
below before trusting them anywhere near a Law 22 boundary call.
  * `normal_restitution = 0.58` is Rod Cross's measured coefficient of
    restitution for a cricket ball dropped onto a rigid surface (a ball
    dropped from 2.0 m rebounds to 0.56-0.76 m; ratio of rebound to impact
    speed ~0.58) — "The Physics of Cricket",
    https://physics.usyd.edu.au/~cross/cricket.html (a university physics
    page, not a peer-reviewed paper; read directly, not from memory).
  * `tangential_retention = 0.63` is chosen so the TOTAL speed lost pitching
    (both components combined, at the shallow angles a fast delivery pitches
    at) falls in the middle of the 30-40% range that source reports for a
    ball pitching on a real pitch — matched by calculation here, not itself a
    number the source states directly, since it does not give a separate
    tangential figure.
  * THE GAP: both numbers come from a rigid/hard-surface drop test, not a
    grass pitch's compliance, nor this project's ball, nor spin's effect on
    grip (which the source says measurably changes the bounce angle and which
    this model ignores). Once a crease-plane sensor exists
    (machine-control/crease_sensor.py), its readings against this model's
    predictions are the real calibration data — the source above narrows the
    starting guess, it does not replace that measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp

from . import constants as c
from .ball import BallProperties, Delivery, Environment
from .dynamics import state_derivative
from .simulate import run_simulation


@dataclass(frozen=True)
class BounceModel:
    normal_restitution: float = 0.58       # share of vertical speed kept, reversed (Cross's measured COR)
    tangential_retention: float = 0.63     # share of horizontal speed kept (fitted to Cross's reported total speed loss)
    max_bounces: int = 2                   # a very short ball can bounce twice before the crease


@dataclass(frozen=True)
class Crossing:
    """The ball at the batting-crease plane (x = crease_x)."""
    t_s: float                # since release
    x_m: float
    y_m: float                # lateral, + = off side for a right-hander
    z_m: float                # height above the ground
    vx: float
    vy: float
    vz: float
    bounces: int              # pitches before the crease (0 = a full toss)
    pitch_x_m: Optional[float]  # where it first landed (None for a full toss)
    pitch_y_m: Optional[float] = None   # lateral position where it first landed
    source: str = "physics model (uncalibrated bounce)"

    @property
    def bounced(self) -> bool:
        return self.bounces > 0

    @property
    def speed_mps(self) -> float:
        return float(np.sqrt(self.vx ** 2 + self.vy ** 2 + self.vz ** 2))


def predict_crossing(
    ball: BallProperties, env: Environment, delivery: Delivery,
    crease_x_m: float = c.PITCH_LENGTH_M, bounce: BounceModel = BounceModel(),
    max_time_s: float = 3.0,
) -> Optional[Crossing]:
    """Fly the delivery to its first ground contact, bounce it, and keep going until it
    crosses x = crease_x_m. Returns None if it never reaches the crease (e.g. a very
    slow, very short ball that dies first)."""
    first = run_simulation(ball, env, delivery, max_time_s=max_time_s)
    traj = first.trajectory
    x, y, z = traj["x"].to_numpy(), traj["y"].to_numpy(), traj["z"].to_numpy()
    vx, vy, vz, t = traj["vx"].to_numpy(), traj["vy"].to_numpy(), traj["vz"].to_numpy(), traj["t"].to_numpy()

    if x.max() >= crease_x_m and (z[-1] > 0.05 or x[-1] >= crease_x_m):
        # already at the crease before (or as) it reached the ground: a full toss
        def at(arr):
            return float(np.interp(crease_x_m, x, arr))
        return Crossing(at(t), crease_x_m, at(y), at(z), at(vx), at(vy), at(vz), 0, None)

    pitch_x = float(x[-1])
    pitch_y = float(y[-1])
    state = np.array([x[-1], y[-1], 1e-4, vx[-1], vy[-1], vz[-1]], dtype=float)
    clock = float(t[-1])
    bounces = 0

    def reached_crease(_t, s, *_):
        return s[0] - crease_x_m
    reached_crease.terminal, reached_crease.direction = True, 1

    def ground(_t, s, *_):
        return s[2]
    ground.terminal, ground.direction = True, -1

    while bounces < bounce.max_bounces:
        bounces += 1
        state = np.array([
            state[0], state[1], 1e-4,
            state[3] * bounce.tangential_retention, state[4] * bounce.tangential_retention,
            -state[5] * bounce.normal_restitution,
        ])
        if state[5] <= 0.3:                                 # too little bounce left to leave the ground
            return None
        sol = solve_ivp(
            state_derivative, (0.0, max_time_s), state, args=(ball, env, delivery),
            events=[reached_crease, ground], max_step=0.002, rtol=1e-7, atol=1e-9,
        )
        clock += float(sol.t[-1])
        state = sol.y[:, -1]
        if len(sol.t_events[0]):                            # crossed the crease plane
            return Crossing(clock, float(state[0]), float(state[1]), float(max(state[2], 0.0)),
                            float(state[3]), float(state[4]), float(state[5]), bounces, pitch_x, pitch_y)
        # otherwise it pitched again before the crease: loop bounces it once more
    return None

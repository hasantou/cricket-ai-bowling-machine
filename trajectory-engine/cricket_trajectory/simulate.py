"""Numerical integration of the trajectory: the actual 'march forward in time' loop."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from .ball import BallProperties, Environment, Delivery
from .dynamics import state_derivative
from . import constants as c


@dataclass
class SimulationResult:
    trajectory: pd.DataFrame   # columns: t, x, y, z, vx, vy, vz, speed
    delivery: Delivery

    @property
    def flight_time_s(self) -> float:
        return float(self.trajectory["t"].iloc[-1])

    @property
    def landing_point_m(self):
        row = self.trajectory.iloc[-1]
        return (float(row.x), float(row.y))

    @property
    def lateral_deviation_m(self) -> float:
        """How far the ball ended up from a straight line drawn from release
        through its initial aim direction -- i.e. how much it swung/turned."""
        traj = self.trajectory
        x0, y0 = traj.x.iloc[0], traj.y.iloc[0]
        theta = math.radians(self.delivery.horizontal_launch_deg)
        # straight-line aim direction in the x-y plane
        dx, dy = math.cos(theta), math.sin(theta)
        xf, yf = traj.x.iloc[-1], traj.y.iloc[-1]
        # perpendicular distance of the final point from the aim line
        rel_x, rel_y = xf - x0, yf - y0
        return rel_x * (-dy) + rel_y * dx

    @property
    def speed_loss_pct(self) -> float:
        v0 = self.trajectory["speed"].iloc[0]
        v1 = self.trajectory["speed"].iloc[-1]
        return 100.0 * (v0 - v1) / v0

    def summary(self) -> str:
        lx, ly = self.landing_point_m
        return (
            f"[{self.delivery.label}] "
            f"flight={self.flight_time_s:.3f}s  "
            f"landing=({lx:.2f}, {ly:.2f})m  "
            f"lateral_dev={self.lateral_deviation_m * 100:.1f}cm  "
            f"speed_loss={self.speed_loss_pct:.1f}%"
        )


def _ground_event(t, state, ball, env, delivery):
    return state[2]  # z
_ground_event.terminal = True
_ground_event.direction = -1  # only trigger when crossing zero going downward


def run_simulation(ball: BallProperties,
                    env: Environment,
                    delivery: Delivery,
                    max_time_s: float = 3.0,
                    max_step_s: float = 0.002) -> SimulationResult:
    """Integrate the trajectory from release until the ball reaches z=0."""
    y0 = delivery.initial_state()

    sol = solve_ivp(
        fun=state_derivative,
        t_span=(0.0, max_time_s),
        y0=y0,
        args=(ball, env, delivery),
        method="RK45",
        max_step=max_step_s,
        events=_ground_event,
        dense_output=False,
    )

    t = sol.t
    x, y, z, vx, vy, vz = sol.y
    speed = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

    df = pd.DataFrame({
        "t": t, "x": x, "y": y, "z": z,
        "vx": vx, "vy": vy, "vz": vz, "speed": speed,
    })
    return SimulationResult(trajectory=df, delivery=delivery)

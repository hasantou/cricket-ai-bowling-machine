"""
The force-summing function: given the ball's current state, compute total
acceleration. This is the single function the integrator calls every step.
"""

from __future__ import annotations

import math
from typing import Sequence

from .aerodynamics import (
    reynolds_number,
    drag_coefficient,
    magnus_lift_coefficient,
    swing_coefficient,
    seam_drag_coefficient,
)
from .ball import BallProperties, Environment, Delivery


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(v):
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def acceleration(state: Sequence[float],
                  ball: BallProperties,
                  env: Environment,
                  delivery: Delivery) -> tuple:
    """
    state = (x, y, z, vx, vy, vz). Returns (ax, ay, az) -- the sum of gravity,
    drag, Magnus (spin) and swing (seam), divided by mass.
    """
    _, _, _, vx, vy, vz = state
    wx, wy, wz = env.wind_mps

    # velocity relative to the air the ball is actually flying through
    rvx, rvy, rvz = vx - wx, vy - wy, vz - wz
    speed = _norm((rvx, rvy, rvz))

    m = ball.mass_kg
    A = ball.area_m2
    rho = env.air_density_kg_m3
    diameter = 2 * ball.radius_m

    # --- gravity --------------------------------------------------------
    ax, ay, az = 0.0, 0.0, -env.gravity_mps2

    if speed < 1e-6:
        return ax, ay, az  # ball at rest relative to air; only gravity acts

    # --- drag -------------------------------------------------------------
    Re = reynolds_number(speed, diameter, rho, env.dynamic_viscosity_pa_s)
    Cd = drag_coefficient(Re)
    drag_scalar = -0.5 * rho * Cd * A * speed / m
    ax += drag_scalar * rvx
    ay += drag_scalar * rvy
    az += drag_scalar * rvz

    # --- Magnus (spin) force -----------------------------------------------
    omega = delivery.spin_rad_s
    spin_rate = _norm(omega)
    if spin_rate > 1e-6:
        spin_parameter = ball.radius_m * spin_rate / speed
        Cl = magnus_lift_coefficient(spin_parameter)
        omega_hat = tuple(o / spin_rate for o in omega)
        v_hat = tuple(v / speed for v in (rvx, rvy, rvz))
        direction = _cross(omega_hat, v_hat)  # already unit-ish, cross of two unit vectors
        magnus_scalar = 0.5 * rho * Cl * A * speed ** 2 / m
        ax += magnus_scalar * direction[0]
        ay += magnus_scalar * direction[1]
        az += magnus_scalar * direction[2]

    # --- swing (seam) force -------------------------------------------------
    if delivery.seam_angle_deg > 1e-6:
        Cs = swing_coefficient(
            seam_angle_deg=delivery.seam_angle_deg,
            speed_mps=speed,
            reverse_swing=delivery.reverse_swing,
            reverse_threshold_mps=delivery.reverse_threshold_kmh / 3.6,
        )
        # Lateral direction: horizontal, perpendicular to the ball's initial
        # direction of travel (seam held roughly fixed relative to the seam
        # bowler set at release, not tied to the ball's spin axis).
        fwd = (rvx, rvy, 0.0)
        fwd_norm = _norm(fwd) or 1.0
        fwd_hat = (fwd[0] / fwd_norm, fwd[1] / fwd_norm, 0.0)
        lateral_hat = (-fwd_hat[1], fwd_hat[0], 0.0)  # rotate 90 deg in the horizontal plane
        swing_scalar = 0.5 * rho * Cs * A * speed ** 2 / m * delivery.swing_sign
        ax += swing_scalar * lateral_hat[0]
        ay += swing_scalar * lateral_hat[1]
        az += swing_scalar * lateral_hat[2]

        # Companion component of the SAME seam-generated force, resolved
        # along the direction of travel instead of across it: presenting the
        # seam at an angle doesn't just deflect the ball, it also costs a
        # little extra drag on top of drag_coefficient()'s Re-based value.
        # This opposes velocity exactly like the main drag term above.
        Cds = seam_drag_coefficient(seam_angle_deg=delivery.seam_angle_deg)
        seam_drag_scalar = -0.5 * rho * Cds * A * speed / m
        ax += seam_drag_scalar * rvx
        ay += seam_drag_scalar * rvy
        az += seam_drag_scalar * rvz

    return ax, ay, az


def state_derivative(t: float, state: Sequence[float],
                      ball: BallProperties, env: Environment, delivery: Delivery):
    """d/dt of (x, y, z, vx, vy, vz) -- the function solve_ivp integrates."""
    x, y, z, vx, vy, vz = state
    ax, ay, az = acceleration(state, ball, env, delivery)
    return [vx, vy, vz, ax, ay, az]

"""
Release-mechanism model for a twin-wheel bowling machine.

Everything in dynamics.py / simulate.py models the ball once it's already in
the air. This module models the other end of the problem: a bowling machine
doesn't have a wrist and fingers, it has two (sometimes three) motor-driven
wheels that compress the ball between them and fling it out. This maps
between "what the wheels are doing" and "what the ball does at release" --
i.e. the actual control variables you'd send to motor controllers.

Physical picture (twin-wheel machine):
    Two wheels of radius R_w, spinning with surface (tangential) speeds
    v1 and v2, squeeze the ball briefly as it passes between them.
      - Their AVERAGE surface speed sets the ball's exit speed (both wheels
        drive the ball forward together).
      - Their DIFFERENCE sets the ball's spin (one wheel's surface moves
        faster than the other across the contact patch, which torques the
        ball -- exactly like rolling a pen between your palms at different
        speeds).
    An efficiency/slip factor < 1 is included because real contact always
    slips a little; calibrate it against your own machine with a radar gun
    and a slow-motion spin count, then treat it as fixed for that machine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .ball import BallProperties


@dataclass
class WheelMachine:
    wheel_radius_m: float = 0.12        # typical drive-wheel radius on a compact machine
    speed_efficiency: float = 0.90      # slip factor for forward speed transfer
    spin_efficiency: float = 0.55       # slip factor for spin transfer (spin slips more than drive)

    # --- forward model: wheel speeds -> ball release state -----------------

    def wheel_rpm_to_surface_speed(self, rpm: float) -> float:
        return 2 * math.pi * self.wheel_radius_m * rpm / 60.0

    def ball_release_state(self, rpm1: float, rpm2: float, ball: BallProperties):
        """Given both wheel RPMs, return (ball_speed_mps, spin_rate_rad_s)."""
        v1 = self.wheel_rpm_to_surface_speed(rpm1)
        v2 = self.wheel_rpm_to_surface_speed(rpm2)
        ball_speed = self.speed_efficiency * (v1 + v2) / 2.0
        spin_rate = self.spin_efficiency * (v1 - v2) / (2.0 * ball.radius_m)
        return ball_speed, spin_rate

    # --- inverse model: desired ball release state -> wheel speeds --------

    def wheel_rpms_for_delivery(self, ball_speed_mps: float, spin_rate_rad_s: float,
                                 ball: BallProperties):
        """
        Solve the forward equations backwards:
            v1 + v2 = 2 * ball_speed / speed_efficiency
            v1 - v2 = 2 * radius * spin_rate / spin_efficiency
        for v1, v2, then convert each to a wheel RPM.
        """
        sum_v = 2 * ball_speed_mps / self.speed_efficiency
        diff_v = 2 * ball.radius_m * spin_rate_rad_s / self.spin_efficiency
        v1 = (sum_v + diff_v) / 2.0
        v2 = (sum_v - diff_v) / 2.0
        rpm1 = v1 * 60.0 / (2 * math.pi * self.wheel_radius_m)
        rpm2 = v2 * 60.0 / (2 * math.pi * self.wheel_radius_m)
        return rpm1, rpm2

    def wheel_pair_tilt_for_spin_axis(self, spin_type: str) -> float:
        """
        Rotation of the wheel-pair assembly about the barrel's own axis (deg),
        needed to align the spin axis with a given stock delivery. The wheel
        pair imparts spin about an axis in the plane perpendicular to the
        line joining the two wheels -- rotating that plane rotates the axis:
            0 deg   -> side-spin axis vertical (z)   : off/leg break
            90 deg  -> side-spin axis horizontal (y) : top/backspin
        Intermediate angles blend the two (useful for a "spinning seam" ball
        that both drifts and dips).
        """
        table = {
            "backspin": 90.0, "topspin": 90.0,
            "offspin": 0.0, "legspin": 0.0,
            "none": 0.0,
        }
        return table.get(spin_type, 0.0)

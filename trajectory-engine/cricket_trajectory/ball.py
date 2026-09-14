"""Data structures describing the ball, the environment, and a delivery."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Tuple

from . import constants as c

Vector3 = Tuple[float, float, float]

SpinType = Literal["none", "backspin", "topspin", "offspin", "legspin", "custom"]


@dataclass
class BallProperties:
    """Physical properties of the ball itself."""

    mass_kg: float = c.BALL_MASS_KG
    radius_m: float = c.BALL_RADIUS_M

    @property
    def area_m2(self) -> float:
        """Cross-sectional area presented to the airflow."""
        return math.pi * self.radius_m ** 2


@dataclass
class Environment:
    """Everything about the air the ball is flying through."""

    air_density_kg_m3: float = c.AIR_DENSITY_KG_M3
    dynamic_viscosity_pa_s: float = c.AIR_DYNAMIC_VISCOSITY_PA_S
    # Wind velocity vector in the same (x=down-pitch, y=lateral, z=up) frame, m/s
    wind_mps: Vector3 = (0.0, 0.0, 0.0)
    gravity_mps2: float = c.G

    @classmethod
    def at_altitude(cls, altitude_m: float, sea_level_density: float = c.AIR_DENSITY_KG_M3) -> "Environment":
        """Rough exponential atmosphere model, for grounds well above sea level."""
        scale_height = 8500.0  # m
        rho = sea_level_density * math.exp(-altitude_m / scale_height)
        return cls(air_density_kg_m3=rho)


@dataclass
class Delivery:
    """
    Initial conditions for one ball in flight (a bowled delivery, or a shot
    off the bat -- the physics engine doesn't care which).

    Coordinate frame:
        x : down the pitch, bowler's end (0) towards the batting end (+)
        y : lateral, positive = off side for a right-handed batter (sign is
            just a convention -- what matters is consistency within a run)
        z : vertical, positive = up, 0 = ground level

    Spin is given directly as an angular-velocity vector (rad/s) in this same
    frame, which is the most general representation -- helper constructors
    below build the common bowling spin axes for you.
    """

    release_pos_m: Vector3 = (0.0, 0.0, 2.2)   # typical fast bowler release height
    speed_mps: float = 30.0
    vertical_launch_deg: float = -2.0           # angle below/above horizontal (bowling is slightly downward)
    horizontal_launch_deg: float = 0.0          # aim line, lateral

    spin_rad_s: Vector3 = (0.0, 0.0, 0.0)       # angular velocity vector, rad/s

    # Seam-induced swing (separate mechanism from Magnus spin above)
    seam_angle_deg: float = 0.0                 # angle of seam to the direction of travel
    swing_sign: float = 1.0                     # +1 / -1, which way the seam is angled
    reverse_swing: bool = False                 # shiny-side-leads regime at high pace
    reverse_threshold_kmh: float = 140.0

    label: str = "delivery"

    def initial_state(self) -> Tuple[float, float, float, float, float, float]:
        """Return (x, y, z, vx, vy, vz) at release."""
        theta = math.radians(self.vertical_launch_deg)
        phi = math.radians(self.horizontal_launch_deg)
        v = self.speed_mps
        vx = v * math.cos(theta) * math.cos(phi)
        vy = v * math.cos(theta) * math.sin(phi)
        vz = v * math.sin(theta)
        x0, y0, z0 = self.release_pos_m
        return (x0, y0, z0, vx, vy, vz)

    # --- convenience constructors for common spin types -------------------

    @staticmethod
    def spin_vector(spin_type: SpinType, rpm: float, custom_axis: Vector3 = (0, 0, 1)) -> Vector3:
        """
        Build an angular-velocity vector (rad/s) for a named spin type.

        Axis conventions (right-handed, x=down-pitch, y=lateral, z=up):
          backspin -> axis along +y  (imparts lift-side Magnus force, i.e. "float")
          topspin  -> axis along -y  (dips the ball down faster)
          offspin  -> axis along +z  (drifts/turns one way, off-spinner's stock ball)
          legspin  -> axis along -z  (drifts/turns the other way)
        """
        omega = rpm * 2 * math.pi / 60.0
        axes = {
            "none": (0.0, 0.0, 0.0),
            "backspin": (0.0, omega, 0.0),
            "topspin": (0.0, -omega, 0.0),
            "offspin": (0.0, 0.0, omega),
            "legspin": (0.0, 0.0, -omega),
        }
        if spin_type == "custom":
            n = math.sqrt(sum(a * a for a in custom_axis)) or 1.0
            ux, uy, uz = (a / n for a in custom_axis)
            return (ux * omega, uy * omega, uz * omega)
        return axes[spin_type]

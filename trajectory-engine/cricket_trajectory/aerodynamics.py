"""
Empirical aerodynamic coefficient models.

None of these are exact -- real values come from wind-tunnel and field-tracking
studies and vary with surface condition, humidity and individual balls. They're
written as small, isolated functions so you can swap in a better-calibrated
curve (e.g. fitted to Hawk-Eye trajectory data) without touching the
integrator.

The seam-swing mechanism modelled in swing_coefficient() below -- a seam held
at a small angle trips the laminar boundary layer into turbulence
asymmetrically, deflecting the ball sideways -- is not a guess; it is the
mechanism established by:

    Mehta, R.D. (1985). "Aerodynamics of Sports Balls."
    Annual Review of Fluid Mechanics, 17, 151-189.

which confirmed the asymmetric separation points on the seam vs. non-seam
sides via smoke-flow visualization. The ~20-25 degree "ideal" seam angle this
module's peak_exponent=6 is tuned to reproduce (see swing_coefficient's
docstring) is the same figure that paper's flow-visualization work supports --
this module's shape is a working reconstruction of that finding, not the
original wind-tunnel data itself.
"""

from __future__ import annotations

import math


def reynolds_number(speed_mps: float, diameter_m: float, air_density: float, viscosity: float) -> float:
    return air_density * speed_mps * diameter_m / viscosity


def drag_coefficient(reynolds: float,
                      cd_subcritical: float = 0.5,
                      cd_supercritical: float = 0.22,
                      re_critical: float = 1.7e5,
                      re_width: float = 2.5e4) -> float:
    """
    Smooth logistic approximation of the cricket-ball 'drag crisis': at low
    Reynolds number the boundary layer separates early (laminar), giving high
    drag; past a critical Re the layer trips turbulent and stays attached
    longer, so drag drops sharply. Bowling speeds (Re ~ 1e5-2e5) sit right in
    this transition, which is part of why swing bowling is even possible.
    """
    if reynolds <= 0:
        return cd_subcritical
    x = (reynolds - re_critical) / re_width
    # sigmoid from 1 (low Re) to 0 (high Re)
    sigmoid = 1.0 / (1.0 + math.exp(x))
    return cd_supercritical + (cd_subcritical - cd_supercritical) * sigmoid


def magnus_lift_coefficient(spin_parameter: float, cl_max: float = 0.35) -> float:
    """
    Lift coefficient for a spinning sphere as a function of the dimensionless
    spin parameter Sp = r*omega / v. Uses the common saturating approximation
    Cl = Sp / (a + b*Sp), which rises roughly linearly for small Sp (gentle
    spin) and flattens out at high Sp (fast spin relative to ball speed) --
    consistent with measurements on spinning balls in several sports.
    """
    if spin_parameter <= 0:
        return 0.0
    a, b = 0.5, 1.0 / cl_max - 0.5 if cl_max else 1.0
    # simplifies to a saturating curve that -> cl_max as spin_parameter -> inf
    return spin_parameter / (a + spin_parameter / cl_max)


def _seam_force_peak(peak_exponent: float) -> float:
    """Peak value of sin(theta)*cos(theta)^n over theta in [0, 90deg], used to
    normalize the seam-force projections below so their maximum equals the
    requested coefficient exactly at the resulting peak angle."""
    theta_peak = math.atan(1.0 / math.sqrt(peak_exponent))
    return math.sin(theta_peak) * math.cos(theta_peak) ** peak_exponent


def swing_coefficient(seam_angle_deg: float,
                       speed_mps: float,
                       reverse_swing: bool,
                       reverse_threshold_mps: float,
                       cs_max: float = 0.30,
                       reverse_gain: float = 1.15,
                       peak_exponent: float = 6.0) -> float:
    """
    Magnitude of the seam-swing (lateral) force coefficient, derived from a
    force-resolution argument rather than fitted directly as a lateral force:
    the seam generates some boundary-layer-driven side force, and geometry
    says the component of that force perpendicular to the direction of
    travel (the swing itself) should carry a cos(theta) projection -- as the
    seam plane rotates away from being broadside to the flow, less of the
    asymmetry's force acts sideways and more of it acts along the direction
    of travel. Combined with the force's own growth from zero frontal
    exposure at theta=0 (a sin(theta) factor), the natural form is:

        C_s(theta) = A0 * sin(theta) * cos(theta)^n

    which is zero at 0 degrees (no asymmetry to project) and at 90 degrees
    (fully side-on, all of it projects into drag instead -- see
    seam_drag_coefficient() below for that companion effect). A plain
    sin(theta)*cos(theta) (n=1) would peak at 45 degrees; the exponent n
    (default 6) folds in the empirically-observed fact that the seam's own
    boundary-layer disruption saturates well before that, pulling the
    combined peak down to the classically-quoted "ideal" seam angle of ~22
    degrees. A0 is solved so the peak value equals cs_max exactly.

    If reverse_swing is enabled and the ball is travelling above the reverse
    threshold, the effective coefficient is boosted (roughness-driven
    transition dominates over the conventional seam effect at high pace) --
    this is a simplified stand-in for a genuinely complex, still-researched
    phenomenon; treat it as illustrative rather than predictive.
    """
    theta = math.radians(min(max(seam_angle_deg, 0.0), 90.0))
    a0 = cs_max / _seam_force_peak(peak_exponent)
    base = a0 * math.sin(theta) * math.cos(theta) ** peak_exponent

    if reverse_swing and speed_mps > reverse_threshold_mps:
        return base * reverse_gain
    return base


def seam_drag_coefficient(seam_angle_deg: float, cds_max: float = 0.06) -> float:
    """
    Extra drag from presenting the seam at an angle to the airflow, on top of
    drag_coefficient()'s Reynolds-based value.

    This is deliberately a SEPARATE curve from swing_coefficient(), not a
    second projection of the same force -- the two are different physical
    mechanisms in real cricket-ball aerodynamics. Orthodox swing comes from a
    boundary-layer transition asymmetry that peaks at a moderate seam angle
    (~20-25 degrees) and is genuinely small again by 90 degrees. Extra seam
    drag comes from simple frontal blockage -- how much of the raised seam's
    profile faces the oncoming air -- which keeps growing all the way to a
    seam fully side-on. That's exactly the "cross-seam" ball bowlers use on
    purpose: seam at ~90 degrees, no swing at all, but a slower, more
    turbulent, harder-gripping delivery off the pitch. Modelled simply as
    C_ds(theta) = C_ds,max * sin(theta)^2 -- zero with the seam aligned dead
    straight, maximum fully side-on.
    """
    theta = math.radians(min(max(seam_angle_deg, 0.0), 90.0))
    return cds_max * math.sin(theta) ** 2

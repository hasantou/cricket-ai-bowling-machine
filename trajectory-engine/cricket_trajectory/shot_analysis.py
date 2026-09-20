"""
"What shot was that?" — a named cricket shot inferred from what the sensors
saw the ball do off the bat, plus how the ball was bowled.

Inputs (all things this project genuinely has on a machine):
  - the ball's exit velocity vector at bat contact (speed, elevation,
    azimuth — net_outcome.ExitVelocity, from the post-contact sensor);
  - the delivery's length (delivery_report.DeliveryReport, from the
    machine's own commanded delivery + physics — no camera needed);
  - optionally, whether the batter went forward or back (from video pose).

Why this works better than reading it off video: *where the ball went* is
what defines most named shots (a cover drive goes through cover; a pull goes
to midwicket off a short ball), and the post-contact sensor measures exactly
that. A pose model cannot see bat-to-ball direction reliably from a phone
clip.

What this is NOT:
  - It is an INFERENCE from a rule table, not a measurement and not a coach's
    judgement. It says the shot was "consistent with a cover drive", which is
    what the direction, height and pace of the ball point to. A batter can
    play a shot and hit it somewhere the textbook name doesn't cover.
  - Direction bands and length groupings are named, adjustable, uncalibrated
    conventions (field-position names are approximate; real fields and
    coaches disagree on exact boundaries).
  - Right-handed batter only, the same fixed convention as ExitVelocity
    (+azimuth = off side). A left-hander would need the sides mirrored.
  - When the batter's footwork isn't supplied, front/back foot is INFERRED
    from the ball's length (full ball -> forward, short ball -> back), and
    the result says so. It is a guess about technique, not an observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .net_outcome import (
    DEAD_BAT_SPEED_MPS, SKIED_ELEVATION_DEG, WELL_STRUCK_SPEED_MPS, ExitVelocity,
)

# --- azimuth bands (degrees from straight down the pitch; +ve = off side) ---
STRAIGHT_MAX_DEG = 15.0        # within this either side: straight
MID_MAX_DEG = 35.0             # mid-off / mid-on
COVER_MAX_DEG = 65.0           # cover / midwicket
SQUARE_MAX_DEG = 100.0         # point / square leg; beyond this is behind square
AERIAL_DEG = 12.0              # off the ground enough to call the shot lofted

FORWARD_LENGTHS = ("full toss", "yorker", "full", "good length")   # played forward
BACK_LENGTHS = ("short of a length", "short (bouncer territory)")  # played back
BOUNCER = "short (bouncer territory)"

FRONT_FOOT, BACK_FOOT = "front foot", "back foot"


@dataclass(frozen=True)
class ShotAnalysis:
    shot: str                      # e.g. "cover drive"
    region: str                    # e.g. "cover" — where the ball went
    trajectory: str                # "along the ground" / "in the air" / "skied"
    exit_speed_kmh: float
    elevation_deg: float
    azimuth_deg: float
    footwork: str                  # "front foot" / "back foot" / "n/a"
    footwork_source: str           # "observed (video)" / "inferred from the ball's length" / "n/a"
    confidence: str                # "firm" | "approximate" | "n/a"
    rationale: str

    def rows(self) -> List[Tuple[str, str]]:
        return [
            ("Shot", self.shot),
            ("Where it went", self.region),
            ("Trajectory", f"{self.trajectory} ({self.elevation_deg:.0f}° launch)"),
            ("Off the bat", f"{self.exit_speed_kmh:.0f} km/h"),
            ("Footwork", f"{self.footwork} — {self.footwork_source}" if self.footwork != "n/a" else "n/a"),
            ("Confidence", self.confidence),
        ]


def region_for(azimuth_deg: float) -> str:
    """Approximate fielding-position zone for a right-handed batter."""
    a = azimuth_deg
    side_off = a > 0
    m = abs(a)
    if m <= STRAIGHT_MAX_DEG:
        return "straight"
    if m <= MID_MAX_DEG:
        return "mid-off" if side_off else "mid-on"
    if m <= COVER_MAX_DEG:
        return "cover" if side_off else "midwicket"
    if m <= SQUARE_MAX_DEG:
        return "point" if side_off else "square leg"
    return "third man / gully (behind square)" if side_off else "fine leg (behind square)"


def _trajectory(elevation_deg: float) -> str:
    if elevation_deg >= SKIED_ELEVATION_DEG:
        return "skied"
    if elevation_deg >= AERIAL_DEG:
        return "in the air"
    return "along the ground"


def _name(region: str, length: str, front: bool, lofted: bool, bouncer: bool) -> str:
    """The shot the direction + length point to, for an attacking hit."""
    pre = "lofted " if lofted else ""
    if region == "straight":
        return f"{pre}straight drive" if front else f"{pre}back-foot straight punch"
    if region == "mid-off":
        return f"{pre}off drive" if front else f"{pre}back-foot off-side punch"
    if region == "mid-on":
        return f"{pre}on drive" if front else "pull"
    if region == "cover":
        return f"{pre}cover drive" if front else f"{pre}back-foot punch through cover"
    if region == "midwicket":
        if front:
            return f"{pre}flick / whip through midwicket"
        return "hook" if bouncer else "pull"
    if region == "point":
        return f"{pre}square drive" if front else "square cut"
    if region == "square leg":
        return f"{pre}flick square of the wicket" if front else ("hook" if bouncer else "pull")
    if region.startswith("third man"):
        return "late cut" if not front else "thick edge / deflection behind square"
    return "leg glance" if front else "fine hook / glance"          # fine leg


def analyse_shot(
    exit_velocity: Optional[ExitVelocity],
    length_label: str,
    footwork: Optional[str] = None,
) -> ShotAnalysis:
    """`exit_velocity` None means the sensor saw no contact. `length_label` is
    delivery_report.DeliveryReport.length_label. `footwork` is "front foot" /
    "back foot" if it was observed (e.g. from video), else None to infer it
    from the length."""
    if exit_velocity is None:
        return ShotAnalysis(
            shot="no shot / missed", region="n/a", trajectory="n/a", exit_speed_kmh=0.0,
            elevation_deg=0.0, azimuth_deg=0.0, footwork="n/a", footwork_source="n/a",
            confidence="n/a", rationale="The sensor detected no contact with the bat.",
        )

    speed = exit_velocity.speed_mps
    elev = exit_velocity.elevation_deg
    az = exit_velocity.azimuth_deg
    region = region_for(az)
    trajectory = _trajectory(elev)

    if footwork in (FRONT_FOOT, BACK_FOOT):
        front, source = footwork == FRONT_FOOT, "observed (video)"
    else:
        front = length_label in FORWARD_LENGTHS
        source = "inferred from the ball's length"
    foot = FRONT_FOOT if front else BACK_FOOT
    bouncer = length_label == BOUNCER

    common = dict(
        region=region, trajectory=trajectory, exit_speed_kmh=speed * 3.6,
        elevation_deg=elev, azimuth_deg=az, footwork=foot, footwork_source=source,
    )

    if elev >= SKIED_ELEVATION_DEG:
        return ShotAnalysis(
            shot="mistimed / skied hit", confidence="firm",
            rationale=f"Launched {elev:.0f}° — steeper than {SKIED_ELEVATION_DEG:.0f}°, so it hung up rather than "
                      "travelled; that is a mistimed hit whatever shot was intended.",
            **common,
        )
    if speed < DEAD_BAT_SPEED_MPS:
        name = "forward defence (block)" if front else "back-foot defence (block)"
        return ShotAnalysis(
            shot=name, confidence="firm" if source.startswith("observed") else "approximate",
            rationale=f"Left the bat at only {speed * 3.6:.0f} km/h — below {DEAD_BAT_SPEED_MPS * 3.6:.0f} km/h "
                      "is a defensive block, whichever way it went.",
            **common,
        )

    lofted = trajectory == "in the air" and speed >= WELL_STRUCK_SPEED_MPS
    name = _name(region, length_label, front, lofted, bouncer)
    # An attacking name that leans on an inferred foot is a weaker claim than one
    # that doesn't depend on footwork (the straight/behind-square regions read the
    # same either way for most lengths) — say so rather than sound certain.
    confidence = "firm" if source.startswith("observed") else "approximate"
    return ShotAnalysis(
        shot=name, confidence=confidence,
        rationale=f"Ball went to {region} ({az:+.0f}°), {trajectory}, off a {length_label} ball played off the "
                  f"{foot} ({source}). That combination is what a {name} looks like.",
        **common,
    )

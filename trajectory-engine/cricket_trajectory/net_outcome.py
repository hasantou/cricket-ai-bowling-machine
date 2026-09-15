"""
Classifies a batter's shot into net-appropriate outcome categories, from
the ball's own post-contact trajectory — not match-ground concepts like
"boundary" or "six", which don't literally apply inside an enclosed
practice net (there's no rope to cross, and a net's length physically
caps how far a shot can ever travel before hitting the back netting).

Deliberately does NOT introduce a second, diverging outcome scale: every
category here maps onto an existing key in adaptive.OUTCOME_SCORES, so
the same Elo rating machinery and the same scorecard.py runs/wickets
logic apply unchanged — this module only adds a named, documented
classification step in front of them, the same "same interface in, only
the decision function changes" migration pattern neural_scorer.py and
outcome_bridge.py already use elsewhere in this project.

This module classifies SHOT QUALITY given that contact has already been
confirmed by some other signal (a bat/ball impact sensor, bat-tracking,
or a human) — it does not itself decide whether contact happened at all.
For no contact, use NET_OUTCOMES["no_contact"] directly rather than
calling classify_net_outcome().

The distance/height it classifies can come from anywhere that produces
real numbers for them: the sparse-detection physics fit tested
separately (see project notes for its real, measured limits — reliable
for slower/shorter shots, not yet for fast/long ones under outlier
contamination), a future bat/ball sensor, or a fully simulated trajectory
for testing. This module only classifies what it's given.
"""

from __future__ import annotations

from dataclasses import dataclass

from .simulate import SimulationResult

# Named, adjustable thresholds — not hidden magic numbers. Physically
# reasoned starting points, not calibrated against anything real yet (no
# post-contact footage or sensor data exists to calibrate against) — the
# same honesty as outcome_bridge.py's heuristic thresholds.
DEAD_BAT_DISTANCE_M = 2.5       # travels only this far -> smothered/defended
WELL_STRUCK_DISTANCE_M = 12.0   # travels at least this far -> a genuine attacking shot
SKIED_HEIGHT_M = 4.0            # rises above this -> mistimed, would likely be caught in a real match

# No "six" tier: a real six needs 70m+ of carry, which cannot physically
# happen inside an enclosed net of realistic length — well_struck is the
# genuine ceiling here, not an oversight.


@dataclass(frozen=True)
class NetOutcome:
    label: str             # net-appropriate descriptive name
    adaptive_outcome: str  # the matching adaptive.OUTCOME_SCORES / scorecard.py key


#: Net-appropriate categories, each tied to the existing outcome
#: vocabulary so runs/wickets/dismissals and the Elo rating all come from
#: the same place they already do — scorecard.py's DEFAULT_SCORING and
#: adaptive.py's OUTCOME_SCORES, unchanged.
NET_OUTCOMES = {
    "no_contact":           NetOutcome("no_contact", "missed"),
    "dead_bat":             NetOutcome("dead_bat", "defended"),
    "controlled_placement": NetOutcome("controlled_placement", "controlled"),
    "well_struck":          NetOutcome("well_struck", "boundary"),
    "mistimed_skied":       NetOutcome("mistimed_skied", "edged"),
}


def classify_net_outcome(post_contact_distance_m: float, post_contact_max_height_m: float) -> NetOutcome:
    """
    Classifies a confirmed-contact shot's quality from two physically
    measurable quantities about its post-contact trajectory: how far the
    ball travelled from the point of contact, and how high it rose.

    Height is checked first — a mistimed shot that balloons up is a real
    "would likely be caught" chance regardless of how far it eventually
    carries, which distance alone would miss entirely.
    """
    if post_contact_max_height_m >= SKIED_HEIGHT_M:
        return NET_OUTCOMES["mistimed_skied"]
    if post_contact_distance_m < DEAD_BAT_DISTANCE_M:
        return NET_OUTCOMES["dead_bat"]
    if post_contact_distance_m < WELL_STRUCK_DISTANCE_M:
        return NET_OUTCOMES["controlled_placement"]
    return NET_OUTCOMES["well_struck"]


def classify_from_simulation(result: SimulationResult) -> NetOutcome:
    """Convenience wrapper: classify directly from a SimulationResult — the
    shape both the sparse-detection physics fit and a fully synthetic test
    already produce via run_simulation()."""
    distance = abs(result.landing_point_m[0] - result.trajectory["x"].iloc[0])
    max_height = float(result.trajectory["z"].max())
    return classify_net_outcome(distance, max_height)

"""
The bridge between the ball and the machine.

Every other module in this package handles one end of the problem:
`simulate.py` says what one ball does in the air, `machine.py` says what
wheel RPMs produce it, `adaptive.py` says which delivery to bowl next. None
of them, on their own, produce anything you could call a scorecard — a
record of runs, wickets, and overs.

This module is the connective tissue. It takes one bowled delivery (a
`Delivery`, already run through `simulate.py`), combines it with what
actually happened when the player faced it, and turns that into:

  1. A fair/wide/no-ball ruling -- computed directly from the delivery's own
     simulated trajectory (line and height at the batting crease), so this
     part needs no human input at all.
  2. A scored outcome (runs, wicket or not, dismissal type) -- derived from
     the same outcome vocabulary the adaptive layer already uses
     (`adaptive.OUTCOME_SCORES`), so reporting one outcome after a ball
     drives BOTH the scorecard AND the player's adaptive rating together.
  3. A running `Scorecard` -- the accumulated state (total runs, wickets,
     overs, ball-by-ball history) that an actual scorecard display would
     read from.

Nothing here needs a new physics model or a new data-collection mechanism:
it's entirely built from what `simulate.py`, `machine.py` and `adaptive.py`
already produce.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np

from .ball import BallProperties, Delivery
from .simulate import SimulationResult
from .adaptive import OUTCOME_SCORES, PlayerProfile, FacedRecord, delivery_difficulty_rating
from . import constants as c

# ---------------------------------------------------------------------------
# 1. Fair / wide / no-ball, straight from the simulated trajectory
# ---------------------------------------------------------------------------

def classify_delivery_legality(result: SimulationResult,
                                crease_distance_m: float = c.PITCH_LENGTH_M,
                                no_ball_height_m: float = 1.0,
                                wide_width_m: float = 1.07) -> Optional[str]:
    """
    Rules on whether a delivery is fair, a no-ball (on height), or a wide
    (on line), using nothing but the trajectory this project already
    simulates -- no separate sensor or umpire input needed for this part.

    - No-ball on height: if the ball is still above `no_ball_height_m` when
      it reaches the batting crease (i.e. it would reach the batter on the
      full, above roughly waist height) rather than having already bounced.
    - Wide: if the ball's lateral position by the time it reaches the
      batter is beyond `wide_width_m` of the stump line. The default,
      1.07 m (~3.5 feet), matches the actual wide-guide markings used in
      limited-overs cricket -- deliberately generous, since normal swing or
      turn drifting the ball sideways during flight is not itself a wide;
      only ending up meaningfully outside the batter's reach is.

    Returns "no_ball", "wide", or None (a fair, legally bowled delivery).
    Both checks can in principle both be true; height takes precedence,
    matching the real Laws of Cricket (a no-ball overrides a wide call).

    Note: since this project's trajectory simulation stops at the ball's
    first ground contact (see the pitch-bounce companion note), the lateral
    position used here for a grounded delivery is where it bounces, as the
    best available proxy for "where it passes the batter" -- a genuinely
    precise version of this check would need the post-bounce model that
    document describes.
    """
    traj = result.trajectory
    x, y, z = traj["x"].to_numpy(), traj["y"].to_numpy(), traj["z"].to_numpy()

    landing_x, _ = result.landing_point_m
    if landing_x <= crease_distance_m:
        # the ball bounces before reaching the batter -- a normal, grounded
        # delivery. Only the line at pitching height matters for a wide.
        y_at_crease = float(np.interp(crease_distance_m, x, y)) if landing_x < crease_distance_m else float(y[-1])
    else:
        # the ball is still in the air when it crosses the crease (a full
        # toss) -- check its height there for a no-ball on the full.
        height_at_crease = float(np.interp(crease_distance_m, x, z))
        if height_at_crease > no_ball_height_m:
            return "no_ball"
        y_at_crease = float(np.interp(crease_distance_m, x, y))

    if abs(y_at_crease) > wide_width_m:
        return "wide"
    return None


# ---------------------------------------------------------------------------
# 2. Turning a reported outcome into runs, wickets, dismissals
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScoringInfo:
    runs: int
    is_wicket: bool = False
    dismissal: Optional[str] = None


#: How each outcome in adaptive.OUTCOME_SCORES translates into an actual
#: scorecard entry. Deliberately a separate table from OUTCOME_SCORES rather
#: than replacing it: OUTCOME_SCORES drives the adaptive rating (how good
#: was this for the player relative to the delivery's difficulty), this
#: table drives the scorecard (what happened in cricket terms) -- related,
#: but not the same axis, and a coach may reasonably want to change one
#: without touching the other.
DEFAULT_SCORING = {
    "missed":     ScoringInfo(runs=0, is_wicket=True, dismissal="bowled"),
    "beaten":     ScoringInfo(runs=0),
    "edged":      ScoringInfo(runs=0, is_wicket=True, dismissal="caught"),
    "defended":   ScoringInfo(runs=0),
    "controlled": ScoringInfo(runs=1),
    "boundary":   ScoringInfo(runs=4),
    "six":        ScoringInfo(runs=6),
}


def score_outcome(outcome: str, scoring_table: Optional[dict] = None) -> ScoringInfo:
    """Look up the cricket-scoring consequence of a reported outcome. Pass a
    custom scoring_table to override any entry (e.g. a coach who wants
    'edged' to sometimes mean 'dropped, no run' rather than always out)."""
    table = scoring_table or DEFAULT_SCORING
    if outcome not in table:
        raise KeyError(f"Unknown outcome '{outcome}'. Known: {list(table)}")
    return table[outcome]


# ---------------------------------------------------------------------------
# 3. One ball's full record, and the running scorecard it belongs to
# ---------------------------------------------------------------------------

@dataclass
class BallRecord:
    over: int
    ball_in_over: int          # 1-6 for a legal ball; 0 for an extra that doesn't count
    delivery_label: str
    delivery_difficulty: float
    legality: Optional[str]    # None (fair), "wide", or "no_ball"
    outcome: Optional[str]     # the reported outcome, or None if the ball was called before a shot mattered
    runs: int                  # total runs from this ball, including any extra run
    is_wicket: bool
    dismissal: Optional[str]
    player_rating_before: Optional[float] = None
    player_rating_after: Optional[float] = None


@dataclass
class Scorecard:
    """Running state for one player's session/innings against the machine."""
    batter_name: str
    total_runs: int = 0
    wickets: int = 0
    legal_balls: int = 0
    extras: int = 0
    balls: List[BallRecord] = field(default_factory=list)

    @property
    def overs_str(self) -> str:
        return f"{self.legal_balls // 6}.{self.legal_balls % 6}"

    @property
    def strike_rate(self) -> float:
        return 100.0 * self.total_runs / self.legal_balls if self.legal_balls else 0.0

    def record_ball(self,
                     profile: PlayerProfile,
                     ball: BallProperties,
                     delivery: Delivery,
                     result: SimulationResult,
                     outcome: Optional[str] = None,
                     scoring_table: Optional[dict] = None,
                     legality: object = "auto") -> BallRecord:
        """
        The single call a control loop makes per ball: pass in what was
        bowled, its simulated flight, and (for a fair ball) the reported
        outcome. Updates the scorecard totals AND the player's adaptive
        rating together, and returns the full record of what happened.

        `outcome` is ignored for a wide or no-ball at the umpiring stage --
        real cricket does allow a batter to still score off an illegal
        delivery, but that's deliberately out of scope for a v1 bridge; a
        wide/no-ball here is always scored as exactly one extra run.

        `legality` defaults to "auto": the single-trajectory proxy in
        classify_delivery_legality(). Pass the call from laws.assess_delivery()
        (None for fair, "wide") - or a crease-plane sensor's decision - to score
        on the bounce-aware, Laws-based judgement instead.
        """
        if legality == "auto":
            legality = classify_delivery_legality(result)
        over = self.legal_balls // 6
        rating_before = rating_after = None

        if legality is not None:
            # wide or no-ball: one extra run, does not consume a legal ball
            self.total_runs += 1
            self.extras += 1
            record = BallRecord(
                over=over, ball_in_over=0, delivery_label=delivery.label,
                delivery_difficulty=delivery_difficulty_rating(delivery, ball),
                legality=legality, outcome=None, runs=1,
                is_wicket=False, dismissal=None,
            )
        else:
            if outcome is None:
                raise ValueError("A fair delivery needs a reported outcome.")
            scoring = score_outcome(outcome, scoring_table)
            self.legal_balls += 1
            self.total_runs += scoring.runs
            if scoring.is_wicket:
                self.wickets += 1

            faced: FacedRecord = profile.record_outcome(delivery, ball, outcome)
            rating_before, rating_after = faced.rating_before, faced.rating_after

            record = BallRecord(
                over=over, ball_in_over=((self.legal_balls - 1) % 6) + 1,
                delivery_label=delivery.label, delivery_difficulty=faced.delivery_rating,
                legality=None, outcome=outcome, runs=scoring.runs,
                is_wicket=scoring.is_wicket, dismissal=scoring.dismissal,
                player_rating_before=rating_before, player_rating_after=rating_after,
            )

        self.balls.append(record)
        return record

    def render_text(self) -> str:
        """A simple ball-by-ball scorecard, readable straight in a terminal
        or dropped into a report."""
        lines = [
            f"{self.batter_name}: {self.total_runs}/{self.wickets}  "
            f"({self.overs_str} overs, SR {self.strike_rate:.1f}, extras {self.extras})",
            "-" * 60,
        ]
        for r in self.balls:
            if r.legality is not None:
                lines.append(f"  {r.legality.upper():>7} -- {r.delivery_label}  (+1 extra)")
                continue
            tag = f"OUT ({r.dismissal})" if r.is_wicket else f"{r.runs} run(s)"
            lines.append(
                f"  {r.over}.{r.ball_in_over}  {r.delivery_label:<32} "
                f"outcome={r.outcome:<10} {tag}"
            )
        return "\n".join(lines)

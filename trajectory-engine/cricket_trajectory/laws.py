"""
The Laws of Cricket, as far as a bowling machine and a net can use them —
sourced, with the checks that can be computed, and an honest map of the rest.

SOURCES (read directly, not from memory):
  [MCC]  Laws of Cricket, 2017 Code (3rd Edition — 2022), Marylebone Cricket Club.
         Read from a published PDF copy of the full text (79 pages):
         https://northwestcricket.com/wp-content/uploads/2023/04/laws-of-cricket-2017-code-3rd-edition-2022_1.pdf
  [ICC]  ICC Men's T20 World Cup 2024 Playing Conditions (a match-specific variation on the Laws):
         https://images.icc-cricket.com/image/upload/prd/my7lr35p9gfrvsafsow6.pdf
  The wide GUIDELINE offset (about 0.89 m from the middle stump) is NOT in either
  document's extractable text (it is drawn in ICC Appendix C); it is derived from
  the return crease (1.32 m, MCC Law 7.4) less the 17 in reported by Wisden's
  explainer, and should be treated as secondary.

What the Laws actually say, that shaped this module:
  * Law 22.1 — a WIDE is a ball that "passes wide of where the striker is standing"
    unless it is "sufficiently within reach for him/her to be able to hit it with the
    bat by means of a normal cricket stroke". There is no fixed distance in the Law.
    It is judged as the ball passes the striker's wicket (22.2). If the striker moves
    so as to make it wide, it is not a wide (22.4.1).
  * ICC 22.1.1.2 — additionally a wide if the ball "passes above the head height of the
    striker standing upright at the popping crease". A no-ball overrides a wide (21.13).
  * Law 21.7 — a NO BALL if the ball bounces more than once or rolls before the popping
    crease, or "pitches wholly or partially off the pitch" (3.05 m wide, Law 6.1) before
    the striker's wicket.
  * Law 41.7.1 — a full toss that passes "above waist height of the striker standing
    upright at the popping crease" is a no-ball. (The Law gives no number for waist.)
  * Law 32.1 — BOWLED: the wicket is broken by a ball delivered by the bowler.
  * Law 36 — LBW needs the ball to strike the person, which a machine cannot see.
  * Law 19.7 — a Boundary 6 needs the ball struck and first grounded beyond the boundary
    without touching the ground inside; a Boundary 4 is grounded beyond having first
    grounded inside.

The tolerances that stand in for umpiring judgement (reach, waist and head height,
leg-side limit) are named, adjustable and NOT calibrated: the Laws leave them to a
human. A machine must pick numbers; these are stated so they can be argued with.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from . import constants as c
from .ball import BallProperties, Delivery, Environment
from .crease_crossing import BounceModel, Crossing, predict_crossing

# --- dimensions, straight from the Laws (metres) -------------------------------------------
PITCH_LENGTH_M = 20.12                     # MCC 6.1: 22 yards between the bowling creases
PITCH_HALF_WIDTH_M = 1.525                 # MCC 6.1: 10 ft wide, 5 ft either side of the stump line
POPPING_CREASE_FROM_STUMPS_M = 1.22        # MCC 7.3: 4 ft in front of the bowling crease
RETURN_CREASE_HALF_WIDTH_M = 1.32          # MCC 7.4: 4 ft 4 in either side of the middle stump
STUMP_SET_WIDTH_M = 0.2286                 # MCC 8.1: each set 9 in wide
STUMP_HEIGHT_M = 0.7112                    # MCC 8.2: tops 28 in above the surface
BALL_CIRCUMFERENCE_M = (0.224, 0.229)      # MCC 4.1: 8.81-9 in
BALL_RADIUS_M = sum(BALL_CIRCUMFERENCE_M) / 2 / (2 * 3.141592653589793)
POPPING_CREASE_X_M = PITCH_LENGTH_M - POPPING_CREASE_FROM_STUMPS_M    # the batting-end popping crease
WICKET_X_M = PITCH_LENGTH_M                                            # the striker's stumps


@dataclass(frozen=True)
class WideRules:
    """Proxies for the umpire's 'within reach of a normal stroke' (MCC 22.1.2). All uncalibrated."""
    off_limit_m: float = 0.89          # off side: the limited-overs wide guideline (secondary source, see above)
    leg_limit_m: float = 0.50          # leg side: no marking exists; a judgement of what a normal stroke reaches
    head_height_m: float = 1.75        # ICC 22.1.1.2: above this at the popping crease is a wide
    waist_height_m: float = 1.00       # MCC 41.7.1 gives no number; about the height of a batter's waist
    striker_y_m: float = 0.0           # where the striker stands laterally (0 = middle-stump guard); + = off side
    tolerance_m: float = 0.03          # inside this of a line, the call is borderline


@dataclass(frozen=True)
class StumpsResult:
    hits: bool
    lateral_margin_m: float            # >0: inside the stumps' width by this much; <0: missed by this much
    height_margin_m: float             # >0: below the top of the stumps by this much


@dataclass(frozen=True)
class DeliveryLegality:
    call: Optional[str]                # "wide" or None (a fair ball). No-balls are informational, see no_ball_flags.
    wide_reasons: List[str]
    wide_margin_m: float               # lateral margin to the nearest wide limit: >0 legal side, <0 wide
    borderline: bool
    no_ball_flags: List[str]           # informational (a machine has no front foot); counted, not scored
    hits_stumps: bool
    stumps: StumpsResult
    at_wicket: Optional[Crossing]
    at_popping_crease: Optional[Crossing]
    notes: List[str] = field(default_factory=list)

    @property
    def is_wide(self) -> bool:
        return self.call == "wide"

    def summary(self) -> str:
        head = "WIDE" if self.is_wide else "fair delivery"
        if self.borderline:
            head += " (borderline)"
        return head


def judge_wide(at_wicket: Optional[Crossing], at_popping: Optional[Crossing], rules: WideRules = WideRules(),
               hand: str = "right-handed") -> Tuple[bool, List[str], float]:
    """Law 22 as a machine can apply it: lateral position as the ball passes the striker's wicket, and
    (ICC) height at the popping crease. + y is the off side for a right-hander; a left-hander mirrors it."""
    if at_wicket is None:
        return False, ["The ball never reached the batter, so no wide can be judged."], 0.0
    y = at_wicket.y_m - rules.striker_y_m
    if hand == "left-handed":
        y = -y
    reasons: List[str] = []
    if y >= 0:
        margin, side, limit = rules.off_limit_m - y, "off", rules.off_limit_m
    else:
        margin, side, limit = rules.leg_limit_m - (-y), "leg", rules.leg_limit_m
    if margin < 0:
        reasons.append(f"passes the wicket {abs(y):.2f} m on the {side} side, beyond the {limit:.2f} m limit (Law 22.1)")
    if at_popping is not None and at_popping.z_m > rules.head_height_m:
        reasons.append(f"passes the popping crease at {at_popping.z_m:.2f} m, above head height {rules.head_height_m:.2f} m (ICC 22.1.1.2)")
        margin = min(margin, rules.head_height_m - at_popping.z_m)
    return bool(reasons), reasons, margin


def no_ball_flags(pitch_x_m: Optional[float], pitch_y_m: Optional[float], bounces: int,
                  at_popping: Optional[Crossing], rules: WideRules = WideRules()) -> List[str]:
    """Informational no-ball conditions a machine can still trigger (not the front-foot ones)."""
    flags: List[str] = []
    if pitch_y_m is not None and abs(pitch_y_m) + BALL_RADIUS_M > PITCH_HALF_WIDTH_M:
        flags.append(f"pitched off the pitch ({abs(pitch_y_m):.2f} m from the stump line; the pitch is 3.05 m wide) — Law 21.7")
    if bounces >= 2:
        flags.append("bounced more than once before the popping crease — Law 21.7")
    if at_popping is not None and not at_popping.bounced and at_popping.z_m > rules.waist_height_m:
        flags.append(f"full toss at {at_popping.z_m:.2f} m at the popping crease, above waist height {rules.waist_height_m:.2f} m — Law 41.7.1")
    return flags


def stumps_check(at_wicket: Optional[Crossing], ball_radius_m: float = BALL_RADIUS_M) -> StumpsResult:
    """Would the ball break the wicket? (MCC 32.1 — the machine's reading of 'would have been bowled'.)
    Treats the stumps as a rectangle and the ball as a disc; ignores the bails and the ball's spin."""
    if at_wicket is None:
        return StumpsResult(False, -1.0, -1.0)
    lateral = STUMP_SET_WIDTH_M / 2 + ball_radius_m - abs(at_wicket.y_m)
    height = STUMP_HEIGHT_M + ball_radius_m - at_wicket.z_m
    return StumpsResult(lateral >= 0 and height >= 0, lateral, height)


def assess_delivery(ball: BallProperties, env: Environment, delivery: Delivery, rules: WideRules = WideRules(),
                    bounce: BounceModel = BounceModel(), hand: str = "right-handed") -> DeliveryLegality:
    """Fly the delivery through its bounce to the batter and apply the checks above."""
    at_pop = predict_crossing(ball, env, delivery, crease_x_m=POPPING_CREASE_X_M, bounce=bounce)
    at_wkt = predict_crossing(ball, env, delivery, crease_x_m=WICKET_X_M, bounce=bounce)
    wide, reasons, margin = judge_wide(at_wkt, at_pop, rules, hand)
    ref = at_wkt or at_pop
    flags = no_ball_flags(ref.pitch_x_m if ref else None, ref.pitch_y_m if ref else None,
                          at_pop.bounces if at_pop else 0, at_pop, rules)
    stumps = stumps_check(at_wkt)
    notes = ["Bounce model is uncalibrated; the tolerances stand in for umpiring judgement (see laws.py)."]
    if at_wkt is None:
        notes.append("The ball does not reach the striker's wicket in this model (a very short or slow delivery).")
    return DeliveryLegality(
        call="wide" if wide else None, wide_reasons=reasons, wide_margin_m=margin,
        borderline=abs(margin) < rules.tolerance_m, no_ball_flags=flags, hits_stumps=stumps.hits, stumps=stumps,
        at_wicket=at_wkt, at_popping_crease=at_pop, notes=notes,
    )


def resolve_no_contact(hits_stumps: bool) -> Tuple[str, str]:
    """(outcome key, explanation) for a ball the bat did not touch. MCC 32.1: the striker is BOWLED if the
    wicket is broken by the delivery, so a miss only costs the wicket if the ball was on target; a ball that
    missed the bat AND the stumps is merely beaten. (LBW needs pad contact, which cannot be seen here.)"""
    if hits_stumps:
        return "missed", "no contact, and the ball was on target, so it would have hit the stumps — bowled (Law 32)"
    return "beaten", "no contact, and the ball would have missed the stumps — beaten, not bowled (Law 32)"


# --- the whole rulebook, as a coverage map ---------------------------------------------------------
APPLIES, INFO, NEEDS_SENSOR, NA = "implemented", "informational", "needs a sensor / person", "not applicable to a machine"


@dataclass(frozen=True)
class LawEntry:
    number: int
    title: str
    status: str
    note: str


LAWS: List[LawEntry] = [
    LawEntry(1, "The players", NA, "Team composition; a machine session has one batter."),
    LawEntry(2, "The umpires", NA, "The software plays the umpire for wide (Law 22) and would-be-bowled (Law 32)."),
    LawEntry(3, "The scorers", APPLIES, "scorecard.py keeps the running score."),
    LawEntry(4, "The ball", INFO, "Weight/size limits (5.5-5.75 oz, 22.4-22.9 cm) fix the ball model's size; the machine's ball must comply."),
    LawEntry(5, "The bat", NA, "Bat dimensions are the batter's equipment."),
    LawEntry(6, "The pitch", APPLIES, "22 yd x 10 ft; used for the pitched-off-the-pitch flag (Law 21.7)."),
    LawEntry(7, "The creases", APPLIES, "Popping crease 4 ft, return crease 4 ft 4 in: positions of the judging planes."),
    LawEntry(8, "The wickets", APPLIES, "9 in wide, 28 in high: the would-be-bowled check (Law 32)."),
    LawEntry(9, "Preparation and maintenance of the playing area", NA, "Groundskeeping."),
    LawEntry(10, "Covering the pitch", NA, "Groundskeeping."),
    LawEntry(11, "Intervals", NA, "Match timing."),
    LawEntry(12, "Start of play; cessation of play", NA, "Match timing."),
    LawEntry(13, "Innings", NA, "Match structure."),
    LawEntry(14, "The follow-on", NA, "Match structure."),
    LawEntry(15, "Declaration and forfeiture", NA, "Match structure."),
    LawEntry(16, "The result", NA, "A net session has no result."),
    LawEntry(17, "The over", APPLIES, "scorecard.py counts six valid balls to an over; a wide does not count (Law 22.8)."),
    LawEntry(18, "Scoring runs", APPLIES, "Runs from shots; running between the wickets is not measured (see Law 19)."),
    LawEntry(19, "Boundaries", NEEDS_SENSOR, "4 needs the ball to reach the boundary along the ground, 6 needs it to clear without grounding; the impact sensor gives speed/angle, not distance, so a six is judged by a rule of thumb or entered by a person."),
    LawEntry(20, "Dead ball", NA, "Umpire's call; each machine delivery is its own ball."),
    LawEntry(21, "No ball", INFO, "Front-foot and fielding no-balls do not apply to a machine. Pitched-off-the-pitch, bounced-twice and waist-high full tosses are counted, not scored."),
    LawEntry(22, "Wide ball", APPLIES, "judge_wide(): lateral position as the ball passes the wicket, plus ICC head height. The reach test is a proxy."),
    LawEntry(23, "Bye and leg bye", NEEDS_SENSOR, "Needs the ball's contact with the batter's body or the keeper's area."),
    LawEntry(24, "Fielder's absence; substitute fielders", NA, "No fielders in a net."),
    LawEntry(25, "Batter's innings; runners", NA, "One batter."),
    LawEntry(26, "Practice on the field", NA, "Conduct."),
    LawEntry(27, "The wicket-keeper", NA, "No keeper in a net."),
    LawEntry(28, "The fielder", NA, "No fielders in a net."),
    LawEntry(29, "The wicket is down", APPLIES, "Approximated by stumps_check() (ball would break the wicket); the bails are not modelled."),
    LawEntry(30, "Batter leaving the wicket", NA, "Umpire's call."),
    LawEntry(31, "Appeals", NA, "The software decides directly instead of on appeal."),
    LawEntry(32, "Bowled", APPLIES, "A miss counts as bowled only if the ball would have hit the stumps (stumps_check); otherwise it is 'beaten'. Bat/pad deflection onto the stumps needs a stump sensor."),
    LawEntry(33, "Caught", NEEDS_SENSOR, "A net has no fielders; a steep-launch hit is scored as 'would probably be caught' by a rule of thumb."),
    LawEntry(34, "Hit the ball twice", NEEDS_SENSOR, "Needs a second-contact sensor."),
    LawEntry(35, "Hit wicket", NEEDS_SENSOR, "Needs a stump/bail sensor."),
    LawEntry(36, "Leg before wicket", NEEDS_SENSOR, "Needs pad-contact detection; the would-have-hit-the-stumps half is available (stumps_check)."),
    LawEntry(37, "Obstructing the field", NA, "Conduct."),
    LawEntry(38, "Run out", NA, "No running measured."),
    LawEntry(39, "Stumped", NA, "No keeper."),
    LawEntry(40, "Timed out", NA, "Match timing."),
    LawEntry(41, "Unfair play", INFO, "41.7.1 (waist-high full toss) is counted as an informational flag; the rest is conduct."),
    LawEntry(42, "Players' conduct", NA, "Conduct."),
]


def coverage() -> dict:
    out: dict = {}
    for law in LAWS:
        out.setdefault(law.status, []).append(law.number)
    return out

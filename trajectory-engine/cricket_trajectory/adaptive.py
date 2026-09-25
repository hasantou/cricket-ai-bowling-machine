"""
Adaptive difficulty layer.

Everything else in this package answers "given these release parameters,
what does the ball do in the air, and what wheel RPMs produce them?". This
module answers a different question: "given how this specific player has
been doing, what should the machine bowl NEXT?"

The design borrows a well-tested idea from game and coaching design: a
learner improves fastest against a challenge that sits just above their
current level -- hard enough to stretch them, not so hard that success
becomes a lottery (sports-science calls this the "challenge point"; game
design calls it the "flow channel"; Vygotsky called it the "zone of
proximal development"). Practically, that means the machine should keep the
player's *expected* success rate hovering a little under 50%, not deliver
its hardest ball every time or repeat deliveries they've already mastered.

To do that we need two numbers on the SAME scale: how good the player
currently is, and how hard a given delivery actually is. We get both from
one well-understood tool -- the Elo rating system chess uses to rate a
player against an opponent -- except here the "opponent" on one side of the
match is a cricket delivery, and its rating is not hand-picked. It is
computed directly from the same aerodynamic coefficient curves
(drag_coefficient, swing_coefficient, seam_drag_coefficient,
magnus_lift_coefficient) that drive the trajectory simulator elsewhere in
this package -- so a delivery that is genuinely nastier in the physics
automatically rates as harder here too.

Feedback loop as implemented: after each ball, someone (coach, player, or
eventually a sensor/vision system -- this module doesn't care which) reports
an outcome. The rating update is the classic Elo rule. The next-delivery
picker then searches a pool of physically varied candidate deliveries for
whichever one's difficulty rating sits closest to the player's personal
target zone, and biases its pick towards that pool -- so the machine
"upgrades with the player's expertise" automatically, ball by ball, without
anyone manually raising a difficulty slider.

One rating is not enough to know a player, though: a single overall number
cannot tell "good against pace, weak against spin" apart from "moderate
against everything" -- two very different batters an overall-only rating
would show as roughly the same number. `PlayerProfile.skill_ratings` fixes
that by running the SAME Elo machinery four more times, once per
`SKILL_DIMENSIONS`, each only moved by however much that specific delivery
actually tested that dimension (see `_dimension_relevance()`) -- a pure
yorker barely touches the spin rating; a big-turning leg-break barely
touches pace. The overall `rating` is kept exactly as before (nothing that
already reads it needs to change); the per-dimension ratings are additional
detail on top, not a replacement.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

from .ball import BallProperties, Delivery
from .aerodynamics import (
    drag_coefficient,
    magnus_lift_coefficient,
    reynolds_number,
    seam_drag_coefficient,
    swing_coefficient,
)
from . import constants as c

# ---------------------------------------------------------------------------
# 1. Turning a delivery's own physics into a single difficulty rating
# ---------------------------------------------------------------------------

#: How to score a reported outcome, on a 0 (complete failure) - 1 (complete
#: success) scale. Numbers are a starting point for a coach to tune, not
#: gospel -- change them to match how strict you want "success" to mean.
OUTCOME_SCORES = {
    "missed": 0.00,       # played and missed entirely / bowled
    "beaten": 0.05,       # beaten by swing/seam/pace, no bat contact of note
    "edged": 0.30,        # thick outside edge, risky contact
    "defended": 0.55,     # solid defensive shot, no attacking intent
    "controlled": 0.75,   # attacking shot, kept along the ground under control
    "boundary": 0.95,     # four
    "six": 1.00,          # six
}

#: Rating bands purely for human-readable labelling (feeds skill_tier()).
#: Listed high-to-low; a rating is assigned the first band it's >= to.
SKILL_TIERS: Tuple[Tuple[str, float], ...] = (
    ("Advanced", 1400.0),
    ("Intermediate", 1150.0),
    ("Improver", 950.0),
    ("Beginner", 0.0),
)


#: The four skill dimensions tracked separately in PlayerProfile.skill_ratings,
#: matching the four physical ingredients _dimension_factors() computes.
SKILL_DIMENSIONS: Tuple[str, ...] = ("pace", "swing", "seam", "spin")

#: How much each dimension counts toward the single overall "nastiness" score
#: in delivery_difficulty_rating() -- pace matters most; spin and swing next;
#: seam-drag least, since a slow, gripping cross-seam ball is unsettling but
#: not violent. Named once here so the overall rating and the per-dimension
#: relevance split (_dimension_relevance()) can't silently drift apart.
_DIMENSION_WEIGHTS = {"pace": 0.45, "swing": 0.20, "seam": 0.10, "spin": 0.25}
_NASTINESS_SCALE = 1.6  # sum of _DIMENSION_WEIGHTS' contributions at every factor capped to 1.0


def _dimension_factors(delivery: Delivery, ball: BallProperties,
                        air_density: float, viscosity: float) -> dict:
    """The four physical ingredients behind delivery_difficulty_rating(),
    computed once and shared by both the overall rating and the
    per-dimension ratings below, so there is exactly one place that turns
    physics into a 0-1ish "how much is this pushing the ball around" number
    per dimension -- see delivery_difficulty_rating()'s docstring for what
    each one means physically.
    """
    v = max(delivery.speed_mps, 1e-6)
    diameter = 2.0 * ball.radius_m

    re = reynolds_number(v, diameter, air_density, viscosity)
    cd = drag_coefficient(re)  # noqa: F841 -- not used in the score directly,
    # kept here because it's what makes Re worth computing at all: overall
    # drag mostly just shortens the ball's flight/pace, which is already
    # captured by the pace factor, so it does not get its own weighted term.

    cs = swing_coefficient(
        delivery.seam_angle_deg, v, delivery.reverse_swing,
        c.kmh_to_ms(delivery.reverse_threshold_kmh),
    )
    cds = seam_drag_coefficient(delivery.seam_angle_deg)

    spin_mag = math.sqrt(sum(w * w for w in delivery.spin_rad_s))
    spin_param = ball.radius_m * spin_mag / v
    cl = magnus_lift_coefficient(spin_param)

    dynamic_term = 0.5 * air_density * ball.area_m2 * v * v  # rho*A*v^2/2, shared by all force terms
    weight_n = ball.mass_kg * c.G

    return {
        "pace": min(v / c.kmh_to_ms(160.0), 1.3),
        "swing": min((cs * dynamic_term) / weight_n, 1.5),
        "seam": min((cds * dynamic_term) / weight_n, 1.5),
        "spin": min((cl * dynamic_term) / weight_n, 1.5),
    }


def delivery_difficulty_rating(delivery: Delivery,
                                ball: BallProperties,
                                air_density: float = c.AIR_DENSITY_KG_M3,
                                viscosity: float = c.AIR_DYNAMIC_VISCOSITY_PA_S,
                                base_rating: float = 1200.0,
                                rating_spread: float = 600.0) -> float:
    """
    Elo-like rating for ONE delivery, derived from its actual physics rather
    than fitted or hand-picked.

    Four physical ingredients are computed with the same coefficient
    functions the trajectory simulator uses, each normalised against a
    generous real-world ceiling, then force-weighted (force / ball weight,
    so every term is a comparable dimensionless "how hard is this pushing
    the ball around" number):

        pace factor       raw speed -- less reaction time for the batter
        swing factor      peak seam-swing force  / ball weight
        seam-drag factor  extra seam-on drag/skid / ball weight
        spin factor       Magnus force from spin  / ball weight

    Weighted sum (see _DIMENSION_WEIGHTS) gives one 0-1ish "nastiness"
    score, linearly mapped onto an Elo-like scale centred on base_rating
    with the given spread.
    """
    factors = _dimension_factors(delivery, ball, air_density, viscosity)
    nastiness = sum(_DIMENSION_WEIGHTS[d] * factors[d] for d in SKILL_DIMENSIONS)
    nastiness = max(0.0, min(nastiness / _NASTINESS_SCALE, 1.0))  # squash to [0, 1]

    return base_rating + (nastiness - 0.5) * 2.0 * rating_spread


def dimension_delivery_rating(dimension: str, factors: dict,
                               base_rating: float = 1200.0,
                               rating_spread: float = 600.0) -> float:
    """The same Elo-like mapping delivery_difficulty_rating() uses for the
    blended nastiness score, applied to just ONE dimension's own factor --
    "how hard was THIS delivery specifically as a test of spin", on the
    same scale as the overall rating so expected_success() can compare a
    player's per-dimension rating against it directly."""
    clipped = max(0.0, min(factors[dimension], 1.0))
    return base_rating + (clipped - 0.5) * 2.0 * rating_spread


def _dimension_relevance(factors: dict) -> dict:
    """What SHARE of a delivery's overall difficulty came from each
    dimension -- a pure 155km/h yorker with no seam or spin should barely
    move a player's spin rating at all. Computed from the same weighted
    contributions delivery_difficulty_rating() sums into one nastiness
    score, just kept separate here instead of being added together.

    An honest asymmetry, checked directly rather than assumed: a genuine
    leg-break does NOT get anywhere near as much relevance-share on spin as
    a fast, seamless ball gets on pace, even at spin near its physical
    ceiling. That's not a weighting choice here -- magnus_lift_coefficient()
    saturates at cl_max=0.35, and a cricket ball's actual mass means even
    that ceiling produces a spin_factor far below pace_factor at a real
    delivery's speed. In other words: this in-flight-aerodynamics model
    structurally cannot represent spin's real difficulty for a batter, which
    is mostly about turn and dip off the pitch (a bounce phenomenon, not an
    in-flight force) -- a known, shared limitation of the underlying
    difficulty rating this module was already built on, not a new gap this
    function introduces. A spin-heavy ball still reads as MORE spin-relevant
    than a spinless one of the same pace, which is the one thing this
    function can honestly promise until a bounce-aware spin difficulty
    exists.

    Falls back to equal relevance (0.25 each) for the one edge case where
    every factor is genuinely zero (a delivery with no pace, seam, swing or
    spin at all isn't physically real, but nothing here should divide by
    zero if it somehow occurs) -- an explicit, named fallback, not a silent
    NaN.
    """
    weighted = {d: _DIMENSION_WEIGHTS[d] * factors[d] for d in SKILL_DIMENSIONS}
    total = sum(weighted.values())
    if total <= 1e-9:
        return {d: 1.0 / len(SKILL_DIMENSIONS) for d in SKILL_DIMENSIONS}
    return {d: weighted[d] / total for d in SKILL_DIMENSIONS}


# ---------------------------------------------------------------------------
# 2. The Elo match itself: player rating vs. delivery rating
# ---------------------------------------------------------------------------

def expected_success(player_rating: float, delivery_rating: float) -> float:
    """Standard Elo expected-score curve: the probability the player comes
    out on top of this particular delivery, given both ratings."""
    return 1.0 / (1.0 + 10.0 ** ((delivery_rating - player_rating) / 400.0))


@dataclass
class FacedRecord:
    """One row of a player's history: what they faced, what happened, and
    how their rating moved as a result."""
    label: str
    delivery_rating: float
    outcome_score: float
    expected_score: float
    rating_before: float
    rating_after: float
    #: What share of this delivery's difficulty came from each dimension
    #: (sums to 1.0) -- e.g. {"pace": 0.85, "swing": 0.05, "seam": 0.03,
    #: "spin": 0.07} for a near-pure pace ball. This is the "why" data a
    #: coach or the next-ball picker can read back later; see
    #: PlayerProfile.record_outcome() for how it drives the per-dimension
    #: rating updates below.
    dimension_relevance: dict = field(default_factory=dict)
    #: Each dimension's rating immediately after this ball (same keys as
    #: dimension_relevance), so a player's per-skill trend over time can be
    #: read straight off their history without recomputing anything.
    skill_ratings_after: dict = field(default_factory=dict)


@dataclass
class PlayerProfile:
    """
    A single batter's adaptive-training state. Everything the machine needs
    to know to keep choosing appropriately challenging deliveries lives
    here -- persist this (e.g. to JSON) between sessions to carry a player's
    progress forward.
    """
    name: str
    rating: float = 1000.0
    k_factor: float = 24.0
    history: List[FacedRecord] = field(default_factory=list)
    #: One Elo-like rating per SKILL_DIMENSIONS entry -- see this module's
    #: docstring for why one overall rating isn't enough to know a player.
    #: Starts at the same 1000.0 base as the overall rating; a fresh player
    #: is assumed equally (un)tested in every dimension until proven
    #: otherwise.
    skill_ratings: dict = field(default_factory=lambda: {d: 1000.0 for d in SKILL_DIMENSIONS})

    def skill_tier(self) -> str:
        for tier_name, lo in SKILL_TIERS:
            if self.rating >= lo:
                return tier_name
        return SKILL_TIERS[-1][0]

    def weakest_skill(self) -> str:
        """Which SKILL_DIMENSIONS entry this player is currently rated
        lowest on -- the dimension a coach (or the next-ball picker) should
        target first. Ties broken by SKILL_DIMENSIONS' own order, so the
        result is deterministic for a fresh, all-equal profile rather than
        depending on dict iteration order."""
        return min(SKILL_DIMENSIONS, key=lambda d: self.skill_ratings[d])

    def record_outcome(self, delivery: Delivery, ball: BallProperties,
                        outcome: Union[str, float]) -> FacedRecord:
        """
        Feed back what happened on one ball. `outcome` is either one of the
        OUTCOME_SCORES keys (manual coach/player input -- the natural
        starting point, since it needs no extra hardware) or a raw 0-1
        float, so this same call is ready to be driven by an automated
        vision/sensor system later without changing its signature.

        Updates the overall rating exactly as before, AND each of
        skill_ratings' four dimensions -- but only in proportion to how much
        THIS delivery actually tested that dimension
        (_dimension_relevance()), using the same Elo rule scaled by that
        relevance share. A pure yorker with negligible spin_factor moves the
        spin rating by only a sliver of what a full k_factor update would;
        a big leg-break moves it by nearly the full amount. This is the same
        Elo machinery run five times (once overall, once per dimension),
        not a new algorithm.
        """
        outcome_score = (OUTCOME_SCORES[outcome] if isinstance(outcome, str)
                          else float(outcome))
        d_rating = delivery_difficulty_rating(delivery, ball)
        expected = expected_success(self.rating, d_rating)

        rating_before = self.rating
        self.rating = self.rating + self.k_factor * (outcome_score - expected)

        factors = _dimension_factors(delivery, ball, c.AIR_DENSITY_KG_M3, c.AIR_DYNAMIC_VISCOSITY_PA_S)
        relevance = _dimension_relevance(factors)
        for dim in SKILL_DIMENSIONS:
            dim_rating = dimension_delivery_rating(dim, factors)
            dim_expected = expected_success(self.skill_ratings[dim], dim_rating)
            self.skill_ratings[dim] += self.k_factor * relevance[dim] * (outcome_score - dim_expected)

        record = FacedRecord(delivery.label, d_rating, outcome_score,
                              expected, rating_before, self.rating,
                              dimension_relevance=relevance,
                              skill_ratings_after=dict(self.skill_ratings))
        self.history.append(record)
        return record


# ---------------------------------------------------------------------------
# 3. Choosing what to bowl next
# ---------------------------------------------------------------------------

_SPIN_TYPES: Tuple[str, ...] = ("none", "backspin", "topspin", "offspin", "legspin")


def _random_candidate(rng: random.Random, speed_range_kmh: Tuple[float, float]) -> Delivery:
    speed_kmh = rng.uniform(*speed_range_kmh)
    seam_angle = rng.uniform(0.0, 90.0)
    spin_type = rng.choice(_SPIN_TYPES)
    rpm = 0.0 if spin_type == "none" else rng.uniform(200.0, 1200.0)
    return Delivery(
        speed_mps=c.kmh_to_ms(speed_kmh),
        seam_angle_deg=seam_angle,
        spin_rad_s=Delivery.spin_vector(spin_type, rpm),
        reverse_swing=speed_kmh > 130.0,
        label=f"{spin_type} {speed_kmh:.0f}km/h seam{seam_angle:.0f}deg",
    )


def suggest_next_delivery(profile: PlayerProfile,
                           ball: BallProperties,
                           challenge_margin: float = 60.0,
                           pool_size: int = 40,
                           shortlist_size: int = 5,
                           speed_range_kmh: Tuple[float, float] = (70.0, 150.0),
                           rng: Optional[random.Random] = None) -> Delivery:
    """
    Generate a pool of physically varied candidate deliveries within the
    machine's realistic speed/seam/spin envelope, rate each one with
    delivery_difficulty_rating(), and pick from whichever candidates land
    closest to the player's personal target zone -- their current rating
    plus a small positive challenge_margin.

    The pick is a weighted random choice among the closest few candidates
    (not always the single nearest one), on purpose: a real training session
    should still vary pace/line/type ball to ball rather than converging on
    one repeated "optimal" delivery, and always facing the exact same ball
    would make this a much less useful training tool.
    """
    rng = rng or random.Random()
    target_rating = profile.rating + challenge_margin

    scored = []
    for _ in range(pool_size):
        candidate = _random_candidate(rng, speed_range_kmh)
        rating = delivery_difficulty_rating(candidate, ball)
        scored.append((abs(rating - target_rating), rating, candidate))

    scored.sort(key=lambda row: row[0])
    shortlist = scored[:shortlist_size]
    weights = [1.0 / (1.0 + gap) for gap, _, _ in shortlist]
    _, _, chosen = rng.choices(shortlist, weights=weights, k=1)[0]
    return chosen


# ---------------------------------------------------------------------------
# 4. Convenience: one call that does record -> re-rate -> suggest
# ---------------------------------------------------------------------------

def next_delivery_after(profile: PlayerProfile,
                         ball: BallProperties,
                         faced_delivery: Delivery,
                         outcome: Union[str, float],
                         **suggest_kwargs) -> Tuple[FacedRecord, Delivery]:
    """Log the outcome of the ball just faced, then immediately suggest the
    next one -- the single call a machine's control loop would make once
    per delivery."""
    record = profile.record_outcome(faced_delivery, ball, outcome)
    next_ball = suggest_next_delivery(profile, ball, **suggest_kwargs)
    return record, next_ball

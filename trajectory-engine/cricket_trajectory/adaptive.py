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

    Seam-drag is, if anything, worse off than spin: checked directly, even
    at seam-drag's own physical peak angle (90 degrees, zero swing, zero
    spin) with a merely-moderate pace, seam's relevance share still comes
    out under 3% -- `seam_drag_coefficient()`'s own cds_max=0.06 ceiling is
    simply smaller than pace's baseline contribution at almost any bowling
    speed. `_isolating_candidate()` (below) still measurably concentrates
    what little seam/spin signal exists over generating everything at
    random -- roughly double for seam, in a direct comparison -- but it
    cannot make either the DOMINANT reason a delivery is rated hard. That
    ceiling is physical, and the honest fix is the same one spin needs: a
    real difficulty contribution from turf/bounce behaviour, not more
    tuning of `_DIMENSION_WEIGHTS`.

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

#: The machine's realistic delivery envelope, named rather than left as inline
#: defaults -- this is what machine-control/tests/test_safety_envelope.py
#: checks SafetyLimits' defaults actually bound, so the two can't silently
#: drift apart if either one changes later.
DEFAULT_SPEED_RANGE_KMH: Tuple[float, float] = (70.0, 150.0)
SPIN_RPM_RANGE: Tuple[float, float] = (200.0, 1200.0)   # ball spin, RPM -- see Delivery.spin_vector()

#: A moderate, "nothing special" pace band used for whichever dimensions a
#: candidate is NOT deliberately isolating -- e.g. varying seam angle to
#: test swing shouldn't also accidentally max out the pace factor. Central
#: enough that pace_factor stays modest without being unrealistically slow.
NEUTRAL_SPEED_RANGE_KMH = (90.0, 110.0)
#: Seam angle bands picked from swing_coefficient()/seam_drag_coefficient()'s
#: OWN documented shapes, not guessed: swing peaks around 20-25 degrees and
#: is genuinely near zero by 90 degrees, while seam-drag keeps RISING all the
#: way to 90 degrees (that is literally what a real cross-seam delivery is).
#: Picking a seam angle from the right band is how this model can isolate
#: swing from seam at all -- they share one input parameter, so they can't be
#: varied independently, but they peak at different, well-separated angles.
SWING_PEAK_SEAM_ANGLE_DEG = (15.0, 30.0)      # conventional swing bowling
SEAM_DRAG_PEAK_ANGLE_DEG = (75.0, 90.0)       # cross-seam
NEUTRAL_SEAM_ANGLE_DEG = (0.0, 5.0)           # minimal swing AND seam-drag, for isolating pace or spin


def _random_candidate(rng: random.Random, speed_range_kmh: Tuple[float, float]) -> Delivery:
    """Broadly varied across every parameter at once -- used when no
    dimension is a clear-enough weakness to target yet (see
    suggest_next_delivery_with_reason()), so a fresh or evenly-matched
    player still sees the same varied training this function has always
    produced."""
    speed_kmh = rng.uniform(*speed_range_kmh)
    seam_angle = rng.uniform(0.0, 90.0)
    spin_type = rng.choice(_SPIN_TYPES)
    rpm = 0.0 if spin_type == "none" else rng.uniform(*SPIN_RPM_RANGE)
    return Delivery(
        speed_mps=c.kmh_to_ms(speed_kmh),
        seam_angle_deg=seam_angle,
        spin_rad_s=Delivery.spin_vector(spin_type, rpm),
        reverse_swing=speed_kmh > 130.0,
        label=f"{spin_type} {speed_kmh:.0f}km/h seam{seam_angle:.0f}deg",
    )


def _isolating_candidate(rng: random.Random, speed_range_kmh: Tuple[float, float], targeted: str) -> Delivery:
    """Vary ONE dimension across its full realistic range while holding the
    others near a neutral setting that doesn't stress them much -- the "test
    one weakness at a time" a coach would deliberately choose, instead of
    _random_candidate()'s "vary everything at once" always leaving it
    ambiguous which attribute actually caused the difficulty.

    Honest limit, not hidden: swing and seam-drag share ONE input parameter
    (seam_angle_deg) in this model, so they cannot be made fully independent
    of each other the way pace and spin can be -- see the module-level seam
    angle constants above for how this picks a band that leans hard toward
    one or the other rather than claiming a false four-way independence.
    """
    seam_angle_by_target = {
        "swing": SWING_PEAK_SEAM_ANGLE_DEG,
        "seam": SEAM_DRAG_PEAK_ANGLE_DEG,
    }
    speed_kmh = rng.uniform(*speed_range_kmh) if targeted == "pace" else rng.uniform(*NEUTRAL_SPEED_RANGE_KMH)
    seam_angle = rng.uniform(*seam_angle_by_target.get(targeted, NEUTRAL_SEAM_ANGLE_DEG))
    if targeted == "spin":
        spin_type = rng.choice([t for t in _SPIN_TYPES if t != "none"])
        rpm = rng.uniform(*SPIN_RPM_RANGE)
    else:
        spin_type, rpm = "none", 0.0
    return Delivery(
        speed_mps=c.kmh_to_ms(speed_kmh),
        seam_angle_deg=seam_angle,
        spin_rad_s=Delivery.spin_vector(spin_type, rpm),
        reverse_swing=speed_kmh > 130.0,
        label=f"[isolating {targeted}] {spin_type} {speed_kmh:.0f}km/h seam{seam_angle:.0f}deg",
    )


#: How far apart the four skill_ratings need to be before one is treated as
#: an actual, worth-targeting weakness rather than Elo noise around a fresh
#: or evenly-tested player. Below this spread, suggest_next_delivery_with_reason()
#: says so honestly instead of inventing a "weak spot" out of a few points'
#: difference that a single lucky/unlucky ball could produce.
SKILL_TIE_SPREAD = 15.0

#: How much extra weight a candidate gets in the shortlist for stressing the
#: player's actual weakest dimension, ON TOP OF already being close to their
#: overall target difficulty (the shortlist is still built from overall
#: difficulty first -- this never picks a wildly-off-difficulty ball just
#: because it happens to be spin-heavy). Named and adjustable, not hidden.
WEAK_SKILL_BIAS = 1.5


@dataclass(frozen=True)
class DeliverySuggestion:
    """A chosen delivery PLUS why it was chosen -- the "why" adaptive.py's
    own docstring flagged as missing: suggest_next_delivery() (below) always
    existed, but never recorded which weakness, if any, it was aiming at."""
    delivery: Delivery
    targeted_skill: Optional[str]   # None when no dimension is a clear-enough weakness yet (see SKILL_TIE_SPREAD)
    reason: str


def _shortlist_weights(shortlist: List[Tuple[float, float, Delivery]], ball: BallProperties,
                        targeted: Optional[str]) -> List[float]:
    """The random-choice weight for each (gap, rating, candidate) row in an
    already-built shortlist. Split out from suggest_next_delivery_with_reason()
    so the weighting rule itself -- closer overall difficulty always weighs
    more (1/(1+gap)), PLUS extra weight for leaning on `targeted` when one is
    given -- can be tested directly against known candidates, instead of only
    inferred statistically from thousands of random picks."""
    if targeted is None:
        return [1.0 / (1.0 + gap) for gap, _, _ in shortlist]
    weights = []
    for gap, _, candidate in shortlist:
        factors = _dimension_factors(candidate, ball, c.AIR_DENSITY_KG_M3, c.AIR_DYNAMIC_VISCOSITY_PA_S)
        relevance = _dimension_relevance(factors)[targeted]
        weights.append((1.0 / (1.0 + gap)) * (1.0 + WEAK_SKILL_BIAS * relevance))
    return weights


def suggest_next_delivery_with_reason(profile: PlayerProfile,
                                       ball: BallProperties,
                                       challenge_margin: float = 60.0,
                                       pool_size: int = 40,
                                       shortlist_size: int = 5,
                                       speed_range_kmh: Tuple[float, float] = DEFAULT_SPEED_RANGE_KMH,
                                       rng: Optional[random.Random] = None) -> DeliverySuggestion:
    """
    Generate a pool of physically varied candidate deliveries within the
    machine's realistic speed/seam/spin envelope, rate each one with
    delivery_difficulty_rating(), and shortlist whichever candidates land
    closest to the player's personal target zone -- their current OVERALL
    rating plus a small positive challenge_margin. Overall difficulty stays
    the primary filter on purpose: a delivery wildly harder or easier than
    the player's current level shouldn't win just because it also happens to
    stress their weakest skill.

    WITHIN that shortlist, candidates that lean more heavily on the player's
    actual weakest `skill_ratings` dimension (profile.weakest_skill()) get
    extra weight in the random pick (WEAK_SKILL_BIAS) -- so the machine
    doesn't just keep the overall challenge level right, it deliberately
    probes whichever specific skill the player has actually shown is
    weakest, the way a coach choosing the next ball would. If the four
    skill_ratings are too close together to call a real weakness
    (SKILL_TIE_SPREAD), no dimension is targeted and the pick is exactly the
    plain overall-difficulty weighting suggest_next_delivery() has always
    used -- said so plainly in the returned reason, not silently guessed.

    The pick is still a weighted random choice among the closest few
    candidates (not always the single nearest one), on purpose: a real
    training session should still vary pace/line/type ball to ball rather
    than converging on one repeated "optimal" delivery.
    """
    rng = rng or random.Random()
    target_rating = profile.rating + challenge_margin

    spread = max(profile.skill_ratings.values()) - min(profile.skill_ratings.values())
    targeted = profile.weakest_skill() if spread >= SKILL_TIE_SPREAD else None

    scored = []
    for _ in range(pool_size):
        # A clear weakness: generate candidates that ISOLATE it (vary that one dimension,
        # hold the others neutral) instead of varying everything at once -- see
        # _isolating_candidate()'s docstring for why this is the real "single vs. multiple
        # variable" decision, not just re-weighting an already broadly-random pool.
        candidate = _isolating_candidate(rng, speed_range_kmh, targeted) if targeted else _random_candidate(rng, speed_range_kmh)
        rating = delivery_difficulty_rating(candidate, ball)
        scored.append((abs(rating - target_rating), rating, candidate))

    scored.sort(key=lambda row: row[0])
    shortlist = scored[:shortlist_size]
    weights = _shortlist_weights(shortlist, ball, targeted)

    _, chosen_rating, chosen = rng.choices(shortlist, weights=weights, k=1)[0]

    if targeted is None:
        reason = (
            f"No dimension is clearly weaker than the others yet (skill ratings within "
            f"{spread:.0f} points) -- picked for overall difficulty ({chosen_rating:.0f} vs a "
            f"target of {target_rating:.0f}) only."
        )
    else:
        chosen_factors = _dimension_factors(chosen, ball, c.AIR_DENSITY_KG_M3, c.AIR_DYNAMIC_VISCOSITY_PA_S)
        chosen_relevance = _dimension_relevance(chosen_factors)[targeted]
        reason = (
            f"Targeting {targeted} ({profile.skill_ratings[targeted]:.0f}, your lowest-rated "
            f"skill) -- this delivery's own difficulty is {chosen_relevance * 100:.0f}% driven by "
            f"{targeted}, while still sitting close to your overall target ({chosen_rating:.0f} vs "
            f"{target_rating:.0f})."
        )

    return DeliverySuggestion(chosen, targeted, reason)


def suggest_next_delivery(profile: PlayerProfile,
                           ball: BallProperties,
                           challenge_margin: float = 60.0,
                           pool_size: int = 40,
                           shortlist_size: int = 5,
                           speed_range_kmh: Tuple[float, float] = DEFAULT_SPEED_RANGE_KMH,
                           rng: Optional[random.Random] = None) -> Delivery:
    """Unchanged public behaviour: just the Delivery, no reason attached.
    See suggest_next_delivery_with_reason() for the same pick plus WHY it
    was made -- this is now a thin wrapper around that, not a separate
    implementation, so the two can never quietly drift apart."""
    return suggest_next_delivery_with_reason(
        profile, ball, challenge_margin, pool_size, shortlist_size, speed_range_kmh, rng,
    ).delivery


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
    per delivery. Unchanged behaviour; see next_delivery_after_with_reason()
    for the same call plus WHY the next ball was chosen."""
    record = profile.record_outcome(faced_delivery, ball, outcome)
    next_ball = suggest_next_delivery(profile, ball, **suggest_kwargs)
    return record, next_ball


def next_delivery_after_with_reason(profile: PlayerProfile,
                                     ball: BallProperties,
                                     faced_delivery: Delivery,
                                     outcome: Union[str, float],
                                     **suggest_kwargs) -> Tuple[FacedRecord, DeliverySuggestion]:
    """Same as next_delivery_after(), but returns the full DeliverySuggestion
    (delivery + targeted_skill + a human-readable reason) instead of a bare
    Delivery -- what a UI wanting to show the player/coach WHY this ball was
    picked should call."""
    record = profile.record_outcome(faced_delivery, ball, outcome)
    suggestion = suggest_next_delivery_with_reason(profile, ball, **suggest_kwargs)
    return record, suggestion

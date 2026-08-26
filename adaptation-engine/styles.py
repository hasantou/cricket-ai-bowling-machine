"""
Bowling style taxonomy for the MVP adaptation engine.

Kept deliberately small and explicit for Phase 1 (the human-in-the-loop
MVP): a coach picks from this fixed list rather than the system inventing
arbitrary combinations. See docs/product/MVP_RD_Plan_Software_First.docx,
Section 2.1, for why this stays rule-based and human-operated for now.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BowlingStyle:
    """One deliverable bowling style, as something a coach can dial into an
    existing programmable machine (e.g. a BOLA Professional)."""

    key: str            # short unique id, e.g. "fast-good-off-outswing"
    label: str           # human-readable name for the operator screen
    pace_band: str        # "slow" | "medium" | "fast"
    line: str             # "off" | "middle" | "leg"
    length: str           # "short" | "good" | "full" | "yorker"
    movement: str          # "none" | "outswing" | "inswing" | "off-spin" | "leg-spin" | "seam"

    def attributes(self):
        return (self.pace_band, self.line, self.length, self.movement)


# A starter library covering a realistic spread of what a BOLA-class
# machine can already produce manually. Extend freely — this is meant to
# be edited as the coaching advisor (see program plan, Section 9) reviews
# it, not treated as fixed.
STYLE_LIBRARY = [
    BowlingStyle("fast-good-off-none", "Fast, good length, off stump, seam-up",
                 "fast", "off", "good", "seam"),
    BowlingStyle("fast-good-off-outswing", "Fast, good length, outswing",
                 "fast", "off", "good", "outswing"),
    BowlingStyle("fast-full-leg-inswing", "Fast, full, inswing into the pads",
                 "fast", "leg", "full", "inswing"),
    BowlingStyle("fast-short-off-seam", "Fast, back of a length, off stump, seam",
                 "fast", "off", "short", "seam"),
    BowlingStyle("fast-yorker-off-none", "Fast, yorker length, off stump",
                 "fast", "off", "yorker", "none"),
    BowlingStyle("medium-good-off-outswing", "Medium pace, good length, outswing",
                 "medium", "off", "good", "outswing"),
    BowlingStyle("slow-good-off-offspin", "Slow, good length, off-spin",
                 "slow", "off", "good", "off-spin"),
    BowlingStyle("slow-good-leg-legspin", "Slow, good length, leg-spin",
                 "slow", "leg", "good", "leg-spin"),
    BowlingStyle("slow-full-middle-legspin", "Slow, full, leg-spin, middle stump",
                 "slow", "middle", "full", "leg-spin"),
    BowlingStyle("medium-short-leg-seam", "Medium pace, short, leg stump, seam",
                 "medium", "leg", "short", "seam"),
]

STYLE_BY_KEY = {s.key: s for s in STYLE_LIBRARY}


def contrast_distance(a: BowlingStyle, b: BowlingStyle) -> float:
    """How different two styles are, 0.0 (identical) to 1.0 (max contrast).

    Simple, explainable Hamming-style distance across the four attributes,
    weighted so pace and movement count for more than line/length — a
    batter who has grooved a *pace and movement* pattern is more disrupted
    by a different pace/movement than by a small line/length shift alone.
    This weighting is a first pass, meant to be tuned against real pilot
    data (program plan, Section 11 — success metrics).
    """
    weights = {"pace_band": 0.35, "movement": 0.35, "line": 0.15, "length": 0.15}
    total = 0.0
    total += weights["pace_band"] if a.pace_band != b.pace_band else 0.0
    total += weights["movement"] if a.movement != b.movement else 0.0
    total += weights["line"] if a.line != b.line else 0.0
    total += weights["length"] if a.length != b.length else 0.0
    return round(total, 3)

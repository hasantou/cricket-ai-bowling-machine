"""
Style recommendation — once a style is mastered, pick the next one.

Implements the "maximum contrast" rule from the program plan (Section
4.3): recommend the style least like the one just mastered, not a random
alternative. Ranks by contrast_distance() from styles.py and returns the
top candidates so a human operator (Phase 1) or a future learned policy
(Phase 4+) can pick among genuinely different options rather than being
handed a single forced choice.
"""

from typing import List, Optional

from styles import STYLE_LIBRARY, STYLE_BY_KEY, contrast_distance, BowlingStyle


def recommend_next_styles(mastered_style_key: str, top_n: int = 3,
                           exclude: Optional[List[str]] = None) -> List[BowlingStyle]:
    """Return up to `top_n` styles ranked by contrast against the mastered one.

    `exclude` lets the caller rule out styles faced too recently even if
    they're a high-contrast match (e.g. to avoid ping-ponging between just
    two styles) — pass the last one or two style keys used.
    """
    mastered = STYLE_BY_KEY.get(mastered_style_key)
    if mastered is None:
        raise ValueError(f"Unknown style key '{mastered_style_key}'")

    exclude = set(exclude or [])
    exclude.add(mastered_style_key)

    candidates = [s for s in STYLE_LIBRARY if s.key not in exclude]
    ranked = sorted(candidates, key=lambda s: contrast_distance(mastered, s), reverse=True)
    return ranked[:top_n]


def explain_recommendation(mastered_style_key: str, recommended: BowlingStyle) -> str:
    """A short, human-readable reason string for the operator screen —
    endorsement/demo audiences respond better to an explainable rule than
    a black-box pick, and it matches the rule-based-first design in the
    MVP plan (Section 2.2, "reinforcement-learning policy" explicitly
    deferred)."""
    mastered = STYLE_BY_KEY[mastered_style_key]
    diffs = []
    if mastered.pace_band != recommended.pace_band:
        diffs.append(f"pace ({mastered.pace_band} → {recommended.pace_band})")
    if recommended.movement != mastered.movement:
        diffs.append(f"movement ({mastered.movement} → {recommended.movement})")
    if recommended.line != mastered.line:
        diffs.append(f"line ({mastered.line} → {recommended.line})")
    if recommended.length != mastered.length:
        diffs.append(f"length ({mastered.length} → {recommended.length})")
    changed = ", ".join(diffs) if diffs else "no attributes"
    return f"Switching because {mastered.label} looks mastered — this changes {changed}."

"""
Named, multi-batter profile storage -- item 3 of the critical review: "build a long-term
profile for each batter rather than treating every session independently."

session_store.py already does the real work of turning a PlayerProfile into JSON and back
(save_profile/load_profile) -- reused here unchanged, not duplicated. What was actually
missing: a returning batter's SECOND session finding their own profile automatically, by
name, instead of a person manually tracking which JSON file belongs to whom. That's the one
gap this module closes.

Deliberately the simplest thing that solves that: one JSON file per batter in a named
directory, looked up by a slugified version of their name. No database, no server, no
concurrency handling. That is NOT the design for a multi-machine or cloud-synced deployment
(see trajectory-engine/README.md's "Where to calibrate first" and the project's own scaling
notes) -- it is what actually removes today's real, manual "which file do I load" step, and
the honest thing to reach for first per this project's own stated preference for the
simplest technology that solves the CURRENT problem. Swapping the two functions below for
a real database later is a contained change: nothing that calls them needs to know the
difference, since they already take a PlayerProfile in and hand one back.
"""

from __future__ import annotations

import json
import os
import re
import statistics
from pathlib import Path
from typing import List, Optional, Union

from cricket_trajectory import PlayerProfile

from .session_store import load_profile, save_profile

PathLike = Union[str, Path]


class ProfileNameCollision(Exception):
    """Two different batters' names slugify to the same filename (e.g. "Sam" and "SAM",
    or "O'Brien" and "OBrien"). Refusing to save is the safe default -- silently
    overwriting one batter's real history with another's would be a much worse failure
    than a clear error asking for a more distinct name."""


def _slug(name: str) -> str:
    """A batter's display name, turned into a safe filename: lowercase, alphanumerics
    only, runs of anything else collapsed to a single underscore. Named and tested on
    its own because it's the one thing silently getting this wrong would corrupt --
    two names that look different to a person but slug identically must be caught
    (see ProfileNameCollision), not merged by accident."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return slug or "unnamed"


def _path_for(name: str, base_dir: PathLike) -> Path:
    return Path(base_dir) / f"{_slug(name)}.json"


def save_named_profile(profile: PlayerProfile, base_dir: PathLike) -> Path:
    """Save `profile` under `base_dir`, keyed by its own name -- the call a session's
    end makes so the SAME batter's next session can find it again by name alone.
    Creates `base_dir` if it doesn't exist yet (the first batter ever saved shouldn't
    need a person to mkdir first)."""
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    path = _path_for(profile.name, base_dir)
    if path.exists():
        existing_name = json.loads(path.read_text()).get("name")
        if existing_name is not None and existing_name != profile.name:
            raise ProfileNameCollision(
                f"'{profile.name}' and the already-saved '{existing_name}' both map to the "
                f"filename '{path.name}' -- refusing to overwrite a different batter's profile. "
                "Use a more distinct name for one of them."
            )
    save_profile(profile, path)
    return path


def load_named_profile(name: str, base_dir: PathLike) -> Optional[PlayerProfile]:
    """The profile previously saved under this name, or None if this batter has never
    been saved before -- a normal, expected first-session case, not an error, so a
    caller can write `profile = load_named_profile(name, dir) or PlayerProfile(name=name)`
    without a try/except."""
    path = _path_for(name, base_dir)
    if not path.exists():
        return None
    return load_profile(path)


def list_known_batters(base_dir: PathLike) -> List[str]:
    """Every batter with a saved profile under `base_dir`, by their real stored name
    (not the filename slug, which is lossy) -- what a "choose a returning batter"
    dropdown in the app would list. Returns an empty list, not an error, if the
    directory doesn't exist yet (nobody has been saved there at all)."""
    directory = Path(base_dir)
    if not directory.exists():
        return []
    names = []
    for path in sorted(directory.glob("*.json")):
        try:
            names.append(json.loads(path.read_text())["name"])
        except (json.JSONDecodeError, KeyError, TypeError):
            # KeyError: valid JSON object, no "name" field. TypeError: valid JSON that
            # isn't even an object (e.g. a bare list) -- found by a test that wrote
            # exactly that. Either way: not a profile this module wrote, so skip it
            # rather than crash a whole listing over one unrelated file.
            continue
    return names


def profile_summary(profile: PlayerProfile, recent_n: int = 10) -> dict:
    """A short, human-readable snapshot of a batter's LONG-TERM state -- deliveries
    faced across every session this profile has been loaded/saved through (its
    `history` persists via session_store.py's JSON round-trip), current overall and
    per-skill ratings, and a simple trend: the average rating across their most
    recent `recent_n` deliveries versus their very first `recent_n`, so "are they
    actually improving" has a real, if rough, number behind it instead of just the
    single current rating.

    `rating_trend` is None until at least `2 * recent_n` deliveries exist (comparing
    two overlapping or near-empty windows would be noise, not a trend) -- stated
    honestly as "not enough history yet" rather than a shaky number.
    """
    history = profile.history
    summary = {
        "name": profile.name,
        "deliveries_faced": len(history),
        "overall_rating": profile.rating,
        "skill_ratings": dict(profile.skill_ratings),
        "skill_tier": profile.skill_tier(),
        "weakest_skill": profile.weakest_skill(),
        "rating_trend": None,
    }
    if len(history) >= 2 * recent_n:
        earliest = statistics.mean(r.rating_after for r in history[:recent_n])
        latest = statistics.mean(r.rating_after for r in history[-recent_n:])
        summary["rating_trend"] = latest - earliest
    return summary

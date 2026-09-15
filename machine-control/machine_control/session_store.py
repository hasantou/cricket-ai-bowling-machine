"""
JSON persistence for trajectory-engine's session state — the piece behind
"is it resumable?": no. Nothing previously saved a PlayerProfile or
Scorecard anywhere; closing the browser tab or restarting the server lost
everything. Both are plain dataclasses (deliberately, per
trajectory-engine/README.md), so this is direct field-by-field JSON, not
a database or new storage system. Nothing here decides WHEN to save or
load — callers (the app, the orchestrator) do that explicitly.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from dataclasses import asdict
from typing import Union

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "trajectory-engine"))

from cricket_trajectory import PlayerProfile, FacedRecord, Scorecard, BallRecord

PathLike = Union[str, Path]


def save_profile(profile: PlayerProfile, path: PathLike) -> None:
    with open(path, "w") as f:
        json.dump(asdict(profile), f, indent=2)


def load_profile(path: PathLike) -> PlayerProfile:
    with open(path) as f:
        data = json.load(f)
    history = [FacedRecord(**r) for r in data.pop("history", [])]
    return PlayerProfile(history=history, **data)


def save_scorecard(card: Scorecard, path: PathLike) -> None:
    with open(path, "w") as f:
        json.dump(asdict(card), f, indent=2)


def load_scorecard(path: PathLike) -> Scorecard:
    with open(path) as f:
        data = json.load(f)
    balls = [BallRecord(**b) for b in data.pop("balls", [])]
    return Scorecard(balls=balls, **data)

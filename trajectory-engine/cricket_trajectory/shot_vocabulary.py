"""
The vocabulary of cricket batting shots this project talks about, and — just
as important — which of them it can actually tell apart today.

Source: transcribed from three screenshots of a general-purpose AI chat
("Cricket shot names") that the project owner saved as the working shot list.
It is a plain reference list, not an authoritative laws-of-cricket or coaching
standard; the one-line descriptions are as given there, lightly tidied. Treat
the categories and regions as conventional, not exact.

Why keep it as data: shot_analysis.analyse_shot() only ever emits names from
this list (or a non-shot outcome), and a test enforces that. And every shot
the classifier can NOT produce is listed here with the reason, so "we detect
shots" never quietly means "we detect the six easy ones".

`detectable` says what it would take:
  "sensor"  — the current classifier can name it from the post-contact
              sensor reading (exit direction, height, speed) plus the ball's
              length. Still an inference, not validated against real shots.
  "needs bat/body information" — the ball's exit vector alone can't
              distinguish it from another shot; it depends on how the bat or
              body moved (which needs a bat tracker or a labelled-clip
              classifier — neither exists yet).
  "needs the bowler type" — defined against spin/slower bowling; the machine
              bowls pace, so it isn't currently meaningful.
  "needs video" — depends on the batter not playing at all (no contact),
              which the contact sensor alone reads as a miss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

FRONT_FOOT = "front-foot"
BACK_FOOT = "back-foot"
MODERN = "modern/creative"
DEFENSIVE = "defensive"

SENSOR = "sensor"
BAT_BODY = "needs bat/body information"
BOWLER_TYPE = "needs the bowler type"
VIDEO = "needs video"


@dataclass(frozen=True)
class ShotEntry:
    name: str
    category: str
    description: str
    detectable: str


def _e(name, category, description, detectable):
    return ShotEntry(name, category, description, detectable)


SHOT_VOCABULARY: List[ShotEntry] = [
    # Front-foot shots
    _e("Straight drive", FRONT_FOOT, "hit straight back past the bowler", SENSOR),
    _e("Cover drive", FRONT_FOOT, "through the covers", SENSOR),
    _e("Off drive", FRONT_FOOT, "between the bowler and mid-off", SENSOR),
    _e("On drive", FRONT_FOOT, "through mid-on", SENSOR),
    _e("Square drive", FRONT_FOOT, "through the off side, square of the wicket", SENSOR),
    _e("Flick", FRONT_FOOT, "wristy shot through midwicket / square leg", SENSOR),
    _e("Leg glance", FRONT_FOOT, "deflecting a ball from the pads toward fine leg", SENSOR),
    _e("Lofted drive", FRONT_FOOT, "aerial version of a drive", SENSOR),
    # Back-foot shots
    _e("Cut", BACK_FOOT, "short/wide ball hit square or behind square on the off side", BAT_BODY),
    _e("Square cut", BACK_FOOT, "sharply behind square", SENSOR),
    _e("Late cut", BACK_FOOT, "played very late toward third man", SENSOR),
    _e("Back-foot punch", BACK_FOOT, "controlled shot through the off side", SENSOR),
    _e("Pull", BACK_FOOT, "horizontal-bat shot, usually against a short ball", SENSOR),
    _e("Hook", BACK_FOOT, "attacking short/pitched ball, usually toward fine leg / square leg", SENSOR),
    # Modern / creative shots
    _e("Sweep", MODERN, "across the line toward the leg side", BOWLER_TYPE),
    _e("Paddle sweep", MODERN, "fine deflection behind square", BOWLER_TYPE),
    _e("Reverse sweep", MODERN, "sweep played toward the off side", BOWLER_TYPE),
    _e("Slog sweep", MODERN, "powerful sweep, often aerial", BOWLER_TYPE),
    _e("Slog", MODERN, "big hit across the line", BAT_BODY),
    _e("Switch hit", MODERN, "batter changes stance / hitting side", BAT_BODY),
    _e("Scoop / ramp", MODERN, "ball redirected over/behind the wicket", BAT_BODY),
    _e("Reverse scoop", MODERN, "ramp/scoop toward the opposite side", BAT_BODY),
    _e("Dilscoop", MODERN, "scoop over/behind the wicketkeeper", BAT_BODY),
    _e("Upper cut", MODERN, "short/wide ball lifted over slips / third man", BAT_BODY),
    _e("Helicopter shot", MODERN, "powerful wristy shot, associated with MS Dhoni", BAT_BODY),
    # Defensive shots
    _e("Forward defence", DEFENSIVE, "front-foot block", SENSOR),
    _e("Back-foot defence", DEFENSIVE, "back-foot block", SENSOR),
    _e("Leave", DEFENSIVE, "deliberately not playing the ball", VIDEO),
    _e("Dead-bat", DEFENSIVE, "soft defensive contact to kill the ball's momentum", BAT_BODY),
]

# Things analyse_shot() may return that are outcomes, not named shots.
NON_SHOT_OUTCOMES = (
    "mistimed / skied hit",
    "thick edge / deflection behind square",
    "no shot / missed",
)

_BY_KEY: Dict[str, ShotEntry] = {e.name.lower(): e for e in SHOT_VOCABULARY}


def lookup(name: str) -> Optional[ShotEntry]:
    return _BY_KEY.get(name.strip().lower())


def names() -> List[str]:
    return [e.name for e in SHOT_VOCABULARY]


def by_detectability() -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for e in SHOT_VOCABULARY:
        out.setdefault(e.detectable, []).append(e.name)
    return out

"""
Two independent witnesses to the same shot, combined.

  * The SENSOR reads the ball: where it went off the bat, how high, how fast
    (shot_analysis.analyse_shot). It sees the outcome of the shot.
  * The VIDEO reads the batter's hands (cv-pipeline/shot_from_video.py). It
    sees the stroke.

Neither is reliable alone in the ways that matter: the sensor can't tell a
pull from a hook or see a batter who swung and missed; the video can't see
where the ball went. They fail in different places, so when they agree the
shot is far more credible than either alone, and when they disagree that is
itself information — usually a tracking glitch or a mis-read, worth a coach's
look rather than a silent pick.

Rules (deliberately simple and stated):
  - The sensor's name wins if the two disagree (it measures the ball, not the
    hands), and the disagreement is reported, never hidden.
  - Video alone is reported as video alone, with its "not validated" label.
  - Nothing is invented: if neither source named a shot, no shot is named.

This module takes plain objects with the attributes it reads (`shot`,
`verdict`, `family` ...) so the physics package does not depend on the
computer-vision package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

DRIVE_NAMES = {"straight drive", "off drive", "cover drive", "on drive", "square drive", "flick",
               "leg glance", "lofted drive"}
HORIZONTAL_NAMES = {"cut", "square cut", "late cut", "pull", "hook", "back-foot punch"}
# The video says "Cut" for any off-side horizontal swing; the sensor is more specific.
CUT_FAMILY = {"cut", "square cut", "late cut"}

NO_CONTACT = "no shot / missed"


@dataclass(frozen=True)
class FusedShot:
    shot: Optional[str]
    source: str                 # which witnesses, and whether they agree
    agreement: str              # "agree" | "family only" | "disagree" | "single source" | "none"
    confidence: str
    notes: List[str] = field(default_factory=list)


def _family(name: str) -> Optional[str]:
    n = name.lower()
    if n in DRIVE_NAMES:
        return "drive"
    if n in HORIZONTAL_NAMES:
        return "horizontal"
    return None


def _names_match(sensor_name: str, video_name: str) -> str:
    s, v = sensor_name.lower(), video_name.lower()
    if s == v or (v == "cut" and s in CUT_FAMILY):
        return "agree"
    if _family(s) is not None and _family(s) == _family(v):
        return "family only"
    return "disagree"


def fuse_shot(sensor, video) -> FusedShot:
    """`sensor`: a shot_analysis.ShotAnalysis or None. `video`: a
    shot_from_video.VideoShotEstimate or None."""
    sensor_named = sensor is not None and sensor.shot != NO_CONTACT
    sensor_missed = sensor is not None and sensor.shot == NO_CONTACT
    video_named = video is not None and video.verdict == "shot named" and video.shot
    video_leave = video is not None and video.verdict == "no shot"
    video_swung = video is not None and video.verdict in ("shot named", "family only")

    if sensor_named and video_named:
        match = _names_match(sensor.shot, video.shot)
        if match == "agree":
            return FusedShot(sensor.shot, "sensor + video", "agree", "corroborated by two independent sources",
                             [f"The ball's direction and the batter's hand path both point to a {sensor.shot.lower()}."])
        if match == "family only":
            return FusedShot(sensor.shot, "sensor + video", "family only",
                             "corroborated at the level of shot type, not the exact name",
                             [f"Sensor: {sensor.shot}; video: {video.shot}. Same kind of shot, different name — the "
                              "sensor's is used."])
        return FusedShot(sensor.shot, "sensor (video disagrees)", "disagree",
                         "sources disagree — the sensor's reading is used",
                         [f"Sensor: {sensor.shot}; video: {video.shot}. Worth a coach's look: either the hand path "
                          "was mis-read or the ball was hit somewhere the stroke wouldn't suggest."])
    if sensor_named and video_leave:
        return FusedShot(sensor.shot, "sensor (video disagrees)", "disagree",
                         "sources disagree — the sensor's reading is used",
                         ["The sensor saw the ball come off the bat, but the video saw almost no hand movement — "
                          "likely a pose-tracking miss, or a very soft block."])
    if sensor_named:
        why = "no usable video" if video is None or video.verdict == "cannot tell" else "video could not name it"
        return FusedShot(sensor.shot, "sensor only", "single source",
                         getattr(sensor, "confidence", "inference from the sensor reading"),
                         [f"Named from the ball's exit reading alone ({why})."])
    if sensor_missed and video_swung:
        return FusedShot("swing and a miss", "sensor + video", "agree", "corroborated by two independent sources",
                         ["The video saw a swing and the sensor saw no contact with the bat."])
    if sensor_missed and video_leave:
        return FusedShot("leave / no shot", "sensor + video", "agree", "corroborated by two independent sources",
                         ["No contact at the sensor and almost no hand movement on video."])
    if sensor_missed:
        return FusedShot("no shot / missed", "sensor only", "single source", "no contact seen at the sensor",
                         ["Whether the batter swung and missed or left it alone needs the video."])
    if video_named:
        return FusedShot(video.shot, "video only", "single source", video.confidence,
                         ["No sensor reading for this delivery; named from the batter's hand path alone."])
    return FusedShot(None, "none", "none", "n/a", ["Neither source could name a shot for this delivery."])

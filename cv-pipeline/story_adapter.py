"""
Turn what the camera pipeline found for one delivery into the input of the whole-delivery story
(trajectory-engine/cricket_trajectory/delivery_story.py).

Kept in its own small module so the physics package never has to import the computer-vision package:
this side knows both, and translates. Every argument is optional; a field the pipeline did not produce
is left None, and the story reports it as "not measured" rather than inventing it.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "trajectory-engine"))

from cricket_trajectory.delivery_story import VideoDeliveryInfo  # noqa: E402


def video_info_from(
    body=None,             # body_vectors.BodyVectorReport
    footwork=None,         # footwork.FootworkReport
    shot=None,             # shot_from_video.VideoShotEstimate
    bowler=None,           # bowler_analysis.BowlerActionEstimate
    footage=None,          # footage_check.FootageReport
    swing_time_s: Optional[float] = None,
    notes: Optional[List[str]] = None,
) -> VideoDeliveryInfo:
    info = VideoDeliveryInfo(swing_time_s=swing_time_s, notes=list(notes or []))
    if footage is not None:
        info.footage_verdict = footage.verdict
    if body is not None:
        info.swing_verdict = body.swing_verdict
        info.peak_hand_speed = body.peak_hand_speed
        info.hand_path_net_direction = body.hand_path_net_direction
        info.notes += [n for n in body.caveats if "whole body was moving" in n or "glitch" in n]
    if footwork is not None:
        info.footwork_class = footwork.footwork_class
        if footwork.footwork_class in ("front-foot", "back-foot", "minimal"):
            info.front_stride = footwork.front_stride
            info.lead_time_s = footwork.lead_time_s
            info.hip_shift = footwork.hip_shift
        info.foreshortened = footwork.foreshortened
    if shot is not None:
        info.shot, info.shot_verdict, info.shot_family = shot.shot, shot.verdict, shot.family
        info.shot_reasons = list(shot.reasons)
    if bowler is not None:
        info.bowler_arm = bowler.arm_side
        info.bowler_action = bowler.arm_action_label
        info.bowler_release_height = bowler.release_height_torsos
    return info

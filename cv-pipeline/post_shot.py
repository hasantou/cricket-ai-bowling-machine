"""
What did the batter do AFTER the shot? — read from the same body-vector track, extended past the
delivery window, into the couple of seconds where a batter would run, stay put, or leave.

Read this before trusting it, because it is the closest this project comes to guessing at a wicket,
and it deliberately stops short of that:

  * It CANNOT detect a wicket, and does not try to. Bowled needs the stumps; caught needs a fielder
    and the ball's flight; lbw and stumped need contact with the pad or a broken wicket. None of
    that is visible in a single camera with no ball tracking (see cv-pipeline/README.md for why ball
    tracking itself was abandoned as infeasible on this footage). Anything claiming to detect a
    dismissal from this signal alone would be a guess dressed up as a measurement.
  * It CANNOT count runs. Whether a run was completed needs the crease lines and the fielding side's
    throw, neither of which this reads.
  * What it DOES measure, honestly: how far the tracked hip position moved, and whether the person
    was still trackable, in the seconds after the swing. That gives three outcomes:
        "ran"                    — sustained, large net movement in a consistent direction
        "stayed at the crease"   — tracked throughout, but net movement stayed small
        "lost track after the shot" — landmarks disappear for a stretch after the shot
    The third is the one to be most careful with: a person can vanish from tracking because they
    walked out of frame after being given out, because the camera cut to a replay, because they
    are obscured by a fielder, or because the pose model simply lost them — this signal cannot tell
    those apart, and is captioned as "worth a look", not "the batter was out".

Built to extend body_vectors.py's already-tested joint-tracking primitives (smoothing, confidence
gating, torso scale) past the single delivery window they normally cover.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from body_vectors import JOINTS, MAX_BRIDGE_SECONDS, MIN_VISIBILITY, SMOOTH_SECONDS, _bridge, _series, _smooth, _torso_scale

POST_SHOT_WINDOW_SECONDS = 2.0     # how far past the swing to look
RUN_MIN_NET_TRAVEL_TORSOS = 3.0    # net hip displacement over the window, to call it "ran"
STAYED_MAX_NET_TRAVEL_TORSOS = 1.5 # below this, "stayed at the crease"; between the two is "unclear"
LOST_TRACK_GAP_SECONDS = 0.5       # a gap at least this long, starting soon after the shot, is "lost track"
LOST_TRACK_STARTS_WITHIN_SECONDS = 0.8

RAN = "ran"
STAYED = "stayed at the crease"
LOST_TRACK = "lost track after the shot"
UNCLEAR = "unclear (movement between the two)"


@dataclass(frozen=True)
class PostShotReport:
    classification: str
    net_travel_torsos: Optional[float]      # None if lost track before any real measurement
    frames_available: int                   # how many frames of the requested window existed (clip may end sooner)
    frames_tracked: int
    gap_started_at_s: Optional[float]       # seconds after the shot the tracking gap began, if classification is LOST_TRACK
    caveats: List[str] = field(default_factory=list)


def analyse_post_shot_movement(
    aligned: Sequence[Optional[list]], fps: float, aspect: float, shot_frame: int,
    window_seconds: float = POST_SHOT_WINDOW_SECONDS,
) -> Optional[PostShotReport]:
    """`aligned` is the WHOLE clip's per-frame landmarks (camera-stabilised), the same list
    video_pipeline.analyse_video already holds — not just one delivery's cropped window, because
    this needs to look further forward than a delivery window normally extends. `shot_frame` is the
    real frame index of the swing (body_vectors.BodyVectorReport.peak_frame mapped back to the whole
    clip). Returns None if there is nothing after the shot to look at (it was the last frame) or the
    batter's own scale (torso length) can't be established near the shot."""
    n_total = len(aligned)
    end = min(n_total, shot_frame + 1 + int(round(window_seconds * fps)))
    window = aligned[shot_frame:end]
    frames_available = len(window) - 1              # frames strictly after the shot
    if frames_available < 1:
        return None

    scale_window = aligned[max(0, shot_frame - int(fps)): shot_frame + 1]
    scale = _torso_scale(scale_window, aspect)
    if scale is None or scale < 1e-6:
        return None

    hip_l, trusted_l = _series(window, JOINTS["left hip"], aspect)
    hip_r, trusted_r = _series(window, JOINTS["right hip"], aspect)
    trusted = trusted_l & trusted_r
    frames_tracked = int(trusted.sum())

    # A gap: `LOST_TRACK_GAP_SECONDS` or more of consecutive untracked frames, starting within
    # `LOST_TRACK_STARTS_WITHIN_SECONDS` of the shot (a gap much later is the NEXT ball's business,
    # not this one's).
    gap_frames_needed = max(1, int(round(LOST_TRACK_GAP_SECONDS * fps)))
    starts_by = int(round(LOST_TRACK_STARTS_WITHIN_SECONDS * fps))
    run_start, run_len = None, 0
    gap_at = None
    for i, ok in enumerate(trusted[1:], start=1):        # frame 0 is the shot itself; look after it
        if not ok:
            if run_start is None:
                run_start = i
            run_len += 1
            if run_len >= gap_frames_needed and run_start <= starts_by and gap_at is None:
                gap_at = run_start
        else:
            run_start, run_len = None, 0

    caveats = [
        "This cannot detect a wicket or count runs — it only measures whether the tracked person "
        "kept moving, stayed put, or dropped out of tracking after the shot.",
    ]

    if gap_at is not None:
        caveats.append(
            "A tracking gap this soon after a shot can mean the batter walked off, the camera cut "
            "away, a fielder blocked the view, or the pose model simply lost them — this signal "
            "can't tell those apart. Worth a look, not a verdict."
        )
        return PostShotReport(LOST_TRACK, None, frames_available, frames_tracked, gap_at / fps, caveats)

    half = max(1, int(round(SMOOTH_SECONDS * fps / 2)))
    gap = max(1, int(round(MAX_BRIDGE_SECONDS * fps)))
    hips = _smooth(_bridge((hip_l + hip_r) / 2.0, gap), half)
    valid = hips[~np.isnan(hips[:, 0])]
    if len(valid) < 2:
        caveats.append("Too little of the batter was tracked after the shot to say anything.")
        return PostShotReport(UNCLEAR, None, frames_available, frames_tracked, None, caveats)

    net = float(np.hypot(*(valid[-1] - valid[0]))) / scale
    if frames_tracked < 0.5 * frames_available:
        caveats.append("The batter was tracked in under half the frames after the shot; treat this as rough.")

    if net >= RUN_MIN_NET_TRAVEL_TORSOS:
        cls = RAN
    elif net <= STAYED_MAX_NET_TRAVEL_TORSOS:
        cls = STAYED
    else:
        cls = UNCLEAR
        caveats.append("Net movement was between the 'stayed' and 'ran' thresholds — not called either way.")
    return PostShotReport(cls, net, frames_available, frames_tracked, None, caveats)

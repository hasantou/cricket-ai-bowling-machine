"""
Did the stumps break? A real, separate visual signal for a wicket — distinct from
outcome_from_video.py's batter-arm-motion guess, which was never built to see this and, tested
against a real clip, guessed "four" and "six" on an actual wicket because a broken stump changes
nothing about how the batter's arms moved.

The idea: a broken wicket is a big, SUSTAINED visual change in a small, known region of the frame —
found by testing on a real clip (2026-09-23, ICC broadcast, a genuine wicket, confirmed by ear from
the commentary) that the stumps go from three neat upright posts to scattered debris within about
6-10 frames at 60fps, and STAY disrupted (unlike a ball merely flying past the stumps, or ordinary
camera noise, which is a brief blip that the region returns from). Sustain, not just size, is what
tells the two apart.

What this genuinely needs and does not invent around: **the stumps' own region of the frame**, the
same honest limitation `video_pipeline.py`'s batter-region selector already has for "who is the
batter" — a fixed camera can be told once where the stumps are; a panning shot would need the region
to move with the camera the same way `advance_roi()` already does for the batter, which this module
does not yet do (see `find_disruption` below — it assumes a still region, and says so).

Tried against the one real clip this was built for, honest result: picking the stumps' pixel region
by eye did NOT work cleanly. In this clip the batter's own body substantially overlaps the stumps
throughout the delivery and follow-through, so a rectangular region wide enough to contain the
stumps also captured his shot and reaction — the measured region-change signal hovered at 15-19
almost continuously through the whole delivery (batter motion), with no clean, isolated spike at the
real break (visually confirmed at ~frame 56-65 of that clip by inspecting frames directly), and the
single largest change in the whole clip (frame 235) was an unrelated EDIT CUT to a different camera
angle, not the wicket. The algorithm below (sustained vs. brief disruption) is sound and unit-tested
on the synthetic case it's designed for; what is unsolved is automatically finding or precisely
hand-picking the stumps' own region on real footage where the batter is right in front of them —
either a person needs to mark it much more precisely per clip than eyeballing a frame achieves, or
the batter's own tracked silhouette (already computed elsewhere in this pipeline) needs to be
excluded from the region before measuring change. Not attempted yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

DISRUPTION_MIN_CHANGE = 18.0      # mean absolute grey-level change (0-255) within the region, to count as "disrupted"
SUSTAIN_FRAMES = 5                # the region must stay disrupted for at least this many consecutive frames
                                   # (a ball passing through is a 1-2 frame blip; a broken wicket stays broken)
BASELINE_FRAMES = 10              # frames before the event used to establish "what this region normally looks like"


@dataclass(frozen=True)
class WicketEvent:
    start_frame: int              # first frame the region is disrupted
    confirmed_frame: int          # the frame at which SUSTAIN_FRAMES of disruption was confirmed
    peak_change: float
    region: Tuple[int, int, int, int]


@dataclass(frozen=True)
class WicketDisruptionReport:
    events: List[WicketEvent]
    region_changes: List[float]   # per-frame mean change within the region, for a human to inspect/re-tune against
    notes: List[str] = field(default_factory=list)

    @property
    def wicket_detected(self) -> bool:
        return len(self.events) > 0


def _region_grey(frame: np.ndarray, region: Tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = region
    return cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)


def find_disruption(
    video_path: str, stumps_region: Tuple[int, int, int, int],
    min_change: float = DISRUPTION_MIN_CHANGE, sustain_frames: int = SUSTAIN_FRAMES,
    baseline_frames: int = BASELINE_FRAMES,
) -> WicketDisruptionReport:
    """`stumps_region` is (x0, y0, x1, y1) in pixels, where the stumps sit in THIS clip's frame —
    told by a human, the same honest limitation as the batter-region selector. Assumes a STILL
    camera: a pan would move the stumps out of a fixed region and either miss a real break or
    manufacture a fake one, exactly the failure mode `camera_motion.py` exists to prevent
    elsewhere in this pipeline — not handled here yet."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    regions: List[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            regions.append(_region_grey(frame, stumps_region))
    finally:
        cap.release()

    n = len(regions)
    notes: List[str] = []
    if n <= baseline_frames:
        return WicketDisruptionReport([], [], ["Clip too short to establish a baseline for this region."])

    baseline = np.mean(regions[:baseline_frames], axis=0)
    changes = [float(np.mean(np.abs(r - baseline))) for r in regions]

    events: List[WicketEvent] = []
    run_start, run_len, run_peak = None, 0, 0.0
    confirmed_until = -1
    for i, change in enumerate(changes):
        if change >= min_change:
            if run_start is None:
                run_start = i
            run_len += 1
            run_peak = max(run_peak, change)
            if run_len == sustain_frames and run_start > confirmed_until:
                events.append(WicketEvent(run_start, i, run_peak, stumps_region))
                confirmed_until = i + sustain_frames  # don't re-report the same ongoing disruption
        else:
            run_start, run_len, run_peak = None, 0, 0.0

    if not events:
        notes.append(
            f"No sustained disruption (>= {sustain_frames} consecutive frames past {min_change:.0f} "
            "change) found in this region across the clip."
        )
    return WicketDisruptionReport(events, changes, notes)

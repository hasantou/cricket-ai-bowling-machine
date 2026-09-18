"""
Turns delivery_segmentation.py's batch "find every delivery in a whole
clip" logic into something usable on a live, continuously-arriving
landmark stream from a camera mounted on the machine — without
reimplementing the peak-picking math itself, since that's already real,
tested logic that shouldn't have two divergent copies.

Three real problems live operation adds, all found by actually running
this against real footage played back frame by frame (see
demo_live_delivery_detection.py) — not assumed up front:

1. find_delivery_windows() needs WINDOW_AFTER_PEAK_SEC of trailing
   context after a swing before it can draw a window around it — on a
   live stream, that context doesn't exist yet the instant the swing
   happens, only ~0.8s later. Straightforward, and by itself not the
   real problem here.

2. find_delivery_windows() is a GLOBAL, retrospective algorithm — greedy
   peak-picking with a wide suppression radius (MIN_DELIVERY_SEPARATION_SEC,
   2 seconds) around each peak it claims. Re-running it on a growing
   buffer means an early, smaller local peak can look "complete" (enough
   WINDOW_AFTER_PEAK_SEC trailing context exists for IT) well before a
   later, genuinely bigger swing has happened — and once that bigger
   swing arrives, the algorithm's own greedy-biggest-first logic would
   have suppressed the smaller one entirely, because they're within 2
   seconds of each other. Fix: a peak isn't safe to report until the
   buffer extends MIN_DELIVERY_SEPARATION_SEC past it, not just
   WINDOW_AFTER_PEAK_SEC — only then is it guaranteed that any bigger,
   closer peak that could have suppressed it has already had the chance
   to.

3. A longer real session (multiple deliveries, minutes apart) trims the
   rolling buffer to keep it bounded — but a "reported up to" marker
   tracked as a position *relative to the current buffer* silently drains
   toward zero every time frames get trimmed off the front, even during
   the quiet gap between deliveries when nothing new is being reported.
   Given enough quiet frames, that drained marker can fall low enough
   that an OLD delivery — still physically sitting in the buffer,
   unchanged — looks unreported again and gets emitted a second (or
   third, or sixth) time. Confirmed on a real ~75-second, 9-delivery
   clip: one delivery was re-reported six times before the next real one
   arrived. Fix: track "reported up to" as an absolute, monotonically
   increasing frame count across the whole session, converted against a
   running trim offset — never a position that trimming itself can erode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from delivery_segmentation import (
    MIN_DELIVERY_SEPARATION_SEC, WINDOW_AFTER_PEAK_SEC, WINDOW_BEFORE_PEAK_SEC, find_delivery_windows,
)
from feature_extraction import FrameLandmarks


@dataclass(frozen=True)
class DetectedDelivery:
    """One reported delivery window, in both coordinate systems a caller
    might need — deliberately explicit about which is which, since
    mixing them up is an easy, real mistake: this project's own
    real-footage validation script (demo_live_delivery_detection.py)
    made exactly this error comparing raw tuples, on its first attempt.

    `start`/`end` are relative to whatever buffered_landmarks() returns
    RIGHT NOW — use these immediately to slice it
    (buffered_landmarks()[start:end]). `absolute_start`/`absolute_end`
    are stable for the life of the whole session, safe to log or compare
    against later — buffer-relative indices get silently reused as the
    sliding window moves forward, so two genuinely different deliveries
    can (and, on real footage, did) end up with identical (start, end)
    while their absolute positions correctly differ.
    """
    start: int
    end: int
    absolute_start: int
    absolute_end: int


class LiveDeliveryDetector:
    def __init__(self, fps: float, buffer_seconds: float = 15.0):
        self.fps = fps
        self._buffer_max_frames = max(1, int(buffer_seconds * fps))
        self._landmarks: List[FrameLandmarks] = []
        self._absolute_offset = 0        # total frames ever trimmed from the buffer's front
        self._reported_up_to_absolute = 0  # ABSOLUTE frame index already reported - never decreases
        self._after_frames = int(WINDOW_AFTER_PEAK_SEC * fps)
        self._full_window_frames = int(WINDOW_BEFORE_PEAK_SEC * fps) + self._after_frames + 1
        self._separation_frames = int(MIN_DELIVERY_SEPARATION_SEC * fps)

    def add_frame(self, landmarks: Optional[FrameLandmarks]) -> List[DetectedDelivery]:
        """Call once per live frame. `landmarks` is None for a frame
        where no person was detected — skipped, same convention
        PoseEstimator.extract_landmarks_from_frames() already uses, so
        indices here stay consistent with the rest of this pipeline.

        Returns newly-safe-to-report deliveries — use `.start`/`.end` to
        slice buffered_landmarks() right away
        (what outcome_bridge.VisionOutcomeEstimator.estimate() expects),
        and `.absolute_start`/`.absolute_end` for anything that logs or
        compares positions across time. Each real delivery is reported
        exactly once for the life of this detector, regardless of how
        long the session runs or how much buffer trimming happens in
        between.
        """
        if landmarks is not None:
            self._landmarks.append(landmarks)

        overflow = len(self._landmarks) - self._buffer_max_frames
        if overflow > 0:
            self._landmarks = self._landmarks[overflow:]
            self._absolute_offset += overflow

        if len(self._landmarks) < 2:
            return []
        try:
            windows = find_delivery_windows(self._landmarks, self.fps)
        except ValueError:
            return []  # no clear swing in the buffer yet - expected, not an error

        buffer_len = len(self._landmarks)
        new_windows = []
        newly_reported_absolute_ends = []
        for start, end in windows:
            if (end - start) != self._full_window_frames:
                continue  # clamped short - not even trailing-context-complete yet
            peak_idx = end - self._after_frames - 1
            safe_to_report = buffer_len - peak_idx >= self._separation_frames
            absolute_start = start + self._absolute_offset
            if safe_to_report and absolute_start >= self._reported_up_to_absolute:
                absolute_end = end + self._absolute_offset
                new_windows.append(DetectedDelivery(start, end, absolute_start, absolute_end))
                newly_reported_absolute_ends.append(absolute_end)
        if newly_reported_absolute_ends:
            self._reported_up_to_absolute = max(newly_reported_absolute_ends)
        return new_windows

    def buffered_landmarks(self) -> List[FrameLandmarks]:
        """The current rolling buffer — window indices from add_frame()
        are relative to this list at the moment they were returned."""
        return self._landmarks

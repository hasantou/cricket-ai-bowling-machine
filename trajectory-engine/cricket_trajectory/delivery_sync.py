"""
Line up the camera's deliveries with the sensors' deliveries in time.

On the real machine there are two independent clocks: the sensors stamp the
release and the bat contact on the machine's clock, and the camera's swing
peaks are timed on the video's. Before anyone can say "this swing produced
that exit reading", the two lists of moments have to be matched — with an
unknown offset between the clocks, with deliveries the camera missed or the
sensor missed, and with a little drift.

Method: try every offset that would line one video moment up with one sensor
moment; keep the offset that puts the most pairs within a tolerance (ties go to
the smaller residuals, then the smaller offset). Then pair them one-to-one
nearest-first. If the offset is wrong nothing lines up, so a bad or ambiguous
alignment shows itself as few matches rather than a confident wrong pairing.

What this deliberately does NOT do: it never forces a match. A video delivery
with no sensor partner (or the reverse) is reported as unmatched — that is the
honest answer, and often the interesting one (a swing with no contact, or
contact the camera missed).

Tested on synthetic timestamps only: there is no real machine yet, so real
clock behaviour (jitter, drift, latency in the sensor path) is unverified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_TOLERANCE_S = 0.15       # a swing peak and the contact it caused are this close on a common clock
MAX_OFFSET_S = 60.0
DRIFT_WARN = 0.002               # >2 ms of clock disagreement per second across the session is worth flagging


@dataclass(frozen=True)
class SyncResult:
    offset_s: Optional[float]                    # add to a sensor time to get the matching video time
    matches: List[Tuple[int, int]]               # (video index, sensor index)
    residuals_s: List[float]                     # video_time - (sensor_time + offset), per match
    unmatched_video: List[int]
    unmatched_sensor: List[int]
    tolerance_s: float
    drift_s_per_s: Optional[float]               # clock drift estimated from the matches, if 3+
    ok: bool
    notes: List[str] = field(default_factory=list)

    def sensor_for_video(self, video_index: int) -> Optional[int]:
        return next((s for v, s in self.matches if v == video_index), None)

    def video_for_sensor(self, sensor_index: int) -> Optional[int]:
        return next((v for v, s in self.matches if s == sensor_index), None)


def _pair(video: np.ndarray, sensor: np.ndarray, offset: float, tol: float):
    """One-to-one nearest-first pairing within tolerance."""
    candidates = sorted(
        (abs(v - (s + offset)), i, j)
        for i, v in enumerate(video) for j, s in enumerate(sensor) if abs(v - (s + offset)) <= tol
    )
    used_v, used_s, pairs = set(), set(), []
    for d, i, j in candidates:
        if i in used_v or j in used_s:
            continue
        used_v.add(i); used_s.add(j); pairs.append((i, j))
    return sorted(pairs)


def align(
    video_times_s: Sequence[float], sensor_times_s: Sequence[float], tolerance_s: float = DEFAULT_TOLERANCE_S,
) -> SyncResult:
    """`video_times_s`: swing-peak times on the video clock. `sensor_times_s`: bat-contact times
    on the sensor clock (or release times, if that is what is being aligned)."""
    video = np.asarray(sorted(video_times_s), dtype=float)
    sensor = np.asarray(sorted(sensor_times_s), dtype=float)
    notes: List[str] = []
    if len(video) == 0 or len(sensor) == 0:
        return SyncResult(None, [], [], list(range(len(video))), list(range(len(sensor))), tolerance_s, None, False,
                          ["One of the two lists is empty, so there is nothing to align."])

    best = None
    for v in video:
        for s in sensor:
            off = float(v - s)
            if abs(off) > MAX_OFFSET_S:
                continue
            pairs = _pair(video, sensor, off, tolerance_s)
            resid = sum(abs(video[i] - (sensor[j] + off)) for i, j in pairs)
            key = (-len(pairs), resid, abs(off))
            if best is None or key < best[0]:
                best = (key, off, pairs)
    if best is None:
        return SyncResult(None, [], [], list(range(len(video))), list(range(len(sensor))), tolerance_s, None, False,
                          [f"No offset within {MAX_OFFSET_S:.0f} s lines any video moment up with a sensor moment."])
    _, off, pairs = best

    # Refine the offset as the median residual of the matches, then re-pair once.
    if pairs:
        off = float(np.median([video[i] - sensor[j] for i, j in pairs]))
        pairs = _pair(video, sensor, off, tolerance_s)
    residuals = [float(video[i] - (sensor[j] + off)) for i, j in pairs]
    unmatched_v = [i for i in range(len(video)) if i not in {p[0] for p in pairs}]
    unmatched_s = [j for j in range(len(sensor)) if j not in {p[1] for p in pairs}]

    drift = None
    if len(pairs) >= 3:
        xs = np.array([sensor[j] for _, j in pairs])
        ys = np.array([video[i] - sensor[j] for i, j in pairs])
        drift = float(np.polyfit(xs, ys, 1)[0])
        if abs(drift) > DRIFT_WARN:
            notes.append(f"The two clocks drift apart by about {drift * 1000:.1f} ms per second across the session; "
                         "matches late in a long session are less certain.")

    ok = len(pairs) >= 2 or (len(pairs) == 1 and len(video) == 1 and len(sensor) == 1)
    if not ok:
        notes.append("Fewer than two deliveries lined up, so the clock offset is not established and no pairing "
                     "should be trusted.")
    if unmatched_v:
        notes.append(f"{len(unmatched_v)} video moment(s) have no sensor partner (a swing with no contact, or contact "
                     "the sensor missed).")
    if unmatched_s:
        notes.append(f"{len(unmatched_s)} sensor moment(s) have no video partner (contact the camera missed, or a delivery "
                     "it did not see).")
    return SyncResult(off, pairs, residuals, unmatched_v, unmatched_s, tolerance_s, drift, ok, notes)

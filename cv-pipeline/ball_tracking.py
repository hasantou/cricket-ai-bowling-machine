"""
Turns sparse, noisy per-frame ball detections into one continuous
trajectory, so a weak detector doesn't need to fire on every frame to be
useful — only on enough of them to constrain a fit.

Honesty check, same discipline as outcome_bridge.py: there is no ball
detector committed to this repo. An earlier YOLOv8 experiment (trained on
Roboflow's CC BY 4.0 cricket-ball dataset) reached ~12% recall on real
nets footage — weak enough, and never checked into git (trained weights
are gitignored, and the training script itself was scratch work, not
committed) — that it's a result to design around, not a component to
import. This module is that design: it assumes nothing about *how*
detections were produced, only that they arrive as
`(frame_index, x_px, y_px, confidence)` tuples, sparse and occasionally
wrong. Feed it real detections once a detector exists (this one or a
better one) and nothing here needs to change.

This is deliberately NOT the real 3D physics in
`trajectory-engine/cricket_trajectory/` — that module has no visibility
into a camera at all. This one only ever sees 2D pixel positions in one
camera's image plane, and fits the two simplest curves that plausibly
explain them: roughly constant horizontal image-velocity (a line) and a
gravity-driven arc (a parabola) in the vertical axis. That is a modelling
choice suited to a broadly side-on camera, not a universal one — a
behind-the-bowler camera would need a different pair of curves. Treat the
degree-1/degree-2 choice below as adjustable, not load-bearing physics.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


class BallTrajectoryFitError(Exception):
    """Too few detections, or no consistent trajectory among them, to
    fit reliably. Raised rather than returning a low-confidence guess —
    a silently-wrong trajectory is worse than a clear refusal here."""


@dataclass
class BallDetection:
    """One per-frame detector output — real or synthetic. `confidence`
    is carried through but not currently used by the fit itself (the
    RANSAC inlier count already down-weights unreliable points by
    construction); it's here so a real detector's score isn't thrown
    away before anyone decides whether to use it."""
    frame_index: int
    x_px: float
    y_px: float
    confidence: float = 1.0


@dataclass
class FittedBallTrajectory:
    """x(frame) = x_coeffs[0]*frame + x_coeffs[1] (a line);
    y(frame) = y_coeffs[0]*frame**2 + y_coeffs[1]*frame + y_coeffs[2] (a parabola).
    `inlier_frames`/`outlier_frames` are which of the *input* detections'
    frame numbers the winning fit accepted or rejected — useful for
    seeing whether a detector's false positives actually got filtered."""
    x_coeffs: Tuple[float, float]
    y_coeffs: Tuple[float, float, float]
    inlier_frames: List[int]
    outlier_frames: List[int]
    fps: float

    def position_at(self, frame_index: int) -> Tuple[float, float]:
        x = self.x_coeffs[0] * frame_index + self.x_coeffs[1]
        y = (self.y_coeffs[0] * frame_index ** 2
             + self.y_coeffs[1] * frame_index
             + self.y_coeffs[2])
        return x, y


def _fit_curves(sample: List[BallDetection]):
    frames = [d.frame_index for d in sample]
    xs = [d.x_px for d in sample]
    ys = [d.y_px for d in sample]
    x_coeffs = tuple(np.polyfit(frames, xs, deg=1))
    y_coeffs = tuple(np.polyfit(frames, ys, deg=2))
    return x_coeffs, y_coeffs


def fit_ball_trajectory(
    detections: List[BallDetection],
    fps: float,
    inlier_threshold_px: float = 15.0,
    min_inliers: int = 3,
    min_span_seconds: float = 0.45,
    max_iterations: int = 300,
    rng: Optional[random.Random] = None,
) -> FittedBallTrajectory:
    """RANSAC over a line (horizontal) + parabola (vertical) motion model:
    repeatedly fit to a random minimal 3-point sample, count how many of
    *all* detections that fit explains within `inlier_threshold_px`
    pixels, keep the best-supported sample that also clears
    `min_span_seconds`, then refit once more using every inlier it found
    for the final, more accurate curve.

    `min_span_seconds` exists because inlier *count* alone isn't enough:
    measured across 250 synthetic trials at the real ~12% recall rate,
    fits whose inliers all happened to cluster in one short stretch of
    time looked fine near that cluster but blew up wildly (worst
    observed: 822px off) when extrapolated further away — a quadratic
    fit from 3 nearby points has zero information about curvature
    anywhere it hasn't seen. The default (0.45s) was picked from that
    same sweep: it caps the worst-case error at 67px (vs. 822px
    unconstrained) in exchange for ~5 percentage points more trials
    correctly refusing to fit at all rather than producing a bad one.

    This is deliberately measured in real seconds (via `fps`), not as a
    fraction of however much time the detections themselves happen to
    span — an earlier version of this function made exactly that mistake
    (requiring inliers to span a fraction of the *detections'* own
    range), which is circular in precisely the way
    `cv-pipeline/outcome_bridge.py`'s original `on_time` heuristic was:
    when there are only 3 detections total, they trivially "span" 100% of
    their own range no matter how clustered in time they really are, so
    the check could never actually reject the worst cases it existed to
    catch. Tying it to real time instead, independent of how many
    detections exist, is the fix (same lesson, second time it's come up).

    Raises BallTrajectoryFitError rather than returning a guess when
    there's too little data, no sample explains enough of it (e.g. all
    detections are false positives with no shared trajectory), or every
    well-supported fit is too temporally narrow to extrapolate safely.
    """
    if len(detections) < 3:
        raise BallTrajectoryFitError(
            f"Need at least 3 ball detections to fit a trajectory, got {len(detections)}."
        )
    min_inliers = max(3, min_inliers)
    rng = rng or random.Random()
    required_span_frames = min_span_seconds * fps

    best_inliers: List[BallDetection] = []
    best_ignoring_span: List[BallDetection] = []  # tracked only to explain a span-only failure
    for _ in range(max_iterations):
        sample = rng.sample(detections, 3)
        if len({d.frame_index for d in sample}) < 3:
            continue  # need 3 distinct frames to determine the curves uniquely
        try:
            x_coeffs, y_coeffs = _fit_curves(sample)
        except np.linalg.LinAlgError:
            continue

        inliers = []
        for d in detections:
            pred_x = x_coeffs[0] * d.frame_index + x_coeffs[1]
            pred_y = y_coeffs[0] * d.frame_index ** 2 + y_coeffs[1] * d.frame_index + y_coeffs[2]
            error_px = ((pred_x - d.x_px) ** 2 + (pred_y - d.y_px) ** 2) ** 0.5
            if error_px <= inlier_threshold_px:
                inliers.append(d)

        if len(inliers) > len(best_ignoring_span):
            best_ignoring_span = inliers

        inlier_frames = [d.frame_index for d in inliers]
        span_ok = (max(inlier_frames) - min(inlier_frames)) >= required_span_frames if inliers else False
        if span_ok and len(inliers) > len(best_inliers):
            best_inliers = inliers

    if len(best_inliers) < min_inliers:
        if len(best_ignoring_span) >= min_inliers:
            best_span_frames = (
                max(d.frame_index for d in best_ignoring_span)
                - min(d.frame_index for d in best_ignoring_span)
            )
            raise BallTrajectoryFitError(
                f"Found {len(best_ignoring_span)} inlier detection(s), but the best-supported "
                f"set only spans {best_span_frames / fps:.2f}s of real time (need at least "
                f"{min_span_seconds:.2f}s) — too clustered together to extrapolate the rest "
                f"of the clip safely."
            )
        raise BallTrajectoryFitError(
            f"Best fit found only {len(best_inliers)} inlier detection(s) out of "
            f"{len(detections)} (need at least {min_inliers}) — too sparse or too "
            f"noisy to fit a trajectory reliably."
        )

    # Refit on every inlier found, not just the 3-point sample that found
    # them, so the final curve isn't overly sensitive to which 3 points
    # RANSAC happened to pick.
    x_coeffs, y_coeffs = _fit_curves(best_inliers)
    inlier_frames = sorted({d.frame_index for d in best_inliers})
    outlier_frames = sorted({d.frame_index for d in detections} - set(inlier_frames))
    return FittedBallTrajectory(
        x_coeffs=x_coeffs, y_coeffs=y_coeffs,
        inlier_frames=inlier_frames, outlier_frames=outlier_frames, fps=fps,
    )


def recover_missing_frames(
    detections: List[BallDetection],
    fps: float,
    frame_range: Tuple[int, int],
    **fit_kwargs,
) -> Dict[int, Tuple[float, float]]:
    """Fits a trajectory from whatever detections exist, then reports an
    interpolated (x, y) for every frame in `frame_range` inclusive — the
    actual point of this module: a detector only needs to be right often
    enough to constrain the fit, not on every single frame, for every
    frame to end up with a usable ball position."""
    trajectory = fit_ball_trajectory(detections, fps, **fit_kwargs)
    start, end = frame_range
    return {frame: trajectory.position_at(frame) for frame in range(start, end + 1)}


# ---------------------------------------------------------------------------
# Bat-ball impact segmentation: Incoming -> Impact -> Outgoing.
#
# A design for post-shot exit-trajectory extraction (stereo cameras,
# ChArUco calibration, 3D triangulation) proposed alongside this project
# described a clean three-phase model worth adopting regardless of
# whether that hardware ever exists: a ball's flight before it's struck
# tells you nothing about where it's going after, so once contact is
# detected, the "incoming" history should be discarded and only the
# "outgoing" phase fitted. The two functions below implement that model
# in 2D image-space (this module's only domain — no camera calibration,
# no real 3D), working on whatever per-frame detections exist, real or
# synthetic. Detecting contact from velocity discontinuity, and trimming
# a net-collision-corrupted tail via an acceleration spike, are both
# techniques from that same design — genuinely useful with a single
# camera's pixel coordinates, not only a calibrated stereo rig.
# ---------------------------------------------------------------------------


@dataclass
class ImpactSplit:
    """One detection sequence split at the moment of bat-ball contact.
    `incoming` and `outgoing` are disjoint, frame-ordered slices of the
    input; `impact_frame` is the frame index the split happened at."""
    incoming: List[BallDetection]
    outgoing: List[BallDetection]
    impact_frame: int


def _frame_velocities(ordered: List[BallDetection]) -> List[Tuple[int, float, float]]:
    """Per-consecutive-pair finite-difference velocity (px/frame) between
    already-frame-sorted detections. Needs reasonably dense detections —
    a large gap between two consecutive entries makes the resulting
    "velocity" over that gap noisy or meaningless; this doesn't correct
    for that, only computes what's asked of it."""
    velocities = []
    for a, b in zip(ordered, ordered[1:]):
        dt_frames = b.frame_index - a.frame_index
        if dt_frames <= 0:
            continue
        velocities.append((b.frame_index, (b.x_px - a.x_px) / dt_frames, (b.y_px - a.y_px) / dt_frames))
    return velocities


def find_impact_and_split(
    detections: List[BallDetection],
    velocity_jump_threshold_px_per_frame: float,
) -> Optional[ImpactSplit]:
    """
    Estimates frame-to-frame velocity from consecutive detections and
    marks the first point where it changes abruptly — a jump in (vx, vy)
    between consecutive estimates at or past
    `velocity_jump_threshold_px_per_frame` — as bat-ball contact.
    Everything from that frame on becomes `outgoing`; everything before
    it becomes `incoming` and should be discarded by the caller, per the
    Incoming -> Impact -> Outgoing model.

    Returns None — not an exception — when there's too little data (fewer
    than 2 velocity estimates, i.e. fewer than 3 detections) or no jump
    clears the threshold. "No clear impact found yet" is a normal,
    expected state (e.g. still watching a delivery in flight, or a ball
    that was never struck at all), not an error the caller should treat
    as a bug.
    """
    ordered = sorted(detections, key=lambda d: d.frame_index)
    velocities = _frame_velocities(ordered)
    if len(velocities) < 2:
        return None

    for (_, prev_vx, prev_vy), (frame, vx, vy) in zip(velocities, velocities[1:]):
        jump = ((vx - prev_vx) ** 2 + (vy - prev_vy) ** 2) ** 0.5
        if jump >= velocity_jump_threshold_px_per_frame:
            return ImpactSplit(
                incoming=[d for d in ordered if d.frame_index < frame],
                outgoing=[d for d in ordered if d.frame_index >= frame],
                impact_frame=frame,
            )
    return None


def trim_before_deceleration_spike(
    detections: List[BallDetection],
    accel_jump_threshold_px_per_frame2: float,
) -> List[BallDetection]:
    """
    Trims a (typically post-impact) detection sequence at the first sign
    of the ball hitting the net: net resistance decelerates a ball far
    more sharply than gravity or drag alone would, so a spike in the
    magnitude of acceleration (the second derivative of position, in
    px/frame^2) past `accel_jump_threshold_px_per_frame2` marks where net
    contact corrupted the data. Returns only the frames strictly before
    that spike, so a subsequent fit_ball_trajectory() call isn't
    contaminated by post-net-contact points.

    Returns the input (sorted by frame) unchanged if no spike is found —
    nothing to trim — including when there's too little data (fewer than
    3 detections) to estimate any acceleration at all.
    """
    ordered = sorted(detections, key=lambda d: d.frame_index)
    velocities = _frame_velocities(ordered)
    if len(velocities) < 2:
        return ordered

    for (prev_frame, prev_vx, prev_vy), (frame, vx, vy) in zip(velocities, velocities[1:]):
        dt_frames = frame - prev_frame
        if dt_frames <= 0:
            continue
        accel_mag = (((vx - prev_vx) / dt_frames) ** 2 + ((vy - prev_vy) / dt_frames) ** 2) ** 0.5
        if accel_mag >= accel_jump_threshold_px_per_frame2:
            return [d for d in ordered if d.frame_index < frame]
    return ordered

"""
End to end: a video file in, (on_time, footwork_correct) out per delivery
detected — the same shape scoring.py and neural_scorer.py already accept
from a human's checkboxes in app.py. estimate_outcomes_from_video() is
the function app.py's "upload a clip" mode calls.

The video does not need to be pre-trimmed to one delivery, and doesn't
need to contain only one — real footage tested against this pipeline ran
6 to 71 seconds long (chunks of a nets session, sometimes with more than
one ball in frame), so delivery_segmentation.py finds every delivery it
can inside whatever's uploaded before the feature math runs on each one.
See that module's docstring for how "a delivery" is detected and its
real limits.

estimate_outcome_from_video() (singular) is kept for callers that only
want the single clearest delivery in a clip.

Requires pose_estimation.download_model() to have been run once (see
that module's docstring for what it fetches and why it's a deliberate,
explicit step rather than automatic).
"""

from dataclasses import dataclass, field
from typing import List, Optional

import cv2

from body_vector_render import draw_body_vectors, to_jpeg
from camera_motion import CameraMotionTracker, build_path, stabilize, stabilize_pose
from post_shot import POST_SHOT_WINDOW_SECONDS, PostShotReport, analyse_post_shot_movement
from outcome_from_video import OutcomeEstimate, estimate_net_outcome
from cut_detection import effective_fps, find_cuts_from_changes, frame_changes, thumbnail, duplicate_fraction
from body_vectors import BodyVectorReport, analyse_body_vectors

from bowler_analysis import hip_center, BowlerActionEstimate, analyse_bowler_action
from delivery_segmentation import WINDOW_BEFORE_PEAK_SEC, find_delivery_windows
from footage_check import FootageReport, assess_footage
from outcome_bridge import VisionOutcomeEstimate, VisionOutcomeEstimator
from pose_estimation import PoseEstimator


def _read_frames(video_path: str):
    cap = cv2.VideoCapture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        return frames, fps
    finally:
        cap.release()


def _expected_frame_count(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()


def estimate_outcomes_from_video(
    video_path: str, estimator: PoseEstimator = None, max_deliveries: int = None
) -> List[VisionOutcomeEstimate]:
    """Returns one VisionOutcomeEstimate per delivery detected in the
    clip, in the order they occur (not ranked by how clear the swing
    was). Raises ValueError if no person is detected anywhere, or if no
    delivery is detected at all (see
    delivery_segmentation.find_delivery_windows)."""
    owns_estimator = estimator is None
    estimator = estimator or PoseEstimator()
    try:
        frames, fps = _read_frames(video_path)
        landmarks = estimator.extract_landmarks_from_frames(frames, fps)
        if not landmarks:
            raise ValueError(
                f"No person detected in any frame of {video_path} — check the clip actually "
                "shows a batter in frame, or that the model downloaded correctly."
            )
        windows = find_delivery_windows(landmarks, fps, max_deliveries=max_deliveries)
        vision_estimator = VisionOutcomeEstimator()
        return [vision_estimator.estimate(landmarks[start:end], fps=fps) for start, end in windows]
    finally:
        if owns_estimator:
            estimator.close()


def estimate_outcome_from_video(video_path: str, estimator: PoseEstimator = None) -> VisionOutcomeEstimate:
    """Single-delivery convenience wrapper: the single clearest delivery
    (strongest swing) in the clip — matches delivery_segmentation.
    find_delivery_window()'s selection, not necessarily the first one in
    time. Prefer estimate_outcomes_from_video() for clips that might
    contain more than one — real nets-session footage usually does."""
    return estimate_outcomes_from_video(video_path, estimator=estimator, max_deliveries=1)[0]


# The bowler runs in and releases BEFORE the batter's swing; the ball's
# flight (~0.4-0.7s) sits between the two. Look this far back from the
# batter's swing peak for the bowler's run-up and release, and this far
# past it in case the flight was short.
BOWLER_LOOKBACK_SECONDS = 3.0
BOWLER_LOOKAHEAD_SECONDS = 0.3
BOWLER_FRAME_STEP = 2   # analyse every 2nd frame - the bowler's motion is smooth, and it halves the cost
BODY_LOOKBACK_SECONDS = 0.6      # body-vector window around a delivery's swing peak
BODY_LOOKAHEAD_SECONDS = 0.8
OVERLAY_MAX_WIDTH = 640          # annotated peak-frame images are downscaled to this width
MAX_POSE_SIDE = 1280             # frames larger than this (longest side) are shrunk for the pose model only
ROI_TARGET_HEIGHT = 720          # a batter's region smaller than this is upscaled to it before pose (more pixels on a small person)
CUT_CLEARANCE_SECONDS = 0.25     # a swing peak this close to an edit cut is the cut, not a swing
CUT_EVENT_GAP_SECONDS = 0.5      # cuts closer together than this are one transition


@dataclass
class VideoAnalysis:
    estimates: List[VisionOutcomeEstimate]                 # the batter, one per detected delivery
    footage: FootageReport                                 # is this clip good enough to trust?
    bowler_actions: List[Optional[BowlerActionEstimate]]   # aligned with `estimates`; None = no bowler found
    body_vectors: List[Optional[BodyVectorReport]] = field(default_factory=list)   # aligned with `estimates`
    vector_images: List[Optional[bytes]] = field(default_factory=list)             # annotated peak frame (JPEG), aligned
    container_fps: float = 0.0
    effective_fps: float = 0.0                                                       # genuinely different frames per second
    cut_events: List[int] = field(default_factory=list)                              # frames where an edit cut/transition begins
    clip_notes: List[str] = field(default_factory=list)                              # facts about the file worth telling the user
    delivery_notes: List[List[str]] = field(default_factory=list)                    # per delivery, aligned
    camera_moved: bool = False                                                       # positions were corrected for camera motion
    body_windows: List[Optional[list]] = field(default_factory=list)                 # per delivery: camera-stabilised landmarks, index-aligned with body_vectors
    aspect: float = 1.0                                                              # frame width / height
    post_shot: List[Optional[PostShotReport]] = field(default_factory=list)          # aligned with `estimates`; see post_shot.py
    outcome_estimates: List[OutcomeEstimate] = field(default_factory=list)           # aligned with `estimates`; see outcome_from_video.py
    swing_frames: List[Optional[int]] = field(default_factory=list)                  # aligned with `estimates`; absolute frame index of the swing peak, or None


def _typical_hip(window_landmarks):
    """Median hip position of the person the batter analysis followed in this
    delivery's window — the batter, by construction."""
    if not window_landmarks:
        return None
    centers = [hip_center(frame) for frame in window_landmarks]
    xs, ys = sorted(c[0] for c in centers), sorted(c[1] for c in centers)
    return xs[len(xs) // 2], ys[len(ys) // 2]


def _swing_frame(window_start: int, frame_of: List[int], fps: float) -> int:
    """The real frame index of a delivery's swing peak. Delivery windows are
    counted in landmark space (frames with no person are absent there), so map
    back through `frame_of` — otherwise every dropout shifts the answer earlier."""
    return frame_of[min(window_start + int(WINDOW_BEFORE_PEAK_SEC * fps), len(frame_of) - 1)]


def _shrink(frame):
    """Downscale for the pose model only (landmarks are normalised, so results carry
    over); a 1080x1920 phone video is otherwise several times slower for no gain."""
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= MAX_POSE_SIDE:
        return frame
    scale = MAX_POSE_SIDE / longest
    return cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)


def crop_to_roi(frame, roi):
    """Cut out the region where the batter is (normalised x0, y0, x1, y1) and give the pose model more
    pixels on them: a small region is upscaled (cubic) to ROI_TARGET_HEIGHT. This does two jobs: it picks
    WHICH person is analysed (a single-person pose model otherwise locks onto whoever is most prominent - on
    broadcast footage from the bowler's end that is the bowler), and it helps the model find a small subject.
    It cannot add detail the camera did not record."""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = roi
    px0, py0 = max(0, int(x0 * w)), max(0, int(y0 * h))
    px1, py1 = min(w, max(px0 + 8, int(x1 * w))), min(h, max(py0 + 8, int(y1 * h)))
    crop = frame[py0:py1, px0:px1]
    ch, cw = crop.shape[:2]
    scale = ROI_TARGET_HEIGHT / ch if ch < ROI_TARGET_HEIGHT else 1.0
    if max(ch, cw) * scale > MAX_POSE_SIDE:
        scale = MAX_POSE_SIDE / max(ch, cw)
    if abs(scale - 1.0) > 1e-6:
        crop = cv2.resize(crop, (max(2, int(round(cw * scale))), max(2, int(round(ch * scale)))),
                          interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    return crop


def clamp_roi(roi):
    """Keep a normalised (x0, y0, x1, y1) region inside [0, 1] and non-degenerate. A moving region
    (tracking a camera pan) can drift with BOTH bounds past an edge, or invert, if the pan carries
    it well off-frame; each axis is clamped by its midpoint, not by flooring/ceilinging a presorted
    pair, so a region that has left the frame entirely collapses to a thin sliver at the near edge
    instead of an inverted, zero-area box (which crashes the resize that follows). This is applied
    identically before cropping AND before mapping landmarks back, so the two always agree on what
    region was actually used."""
    def axis(a, b):
        lo, hi = max(0.0, min(min(a, b), 1.0)), max(0.0, min(max(a, b), 1.0))
        if hi - lo < 0.02:
            mid = min(max((lo + hi) / 2.0, 0.01), 0.99)
            return mid - 0.01, mid + 0.01
        return lo, hi

    x0, x1 = axis(roi[0], roi[2])
    y0, y1 = axis(roi[1], roi[3])
    return (x0, y0, x1, y1)


def advance_roi(roi, running_dx, running_dy):
    """The region shifted by the camera's cumulative causal displacement so far, clamped to the
    frame. Pulled out as its own function so the pan-following logic is testable without driving
    the whole pipeline through pose estimation and delivery detection."""
    x0, y0, x1, y1 = roi
    return clamp_roi((x0 + running_dx, y0 + running_dy, x1 + running_dx, y1 + running_dy))


def map_roi_landmarks(lm, roi):
    """Landmarks normalised to the crop -> normalised to the whole frame."""
    if lm is None:
        return None
    x0, y0, x1, y1 = roi
    return [(x0 + x * (x1 - x0), y0 + y * (y1 - y0), v) for (x, y, v) in lm]


def _iter_wanted_frames(video_path: str, wanted):
    """Yield (index, frame) for just the wanted frame indices, decoding sequentially
    (exact, unlike seeking) and holding one frame at a time."""
    if not wanted:
        return
    last = max(wanted)
    cap = cv2.VideoCapture(video_path)
    try:
        i = 0
        while i <= last:
            ok, frame = cap.read()
            if not ok:
                break
            if i in wanted:
                yield i, frame
            i += 1
    finally:
        cap.release()


def _cluster(cuts: List[int], fps: float) -> List[int]:
    events: List[int] = []
    for c in cuts:
        if not events or c - events[-1] > CUT_EVENT_GAP_SECONDS * fps:
            events.append(c)
    return events


def analyse_video(
    video_path: str, estimator: PoseEstimator = None, analyse_bowler: bool = False,
    max_deliveries: int = None, roi=None,
) -> VideoAnalysis:
    """Everything estimate_outcomes_from_video() does (the batter's footwork and
    timing per delivery), plus a footage-quality verdict, body-movement vectors per
    delivery, and — only if asked, since it needs a second multi-person pose pass — the
    bowler's action.

    Memory: frames are decoded one at a time and only landmarks and a tiny thumbnail per
    frame are kept, so a long or high-resolution phone video no longer has to fit in RAM
    (the earlier version held every decoded frame: ~1.8 GB for a 26 s 720x1040 clip, and
    ~5 GB for a 30 s 1080x1920 one). The few frames needed later (overlays, the bowler
    pass) are re-read from the file.

    Edited clips: hard cuts are detected; a swing peak sitting on a cut is dropped (the
    person 'teleporting' reads as a huge fake swing) and measurements never span a cut. The
    file's real frame rate is also reported, because clips exported at '60 fps' are often 30
    unique fps with every frame doubled.

    `roi` (normalised x0, y0, x1, y1), if given, is where the batter is: the pose model sees only that region,
    upscaled (see crop_to_roi), and landmarks are mapped back to the whole frame. Use it whenever the most
    prominent person is not the batter, or the batter is small. Camera motion is still measured on the whole
    frame; a fixed region does not follow a panning camera."""
    owns_estimator = estimator is None
    estimator = estimator or PoseEstimator()
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        thumbs = []
        shifts = []
        rois_used = []
        tracker = CameraMotionTracker()

        def stream():
            running = [0.0, 0.0]                 # cumulative camera displacement seen so far (causal)
            while True:
                ok, frame = cap.read()
                if not ok:
                    return
                thumbs.append(thumbnail(frame))
                shift = tracker.push(frame)
                shifts.append(shift)
                if roi:
                    if shift is not None:
                        running[0] += shift[0]
                        running[1] += shift[1]
                    # A region tracking the CAMERA's own pan, not the batter's own movement: it follows
                    # a panning shot (found not to on a real broadcast clip - see cv-pipeline/README.md)
                    # but a batter who moves independently (steps out, runs) still drifts out of it.
                    moved = advance_roi(roi, running[0], running[1])
                    rois_used.append(moved)
                    yield crop_to_roi(frame, moved)
                else:
                    yield _shrink(frame)

        try:
            aligned = estimator.extract_aligned_landmarks_from_frames(stream(), fps)
        finally:
            cap.release()
        if roi:
            aligned = [map_roi_landmarks(lm, rois_used[i]) for i, lm in enumerate(aligned)]
        n_frames = len(aligned)
        frame_of = [i for i, lm in enumerate(aligned) if lm is not None]   # landmark index -> real frame index
        if not frame_of:
            raise ValueError(
                f"No person detected in any frame of {video_path} — check the clip actually "
                "shows a batter in frame, or that the model downloaded correctly."
            )
        changes = frame_changes(thumbs)
        del thumbs
        cut_events = _cluster(find_cuts_from_changes(changes), fps)
        # Remove the camera's own movement from every landmark, so a person standing still in the
        # world has constant coordinates (identical output for tripod footage).
        path = build_path(shifts, cut_events)
        raw_aligned = aligned
        aligned = stabilize(aligned, path)
        landmarks = [aligned[i] for i in frame_of]
        eff_fps = effective_fps(fps, changes)
        clip_notes: List[str] = []
        if roi:
            drift = max(abs(rois_used[-1][0] - roi[0]), abs(rois_used[-1][1] - roi[1])) if rois_used else 0.0
            clip_notes.append(
                f"The batter was located by a region you set ({roi[0]:.2f},{roi[1]:.2f})-({roi[2]:.2f},{roi[3]:.2f}), "
                "upscaled before pose estimation, and moved to track the camera's own panning as it was measured"
                + (f" (drifted {drift:.2f} by the end of the clip)" if drift > 0.02 else "")
                + ". It does NOT follow the batter's own movement (stepping out, running) independent of the camera."
            )
        if eff_fps < 0.8 * fps:
            clip_notes.append(
                f"The file says {fps:.0f} fps but only about {eff_fps:.0f} frames per second are genuinely different "
                f"({duplicate_fraction(changes):.0%} are repeats) — typical of an exported/converted video. Timing "
                "is still correct, but fast movements are sampled less finely."
            )
        clip_notes.extend(path.notes)
        if cut_events:
            clip_notes.append(
                f"{len(cut_events)} edit cut/transition point(s) at frame(s) {cut_events}. Deliveries are measured only "
                "within an uncut stretch, and a swing peak sitting on a cut is dropped."
            )

        footage = assess_footage(landmarks, n_frames, frames_expected=_expected_frame_count(video_path))
        raw_windows = find_delivery_windows(landmarks, fps, max_deliveries=max_deliveries)
        clearance = CUT_CLEARANCE_SECONDS * fps
        windows, dropped = [], 0
        for w in raw_windows:
            peak = _swing_frame(w[0], frame_of, fps)
            if any(abs(peak - c) <= clearance for c in cut_events):
                dropped += 1
            else:
                windows.append(w)
        if dropped:
            clip_notes.append(f"{dropped} apparent delivery(ies) were on an edit cut, not a swing, and were dropped.")

        vision_estimator = VisionOutcomeEstimator()
        estimates = [vision_estimator.estimate(landmarks[start:end], fps=fps) for start, end in windows]
        delivery_notes: List[List[str]] = [[] for _ in windows]

        bowler_actions: List[Optional[BowlerActionEstimate]] = [None] * len(windows)
        if analyse_bowler and windows:
            step, eff = BOWLER_FRAME_STEP, fps / BOWLER_FRAME_STEP
            ranges = []
            for start, _end in windows:
                sw = _swing_frame(start, frame_of, fps)
                ranges.append((max(0, sw - int(BOWLER_LOOKBACK_SECONDS * fps)), min(n_frames, sw + int(BOWLER_LOOKAHEAD_SECONDS * fps)), sw))
            wanted = {i for lo, hi, _ in ranges for i in range(lo, hi, step)}
            poses = {}
            with PoseEstimator(num_poses=4) as multi:
                for i, frame in _iter_wanted_frames(video_path, wanted):
                    found = multi.extract_all_poses_from_one_live_frame(_shrink(frame), eff)
                    poses[i] = [stabilize_pose(pz, *path.at(i)) for pz in found] if path.moving else found
            for k, ((start, end), (lo, hi, sw)) in enumerate(zip(windows, ranges)):
                bowler_actions[k] = analyse_bowler_action(
                    [poses.get(i, []) for i in range(lo, hi, step)], eff,
                    batter_swing_frame=(sw - lo) // step, batter_hip=_typical_hip(landmarks[start:end]),
                )

        body_vectors: List[Optional[BodyVectorReport]] = []
        body_windows: List[Optional[list]] = []
        peak_frames: List[Optional[int]] = []
        aspect = width / height if height else 1.0
        for k, (start, _end) in enumerate(windows):
            swing = _swing_frame(start, frame_of, fps)
            lo, hi = max(0, swing - int(BODY_LOOKBACK_SECONDS * fps)), min(n_frames, swing + int(BODY_LOOKAHEAD_SECONDS * fps))
            before = [c for c in cut_events if c <= swing]
            after = [c for c in cut_events if c > swing]
            if before and before[-1] > lo:
                lo = before[-1]
            if after and after[0] < hi:
                hi = after[0]
            if lo > 0 or hi < n_frames:
                pass
            report = analyse_body_vectors(aligned[lo:hi], fps, aspect)
            if report is None and (lo, hi) != (max(0, swing - int(BODY_LOOKBACK_SECONDS * fps)), min(n_frames, swing + int(BODY_LOOKAHEAD_SECONDS * fps))):
                delivery_notes[k].append("An edit cut leaves too little uncut footage around this swing to measure it.")
            elif report is not None and (lo > max(0, swing - int(BODY_LOOKBACK_SECONDS * fps)) or hi < min(n_frames, swing + int(BODY_LOOKAHEAD_SECONDS * fps))):
                delivery_notes[k].append("Measured only within the uncut part of the window (an edit cut is nearby).")
            body_vectors.append(report)
            body_windows.append(aligned[lo:hi] if report is not None else None)
            peak_frames.append(None if report is None or aligned[lo + report.peak_frame] is None else lo + report.peak_frame)

        # How far this delivery's post-shot lookahead may run before it would cross into an edit
        # cut or the next detected delivery — never conflate two balls' aftermaths.
        post_shot: List[Optional[PostShotReport]] = []
        for k, pf in enumerate(peak_frames):
            if pf is None:
                post_shot.append(None)
                continue
            candidates = [n_frames] + [c for c in cut_events if c > pf]
            if k + 1 < len(windows):
                candidates.append(_swing_frame(windows[k + 1][0], frame_of, fps))
            post_shot.append(analyse_post_shot_movement(aligned[:min(candidates)], fps, aspect, shot_frame=pf))

        outcome_estimates = [estimate_net_outcome(bv, ps) for bv, ps in zip(body_vectors, post_shot)]

        vector_images: List[Optional[bytes]] = [None] * len(windows)
        wanted_peaks = {pf for pf in peak_frames if pf is not None}
        for i, frame in _iter_wanted_frames(video_path, wanted_peaks):
            for k, pf in enumerate(peak_frames):
                if pf == i:
                    drawn = draw_body_vectors(frame, raw_aligned[i], body_vectors[k], offset=path.at(i) if path.moving else (0.0, 0.0))
                    if drawn.shape[1] > OVERLAY_MAX_WIDTH:
                        scale = OVERLAY_MAX_WIDTH / drawn.shape[1]
                        drawn = cv2.resize(drawn, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    vector_images[k] = to_jpeg(drawn)

        return VideoAnalysis(
            estimates=estimates, footage=footage, bowler_actions=bowler_actions,
            body_vectors=body_vectors, vector_images=vector_images,
            container_fps=fps, effective_fps=eff_fps, cut_events=cut_events,
            clip_notes=clip_notes, delivery_notes=delivery_notes, camera_moved=path.moving,
            body_windows=body_windows, aspect=aspect, post_shot=post_shot, outcome_estimates=outcome_estimates,
            swing_frames=peak_frames,
        )
    finally:
        if owns_estimator:
            estimator.close()

"""
Find hard edit cuts in a video: the frame where one shot ends and an unrelated
one begins.

Why this matters: pose analysis assumes one continuous camera view. Edited
clips (TikTok, highlight reels, anything cut in an app) violate that. At a cut
the person "teleports", so a joint appears to move across the frame in one
frame — which reads as an enormous, entirely fake swing — and smoothing or
tracking across the cut blends two different scenes. Nothing about the pose
output says a cut happened, so it has to be detected from the pictures.

Method: compare each frame's tiny greyscale thumbnail with the previous one.
Ordinary motion (a swing, a pan, camera shake) changes a thumbnail by a few
grey levels; a hard cut changes most of it at once. A cut is a jump that is
both large in absolute terms AND far above the recent typical change, so a
generally lively clip does not trigger it constantly.

Two real-footage lessons are built in (found on TikTok clips):
  * A fast whip-pan changes the picture a lot on every new frame, sustained. That
    is NOT a cut: a cut is an ISOLATED jump after a stretch of ordinary change.
  * Clips exported at "60 fps" are often 30 unique frames per second with every
    other frame a duplicate (change of ~0). Duplicates must not count as "quiet"
    when judging what is typical, and the true frame rate is reported by
    effective_fps() because every speed in torso-lengths/s depends on it.

Limits, stated: only HARD cuts are found. A slow fade or dissolve changes the
picture gradually and is not detected; neither is a cut between two shots that
happen to look nearly identical (rare, and harmless for this purpose). The
thresholds are judgment calls checked on synthetic clips and on the clips
available, not tuned on a labelled set of real edits.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

THUMB_SIZE = (48, 48)         # (w, h) of the comparison thumbnail
MIN_JUMP = 25.0               # mean absolute grey-level change (0-255) a cut must exceed
JUMP_OVER_TYPICAL = 5.0       # ...and this many times the recent median change
HISTORY = 15                  # how many preceding frames define "typical change"
TYPICAL_FLOOR = 2.0           # typical change is never treated as less than this (a static tripod shot)
DUPLICATE_LEVEL = 0.35        # a frame changing by less than this is a repeat of the one before (compression noise ~0.1-0.3)
NEW_FRAME_LEVEL = 1.0         # a frame that genuinely differs from its neighbour changes at least this much
MIN_TYPICAL_SAMPLES = 3       # need this many real (non-duplicate) recent changes to judge what is typical
CALM_AFTER = 0.4              # after a real cut the typical change falls below this share of the jump
MIN_GAP_FRAMES = 3            # two "cuts" closer than this are one event


def thumbnail(frame_bgr: np.ndarray) -> np.ndarray:
    """A small float greyscale copy of a frame, for cheap comparison."""
    import cv2
    grey = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(grey, THUMB_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)


def frame_changes(thumbs: Sequence[np.ndarray]) -> List[float]:
    """changes[i] = how different frame i is from frame i-1 (changes[0] = 0)."""
    out = [0.0]
    for a, b in zip(thumbs, thumbs[1:]):
        out.append(float(np.mean(np.abs(b - a))))
    return out


def find_cuts_from_changes(changes: Sequence[float]) -> List[int]:
    """Frame indices at which a new shot begins."""
    cuts: List[int] = []
    for i, jump in enumerate(changes):
        if i == 0 or jump < MIN_JUMP:
            continue
        recent = [c for c in changes[max(0, i - HISTORY):i] if c > DUPLICATE_LEVEL]   # duplicates are not "quiet"
        typical = max(float(np.median(recent)) if len(recent) >= MIN_TYPICAL_SAMPLES else 0.0, TYPICAL_FLOOR)
        if jump < JUMP_OVER_TYPICAL * typical or (cuts and i - cuts[-1] < MIN_GAP_FRAMES):
            continue
        # A cut lands in a new, calm shot; a whip-pan keeps changing the picture after the "jump".
        after = [c for c in changes[i + 1:i + 1 + HISTORY] if c > DUPLICATE_LEVEL]
        if len(after) >= MIN_TYPICAL_SAMPLES and float(np.median(after)) > CALM_AFTER * jump:
            continue
        cuts.append(i)
    return cuts


def find_cuts(thumbs: Sequence[np.ndarray]) -> List[int]:
    return find_cuts_from_changes(frame_changes(thumbs))


def duplicate_fraction(changes: Sequence[float]) -> float:
    """Share of frames that are exact-looking repeats of the frame before AND sit next to a
    frame that genuinely changed. The second condition matters: a still scene with a little
    movement changes by tiny amounts every frame, but that is a quiet picture, not repeated
    frames. Doubled-frame video alternates: repeat, new, repeat, new."""
    body = list(changes)
    dups = 0
    for i in range(1, len(body)):
        if body[i] >= DUPLICATE_LEVEL:
            continue
        neighbours = [body[j] for j in (i - 1, i + 1) if 1 <= j < len(body)]
        if any(c >= NEW_FRAME_LEVEL for c in neighbours):
            dups += 1
    return dups / (len(body) - 1) if len(body) > 1 else 0.0


def effective_fps(container_fps: float, changes: Sequence[float]) -> float:
    """The number of genuinely different frames per second. A container that
    says 60 fps but repeats each frame twice is really 30; a static shot with
    few repeats-by-chance is left alone by requiring the repeats to be regular
    (at least a third of frames) before adjusting."""
    frac = duplicate_fraction(changes)
    if frac < 0.33:
        return container_fps
    return container_fps * (1.0 - frac)

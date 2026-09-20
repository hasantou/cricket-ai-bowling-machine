"""
Tie a hand-annotated screenshot back to the exact frame of the clip it came
from, and keep those annotations as labelled examples.

Why this exists: video shot recognition needs labelled examples — a clip, the
moment of the shot, and what the shot was — before anything can be trained
or even honestly scored. People naturally produce labels as annotated
screenshots (a paused frame with an arrow drawn on it). This module finds
which frame such a screenshot is, and `labels/shot_labels.json` records what
was said about it. `evaluate()` is the piece that will report real accuracy
once there are enough labelled shots; with a handful it only shows
agreement/disagreement, and says so.

Matching is plain image similarity on a downscaled greyscale copy (the
screenshot is the video frame plus black bars, so the video band is cropped
out first). It returns the score alongside the frame, because a clean match
is a low number and a wrong one is not — callers should look at it.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

LABELS_PATH = os.path.join(os.path.dirname(__file__), "labels", "shot_labels.json")
_COMPARE_W = 72
DARK_ROW_LEVEL = 25          # mean brightness below this = a black bar, not picture


def crop_video_band(screenshot_bgr: np.ndarray) -> np.ndarray:
    """A phone screenshot of a paused video has black bars above and below the
    picture; keep only the tallest run of non-black rows."""
    brightness = screenshot_bgr.mean(axis=(1, 2))
    rows = np.where(brightness > DARK_ROW_LEVEL)[0]
    if rows.size == 0:
        return screenshot_bgr
    runs = np.split(rows, np.where(np.diff(rows) > 3)[0] + 1)
    longest = max(runs, key=len)
    return screenshot_bgr[longest[0]: longest[-1] + 1]


def _small_grey(img_bgr: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return cv2.cvtColor(cv2.resize(img_bgr, size), cv2.COLOR_BGR2GRAY).astype(np.float32)


def locate_frame_in_video(screenshot_path: str, video_path: str, top_k: int = 3) -> List[Tuple[int, float]]:
    """Returns the `top_k` best (frame_index, mean_abs_difference) matches,
    best first. Lower difference = closer; neighbouring frames legitimately
    score nearly as well, so the top hits are usually a run of adjacent
    frames around the true one."""
    shot = cv2.imread(screenshot_path)
    if shot is None:
        raise FileNotFoundError(f"could not read image {screenshot_path}")
    band = crop_video_band(shot)
    h, w = band.shape[:2]
    size = (_COMPARE_W, max(1, round(_COMPARE_W * h / w)))
    target = _small_grey(band, size)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video {video_path}")
    scored: List[Tuple[float, int]] = []
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            scored.append((float(np.mean(np.abs(_small_grey(frame, size) - target))), index))
            index += 1
    finally:
        cap.release()
    scored.sort()
    return [(i, d) for d, i in scored[:top_k]]


def load_labels(path: str = LABELS_PATH) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)["labels"]


def evaluate(predictions: Dict[str, str], labels: Optional[List[Dict]] = None) -> Dict:
    """`predictions` maps a label id to the shot name a pipeline predicted.
    Compares only labels that actually carry a shot name (`shot` not null);
    returns counts and the disagreements. Accuracy is reported only as a
    fraction of what was compared — with a handful of labels it is an
    anecdote, and the result says how many it rests on."""
    labels = load_labels() if labels is None else labels
    compared, correct, misses = 0, 0, []
    for label in labels:
        truth = label.get("shot")
        if not truth or label["id"] not in predictions:
            continue
        compared += 1
        predicted = predictions[label["id"]]
        if predicted.strip().lower() == truth.strip().lower():
            correct += 1
        else:
            misses.append({"id": label["id"], "truth": truth, "predicted": predicted})
    return {
        "labelled_and_predicted": compared,
        "correct": correct,
        "accuracy": (correct / compared) if compared else None,
        "disagreements": misses,
        "note": f"Rests on only {compared} labelled shot(s)." if compared < 30 else "",
    }

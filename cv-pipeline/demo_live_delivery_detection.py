"""
Validates the live-shaped pipeline (live_video_source.py +
live_delivery_detector.py) against real footage, not just synthetic
data — by playing a real, already-tested WhatsApp clip back frame by
frame AS IF it were a live camera feed, and confirming it detects the
same deliveries the existing batch pipeline (video_pipeline.py) already
found on that same clip. A file and a live camera look identical to
cv2.VideoCapture.read() in a loop, which is exactly what makes this a
fair, honest test of "will the live path actually work," not just "does
this new code run."

Run with: python3 demo_live_delivery_detection.py path/to/clip.mp4
"""

import sys
import time

from delivery_segmentation import (
    MIN_DELIVERY_SEPARATION_SEC, WINDOW_AFTER_PEAK_SEC, WINDOW_BEFORE_PEAK_SEC, find_delivery_windows,
)
from live_video_source import LiveVideoSource
from live_delivery_detector import LiveDeliveryDetector
from pose_estimation import PoseEstimator
from outcome_bridge import VisionOutcomeEstimator


def run_live(video_path: str, estimator: PoseEstimator):
    source = LiveVideoSource(device=video_path)
    fps = source.fps
    detector = LiveDeliveryDetector(fps=fps, buffer_seconds=15.0)
    vision_estimator = VisionOutcomeEstimator()

    live_windows = []  # absolute (start, end) - stable across the whole session, safe to compare
    live_results = []
    frame_count = 0
    t0 = time.perf_counter()
    for frame in source.frames():
        frame_count += 1
        # Exactly one frame at a time - the realistic live shape.
        landmarks = estimator.extract_landmarks_from_one_live_frame(frame, fps)

        for delivery in detector.add_frame(landmarks):
            # .start/.end slice the CURRENT buffer right now; .absolute_*
            # is what's safe to log/compare later - see DetectedDelivery's
            # own docstring for why conflating the two is a real mistake
            # (this script made it once, on its first version).
            window_landmarks = detector.buffered_landmarks()[delivery.start:delivery.end]
            estimate = vision_estimator.estimate(window_landmarks, fps=fps)
            live_windows.append((delivery.absolute_start, delivery.absolute_end))
            live_results.append(estimate)
            print(f"  [live, frame {frame_count}] delivery detected — "
                  f"on_time={estimate.on_time}, footwork_correct={estimate.footwork_correct}")
    elapsed = time.perf_counter() - t0
    source.close()
    return live_windows, live_results, frame_count, elapsed


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 demo_live_delivery_detection.py path/to/clip.mp4")
        sys.exit(1)
    video_path = sys.argv[1]

    print(f"=== Playing {video_path} back as a live feed ===")
    with PoseEstimator() as estimator:
        live_windows, live_results, frame_count, elapsed = run_live(video_path, estimator)

    print(f"\nProcessed {frame_count} frames in {elapsed:.1f}s "
          f"({frame_count / elapsed:.1f} fps sustained, live pipeline).")
    print(f"Live pipeline found {len(live_results)} deliveries.\n")

    print("=== Cross-checking against the existing batch pipeline ===")
    with PoseEstimator() as estimator2:
        source2 = LiveVideoSource(device=video_path)
        fps = source2.fps
        all_frames = list(source2.frames())
        source2.close()
        batch_landmarks = estimator2.extract_landmarks_from_frames(all_frames, fps)

    after_frames = int(WINDOW_AFTER_PEAK_SEC * fps)
    full_window_frames = int(WINDOW_BEFORE_PEAK_SEC * fps) + after_frames + 1
    separation_frames = int(MIN_DELIVERY_SEPARATION_SEC * fps)
    batch_windows = find_delivery_windows(batch_landmarks, fps)
    total_frames = len(batch_landmarks)

    # A window is only fair to expect from the live pipeline if the
    # RECORDING itself contains enough frames past its peak for the same
    # safety margin LiveDeliveryDetector requires (MIN_DELIVERY_SEPARATION_SEC,
    # not just WINDOW_AFTER_PEAK_SEC - see that module's docstring for why
    # the extra margin exists). A window that can't clear that bar isn't a
    # live-detection failure - it means the recording stopped too soon
    # after that swing, which can only happen to a finite file, never to
    # a genuinely continuous live camera.
    complete_batch_windows = []
    clamped_at_end = []
    for s, e in batch_windows:
        peak_idx = e - after_frames - 1
        if (e - s) == full_window_frames and total_frames - peak_idx >= separation_frames:
            complete_batch_windows.append((s, e))
        else:
            clamped_at_end.append((s, e))

    print(f"Batch pipeline found {len(batch_windows)} window(s) total: "
          f"{len(complete_batch_windows)} complete, {len(clamped_at_end)} clamped by the "
          f"recording ending mid-follow-through (expected on a finite file, not a live bug).\n")

    if live_windows != complete_batch_windows:
        print(f"MISMATCH on the windows that had a fair chance to complete: "
              f"live={live_windows} vs batch(complete only)={complete_batch_windows}")
        sys.exit(1)

    vision_estimator = VisionOutcomeEstimator()
    for i, (start, end) in enumerate(complete_batch_windows):
        batch_estimate = vision_estimator.estimate(batch_landmarks[start:end], fps=fps)
        live_estimate = live_results[i]
        agree = (live_estimate.on_time == batch_estimate.on_time
                 and live_estimate.footwork_correct == batch_estimate.footwork_correct)
        tag = "OK" if agree else "MISMATCH"
        print(f"  delivery {i + 1}: live(on_time={live_estimate.on_time}, "
              f"footwork={live_estimate.footwork_correct}) vs "
              f"batch(on_time={batch_estimate.on_time}, footwork={batch_estimate.footwork_correct}) — {tag}")
        if not agree:
            sys.exit(1)

    print("\nLive pipeline exactly matches the batch pipeline on every window "
          "that a genuinely continuous camera would also have completed.")


if __name__ == "__main__":
    main()

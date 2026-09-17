"""
Demonstrates ball_tracking.py against a synthetic flight sampled at the
real-world ~12% recall rate measured on actual nets footage from an
earlier (uncommitted) YOLOv8 experiment — not real detections, since no
detector is checked into this repo (see ball_tracking.py's docstring for
why). This shows the fitting logic is real and does something useful;
it does not show a real detector's output being recovered.

Run with: python3 demo_ball_tracking.py
"""
import random

from ball_tracking import BallDetection, BallTrajectoryFitError, fit_ball_trajectory

# A synthetic flight in pixel space - constant horizontal image velocity,
# a gravity-style parabola vertically. Not real physics (that's
# trajectory-engine's job); just a curve to demonstrate recovery against.
TRUE_X = (12.0, 40.0)
TRUE_Y = (-0.35, 9.0, 380.0)
N_FRAMES = 40
RECALL = 0.12  # the measured real-footage number


def true_position(frame: int):
    x = TRUE_X[0] * frame + TRUE_X[1]
    y = TRUE_Y[0] * frame ** 2 + TRUE_Y[1] * frame + TRUE_Y[2]
    return x, y


def main():
    rng = random.Random(42)
    ground_truth = [true_position(f) for f in range(N_FRAMES)]

    detections = [
        BallDetection(f, x + rng.uniform(-3, 3), y + rng.uniform(-3, 3))
        for f, (x, y) in enumerate(ground_truth)
        if rng.random() < RECALL
    ]
    print(f"Simulated a {N_FRAMES}-frame delivery at {RECALL:.0%} recall: "
          f"{len(detections)} of {N_FRAMES} frames actually detected.")
    print(f"Detected frames: {[d.frame_index for d in detections]}\n")

    try:
        trajectory = fit_ball_trajectory(detections, fps=30.0, inlier_threshold_px=10.0, rng=rng)
    except BallTrajectoryFitError as e:
        print(f"Refused to fit: {e}")
        return

    print(f"Fit accepted {len(trajectory.inlier_frames)} of {len(detections)} "
          f"detections as inliers; rejected {len(trajectory.outlier_frames)}.\n")

    print(f"{'frame':>5}  {'true (x, y)':>18}  {'recovered (x, y)':>20}  {'error (px)':>10}  detected?")
    for f in range(0, N_FRAMES, 4):
        tx, ty = true_position(f)
        px, py = trajectory.position_at(f)
        error = ((px - tx) ** 2 + (py - ty) ** 2) ** 0.5
        detected = "yes" if f in {d.frame_index for d in detections} else "no"
        print(f"{f:>5}  ({tx:6.1f}, {ty:6.1f})  ({px:6.1f}, {py:6.1f})  {error:10.1f}  {detected}")


if __name__ == "__main__":
    main()

"""
Bridges cv-pipeline features to the (on_time, footwork_correct) booleans
adaptation-engine's scorers already consume — see
adaptation-engine/scoring.py and neural_scorer.py's record_delivery().

Honesty check, same as neural_scorer.py: there is no labelled real
footage yet (a delivery clip paired with a coach's real on_time/footwork
verdict), so this cannot be a trained classifier today. It is a
documented, adjustable heuristic over feature_extraction.py's output
instead of a black box — every threshold below is named and easy to
tune or replace, not a hidden magic number.

When labelled clips exist (coach-reviewed footage with a footwork/timing
verdict attached), replace heuristic_bridge() with a trained classifier
over the same DeliveryFeatures fields — VisionOutcomeEstimator.estimate()
does not need to change, matching the pattern already used for
neural_scorer.py's simulated-to-real migration path (same feature
schema in, same interface out, only the decision function changes).
"""

from dataclasses import dataclass
from typing import List, Tuple

from feature_extraction import DeliveryFeatures, FrameLandmarks, extract_delivery_features

# Named, adjustable thresholds — not hidden magic numbers.
MIN_FOOTWORK_DISPLACEMENT = 0.04     # normalised units; below this reads as "didn't move the feet"
MAX_BALANCED_KNEE_BEND_DEG = 165     # a fully locked-straight knee (~180) reads as poor balance
LATE_SWING_START_FRAME_RATIO = 0.75  # swing starting in the last quarter of the clip reads as late


@dataclass
class VisionOutcomeEstimate:
    on_time: bool
    footwork_correct: bool
    features: DeliveryFeatures


def heuristic_bridge(features: DeliveryFeatures) -> Tuple[bool, bool]:
    footwork_correct = (
        features.front_foot_displacement >= MIN_FOOTWORK_DISPLACEMENT
        and features.front_knee_bend_deg <= MAX_BALANCED_KNEE_BEND_DEG
    )
    late_threshold_frame = features.n_frames * LATE_SWING_START_FRAME_RATIO
    on_time = features.swing_start_frame <= late_threshold_frame
    return on_time, footwork_correct


class VisionOutcomeEstimator:
    """What a future 'auto-log this delivery from video' mode in app.py
    would call once per clip, instead of a human ticking checkboxes.
    Does not judge shot outcome (middled/edged/missed/...) — that needs
    ball-tracking against the bat, which isn't built; this covers timing
    and footwork only, the two signals scoring.py already consumes
    alongside a human-entered outcome."""

    def estimate(self, frames: List[FrameLandmarks]) -> VisionOutcomeEstimate:
        features = extract_delivery_features(frames)
        on_time, footwork_correct = heuristic_bridge(features)
        return VisionOutcomeEstimate(on_time=on_time, footwork_correct=footwork_correct, features=features)

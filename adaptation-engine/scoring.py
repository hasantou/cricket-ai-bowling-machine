"""
Mastery scoring — the rolling, rule-based "has this batter solved this
style yet?" score described in the program plan (Section 4.2) and the MVP
plan (Section 2.1). Deliberately simple and explainable for Phase 1: an
exponential moving average per style, with a minimum-sample gate so a
lucky single ball can't trigger a switch.
"""

from dataclasses import dataclass, field
from typing import Dict, List

# Outcome -> quality score. Tune against coach judgement during pilot
# sessions (program plan, Section 11).
OUTCOME_QUALITY = {
    "middled": 1.0,
    "attacked": 1.0,
    "defended": 0.6,
    "edged": 0.3,
    "missed": 0.0,
}

# How much a new delivery's outcome moves the running score. Higher =
# more reactive to the last few balls; lower = more conservative.
EMA_ALPHA = 0.35

# A style counts as "mastered" once its score is at or above this line...
MASTERY_THRESHOLD = 0.75
# ...and at least this many deliveries of that style have been logged.
MIN_SAMPLE = 4


@dataclass
class StyleRecord:
    style_key: str
    score: float = 0.0
    deliveries_seen: int = 0
    history: List[str] = field(default_factory=list)  # outcome log, most recent last

    @property
    def mastered(self) -> bool:
        return self.deliveries_seen >= MIN_SAMPLE and self.score >= MASTERY_THRESHOLD


class MasteryScorer:
    """Tracks a live mastery score per bowling style for one batter."""

    def __init__(self):
        self._records: Dict[str, StyleRecord] = {}

    def record_delivery(self, style_key: str, outcome: str,
                         on_time: bool = True, footwork_correct: bool = True) -> StyleRecord:
        """Log one delivery's outcome and update that style's rolling score.

        `outcome` must be one of OUTCOME_QUALITY's keys. `on_time` and
        `footwork_correct` are small modifiers representing what a CV
        pipeline would eventually supply automatically (see
        cv-pipeline/README.md) — for the MVP these can be set by a human
        operator reviewing the footage, or left at their defaults.
        """
        if outcome not in OUTCOME_QUALITY:
            raise ValueError(f"Unknown outcome '{outcome}'. Expected one of {list(OUTCOME_QUALITY)}")

        record = self._records.setdefault(style_key, StyleRecord(style_key=style_key))

        quality = OUTCOME_QUALITY[outcome]
        if not on_time:
            quality *= 0.7
        if not footwork_correct:
            quality *= 0.85

        if record.deliveries_seen == 0:
            record.score = quality
        else:
            record.score = EMA_ALPHA * quality + (1 - EMA_ALPHA) * record.score

        record.deliveries_seen += 1
        record.history.append(outcome)
        return record

    def get(self, style_key: str) -> StyleRecord:
        return self._records.get(style_key, StyleRecord(style_key=style_key))

    def all_records(self) -> Dict[str, StyleRecord]:
        return dict(self._records)

    def mastered_styles(self) -> List[str]:
        return [k for k, r in self._records.items() if r.mastered]

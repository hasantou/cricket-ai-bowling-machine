# adaptation-engine/

The decision logic: turns the CV pipeline's readings into a per-style
"mastery score" and, when a style is mastered, recommends the next one.

## Scope (Phase 1 — MVP)

- Rolling, rule-based mastery score per bowling style (pace band,
  line/length zone, swing/spin type).
- Threshold-based trigger: when a style's score crosses the "mastered"
  line over a minimum sample of deliveries, recommend a new style.
- "Maximum contrast" selection: recommend the style least like the one
  just mastered — not a random alternative.
- Output is a **recommendation shown to a human operator**, not a direct
  machine control signal, in this phase.

## Later phases (not yet in scope)

- Replace rule-based thresholds with a learned policy (reinforcement
  learning / multi-armed bandit) once real session data exists — see
  `docs/product/AI_Adaptive_Bowling_Machine_Program_Plan.docx`, Section 4.3.
- Direct, low-latency machine control (Phase 2+).

## Suggested structure

```
adaptation-engine/
  scoring/        mastery-score calculation from CV pipeline outputs
  recommendation/ style-selection rules ("maximum contrast" logic)
  tests/
```

No code yet — this is scaffolding for the first commits.

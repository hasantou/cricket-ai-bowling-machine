# cv-pipeline/

Computer-vision pipeline: pose estimation, ball tracking, and shot-outcome
classification from camera footage.

## Scope (Phase 1 — MVP)

- Pose estimation on the batter's stance, backlift, and footwork.
- Ball tracking: release point, trajectory, pitch location, deviation.
- Bat-ball contact analysis and shot-outcome classification (middled,
  edged, missed, defended, attacked).
- Runs offline or with a short processing delay on a laptop/cloud GPU —
  **not** on embedded hardware yet (see `docs/product/MVP_RD_Plan_Software_First.docx`).

## Suggested structure (to be filled in as work starts)

```
cv-pipeline/
  pose/            pose-estimation model + fine-tuning for batting stance
  ball_tracking/   trajectory and pitch-location detection
  shot_outcome/    contact-quality and outcome classification
  tests/
    fixtures/      small sample clips/labels for unit tests (keep tiny)
```

No code yet — this is scaffolding for the first commits.

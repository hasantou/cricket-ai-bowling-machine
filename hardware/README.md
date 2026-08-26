# hardware/

Mechanical delivery-head design, sensor integration, and embedded/edge
compute work.

## Status: not started (Phase 2+)

Deliberately out of scope for the current MVP (Phase 1), which validates
the adaptation logic using an existing off-the-shelf programmable machine
operated by a human — see `docs/product/MVP_RD_Plan_Software_First.docx`.

This folder is scaffolded now so the transition into Phase 2 (on-device
inference, low-latency automated switching) has a home to land in without
restructuring the repo.

## Anticipated scope (Phase 2)

- Delivery-head actuation: motorised adjustment of pace, line, length,
  swing, and spin between deliveries.
- Sensor suite integration: cameras, optional speed radar, pitch-side
  calibration.
- Edge compute unit: porting the validated `cv-pipeline/` and
  `adaptation-engine/` logic to run on-device with a real latency budget.

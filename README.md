# AI-Adaptive Bowling Machine

A cricket training system that senses how a batter is playing each bowling
style in real time, and switches pace, line, length, swing, or spin the
moment the player has mastered the current pattern — instead of varying on
a timer or at random, the way existing programmable machines do.

This repository is the private home for the whole project: planning docs,
research, and (as they're built) the computer-vision pipeline, the
adaptation engine, and the hardware/mechanical work.

## Repo layout

```
docs/
  product/    Program plan and the MVP research & development plan
  business/   Business plan (UK Innovator Founder visa version)
  research/   Source notes on the competitive landscape and market
app/               The working MVP demo (Streamlit) — run this
adaptation-engine/ Mastery scoring + style-recommendation logic (real, tested code)
cv-pipeline/       Pose estimation, ball tracking, shot-outcome classification (not built yet)
hardware/          Delivery-head and sensor integration work (later phase)
data/              Session recordings and datasets (gitignored — not committed)
```

## Run the MVP demo

```
pip install -r requirements.txt
streamlit run app/app.py
```

Log a few deliveries against the same bowling style with good outcomes and
watch it get flagged "mastered," then see the recommended next style
appear — the actual adaptation logic, running. See `app/README.md` for
what's real vs. still a placeholder in this build.

## Start here

1. [`docs/product/AI_Adaptive_Bowling_Machine_Program_Plan.docx`](docs/product/AI_Adaptive_Bowling_Machine_Program_Plan.docx) —
   the full concept: the problem, the AI/ML architecture, hardware plan,
   phased roadmap, risks, and success metrics.
2. [`docs/product/MVP_RD_Plan_Software_First.docx`](docs/product/MVP_RD_Plan_Software_First.docx) —
   the current, active plan: a ~10-week, human-in-the-loop MVP that tests
   whether the adaptation logic works at all, before any embedded hardware
   or low-latency actuation work begins.
3. [`docs/business/UK_Innovator_Founder_Business_Plan.docx`](docs/business/UK_Innovator_Founder_Business_Plan.docx) —
   the same concept restructured around the UK Innovator Founder visa
   endorsement criteria (innovation, viability, scalability). Contains
   placeholders that still need founder-specific detail — not submission-ready.
4. [`ROADMAP.md`](ROADMAP.md) — the phased plan (0–5) at a glance.

## Current phase

**MVP (software-first).** On-device inference and low-latency automated
switching are deliberately deferred — see the MVP plan for why. The active
build is a human-in-the-loop prototype: a camera + laptop/cloud pipeline
recommends the next delivery style, a coach manually sets it on an existing
programmable machine.

## Status

MVP demo working: the mastery-scoring and style-recommendation logic in
`adaptation-engine/` is real, tested code (`pytest adaptation-engine/tests/`
— 11 passing), wired into a runnable Streamlit app in `app/`. Computer
vision (`cv-pipeline/`) and any machine hardware integration are not built
yet — the demo uses human-entered delivery outcomes in their place, which
is the deliberate Phase 1 scope (see the MVP plan).

## License / confidentiality

Private repository. All rights reserved — not licensed for external use or
distribution. Update this section once the team has settled on IP terms
(patents, contributor agreements) for anyone joining the project.

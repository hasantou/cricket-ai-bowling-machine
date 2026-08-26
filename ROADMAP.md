# Roadmap

Phased so the riskiest, cheapest-to-test question — does the adaptation
logic actually work? — gets answered before any bespoke hardware spend.

| Phase | Focus | Key activity | Exit criteria |
|---|---|---|---|
| 0 | Feasibility | Confirm mechanical base (license/partner vs. build); define the style taxonomy with a coaching advisor. | Go/no-go decision with a chosen mechanical platform. |
| **1 (current)** | **MVP — proof of concept** | Human-in-the-loop: camera + laptop/cloud CV pipeline recommends a style; a coach manually sets it on an existing machine. See `docs/product/MVP_RD_Plan_Software_First.docx`. | AI's "switch now" call agrees with an independent coach's judgement on a meaningful majority of deliveries. |
| 2 | MVP hardware prototype | Integrate sensors + edge compute on one machine; real-time mastery scoring. | A single working prototype used in supervised sessions. |
| 3 | Pilot | Deploy 2–4 units at partner academies; collect data; gather feedback. | Validated detection accuracy against coach judgement. |
| 4 | Intelligent engine | Replace rule-based triggers with a learned (RL/bandit) policy; expand style library. | Learned policy outperforms the rule-based baseline. |
| 5 | Launch & scale | Ruggedised hardware, coach dashboard, cloud profiles, first hires, international partnerships. | First commercial units shipped and supported. |

Full detail for each phase lives in `docs/product/`.

## Deliberately deferred (Phase 1)

- **On-device / edge inference** — the CV and scoring pipeline runs on a
  laptop or cloud GPU for now, not embedded on the machine.
- **Low-latency automated switching** — a human operator executes the
  recommendation; no actuator engineering or ball-to-ball timing budget yet.
- **Reinforcement-learning policy** — rule-based thresholds only, until
  there's real session data to learn from.

Both return as the focus of Phase 2, once the MVP has validated the core
hypothesis.

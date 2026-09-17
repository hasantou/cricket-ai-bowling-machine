# Competitive landscape — source notes

Working notes on existing bowling-machine products and adjacent training
tech, kept here so claims in the planning docs are traceable back to a
source. Last researched 2026-09 via web search — re-verify before relying
on pricing or feature claims in a live pitch, since this market moves
fast (see the TrueMan3 deployment below, which happened in July 2026).

## Direct competitors: cricket bowling machines

### BOLA (incl. TrueMan3)

BOLA (Bristol, UK) is the dominant incumbent, from entry-level machines up
to its flagship:

- **Professional**: line/length via ball joint + vernier adjusters, pace
  in 1mph increments, swing via push-buttons, spin via angling the
  delivery head. **Electronic random mode** varies speed/swing per ball
  automatically via the feeder — scheduled/random, not performance-aware.
  **£2,532 (£2,110 ex VAT)** as of 2026.
- **TrueMan3** (the machine Pakistan's PCB deployed at its National
  Cricket Academy in July 2026, reported as "AI-powered"): a 3-wheel head
  (adds controllable backspin over the 2-wheel design), and — the actual
  headline feature — a full-colour LED screen playing real video of a
  bowler's run-up, synced precisely to ball release, so batters can read
  a real release point instead of a mechanical arm. App-programmable
  (line/length/speed/swing), and individual balls/overs/spells can be
  saved and shuffled for "random match-like scenarios." **This is
  realism-of-presentation and coach-programmed/shuffled variation — not a
  closed loop that senses the batter's actual success rate and adapts.**
  No published price found; sold on request.
- Sources: [bola.co.uk/cricket.html](https://www.bola.co.uk/cricket.html),
  [BOLA TrueMan3](https://www.bola.co.uk/shop/trueman3-bowling-machine),
  [2026 BOLA price list](https://www.bola.co.uk/productlist.html),
  [PCB TrueMan3 deployment, July 2026](https://propakistani.pk/2026/07/06/pcb-gets-state-of-the-art-bowling-machine/)

### ProBatter PX3

- Projects a real bowler's run-up and action for realistic visual timing.
- Large programmable library of bowler styles and speeds, set by a coach
  in advance — same category as TrueMan3 (pre-programmed, not adaptive).
- Source: [probatter.com/projects/px3-cricket-simulator](https://probatter.com/projects/px3-cricket-simulator/)

## Adjacent: cricket AI/CV coaching apps (crowded — not this project's differentiator)

Pure video-analysis apps already do what `cv-pipeline/` does for footwork
and timing — this space is active and funded. None of them drive a
physical machine; all put a human in the loop to act on the feedback
later, which is the key structural difference from what this project is
building.

- **CricVision** — footwork, timing, body position, shot selection
  feedback from video. [cricvision.ai](https://cricvision.ai/)
- **Advanced Impactor** — AI video + biomechanics, batting/bowling
  analytics. [tezeract.ai writeup](https://tezeract.ai/ai-powered-cricket-coaching-platform-improves-training/)
- **Fulltrack AI** — has genuinely solved single-smartphone ball tracking
  (3D ball tracks, speed, spin, pitchmaps), and it's peer-reviewed, not
  just marketing: validated against 3D motion capture and a radar gun.
  Calibrates using the cricket stumps as fixed reference keypoints. Known
  limits: ~0.47m standard error vs. motion capture, and it overestimates
  speed vs. radar. Directly relevant to this project's own weak ball
  tracking — the stump-calibration approach is worth adopting regardless
  of competitive concerns. Sources:
  [validation vs. 3D mocap](https://www.tandfonline.com/doi/full/10.1080/14763141.2024.2381108),
  [vs. radar gun](https://journals.sagepub.com/doi/10.1177/17479541241284714),
  [fulltrack.ai/technology](https://www.fulltrack.ai/technology)
- **Arc Simulations** (UK-based) — "millimetre-accurate" ball tracking
  and simulation, cricket first, expanding across ball sports. Worth
  naming directly in the visa business plan since it's a UK competitor in
  the same ecosystem. Source: [Top Cricket Tech Startups 2026](https://www.sportstartups.org/top/cricket/)

## Adjacent: instrumented ball (different sensing modality entirely)

- **Kookaburra SmartBall** (with Sportcor, Australia) — an IMU chip
  (accelerometer + gyroscope + magnetometer) embedded inside the ball
  itself, streaming speed/spin via Bluetooth LE to an app. Measures speed
  at release, pre-bounce, and post-bounce; swing detection was in
  development. A completely different approach from vision: senses the
  ball directly rather than trying to see it. Given this project's own
  measured difficulty detecting a ball in real footage (see
  `cv-pipeline/README.md`), an instrumented-ball approach may be a more
  achievable path to real delivery-outcome sensing than out-competing
  Fulltrack AI on pixels. Source:
  [Nordic Semi / Sportcor writeup](https://www.nordicsemi.com/Nordic-news/2019/10/Sportcor-Smartball-employs-nRF52840-to-send-tracking-data-from-Kookaburra-Sports-Smartball-to-app)

## Adjacent sports: proof the core mechanism already works elsewhere

This is the most important finding: **the closed loop this project is
built around — sense real-time performance, autonomously adjust
difficulty — already exists and ships commercially in tennis. It does
not exist yet in cricket.** That reframes the pitch: not "adaptive
training is a novel idea," but "cricket hasn't caught up to what tennis
already proved works."

- **Pongbot PACE S Pro** (tennis) — position-aware sensor reads the
  player's court position at 100Hz and "waits until you're ready" before
  firing the next ball, matching feed timing to footwork recovery. Real
  closed-loop adaptation, shipping today. Source:
  [The Gadgeteer review](https://the-gadgeteer.com/2026/01/08/pongbot-pace-s-pro-is-the-ai-tennis-training-robot-that-waits-until-youre-ready-before-firing-the-next-ball/)
- **Pongbot Aura** — $499 multi-sport (tennis/pickleball/padel) machine
  from one chassis, with a detachable AI camera module ("Spotter") for
  swing feedback. Shows a real, much lower price point than BOLA's
  £2,532+ exists for AI-adaptive ball machines in an adjacent sport.
  Source: [The Gadgeteer](https://the-gadgeteer.com/2026/05/15/pongbot-aura-is-a-499-ai-ball-machine-for-tennis-pickleball-and-padel/)
- **"The Partner"** and multiple patents (US20230123704A1,
  US8419560 — the latter broadly titled "adaptive delivery of game balls
  based on player-specific performance data analysis") confirm this
  mechanism is already IP-protected territory in tennis. See patent notes
  below — this is a genuine prior-art consideration, not just a
  competitor list entry.
- **Dr. Dish** (basketball) — tracks makes/misses and shot arc in detail,
  and lets players/coaches reconfigure distance/speed/locations to
  simulate scenarios. Feedback-rich, but adjustment appeared
  player/coach-configured rather than a fully autonomous loop in what was
  found — worth a deeper look if basketball's exact mechanism matters for
  positioning. Source: [Dr. Dish Arc1](https://www.drdishbasketball.com/schools/dr-dish-arc-1)

## The gap this project actually targets (revised, evidence-based)

Not "AI in cricket training" — that's crowded (coaching apps) and partly
solved (Fulltrack AI's ball tracking, BOLA/TrueMan3's realism). The real,
still-open gap, confirmed by the above: **nothing in cricket closes the
loop from batter performance to autonomous machine reconfiguration.**
Every cricket product found is one half of that loop — a machine varying
on a schedule or a coach's program (BOLA, TrueMan3, ProBatter), or a pure
analytics app needing a human to act on it (CricVision, Fulltrack,
Advanced Impactor). Tennis has already proven the full loop works and is
commercially viable (Pongbot, The Partner); cricket hasn't caught up.

Three concrete positioning angles this research supports, beyond "we
close the loop":

1. **Retrofit, not replace.** Every competitor sells a whole new machine.
   Thousands of clubs already own a £2,500+ BOLA. `machine-control/`'s
   hardware-agnostic `MachineController` interface plus the documented
   serial protocol (`machine-control/PROTOCOL.md`) means this could be
   pitched as a brain that upgrades an existing machine, not a new
   capital purchase — nobody else found is doing this.
2. **Nets-first framing, not match-simulation.** BOLA/TrueMan3 sell
   realism (LED bowler video, "face a real bowler"). This project's own
   pivot to physically-measurable nets outcomes
   (`trajectory-engine/cricket_trajectory/net_outcome.py`:
   dead_bat/controlled_placement/well_struck/mistimed_skied, not
   fours/sixes) is a genuinely different mental model built around what
   nets sessions actually are.
3. **Continuous physics-generated deliveries vs. a preset dial.**
   BOLA/TrueMan3 vary a small set of manually-tuned knobs, saved as
   discrete "balls." This project's physics engine generates from real
   aerodynamics (cited: Mehta 1985) — effectively unbounded variation
   within a controlled difficulty band, not a cycle through presets.

See `docs/product/AI_Adaptive_Bowling_Machine_Program_Plan.docx`, Section
7, for the fuller comparison — due for an update against this file.

## Patent / IP note (not legal advice — flagging for a professional)

US20230123704A1 ("Adaptive tennis ball machine") and US8419560 ("System
and method for adaptive delivery of game balls based on player-specific
performance data analysis") are real, existing patents covering the
general "ball machine adapts to player performance" mechanism — the
latter's title is broad enough to warrant a proper freedom-to-operate
search before any patent strategy is built around this project's core
loop, regardless of the cricket-specific execution being different. This
repository is public and already indexed (confirmed: search engine
summaries have quoted this project's own README back during this
research), which has practical implications for patent filing timing —
absolute-novelty jurisdictions (UK/Europe, relevant given the visa plan)
have no grace period for an inventor's own prior public disclosure,
unlike the US's 1-year grace period. Get a patent attorney's read on
this specifically before relying on any "novel" claim in the business
plan.

## Open research questions (not yet answered)

- Real-world pricing for ProBatter PX3 and BOLA TrueMan3 specifically
  (only base BOLA Professional pricing was found).
- Market sizing: number of academies/clubs in target regions, addressable
  spend on training technology (flagged as a placeholder in the visa
  business plan, Section 4.2) — still open.
- Dr. Dish's exact adaptation mechanism (autonomous vs. manually
  reconfigured) — worth confirming if basketball is used as a comparison
  point in the pitch.
- A proper (non-web-search) patent search via a patent attorney or
  professional prior-art tool (Espacenet, USPTO full-text search) — this
  research used general web search, not a real patent search, and should
  not be treated as one.

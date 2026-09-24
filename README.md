# AI-Adaptive Bowling Machine

A cricket training system that senses how a batter is playing each bowling
style in real time, and switches pace, line, length, swing, or spin the
moment the player has mastered the current pattern — instead of varying on
a timer or at random, the way existing programmable machines do.

**Live demo:** https://cricket-ai-bowling-machine-qnsjkesghkquezmehxrbyg.streamlit.app

## Repo layout

```
docs/
  product/            Program plan and the MVP research & development plan
  business/           Business plan (UK Innovator Founder visa version)
  research/           Source notes on the competitive landscape and market
app/                  The working demo (Streamlit) — run this
adaptation-engine/    Style-library mastery scoring + recommendation (rule-based + trained neural)
cv-pipeline/          Real pretrained pose estimation + automatic delivery detection from video
trajectory-engine/    Physics-based ball flight, twin-wheel machine control, Elo-style adaptive rating
machine-control/      Hardware-agnostic control loop — the architecture real hardware plugs into
hardware/             Concrete hardware requirements derived from the software's actual interfaces — no physical build yet
data/                 Session recordings and datasets (gitignored — not committed)
```

## Run it

```
pip install -r requirements.txt
streamlit run app/app.py
```

Two independent engines, switchable in the sidebar: a style-library engine
(log deliveries against a fixed set of named styles, get a recommendation
once one is "mastered") and a physics-based engine (continuously varied
deliveries targeting a 50% "challenge point," with an automatic scorecard).
See `app/README.md` for what's real vs. still a placeholder.

## Research basis, not guessed physics

The swing model in `trajectory-engine/cricket_trajectory/aerodynamics.py`
is built on a real, peer-reviewed mechanism, not an invented curve:

> Mehta, R.D. (1985). "Aerodynamics of Sports Balls."
> *Annual Review of Fluid Mechanics*, 17, 151-189.

That paper established that a cricket ball's seam, held at a small angle to
the airflow, trips the laminar boundary layer into turbulence asymmetrically
— confirmed via smoke-flow visualization — which is the physical mechanism
`swing_coefficient()` reproduces, tuned to peak at the same ~20-25 degree
seam angle that paper's findings support. Every other empirical curve in
that module (drag, Magnus lift, seam drag) is named, documented, and stated
honestly as a physically-reasoned approximation pending real calibration —
see that file's docstrings and `trajectory-engine/docs/limitations_report.pdf`
for the full, unvarnished list of what still needs measuring against a real
ball and machine.

## Built to be hardwired, before any hardware exists

`machine-control/` is the actual control loop — decide a delivery, command
the machine, sense the outcome, update the rating, repeat — built and fully
tested against a simulated machine and a scripted outcome sequence, so it's
ready for real hardware and real sensing to be substituted in without the
loop itself changing:

- `MachineController` (abstract) is the only thing the software knows about
  a physical machine — a real driver for a specific machine implements the
  same four methods; nothing else needs to change when hardware does.
  `SerialMachineController` is that driver, already built: it speaks a
  documented line protocol (`machine-control/PROTOCOL.md`) that any
  microcontroller firmware can implement, regardless of which physical
  machine it ends up wired to — the app's sidebar lets you connect to a
  real serial port with it today, there's just no firmware on the other
  end yet.
- `SafeMachineController` wraps any controller with hard limits and an
  emergency-stop that needs an explicit human reset — never a silent
  auto-resume.
- `OutcomeObserver` (abstract) mirrors this on the sensing side, ready for
  real ball/bat tracking or a sensor once either matures past today's honest
  limits (ball detection ~12% recall on real footage; bat tracking unbuilt).

The one thing this repo cannot do on its own: choose which physical machine
to target. That's a decision, not an engineering task — see `ROADMAP.md`.

## Status

524 automated tests passing across all four Python packages
(`adaptation-engine/`, `cv-pipeline/`, `trajectory-engine/`, `machine-control/`).
What that number covers, and what it doesn't:

- **Real and tested**: mastery scoring (rule-based + a trained neural
  network), style recommendation, pose estimation on real video with
  automatic multi-delivery detection, the full physics trajectory
  simulation, twin-wheel machine-control mapping, Elo-style adaptive
  rating, automatic wide/no-ball scoring, and the hardware-agnostic control
  loop above.
- **Real but never validated on a real player or a real machine**: every
  decision engine here has been checked against simulated batters and
  simulated hardware, never a human being coached in front of an actual
  bowling machine.
- **Not built**: automatic shot-outcome detection. Two different ball-
  detection approaches were tried and both fell short: a trained YOLOv8
  detector reached ~12% recall on real footage (weak enough it was never
  committed to this repo), and an untrained classical approach
  (`cv-pipeline/motion_ball_detector.py`) tested directly against real
  clips couldn't reliably find the ball either — direct visual inspection
  of that same footage suggests the ball is at or below the visibility
  floor there, pointing at camera setup (distance, zoom, shutter speed)
  as the real blocker, not detector choice. `cv-pipeline/ball_tracking.py`
  exists to turn sparse, noisy detections into one continuous trajectory
  once a detector produces some, but there is currently no detector
  feeding it real data; bat tracking doesn't exist at all. A different
  angle on sensing exists in `machine-control/`, sidestepping vision
  entirely: `release_sensor.py` calibrates the machine's own RPM-to-speed
  mapping from a simple release-point sensor reading (since this system,
  unlike a human bowler, already knows what it commanded), and
  `impact_sensor.py` classifies shot outcome from a simple post-contact
  sensor reading (distance, height) rather than tracking the ball
  visually — both real, tested software, against simulated sensors, since
  no physical sensor exists yet either. The live-camera side is
  real, tested software now too — `cv-pipeline/live_video_source.py` and
  `live_delivery_detector.py` process a continuous frame stream instead
  of an uploaded file, validated by playing real WhatsApp clips back
  frame by frame and matching the existing batch pipeline exactly on
  every delivery a genuinely continuous camera would also complete
  (three real bugs were found and fixed doing this — see
  `cv-pipeline/README.md`). What's still missing is the camera itself —
  `LiveVideoSource` opening a real device is unexercised, identical in
  shape to opening a file but unexercised all the same — and any real
  hardware connection — `machine-control/` has a real serial driver and
  documented protocol ready for a microcontroller to speak, but no
  physical machine has been
  chosen, so nothing is plugged in yet.

## Start here

1. [`docs/product/AI_Adaptive_Bowling_Machine_Program_Plan.docx`](docs/product/AI_Adaptive_Bowling_Machine_Program_Plan.docx) —
   the full concept: the problem, the AI/ML architecture, hardware plan,
   phased roadmap, risks, and success metrics.
2. [`docs/product/MVP_RD_Plan_Software_First.docx`](docs/product/MVP_RD_Plan_Software_First.docx) —
   the current, active plan: a human-in-the-loop MVP that tests whether the
   adaptation logic works at all, before any embedded hardware or
   low-latency actuation work begins.
3. [`docs/business/UK_Innovator_Founder_Business_Plan.docx`](docs/business/UK_Innovator_Founder_Business_Plan.docx) —
   the same concept restructured around the UK Innovator Founder visa
   endorsement criteria (innovation, viability, scalability). Contains
   placeholders that still need founder-specific detail — not submission-ready.
4. [`ROADMAP.md`](ROADMAP.md) — the phased plan (0–5) at a glance.

## Current phase

**MVP (software-first).** On-device inference and low-latency automated
switching are deliberately deferred — see the MVP plan for why. The active
build is human-in-the-loop: a coach reads a recommendation and manually sets
it on an existing programmable machine; nothing here actuates hardware yet.

## License

Public repository. Source is visible for demo/deployment purposes; no
license is granted for reuse or redistribution. Update this section once
the team has settled on formal IP terms (patents, contributor agreements,
an actual open-source or proprietary license) for anyone joining the
project.

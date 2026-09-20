# cricket_bowling_machine

A physics-based cricket-ball trajectory simulator, a twin-wheel bowling-machine
control mapping, and an adaptive-difficulty layer that adjusts what the
machine bowls based on how the player is performing — packaged together as
one pipeline.

Read this alongside the three PDFs in `docs/`, which explain the physics,
the algorithm, and the honest limitations in full. This README is the code
map and quickstart; the PDFs are the "why".

## What's here

```
cricket_bowling_machine/
├── cricket_trajectory/       the actual algorithm, as an importable package
│   ├── constants.py          physical constants (ball mass, air density, pitch length, ...)
│   ├── ball.py                BallProperties, Environment, Delivery data classes
│   ├── aerodynamics.py        Cd(Re), Cl(spin), Cs(seam angle), Cds(seam angle) coefficient curves
│   ├── dynamics.py            sums gravity + drag + Magnus + swing + seam-drag into one acceleration
│   ├── simulate.py            numerically integrates one delivery into a full trajectory
│   ├── machine.py             twin-wheel machine <-> ball release state, forward and inverse
│   ├── adaptive.py            player rating, delivery difficulty rating, next-delivery selection
│   └── scorecard.py           the bridge: turns a bowled delivery + reported outcome into a scorecard
├── demo.py                    quickstart: one delivery, through the whole pipeline
├── demo_adaptive.py           a longer synthetic 70-ball adaptive session, with a chart
├── demo_scorecard.py          full loop: bowl -> legality check -> outcome -> scorecard -> next delivery
├── docs/
│   ├── cricket_ball_physics_report.pdf     the physics, equations, and algorithm in full
│   ├── cricket_adaptive_difficulty.pdf     the adaptive-difficulty layer explained
│   └── limitations_report.pdf              honest limitations of both, in one place
└── requirements.txt
```

## Install

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ (uses `from __future__ import annotations` throughout).

## Quickstart

```bash
python3 demo.py
```

This runs one delivery through all three stages: the trajectory simulation,
the wheel-RPM mapping for a physical machine, and a handful of adaptive-layer
ball-by-ball outcomes so you can see the player rating move.

For a longer, chart-producing adaptive session:

```bash
python3 demo_adaptive.py
```

## The pipeline, in order

**1. Describe a delivery.** `Delivery` holds everything about one ball at
release — speed, launch angle, seam angle, spin vector, reverse-swing flag.
`BallProperties` and `Environment` describe the ball itself and the air it's
flying through.

```python
from cricket_trajectory import BallProperties, Environment, Delivery, run_simulation

ball = BallProperties()
env = Environment()
delivery = Delivery(speed_mps=37.5, seam_angle_deg=22, label="outswinger")

result = run_simulation(ball, env, delivery)
print(result.summary())
```

**2. Turn it into machine control.** `WheelMachine` converts between wheel
RPMs and the ball's release speed/spin (both directions).

```python
from cricket_trajectory.machine import WheelMachine

machine = WheelMachine()
rpm1, rpm2 = machine.wheel_rpms_for_delivery(
    ball_speed_mps=delivery.speed_mps, spin_rate_rad_s=0.0, ball=ball,
)
```

**3. Adapt to the player.** `PlayerProfile` tracks one batter's rating.
`next_delivery_after()` logs what happened on the ball just faced and
immediately returns what to bowl next, so a real control loop only needs
one call per ball.

```python
from cricket_trajectory import PlayerProfile, next_delivery_after

profile = PlayerProfile(name="Player 1", rating=1000.0)
record, next_ball = next_delivery_after(profile, ball, delivery, outcome="defended")
# next_ball is a Delivery -> feed it back into step 2 for the next set of wheel RPMs
```

Outcomes are one of `"missed"`, `"beaten"`, `"edged"`, `"defended"`,
`"controlled"`, `"boundary"`, `"six"` (see `adaptive.OUTCOME_SCORES`), or any
raw float from 0 to 1 if you want a finer-grained scoring scheme — e.g. from
an automated vision/sensor system instead of a person reporting it.

**4. Turn it into an actual scorecard.** `scorecard.py` is the bridge between
a bowled ball and a real scorecard entry. It needs no sensor for legality —
`classify_delivery_legality()` computes wide/no-ball directly from the
delivery's own simulated trajectory, since the machine already knows exactly
what it told the ball to do. It does need a reported outcome for anything
else (runs, wickets) — that information only exists on the batter's side of
the ball, so a person (or eventually a vision/sensor system) has to supply
it, same as step 3 above.

```python
from cricket_trajectory import Scorecard, classify_delivery_legality

card = Scorecard(batter_name="Player 1")
result = run_simulation(ball, env, delivery)

if classify_delivery_legality(result) is None:      # a fair delivery
    card.record_ball(profile, ball, delivery, result, outcome="controlled")
else:                                                 # wide or no-ball
    card.record_ball(profile, ball, delivery, result)

print(card.render_text())
```

See `demo_scorecard.py` for the full loop: bowl, check legality, score the
outcome, update the player's adaptive rating, ask for the next delivery, repeat.

**5. Describe how the ball is bowled.** `delivery_report.py` turns a
delivery and its simulated flight into what a coach would say: length
(yorker / full / good / short-of-a-length / short, plus metres from the
stumps), line (off / middle / leg, wide), swing, spin, speed at release and
at the pitch, and legality. `build_delivery_report(delivery, result,
measured_release_speed_mps=...)` also carries the release sensor's reading,
so the report shows commanded vs. measured speed side by side. The pitch
point and swing are the physics model's **prediction** from the commanded
delivery, not a measurement of the real ball — on hardware the release
sensor supplies the speed, and nothing yet measures where the real ball
lands. Length/line boundaries are named, adjustable constants (judgment
calls, not a governing-body standard).

```python
from cricket_trajectory import build_delivery_report
report = build_delivery_report(delivery, run_simulation(ball, env, delivery))
for label, value in report.rows():
    print(f"{label}: {value}")
```

## Persisting a player's progress

`PlayerProfile` is a plain dataclass, so saving it between sessions is a
matter of your own serialization of choice, e.g.:

```python
import json
from dataclasses import asdict

with open("player_1.json", "w") as f:
    json.dump(asdict(profile), f)
```

## Where to calibrate first

Every coefficient in `aerodynamics.py` is a physically-shaped curve, not a
number measured off your specific ball and machine — see
`docs/limitations_report.pdf` for the full list. In priority order:

1. Wheel efficiencies (`WheelMachine.speed_efficiency`, `spin_efficiency`) —
   cheapest to measure (radar gun + slow-motion spin count), biggest accuracy
   win on pace and line.
2. The drag curve (`aerodynamics.drag_coefficient`) — matters most if
   predicting pace accurately matters to you.
3. The swing/spin curves (`swing_coefficient`, `seam_drag_coefficient`,
   `magnus_lift_coefficient`) — most expensive to measure accurately and the
   least forgiving of noisy data, so calibrate last.

The adaptive layer's *ranking* of which deliveries are harder than others
stays sound even before calibration (it only needs relative difficulty to be
right); the exact difficulty numbers will not be exact until these curves are.

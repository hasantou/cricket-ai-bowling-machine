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

**6. Name the shot.** `shot_analysis.py` infers which shot was played from
what the post-contact sensor saw the ball do (exit speed, launch angle,
direction) plus the delivery's length: "cover drive", "pull", "square cut",
"forward defence", "mistimed / skied hit", and so on, with where the ball
went (mid-off, cover, midwicket, ...). Where the ball went is what defines
most named shots, which is why this comes from the sensor and not from video.

It is an **inference from a rule table**, not a measurement and not a coach's
verdict: direction bands and length groupings are named, uncalibrated
conventions, and it assumes a right-handed batter. If the batter's footwork
isn't supplied, front/back foot is *inferred from the ball's length* and the
result is marked "approximate"; pass an observed `footwork` (e.g. from video)
and it is marked "firm". It has not been checked against a coach's labels.

`shot_vocabulary.py` is the shot list it speaks (transcribed from a saved
general-purpose shot-name reference — a plain list, not a coaching standard),
and it states which shots can be named from the sensor today (drives, flick,
leg glance, square cut, late cut, back-foot punch, pull, hook, defences) and
which cannot, with the reason: sweeps need the bowler type; cut (generic),
slog, switch hit, scoops, upper cut, helicopter and dead-bat need bat or body
information; leave needs video. Tests enforce both directions — the
classifier never outputs a name outside the list, and never claims a shot the
vocabulary says it can't tell apart.

```python
from cricket_trajectory import analyse_shot
shot = analyse_shot(exit_velocity, delivery_report.length_label)   # exit_velocity None = no contact
print(shot.shot, shot.region, shot.confidence)
```

**7. Combine the sensor with video.** `shot_fusion.fuse_shot(sensor_shot,
video_shot)` treats the ball (sensor) and the batter's hands (video) as two
independent witnesses to the same shot. Agreement is reported as corroborated;
a disagreement is reported, not hidden, and the sensor's name is used (it
measures the ball); a swing with no contact at the sensor is "swing and a miss"
only if the video also saw the swing; and nothing is named if neither source
named anything. It takes plain objects, so this package does not depend on the
computer-vision package. Whether agreement really makes the answer more
reliable is one of the questions in `docs/real_world_test_protocol.md`.

**8. Judge the delivery by the Laws.** `laws.py` holds the dimensions and rules a machine can apply,
read from the source: the MCC *Laws of Cricket* (2017 Code, 3rd edition 2022, 79-page text) and the ICC
T20 World Cup 2024 playing conditions (URLs and clause numbers are in the module docstring). It also maps
**all 42 Laws** to what a bowling machine and a net can do with them (`LAWS`, `coverage()`): implemented,
informational, needs a sensor or a person, or not applicable.

What the source changed in the design: **the Laws do not define a wide by a fixed distance.** Law 22.1 says
a ball is wide if it passes wide of where the striker stands and is not "sufficiently within reach... by
means of a normal cricket stroke", judged as it passes the striker's wicket (22.2); the ICC adds that a ball
above head height at the popping crease is wide (22.1.1.2). The fixed limits used here (0.89 m off side,
0.5 m leg side) are **proxies for the umpire's judgement**, not the Law; the 0.89 m off-side figure is derived
from the return crease (Law 7.4, 1.32 m) and a 17 in offset reported by Wisden's explainer, and is not in the
extractable text of either primary document. Reach, waist height and head height are named `WideRules` fields,
uncalibrated. `assess_delivery()` returns the call, the margin to the limit (with a 3 cm "borderline" band),
whether the ball would hit the stumps (Law 32.1; stumps 9 in x 28 in), and informational no-ball conditions
(pitched off the 3.05 m pitch, Law 21.7; bounced twice; waist-high full toss, Law 41.7.1). **No-balls are
counted, not scored**: a machine has no front foot.

It rests on `crease_crossing.py`, which carries the ball through its bounce to the batter (the older
simulation stopped at the first ground contact, so "where it passes the batter" was a proxy). The bounce is a
simple model (vertical restitution 0.55, horizontal retention 0.75) that is **uncalibrated** - a real
crease-plane sensor's readings against its predictions are exactly the data to calibrate it
(`machine-control/machine_control/crease_sensor.py`, protocol appendix, `hardware/README.md` 5b).
A miss now costs a wicket only if the ball would have hit the stumps (`resolve_no_contact`); otherwise it is
"beaten". LBW needs pad contact and is not detectable.

**9. Aim the delivery.** `targeting.py` solves the launch angles that land the ball at a chosen line and
length (bisection on vertical angle for length, secant on horizontal for line). Measured on 120 generated
deliveries **before** it existed: 35% wides, 8% on target, and the length at the batter was 61% full toss,
32% yorker, 8% full, **no good-length or short balls** (the launch angle never varied). Measured on 60 aimed
deliveries **after**: 0% wides, 22% on target (the corridor is deliberately mostly outside off), and a real length
mix (57% good length, 17% full, 13% yorker, 10% short of a length, 3% short). The adaptive difficulty logic is
unchanged - difficulty depends only on pace, seam and spin - so a candidate is picked for difficulty, then aimed.
The length mix and line corridor are choices, not coaching norms. Solve time is about 2 s per delivery.

**10. Tell the whole story of one delivery.** `delivery_story.build_story()` joins everything known about a
single ball: how it was bowled (commanded, physics prediction, release sensor), where it went at the batter
(the Laws-based call, from the crease sensor if there is one), what the batter did (swing, footwork, weight
transfer, from video), what happened at contact (impact sensor, shot from the ball, shot from the hands, and the
fused shot), and the result. **Every fact carries its source** - `commanded`, `physics prediction`, `release /
crease / impact sensor`, `video (pose)`, `video (rule of thumb)`, `inferred` or `not measured` - and anything from a
simulated sensor says `simulated`. It never fills a gap. On top of the table it adds:

- **cross-checks** (rules of thumb, prompts for a coach): footwork against the length bowled ("went forward to a
  short ball"), and stride timing against the swing (too late / too early);
- **flags** where sources disagree: the sensor felt bat but the camera saw no swing; the release sensor differs
  from the command; the crease sensor overrules the model; the camera and sensor clocks do not line up; the footage
  is poor;
- a plain-language **narrative** hedged by source.

`delivery_sync.align()` matches the camera's swing peaks to the sensors' contact times across the two clocks
(offset search, missed and spurious events left unmatched not forced, drift flagged). It is tested on synthetic
timestamps only - there is no real machine yet, so real clock jitter, drift and latency are unverified.
The camera side supplies `cv-pipeline/story_adapter.py`; the video-only view builds a story with no machine data
and says so.

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

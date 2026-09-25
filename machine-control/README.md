# machine-control/

The software architecture connecting the existing decision engines to real
bowling-machine hardware and real outcome sensing — built and fully tested
against **simulated** stand-ins for both, so it's ready for real hardware
and real sensing to be substituted in without anything here changing.

## Why this exists

Everything else in this repo (`adaptation-engine/`, `cv-pipeline/`,
`trajectory-engine/`) is real, tested decision logic — but nothing tied it
into an actual running cycle: decide a delivery, command a machine, sense
what happened, update the rating, repeat. `app.py` today is a human
clicking through a form once per ball. This package is that missing cycle,
built now, on the software side, deliberately *before* any real hardware
exists to plug into it — see project notes for why "which machine" is a
decision blocking real hardware integration, not a queued task.

## What's here

```
machine_control/
  controller.py          MachineController (abstract) + SimulatedMachineController
  serial_controller.py   SerialMachineController — a real driver, speaks PROTOCOL.md
  safety.py              SafeMachineController — hard limits + kill-switch, wraps any controller
  outcome_observer.py    OutcomeObserver (abstract) + ScriptedOutcomeObserver
  impact_sensor.py       ImpactSensor + VelocitySensor (abstract) — two real OutcomeObserver implementations
  release_sensor.py      ReleaseSensor (abstract) + SpeedCalibrator — measures and corrects machine.py's RPM->speed mapping
  orchestrator.py         MachineOrchestrator — the actual decide->command->sense->score loop
  session_store.py       JSON save/load for PlayerProfile and Scorecard (resumability)
PROTOCOL.md              The serial wire protocol SerialMachineController speaks
tests/                   63 tests covering every piece above
demo_orchestrator.py     runnable end-to-end demo against simulated hardware
```

## The two abstraction boundaries

**`MachineController`** (`controller.py`) is the only thing the rest of this
software knows about a physical machine: `set_delivery(rpm1, rpm2)`,
`emergency_stop()`, `is_ready()`, `reset_after_stop()`. A real driver for a
specific machine (BOLA, ProBatter, a custom rig) implements this against
whatever interface that machine actually exposes — that implementation
doesn't exist yet and can't, without knowing which machine this targets.
`SimulatedMachineController` implements the same interface with no real
hardware, so everything above it is real, runnable, tested software today.
`SerialMachineController` (`serial_controller.py`) is a second, real
implementation of that same interface — it drives an actual serial link
using the line protocol documented in `PROTOCOL.md`, so any microcontroller
that implements that protocol works with this software already, without
knowing which specific bowling machine it ends up wired to. Tested against
a fake serial connection (`tests/test_serial_controller.py`) — the protocol
logic (what gets sent, how a response is interpreted) is real and covered;
the physical link and the firmware on the other end of it are not, because
neither exists yet (see `PROTOCOL.md` for what's deliberately left open).

**`OutcomeObserver`** (`outcome_observer.py`) mirrors this on the sensing
side: `observe(delivery, ball) -> outcome`. `ScriptedOutcomeObserver`
replays a fixed sequence for tests and demos. `impact_sensor.py` has the
first two *real* implementations, built on a different idea from ball
tracking: rather than trying to visually resolve a small fast ball (the
approach `cv-pipeline/motion_ball_detector.py` tried against real footage
and found genuinely hard), each wraps a much simpler post-contact
sensor. `ImpactSensorOutcomeObserver` wraps trip-beams or a
pressure-sensitive net panel giving (distance, height).
`VelocitySensorOutcomeObserver` wraps a sensor that reports an exit
velocity vector directly — exactly what a stereo-camera post-shot
trajectory rig (triangulated 3D position over the first 50-200ms after
contact, differentiated for velocity) would produce. Both run their
reading straight through `trajectory-engine/cricket_trajectory/net_outcome.py`'s
already-tested classification logic (`classify_net_outcome()` and
`classify_exit_velocity()` respectively) rather than a second decision
path. No real sensor exists yet; `SimulatedImpactSensor` and
`SimulatedVelocitySensor` stand in for one.

**`release_sensor.py`** closes a different, pre-contact gap: unlike a
human bowler, this system commands the delivery it's about to make, so it
already knows the intended speed — it only needs one cheap sensor (a
photogate, the same tech radar guns use) at the release point to confirm
what actually happened and correct
`trajectory-engine/cricket_trajectory/machine.py`'s RPM-to-speed mapping
over time, exactly what that module's own docstring already asks for
("calibrate ... with a radar gun ... then treat it as fixed") but never
had code for. `SpeedCalibrator` fits the correction from real
measurements; tested by recovering a known true efficiency from noisy
synthetic measurements (verified stable across 100 random seeds, worst
recovered error 0.4%).

## The safety layer

`SafeMachineController` (`safety.py`) wraps any `MachineController` and
validates every command against `SafetyLimits` — a hard RPM range and a max
wheel-speed differential — **before** it reaches the underlying controller.
Violations raise `SafetyViolation` rather than silently clamping (a wildly
wrong RPM reaching here is more likely a bug upstream than a legitimate
edge case worth quietly tolerating). `emergency_stop()` disables the layer
outright; resuming needs an explicit `reset()` call — nothing here resumes
just because time has passed.

`SafetyLimits`' defaults are a conservative placeholder, not a calibrated
value — replace them with the real target machine's documented safe
operating range once one is chosen. That placeholder isn't an arbitrary
guess, though: `tests/test_safety_envelope.py` checks it directly against
what the adaptive engine can actually generate (trajectory-engine's own
`DEFAULT_SPEED_RANGE_KMH` / `SPIN_RPM_RANGE`, run through `WheelMachine`'s
real forward model across the whole envelope, not just spot values) — the
worst realistic case needs under 4200 RPM per wheel and under 1350 RPM of
differential, comfortably inside the 5000/3000 defaults, with margin left
for calibration error. If either envelope constant ever widens, this test
is what catches the two silently drifting apart.

**For a machine that has never been run before**, use
`SafetyLimits.commissioning()` instead of the defaults — deliberately much
tighter (1000 RPM / 500 RPM differential), standard practice for bringing
up new, unverified hardware at reduced power before trusting it with a
full-range command. Confirmed directly that it actually would reject a
real full-pace delivery, not just that the numbers look smaller. Move to
the full `SafetyLimits()` (or the real machine's own measured spec) only
after a person has watched it behave safely at reduced power — never
automatically.

## Resumability

`session_store.py` gives `PlayerProfile` and `Scorecard` — both plain
dataclasses already, by design (see `trajectory-engine/README.md`) — actual
JSON persistence. Nothing decides *when* to save automatically; callers
(the app, or a real deployment) do that explicitly.

**Long-term, per-batter profiles.** `session_store.py` alone still needed a
person to manage individual JSON files by hand — nothing found "this
batter's" file automatically. `profile_store.py` closes that one gap:
`save_named_profile`/`load_named_profile` key a profile by the batter's own
name (slugified to a safe filename), so a returning batter's second session
finds their real history — every skill rating, every delivery ever faced —
without anyone tracking file paths. `list_known_batters` is what a "pick a
returning batter" dropdown would call, and `profile_summary` gives a
long-term snapshot (deliveries faced, current skill ratings, and a rating
trend once at least 20 deliveries exist) instead of just the current number.

Deliberately the simplest thing that solves today's actual problem: one
JSON file per batter in a directory, no database, no server. That's not the
design for multiple machines needing the same batter's profile at once —
this only becomes wrong once that's a real requirement, and swapping the
storage functions for a real database then is a contained change, since
nothing calling them needs to know the difference. The one real safety
property this DOES enforce now: two different batters whose names slugify
to the same filename (`"Sam"` / `"SAM"`) raise `ProfileNameCollision`
instead of one silently overwriting the other's history.

## Run it

```
python3 demo_orchestrator.py
```

Runs 12 simulated deliveries through the full loop, prints each machine
command and outcome, renders the resulting scorecard, and saves the session
to `output/` — restart from those files to see resumability rather than
losing everything when the process ends.

## Tests

```
python3 -m pytest machine-control/tests/ -v
```

63 tests: the simulated controller's readiness/stop semantics, every
safety-limit rejection path (including a real bug this caught — `reset()`
originally only cleared this layer's own flag, not the inner controller's
stopped state, which would have silently left the machine stuck even
though the safety layer reported itself re-enabled), the scripted
observer, session-store round-trips, the orchestrator's full run loop
including a safety-violation stopping it immediately, the serial
protocol's encode/decode logic against a fake connection, and — the
sensor modules — `ImpactSensorOutcomeObserver` classifying every net
outcome correctly from raw (distance, height) readings,
`VelocitySensorOutcomeObserver` doing the same from an exit-velocity
vector (speed, elevation, azimuth) via `net_outcome.classify_exit_velocity()`, and
`SpeedCalibrator` recovering a machine's true (initially unknown)
speed_efficiency from noisy simulated measurements and confirming the
correction actually improves future delivery accuracy, not just that the
recovered number "looks right."

## What this does *not* claim to solve

This is real, tested control-loop **software** — it is not a claim that
automatic outcome sensing works. `ScriptedOutcomeObserver` is a stand-in
for a real sensing method that doesn't exist reliably yet. The honest gap
is unchanged: shot outcome still needs either a human, or the ball/bat
tracking and sensor work described in project notes to mature further.
What this closes is the *architecture* gap — there was no control loop to
plug real sensing into at all; now there is.

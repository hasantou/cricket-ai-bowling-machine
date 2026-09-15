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
  controller.py         MachineController (abstract) + SimulatedMachineController
  safety.py              SafeMachineController — hard limits + kill-switch, wraps any controller
  outcome_observer.py    OutcomeObserver (abstract) + ScriptedOutcomeObserver
  orchestrator.py         MachineOrchestrator — the actual decide->command->sense->score loop
  session_store.py       JSON save/load for PlayerProfile and Scorecard (resumability)
tests/                   28 tests covering every piece above
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

**`OutcomeObserver`** (`outcome_observer.py`) mirrors this on the sensing
side: `observe(delivery, ball) -> outcome`. No real implementation exists
yet either — ball tracking is real but weak (~12% recall on real footage),
bat tracking isn't built, and the sparse-detection physics fit is only
reliable for slower/shorter shots (see project notes for the measured
limits). `ScriptedOutcomeObserver` replays a fixed sequence for tests and
demos.

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
operating range once one is chosen.

## Resumability

`session_store.py` gives `PlayerProfile` and `Scorecard` — both plain
dataclasses already, by design (see `trajectory-engine/README.md`) — actual
JSON persistence. Nothing decides *when* to save automatically; callers
(the app, or a real deployment) do that explicitly.

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

28 tests: the simulated controller's readiness/stop semantics, every
safety-limit rejection path (including a real bug this caught — `reset()`
originally only cleared this layer's own flag, not the inner controller's
stopped state, which would have silently left the machine stuck even
though the safety layer reported itself re-enabled), the scripted
observer, session-store round-trips, and the orchestrator's full run loop
including a safety-violation stopping it immediately.

## What this does *not* claim to solve

This is real, tested control-loop **software** — it is not a claim that
automatic outcome sensing works. `ScriptedOutcomeObserver` is a stand-in
for a real sensing method that doesn't exist reliably yet. The honest gap
is unchanged: shot outcome still needs either a human, or the ball/bat
tracking and sensor work described in project notes to mature further.
What this closes is the *architecture* gap — there was no control loop to
plug real sensing into at all; now there is.

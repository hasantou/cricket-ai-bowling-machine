# hardware/

Concrete hardware requirements for adopting the software this repo
already has — not a wishlist, a spec derived directly from the actual
interfaces (`machine-control/`, `trajectory-engine/`, `cv-pipeline/`)
that are built, tested, and waiting for real hardware to plug into.

**How to read this**: every requirement below traces back to a specific
file and, where relevant, a number pulled directly from the code, not
guessed. Where a real number doesn't exist yet (because it depends on
which machine gets chosen), that's stated explicitly rather than filled
in with a plausible-sounding placeholder.

## Overview: six subsystems, six existing interfaces

| # | Subsystem | Plugs into | Status |
|---|---|---|---|
| 1 | Twin-wheel delivery head | `trajectory-engine/cricket_trajectory/machine.py` (`WheelMachine`) | Physics model exists; no physical wheels exist |
| 2 | Microcontroller + firmware | `machine-control/PROTOCOL.md`, `serial_controller.py` | Protocol documented and tested against a fake link; no firmware written |
| 3 | Release-point sensor | `machine-control/machine_control/release_sensor.py` | `ReleaseSensor` interface + calibration math tested; no physical sensor |
| 4 | Camera (batter side) | `cv-pipeline/` (pose estimation, live pipeline) | Software works today on real footage; no camera mounted |
| 5 | Post-contact sensor (net side) | `machine_control/impact_sensor.py` | Two real `OutcomeObserver` implementations tested; no physical sensor |
| 6 | Compute unit | Everything above | Runs today on a laptop; needs a permanent home on/near the machine |

---

## 1. Twin-wheel delivery head

`WheelMachine` (`trajectory-engine/cricket_trajectory/machine.py`) already
assumes a specific mechanical design — two independently speed-controlled
wheels squeezing the ball between them, spin driven by the *difference*
in their surface speeds, pace by their *average*. Whatever machine gets
chosen has to match this shape, or `machine.py`'s forward/inverse model
needs rederiving for a different mechanism (e.g. a single wheel + a
seam-angling head).

**Derived requirement, not a guess**: feeding the software's own
delivery-generation range (70–150 km/h pace, per `adaptive.py`'s
`suggest_next_delivery`, and up to 1200 rpm of imparted ball spin)
through `WheelMachine`'s current defaults (12cm wheel radius, 0.90 speed
efficiency, 0.55 spin efficiency) computes a required wheel-surface RPM
range of **roughly 1000–4400 RPM**, per wheel, independently controllable.
`SafeMachineController`'s configured ceiling in `app.py` is 6000 RPM —
chosen with headroom above that computed range, not arbitrarily.

- Two motors capable of sustained operation in the ~0–4500 RPM range at
  the wheel surface (gearing/wheel-diameter choice is a tradeoff against
  motor RPM — the number that matters is *wheel surface* RPM, not motor
  shaft RPM, if there's a gearbox in between).
- Independent electronic speed control per wheel (not a single shared
  speed dial) — spin control depends entirely on being able to command
  the two wheels differently.
- Command latency low enough that a delivery can be set and the ball
  released within the existing ball-to-ball interval — BOLA's own
  documented interval is 7–11 seconds; that's the real budget, not a
  guess.
- Whatever the actual RPM/wheel-radius numbers turn out to be,
  `WheelMachine`'s `wheel_radius_m`, `speed_efficiency`, and
  `spin_efficiency` fields need setting to the real machine's measured
  values — `machine.py`'s own docstring already says to calibrate these
  with a radar gun, and `release_sensor.py`'s `SpeedCalibrator` (built,
  tested, see below) does exactly that automatically once a real release
  sensor exists.

## 2. Microcontroller + firmware

Nothing here exists yet — `machine-control/PROTOCOL.md` was written
*for* a firmware author to implement, not derived from existing firmware.
`SerialMachineController` (the host side) is real and tested against a
fake serial connection; the actual microcontroller code is the gap.

**Requirements**:
- Any microcontroller with a serial interface: Arduino (Uno/Mega/Nano),
  ESP32, or a Raspberry Pi Pico all work — the protocol is deliberately
  generic ASCII-over-serial, not tied to a specific chip.
- Must implement the exact four commands and five responses in
  `PROTOCOL.md`: `PING` → `READY`/`BUSY`/`STOPPED`; `SET <rpm1> <rpm2>` →
  `OK`/`BUSY`/`ERR <reason>`; `STOP` → `OK`; `RESET` → `OK`/`ERR <reason>`
  (only after confirming it's mechanically safe to resume — never just
  because time passed).
- Serial: 8N1, 115200 baud by default (configurable on both ends — see
  `SerialMachineController.__init__`'s `baudrate` parameter).
- Firmware must translate a commanded `SET` RPM pair into whatever its
  actual motor drivers need (PWM duty cycle, ESC signal, stepper pulses —
  depends entirely on the motor choice in section 1) — that translation
  layer doesn't exist in this repo and is genuinely hardware-specific.
- **A physical emergency-stop button, wired directly into the motor
  power path, independent of the microcontroller.** `SafeMachineController`
  (`safety.py`) is the *software* kill-switch — real, tested, and it
  requires an explicit human `reset()` call, never an automatic resume.
  But software can hang, crash, or lose the serial connection; a hardware
  E-stop that cuts motor power directly, with no microcontroller in the
  loop, is the physical half of that same safety design and doesn't
  exist yet. This is not optional — see `safety.py`'s own docstring:
  "the physical half (a real emergency-stop button, mechanical limits) is
  hardware work this can't do."

## 3. Release-point sensor

Feeds `machine_control/release_sensor.py`'s `ReleaseSensor` interface —
one method, `measure_speed_mps() -> float`, called once per delivery.
`SpeedCalibrator` (same file) is real, tested software that turns a
stream of these readings into a corrected `WheelMachine.speed_efficiency`
— verified to recover a machine's true efficiency to within 0.4% across
100 simulated trials. It has never seen a real reading.

**Requirements** (photogate approach, the cheapest option that satisfies
the interface):
- Two IR emitter/receiver (break-beam) pairs, mounted a known, fixed
  distance apart (e.g. 20–30cm) directly in front of the wheel head,
  square across the ball's exit path.
- Beam-break timing resolution good enough that a 30 m/s ball crossing a
  20cm gap (≈6.7ms) is measured to within a few percent — a
  microcontroller-timed interrupt on each beam, not a polled loop, is
  needed at that speed.
- Reports one float (m/s) per delivery back over the same serial link (a
  natural fifth `PROTOCOL.md` message, e.g. `SPEED <value>`, would need
  adding — not currently specified, since no sensor existed to motivate
  its exact shape yet).
- Alternative: a small Doppler radar module (the same underlying tech as
  a cricket speed-gun) if precise beam-break timing circuitry is harder
  to get right than budget for a radar module — either satisfies the same
  one-float-per-delivery interface.

## 4. Camera (batter side)

This is the one subsystem whose software is already validated against
real footage — `cv-pipeline/pose_estimation.py`,
`live_video_source.py`, and `live_delivery_detector.py` correctly detect
deliveries and compute footwork/timing in real time, sustained ~28-30fps
on ordinary CPU inference (see `cv-pipeline/README.md`'s real-time
section). The hardware gap is narrow: a camera has never actually been
pointed at anything live.

**Requirements**:
- Any USB webcam or a Raspberry Pi Camera Module, exposed to Python as a
  standard `cv2.VideoCapture` device — `LiveVideoSource` accepts a camera
  index or stream URL identically to a file, so this is a plug-in, not new
  code.
- ≥30fps at whatever resolution keeps a full-body, broadly side-on view
  of the batter in frame — `feature_extraction.py`'s pose math
  assumes a side-on view of a right-handed batter (a stated, real
  limitation — see `cv-pipeline/README.md` — not yet handled for
  left-handers or a different camera angle).
- Positioned so the batter's whole body (head to feet) stays in frame
  through the stance, stride, and swing — pose landmarks silently drop
  for any frame where MediaPipe can't find a full person.
- **Not required for this camera**: any ability to see the ball. That's
  deliberately a separate, unsolved problem (see `cv-pipeline/README.md`'s
  honest writeup on `motion_ball_detector.py`) — this camera's whole job
  is the batter, not the ball.

## 5. Post-contact sensor (net side)

Feeds `machine_control/impact_sensor.py` — two independent, already-real
`OutcomeObserver` implementations exist, both untested against real
hardware because none exists. Pick one tier:

**Tier A — cheap, `ImpactSensor` interface** (`measure_impact() -> (distance_m, height_m)`):
- Trip-beam pairs at 2-3 known distances along the net's length (matching
  `net_outcome.py`'s named thresholds: `DEAD_BAT_DISTANCE_M = 2.5`,
  `WELL_STRUCK_DISTANCE_M = 12.0`), or a pressure-sensitive strip laid
  along the net floor, to localise where the ball lands.
- A simple height gate (e.g. a beam at `SKIED_HEIGHT_M = 4.0`) to detect
  a ball that ballooned up.
- Reports (distance, height) once per delivery, or nothing if no contact
  was detected (`NoContactDetected`, per the interface's own contract).

**Tier B — richer, `VelocitySensor` interface** (`measure_exit_velocity() -> (vx, vy, vz)`):
- The stereo-camera exit-trajectory design already reviewed for this
  project: two synchronised, global-shutter cameras, ≤1/1000s exposure,
  120-240fps, calibrated via a ChArUco board, triangulating the ball's
  first 50-200ms of flight after contact. Gives real speed, launch angle,
  and shot direction (`net_outcome.ExitVelocity`) instead of just two
  numbers — but is a materially bigger hardware and calibration
  commitment than Tier A (see the design review already in project
  notes for the tradeoffs, including the recommendation to pair it with
  an audio/accelerometer contact-timing trigger on the bat rather than
  relying on vision alone to find the moment of contact).

Either tier plugs into the exact same `OutcomeObserver.observe()` call
the orchestrator already uses — nothing above this layer needs to know
which tier is installed.

## 6. Compute unit

Everything above needs somewhere to actually run the Python software.

- Anything satisfying `requirements.txt` — a Raspberry Pi 4/5 (if the
  camera is Pi-native), a small mini-PC, or a laptop physically mounted
  near the machine, all work identically from the software's point of
  view.
- Needs simultaneously: a USB (or CSI) connection for the batter-side
  camera, a serial (USB) connection to the microcontroller in section 2,
  and — depending on where the net-side sensor physically sits relative
  to the compute unit — either a wired serial link or a wireless one
  (an ESP32 with WiFi reporting net-side readings back over the network
  is a reasonable choice if running a cable the length of the net isn't
  practical).
- No specific CPU/GPU requirement is set by this repo today: the CV
  pipeline's real-time validation (`cv-pipeline/README.md`) was measured
  on ordinary CPU inference, comfortably within the 7-11 second
  ball-to-ball budget — a dedicated GPU/NPU is not a stated requirement,
  only a possible future optimisation if a more demanding model
  (e.g. the full stereo pipeline in section 5, Tier B) turns out to need
  one.

## What this document deliberately does not decide

- **Which physical machine** — a BOLA, a Sportsmech, or a custom-built
  rig all satisfy section 1's requirements differently; this is a
  sourcing/partnership decision, not an engineering one, and remains the
  single blocking decision noted throughout this project (see
  `ROADMAP.md` and `docs/research/competitive-landscape.md`).
- **Exact motor/driver part numbers** — depends on the wheel-radius and
  gearing choice made once a machine is chosen; the RPM range in section
  1 is the number that constrains that choice, not a specific motor.
- **Budget.** Tier A vs. Tier B in section 5 alone represents a large
  cost spread; no attempt is made here to cost out either, since neither
  has been priced against real supplier quotes.

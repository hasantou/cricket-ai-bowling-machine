# Serial machine protocol

The contract `SerialMachineController` (`machine_control/serial_controller.py`)
speaks. Any microcontroller firmware that implements this — an Arduino, an
ESP32, a Raspberry Pi Pico wired to the actual wheel motors and an
emergency-stop relay — works with this software unchanged, regardless of
which physical bowling machine it ends up driving. This is deliberately
the simplest protocol that covers what `MachineController` needs: plain
ASCII, one line per command, one line per response, nothing binary and
nothing stateful beyond "ready / busy / stopped."

This has never been run against a real microcontroller — there is no
firmware yet, because there is no chosen machine yet (see the root
`ROADMAP.md`). Writing this down now, and building `SerialMachineController`
against it, means a firmware author has a concrete, already-tested target
to implement rather than a protocol invented at the same time as the
firmware.

## Transport

- Serial, 8N1, default baud rate 115200 (configurable — both sides just
  need to agree).
- Every command and response is one line, terminated by `\n`.
- The host (this software) always speaks first; the firmware never sends
  a line unprompted.
- One command in flight at a time — the host waits for a response line
  before sending the next command.

## Commands (host → machine)

| Command | Meaning |
|---|---|
| `PING` | Liveness/readiness check. No side effect. |
| `SET <rpm1> <rpm2>` | Spin wheel 1 and wheel 2 at the given surface speeds (decimal RPM, e.g. `SET 1450.3 1602.8`) and release the next ball once at speed. |
| `STOP` | Emergency stop — halt the mechanism immediately, regardless of what it's doing. Must be safe to send at any time, including when already stopped. |
| `RESET` | Clear an emergency-stopped state. The firmware should only answer `OK` once it has confirmed the machine is actually mechanically safe to resume (e.g. homed, no fault latched) — never just because time has passed. |

## Responses (machine → host)

One line, always sent in reply to the command just received:

| Response | Meaning | Valid after |
|---|---|---|
| `READY` | Idle and able to accept `SET` right now. | `PING` |
| `BUSY` | Still cycling from the previous ball; not ready yet. | `PING`, `SET` |
| `STOPPED` | Currently emergency-stopped. | `PING` |
| `OK` | Command accepted. | `SET`, `STOP`, `RESET` |
| `ERR <reason>` | Command rejected; `<reason>` is free-text for logs, not parsed by the host. | any |

If no response line arrives before the host's configured timeout
(default 2 seconds), the host treats that as a communication failure —
`is_ready()` returns `False`, `set_delivery()` and `reset_after_stop()`
raise, and `emergency_stop()` — which must never raise — logs the failure
locally and otherwise does nothing further, on the assumption that a dead
serial link means the physical estop button is the only thing left that
can actually be trusted.

## Example exchange

```
host:     PING
firmware: READY
host:     SET 1450.3 1602.8
firmware: OK
host:     PING
firmware: BUSY
...
host:     PING
firmware: READY
host:     STOP
firmware: OK
host:     PING
firmware: STOPPED
host:     RESET
firmware: OK
```

## Appendix: crease-plane sensor (a separate device)

The wide / head-height sensor is its own board on its own serial port, not part of the
machine's command channel above. Same conventions: 8N1, plain ASCII, one line per message,
host speaks first. `machine_control/crease_sensor.py` (`parse_crease_line`) is the reference parser.

| Host sends | Sensor answers | Meaning |
|---|---|---|
| `CREASE? <plane>` | `CREASE <plane> <t_us> <y_mm> <z_mm>` | The last delivery's crossing of `<plane>` |
| `CREASE? <plane>` | `NONE` | Nothing has crossed that plane |
| any | `ERR <reason>` | Rejected; free text for logs |

- `<plane>` is `wicket` (the striker's stumps, where a wide is judged - MCC Law 22.2) or
  `popping` (the popping crease, 1.22 m in front of the stumps, where head height is judged -
  ICC 22.1.1.2).
- `<t_us>` is integer microseconds on the sensor's own clock; `<y_mm>` is the ball's lateral
  position, signed, **positive toward the off side of a right-handed batter**, measured from the
  line through the middle stump; `<z_mm>` is its height above the playing surface. Integers only.
- The host rejects any line that is malformed, has the wrong number of fields, or is outside a
  physical range (|y| > 6 m, z outside -5 cm..6 m). It never guesses at a garbled line.
- Required accuracy: about **+/-2 cm lateral and +/-3 cm height**, because the rules treat a
  reading within 3 cm of a limit as borderline (`laws.WideRules.tolerance_m`).
- Time alignment with the camera and the other sensors is done in software
  (`trajectory-engine/cricket_trajectory/delivery_sync.py`); the sensor needs only a monotonic clock.

Never run against real hardware: there is no board yet.

## What this protocol deliberately does not cover

- **Calibration.** `RPM → actual ball speed/spin` is currently a purely
  theoretical mapping (`trajectory-engine/cricket_trajectory/machine.py`).
  Once real hardware exists, that mapping needs measuring against real
  radar-gun or high-speed-camera data — this protocol only carries the
  RPM values `machine.py` computes, it doesn't validate them.
- **Which physical machine.** This is intentionally generic — it says
  nothing about wheel diameter, motor type, or mechanical design, because
  no machine has been chosen yet.
- **Telemetry beyond ready/busy/stopped.** No temperature, no motor
  current, no fault codes beyond the free-text `ERR` reason. A real
  deployment likely wants more; add fields as new commands/responses
  once there's real hardware to motivate the exact shape, rather than
  guessing now.

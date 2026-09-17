"""
A real MachineController implementation, ready to flash into firmware —
the piece project notes flagged as blocked on "which machine", not on
software. This doesn't solve that decision; it removes it as a blocker
for everything else. Any microcontroller (Arduino, ESP32, a Raspberry Pi
Pico) that implements the line protocol documented in
`machine-control/PROTOCOL.md` works with this class unchanged, regardless
of which physical machine it ends up driving.

Tested here without any real hardware by injecting a fake serial
connection (see tests/test_serial_controller.py) that speaks the same
protocol — this class's own logic (encode a command, wait for a
response, decide what that response means) is exercised for real; only
the physical link underneath is faked, exactly the same boundary
SimulatedMachineController stands in for at a higher level.
"""

from __future__ import annotations

import time
from typing import Callable, List, Optional

from .controller import CommandedDelivery, MachineController


class SerialCommunicationError(RuntimeError):
    """The serial link didn't behave as the protocol requires (timeout,
    garbled response, or the port itself is unavailable)."""


def _default_connection_factory(port: str, baudrate: int, timeout_s: float):
    try:
        import serial as pyserial
    except ImportError as e:
        raise SerialCommunicationError(
            "pyserial is required to talk to a real machine over serial "
            "(it's in requirements.txt — run `pip install -r requirements.txt`)."
        ) from e
    return pyserial.Serial(port=port, baudrate=baudrate, timeout=timeout_s)


class SerialMachineController(MachineController):
    """Drives a real machine over a serial link using the line protocol in
    `PROTOCOL.md`: one ASCII command per line out, one ASCII response line
    back, always. `connection` is dependency-injected (a real
    `serial.Serial` by default, or a fake for tests) so this class's own
    protocol logic is what's actually under test, not a physical port.
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        timeout_s: float = 2.0,
        connection_factory: Optional[Callable] = None,
    ):
        factory = connection_factory or _default_connection_factory
        self._conn = factory(port, baudrate, timeout_s)
        self.port = port
        self.history: List[CommandedDelivery] = []

    def _send(self, command: str) -> str:
        """Writes one command line and reads back exactly one response
        line. Returns the response with whitespace stripped, or raises
        SerialCommunicationError if nothing came back before the
        connection's own timeout."""
        self._conn.write((command + "\n").encode("ascii"))
        raw = self._conn.readline()
        if not raw:
            raise SerialCommunicationError(
                f"No response from machine within the configured timeout "
                f"(sent {command!r})."
            )
        return raw.decode("ascii", errors="replace").strip()

    def set_delivery(self, wheel1_rpm: float, wheel2_rpm: float) -> None:
        response = self._send(f"SET {wheel1_rpm:.1f} {wheel2_rpm:.1f}")
        self.history.append(CommandedDelivery(wheel1_rpm, wheel2_rpm, timestamp=time.monotonic()))
        if response == "OK":
            return
        if response == "BUSY":
            raise RuntimeError("Machine reported BUSY — previous cycle hasn't finished yet.")
        raise SerialCommunicationError(f"Machine refused SET: {response!r}")

    def emergency_stop(self) -> None:
        # Contract: must never raise, even if the link itself is dead —
        # an exception here must not be what stops a human from also
        # hitting the machine's own physical stop button.
        self.history.append(CommandedDelivery(0.0, 0.0, timestamp=time.monotonic(), stopped=True))
        try:
            self._send("STOP")
        except SerialCommunicationError:
            pass

    def is_ready(self) -> bool:
        # Fail safe: any communication problem reads as "not ready",
        # never as "assume it's fine."
        try:
            return self._send("PING") == "READY"
        except SerialCommunicationError:
            return False

    def reset_after_stop(self) -> None:
        response = self._send("RESET")
        if response != "OK":
            raise SerialCommunicationError(
                f"Machine refused RESET (stayed stopped): {response!r}"
            )

    def close(self) -> None:
        self._conn.close()

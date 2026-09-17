import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from machine_control.safety import SafeMachineController, SafetyLimits
from machine_control.serial_controller import SerialCommunicationError, SerialMachineController


class FakeSerialConnection:
    """Stands in for a real `serial.Serial` — a real microcontroller
    running the PROTOCOL.md firmware is what this fake is rehearsing
    against, so SerialMachineController's own protocol logic (what it
    sends, how it interprets what comes back) is genuinely exercised here,
    only the physical link itself is faked."""

    def __init__(self, responses):
        # queue of raw bytes readline() should hand back, one per call;
        # b"" reproduces what pyserial returns on a real timeout.
        self._responses = list(responses)
        self.sent = []
        self.closed = False

    def write(self, data: bytes):
        self.sent.append(data)

    def readline(self) -> bytes:
        if not self._responses:
            return b""
        return self._responses.pop(0)

    def close(self):
        self.closed = True


def make_serial(responses):
    fake = FakeSerialConnection(responses)
    controller = SerialMachineController(
        port="FAKE0", connection_factory=lambda port, baud, timeout: fake
    )
    return controller, fake


def test_set_delivery_sends_the_documented_command_line():
    controller, fake = make_serial([b"OK\n"])
    controller.set_delivery(1450.3, 1602.8)
    assert fake.sent == [b"SET 1450.3 1602.8\n"]


def test_set_delivery_accepts_ok_and_records_history():
    controller, fake = make_serial([b"OK\n"])
    controller.set_delivery(2000.0, 2100.0)
    assert len(controller.history) == 1
    assert controller.history[0].wheel1_rpm == 2000.0
    assert controller.history[0].stopped is False


def test_set_delivery_raises_runtime_error_on_busy():
    controller, fake = make_serial([b"BUSY\n"])
    with pytest.raises(RuntimeError):
        controller.set_delivery(2000.0, 2000.0)


def test_set_delivery_raises_on_unexpected_response():
    controller, fake = make_serial([b"ERR bad checksum\n"])
    with pytest.raises(SerialCommunicationError):
        controller.set_delivery(2000.0, 2000.0)


def test_set_delivery_raises_on_timeout():
    controller, fake = make_serial([b""])  # empty line == pyserial timeout
    with pytest.raises(SerialCommunicationError):
        controller.set_delivery(2000.0, 2000.0)


def test_is_ready_true_when_machine_reports_ready():
    controller, fake = make_serial([b"READY\n"])
    assert controller.is_ready() is True
    assert fake.sent == [b"PING\n"]


@pytest.mark.parametrize("response", [b"BUSY\n", b"STOPPED\n"])
def test_is_ready_false_on_busy_or_stopped(response):
    controller, fake = make_serial([response])
    assert controller.is_ready() is False


def test_is_ready_fails_safe_on_communication_timeout():
    controller, fake = make_serial([b""])
    assert controller.is_ready() is False  # never raises - fails safe as "not ready"


def test_emergency_stop_never_raises_even_on_dead_link():
    controller, fake = make_serial([b""])  # simulated dead link, no response at all
    controller.emergency_stop()  # must not raise
    assert len(controller.history) == 1
    assert controller.history[0].stopped is True


def test_emergency_stop_sends_stop_command():
    controller, fake = make_serial([b"OK\n"])
    controller.emergency_stop()
    assert fake.sent == [b"STOP\n"]


def test_reset_after_stop_succeeds_on_ok():
    controller, fake = make_serial([b"OK\n"])
    controller.reset_after_stop()  # should not raise


def test_reset_after_stop_raises_when_machine_refuses():
    controller, fake = make_serial([b"ERR not homed\n"])
    with pytest.raises(SerialCommunicationError):
        controller.reset_after_stop()


def test_default_factory_gives_a_clear_error_without_pyserial(monkeypatch):
    monkeypatch.setitem(sys.modules, "serial", None)  # simulate pyserial not installed
    with pytest.raises(SerialCommunicationError, match="pyserial"):
        SerialMachineController(port="FAKE0")


def test_close_closes_the_underlying_connection():
    controller, fake = make_serial([])
    controller.close()
    assert fake.closed is True


def test_wrapped_in_safe_machine_controller_end_to_end():
    """The whole point of the MachineController abstraction: SafeMachineController
    doesn't need to know or care that this is a serial link instead of the
    simulated one — same safety checks, same kill-switch behavior."""
    controller, fake = make_serial([b"OK\n", b"READY\n", b"OK\n", b"OK\n", b"OK\n"])
    safe = SafeMachineController(controller, SafetyLimits(max_wheel_rpm=6000.0))

    safe.set_delivery(2000.0, 2100.0)
    assert safe.is_ready() is True

    safe.emergency_stop()
    assert safe.enabled is False
    # is_ready() short-circuits on the safety layer's own flag - never even
    # asks the (now stopped) machine, so no serial round-trip is spent here.
    assert safe.is_ready() is False

    safe.reset()
    assert safe.enabled is True
    safe.set_delivery(1800.0, 1900.0)
    assert len(controller.history) == 3  # SET, STOP, SET

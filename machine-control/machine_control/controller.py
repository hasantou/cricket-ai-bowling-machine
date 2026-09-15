"""
The hardware boundary — one abstract interface every real machine driver
implements, so nothing above this line (the orchestrator, safety layer,
decision engines) needs to know or care which physical machine is
actually connected.

Deliberately minimal: three methods, because that's genuinely all the
software above this needs from hardware. A real driver for a specific
machine (BOLA, ProBatter, a custom rig) implements MachineController by
talking to whatever interface that machine actually exposes (serial,
Bluetooth, a network API, raw GPIO) — that implementation doesn't exist
yet and can't, without knowing which machine this targets (see project
notes — "which machine" is a decision blocking this, not a queued task).
What CAN be built now, and is: the contract itself, a simulated
implementation to test everything else against, and everything that sits
on top of this interface without needing real hardware to be exercised.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple


class MachineController(ABC):
    """The only thing the rest of this software knows about a physical
    bowling machine. A real driver for a specific machine implements
    this; nothing else changes when the hardware does."""

    @abstractmethod
    def set_delivery(self, wheel1_rpm: float, wheel2_rpm: float) -> None:
        """Command the machine to spin its wheels at the given surface
        speeds for the next ball. Should raise if the machine can't
        currently accept a new command (see is_ready())."""

    @abstractmethod
    def emergency_stop(self) -> None:
        """Immediately halt the machine, regardless of what it's doing.
        Must be safe to call at any time, including when already
        stopped, and must never raise."""

    @abstractmethod
    def is_ready(self) -> bool:
        """Whether the machine is ready to accept the next set_delivery()
        call — e.g. the previous ball's mechanism cycle has finished."""

    @abstractmethod
    def reset_after_stop(self) -> None:
        """Clear an emergency-stopped state and resume normal operation.
        Part of the contract, not just SimulatedMachineController's own
        detail — any real driver needs an explicit "resume" distinct from
        is_ready(), since nothing should resume just because time has
        passed since a stop."""


@dataclass
class CommandedDelivery:
    """One record of a command actually sent to (or attempted on) a
    controller — the audit trail a real deployment would want, and what
    tests check against."""
    wheel1_rpm: float
    wheel2_rpm: float
    timestamp: float
    stopped: bool = False  # True if this entry is an emergency_stop(), not a delivery


class SimulatedMachineController(MachineController):
    """No real hardware — records what it was told to do and simulates a
    simple mechanical cycle time, so the orchestrator loop above it can be
    built and tested end-to-end before any real machine exists. Swapping
    this for a real driver later means implementing the same three
    methods against real hardware; nothing that uses this class changes.
    """

    def __init__(self, cycle_time_s: float = 0.05):
        # Deliberately tiny default (not a real inter-delivery gap) so
        # automated tests run fast; a real driver's is_ready() would
        # reflect the machine's actual mechanical reset time instead.
        self.cycle_time_s = cycle_time_s
        self.history: List[CommandedDelivery] = []
        self._busy_until: float = 0.0
        self._stopped = False

    def set_delivery(self, wheel1_rpm: float, wheel2_rpm: float) -> None:
        if not self.is_ready():
            raise RuntimeError(
                "SimulatedMachineController is not ready for a new delivery yet "
                "(previous cycle still in progress, or emergency-stopped)."
            )
        now = time.monotonic()
        self.history.append(CommandedDelivery(wheel1_rpm, wheel2_rpm, timestamp=now))
        self._busy_until = now + self.cycle_time_s
        self._stopped = False

    def emergency_stop(self) -> None:
        self._stopped = True
        self._busy_until = 0.0
        self.history.append(CommandedDelivery(0.0, 0.0, timestamp=time.monotonic(), stopped=True))

    def is_ready(self) -> bool:
        if self._stopped:
            return False
        return time.monotonic() >= self._busy_until

    def reset_after_stop(self) -> None:
        """A human clearing an emergency stop and confirming it's safe to
        resume — deliberately a separate, explicit call from is_ready(),
        so nothing resumes automatically just because the cycle timer
        would otherwise have elapsed."""
        self._stopped = False
        self._busy_until = 0.0

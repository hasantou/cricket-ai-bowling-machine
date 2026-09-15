"""
The safety layer every machine command passes through before it reaches
real hardware — built now, independent of which machine eventually gets
connected, because "a machine that changes pace on its own with no one
confirming is a safety question, not just a software one" was flagged as
a real, unbuilt gap. This closes the software half of that; the physical
half (a real emergency-stop button, mechanical limits) is hardware work
this can't do.

Design choices, stated rather than left implicit:
  - Violations RAISE rather than silently clamp. A wildly-wrong RPM
    reaching here is more likely a bug upstream than a legitimate edge
    case — surfacing it loudly matches the "no silent fake success"
    pattern already used throughout this project (pose_estimation.py,
    neural_scorer.py, delivery_segmentation.py all fail loudly rather
    than guess), and silently capping a bad command could hide the
    actual bug instead of stopping it.
  - The kill-switch is MORE persistent than the underlying controller's
    own busy/ready cycle. emergency_stop() disables this layer outright;
    resuming needs an explicit, deliberate reset() call — never just
    "enough time passed since the stop."
"""

from __future__ import annotations

from dataclasses import dataclass

from .controller import MachineController


class SafetyViolation(Exception):
    """A commanded delivery fell outside configured hard limits, or a
    command was attempted while this layer is disabled (e.g. after an
    emergency stop nobody has cleared yet)."""


@dataclass(frozen=True)
class SafetyLimits:
    """Hard limits on what this software will EVER send to a machine,
    regardless of what any decision engine computes. Named, adjustable,
    not hidden — set these to the real machine's documented safe
    operating range once one is chosen; these defaults are a
    conservative placeholder, not a calibrated value."""
    min_wheel_rpm: float = 0.0
    max_wheel_rpm: float = 5000.0        # placeholder ceiling - real machine spec should replace this
    max_rpm_differential: float = 3000.0  # caps commanded spin indirectly (see machine.py's v1-v2 relation)


class SafeMachineController:
    """Wraps any MachineController (simulated or real) and validates every
    command against SafetyLimits before forwarding it. Nothing above this
    layer — the orchestrator, either decision engine — ever talks to a
    raw MachineController directly; this is the only path a command takes
    to reach hardware.
    """

    def __init__(self, inner: MachineController, limits: SafetyLimits = None):
        self._inner = inner
        self.limits = limits or SafetyLimits()
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_delivery(self, wheel1_rpm: float, wheel2_rpm: float) -> None:
        if not self._enabled:
            raise SafetyViolation(
                "Machine is disabled (emergency-stopped or never enabled) — "
                "call reset() after confirming it's safe to resume."
            )
        for name, rpm in (("wheel1_rpm", wheel1_rpm), ("wheel2_rpm", wheel2_rpm)):
            if not (self.limits.min_wheel_rpm <= rpm <= self.limits.max_wheel_rpm):
                raise SafetyViolation(
                    f"{name}={rpm:.0f} is outside the configured safe range "
                    f"[{self.limits.min_wheel_rpm:.0f}, {self.limits.max_wheel_rpm:.0f}] — refusing to send it."
                )
        differential = abs(wheel1_rpm - wheel2_rpm)
        if differential > self.limits.max_rpm_differential:
            raise SafetyViolation(
                f"wheel RPM differential {differential:.0f} exceeds the configured "
                f"max {self.limits.max_rpm_differential:.0f} (this drives commanded spin) — refusing to send it."
            )
        self._inner.set_delivery(wheel1_rpm, wheel2_rpm)

    def emergency_stop(self) -> None:
        self._enabled = False
        self._inner.emergency_stop()

    def reset(self) -> None:
        """A human explicitly confirming it's safe to resume after a stop.
        Deliberately separate from is_ready() — nothing here resumes on
        its own just because time has passed. Clears both this layer's
        own disabled flag AND the inner controller's stopped state —
        resetting only one and not the other would leave the machine
        silently stuck even though this layer reports itself enabled."""
        self._inner.reset_after_stop()
        self._enabled = True

    def is_ready(self) -> bool:
        return self._enabled and self._inner.is_ready()

from __future__ import annotations

import time
from collections import deque
from enum import Enum
from typing import Deque, Dict, Optional, Tuple


class LinkState(str, Enum):
    """High-level state of the serial telemetry link."""

    IDLE = "idle"
    CONNECTING = "connecting"
    OPEN = "open"
    SYNCING = "syncing"
    HEALTHY = "healthy"
    STALE = "stale"
    ERROR = "error"
    CLOSED = "closed"


class LinkMonitor:
    """
    Lightweight helper that tracks link health and computes telemetry stats.

    The serial worker feeds connection events through :meth:`transition` and
    notifies each received line via :meth:`record_rx`.  Consumers can poll
    :meth:`evaluate` to detect stale periods without sprinkling timing logic
    across the UI layer.
    """

    def __init__(
        self,
        expected_rx_hz: float = 5.0,
        warmup_s: float = 1.2,
        stale_s: float = 2.5,
        history: int = 64,
    ) -> None:
        self.expected_rx_hz = max(0.1, float(expected_rx_hz))
        self.warmup_s = max(0.0, float(warmup_s))
        self.stale_s = max(0.1, float(stale_s))
        self._rx_times: Deque[float] = deque(maxlen=max(2, history))
        self._state: LinkState = LinkState.IDLE
        self._last_change: float = time.monotonic()
        self._open_time: Optional[float] = None
        self._last_rx: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    @property
    def state(self) -> LinkState:
        return self._state

    def reset(self) -> None:
        """Return the monitor to IDLE state and clear timing history."""
        self._rx_times.clear()
        self._state = LinkState.IDLE
        self._last_change = time.monotonic()
        self._open_time = None
        self._last_rx = None

    def transition(self, state: LinkState, *, timestamp: Optional[float] = None) -> bool:
        """
        Record a state transition originating from the worker.

        Returns True if the state actually changed.
        """
        now = timestamp or time.monotonic()
        if state == self._state:
            return False
        self._state = state
        self._last_change = now
        if state == LinkState.OPEN:
            self._open_time = now
        if state in (LinkState.CLOSED, LinkState.ERROR, LinkState.IDLE):
            self._rx_times.clear()
            self._last_rx = None
        return True

    def record_rx(self, *, timestamp: Optional[float] = None) -> Tuple[bool, Dict[str, Optional[float]]]:
        """
        Register an incoming telemetry line and update health stats.

        Returns (state_changed, info).
        """
        now = timestamp or time.monotonic()
        self._rx_times.append(now)
        self._last_rx = now
        changed = self.transition(LinkState.HEALTHY, timestamp=now)
        return changed, self.snapshot(now)

    def evaluate(self, *, timestamp: Optional[float] = None) -> Tuple[bool, LinkState, Dict[str, Optional[float]]]:
        """
        Inspect recent activity and emit STALE when the stream pauses.

        Returns a tuple of (state_changed, current_state, info).
        """
        now = timestamp or time.monotonic()
        info = self.snapshot(now)

        if self._state == LinkState.HEALTHY and self._last_rx is not None:
            if now - self._last_rx >= self.stale_s:
                self.transition(LinkState.STALE, timestamp=now)
                info = self.snapshot(now)
                return True, LinkState.STALE, info

        if self._state == LinkState.SYNCING and self._open_time is not None:
            if now - self._open_time >= max(self.warmup_s, self.stale_s):
                self.transition(LinkState.STALE, timestamp=now)
                info = self.snapshot(now)
                return True, LinkState.STALE, info

        return False, self._state, info

    def snapshot(self, timestamp: Optional[float] = None) -> Dict[str, Optional[float]]:
        """
        Return a dictionary of telemetry metrics (rx rate, age, warmup time).
        """
        now = timestamp or time.monotonic()
        rx_hz: Optional[float] = None
        if len(self._rx_times) >= 2:
            times = list(self._rx_times)
            intervals = [t2 - t1 for t1, t2 in zip(times, times[1:]) if t2 > t1]
            if intervals:
                avg = sum(intervals) / len(intervals)
                if avg > 0:
                    rx_hz = 1.0 / avg

        age = (now - self._last_rx) if self._last_rx is not None else None
        warmup = (now - self._open_time) if self._open_time is not None else None
        elapsed = now - self._last_change if self._last_change else None

        return {
            "rx_hz": rx_hz,
            "age": age,
            "warmup": warmup,
            "elapsed": elapsed,
            "state": self._state.value,
        }

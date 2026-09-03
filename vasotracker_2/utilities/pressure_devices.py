# #################################################
# # VasoTracker 2 - Blood Vessel Diameter Measurement Software
# #
# # Author: Calum Wilson, Matthew D Lee, and Chris Osborne
# # License: BSD 3-Clause License (See main file for details)
# # Website: www.vasostracker.com
# #
# #################################################

# Vendor-neutral pressure device abstractions. These classes allow
# VasoTracker to support multiple hardware backends (Arduino VasoMoto,
# NI-DAQ, software simulation, or a null device) without coupling the
# UI or data logging logic to any particular vendor API.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple
import re
import threading
import time

from .arduino_async_worker import ArduinoSerialWorker


class PressureDevice(Protocol):
    """Interface implemented by every concrete pressure backend."""

    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def set_pressure(self, value_mmHg: float) -> None:
        ...

    def read_latest(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Return (p1_mmHg, p2_mmHg, setpoint_mmHg)."""


@dataclass
class NullPressureDevice:
    """No-op device used when no hardware is selected."""

    def __post_init__(self) -> None:
        self._latest: Tuple[Optional[float], Optional[float], Optional[float]] = (
            None,
            None,
            None,
        )

    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def set_pressure(self, value_mmHg: float) -> None:
        self._latest = (self._latest[0], self._latest[1], float(value_mmHg))

    def read_latest(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        return self._latest


class SimPressureDevice:
    """Simple first-order plant that drifts toward setpoint (for offline tests)."""

    def __init__(self, update_hz: float = 20.0, tau_s: float = 0.8) -> None:
        self._sp = 0.0
        self._p1 = 0.0
        self._p2 = 0.0
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._dt = 1.0 / max(1.0, update_hz)
        self._tau = max(1e-3, tau_s)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def set_pressure(self, value_mmHg: float) -> None:
        self._sp = max(0.0, float(value_mmHg))

    def _run(self) -> None:
        k = self._dt / self._tau
        while not self._stop_evt.is_set():
            self._p1 += k * (self._sp - self._p1)
            # Small mismatch to avoid perfect symmetry
            self._p2 += k * (self._sp - self._p2 * 0.98)
            time.sleep(self._dt)

    def read_latest(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        return (round(self._p1, 1), round(self._p2, 1), round(self._sp, 1))


class ArduinoPressureDevice:
    """
    Arduino-based implementation that speaks the minimal VasoMoto protocol.

    Serial protocol (ASCII, newline terminated):
      PC -> Arduino: 'PING\\n' | 'START\\n' | 'STOP\\n'
        Setpoint: 'SET:<mmHg>\\n' (modern) and '<mmHg>\\n' (legacy VasoMoto firmware)
      Arduino -> PC: 'P1:<v>,P2:<v>,SET:<v>\\n' (modern) or '<P1:<v>;P2:<v>>' (legacy)
    """

    def __init__(
        self,
        arduino=None,
        stream_rate_hz: float = 20.0,
        worker: Optional[ArduinoSerialWorker] = None,
    ) -> None:
        """
        Parameters
        ----------
        arduino:
            Instance exposing `sendData(text: str) -> None` and
            `readline(timeout: float) -> Optional[str]`.
            The utilities.VT_Arduino.Arduino class satisfies this interface.
        worker:
            Optional asynchronous worker that pushes telemetry lines via
            callbacks. When provided the legacy polling thread is bypassed.
        stream_rate_hz:
            Expected telemetry rate; used to set the poll timeout.
        """

        self.arduino = arduino
        self.worker = worker
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._latest: Tuple[Optional[float], Optional[float], Optional[float]] = (
            None,
            None,
            None,
        )
        self._pattern = re.compile(
            r"P1:(-?\d+\.?\d*),\s*P2:(-?\d+\.?\d*),\s*SET:(-?\d+\.?\d*)"
        )
        self._vm_pattern = re.compile(
            r"^DATA\s+T=(\d+)\s+P=([\d\.\-NaN]+)\s+P_SET=([\d\.\-NaN]+)",
            re.IGNORECASE,
        )
        self._vm_ack_pattern = re.compile(
            r"^ACK\s+SET\s+P=([\d\.\-]+)\s+T=(\d+)",
            re.IGNORECASE,
        )
        self._pattern_legacy = re.compile(
            r"<\s*P1:(-?\d+\.?\d*)\s*[,;]\s*P2:(-?\d+\.?\d*)\s*(?:[,;]\s*SET:(-?\d+\.?\d*))?\s*>"
        )
        self._last_sent_set_mmHg: Optional[float] = None
        self._interval = 1.0 / max(1.0, stream_rate_hz)
        self._keepalive_interval_s = 0.8
        self._keepalive_evt = threading.Event()
        self._keepalive_thread: Optional[threading.Thread] = None
        self._vm_last_device_time: Optional[float] = None
        self._last_command_ts: float = 0.0
        self._awaiting_app_setpoint: bool = False
        self._manual_override: bool = False
        self._manual_override_timeout_s: float = 2.5  # Increased for smoother knob operation
        self._last_broadcasted_setpoint: Optional[float] = None

    def bind_worker(self, worker: ArduinoSerialWorker) -> None:
        """Attach an async serial worker after construction."""
        self.worker = worker

    def start(self) -> None:
        if self.worker is not None:
            self.worker.start()
            self._queue_command("PING")
            self._queue_command("START")
            self._start_keepalive()
            return

        if not getattr(self.arduino, "is_connected", False):
            return

        self._stop_evt.clear()
        self._queue_command("PING")
        self._queue_command("START")
        self._start_keepalive()

        if self._thread and self._thread.is_alive():
            return

        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self.worker is not None:
            self._queue_command("STOP")
            self._stop_keepalive()
            return

        self._stop_evt.set()
        self._queue_command("STOP")
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._stop_keepalive()

    def set_pressure(self, value_mmHg: float) -> bool:
        v = max(0.0, float(value_mmHg))
        self._last_sent_set_mmHg = v
        self._last_command_ts = time.monotonic()
        self._awaiting_app_setpoint = True

        # Track if we're exiting manual override mode
        was_manual = self._manual_override
        self._manual_override = False

        # If transitioning from manual to app control, could notify UI
        if was_manual:
            # App has taken control back from manual knob
            pass
        if self.worker is not None:
            # VasoMoto's firmware protocol is unambiguous.  Sending a burst of
            # legacy variants can overflow or confuse its line parser, so issue
            # one authoritative command and wait for its ACK/telemetry echo.
            return self._queue_command(f"SET P={v:.1f}")

        if not getattr(self.arduino, "is_connected", False):
            return False

        try:
            self.arduino.sendData(f"SET P={v:.1f}\n")
            return True
        except Exception:
            return False

    def _start_keepalive(self) -> None:
        if self._keepalive_thread and self._keepalive_thread.is_alive():
            return
        self._keepalive_evt.clear()
        self._keepalive_thread = threading.Thread(target=self._keepalive_loop, daemon=True)
        self._keepalive_thread.start()

    def _stop_keepalive(self) -> None:
        self._keepalive_evt.set()
        thread = self._keepalive_thread
        if thread and thread.is_alive():
            thread.join(timeout=0.5)
        self._keepalive_thread = None

    def _keepalive_loop(self) -> None:
        while not self._keepalive_evt.wait(self._keepalive_interval_s):
            value = self._last_sent_set_mmHg
            if value is None or self._manual_override:
                continue
            last_cmd_age = (
                time.monotonic() - self._last_command_ts
                if self._last_command_ts > 0.0
                else float("inf")
            )
            # After a short idle period, back off so the physical knob can take over
            if last_cmd_age > self._manual_override_timeout_s:
                continue
            if self.worker is None and not getattr(self.arduino, "is_connected", False):
                continue
            self._queue_command(f"SET P={value:.1f}")

    def _reader_loop(self) -> None:
        while not self._stop_evt.is_set():
            if not getattr(self.arduino, "is_connected", False):
                time.sleep(self._interval)
                continue
            try:
                line = self.arduino.readline(timeout=self._interval)
            except Exception:
                line = None
            if not line:
                continue
            self._process_line(line.strip())

    def read_latest(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        return self._latest

    def is_manual_override_active(self) -> bool:
        """
        Check if manual override mode is currently active.

        Returns True when the physical knob is being used and the app has
        paused sending keepalive commands to avoid fighting with manual adjustments.
        """
        return self._manual_override

    def get_control_status(self) -> dict:
        """
        Get detailed status about current control state.

        Returns a dictionary with:
        - manual_override: bool - Whether knob control is active
        - last_setpoint: float or None - Last known setpoint
        - time_since_command: float - Seconds since last app command
        """
        now = time.monotonic()
        return {
            "manual_override": self._manual_override,
            "last_setpoint": self._last_sent_set_mmHg,
            "time_since_command": now - self._last_command_ts if self._last_command_ts > 0 else float('inf'),
        }

    def handle_line(self, line: str) -> None:
        """Entry point used by the async worker to feed telemetry."""
        self._process_line(line.strip())

    # Internal helpers -------------------------------------------------------
    def _process_line(self, s: str) -> None:
        if not s:
            return
        if vm_match := self._vm_pattern.match(s):
            _, p_raw, sp_raw = vm_match.groups()
            p = self._safe_float(p_raw)
            sp = self._safe_float(sp_raw)
            self._latest = (p, None, sp)
            self._maybe_detect_manual_override(sp)
            return
        if ack_match := self._vm_ack_pattern.match(s):
            sp = self._safe_float(ack_match.group(1))
            if sp is None or (isinstance(sp, float) and sp != sp):
                return
            self._latest = (self._latest[0], self._latest[1], sp)
            if (
                self._last_sent_set_mmHg is not None
                and abs(sp - self._last_sent_set_mmHg) <= 0.25
            ):
                self._awaiting_app_setpoint = False
                self._manual_override = False
            self._last_sent_set_mmHg = sp
            self._maybe_detect_manual_override(sp)
            print(f"[VasoMoto] Set pressure acknowledged: {sp:.1f} mmHg")
            return

        match = self._pattern.search(s)
        match_legacy = None
        sp: Optional[float] = None
        if match:
            p1 = float(match.group(1))
            p2 = float(match.group(2))
            sp = float(match.group(3))
        else:
            match_legacy = self._pattern_legacy.search(s)
            if match_legacy:
                p1 = float(match_legacy.group(1))
                p2 = float(match_legacy.group(2))
                g3 = match_legacy.group(3)
                sp = float(g3) if g3 is not None else None
        if match or match_legacy:
            if sp is None:
                previous = self._latest[2]
                sp = previous if previous is not None else self._last_sent_set_mmHg
            self._latest = (p1, p2, sp)
            self._maybe_detect_manual_override(sp)

    @staticmethod
    def _safe_float(value: Optional[str]) -> Optional[float]:
        if value is None:
            return None
        raw = value.strip()
        if raw.lower() == "nan":
            return float("nan")
        try:
            return float(raw)
        except ValueError:
            return None

    def _queue_command(self, text: str) -> bool:
        payload = text if text.endswith("\n") else f"{text}\n"
        if self.worker is not None:
            return self.worker.write(payload, ensure_newline=False)
        elif self.arduino is not None:
            try:
                self.arduino.sendData(payload)
                return True
            except Exception:
                return False
        return False

    def _maybe_detect_manual_override(self, setpoint: Optional[float]) -> None:
        """
        Detect when the device setpoint changes without a recent app command and pause keepalives.

        When manual override is detected (e.g., physical knob adjustment), broadcasts the new
        setpoint to update the UI in real-time, creating seamless bidirectional synchronization.
        """
        if setpoint is None:
            return
        if isinstance(setpoint, float) and setpoint != setpoint:
            return
        now = time.monotonic()
        last_cmd_age = (
            now - self._last_command_ts if self._last_command_ts > 0.0 else float("inf")
        )
        delta = None if self._last_sent_set_mmHg is None else abs(setpoint - self._last_sent_set_mmHg)

        if self._awaiting_app_setpoint:
            if delta is not None and delta <= 0.25:
                self._awaiting_app_setpoint = False
                self._manual_override = False
                return
            # VasoMoto may report the previous P_SET for a few frames while it
            # handles an app request. If no matching echo arrives, eventually
            # release the guard and accept the device as authoritative again.
            if last_cmd_age <= self._manual_override_timeout_s:
                return
            self._awaiting_app_setpoint = False

        # A matching echo is not a new manual adjustment. Preserve the current
        # mode: it may be the steady echo after a knob change detected below.
        if delta is not None and delta <= 0.25:
            return

        # Any meaningful change away from the app-set target counts as a manual override.
        if delta is None or delta > 0.5:
            was_override = self._manual_override
            self._manual_override = True
            self._last_sent_set_mmHg = setpoint

            # Broadcast setpoint change to UI for bidirectional sync
            # Only broadcast if value changed to avoid spamming listeners
            if self._last_broadcasted_setpoint is None or abs(setpoint - self._last_broadcasted_setpoint) > 0.1:
                self._last_broadcasted_setpoint = setpoint
                try:
                    # Import here to avoid circular dependency
                    try:
                        from ..setpoint_bus import broadcast_setpoint
                    except ImportError:
                        from setpoint_bus import broadcast_setpoint

                    # Broadcast with source indicating manual knob control
                    # UI listeners can use this to update sliders, displays, and show indicators
                    broadcast_setpoint(setpoint, source="device_knob")

                    if not was_override:
                        # First detection of manual override - physical knob is now in control
                        # UI can display a visual indicator that manual mode is active
                        # Keepalive commands will be paused until app sends a new command
                        pass
                except Exception:
                    # Fail silently if broadcast isn't available (e.g., during testing)
                    pass


class NIDaqPressureDevice:
    """Thin wrapper around NI-DAQ PyDAQmx tasks (optional backend)."""

    def __init__(self, ai_task=None, ao_task=None, scale: float = 1.0) -> None:
        self._latest: Tuple[Optional[float], Optional[float], Optional[float]] = (
            None,
            None,
            None,
        )
        self._ai = ai_task
        self._ao = ao_task
        self._scale = float(scale)

    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def set_pressure(self, value_mmHg: float) -> None:
        try:
            from PyDAQmx import Task  # noqa: F401
        except Exception as exc:
            raise RuntimeError("PyDAQmx not available for NI-DAQ pressure control") from exc

        if self._ao is None:
            raise RuntimeError("Analog output task not configured for NI-DAQ pressure device")

        volts = float(value_mmHg) * self._scale
        self._ao.WriteAnalogScalarF64(True, 10.0, volts, None)
        self._latest = (self._latest[0], self._latest[1], float(value_mmHg))

    def read_latest(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        # Keeping async reads optional – callers may supply ai_task to refresh.
        return self._latest

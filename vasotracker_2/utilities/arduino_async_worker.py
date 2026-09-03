from __future__ import annotations

import queue
import threading
import time
from typing import Callable, Optional, Sequence, Tuple, Union

try:
    import serial
    from serial.tools import list_ports
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "pyserial is required for VasoMoto communication. Install with `pip install pyserial`."
    ) from exc

from .arduino_link_monitor import LinkMonitor, LinkState

LineCallback = Callable[[str], None]
StatusCallback = Callable[[str, dict], None]


class ArduinoSerialWorker:
    """
    Background worker that opens the serial port without blocking the UI thread.

    The worker owns the serial handle, pushes outbound commands from a thread-safe
    queue, and streams inbound lines to a callback.  Connection status updates
    are emitted for UI components (e.g. link badge) via a second callback.
    """

    DEFAULT_WRITE_QUEUE = 256
    PRIORITY_HINTS = ("vasomoto", "arduino", "ch340", "usb serial", "ftdi", "silicon labs")

    def __init__(
        self,
        port: Optional[str],
        baud: int = 115200,
        *,
        monitor: Optional[LinkMonitor] = None,
        line_callback: Optional[LineCallback] = None,
        status_callback: Optional[StatusCallback] = None,
        read_timeout: float = 0.1,
        write_timeout: float = 0.5,
        warmup_s: float = 1.2,
        handshake_s: float = 0.0,
        retry_delay: float = 1.0,
        auto_reconnect: bool = True,
        write_queue_size: int = DEFAULT_WRITE_QUEUE,
    ) -> None:
        self._requested_port = (port or "").strip() or None
        self.baud = int(baud)
        self.monitor = monitor or LinkMonitor()
        self.line_callback = line_callback
        self.status_callback = status_callback
        self.read_timeout = max(0.01, float(read_timeout))
        self.write_timeout = max(0.05, float(write_timeout))
        self.warmup_s = max(0.0, float(warmup_s))
        self.handshake_s = max(0.0, float(handshake_s))
        self.retry_delay = max(0.1, float(retry_delay))
        self.auto_reconnect = bool(auto_reconnect)
        self._write_queue: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=max(1, write_queue_size))
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._serial: Optional[serial.SerialBase] = None
        self._current_port: Optional[str] = None
        self._last_status: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    @property
    def current_port(self) -> Optional[str]:
        return self._current_port

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, *, wait: bool = True, timeout: Optional[float] = 2.0) -> None:
        self._stop_evt.set()
        try:
            self._write_queue.put_nowait(None)
        except queue.Full:
            pass
        if wait and self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._thread = None

    def write(self, payload: Union[str, bytes], *, ensure_newline: bool = True, timeout: float = 0.1) -> bool:
        """
        Queue a command for asynchronous transmission. Returns False if the
        queue is saturated.
        """
        data = payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)
        if ensure_newline and not data.endswith(b"\n"):
            data += b"\n"
        try:
            self._write_queue.put(data, timeout=max(0.0, timeout))
            return True
        except queue.Full:
            return False

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _emit_status(self, event: str, extra: Optional[dict] = None) -> None:
        now = time.monotonic()
        extra = extra or {}
        monitor_info = {}
        if event not in ("healthy", "stale"):
            state = _EVENT_TO_STATE.get(event)
            if state is not None:
                self.monitor.transition(state, timestamp=now)
        monitor_info = self.monitor.snapshot(now)
        info = {**monitor_info, **extra}

        if event == "healthy":
            info = {**monitor_info, **extra}
        elif event == "stale":
            info = {**monitor_info, **extra}

        if self.status_callback:
            try:
                self.status_callback(event, info)
            except Exception:
                pass
        self._last_status = event

    def _run(self) -> None:
        self.monitor.reset()
        while not self._stop_evt.is_set():
            for port in self._candidate_ports():
                if self._stop_evt.is_set():
                    break
                self._emit_status("connecting", {"port": port})
                try:
                    handle = self._open_serial(port)
                except Exception as exc:  # noqa: BLE001 - surface raw message
                    self._emit_status(
                        "error",
                        {
                            "port": port,
                            "message": str(exc).strip() or exc.__class__.__name__,
                        },
                    )
                    self._wait_with_stop(self.retry_delay)
                    continue

                self._serial = handle
                self._current_port = port
                try:
                    self._session_loop(handle, port)
                finally:
                    try:
                        handle.close()
                    except Exception:
                        pass
                    self._serial = None
                    self._current_port = None
                    self._emit_status("closed", {"port": port})

                if not self.auto_reconnect:
                    self._stop_evt.set()
                    break

            if not self.auto_reconnect or self._stop_evt.is_set():
                break
            self._wait_with_stop(self.retry_delay)

    def _candidate_ports(self) -> Sequence[str]:
        if self._requested_port:
            return [self._requested_port]

        ports = list(list_ports.comports())
        results: list[Tuple[int, str]] = []
        for info in ports:
            description = " ".join(
                part.lower()
                for part in (
                    info.manufacturer or "",
                    info.description or "",
                    info.hwid or "",
                )
                if part
            )
            score = 0
            for hint in self.PRIORITY_HINTS:
                if hint in description:
                    score -= 1
                    break
            results.append((score, info.device))
        if not results:
            return []
        results.sort()
        return [device for _, device in results]

    def _open_serial(self, port: str) -> serial.Serial:
        handle = serial.Serial(
            port,
            baudrate=self.baud,
            timeout=self.read_timeout,
            write_timeout=self.write_timeout,
        )
        try:
            handle.reset_input_buffer()
        except Exception:
            pass
        try:
            handle.reset_output_buffer()
        except Exception:
            pass
        return handle

    def _session_loop(self, handle: serial.Serial, port: str) -> None:
        self._emit_status("open", {"port": port})
        self._emit_status("syncing", {"port": port})

        warmup_deadline = (
            time.monotonic() + self.warmup_s if self.warmup_s > 0 else None
        )
        warmup_notified = False
        handshake_deadline = (
            time.monotonic() + self.handshake_s if self.handshake_s > 0 else None
        )
        handshake_warned = False

        while not self._stop_evt.is_set():
            self._flush_outbox(handle)

            try:
                raw = handle.readline()
            except Exception as exc:  # noqa: BLE001 - propagate up as error
                self._emit_status(
                    "error",
                    {
                        "port": port,
                        "message": str(exc).strip() or exc.__class__.__name__,
                    },
                )
                return

            now = time.monotonic()
            if raw:
                line = raw.decode("utf-8", errors="ignore").strip()
                if line:
                    changed, _ = self.monitor.record_rx(timestamp=now)
                    if self.line_callback:
                        try:
                            self.line_callback(line)
                        except Exception:
                            pass
                    if changed:
                        self._emit_status("healthy", {"port": port})

            if (
                warmup_deadline
                and not warmup_notified
                and now >= warmup_deadline
                and self.monitor.state == LinkState.SYNCING
            ):
                # Still waiting for telemetry after the warm-up window.
                self._emit_status("syncing", {"port": port, "warmup": self.warmup_s})
                warmup_notified = True

            changed, state, info = self.monitor.evaluate(timestamp=now)
            if changed and state == LinkState.STALE:
                self._emit_status("stale", {"port": port, **info})

            if (
                handshake_deadline
                and not handshake_warned
                and now >= handshake_deadline
                and self.monitor.state in (LinkState.SYNCING, LinkState.OPEN)
            ):
                # Promote to stale once, but keep the port open for late telemetry.
                self.monitor.transition(LinkState.STALE, timestamp=now)
                info = self.monitor.snapshot(now)
                info.update({"port": port, "reason": "handshake_timeout", "timeout": self.handshake_s})
                self._emit_status("stale", info)
                handshake_warned = True

        # Graceful shutdown path
        self._flush_outbox(handle)

    def _flush_outbox(self, handle: serial.Serial) -> None:
        wrote_payload = False
        while True:
            try:
                payload = self._write_queue.get_nowait()
            except queue.Empty:
                break
            if payload is None:
                continue
            try:
                handle.write(payload)
                wrote_payload = True
            except Exception:
                continue
        if wrote_payload:
            try:
                handle.flush()
            except Exception:
                pass

    def _wait_with_stop(self, duration: float) -> None:
        deadline = time.monotonic() + duration
        while not self._stop_evt.is_set() and time.monotonic() < deadline:
            time.sleep(0.05)


_EVENT_TO_STATE = {
    "connecting": LinkState.CONNECTING,
    "open": LinkState.OPEN,
    "syncing": LinkState.SYNCING,
    "error": LinkState.ERROR,
    "closed": LinkState.CLOSED,
}

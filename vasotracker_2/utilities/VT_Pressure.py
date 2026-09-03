##################################################
## VasoTracker 2 - Blood Vessel Diameter Measurement Software
##
## Author: Calum Wilson, Matthew D Lee, and Chris Osborne
## License: BSD 3-Clause License (See main file for details)
## Website: www.vasostracker.com
##
##################################################


## We found the following to be useful:
## https://www.safaribooksonline.com/library/view/python-cookbook/0596001673/ch09s07.html
## http://code.activestate.com/recipes/82965-threads-tkinter-and-asynchronous-io/
## https://www.physics.utoronto.ca/~phy326/python/Live_Plot.py
## http://forum.arduino.cc/index.php?topic=225329.msg1810764#msg1810764
## https://stackoverflow.com/questions/9917280/using-draw-in-pil-tkinter
## https://stackoverflow.com/questions/37334106/opening-image-on-canvas-cropping-the-image-and-update-the-canvas

import sys
import os
import time
from datetime import timedelta
import tkinter.messagebox as tmb
from typing import Callable, Optional

from .arduino_async_worker import ArduinoSerialWorker
from .arduino_link_monitor import LinkMonitor
from .pressure_devices import ArduinoPressureDevice

#########################################################################################
# Calum trying to sort out the National Instruments problem....
#########################################################################################
'''
def get_resource_path(relative_path):
    """Get the path to a resource, whether it's bundled with PyInstaller or not."""
    base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
    return os.path.join(base_path, relative_path)



# If running as an exe, then get the included nidaqmax.h file location
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    # running in a PyInstaller bundle
    header_dir = os.path.join(sys._MEIPASS, 'include')
else:
    # running in a normal Python environment
    # This is wrong, but it won't matter because I set the DAQmxConfig.py to check the header_dir last.
    header_dir = os.path.join(os.getcwd(), 'include')
print("header_dir = ", header_dir)

os.environ['NIDAQMX_INCLUDE_PATH'] = header_dir
print("NIDAQMX_INCLUDE_PATH set to:", os.environ['NIDAQMX_INCLUDE_PATH'])

from PyDAQmx.DAQmxConfig import is_pydaqmx_installed
print(f"PyDAQmx installed: {is_pydaqmx_installed()}")

'''
try:
    import PyDAQmx
    from PyDAQmx import *
    pydaqmx_available = True
except:
    pydaqmx_available = False

def is_pydaqmx_available():
    return pydaqmx_available


#########################################################################################
# End of Calum trying to sort out the National Instruments problem....
#########################################################################################






class PressureController:
    def __init__(self, model, view, pydaqmx_available):
        self.model = model
        self.view = view
        self.pydaqmx_available = pydaqmx_available
        self.task = None
        #self.initialize_pressure_system()
        self.start_pressure = None
        self.stop_pressure = None
        self.pressure_interval = None
        self.pressure_time_interval = None
        self.set_pressure = None
        self.pressure_start_time = None
        self.multiplier = 1
        self.last_update_time = None
        self.update_threshold = 1  # Minimum time interval in seconds between updates
        self.protocol_completed = False
        self.completed = False
        self.next_pressure_update_time = 0

        # VasoMoto is deliberately a headless backend.  The official pressure
        # panel remains unchanged; its existing power button starts/stops this
        # connection and all existing protocol/manual controls route through it.
        self._vasomoto_worker = None
        self._vasomoto_device = None
        self._vasomoto_monitor = None
        self._vasomoto_state = "disconnected"
        self._active_backend = None
        self._connection_status_callback: Optional[Callable[[str, dict], None]] = None
        self.measured_pressure_1 = None
        self.measured_pressure_2 = None
        self.measured_pressure_avg = None
        self.measured_temperature = None

    def set_connection_status_callback(self, callback):
        self._connection_status_callback = callback

    def _emit_connection_status(self, event, info=None):
        callback = self._connection_status_callback
        if callback is None:
            return
        payload = dict(info or {})
        payload.setdefault("backend", self._active_backend)

        def deliver():
            try:
                callback(event, payload)
            except Exception:
                pass

        try:
            self.view.after(0, deliver)
        except Exception:
            deliver()

    @staticmethod
    def _normalise_backend(value):
        name = str(value or "").strip().lower()
        if name in ("arduino", "vasomoto"):
            return "VasoMoto"
        if name in ("ni", "nidaq", "ni-daq"):
            return "NI-DAQ"
        return ""

    def configured_backend(self):
        settings = getattr(self.model.state.toolbar, "pressure_device", None)
        if settings is None:
            return "NI-DAQ"
        return self._normalise_backend(settings.device_type.get())

    def active_backend(self):
        return self._active_backend

    def connect_vasomoto(self, port="Auto", baud=115200):
        """Start the configured VasoMoto serial connection in the background."""
        if self._vasomoto_worker is not None:
            self.disconnect_vasomoto()

        try:
            from serial.tools import list_ports
            ports = list(list_ports.comports())
        except Exception as exc:
            tmb.showinfo("VasoMoto", f"Unable to scan serial ports:\n{exc}")
            return False

        if not ports:
            tmb.showinfo(
                "VasoMoto",
                "No serial devices were detected. Connect VasoMoto by USB and try again.",
            )
            return False

        requested_port = str(port or "Auto").strip()
        auto_detect = requested_port.lower() in ("", "auto", "detect", "autodetect")
        port_names = [info.device for info in ports]
        if not auto_detect and requested_port not in port_names:
            tmb.showinfo(
                "VasoMoto",
                f"The configured port {requested_port} is not available.\n"
                "Open Settings > Pressure Controller Setup and select the current port.",
            )
            return False

        monitor = LinkMonitor(expected_rx_hz=5.0, warmup_s=2.0, stale_s=5.0)
        device = ArduinoPressureDevice(worker=None)

        def status_callback(event, info):
            if worker is not self._vasomoto_worker:
                return
            self._vasomoto_state = event
            port = info.get("port") or "auto"
            if event == "healthy":
                hz = info.get("rx_hz")
                rate = f" ({hz:.1f} Hz)" if hz else ""
                print(f"[VasoMoto] Connected on {port}{rate}")
            elif event == "error":
                message = info.get("message") or "unknown serial error"
                print(f"[VasoMoto] {port}: {message}")
            elif event == "stale":
                print(f"[VasoMoto] Waiting for telemetry on {port}")
            elif event == "closed" and self._vasomoto_worker is not None:
                print(f"[VasoMoto] Connection closed on {port}; retrying")
            self._emit_connection_status(event, info)

        worker = ArduinoSerialWorker(
            port=None if auto_detect else requested_port,
            baud=int(baud),
            monitor=monitor,
            line_callback=device.handle_line,
            status_callback=status_callback,
        )
        device.bind_worker(worker)
        self._vasomoto_worker = worker
        self._vasomoto_device = device
        self._vasomoto_monitor = monitor
        self._vasomoto_state = "connecting"
        self._active_backend = "VasoMoto"
        self._emit_connection_status(
            "connecting", {"port": "Auto" if auto_detect else requested_port}
        )
        device.start()
        return True

    def disconnect_vasomoto(self):
        """Stop VasoMoto telemetry and release its serial port."""
        device = self._vasomoto_device
        worker = self._vasomoto_worker
        self._vasomoto_device = None
        self._vasomoto_worker = None
        self._vasomoto_monitor = None
        self._vasomoto_state = "disconnected"
        if device is not None:
            try:
                device.stop()
            except Exception:
                pass
        if worker is not None:
            try:
                worker.stop()
            except Exception:
                pass
        if self._active_backend == "VasoMoto":
            self._active_backend = None
        self._emit_connection_status("disconnected", {"backend": "VasoMoto"})

    def vasomoto_active(self):
        return self._vasomoto_worker is not None

    def disconnect_nidaq(self):
        task = self.task
        self.task = None
        if task is not None:
            try:
                task.StopTask()
            except Exception:
                pass
            try:
                task.ClearTask()
            except Exception:
                pass
        if self._active_backend == "NI-DAQ":
            self._active_backend = None
        self._emit_connection_status("disconnected", {"backend": "NI-DAQ"})

    def connect_nidaq(self):
        self.disconnect_vasomoto()
        if self.set_dev():
            self._active_backend = "NI-DAQ"
            self._emit_connection_status("healthy", {"backend": "NI-DAQ"})
            return True
        return False

    def disconnect_device(self):
        self.disconnect_vasomoto()
        self.disconnect_nidaq()
        self._active_backend = None

    def connect_configured_device(self):
        backend = self.configured_backend()
        settings = getattr(self.model.state.toolbar, "pressure_device", None)
        if backend == "VasoMoto" and settings is not None:
            try:
                baud = int(settings.baud.get())
            except Exception:
                tmb.showinfo("VasoMoto", "The configured baud rate is not valid.")
                return False
            self.disconnect_nidaq()
            return self.connect_vasomoto(settings.port.get(), baud)
        if backend == "NI-DAQ":
            return self.connect_nidaq()
        tmb.showinfo(
            "Pressure controller",
            "Select NI-DAQ or VasoMoto under Settings > Pressure Controller Setup.",
        )
        return False

    def toggle_configured_device(self):
        """Connect, disconnect, or force-reconnect the device selected in settings."""
        configured = self.configured_backend()
        if self._active_backend == configured:
            if configured == "VasoMoto" and self._vasomoto_state in (
                "error",
                "stale",
                "closed",
            ):
                self.disconnect_vasomoto()
                return self.connect_configured_device()
            self.disconnect_device()
            return False
        self.disconnect_device()
        return self.connect_configured_device()

    def shutdown(self):
        self.disconnect_device()

    # Compatibility with the official Model polling path, which historically
    # read a separate Arduino object for pressure telemetry.
    def getData(self):
        return [[""]]

    def sortdata(self, _data=None):
        if self._vasomoto_device is None:
            self.measured_pressure_1 = None
            self.measured_pressure_2 = None
            self.measured_pressure_avg = None
            return None, None, None, self.measured_temperature
        p1, p2, device_setpoint = self._vasomoto_device.read_latest()
        measured = [float(v) for v in (p1, p2) if v is not None]
        average = sum(measured) / len(measured) if measured else None
        self.measured_pressure_1 = p1
        self.measured_pressure_2 = p2
        self.measured_pressure_avg = average

        # Mirror physical-knob changes into the official +/- target display.
        # process_updates() calls sortdata() on Tk's event loop, so updating the
        # Tk variable here stays on the UI thread.
        is_manual = getattr(
            self._vasomoto_device, "is_manual_override_active", lambda: False
        )()
        if (
            is_manual
            and device_setpoint is not None
            and not (
                isinstance(device_setpoint, float)
                and device_setpoint != device_setpoint
            )
        ):
            target = float(device_setpoint)
            display_var = self.model.state.toolbar.pressure_protocol.set_pressure
            try:
                displayed = float(display_var.get())
            except (TypeError, ValueError):
                displayed = None
            if displayed is None or abs(displayed - target) > 0.05:
                display_var.set(target)
                print(f"[VasoMoto] Device set pressure: {target:.1f} mmHg")
        return p1, p2, average, self.measured_temperature
    
    def end_protocol(self):
        try:
            self.view.toolbar.pressure_control_settings.toggle_protocol_button()
        except Exception as e:
            print(f"Error in end_protocol: {e}")



    def initialize_pressure_system(self):
        if self.pydaqmx_available:
            self.task = PyDAQmx.Task()
            self.set_dev()

        settings = getattr(
            self.model.state.toolbar,
            "pressure_device",
            self.model.state.toolbar.servo,
        )
        device_var = getattr(settings, "ni_device", None) or getattr(settings, "device")
        channel_var = getattr(settings, "ni_ao_channel", None) or getattr(settings, "ao_channel")
        device = device_var.get()
        ao_channel = channel_var.get()

        print(f"The device is {device}, and the aochannel is {ao_channel}")

    def on_option_changed(self, *args):
        settings = getattr(
            self.model.state.toolbar,
            "pressure_device",
            self.model.state.toolbar.servo,
        )
        device = (getattr(settings, "ni_device", None) or getattr(settings, "device")).get()
        ao_channel = (getattr(settings, "ni_ao_channel", None) or getattr(settings, "ao_channel")).get()
        if device != "" and ao_channel != "":
            
            if self.set_dev():
                self.view.toolbar.pressure_control_settings.enable_buttons()

    def update_intvl(self):
        current_time = time.time()

        # Check if sufficient time has elapsed since the last update
        if self.last_update_time is not None and (current_time - self.last_update_time) < self.update_threshold:
            return  # Exit if not enough time has passed

        pressure_protocol_settings = self.model.state.toolbar.pressure_protocol
        if pressure_protocol_settings.pressure_protocol_flag.get() == 0:
            if self.protocol_completed:
                self.reset_protocol()  # Reset protocol for next run
            return  # Exit if the protocol is not active

        if self.pressure_start_time is None:
            self.initialize_pressure_protocol(pressure_protocol_settings)

        elapsed_seconds = current_time - self.pressure_start_time

        time_to_update_secs = self.multiplier * self.pressure_time_interval - int(elapsed_seconds)
        self.model.state.toolbar.data_acq.countdown.set(str(timedelta(seconds=time_to_update_secs)))

        if elapsed_seconds >= self.next_pressure_update_time:
            # Advance the deadline before commanding/logging the step.  A table
            # or UI failure must never leave the old deadline active and cause
            # the remaining ramp steps to fire once per update cycle.
            self.next_pressure_update_time += self.pressure_time_interval
            self.update_pressure()

        self.last_update_time = current_time

    def initialize_pressure_protocol(self, settings):
        self.start_pressure = settings.pressure_start.get()
        self.stop_pressure = settings.pressure_stop.get()
        self.pressure_interval = settings.pressure_intvl.get()
        self.pressure_time_interval = settings.time_intvl.get()
        self.pressure_start_time = time.time()
        self.next_pressure_update_time = self.pressure_time_interval
        self.multiplier = 1
        self.protocol_completed = False

        # Immediately set pressure to start_pressure when the protocol begins
        self.set_pressure = self.start_pressure
        self.adjust_pressure(self.set_pressure)


    def update_pressure(self):
        self.stop_protocol_on_completion = True
        self.completed = False
        if self.set_pressure < self.stop_pressure:
            self.set_pressure += self.pressure_interval
            self.adjust_pressure(self.set_pressure)
            self.multiplier += 1
        else:
            

            if not self.model.state.toolbar.pressure_protocol.hold_pressure.get():# Reset to start pressure or stop protocol
                self.set_pressure = self.start_pressure
                self.adjust_pressure(self.set_pressure)
            self.multiplier = 1  # Reset the multiplier
            self.completed = True
            self.model.state.toolbar.pressure_protocol.pressure_protocol_flag.set(0)
            self.reset_protocol()  # Reset protocol for next run

        if self.completed:
            self.end_protocol()



    def reset_protocol(self):
        # Reset all protocol control variables
        settings = self.model.state.toolbar.pressure_protocol
        self.start_pressure = settings.pressure_start.get()
        self.stop_pressure = settings.pressure_stop.get()
        self.pressure_interval = settings.pressure_intvl.get()
        self.pressure_time_interval = settings.time_intvl.get()
        self.set_pressure = settings.set_pressure.get()
        self.pressure_start_time = None
        self.multiplier = 1
        self.next_pressure_update_time = 0
        self.protocol_completed = False



    def set_dev(self):
        if not self.pydaqmx_available:
            tmb.showinfo(
                "NI-DAQ unavailable",
                "PyDAQmx is not installed. Install the NI-DAQ requirements or select VasoMoto.",
            )
            return False

        settings = getattr(
            self.model.state.toolbar,
            "pressure_device",
            self.model.state.toolbar.servo,
        )
        device_var = getattr(settings, "ni_device", None) or getattr(settings, "device")
        channel_var = getattr(settings, "ni_ao_channel", None) or getattr(settings, "ao_channel")
        device = str(device_var.get()).strip()
        ao_channel = str(channel_var.get()).strip()
        if not device or not ao_channel:
            tmb.showinfo(
                "NI-DAQ configuration",
                "Enter both the NI device and analog output channel under "
                "Settings > Pressure Controller Setup.",
            )
            return False

        old_task = self.task
        self.task = None
        if old_task is not None:
            try:
                old_task.StopTask()
            except Exception:
                pass
            try:
                old_task.ClearTask()
            except Exception:
                pass
        try:
            self.task = PyDAQmx.Task()
            self.task.CreateAOVoltageChan(f"/{device}/{ao_channel}", "", -10.0, 10.0, PyDAQmx.DAQmx_Val_Volts, None)
            self.task.StartTask()
            self.view.toolbar.pressure_protocol_settings.set_unlock_state()
            return True
        except Exception as e:
            print("Failed to connect to NI device:", e)
            failed_task = self.task
            self.task = None
            if failed_task is not None:
                try:
                    failed_task.ClearTask()
                except Exception:
                    pass
            try:
                self.view.toolbar.pressure_protocol_settings.set_lock_state()
            except Exception:
                pass
            tmb.showinfo(
                "NI-DAQ connection",
                "Cannot connect to the configured NI-DAQ device.\n"
                "Check the USB connection, device name, and AO channel, then press Power to retry.",
            )
            self._emit_connection_status(
                "error", {"backend": "NI-DAQ", "message": str(e)}
            )
            return False
            
        

    def adjust_pressure(self, pressure_value, update_table=True):
        # Validate and adjust pressure value to be within the acceptable range
        pressure_value = max(min(200, pressure_value), 0)

        pressure_protocol_settings = self.model.state.toolbar.pressure_protocol
        pressure_protocol_settings.set_pressure.set(pressure_value)

        if self._vasomoto_device is not None:
            queued = self._vasomoto_device.set_pressure(pressure_value)
            if not queued:
                print("[VasoMoto] Could not queue the pressure command.")
                return False
            print(f"[VasoMoto] Set pressure requested: {pressure_value:.1f} mmHg")
        elif self.pydaqmx_available and self.task is not None:
            # Preserve the official NI-DAQ backend when it is explicitly set up.
            try:
                self.task.WriteAnalogScalarF64(1, 10.0, pressure_value / 100, None)
            except Exception as e:
                print("Exception occurred while setting pressure:", e)
                return False
        else:
            print("No pressure controller is connected.")
            return False

        # Optionally update the table
        # If update_table is True, this will update the UI to reflect the new pressure
        if update_table:
            try:
                self.model.state.table.label.set(f"Set pressure = {pressure_value} mmHg")
                self.model.add_table_row()
            except Exception as exc:
                # Pressure timing and hardware control must remain independent
                # of optional table annotation failures.
                print(f"Could not add pressure-step table label: {exc}")
        return True

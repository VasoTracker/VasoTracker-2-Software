import unittest
from unittest.mock import patch

from vasotracker_2.config import AcquisitionSettings, Config
from vasotracker_2.utilities.VT_Pressure import PressureController
from vasotracker_2.utilities.pressure_devices import ArduinoPressureDevice


class _Var:
    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _PressureProtocol:
    def __init__(self):
        self.set_pressure = _Var(0.0)
        self.pressure_protocol_flag = _Var(1)


class _PressureDeviceConfig:
    def __init__(self):
        self.device_type = _Var("VasoMoto")
        self.port = _Var("COM7")
        self.baud = _Var(115200)
        self.ni_device = _Var("Dev1")
        self.ni_ao_channel = _Var("ao1")


class _Model:
    def __init__(self):
        self.state = type("State", (), {})()
        self.state.toolbar = type("Toolbar", (), {})()
        self.state.toolbar.pressure_protocol = _PressureProtocol()
        self.state.toolbar.data_acq = type(
            "DataAcq", (), {"countdown": _Var("0:00:00")}
        )()
        self.state.toolbar.pressure_device = _PressureDeviceConfig()
        self.state.toolbar.servo = type(
            "Servo", (), {"device": _Var("Dev1"), "ao_channel": _Var("ao1")}
        )()
        self.state.table = type("Table", (), {"label": _Var("")})()
        self.table_rows = 0

    def add_table_row(self):
        self.table_rows += 1


class _VasoMoto:
    def __init__(self, latest=(None, None, None), manual_override=False):
        self.latest = latest
        self.manual_override = manual_override
        self.pressures = []
        self.stopped = False

    def set_pressure(self, pressure):
        self.pressures.append(pressure)
        return True

    def read_latest(self):
        return self.latest

    def is_manual_override_active(self):
        return self.manual_override

    def stop(self):
        self.stopped = True


class _NITask:
    def __init__(self):
        self.writes = []

    def WriteAnalogScalarF64(self, *args):
        self.writes.append(args)

    def StopTask(self):
        return None

    def ClearTask(self):
        return None


class _Worker:
    def __init__(self):
        self.writes = []
        self.stopped = False

    def write(self, payload, **kwargs):
        self.writes.append((payload, kwargs))
        return True

    def stop(self):
        self.stopped = True


class PressureBackendTests(unittest.TestCase):
    def setUp(self):
        self.model = _Model()
        self.controller = PressureController(self.model, object(), True)

    def test_vasomoto_is_used_when_connected(self):
        vasomoto = _VasoMoto()
        ni_task = _NITask()
        self.controller._vasomoto_device = vasomoto
        self.controller.task = ni_task

        self.controller.adjust_pressure(125.0, update_table=False)

        self.assertEqual(vasomoto.pressures, [125.0])
        self.assertEqual(ni_task.writes, [])

    def test_pressure_ramp_step_requests_a_labelled_table_row(self):
        self.controller._vasomoto_device = _VasoMoto()

        applied = self.controller.adjust_pressure(40.0, update_table=True)

        self.assertTrue(applied)
        self.assertEqual(self.model.state.table.label.get(), "Set pressure = 40.0 mmHg")
        self.assertEqual(self.model.table_rows, 1)

    def test_ramp_deadline_advances_even_if_step_logging_fails(self):
        self.controller.pressure_start_time = 0.0
        self.controller.pressure_time_interval = 300.0
        self.controller.next_pressure_update_time = 300.0
        self.controller.multiplier = 1

        with (
            patch(
                "vasotracker_2.utilities.VT_Pressure.time.time",
                return_value=300.0,
            ),
            patch.object(
                self.controller,
                "update_pressure",
                side_effect=RuntimeError("table failure"),
            ),
            self.assertRaises(RuntimeError),
        ):
            self.controller.update_intvl()

        self.assertEqual(self.controller.next_pressure_update_time, 600.0)

    def test_ni_daq_remains_the_fallback_backend(self):
        ni_task = _NITask()
        self.controller.task = ni_task

        self.controller.adjust_pressure(125.0, update_table=False)

        self.assertEqual(ni_task.writes, [(1, 10.0, 1.25, None)])

    def test_vasomoto_telemetry_uses_official_cached_interface(self):
        self.controller._vasomoto_device = _VasoMoto((40.0, 42.0, 50.0))

        values = self.controller.sortdata()

        self.assertEqual(values, (40.0, 42.0, 41.0, None))
        self.assertEqual(self.controller.measured_pressure_1, 40.0)
        self.assertEqual(self.controller.measured_pressure_2, 42.0)
        self.assertEqual(self.controller.measured_pressure_avg, 41.0)

    def test_vasomoto_protocol_parses_device_telemetry(self):
        device = ArduinoPressureDevice()

        device.handle_line("DATA T=100 P=42.5 P_SET=50.0")

        self.assertEqual(device.read_latest(), (42.5, None, 50.0))

    def test_physical_vasomoto_setpoint_updates_manual_control_display(self):
        self.controller._vasomoto_device = _VasoMoto(
            (40.0, None, 37.0), manual_override=True
        )

        self.controller.sortdata()

        self.assertEqual(
            self.model.state.toolbar.pressure_protocol.set_pressure.get(), 37.0
        )

    def test_old_telemetry_does_not_override_recent_app_command(self):
        worker = _Worker()
        device = ArduinoPressureDevice(worker=worker)

        device.set_pressure(60.0)
        device.handle_line("DATA T=100 P=42.5 P_SET=20.0")

        self.assertFalse(device.is_manual_override_active())

    def test_knob_change_is_detected_immediately_after_app_ack(self):
        worker = _Worker()
        device = ArduinoPressureDevice(worker=worker)

        device.set_pressure(60.0)
        device.handle_line("ACK SET P=60.0 T=100")
        device.handle_line("DATA T=110 P=42.5 P_SET=45.0")

        self.assertTrue(device.is_manual_override_active())
        self.assertEqual(device.read_latest()[2], 45.0)

    def test_vasomoto_setpoint_is_sent_to_worker(self):
        worker = _Worker()
        device = ArduinoPressureDevice(worker=worker)

        device.set_pressure(60.0)

        commands = [payload.strip() for payload, _kwargs in worker.writes]
        self.assertEqual(commands, ["SET P=60.0"])

    def test_power_connects_the_backend_selected_by_settings(self):
        with (
            patch.object(self.controller, "disconnect_nidaq"),
            patch.object(self.controller, "connect_vasomoto", return_value=True) as connect,
        ):
            connected = self.controller.connect_configured_device()

        self.assertTrue(connected)
        connect.assert_called_once_with("COM7", 115200)

        self.model.state.toolbar.pressure_device.device_type.set("NI-DAQ")
        with patch.object(self.controller, "connect_nidaq", return_value=True) as connect_ni:
            connected = self.controller.connect_configured_device()

        self.assertTrue(connected)
        connect_ni.assert_called_once_with()

    def test_power_forces_a_reconnect_when_vasomoto_is_stale(self):
        self.controller._active_backend = "VasoMoto"
        self.controller._vasomoto_state = "stale"
        with (
            patch.object(self.controller, "disconnect_vasomoto") as disconnect,
            patch.object(self.controller, "connect_configured_device", return_value=True) as connect,
        ):
            connected = self.controller.toggle_configured_device()

        self.assertTrue(connected)
        disconnect.assert_called_once_with()
        connect.assert_called_once_with()

    def test_official_servo_settings_migrate_to_ni_daq(self):
        data = {"servo": {"device": "Dev2", "ao_channel": "ao0"}}

        Config._add_pressure_defaults(data)

        self.assertEqual(
            data["pressure"],
            {
                "device": "NI-DAQ",
                "port": "Auto",
                "baud": 115200,
                "ni_device": "Dev2",
                "ni_ao_channel": "ao0",
            },
        )

    def test_existing_vasomoto_settings_are_not_overwritten(self):
        pressure = {
            "device": "VasoMoto",
            "port": "COM3",
            "baud": 115200,
            "ni_device": "Dev1",
            "ni_ao_channel": "ao1",
        }
        data = {"pressure": pressure.copy()}

        Config._add_pressure_defaults(data)

        self.assertEqual(data["pressure"], pressure)

    def test_acquisition_settings_save_camera_selection(self):
        state = type("State", (), {})()
        state.toolbar = type("Toolbar", (), {})()
        state.toolbar.acq = type(
            "Acq",
            (),
            {
                "camera": _Var("DCC1545M"),
                "scale": _Var(0.52),
                "exposure": _Var(10),
                "pixel_clock": _Var(10),
                "rec_interval": _Var(10.0),
                "target_fps": _Var(10.0),
            },
        )()
        state.toolbar.start_stop = type(
            "StartStop", (), {"save_overlay": _Var(True)}
        )()

        settings = AcquisitionSettings.from_state(state)

        self.assertEqual(settings.camera, "DCC1545M")

    def test_old_settings_without_camera_preserve_current_selection(self):
        state = type("State", (), {})()
        state.toolbar = type("Toolbar", (), {})()
        state.toolbar.acq = type(
            "Acq",
            (),
            {
                "camera": _Var("DCC1545M"),
                "scale": _Var(),
                "exposure": _Var(),
                "pixel_clock": _Var(),
                "rec_interval": _Var(),
                "target_fps": _Var(),
            },
        )()
        state.toolbar.start_stop = type(
            "StartStop", (), {"save_overlay": _Var()}
        )()

        AcquisitionSettings().set_values(state)

        self.assertEqual(state.toolbar.acq.camera.get(), "DCC1545M")


if __name__ == "__main__":
    unittest.main()

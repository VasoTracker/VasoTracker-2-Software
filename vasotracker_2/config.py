##################################################
## VasoTracker 2 - Blood Vessel Diameter Measurement Software
##
## Author: Calum Wilson, Matthew D Lee, and Chris Osborne
## License: BSD 3-Clause License (See main file for details)
## Website: www.vasostracker.com
##
##################################################


from dataclasses import dataclass, field, asdict, fields, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union
import toml
import dacite
import os

if TYPE_CHECKING:
    from vt_mvc import VtState

class Configurator:
    def set_values(self, state: "VtState"):
        pass

    @classmethod
    def from_state(cls, state: "VtState"):
        return cls()


@dataclass
class AcquisitionSettings(Configurator):
    # Empty means the settings file predates camera persistence. In that case,
    # loading the file must leave the currently selected camera alone.
    camera: str = ""
    pixel_world_scale: float = 1.0
    exposure: int = 50
    pixel_clock: int = 10
    recording_interval: float = 10.0
    target_fps: float = 10.0  # Target frame rate in Hz
    save_overlay: bool = True  # also write the tracked-overlay TIFF stack

    def set_values(self, state: "VtState"):
        acq = state.toolbar.acq
        if self.camera:
            acq.camera.set(self.camera)
        acq.scale.set(self.pixel_world_scale)
        acq.exposure.set(self.exposure)
        acq.pixel_clock.set(self.pixel_clock)
        acq.rec_interval.set(self.recording_interval)
        acq.target_fps.set(self.target_fps)
        state.toolbar.start_stop.save_overlay.set(self.save_overlay)

    @classmethod
    def from_state(cls, state: "VtState"):
        # Coerce to the declared field types: several of these live in the UI as
        # IntVars, and dumping them straight back rewrites "300.0" as "300" etc.
        # on every save, churning settings.toml for no reason.
        acq = state.toolbar.acq
        return cls(
            camera=str(acq.camera.get()),
            pixel_world_scale=float(acq.scale.get()),
            exposure=int(acq.exposure.get()),
            pixel_clock=int(acq.pixel_clock.get()),
            recording_interval=float(acq.rec_interval.get()),
            target_fps=float(acq.target_fps.get()),
            save_overlay=bool(state.toolbar.start_stop.save_overlay.get()),
        )


@dataclass
class AnalysisSettings(Configurator):
    num_lines: int = 10
    smooth: int = 21
    integration: int = 20
    threshold: float = 5.5
    num_threads: int = 1

    def set_values(self, state: "VtState"):
        ana = state.toolbar.analysis
        ana.num_lines.set(self.num_lines)
        ana.smooth_factor.set(self.smooth)
        ana.integration_factor.set(self.integration)
        ana.thresh_factor.set(self.threshold)

    @classmethod
    def from_state(cls, state: "VtState"):
        ana = state.toolbar.analysis
        return cls(
            num_lines=int(ana.num_lines.get()),
            smooth=int(ana.smooth_factor.get()),
            integration=int(ana.integration_factor.get()),
            threshold=float(ana.thresh_factor.get()),
        )


@dataclass
class GraphAxisSettings(Configurator):
    x_min: float = -1200.0
    x_max: float = 0.0
    y_min1: float = 50.0
    y_max1: float = 250.0
    y_min2: float = 25.0
    y_max2: float = 200.0
    y_min_p: float = 0.0
    y_max_p: float = 200.0
    axis1_metric: str = "Outer diameter"
    axis2_metric: str = "Inner diameter"

    def set_values(self, state: "VtState"):
        g = state.toolbar.graph
        g.x_min.set(self.x_min) #
        g.x_max.set(self.x_max)
        g.y_min_od.set(self.y_min1)
        g.y_max_od.set(self.y_max1)
        g.y_min_id.set(self.y_min2)
        g.y_max_id.set(self.y_max2)
        g.y_min_p.set(self.y_min_p)
        g.y_max_p.set(self.y_max_p)
        g.axis1_metric.set(self.axis1_metric)
        g.axis2_metric.set(self.axis2_metric)
    @classmethod
    def from_state(cls, state: "VtState"):
        g = state.toolbar.graph
        return cls(
            x_min=float(g.x_min.get()),
            x_max=float(g.x_max.get()),
            y_min1=float(g.y_min_od.get()),
            y_max1=float(g.y_max_od.get()),
            y_min2=float(g.y_min_id.get()),
            y_max2=float(g.y_max_id.get()),
            y_min_p=float(g.y_min_p.get()),
            y_max_p=float(g.y_max_p.get()),
            axis1_metric=g.axis1_metric.get(),
            axis2_metric=g.axis2_metric.get(),
        )


@dataclass
class MemorySettings(Configurator):
    num_plot_points: int = 500000
    num_data_points: int = 500000

    def set_values(self, state: "VtState"):
        state.measure.max_len = self.num_data_points

    @classmethod
    def from_state(cls, state: "VtState"):
        if state.measure.max_len is not None:
            return cls(
                num_data_points=state.measure.max_len
            )
        return cls()
    

@dataclass
class ServoSettings(Configurator):
    
    device: str = "Dev1"
    ao_channel: str = "ao1"

    def set_values(self, state: "VtState"):
        servo = state.toolbar.servo
        servo.device.set(self.device)
        servo.ao_channel.set(self.ao_channel)
        print("Device: ", self.device)

    @classmethod
    def from_state(cls, state: "VtState"):
        servo = state.toolbar.servo
        device = servo.device.get()
        ao_channel= servo.ao_channel.get()
        return cls(
            device=device,
            ao_channel=ao_channel,
        )


@dataclass
class PressureHardwareSettings(Configurator):
    """Persisted pressure backend selection and both backends' connection details."""

    device: str = "NI-DAQ"
    port: str = "Auto"
    baud: int = 115200
    ni_device: str = "Dev1"
    ni_ao_channel: str = "ao1"

    @staticmethod
    def normalize_device(name: str) -> str:
        value = str(name or "").strip().lower()
        if value in ("arduino", "vasomoto"):
            return "VasoMoto"
        return "NI-DAQ"

    def set_values(self, state: "VtState"):
        hardware = state.toolbar.pressure_device
        hardware.device_type.set(self.normalize_device(self.device))
        hardware.port.set(self.port or "Auto")
        hardware.baud.set(int(self.baud))
        hardware.ni_device.set(self.ni_device)
        hardware.ni_ao_channel.set(self.ni_ao_channel)

        # Keep the official servo state in sync for backwards compatibility.
        state.toolbar.servo.device.set(self.ni_device)
        state.toolbar.servo.ao_channel.set(self.ni_ao_channel)

    @classmethod
    def from_state(cls, state: "VtState"):
        hardware = state.toolbar.pressure_device
        return cls(
            device=cls.normalize_device(hardware.device_type.get()),
            port=hardware.port.get() or "Auto",
            baud=int(hardware.baud.get()),
            ni_device=hardware.ni_device.get(),
            ni_ao_channel=hardware.ni_ao_channel.get(),
        )


@dataclass
class PressureControlSettings(Configurator):
    default_pressure: float = 20.0
    time_interval: float = 300.0
    start_pressure: float = 20.0
    stop_pressure: float = 100.0
    pressure_interval: float = 20.0

    def set_values(self, state: "VtState"):
        p = state.toolbar.pressure_protocol
        p.pressure_start.set(self.start_pressure)
        p.pressure_stop.set(self.stop_pressure)
        p.pressure_intvl.set(self.pressure_interval)
        p.time_intvl.set(self.time_interval)
        s = state.toolbar.servo
        p.set_pressure.set(self.default_pressure)

    @classmethod
    def from_state(cls, state: "VtState"):
        p = state.toolbar.pressure_protocol
        # default_pressure mirrors set_values, which writes it to
        # pressure_protocol.set_pressure (not the servo's live set point).
        return cls(
            default_pressure=float(p.set_pressure.get()),
            time_interval=float(p.time_intvl.get()),
            start_pressure=float(p.pressure_start.get()),
            stop_pressure=float(p.pressure_stop.get()),
            pressure_interval=float(p.pressure_intvl.get()),
        )

@dataclass
class TisDcamSettings:
    property_gain: int = 240


@dataclass
class ProxyCameraSettings:
    #initialdir = os.getcwd()
    path_template: str = "\\SampleData\\TEST{:04d}.tif" #f'{initialdir}' + "\\SampleData\\TEST{:d}.tif" #
    max_frame: int = 300

@dataclass
class RegistrationSettings:
    #initialdir = os.getcwd()
    register_flag: int = 0
    neveragain_flag: int = 0


@dataclass
class MicroManagerSettings(Configurator):
    # Micro-Manager system configuration (.cfg) loaded when the "MMConfig"
    # camera is selected. Empty string means auto-resolve: the MMConfig.cfg
    # sitting next to the active Micro-Manager install, else the placeholder
    # bundled with VasoTracker. Set via the config chooser dialog that opens
    # when "MMConfig" is picked in the camera dropdown.
    config_file: str = ""

    def set_values(self, state: "VtState"):
        state.toolbar.acq.mm_config_file.set(self.config_file)

    @classmethod
    def from_state(cls, state: "VtState"):
        return cls(config_file=state.toolbar.acq.mm_config_file.get())

@dataclass
class Config(Configurator):
    acquisition: AcquisitionSettings = field(default_factory=AcquisitionSettings)
    analysis: AnalysisSettings = field(default_factory=AnalysisSettings)
    servo: ServoSettings = field(default_factory=ServoSettings)
    pressure: PressureHardwareSettings = field(default_factory=PressureHardwareSettings)
    graph_axes: GraphAxisSettings = field(default_factory=GraphAxisSettings)
    memory: MemorySettings = field(default_factory=MemorySettings)
    pressure_control: PressureControlSettings = field(
        default_factory=PressureControlSettings
    )
    TIS_DCAM: TisDcamSettings = field(default_factory=TisDcamSettings)
    proxy_camera: ProxyCameraSettings = field(default_factory=ProxyCameraSettings)
    registration: RegistrationSettings = field(default_factory=RegistrationSettings)
    micromanager: MicroManagerSettings = field(default_factory=MicroManagerSettings)

    path: Optional[str] = None

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "Config":
        data = toml.load(path)
        cls._add_pressure_defaults(data)
        result = dacite.from_dict(data_class=cls, data=data)
        result.pressure.device = PressureHardwareSettings.normalize_device(
            result.pressure.device
        )
        result.path = str(path)
        return result

    @staticmethod
    def _add_pressure_defaults(data: dict) -> None:
        """Upgrade official/older settings files that only contain [servo]."""
        if "pressure" in data:
            return
        servo = data.get("servo", {}) or {}
        data["pressure"] = {
            "device": "NI-DAQ",
            "port": "Auto",
            "baud": 115200,
            "ni_device": servo.get("device", "Dev1"),
            "ni_ao_channel": servo.get("ao_channel", "ao1"),
        }

    @classmethod
    def load(cls, bundled_path: Union[str, Path],
             user_path: Union[str, Path]) -> "Config":
        """The bundled settings.toml is the base; an optional per-user
        settings.toml is layered on top. Runtime writes go to the user file
        (``self.path``), so the shipped defaults - and a read-only Program
        Files install - are never modified."""
        data: dict = {}
        try:
            data = toml.load(bundled_path)
        except Exception:
            pass
        user_path = Path(user_path)
        if user_path.is_file():
            def _merge(base: dict, over: dict):
                for k, v in over.items():
                    if isinstance(v, dict) and isinstance(base.get(k), dict):
                        _merge(base[k], v)
                    else:
                        base[k] = v
            try:
                _merge(data, toml.load(user_path))
            except Exception:
                pass
        cls._add_pressure_defaults(data)
        result = dacite.from_dict(data_class=cls, data=data)
        result.pressure.device = PressureHardwareSettings.normalize_device(
            result.pressure.device
        )
        result.path = str(user_path)
        return result

    def save(self, override_path: Optional[Union[str, Path]] = None):
        path = Path(override_path if override_path is not None else self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data.pop("path", None)
        with open(path, "w") as f:
            toml.dump(data, f)

    def set_values(self, state: "VtState"):
        class_fields = fields(self)
        for f in class_fields:
            # NOTE(cmo): Check is_dataclass first, because the Union[str, None]
            # breaks older Python issubclass
            if is_dataclass(f.type) and issubclass(f.type, Configurator):
                item: Configurator = getattr(self, f.name)
                item.set_values(state)

    @classmethod
    def from_state(cls, state: "VtState"):
        attrs = {}
        class_fields = fields(cls)
        for f in class_fields:
            # NOTE(cmo): Check is_dataclass first, because the Union[str, None]
            # breaks older Python issubclass
            if is_dataclass(f.type) and issubclass(f.type, Configurator):
                attrs[f.name] = f.type.from_state(state)
        return cls(**attrs)

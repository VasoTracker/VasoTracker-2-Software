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
    pixel_world_scale: float = 1.0
    exposure: int = 50
    pixel_clock: int = 10
    recording_interval: float = 300.0
    refresh_min_interval: float = 0.2
    refresh_faster_interval: float = 0.01

    def set_values(self, state: "VtState"):
        acq = state.toolbar.acq
        acq.scale.set(self.pixel_world_scale)
        acq.exposure.set(self.exposure)
        acq.pixel_clock.set(self.pixel_clock)
        acq.rec_interval.set(self.recording_interval)

    @classmethod
    def from_state(cls, state: "VtState"):
        acq = state.toolbar.acq
        scale = acq.scale.get()
        exposure = acq.exposure.get()
        pixel_clock = acq.pixel_clock.get()
        recording_interval = acq.rec_interval.get()
        return cls(
            pixel_world_scale=scale,
            exposure=exposure,
            pixel_clock=pixel_clock,
            recording_interval=recording_interval,
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
        num_lines = ana.num_lines.get()
        smooth = ana.smooth_factor.get()
        integration = ana.smooth_factor.get()
        threshold = ana.thresh_factor.get()
        return cls(
            num_lines=num_lines,
            smooth=smooth,
            integration=integration,
            threshold=threshold,
        )


@dataclass
class GraphAxisSettings(Configurator):
    x_min: float = -1200.0
    x_max: float = 0.0
    y_min1: float = 50.0
    y_max1: float = 250.0
    y_min2: float = 25.0
    y_max2: float = 200.0

    def set_values(self, state: "VtState"):
        g = state.toolbar.graph
        g.x_min.set(self.x_min) #
        g.x_max.set(self.x_max)
        g.y_min_od.set(self.y_min1)
        g.y_max_od.set(self.y_max1)
        g.y_min_id.set(self.y_min2)
        g.y_max_id.set(self.y_max2)
    @classmethod
    def from_state(cls, state: "VtState"):
        g = state.toolbar.graph
        return cls(
            x_min=g.x_min.get(),
            x_max=g.x_max.get(),
            y_min1=g.y_min_od.get(),
            y_max1=g.y_max_od.get(),
            y_min2=g.y_min_id.get(),
            y_max2=g.y_max_od.get(),
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
        start_p = p.pressure_start.get()
        stop_p = p.pressure_stop.get()
        p_interval = p.pressure_intvl.get()
        t_interval = p.time_intvl.get()

        s = state.toolbar.servo
        default_pressure = s.set_pressure.get()
        return cls(
            default_pressure=default_pressure,
            time_interval=t_interval,
            start_pressure=start_p,
            stop_pressure=stop_p,
            pressure_interval=p_interval,
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
class Config(Configurator):
    acquisition: AcquisitionSettings = field(default_factory=AcquisitionSettings)
    analysis: AnalysisSettings = field(default_factory=AnalysisSettings)
    servo: ServoSettings = field(default_factory=ServoSettings)
    graph_axes: GraphAxisSettings = field(default_factory=GraphAxisSettings)
    memory: MemorySettings = field(default_factory=MemorySettings)
    pressure_control: PressureControlSettings = field(
        default_factory=PressureControlSettings
    )
    TIS_DCAM: TisDcamSettings = field(default_factory=TisDcamSettings)
    proxy_camera: ProxyCameraSettings = field(default_factory=ProxyCameraSettings)
    registration: RegistrationSettings = field(default_factory=RegistrationSettings)

    path: Optional[str] = None

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "Config":
        data = toml.load(path)
        result = dacite.from_dict(data_class=cls, data=data)
        result.path = str(path)
        return result

    def save(self, override_path: Optional[Union[str, Path]] = None):
        path = self.path
        print("The path is: ", path)
        if override_path is not None:
            path = override_path
        print("The path is now: ", path)
        data = asdict(self)
        print("Data: ", data)
        if self.path is not None:
            del data["path"]
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

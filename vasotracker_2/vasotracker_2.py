## TODO: Update paper link in status_bar 


##################################################
## VasoTracker 2 Blood Vessel Diameter Measurement Software
## 
## This software provides diameter measurements (inner and outer) of blood vessels
## 
## For additional info see www.vasostracker.com
## 
##################################################
## 
## BSD 3-Clause License
## 
## Copyright (c) 2025, VasoTracker
## All rights reserved.
## 
## Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are met:
## 
## * Redistributions of source code must retain the above copyright notice, this
##   list of conditions and the following disclaimer.
## 
## * Redistributions in binary form must reproduce the above copyright notice,
##   this list of conditions and the following disclaimer in the documentation
##   and/or other materials provided with the distribution.
## 
## * Neither the name of the copyright holder nor the names of its
##   contributors may be used to endorse or promote products derived from
##   this software without specific prior written permission.
## 
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
## AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
## IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
## DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
## FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
## DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
## SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
## CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
## OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
## OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
## 
##################################################
## 
## Author: Calum Wilson, Matthew D Lee, and Chris Osborne
## Copyright: Copyright 2025, VasoTracker
## Credits: Calum Wilson, Matthew D Lee, and Chris Osborne
## License: BSD 3-Clause License
## Version: 2.0.0
## Maintainer: Calum Wilson
## Email: vasotracker@gmail.com
## Status: Production
## Last updated: 20250130
## 
##################################################

#TODO: Add memory warning for less than 5 seconds recording interval.
import version
from version import __version__

print(f"VasoTracker Version: {__version__}")

import subprocess
import os


# Standard library imports
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
import csv
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import IntEnum, auto
from functools import partial
from math import hypot
import os
from pathlib import Path
import queue
import sys
import threading
import time
import traceback
from typing import Callable, Dict, List, Optional, Tuple, Type
import webbrowser
# Suppress pygame welcome message
sys.stdout = open(os.devnull, 'w')
import pygame
sys.stdout = sys.__stdout__  # Restore stdout

# Third-party imports
import random
from PIL import Image, ImageTk
import cv2
import numpy as np
from multiprocessing import freeze_support
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.path import Path as MplPath
import skimage
import tifffile as tf
import tkinter as tk
from tkinter import filedialog, scrolledtext, IntVar, StringVar, DoubleVar, BooleanVar, Scale
import tkinter.messagebox as tmb
import tkinter.ttk as ttk
from tkinter import font
import runpy
import json

# Local application/library specific imports
from utilities.VT_Diameter import ImageDiameters, calculate_diameter
from utilities.VT_NavBar import CustomVTToolbar
from utilities.VasoTrackerSplashScreen import VasoTrackerSplashScreen
from utilities.ToolTip import ToolTip
from utilities.VT_Arduino import Arduino as ArduinoController
import utilities.VT_Pressure
from utilities.VT_Pressure import PressureController
from cameras import Camera, CameraBase
from config import AcquisitionSettings, Config, GraphAxisSettings
import customtkinter as ctk
import sv_ttk


# Conditional imports with user notification for optional dependencies
try:
    from pymmcore_plus import CMMCorePlus, find_micromanager
    micromanager_available = True
except:
    micromanager_available = False

is_pydaqmx_available = utilities.VT_Pressure.is_pydaqmx_available()#False

print("Is PYDAQMX = ", is_pydaqmx_available)

# Constants
SYS32_PATH = "C:/WINDOWS/SYSTEM32/DRIVERs/"
BASLER_PATH = os.path.join("C:", os.sep, "Program Files", "Basler/pylon 6/Runtime/x64/")
BASLER_PATH2 = os.path.join("C:", os.sep, "Program Files", "Basler/pylon 6/Runtime/Win32/")

NUM_LINES = 10
NUM_ROIS = 10
ELLIPSIS = "..."
CMAP = plt.get_cmap("tab10")

C1 = (0, 0, 200) #Blue outer
C2 = (0,125, 0) #Dark green inner
C3 = (20, 20, 20)
C4 = (10, 131, 135)
VasoTracker_Green = (10, 131, 135)
VasoTracker_Green_hex = "#{:02x}{:02x}{:02x}".format(*VasoTracker_Green)

default_font = 'Arial'
default_font_size = 14
VasoTracker_Blue = '#203C57'
frame_label_color = VasoTracker_Blue
frame_label_height = 25
entry_disabled_color="#BDC3C7"

# The following is so that the required resources are included in the PyInstaller build.
# Utility functions
def get_resource_path(relative_path):
    """Get the path to a resource, whether it's bundled with PyInstaller or not."""
    base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
    return os.path.join(base_path, relative_path)



# Resource paths
images_folder = get_resource_path("images\\")
sample_data_path = get_resource_path("SampleData\\")
gui_json_path = get_resource_path("VasoTrackerblue.json")

# TODOs and Future Improvements
# TODO:


@dataclass
class SourcePaneState:
    path: StringVar = field(default_factory=StringVar)
    settings: StringVar = field(default_factory=StringVar)
    filename: StringVar = field(default_factory=StringVar)


@dataclass
class AcquisitionPaneState:
    camera: StringVar = field(default_factory=StringVar)
    scale: DoubleVar = field(default_factory=DoubleVar)
    exposure: IntVar = field(default_factory=IntVar)
    pixel_clock: IntVar = field(default_factory=IntVar)
    acq_rate: DoubleVar = field(default_factory=DoubleVar)
    rec_interval: IntVar = field(default_factory=IntVar)
    default_settings: BooleanVar = field(default_factory=BooleanVar)
    fast_mode: BooleanVar = field(default_factory=BooleanVar)
    res: StringVar = field(default_factory=StringVar)
    fov: StringVar = field(default_factory=StringVar)


@dataclass
class AnalysisPaneState:
    num_lines: IntVar = field(default_factory=IntVar)
    smooth_factor: IntVar = field(default_factory=IntVar)
    integration_factor: IntVar = field(default_factory=IntVar)
    thresh_factor: DoubleVar = field(default_factory=DoubleVar)
    filter: BooleanVar = field(default_factory=BooleanVar)
    ID: BooleanVar = field(default_factory=BooleanVar)
    org: BooleanVar = field(default_factory=BooleanVar)
    roi: BooleanVar = field(default_factory=BooleanVar)
    rotate_tracking: BooleanVar = field(default_factory=BooleanVar)
    ultrasound_tracking: BooleanVar = field(default_factory=BooleanVar)


@dataclass
class GraphPaneState:
    x_min: IntVar = field(default_factory=IntVar)
    x_max: IntVar = field(default_factory=IntVar)
    y_min_od: IntVar = field(default_factory=IntVar)
    y_max_od: IntVar = field(default_factory=IntVar)
    y_min_id: IntVar = field(default_factory=IntVar)
    y_max_id: IntVar = field(default_factory=IntVar)
    dirty: BooleanVar = field(default_factory=BooleanVar)
    limits_dirty: BooleanVar = field(default_factory=BooleanVar)


@dataclass
class CaliperROIPaneState:
    roi_flag: StringVar = field(default_factory=StringVar)
    caliper_flag: StringVar = field(default_factory=StringVar)


@dataclass
class PlottingPaneState:
    line_show: List[BooleanVar] = field(
        default_factory=lambda: [BooleanVar() for _ in range(NUM_LINES)]
    )
    outer_diam_roi: Dict[str, List[float]] = field(default_factory=dict)
    inner_diam_roi: Dict[str, List[float]] = field(default_factory=dict)
    
    # Using DoubleVar for multiple entry boxes
    outer_diam_values: List[DoubleVar] = field(
        default_factory=lambda: [DoubleVar() for _ in range(NUM_LINES)]
    )

    inner_diam_values: List[DoubleVar] = field(
        default_factory=lambda: [DoubleVar() for _ in range(NUM_LINES)]
    )



@dataclass
class DataAcqPaneState:
    time: DoubleVar = field(default_factory=DoubleVar)
    #time_string: StringVar = field(default_factory=StringVar)
    time_string: StringVar = field(default_factory=lambda: StringVar(value="00:00:00"))
    temperature: DoubleVar = field(default_factory=DoubleVar)
    pressure: DoubleVar = field(default_factory=DoubleVar)
    outer_diam: DoubleVar = field(default_factory=DoubleVar)
    inner_diam: DoubleVar = field(default_factory=DoubleVar)
    diam_percent: DoubleVar = field(default_factory=DoubleVar)
    caliper_length: DoubleVar = field(default_factory=DoubleVar)
    countdown: IntVar = field(default_factory=IntVar)



@dataclass
class ImageDimensionsPaneState:
    cam_width: IntVar = field(default_factory=IntVar)
    cam_height: IntVar = field(default_factory=IntVar)
    fov_width: IntVar = field(default_factory=IntVar)
    fov_height: IntVar = field(default_factory=IntVar)
    file_length: IntVar = field(default_factory=IntVar)


@dataclass
class ServoSettingsState:
    flag: StringVar = field(default_factory=BooleanVar)
    device: StringVar = field(default_factory=StringVar)
    ao_channel: StringVar = field(default_factory=StringVar)
    set_pressure: IntVar = field(default_factory=IntVar)



@dataclass
class PressureProtocolSettingsState:
    pressure_start: IntVar = field(default_factory=IntVar)
    pressure_stop: IntVar = field(default_factory=IntVar)
    pressure_protocol_flag: IntVar = field(default_factory=IntVar)
    pressure_intvl: IntVar = field(default_factory=IntVar)
    time_intvl: IntVar = field(default_factory=IntVar)
    #countdown: IntVar = field(default_factory=IntVar)
    protocol_start_time: IntVar = field(default_factory=IntVar)
    set_pressure: IntVar = field(default_factory=IntVar)
    pressure_increment: IntVar = field(default_factory=IntVar)
    hold_pressure: BooleanVar = field(default_factory=BooleanVar)

@dataclass
class StartStopState:
    record: BooleanVar = field(default_factory=BooleanVar)


@dataclass
class ToolbarState:
    source: SourcePaneState = field(default_factory=SourcePaneState)
    acq: AcquisitionPaneState = field(default_factory=AcquisitionPaneState)
    analysis: AnalysisPaneState = field(default_factory=AnalysisPaneState)
    graph: GraphPaneState = field(default_factory=GraphPaneState)
    caliper_roi: CaliperROIPaneState = field(default_factory=CaliperROIPaneState)
    plotting: PlottingPaneState = field(default_factory=PlottingPaneState)
    data_acq: DataAcqPaneState = field(default_factory=DataAcqPaneState)
    image_dim: ImageDimensionsPaneState = field(
        default_factory=ImageDimensionsPaneState
    )
    servo: ServoSettingsState = field(default_factory=ServoSettingsState)
    pressure_protocol: PressureProtocolSettingsState = field(
        default_factory=PressureProtocolSettingsState
    )
    start_stop: StartStopState = field(default_factory=StartStopState)


@dataclass
class TableState:
    label: StringVar = field(default_factory=StringVar)
    ref_diam: DoubleVar = field(default_factory=DoubleVar)
    rows_to_add: List[str] = field(default_factory=list)
    dirty: BooleanVar = field(default_factory=BooleanVar)
    dirty_marker: BooleanVar = field(default_factory=BooleanVar)
    clear: BooleanVar = field(default_factory=BooleanVar)

    def headers(self) -> Tuple[str]:
        return (
            "#",
            "Time",
            "Frame",
            "Label",
            "OD",
            "%OD ref",
            "ID",
            "Caliper",
            "Pavg",
            "P1",
            "P2",
            "Temp",
        )


@dataclass
class AppState:
    acquiring: BooleanVar = field(default_factory=BooleanVar)
    tracking: BooleanVar = field(default_factory=BooleanVar)
    tracking_file: BooleanVar = field(default_factory=BooleanVar)
    auto_pressure: BooleanVar = field(default_factory=BooleanVar)
    file_analysed: BooleanVar = field(default_factory=BooleanVar)


@dataclass
class LineData:
    x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    y: np.ndarray = field(default_factory=lambda: np.zeros(0))


@dataclass
class GraphState:
    od_avg: LineData = field(default_factory=LineData)
    id_avg: LineData = field(default_factory=LineData)
    markers: LineData = field(default_factory=LineData)
    od_lines: List[LineData] = field(
        default_factory=lambda: [LineData() for _ in range(NUM_LINES)]
    )
    id_lines: List[LineData] = field(
        default_factory=lambda: [LineData() for _ in range(NUM_LINES)]
    )
    vertical_indicator: Optional[float] = None
    dirty: BooleanVar = field(default_factory=BooleanVar)
    clear: BooleanVar = field(default_factory=BooleanVar)


@dataclass
class Roi:
    x1: int
    x2: int
    y1: int
    y2: int
    handle: Optional[int] = None
    dirty: bool = False

    def fixed_corners(self):
        """Returns the corners in expected tkinter order (top-left,
        bottom-right), ready for splatting"""
        x1 = min(self.x1, self.x2)
        x2 = max(self.x1, self.x2)
        y1 = min(self.y1, self.y2)
        y2 = max(self.y1, self.y2)
        return (x1, y1, x2, y2)


@dataclass
class Caliper:
    x1: int
    x2: int
    y1: int
    y2: int
    length: float


@dataclass
class CanvasDrawState:
    # NOTE(cmo): These are only used for the drawing step, and are in
    # screen-space for the frame. Resizing during drawing could be an issue, but
    # highly unlikely as one needs to hold the mouse to draw. On mouse-up, these
    # get added to the RasterDrawState to be rasterised onto the image.
    roi: Optional[Roi] = None
    caliper: Optional[Roi] = None
    multi_roi: Dict[str, Roi] = field(default_factory=dict)
    autocaliper: Dict[str, Roi] = field(default_factory=dict)
    roi_cleanup: List[Roi] = field(default_factory=list)
    # NOTE(cmo): True only when user is currently drawing on the canvas (i.e.
    # clicking and dragging)
    user_drawing: BooleanVar = field(default_factory=BooleanVar)


@dataclass
class RasterDrawState:
    roi: Optional[Roi] = None
    caliper: Optional[Caliper] = None
    multi_roi: Dict[str, Roi] = field(default_factory=dict)
    autocaliper: Dict[str, Caliper] = field(default_factory=dict)


@dataclass
class CameraViewState:
    dirty: BooleanVar = field(default_factory=BooleanVar)
    slider_position: IntVar = field(default_factory=IntVar)
    slider_dirty: BooleanVar = field(default_factory=BooleanVar)
    slider_change_state: BooleanVar = field(default_factory=BooleanVar)
    slider_toggle_dirty: BooleanVar = field(default_factory=BooleanVar)
    slider_length_dirty: BooleanVar = field(default_factory=BooleanVar)
    im_data: Optional[np.ndarray] = None
    raw_im_data: Optional[np.ndarray] = None
    im_centre: Tuple[int, int] = field(default_factory=lambda: (0, 0))
    im_presented_size: Tuple[int, int] = field(default_factory=lambda: (1, 1))
    canvas_draw_state: CanvasDrawState = field(default_factory=CanvasDrawState)
    raster_draw_state: RasterDrawState = field(default_factory=RasterDrawState)
    slider_position_manual: IntVar = field(default_factory=IntVar)


class CsvListWrapper(list):
    def __str__(self):
        # NOTE(cmo): strip the [ ] off each end
        return super().__str__()[1:-1]


@dataclass
class MeasureStore:
    times: List[float] = field(default_factory=list)
    formatted_times: List[float] = field(default_factory=list)
    outer_diam: List[float] = field(default_factory=list)
    inner_diam: List[float] = field(default_factory=list)
    markers: List[float] = field(default_factory=list)
    #markers: List[float] = field(default_factory=list) ## Working here
    temperature: List[float] = field(default_factory=list)
    pressure1: List[float] = field(default_factory=list)
    pressure2: List[float] = field(default_factory=list)
    avg_pressure: List[float] = field(default_factory=list)
    set_pressure: List[float] = field(default_factory=list)
    caliper_length: List[float] = field(default_factory=list)
    outer_diam_profile: List[np.ndarray] = field(default_factory=list)
    inner_diam_profile: List[np.ndarray] = field(default_factory=list)
    outer_diam_good: List[np.ndarray] = field(default_factory=list)
    inner_diam_good: List[np.ndarray] = field(default_factory=list)
    outer_diam_roi: Dict[str, List[float]] = field(default_factory=dict)
    inner_diam_roi: Dict[str, List[float]] = field(default_factory=dict)

    max_len: Optional[int] = None

    def append(
        self,
        t: float,
        od: float,
        id: float,
        marker: float,
        temperature: float,
        pavg: float,
        p1: float,
        p2: float,
        set_p: float,
        caliper_length: float,
        ods: np.ndarray,
        #ods_smooth: np.ndarray,
        ids: np.ndarray,
        ods_valid: np.ndarray,
        ids_valid: np.ndarray,
    ):
        self.times.append(round(t, 1))
        self.formatted_times.append(time.strftime("%H:%M:%S", time.gmtime(np.round(t, 1))))
        self.outer_diam.append(od)
        self.inner_diam.append(id)
        self.markers.append(marker)
        self.temperature.append(temperature)
        self.pressure1.append(p1)
        self.pressure2.append(p2)
        self.avg_pressure.append(0.5 * (p1 + p2))
        self.set_pressure.append(set_p)
        self.caliper_length.append(caliper_length)
        self.outer_diam_profile.append(ods)
        self.outer_diam_profile = self.outer_diam_profile
        self.inner_diam_profile.append(ids)
        self.outer_diam_good.append(ods_valid)
        self.inner_diam_good.append(ids_valid)

        if self.max_len is not None and len(self.times) > self.max_len:
            self.times = self.times[-self.max_len :]
            self.formatted_times = self.formatted_times[-self.max_len :]
            self.outer_diam = self.outer_diam[-self.max_len :]
            self.inner_diam = self.inner_diam[-self.max_len :]
            self.markers = self.markers[-self.max_len :] # Working
            self.temperature = self.temperature[-self.max_len :]
            self.pressure1 = self.pressure1[-self.max_len :]
            self.pressure2 = self.pressure2[-self.max_len :]
            self.avg_pressure = self.avg_pressure[-self.max_len :]
            self.set_pressure = self.set_pressure[-self.max_len :]
            self.caliper_length = self.caliper_length[-self.max_len :]
            self.outer_diam_profile = self.outer_diam_profile[-self.max_len :]
            self.inner_diam_profile = self.inner_diam_profile[-self.max_len :]
            self.outer_diam_good = self.outer_diam_good[-self.max_len :]
            self.inner_diam_good = self.inner_diam_good[-self.max_len :]

    def get_last_row(self):
        return (
            self.times[-1],
            self.formatted_times[-1],
            self.outer_diam[-1],
            self.inner_diam[-1],
            self.markers[-1],
            self.temperature[-1],
            self.pressure1[-1],
            self.pressure2[-1],
            self.avg_pressure[-1],
            self.set_pressure[-1],
            self.caliper_length[-1],
            CsvListWrapper(self.outer_diam_profile[-1].tolist()),
            CsvListWrapper(self.inner_diam_profile[-1].tolist()),
            CsvListWrapper(self.outer_diam_good[-1].astype(np.int32).tolist()),
            CsvListWrapper(self.inner_diam_good[-1].astype(np.int32).tolist()),
        )

    def headers(self):
        return (
            "Time (s)",
            "Time (hh:mm:ss)",
            "Outer Diameter",
            "Inner Diameter",
            "Table Marker",
            "Temperature (oC)",
            "Pressure 1 (mmHg)",
            "Pressure 2 (mmHg)",
            "Avg Pressure (mmHg)",
            "Set Pressure (mmHg)",
            "Caliper length",
            "Outer Profiles",
            "Inner Profiles",
            "Outer Profiles Valid",
            "Inner Profiles Valid",
        )

    def clear(self):
        self.times.clear()
        self.formatted_times.clear()
        self.outer_diam.clear()
        self.inner_diam.clear()
        self.markers.clear()
        self.temperature.clear()
        self.pressure1.clear()
        self.pressure2.clear()
        self.avg_pressure.clear()
        self.set_pressure.clear()
        self.caliper_length.clear()
        self.outer_diam_profile.clear()
        self.inner_diam_profile.clear()
        self.outer_diam_good.clear()
        self.inner_diam_good.clear()


class MessageType(IntEnum):
    Info = auto()
    Warning = auto()
    Error = auto()


# NOTE(cmo): This error display mechanism is pretty simple and can't handle
# overlapping messages (would need a list for that). This is likely fine unless
# abused.
@dataclass
class MessageState:
    type: MessageType = MessageType.Info
    title: str = ""
    message: str = ""
    dirty: BooleanVar = field(default_factory=BooleanVar)




@dataclass
class VtState:
    toolbar: ToolbarState = field(default_factory=ToolbarState)
    table: TableState = field(default_factory=TableState)
    graph: GraphState = field(default_factory=GraphState)
    app: AppState = field(default_factory=AppState)
    camera: Optional[CameraBase] = None
    cam_show: CameraViewState = field(default_factory=CameraViewState)
    diameters: Optional[ImageDiameters] = None
    measure: MeasureStore = field(default_factory=MeasureStore)
    message: MessageState = field(default_factory=MessageState)
    arduino_controller: Optional[ArduinoController] = None
    pressure_controller: Optional[PressureController] = None
    servo: ServoSettingsState = field(default_factory=ServoSettingsState)
    pressure_protocol: PressureProtocolSettingsState = field(default_factory=PressureProtocolSettingsState)


def rasterise_camera_state(
    image: np.ndarray,
    state: RasterDrawState,
    diams: Optional[ImageDiameters] = None,
    filter_diams: bool = True,
    rotate_tracking: bool = True,

) -> np.ndarray:
    # NOTE(cmo): Draw ROI
    result = np.copy(image)
    ny, nx, _ = image.shape  # Adding an underscore to capture the number of color channels, if present

    roi = state.roi
    black = (10, 131, 135)
    C1_good = C1
    C2_good = C2
    C1_bad = (0, 0, 0)
    C2_bad = (90, 90, 90)
    cmap = CMAP.colors
    if roi is not None:
        x1, y1, x2, y2 = roi.fixed_corners()
        cv2.rectangle(result, (x1, y1), (x2, y2), black, 2)

    

    # NOTE(cmo): Draw calipers
    cal = state.caliper
    if cal is not None:
        cv2.line(result, (cal.x1, cal.y1), (cal.x2, cal.y2), black, 3)

    for idx, roi in enumerate(state.multi_roi.values()):
        colour = [int(255 * x) for x in cmap[idx]]
        x1, y1, x2, y2 = roi.fixed_corners()
        cv2.rectangle(result, (x1, y1), (x2, y2), colour, 2)

    for idx, cal in enumerate(state.autocaliper.values()):
        colour = [int(255 * x) for x in cmap[idx]]
        cv2.line(result, (cal.x1, cal.y1), (cal.x2, cal.y2), colour, 2)

    

    # Drawing diameter overlays
    if diams is not None:
        for idx in range(diams.outer_diam_x.shape[0]):
            od_x = diams.outer_diam_x[idx].astype(np.int32)
            od_y = diams.outer_diam_y[idx].astype(np.int32)
            id_x = diams.inner_diam_x[idx].astype(np.int32)
            id_y = diams.inner_diam_y[idx].astype(np.int32)

            if np.all(od_x == 0) and np.all(id_x == 0):
                continue  # Skip if diameter coordinates are all zeros

            # Check if rotate_tracking was applied during diameter calculation
            if rotate_tracking:
                # Adjusting for 90 degrees counterclockwise rotation if rotate_tracking is True
                od_x, od_y = nx - od_y, od_x  # Flip x-axis
                id_x, id_y = nx - id_y, id_x  # Flip x-axis


            # Drawing diameters with colors based on outlier status and filter setting
            od_colour = C1_bad if filter_diams and diams.od_outliers[idx] else C1_good
            id_colour = C2_bad if filter_diams and diams.id_outliers[idx] else C2_good

            # Draw outer and inner diameters with caps
            line_with_caps(result, (od_x[0], od_y[0]), (od_x[1], od_y[1]), od_colour, 4, rotated=rotate_tracking)
            line_with_caps(result, (id_x[0], id_y[0]), (id_x[1], id_y[1]), id_colour, 4, rotated=rotate_tracking)

    return result


def line_with_caps(result, pt1, pt2, color, thickness, cap_half_height=6, rotated=False):
    cv2.line(result, pt1, pt2, color, thickness)
    if rotated:
        # Horizontal caps for a line considered vertical in rotated space
        cv2.line(
            result,
            (pt1[0] - cap_half_height, pt1[1]),
            (pt1[0] + cap_half_height, pt1[1]),
            color,
            thickness,
        )
        cv2.line(
            result,
            (pt2[0] - cap_half_height, pt2[1]),
            (pt2[0] + cap_half_height, pt2[1]),
            color,
            thickness,
        )
    else:
        # Regular vertical caps
        cv2.line(
            result,
            (pt1[0], pt1[1] - cap_half_height),
            (pt1[0], pt1[1] + cap_half_height),
            color,
            thickness,
        )
        cv2.line(
            result,
            (pt2[0], pt2[1] - cap_half_height),
            (pt2[0], pt2[1] + cap_half_height),
            color,
            thickness,
        )


@dataclass
class DiamsAndRasterResult:
    frame_id: int
    frame_time: float
    diameters: Optional[ImageDiameters]
    raw_im: np.ndarray
    rasterised: np.ndarray


def compute_diameters_and_rasterise(
    im: np.ndarray,
    raster_draw_state: RasterDrawState,
    frame_id: int,
    frame_time: float,
    compute_id: bool,
    default_detection_alg: bool,
    lines_to_avg: int,
    num_lines: int,
    scale: float,
    smooth_factor: int,
    thresh_factor: float,
    filter_diams: bool,
    rotate_tracking: bool,
    ultrasound_tracking: bool,
):
    #print ("frame id == ", frame_id)
    diams = calculate_diameter(
        image=im,
        rds=raster_draw_state,
        compute_id=compute_id,
        default_detection_alg=default_detection_alg,
        lines_to_avg=lines_to_avg,
        num_lines=num_lines,
        scale=scale,
        smooth_factor=smooth_factor,
        thresh_factor=thresh_factor,
        filter_means=filter_diams,
        rotate_tracking=rotate_tracking,
        ultrasound_tracking=ultrasound_tracking,
    )
    image_colour = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    rasterised = rasterise_camera_state(
        image_colour,
        raster_draw_state,
        diams,
        filter_diams=filter_diams,
        rotate_tracking=rotate_tracking,
    )
    return DiamsAndRasterResult(
        frame_id=frame_id,
        frame_time=frame_time,
        diameters=diams,
        raw_im=im,
        rasterised=rasterised,
    )


@dataclass
class FutureAndCallbackFlag:
    future: Future
    callback_bound: bool = False


class Model:
    def __init__(self, mmc: CMMCorePlus, set_timeout):
        self.pressure_controller = None
        self.state = VtState()
        self.set_timeout = set_timeout
        self.run_acq_thread = True
        self.acquiring = False
        self.file_analysed = False
        self.tracking = False
        self.tracking_file = False
        self.notepad_path = None
        self.queue = queue.Queue()
        self.mmc = mmc
        self.config_path = "settings.toml"
        self.current_table_row = 1  # Initialize row number to 0

        # For saving the output tiffs
        self.output_path1 = None
        self.output_path2 = None
        self.tiff_writer1 = None
        self.tiff_writer2 = None


        try:
            self.configure = Config.from_file(Path(__file__).parent / self.config_path)
        except:
            traceback.print_exc()
            self.state.message.type = MessageType.Warning
            self.state.message.title = "Failed to load config"
            self.state.message.message = (
                f"Failed to load config from path... loading defaults: {Path(__file__).parent / self.config_path}."
            )
            self.state.message.dirty.set(True)
            self.configure = Config(path=self.config_path)

        self.configure.set_values(self.state)
        self.setup_thread_pool()

        self.sleep_duration = self.configure.acquisition.refresh_min_interval
        self.start_time = 0.0
        self.prev_update = 0.0
        self.time_elapsed = 0.0
        self.frame_count = 0

        self.setup_default_ui_state()

        self.register_callbacks()

        self.worker = threading.Thread(target=self.acq_thread)
        self.worker.start()

    def set_pressure_controller(self, pressure_controller):
        self.pressure_controller = pressure_controller

    def set_arduino_controller(self, arduino_controller):
        self.arduino_controller = arduino_controller

    def setup_output_files(self, output_path):
        """Needs to be called before acquiring anything"""
        self.output_path = output_path
        self.output_dir, self.output_filename = os.path.split(output_path)
        self.output_stem = os.path.splitext(self.output_filename)[0]

        # NOTE(cmo): Setup output file
        # NOTE(cmo): This nested output file can be worked with pretty easily by
        # splitting the each of the quoted variadic columns in Excel's
        # PowerQuery (better than text to columns as it won't overwrite).
        self.output_file = open(self.output_path, "w", newline="")
        self.output_writer = csv.writer(self.output_file)
        self.output_writer.writerow(self.state.measure.headers())
        self.output_file.flush()

        self.notepad_path = os.path.splitext(output_path)[0] + "_notes" + ".txt"

        self.table_path = os.path.splitext(output_path)[0] + "_table" + ".csv"
        self.table_file = open(self.table_path, "w", newline="")
        self.table_writer = csv.writer(self.table_file)
        self.table_writer.writerow(self.state.table.headers())
        self.table_file.flush()

        tb = self.state.toolbar
        tb.source.path.set(self.output_dir)
        tb.source.filename.set(self.output_filename)

    def setup_default_ui_state(self):
        """Set up UI element state that isn't part of the config"""
        tb = self.state.toolbar
        tb.acq.default_settings.set(True)
        tb.source.settings.set(self.config_path)
        tb.acq.fast_mode.set(False)

        tb.analysis.filter.set(True)
        tb.analysis.ID.set(True)
        tb.analysis.org.set(False)

        tb.caliper_roi.roi_flag.set("ROI")

        tb.pressure_protocol.hold_pressure.set(True)

        tb.start_stop.record.set(True)

    def setup_default_ui_state_loadfile(self):
        tb = self.state.toolbar
        tb.acq.default_settings.set(False)
        tb.acq.fast_mode.set(True)
        tb.acq.rec_interval.set(1)

    def setup_thread_pool(self):
        num_threads = self.configure.analysis.num_threads
        try:
            if self.executor is not None:
                self.executor.shutdown(wait=False)
        except AttributeError:
            pass

        if num_threads <= 1:
            self.executor = None
        else:
            self.executor = ProcessPoolExecutor(max_workers=num_threads)
        self.futures_to_resolve = deque()

    def load_config(self, config: Config):
        self.configure = config
        config.set_values(self.state)
        self.config_path = config.path
        self.state.toolbar.source.settings.set(os.path.split(self.config_path)[1])
        # NOTE(cmo): refresh graph lims
        self.state.toolbar.graph.dirty.set(True)
        self.setup_thread_pool()

    def to_config(self):
        config = Config.from_state(self.state)
        return config

    def get_shutdown_callback(self):
        def cb():
            self.run_acq_thread = False
            if self.state.camera is not None:
                self.state.camera.shutdown()

        return cb

    def register_callbacks(self):
        tb = self.state.toolbar

        def handle_exposure(*args):
            if self.state.camera is None:
                return
            try:
                # Attempt to get the entry value
                exposure_entry = self.state.toolbar.acq.exposure.get()

                # If entry is empty, default to 1
                if exposure_entry.strip() == "":
                    exposure_entry = 1
                else:
                    # Convert to float first (to support decimal values) and then to int
                    exposure_entry = int(float(exposure_entry))

            except:
                # If .get() fails or contains non-numeric data, set a safe default
                exposure_entry = 1
            exposure = np.clip(exposure_entry, 1, 500)
            self.state.camera.set_exposure(exposure)
            if exposure != exposure_entry:
                self.state.toolbar.acq.exposure.set(exposure)

        tb.acq.exposure.trace_add("write", handle_exposure)

        def handle_pix_clock(*args):
            if self.state.camera is None:
                return
            pix_clock = tb.acq.pixel_clock.get()
            self.set_camera_pix_clock(pix_clock)

        tb.acq.pixel_clock.trace_add("write", handle_pix_clock)

        # NOTE(cmo): Set self.acquiring directly off the app.acquiring variable
        def set_acquiring(*args):
            acquiring = self.state.app.acquiring.get()
            if self.state.camera is not None:
                if acquiring:
                    try:
                        self.state.camera.start_acquisition()
                    except:
                        pass
                else:
                    self.state.camera.stop_acquisition()
            self.acquiring = acquiring

        self.state.app.acquiring.trace_add(
            "write",
            set_acquiring,
        )




        # NOTE(cmo): Set self.tracking directly off the app.tracking variable
        def set_tracking(*args):
            tracking = self.state.app.tracking.get()
            if self.state.camera is not None:
                if tracking:
                    try:
                        self.state.camera.start_acquisition()
                    except:
                        pass
                else:
                    self.state.camera.stop_acquisition()
            self.tracking = tracking

        self.state.app.tracking.trace_add(
            "write",
            set_tracking,
        )

        # NOTE(cmo): Set self.tracking_file directly off the app.tracking variable
        def set_tracking_file(*args):
            tracking_file = self.state.app.tracking_file.get()
            self.tracking_file = tracking_file

        self.state.app.tracking_file.trace_add(
            "write",
            set_tracking_file,
        )


        def set_acq_thread_sleep(*args):
            if self.state.toolbar.acq.fast_mode.get():
                self.sleep_duration = self.configure.acquisition.refresh_faster_interval
            else:
                self.sleep_duration = self.configure.acquisition.refresh_min_interval
        self.state.toolbar.acq.fast_mode.trace_add("write", set_acq_thread_sleep)

        def update_scale(*args):
            scale = tb.acq.scale.get()
            if (cal := self.state.cam_show.raster_draw_state.caliper) is not None:
                tb.data_acq.caliper_length.set(cal.length * scale)

        tb.acq.scale.trace_add("write", update_scale)

    def process_images(self):
        got_im = False
        while not self.queue.empty():
            im = self.queue.get(block=False)
            got_im = True

        if not got_im:
            return

        tb = self.state.toolbar
        current_time = time.time()

        if self.start_time == 0:
            if self.tracking:
                self.start_time = current_time
                self.frames_elapsed = 0

        if self.executor is None:
            result = compute_diameters_and_rasterise(
                im=im,
                raster_draw_state=self.state.cam_show.raster_draw_state,
                frame_id=self.frame_count,
                frame_time=current_time,
                compute_id=tb.analysis.ID.get(),
                default_detection_alg=tb.analysis.org.get(),
                lines_to_avg=tb.analysis.integration_factor.get(),
                num_lines=tb.analysis.num_lines.get(),
                scale=tb.acq.scale.get(),
                smooth_factor=tb.analysis.smooth_factor.get(),
                thresh_factor=tb.analysis.thresh_factor.get(),
                filter_diams=tb.analysis.filter.get(),
                rotate_tracking=tb.analysis.rotate_tracking.get(),
                ultrasound_tracking=tb.analysis.ultrasound_tracking.get(),
            )
            self.complete_processing(result)
        else:
            future = self.executor.submit(
                compute_diameters_and_rasterise,
                im=im,
                raster_draw_state=self.state.cam_show.raster_draw_state,
                frame_id=self.frame_count,
                frame_time=current_time,
                compute_id=tb.analysis.ID.get(),
                default_detection_alg=tb.analysis.org.get(),
                lines_to_avg=tb.analysis.integration_factor.get(),
                num_lines=tb.analysis.num_lines.get(),
                scale=tb.acq.scale.get(),
                smooth_factor=tb.analysis.smooth_factor.get(),
                thresh_factor=tb.analysis.thresh_factor.get(),
                filter_diams=tb.analysis.filter.get(),
                rotate_tracking=tb.analysis.rotate_tracking.get(),
                ultrasound_tracking=tb.analysis.ultrasound_tracking.get(),
            )
            self.futures_to_resolve.append(FutureAndCallbackFlag(future))
            self.resolve_next_pending_future()

            
        self.frame_count += 1

    def resolve_next_pending_future(self):
        def resolve_future(f: Future):
            try:
                self.complete_processing(f.result())
            except:
                traceback.print_exc()
            # NOTE(cmo): This callback is only ever bound to the first
            # future at a time. This means we can do the pop and shuffle of
            # remaining jobs _in_ the callback, whilst still ensuring ordering.
            self.futures_to_resolve.popleft()
            self.resolve_next_pending_future()

        if len(self.futures_to_resolve) > 0:
            f: FutureAndCallbackFlag = self.futures_to_resolve[0]
            if not f.callback_bound:
                f.callback_bound = True
                f.future.add_done_callback(resolve_future)

    def complete_processing(self, result: DiamsAndRasterResult):

        # Working here
        # This section should probably be in the processing part of model.
        if self.state.graph.clear.get():
            self.state.measure.clear()
            #self.state.graph.clear()
            self.state.graph.clear.set(False)    # TODO: Add other measures here.

        tb = self.state.toolbar
        # NOTE(cmo): Condition added to show image when scrolling through image from file
        if self.tracking or self.state.camera.camera_name == "Image from file":
            self.state.diameters = result.diameters
            # NOTE(cmo): Drop frames if the UI can't keep up
            if not self.state.cam_show.dirty.get():
                self.state.cam_show.raw_im_data = result.raw_im
                self.state.cam_show.im_data = result.rasterised
                self.state.cam_show.dirty.set(True)
            
            if not self.tracking:
                return
            

            current_time = result.frame_time
            time_elapsed = current_time - self.start_time
            self.time_elapsed = time_elapsed

            if self.state.camera.camera_name == "Image from file":
                self.frames_elapsed += 1
                self.time_elapsed = self.frames_elapsed

            diams = self.state.diameters
            #print("Length of diameter avg: ", len(diams.avg_outer_diam))
            record_data = self.state.toolbar.start_stop.record.get()
            rec_interval = self.state.toolbar.acq.rec_interval.get()

            if record_data and int(self.time_elapsed) % rec_interval == 0:
                # Save the raw and rasterised images
                # ----------------------------------
                self.save_image(result.raw_im, subdir1="Raw")
                self.save_image(result.rasterised, subdir2="Result")
        else:
            self.state.diameters = None
            diams = self.state.diameters
            # NOTE(cmo): Drop frames if the UI can't keep up
            if not self.state.cam_show.dirty.get():
                self.state.cam_show.raw_im_data = result.raw_im
                self.state.cam_show.im_data = result.raw_im
                self.state.cam_show.dirty.set(True)

        if diams is not None:

            marker = 0
            if self.state.table.dirty_marker.get():
                marker = 1
                self.state.table.dirty_marker.set(False)


            # Record measurements
            # -------------------
            self.state.measure.append(
                t=self.time_elapsed,
                od=diams.avg_outer_diam,
                id=diams.avg_inner_diam,
                marker=marker,
                temperature=tb.data_acq.temperature.get(),
                pavg = tb.data_acq.pressure.get(),
                p1 = self.state.arduino_controller.measured_pressure_1 if self.state.arduino_controller.measured_pressure_1 is not None else np.nan,
                p2 = self.state.arduino_controller.measured_pressure_2 if self.state.arduino_controller.measured_pressure_2 is not None else np.nan,
                set_p = tb.pressure_protocol.set_pressure.get(),
                caliper_length=tb.data_acq.caliper_length.get(),
                ods=diams.outer_diam,
                ids=diams.inner_diam,
                ods_valid=~diams.od_outliers,
                ids_valid=~diams.id_outliers,
            )

            tracking = self.state.app.tracking.get()
            if tracking:
                self.output_writer.writerow(self.state.measure.get_last_row())
                self.output_file.flush()

        # NOTE(cmo): Drop frames if the UI can't keep up
        if diams is not None and not self.state.graph.dirty.get():
            # Add measurements to plot
            # ------------------------
            rds = self.state.cam_show.raster_draw_state
            graph = self.state.graph
            have_autocaliper = len(rds.autocaliper) > 0
            have_multi_roi = len(rds.multi_roi) > 0

            max_pts = min(
                self.configure.memory.num_plot_points,
                len(self.state.measure.times),
            )

            measure = self.state.measure
            new_x = np.asarray(measure.times[-max_pts:]) - measure.times[-1]

            # NOTE(cmo): Average size
            od_ordinates = np.asarray(measure.outer_diam[-max_pts:])
            id_ordinates = np.asarray(measure.inner_diam[-max_pts:])
            marker_ordinates = np.asarray(measure.markers[-max_pts:])
            graph.od_avg.x = new_x
            graph.od_avg.y = od_ordinates
            graph.id_avg.x = new_x
            graph.id_avg.y = id_ordinates
            graph.markers.x = new_x
            graph.markers.y = marker_ordinates

            if have_autocaliper or have_multi_roi:
                filter_diams=tb.analysis.filter.get()
                def compute_masked_diams(diam_list, good_list):
                    masked_diams = []
                    for d, good in zip(diam_list[-max_pts:], good_list[-max_pts:]):
                        masked_d = d.copy()
                        if filter_diams:
                            masked_d[~good] = np.nan
                        masked_diams.append(masked_d)
                    return masked_diams

                def sample_masked_diams(masked_diams, idx):
                    result = []
                    for md in masked_diams:
                        if idx < md.shape[0]:
                            result.append(md[idx])
                        else:
                            result.append(np.nan)
                    return result

                masked_ods = compute_masked_diams(
                    measure.outer_diam_profile, measure.outer_diam_good
                )
                masked_ids = compute_masked_diams(
                    measure.inner_diam_profile, measure.inner_diam_good
                )

                '''
                This should probably be moved somehwere better.
                '''
                for i in range(NUM_LINES):

                    # Ensure the key exists in the dictionary, if not, initialize with an empty list
                    if i not in measure.outer_diam_roi:
                        measure.outer_diam_roi[i] = []

                    if i not in measure.inner_diam_roi:
                        measure.inner_diam_roi[i] = []

                    

                    graph.od_lines[i].x = new_x
                    graph.od_lines[i].y = sample_masked_diams(masked_ods, i)
                    graph.id_lines[i].x = new_x
                    graph.id_lines[i].y = sample_masked_diams(masked_ids, i)


                    if len(graph.od_lines[i].y) > 0 and not np.all(np.isnan(graph.od_lines[i].y)):
                        line_od_value = np.round(np.nanmean(graph.od_lines[i].y), 1)
                    else:
                        # If the od_lines array is empty, set a default value (e.g., np.nan)
                        line_od_value = np.nan

                    if len(graph.id_lines[i].y) > 0 and not np.all(np.isnan(graph.id_lines[i].y)):
                        line_id_value = np.round(np.nanmean(graph.id_lines[i].y), 1)
                    else:
                        # If the id_lines array is empty, set a default value (e.g., np.nan)
                        line_id_value = np.nan

                    #Store the values from the od_rois.
                    measure.outer_diam_roi[i].append(line_od_value)
                    measure.inner_diam_roi[i].append(line_id_value)

                    # This is used to update the variable in the entry box the show/ hides traces.
                    tb.plotting.outer_diam_values[i].set(line_od_value)
                    tb.plotting.inner_diam_values[i].set(line_id_value)
                    


                    

            # If image is from file, only update the graph on the last frame.
            if self.state.camera.camera_name == "Image from file":
                last_frame = self.state.camera.max_frame_count
                current_frame = self.state.camera.frame_count
                #print ("last_frame == ", current_frame)
                #print ("last_frame == ", last_frame)

                if current_frame == last_frame -1:
                    graph.dirty.set(True)
            else:
                graph.dirty.set(True)

        if diams is not None:
            if self.prev_update == 0:
                acq_rate = 0.0
            else:
                acq_rate = 1.0 / (current_time - self.prev_update)
            self.prev_update = current_time
            tb.acq.acq_rate.set(np.round(acq_rate, 2))
            tb.data_acq.time.set(np.round(time_elapsed, 1))
            formatted_time = time.strftime("%H:%M:%S", time.gmtime(np.round(time_elapsed, 1)))
            tb.data_acq.time_string.set(formatted_time)
            if diams is not None:
                tb.data_acq.outer_diam.set(np.round(diams.avg_outer_diam, 1))
                tb.data_acq.inner_diam.set(np.round(diams.avg_inner_diam, 1))
            ref_diam = self.state.table.ref_diam.get()
            if not np.isnan(ref_diam) and ref_diam != 0.0:
                outer_percentage = np.round((diams.avg_outer_diam / ref_diam) * 100, 2)
                tb.data_acq.diam_percent.set(outer_percentage)



    def initialize_tiff_writer1(self):
        print(self.output_path1)
        self.tiff_writer1 = tf.TiffWriter(self.output_path1, mode='w')

    def initialize_tiff_writer2(self):
        print(self.output_path2)
        self.tiff_writer2 = tf.TiffWriter(self.output_path2, mode='w')

    def close_tiff_writers(self):
        if self.tiff_writer1 is not None:
            self.tiff_writer1.close()
        if self.tiff_writer2 is not None:
            self.tiff_writer2.close()

    def save_snapshot(self, image: np.ndarray, subdir: Optional[str] = None):
        current_time = int(self.time_elapsed)
        current_frame = int(self.frame_count)
        directory = Path(self.output_dir) / "snapshots"
        if subdir is not None:
            directory = directory / subdir
        directory.mkdir(parents=True, exist_ok=True)
        image_path = directory / f"{self.output_stem}[t={current_time:06d}].tiff"
        skimage.io.imsave(image_path, image)

    def save_image(self, image: np.ndarray, subdir1: Optional[str] = None, subdir2: Optional[str] = None, metadata: Optional[dict] = None):
        directory = Path(self.output_dir)

        if self.tiff_writer1 is None and subdir1 is not None:
            self.output_path1 = directory / f"{self.output_stem}_{subdir1}.tiff"
            self.initialize_tiff_writer1()
        elif self.tiff_writer2 is None and subdir2 is not None:
            self.output_path2 = directory / f"{self.output_stem}_{subdir2}.tiff"
            self.initialize_tiff_writer2()

        # Default metadata if none provided
        if metadata is None:
            metadata = {}

        # Add standard metadata
        default_metadata = {
            'Timestamp': datetime.now().isoformat(),
            'FrameNumber': self.frame_count,
            'TimeElapsed': self.time_elapsed,
            'FilePath': str(self.output_path1 if subdir1 is not None else self.output_path2)
        }

        # Merge user provided metadata with default metadata
        combined_metadata = {**default_metadata, **metadata}

        # Convert metadata to JSON string for storage
        metadata_json = json.dumps(combined_metadata)

        if subdir1 is not None:
            description = metadata_json
            self.tiff_writer1.write(image, description=description)
        elif subdir2 is not None:
            description = metadata_json
            self.tiff_writer2.write(image, description=description)

    def process_updates(self):
        tb = self.state.toolbar
        try:
            self.process_images()
        except:
            traceback.print_exc()

        # need to update the timer here
        if self.state.toolbar.pressure_protocol.pressure_protocol_flag.get() == 1:
            #update the timer here
            # new if based on timer to set pressure
             #Timenow + interval = next pressure
            try:
                self.pressure_controller.update_intvl()
            except:
                traceback.print_exc()

        else:
            pass

        temppres = self.arduino_controller.getData()
        self.measured_pressure_1, self.measured_pressure_2, self.measured_pressure_avg, self.measured_temperature = self.arduino_controller.sortdata(temppres)
        if self.measured_temperature:
            tb.data_acq.temperature.set(np.round(self.measured_temperature, 1))
        if self.measured_pressure_avg:
            tb.data_acq.pressure.set(np.round(self.measured_pressure_avg, 1))
        #tb.data_acq.temperature.set(np.round(self.measured_temperature, 2))
        #tb.data_acq.temperature.set(np.round(self.measured_temperature, 2))

        if self.run_acq_thread:
            # NOTE(cmo): This is only set False when we're exiting, at which
            # point stop handling future events
            self.set_timeout(10, self.process_updates)

    ##### WORKING HERE


    def acq_thread(self):
        """
        Acquisition thread function responsible for managing image acquisition,
        buffering, and processing for both live camera and file-based camera data.

        The function continuously checks for new images from the camera and updates
        the queue with the latest image. It manages specific logic for when data is
        being loaded from a file and when dealing with live camera feeds. Additionally,
        it handles edge cases such as ensuring the camera buffer is not empty and
        controlling the slider position when working with image files.

        Behavior Overview:
            1. Continuously checks if the acquisition thread should keep running (`self.run_acq_thread`).
            2. If acquiring and no image is available in the queue, it fetches an image from the camera.
            3. For live camera feeds:
                - Ensures that the camera buffer is not empty before retrieving an image.
            4. For image files:
                - Retrieves images based on the slider position and updates the slider.
            5. Stops acquiring and resets the system when the last frame is reached in file-based acquisition.
            6. Handles special cases for loading data and updates the system state accordingly.

        Detailed Steps:
            1. Checks if the queue is empty and if acquiring is active.
            2. If acquiring, retrieves the current image from the camera:
                - For live cameras, checks the buffer status and ensures it is not empty before fetching an image.
                - For image files, retrieves the image at the current slider position.
            3. Puts the retrieved image into the processing queue.
            4. Special case for image files:
                - Updates the slider and checks if the last frame has been reached. If the last frame is reached, acquisition and tracking are stopped, and the system is reset.
            5. If acquisition is off and file-based tracking is active:
                - Retrieves an image based on the slider position and updates diameter values.
                - Ensures the slider position has changed before updating.
                - Avoids unnecessary fast spinning on the same frame.
            6. Sleep duration is adjusted depending on the state of acquisition.

        Attributes:
            self.run_acq_thread (bool): Flag controlling the thread's active state.
            self.sleep_duration (float): Duration for which the thread sleeps between iterations.
            self.state: Contains various state attributes related to camera, acquisition, tracking, toolbar, etc.
            self.queue: Queue for storing acquired images for processing.
            self.prev_slider_index (int): The previously used slider index for tracking changes.
        """

        while self.run_acq_thread:
            camera = self.state.camera
            file_analysed = self.state.app.file_analysed.get()
            '''
            Whenever an image is shown acquiring must be set to true, otherwise an image will not show.
            Therefore, the logic below cannot rely on acquire == True as this will always be the case when showing an image.
            '''
            sleep_duration = self.sleep_duration
            if self.queue.empty() and self.acquiring and not file_analysed:
                if self.state.camera.camera_name == "Image from file" and not self.tracking:
                    '''
                    Load an image and have it show. The image will not be analysed until the analyse button is pressed and self.tracking is True.
                    '''
                    slider_img = camera.get_specific_frame(self.state.cam_show.slider_position_manual)
                    slider_index = int(self.state.cam_show.slider_position_manual)
                    self.queue.put(slider_img)
                    self.queue.empty()
                    # NOTE(cmo): Don't spin super fast on the same frame in this state!
                    sleep_duration *= 10

                elif self.state.camera.camera_name == "Image from file" and self.tracking and self.tracking_file:
                    '''
                    Analyse the loaded file when the analyse button is pressed.
                    '''
                    camera.next_position(self.state.app.tracking.get())
                    self.state.cam_show.slider_dirty.set(True) # Set the slider to the current potition
                    last_frame = self.state.camera.max_frame_count
                    current_frame = self.state.camera.frame_count

                    # Clear table and graph
                    if current_frame == 1:
                        self.state.table.clear.set(True)
                        self.state.graph.clear.set(True)

                    # Stop analysis and show graph.
                    if current_frame == last_frame:
                        self.state.app.acquiring.set(1)
                        self.state.app.tracking.set(0)
                        self.state.app.file_analysed.set(1)
                        self.state.camera.reinitialize()
                        self.frames_elapsed = 0

                        # Update the slider to the current position
                        self.state.cam_show.slider_toggle_dirty.set(True)

                        # Update the graph
                        self.state.toolbar.graph.limits_dirty.set(True)

                        # Enable the slider
                        self.state.cam_show.slider_change_state.set(True)
                    
                    # Show the images as we analyse them
                    slider_img = camera.get_specific_frame(self.state.cam_show.slider_position_manual)
                    self.queue.put(slider_img)
                    self.queue.empty()

                else:
                    camera = self.state.camera
                    if camera and camera.image_ready():
                        # Need to make sure circular buffer has not reset for uManager cameras (crashes if the buffer is 0)
                        buffer = camera.is_buffer_empty()
                        # Logic: If not Offline Analyzer and if live camera buffer is empty, then do not try to get an image. Otherwise, get an image.
                        if not self.state.camera.camera_name == "Image from file":
                            if buffer < 1:
                                time.sleep(sleep_duration)
                                continue
                            else:
                                pass
                        else:
                            pass
                        try:
                            img = camera.get_image()
                        except:
                            pass
                        self.queue.put(img)
                    # NOTE(cmo): Don't spin super fast on the same frame in this state!
                    sleep_duration *= 10

            elif self.acquiring and self.tracking_file and file_analysed:
                '''
                This runs after we have analysed a file and allows us to scroll through the images and plotted graph.
                '''
                camera = self.state.camera
                if self.state.camera.camera_name == "Image from file":
                    slider_img = camera.get_specific_frame(self.state.cam_show.slider_position_manual)
                    slider_index = int(self.state.cam_show.slider_position_manual) - 1

                    self.state.toolbar.data_acq.outer_diam.set(np.round(self.state.measure.outer_diam[slider_index], 1))
                    self.state.toolbar.data_acq.inner_diam.set(np.round(self.state.measure.inner_diam[slider_index], 1))
                    # If no multi ROIs have been drawn then an error is returned as cant set the value. There way be a better way to do this other than try.
                    try:
                        for i in range(NUM_LINES):
                            self.state.toolbar.plotting.outer_diam_values[i].set(self.state.measure.outer_diam_roi[i][slider_index])
                            self.state.toolbar.plotting.inner_diam_values[i].set(self.state.measure.inner_diam_roi[i][slider_index])
                    except:
                        pass
                    self.prev_slider_index = slider_index
                    self.queue.put(slider_img)
                    self.queue.empty() #This stops the program from constantly trying to find a new image and analysing it when running from file.

                    # NOTE(cmo): Don't spin super fast on the same frame in this state!
                    sleep_duration *= 10         
            else:
                pass
            time.sleep(sleep_duration)

    def set_default_graph_lims(self):
        defaults = GraphAxisSettings()
        defaults.set_values(self.state)
        self.state.toolbar.graph.dirty.set(True)

    def set_default_acq_settings(self):
        defaults = AcquisitionSettings()
        defaults.set_values(self.state)

    def set_ref_diameter(self):
        if self.state.diameters is not None:
            ref_diam = self.state.diameters.avg_outer_diam
            self.state.table.ref_diam.set(np.round(ref_diam, 2))

    def set_camera(self, cam_name):
        if cam_name == ELLIPSIS:
            return

        if self.state.camera is not None:
            self.state.camera.reset()

        if self.state.camera and cam_name == "Image from file":
            self.state.camera.reset()
            self.mmc.unloadAllDevices()
            self.mmc.reset()

        try:
            self.state.camera = Camera(cam_name, self.mmc, self.state, self.configure)
            image_dim = self.state.toolbar.image_dim
            if cam_name == "Image from file":
                w, h, l = self.state.camera.get_camera_dims()
                image_dim.file_length.set(l)
                self.state.cam_show.slider_length_dirty.set(True)
            else:
                w, h = self.state.camera.get_camera_dims()
            image_dim.cam_width.set(w)
            image_dim.cam_height.set(h)
            image_dim.fov_width.set(w)
            image_dim.fov_height.set(h)
        except Exception as e:
            print(f"An error occurred: {e}")
            self.state.camera.reset()
            self.mmc.unloadAllDevices()
            self.mmc.reset()


    def set_camera_pix_clock(self, pix_clock, quiet_fail=True):
        if self.state.camera is None:
            return
        try:
            self.state.camera.set_pixel_clock(pix_clock)
        except:
            traceback.print_exc()
            if not quiet_fail:
                self.state.message.type = MessageType.Error
                self.state.message.title = "Set pix clock Failed"
                self.state.message.message = "Failed to set pixel clock"
                self.state.message.dirty.set(True)

    def set_camera_resolution(self, x, y):
        if self.state.camera is None:
            return
        try:
            self.state.camera.set_resolution(x, y)
        except NotImplementedError as e:
            # NOTE(cmo): Pop up an error
            self.state.message.type = MessageType.Error
            self.state.message.title = "Set Resolution Failed"
            self.state.message.message = e.args[0]
            self.state.message.dirty.set(True)

    def set_camera_fov(self, x, y, xSize, ySize):
        if self.state.camera is None:
            return
        try:
            self.state.camera.set_fov(x, y, xSize, ySize)
            image_dim = self.state.toolbar.image_dim
            image_dim.fov_width.set(xSize)
            image_dim.fov_height.set(ySize)
        except NotImplementedError as e:
            # NOTE(cmo): Pop up an error
            self.state.message.type = MessageType.Error
            self.state.message.title = "Set FOV Failed"
            self.state.message.message = e.args[0]
            self.state.message.dirty.set(True)

    def rerasterise_current_image(self):
        data = self.state.cam_show.im_data
        if data is None:
            return

        tb = self.state.toolbar
        filter_diams = tb.analysis.filter.get()
        self.state.diameters = calculate_diameter(
            image=self.state.cam_show.raw_im_data,
            rds=self.state.cam_show.raster_draw_state,
            compute_id=tb.analysis.ID.get(),
            default_detection_alg=tb.analysis.org.get(),
            lines_to_avg=tb.analysis.integration_factor.get(),
            num_lines=tb.analysis.num_lines.get(),
            scale=tb.acq.scale.get(),
            smooth_factor=tb.analysis.smooth_factor.get(),
            thresh_factor=tb.analysis.thresh_factor.get(),
            filter_means=filter_diams,
            rotate_tracking=tb.analysis.rotate_tracking.get(),
            ultrasound_tracking=tb.analysis.ultrasound_tracking.get(),
        )
        im_data = cv2.cvtColor(
            self.state.cam_show.raw_im_data,
            cv2.COLOR_GRAY2RGB,
        )
        rasterised = rasterise_camera_state(
            im_data,
            self.state.cam_show.raster_draw_state,
            self.state.diameters,
            filter_diams=filter_diams,
            rotate_tracking=tb.analysis.rotate_tracking.get(),
        )
        self.state.cam_show.im_data = rasterised
        self.state.cam_show.dirty.set(True)

    def set_roi(self, x1, y1, x2, y2):
        if self.state.cam_show.im_data is None:
            return

        self.state.cam_show.raster_draw_state.roi = Roi(
            x1=x1,
            x2=x2,
            y1=y1,
            y2=y2,
        )
        self.rerasterise_current_image()

    def delete_roi(self):
        self.state.cam_show.raster_draw_state.roi = None
        self.rerasterise_current_image()

    def set_caliper(self, x1, y1, x2, y2):
        if self.state.cam_show.im_data is None:
            return

        length = hypot(abs(x2 - x1), abs(y2 - y1))
        scaled_length = length
        if (scale := self.state.toolbar.acq.scale.get()) != 0.0:
            scaled_length = length * scale

        cal = Caliper(x1=x1, y1=y1, x2=x2, y2=y2, length=length)
        self.state.cam_show.raster_draw_state.caliper = cal
        self.state.toolbar.data_acq.caliper_length.set(np.round(scaled_length, 2))
        self.rerasterise_current_image()

    def delete_caliper(self):
        self.state.cam_show.raster_draw_state.caliper = None
        self.state.toolbar.data_acq.caliper_length.set(np.nan)
        self.rerasterise_current_image()

    def add_multi_roi(self, x1, y1, x2, y2):
        if self.state.cam_show.im_data is None:
            return

        state = self.state.cam_show.raster_draw_state
        if len(state.multi_roi) >= NUM_ROIS:
            return
        roi = Roi(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
        )
        key = f"ROI{len(state.multi_roi)}"
        state.multi_roi[key] = roi
        try:
            self.rerasterise_current_image()
        except:
            traceback.print_exc()

    def delete_most_recent_multi_roi(self):
        state = self.state.cam_show.raster_draw_state
        if (idx := len(state.multi_roi)) == 0:
            return

        key = f"ROI{idx-1}"
        del state.multi_roi[key]
        self.rerasterise_current_image()

    def delete_all_multi_roi(self):
        state = self.state.cam_show.raster_draw_state
        if not state.multi_roi:  # Check if autocaliper is empty
            return

        # Create a list of all keys in autocaliper
        keys_to_delete = list(state.multi_roi.keys())
        # Iterate over the list and delete each key from the autocaliper
        for key in keys_to_delete:
            del state.multi_roi[key]

        self.rerasterise_current_image()


    def add_auto_caliper(self, x1, y1, x2, y2):
        if self.state.cam_show.im_data is None:
            return

        state = self.state.cam_show.raster_draw_state
        if len(state.autocaliper) >= NUM_LINES:
            return

        key = f"Caliper{len(state.autocaliper)}"

        length = hypot(abs(x2 - x1), abs(y2 - y1))
        cal = Caliper(x1=x1, y1=y1, x2=x2, y2=y2, length=length)
        state.autocaliper[key] = cal
        try:
            self.rerasterise_current_image()
        except:
            traceback.print_exc()

    def delete_most_recent_autocaliper(self):
        state = self.state.cam_show.raster_draw_state
        if (idx := len(state.autocaliper)) == 0:
            return

        key = f"Caliper{idx-1}"
        del state.autocaliper[key]
        self.rerasterise_current_image()

    def delete_all_autocaliper(self):
        state = self.state.cam_show.raster_draw_state
        if not state.autocaliper:  # Check if autocaliper is empty
            return

        # Create a list of all keys in autocaliper
        keys_to_delete = list(state.autocaliper.keys())
        # Iterate over the list and delete each key from the autocaliper
        for key in keys_to_delete:
            del state.autocaliper[key]

        self.rerasterise_current_image()


    def add_table_row(self):
        if self.state.diameters is None:
            return

        diams = self.state.diameters
        table = self.state.table
        label = table.label.get()
        ref_diam = table.ref_diam.get()
        percentage = (diams.avg_outer_diam / ref_diam) * 100.0
        percentage_as_str = str(np.round(percentage, 2))
        if np.isnan(ref_diam) or ref_diam == 0.0:
            percentage = np.nan
            percentage_as_str = "-"
        caliper_length = self.state.toolbar.data_acq.caliper_length.get()
        pavg = self.measured_pressure_avg
        p1 = self.measured_pressure_1
        p2 = self.measured_pressure_2
        temp = self.measured_temperature

        # Get the current number of rows in the table
        current_rows = len(table.rows_to_add) + 1

        values = [
            self.current_table_row,  # Add row number
            self.state.toolbar.data_acq.time_string.get(),#self.time_elapsed,
            self.frame_count,# Get the frame here
            label,
            diams.avg_outer_diam,
            percentage,
            diams.avg_inner_diam,
            caliper_length,
            pavg,
            p1,
            p2,
            temp
        ]
        self.table_writer.writerow(values)
        self.table_file.flush()

        disp_values = [
            str(self.current_table_row),  # Add row number
            self.state.toolbar.data_acq.time_string.get(),#str(np.round(self.time_elapsed, 2)),
            self.frame_count,
            label,
            str(np.round(diams.avg_outer_diam, 2)),
            percentage_as_str,
            str(np.round(diams.avg_inner_diam, 2)),
            str(caliper_length),
            str(np.round(pavg, 2)) if p1 is not None else "",
            str(np.round(p1, 2)) if p1 is not None else "",
            str(np.round(p2, 2)) if p2 is not None else "",
            str(np.round(temp, 2)) if p1 is not None else "",
        ]
        table.rows_to_add.append(disp_values)
        table.dirty.set(True)
        table.dirty_marker.set(True)
        self.current_table_row += 1

def make_entry_factory(self):
    def make_entry(EntryType: Type[tk.Widget], row, column=1, sticky="",padx=0, pady=2, disabled=False, **kwargs):
        # Set default width to 8 unless specified in kwargs
        kwargs.setdefault('width', 50)
        # NOTE(cmo): The need for this is due to tkinter being silly and
        # requiring *args be used for the options in an OptionMenu
        if "args" in kwargs:
            entry = EntryType(self, *kwargs["args"])
        else:
            entry = EntryType(self, **kwargs)
        entry.grid(row=row, column=column, padx=padx, pady=pady, sticky=sticky)
        if disabled:
            entry.configure(state=tk.DISABLED)
        return entry

    return make_entry


class ToolbarPane(ctk.CTkFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_lock_state(self, state=tk.DISABLED):
        pass

    def set_edit_state(self):
        self.set_lock_state(state=tk.NORMAL)

    def set_acquire_state(self):
        self.set_lock_state()


class SourcePane(ToolbarPane):
    def __init__(self, parent, model_vars: VtState):
        super().__init__(parent,  height=175, width=150)
        self.parent = parent
        self.model_vars = model_vars
        sv = model_vars.toolbar.source
        default_disabled = True
        make_entry = make_entry_factory(self)

        self.frame_label = ctk.CTkLabel(self, text="File details", font=(default_font, 16, 'bold'), fg_color=frame_label_color, height=frame_label_height, text_color='white').grid(row=0, column=0, columnspan=2,padx=1,pady=1, sticky="nsew")

        #self.pack(side=tk.LEFT, anchor=tk.N, padx=3, fill=tk.Y)

        ctk.CTkLabel(self, text="File:", font=(default_font, default_font_size)).grid(row=1, column=0, sticky=tk.E)
        self.save_entry = make_entry(
            ctk.CTkEntry,
            row=1,
            column=1,
            disabled=default_disabled,
            textvariable=sv.filename,
            font=(default_font, default_font_size),
            fg_color = entry_disabled_color,
            width=300,
        )

        ctk.CTkLabel(self, text="Settings:", font=(default_font, default_font_size)).grid(row=2, column=0, sticky=tk.E)
        self.settings_entry = make_entry(
            ctk.CTkEntry,
            row=2,
            column=1,
            disabled=default_disabled,
            textvariable=sv.settings,
            font=(default_font, default_font_size),
            fg_color = entry_disabled_color,
            width=300,
        )


class AcquisitionSettingsPane(ToolbarPane):
    def __init__(self, parent, model_vars: VtState, set_camera_callback):
        super().__init__(parent)
        self.set_camera_callback = set_camera_callback
        self.parent = parent
        self.model_vars = model_vars
        sv = model_vars.toolbar.acq
        make_entry = make_entry_factory(self)

        # Set a fixed size
        self.configure(width=220, height=150)  
        self.grid_propagate(False)  # Prevent shrinking/expanding if using grid
        self.pack_propagate(False)  # Prevent resizing if using pack

        # Apply grid column weight
        #self.columnconfigure(0, weight=1)  # Make column 0 expand
        #self.columnconfigure(1, weight=1)  # Make column 1 expand

        default_disabled = True
        self.pack(side=tk.LEFT, anchor=tk.N, padx=5, pady=5, fill=tk.Y)

        self.frame_label = ctk.CTkLabel(self, text="Acquisition Settings", font=(default_font, 16, 'bold'), fg_color=frame_label_color, height=frame_label_height, text_color='white').grid(row=0, column=0, columnspan=2,padx=1,pady=1, sticky="nsew")



        self.res_options = [
            "...",
            "320x240",
            "640x480",
            "720x480",
            "1280x720",
            "1280x960",
            "1280x1024",
        ]
        self.fov_options = ["w x h", "w/2 x h/2"]

        self.camera_options = ["..."] + list(Camera.registry.keys())


        sv.camera = tk.StringVar()  # Assuming sv.camera is a StringVar
        sv.camera.set(self.camera_options[0])  # Set the default value

        # Calculate the length of the longest string in camera_options
        max_length = max(len(option) for option in self.camera_options)

        entry_width = 50
        entry_height = 25
        padx=(0,10)

        ctk.CTkLabel(self, text="Camera:", font=(default_font, default_font_size)).grid(row=1, column=0, padx=padx, sticky=tk.E)
        self.camera_entry = ttk.OptionMenu(self, sv.camera, *self.camera_options, command=self.set_camera_callback)

        self.camera_entry.grid(row=1, column=1, sticky=tk.EW)

        #self.camera_entry.configure(width=8) #### Do not have this! It interferes with loading a file!
        default_disabled = True

        self.model_vars.toolbar.acq.camera.trace_add(
            "write", lambda *args: self.set_lock_state_when_no_camera()
        )

        ctk.CTkLabel(self, text="Scale (\u03bcm/px):", font=(default_font, default_font_size)).grid(row=2, column=0,padx=padx, sticky=tk.E)
        self.scale_entry = make_entry(
            ctk.CTkEntry,
            textvariable=sv.scale,
            font=(default_font, default_font_size),
            fg_color=entry_disabled_color,
            row=2,
            column=1,
            width = entry_width,
            height=entry_height,
            disabled=True,
            sticky=tk.W
        )

        ctk.CTkLabel(self, text="Exp (ms):", font=(default_font, default_font_size)).grid(row=3, column=0, padx=padx, sticky=tk.E)
        self.exposure_entry = make_entry(
            ctk.CTkEntry,
            textvariable=sv.exposure,
            font=(default_font, default_font_size),
            fg_color=entry_disabled_color,
            row=3,
            column=1,
            width = entry_width,
            height=entry_height,
            disabled=True,
            sticky=tk.W
        )

        ctk.CTkLabel(self, text="Acq rate (Hz):", font=(default_font, default_font_size)).grid(row=4, column=0, padx=padx, sticky=tk.E)
        self.acq_rate_entry = make_entry(
            ctk.CTkEntry,
            textvariable=sv.acq_rate,
            font=(default_font, default_font_size),
            fg_color=entry_disabled_color,
            row=4,
            column=1,
            width = entry_width,
            height=entry_height,
            disabled=True,
            sticky=tk.W
        )
        ctk.CTkLabel(self, text="Rec intvl (s):", font=(default_font, default_font_size)).grid(row=5, column=0, padx=padx, sticky=tk.E)
        self.rec_interval_entry = make_entry(
            ctk.CTkEntry,
            textvariable=sv.rec_interval,
            font=(default_font, default_font_size),
            fg_color=entry_disabled_color,
            row=5,
            column=1,
            width = entry_width,
            height=entry_height,
            disabled=True,
            sticky=tk.W
        )

        self.default_settings = ctk.CTkCheckBox(
            self,
            text="Default",
            font=(default_font, default_font_size),
            variable=sv.default_settings,
            checkbox_width=20,
            checkbox_height=20,
            border_width=2
        )
        self.default_settings.grid(row=6, column=0, padx=5, pady=0, sticky=tk.E)

        self.faster_settings = ctk.CTkCheckBox(
            self,
            text="Fast",
            font=(default_font, default_font_size),
            variable=sv.fast_mode,
            checkbox_width=20,
            checkbox_height=20,
            border_width=2
        )
        self.faster_settings.grid(row=6, column=1, padx=5, pady=0, sticky=tk.E)
        self.faster_settings.configure(state=tk.DISABLED)

        # Uncomment to add ability to set binning and FOV

        '''
        ctk.CTkLabel(self, text="Res:").grid(row=7, column=0, sticky=tk.E)
        self.res_entry = make_entry(
            ttk.OptionMenu,
            args=(sv.res, self.res_options[0], *self.res_options),
            disabled=default_disabled,
            row=7,
            column=1,
        )

        ctk.CTkLabel(self, text="FOV:").grid(row=8, column=0, sticky=tk.E)
        self.fov_entry = make_entry(
            ttk.OptionMenu,
            args=(sv.fov, self.fov_options[0], *self.fov_options),
            disabled=default_disabled,
            row=8,
            column=1,
        )
        '''

        self.setup_default_settings_lock()
        self.setup_faster_settings_warning()


        # Create a single tooltip instance for the container
        tooltip = ToolTip(self)

        # Bind tooltips to the buttons
        tooltips = {
            self.camera_entry: "Select your camera driver.",
            self.scale_entry: "Set the scale in micrometers per pixel.",
            self.exposure_entry: "Set the camera exposure.",
            self.acq_rate_entry: "Current acquisition rate (Hz).",
            self.rec_interval_entry: "Set the number of seconds between saved images.",
            self.default_settings: "Enable/disable default settings.",
            self.faster_settings: "Enable/disable fast mode (experimental).",
        }

        for widget, text in tooltips.items():
            tooltip.register(widget, text)


    def set_lock_state_when_no_camera(self):
        if self.model_vars.toolbar.acq.camera.get() == ELLIPSIS:
            state_to_set = tk.DISABLED
        else:
            state_to_set = tk.NORMAL

        #self.res_entry.configure(state=state_to_set)
        #self.fov_entry.configure(state=state_to_set)

    def set_lock_state(self, state=tk.DISABLED):
        self.camera_entry.configure(state=state)
        #self.res_entry.configure(state=state)
        #self.fov_entry.configure(state=state)
        self.default_settings.configure(state=state)
        use_defaults = self.model_vars.toolbar.acq.default_settings.get()
        # NOTE(cmo): If defaults are set, then don't mess with the ability
        # to adjust these.
        if not use_defaults:
            self.faster_settings.configure(state=state)
            self.scale_entry.configure(state=state)
            self.exposure_entry.configure(state=state)
            #self.pix_clock_entry.configure(state=state)
            self.rec_interval_entry.configure(state=state)

    def setup_default_settings_lock(self):
        def callback(*args):
            default = self.model_vars.toolbar.acq.default_settings.get()
            state = tk.DISABLED if default else tk.NORMAL
            fg_color = entry_disabled_color if default else "white"

            # If using CustomTkinter (CTkEntry)
            self.scale_entry.configure(state=state, fg_color=fg_color)
            self.exposure_entry.configure(state=state, fg_color=fg_color)
            self.rec_interval_entry.configure(state=state, fg_color=fg_color)
            self.faster_settings.configure(state=state, fg_color=fg_color)


        self.model_vars.toolbar.acq.default_settings.trace_add("write", callback)

    def setup_faster_settings_warning(self):
        faster_settings = self.model_vars.toolbar.acq.fast_mode

        def callback(*args):
            if self.model_vars.camera and self.model_vars.camera.camera_name != "Image from file":
                if faster_settings.get():
                    tmb.showwarning(
                        title="Warning",
                        message="This might make things go faster, and it might make things crash. You were warned. SET EXPOSURE AS LOW AS POSSIBLE!",
                    )

        faster_settings.trace_add("write", callback)

class AnalysisSettingsPane(ToolbarPane):
    def __init__(self, parent, model_vars: VtState):
        super().__init__(parent)
        self.parent = parent
        self.model_vars = model_vars
        sv = model_vars.toolbar.analysis

                # Set a fixed size
        self.configure(width=250, height=150)  
        self.grid_propagate(False)  # Prevent shrinking/expanding if using grid
        self.pack_propagate(False)  # Prevent resizing if using pack

        # Apply grid column weight
        self.columnconfigure(0, weight=1)  # Make column 0 expand
        self.columnconfigure(1, weight=1)  # Make column 1 expand

        self.pack(side=tk.LEFT, anchor=tk.N, padx=5, pady=5, fill=tk.Y)
        
        make_entry = make_entry_factory(self)


        
        # === Sliders Frame ===
        self.sliders_frame = ctk.CTkFrame(self, border_width=0)
        self.sliders_frame.pack(fill="x")

        self.sliders_frame.columnconfigure(0, weight=1)
        self.sliders_frame.columnconfigure(1, weight=3)
        self.sliders_frame.columnconfigure(2, weight=1)

        # === Checkboxes Frame ===
        self.checkboxes_frame = ctk.CTkFrame(self, border_width=0)
        self.checkboxes_frame.pack(fill="x")


        padx=3
        pady=3
        slider_width = 120
        checkbox_height = 20
        checkbox_width = 20

        self.frame_label = ctk.CTkLabel(self.sliders_frame, text="Analysis settings", font=(default_font, 16, 'bold'), fg_color=frame_label_color, height=frame_label_height, text_color='white').grid(row=0, column=0, columnspan=3,padx=1,pady=1, sticky="nsew")


        ctk.CTkLabel(self.sliders_frame, text="# of lines:", font=(default_font, default_font_size)).grid(row=1, column=0, padx=padx, pady=pady, sticky=tk.E)
        self.num_lines_entry = ctk.CTkSlider(self.sliders_frame, from_=2, to=25, orientation=tk.HORIZONTAL, variable=sv.num_lines, width=slider_width)
        self.num_lines_entry.grid(row=1, column=1, padx=padx, pady=pady)  # Span two columns

        self.num_lines_value_label = ctk.CTkLabel(self.sliders_frame, textvariable=sv.num_lines, font=(default_font, default_font_size))
        self.num_lines_value_label.grid(row=1, column=2, padx=padx, pady=pady, sticky=tk.W)  # Adjusted to be in the next column after the span

        ctk.CTkLabel(self.sliders_frame, text="Smooth:", font=(default_font, default_font_size)).grid(row=2, column=0, padx=padx, pady=pady, sticky=tk.E)
        self.smooth_scale = ctk.CTkSlider(self.sliders_frame, from_=1, to=101, orientation=tk.HORIZONTAL, variable=sv.smooth_factor, width=slider_width)
        self.smooth_scale.grid(row=2, column=1, padx=padx, pady=pady)  # Span two columns

        self.smooth_value_label = ctk.CTkLabel(self.sliders_frame, textvariable=sv.smooth_factor, font=(default_font, default_font_size))
        self.smooth_value_label.grid(row=2, column=2, padx=padx, pady=pady, sticky=tk.W)  # Adjusted accordingly

        ctk.CTkLabel(self.sliders_frame, text="Integration:", font=(default_font, default_font_size)).grid(row=3, column=0, padx=padx, pady=pady, sticky=tk.E)
        self.integration_scale = ctk.CTkSlider(self.sliders_frame, from_=2, to=100, orientation=tk.HORIZONTAL, variable=sv.integration_factor, width=slider_width)
        self.integration_scale.grid(row=3, column=1, padx=padx, pady=pady)

        self.integration_value_label = ctk.CTkLabel(self.sliders_frame, textvariable=sv.integration_factor, font=(default_font, default_font_size))
        self.integration_value_label.grid(row=3, column=2, padx=padx, pady=pady, sticky=tk.W)

        formatted_thresh = tk.StringVar(value=f"{sv.thresh_factor.get():.1f}")
        def update_slider_value(value):
            rounded_value = round(float(value), 1)  # Proper rounding
            sv.thresh_factor.set(rounded_value)  # Update DoubleVar
            formatted_thresh.set(f"{rounded_value:.1f}")  # Store formatted display value

        ctk.CTkLabel(self.sliders_frame, text="Threshold:", font=(default_font, default_font_size)).grid(row=4, column=0, padx=padx, pady=pady,sticky=tk.E)
        self.thresh_scale = ctk.CTkSlider(self.sliders_frame, from_=0.5, to=9.5, number_of_steps=90, orientation=tk.HORIZONTAL, variable=sv.thresh_factor, width=slider_width, command=update_slider_value)
        self.thresh_scale.grid(row=4, column=1, padx=padx, pady=pady)

        self.thresh_value_label = ctk.CTkLabel(self.sliders_frame, textvariable=formatted_thresh, font=(default_font, default_font_size))
        self.thresh_value_label.grid(row=4, column=2, padx=padx, pady=pady, sticky=tk.W)

        # CheckBoxes
        self.checkboxes_frame.columnconfigure(0, weight=1)
        self.checkboxes_frame.columnconfigure(1, weight=1)
        self.checkboxes_frame.columnconfigure(2, weight=1)


        self.filter_entry = ctk.CTkCheckBox(self.checkboxes_frame, text="Filter", font=(default_font, default_font_size), variable=sv.filter, checkbox_height=checkbox_height, checkbox_width=checkbox_width)
        self.filter_entry.grid(row=0, column=0, padx=padx, pady=pady, sticky=tk.NS)
        '''
        self.roi_entry = ctk.CTkCheckBox(self.checkboxes_frame, text="ROI", variable=sv.roi, checkbox_height=checkbox_height, checkbox_width=checkbox_width)
        self.roi_entry.grid(row=0, column=1, padx=padx, pady=pady, sticky=tk.E)  # Moved to the third column to align with the layout
        self.roi_entry.configure(state=tk.DISABLED)
        '''
        self.ID_entry = ctk.CTkCheckBox(self.checkboxes_frame, text="ID", font=(default_font, default_font_size), variable=sv.ID, checkbox_height=checkbox_height, checkbox_width=checkbox_width)
        self.ID_entry.grid(row=0, column=1, sticky=tk.NS)

        self.org_entry = ctk.CTkCheckBox(self.checkboxes_frame, text="Fluor", font=(default_font, default_font_size), variable=sv.org, checkbox_height=checkbox_height, checkbox_width=checkbox_width)
        self.org_entry.grid(row=1, column=0, padx=padx, pady=pady, sticky=tk.NS)  # Moved to the third column for consistency

        self.rotate_entry = ctk.CTkCheckBox(self.checkboxes_frame, text="90\u00B0", font=(default_font, default_font_size), variable=sv.rotate_tracking, checkbox_height=checkbox_height, checkbox_width=checkbox_width)
        self.rotate_entry.grid(row=1, column=1, padx=padx, pady=pady, sticky=tk.NS)  # Moved to the third column for consistency

        self.org_entry = ctk.CTkCheckBox(self.checkboxes_frame, text="US", font=(default_font, default_font_size), variable=sv.ultrasound_tracking, checkbox_height=checkbox_height, checkbox_width=checkbox_width)
        self.org_entry.grid(row=1, column=2, padx=padx, pady=pady, sticky=tk.NS)  # Moved to the third column for consistency

        # Create a single tooltip instance for the container
        tooltip = ToolTip(self)

        # Bind tooltips to widgets
        tooltips = {
            self.num_lines_entry: "Select the number of lines to track.",
            self.smooth_scale: "Set the width (in pixels) of the smoothing window.",
            self.integration_scale: "Set the number of pixel rows used for each line profile.",
            self.thresh_scale: "Set the threshold for identifying outliers.",
            self.filter_entry: "Enable or disable outlier detection.",
            #self.roi_entry: "Enable or disable region of interest (ROI) tracking.",
            self.ID_entry: "Enable or disable inner diameter tracking.",
            self.org_entry: "Enable or disable fluorescence tracking mode.",
            self.rotate_entry: "Switch between horizontal and vertical tracking.",
        }

        for widget, text in tooltips.items():
            tooltip.register(widget, text)



class GraphSettingsPane(ToolbarPane):
    def __init__(self, parent, model_vars: VtState):
        super().__init__(parent, height=175, width=150)
        self.parent = parent
        self.model_vars = model_vars
        sv = model_vars.toolbar.graph
        padx = 5
        pady = 5

        #self.pack(side=tk.LEFT, anchor=tk.N, padx=5, pady=5, fill=tk.Y)

        make_entry = make_entry_factory(self)
        self.frame_label = ctk.CTkLabel(self, text="Graph settings", font=(default_font, 16, 'bold'), fg_color=frame_label_color, height=frame_label_height, text_color='white').grid(row=0, column=0, columnspan=3,padx=1,pady=1, sticky="nsew")

        ctk.CTkLabel(self, text="Min:", font=(default_font, default_font_size)).grid(row=1, column=1, sticky=tk.NS, padx=padx, pady=pady)
        ctk.CTkLabel(self, text="Max:", font=(default_font, default_font_size)).grid(row=1, column=2, sticky=tk.NS, padx=padx, pady=pady)
        ctk.CTkLabel(self, text="Time:", font=(default_font, default_font_size)).grid(row=2, column=0, sticky=tk.E, padx=padx, pady=pady)
        ctk.CTkLabel(self, text="OD:", font=(default_font, default_font_size)).grid(row=3, column=0, sticky=tk.E, padx=padx, pady=pady)
        ctk.CTkLabel(self, text="ID:", font=(default_font, default_font_size)).grid(row=4, column=0, sticky=tk.E, padx=padx, pady=pady)

        graphaxes_entry_width = 75

        self.x_min_entry = make_entry(
            ctk.CTkEntry,
            textvariable=sv.x_min,
            font=(default_font, default_font_size),
            fg_color = "white",
            width=graphaxes_entry_width,
            row=2,
            column=1,
            padx=padx,
            pady=pady
        )

        self.x_max_entry = make_entry(
            ctk.CTkEntry,
            textvariable=sv.x_max,
            font=(default_font, default_font_size),
            fg_color = "white",
            width=graphaxes_entry_width,
            row=2,
            column=2,
            padx=padx,
            pady=pady
        )
        self.y_min_od_entry = make_entry(
            ctk.CTkEntry,
            textvariable=sv.y_min_od,
            font=(default_font, default_font_size),
            fg_color = "white",
            width=graphaxes_entry_width,
            row=3,
            column=1,
            padx=padx,
            pady=pady
        )
        self.y_max_od_entry = make_entry(
            ctk.CTkEntry,
            textvariable=sv.y_max_od,
            font=(default_font, default_font_size),
            fg_color = "white",
            width=graphaxes_entry_width,
            row=3,
            column=2,
            padx=padx,
            pady=pady
        )
        self.y_min_id_entry = make_entry(
            ctk.CTkEntry,
            textvariable=sv.y_min_id,
            font=(default_font, default_font_size),
            fg_color = "white",
            width=graphaxes_entry_width,
            row=4,
            column=1,
            padx=padx,
            pady=pady
        )
        self.y_max_id_entry = make_entry(
            ctk.CTkEntry,
            textvariable=sv.y_max_id,
            font=(default_font, default_font_size),
            fg_color = "white",
            width=graphaxes_entry_width,
            row=4,
            column=2,
            padx=padx,
            pady=pady
        )
        self.set_button = ctk.CTkButton(self, width=70, text="Set", font=(default_font, default_font_size),text_color="black")
        self.set_button.grid(row=6, column=1, padx=padx, pady=pady)
        self.default_button = ctk.CTkButton(self, width=70, text="Default", font=(default_font, default_font_size), text_color="black")
        self.default_button.grid(row=6, column=2, padx=padx, pady=pady)


class CaliperROIPane(ToolbarPane):
    def __init__(self, parent, model_vars: VtState):
        super().__init__(parent)
        self.parent = parent
        self.model_vars = model_vars
        sv = model_vars.toolbar.caliper_roi

        self.pack(side=tk.LEFT, anchor=tk.N, padx=5, pady=5, fill=tk.Y)

        BUTTON_WIDTH = 30
        BUTTON_HEIGHT = 30
        RADIO_DIAM = 20
        padx = (8,1)
        pady = 3

        # Adjusted resize_img method calls
        self.roi_img = self.resize_img(os.path.join(images_folder, 'ROI Button.png'), BUTTON_WIDTH, BUTTON_HEIGHT)
        self.caliper_img = self.resize_img(os.path.join(images_folder, 'Caliper Button.png'), BUTTON_WIDTH, BUTTON_HEIGHT)
        self.add_img = self.resize_img(os.path.join(images_folder, 'Add Button.png'), BUTTON_WIDTH, BUTTON_HEIGHT)
        self.remove_img = self.resize_img(os.path.join(images_folder, 'Remove Button.png'), BUTTON_WIDTH, BUTTON_HEIGHT)
        self.bin_img = self.resize_img(os.path.join(images_folder, 'Delete Button.png'), BUTTON_WIDTH, BUTTON_HEIGHT)

        self.frame_label = ctk.CTkLabel(self, text="Regions of interest", font=(default_font, 16, 'bold'), fg_color=frame_label_color, height=frame_label_height, text_color='white').grid(row=0, column=0, columnspan=3,padx=1,pady=1, sticky="nsew")

        self.single_label = ctk.CTkLabel(self, text="Single ROIs:", font=(default_font, default_font_size)).grid(row=1, column=0, padx=padx, pady=0, columnspan=3, sticky="w")

        self.draw_roi_button = ctk.CTkButton(self, image=self.roi_img, text="", height=BUTTON_HEIGHT, width=BUTTON_WIDTH)
        self.draw_roi_button.grid(row=2, column=0, padx=padx, pady=pady, sticky="ns")
        self.draw_roi_button.image = self.roi_img  # Keep a reference

        self.draw_caliper_button = ctk.CTkButton(self, image=self.caliper_img, text="", height=BUTTON_HEIGHT, width=BUTTON_WIDTH)
        self.draw_caliper_button.grid(row=2, column=1, padx=padx, pady=pady, sticky="ns")
        self.draw_caliper_button.image = self.caliper_img  # Keep a reference

        self.delete_roi_caliper_button = ctk.CTkButton(self, image=self.bin_img, text="", height=BUTTON_HEIGHT, width=BUTTON_WIDTH)
        self.delete_roi_caliper_button.grid(row=2, column=2, padx=(8,8), pady=pady, sticky="ns")
        self.delete_roi_caliper_button.image = self.bin_img  # Keep a reference

        self.multi_label = ctk.CTkLabel(self, text="Multiple ROIs:", font=(default_font, default_font_size)).grid(row=3, column=0, padx=padx, pady=0, columnspan=3, sticky="w")

        self.roi_button = ctk.CTkRadioButton(self, variable=sv.roi_flag, text="Box", value='ROI', font=(default_font, default_font_size), radiobutton_height=RADIO_DIAM, radiobutton_width=RADIO_DIAM)
        self.roi_button.grid(row=4, column=0, padx=padx, pady=pady, columnspan=2, sticky="w")

        self.caliper_button = ctk.CTkRadioButton(self, variable=sv.roi_flag, text="Line", value='Caliper', font=(default_font, default_font_size), radiobutton_height=RADIO_DIAM, radiobutton_width=RADIO_DIAM)
        self.caliper_button.grid(row=4, column=1, padx=padx, pady=pady, columnspan=2, sticky="w")

        self.auto_add_button = ctk.CTkButton(self, image=self.add_img, text="", height=BUTTON_HEIGHT, width=BUTTON_WIDTH)
        self.auto_add_button.grid(row=5, column=0, padx=padx, pady=pady, sticky="ns")
        self.auto_add_button.image = self.add_img  # Keep a reference

        self.auto_delete_button = ctk.CTkButton(self, image=self.remove_img, text="", height=BUTTON_HEIGHT, width=BUTTON_WIDTH)
        self.auto_delete_button.grid(row=5, column=1, padx=padx, pady=pady, sticky="ns")
        self.auto_delete_button.image = self.remove_img  # Keep a reference

        self.auto_delete_all_button = ctk.CTkButton(self, image=self.bin_img, text="", height=BUTTON_HEIGHT, width=BUTTON_WIDTH)
        self.auto_delete_all_button.grid(row=5, column=2, padx=(8,8), pady=pady, sticky="ns")
        self.auto_delete_all_button.image = self.bin_img  # Keep a reference

        self.showtraces_button = ctk.CTkButton(self, text="Show/ Hide Traces", font=(default_font, 16), text_color=VasoTracker_Blue, height=30, width=150)
        self.showtraces_button.grid(row=6, column=0, columnspan=3, padx=(8,8), pady=pady, sticky="ns")

        '''
        # Ensure equal column width distribution
        for i in range(3):  # 3 columns
            self.grid_columnconfigure(i, weight=1, uniform="cols")

        # Ensure equal row spacing
        for i in range(5):  # Adjust based on your row count
            self.grid_rowconfigure(i, weight=1)

        '''
        # Create a single tooltip instance for the container
        tooltip = ToolTip(self)

        # Bind tooltips to the buttons
        tooltips = {
            self.draw_roi_button: "Draw a single rectangular ROI.",
            self.draw_caliper_button: "Draw a single caliper line for manual measurement.",
            self.delete_roi_caliper_button: "Delete the ROI/caliper line.",
            self.auto_add_button: "Add an ROI/line.",
            self.auto_delete_button: "Delete last drawn ROI/line.",
            self.auto_delete_all_button: "Delete all ROIs/lines.",
        }

        for widget, text in tooltips.items():
            tooltip.register(widget, text)

    def resize_img(self, img_path, width=50, height=50):  # Match BUTTON_WIDTH and BUTTON_HEIGHT
        img = Image.open(img_path)
        resized_image = img.resize((width, height), Image.LANCZOS)
        tk_image = ctk.CTkImage(resized_image, size=(width, height))  # Ensure proper scaling
        return tk_image

class PlottingPane(ToolbarPane):
    def __init__(self, parent, model_vars: VtState):
        super().__init__(parent, height=175, width=150)
        self.parent = parent
        self.model_vars = model_vars
        sv = model_vars.toolbar.plotting

        self.frame_label = ctk.CTkLabel(self, text="Region of Interest information", font=(default_font, 16, 'bold'), fg_color=frame_label_color, height=frame_label_height, text_color='white').grid(row=0, column=0, columnspan=4,padx=1,pady=1, sticky="nsew")
       
        #self.pack(side=tk.LEFT, anchor=tk.N, padx=3, fill=tk.Y)
        pady=3
        padx=3
        def rgb_to_hex(r, g, b):
            return f"#{r:02x}{g:02x}{b:02x}"

        cmap = CMAP.colors
        colours = [rgb_to_hex(*[int(c * 255) for c in colour]) for colour in cmap]

        od_label = ctk.CTkLabel(self, text="OD", text_color="black", font=(default_font, default_font_size, "bold"))
        od_label.grid(row=1, column=1, pady=0, padx=padx)

        id_label = ctk.CTkLabel(self, text="ID", text_color="black", font=(default_font, default_font_size, "bold"))
        id_label.grid(row=1, column=2, pady=0, padx=padx)

        def add_Label(text, colour, row, col=0):
            label = ctk.CTkLabel(self, text=text, text_color=colour, font=(default_font, default_font_size, "bold"))
            label.grid(row=row+2, column=col, pady=pady, padx=padx)
            return label
        
        def add_entry(i, color, row, col=1, text_variable=None):
            entry = ctk.CTkEntry(self, text_color=color,textvariable=text_variable[i],  font=(default_font, default_font_size, "bold"),fg_color="white", justify='center', width=60, height=20)
            entry.configure()
            entry.grid(row=row+2, column=col, pady=pady, padx=padx)
            return entry
        
        def add_button(text, colour, row, col=3):
            button = ctk.CTkButton(self, text=text, font=(default_font, default_font_size), text_color=colour, width=70, height=25)
            #button.configure(fg=colour)
            button.grid(row=row+2, column=col, pady=pady, padx=padx)
            return button
        

        self.show_buttons = [
            add_button(f"Show", colours[i], row=i)
            for i in range(NUM_LINES)
        ]


        self.line_buttons = [
            add_Label(f"ROI {i+1}:", colours[i], row=i)
            for i in range(NUM_LINES)
        ]

        self.roi_od_plot_entry = [
            add_entry(i, colours[i], row=i, text_variable=sv.outer_diam_values)
            for i in range(NUM_LINES)
        ]

        self.roi_id_plot_entry = [
            add_entry(i, colours[i], row=i, col=2, text_variable=sv.inner_diam_values)
            for i in range(NUM_LINES)
        ]


    def update_button_states(self):
        for i, button in enumerate(self.show_buttons):
            state = self.model_vars.toolbar.plotting.line_show[i].get()
            try:
                if state:
                    button.configure(text="Hide")
                else:
                    button.configure(text='Show')
            except:
                pass

    

    




class DataAcquisitionPane(ToolbarPane):
    def __init__(self, parent, model_vars: VtState):
        super().__init__(parent, height=400, width=400)
        self.model_vars = model_vars
        sv = model_vars.toolbar.data_acq

        self.pack(side=tk.LEFT, anchor=tk.N, padx=5, pady=5, fill=tk.Y)
        self.frame_label = ctk.CTkLabel(self, text="Data Acquisition", font=(default_font, 16, 'bold'), fg_color=frame_label_color, height=frame_label_height, text_color='white').grid(row=0, column=0, columnspan=4,padx=1,pady=1, sticky="nsew")

        color_inner = '#{:02x}{:02x}{:02x}'.format(*C2)
        color_outer = '#{:02x}{:02x}{:02x}'.format(*C1)
        color_gray = '#{:02x}{:02x}{:02x}'.format(*C3)
        color_vt = '#{:02x}{:02x}{:02x}'.format(*C4)


        entry_fg_color = "white"
        entry_font_size = 18
        entry_width = 100
        justify = 'center'
        padx = (0,30)

        # Configuring the grid
        for col in range(3):
            self.grid_columnconfigure(col, weight=1)

        # Labels for OD, ID, and Pressure
        ctk.CTkLabel(self, text="OD (\u03bcm):", anchor="center", font=(default_font, default_font_size)).grid(row=1, column=0, padx=(20, 30), pady=0, sticky=tk.EW)
        ctk.CTkLabel(self, text="ID (\u03bcm):", anchor="center", font=(default_font, default_font_size)).grid(row=1, column=1, padx=padx, pady=0, sticky=tk.EW)
        ctk.CTkLabel(self, text="Pressure (mmHg):", anchor="center", font=(default_font, default_font_size)).grid(row=1, column=2, padx=padx, pady=0, sticky=tk.EW)

        # Recessed Entry for OD, ID, and Pressure
        self.outer_diam_entry = ctk.CTkEntry(self, textvariable=sv.outer_diam, font=(default_font, entry_font_size, "bold"), justify=justify, width=entry_width, fg_color=entry_fg_color, text_color=color_outer, state=tk.DISABLED)
        self.outer_diam_entry.grid(row=2, column=0, padx=(20, 30), pady=5)
        self.inner_diam_entry = ctk.CTkEntry(self, textvariable=sv.inner_diam, font=(default_font, entry_font_size, "bold"), justify=justify, width=entry_width, fg_color=entry_fg_color, text_color=color_inner, state=tk.DISABLED)
        self.inner_diam_entry.grid(row=2, column=1, padx=padx, pady=5)
        self.pressure_entry = ctk.CTkEntry(self, textvariable=sv.pressure, font=(default_font, entry_font_size, "bold"), justify=justify, width=entry_width, fg_color=entry_fg_color, text_color=color_vt, state=tk.DISABLED)
        self.pressure_entry.grid(row=2, column=2, padx=padx, pady=5)

        # Labels for OD %, Caliper μm, and Temp °C
        ctk.CTkLabel(self, text="OD (%)", anchor="center", font=(default_font, default_font_size)).grid(row=3, column=0, padx=(20, 30), pady=0, sticky=tk.EW)
        ctk.CTkLabel(self, text="Caliper (μm)", anchor="center", font=(default_font, default_font_size)).grid(row=3, column=1, padx=padx, pady=0, sticky=tk.EW)
        ctk.CTkLabel(self, text="Temp (°C)", anchor="center", font=(default_font, default_font_size)).grid(row=3, column=2, padx=padx, pady=0, sticky=tk.EW)

        # Recessed Entry for Diameter (%), Caliper, and Temp
        self.diam_percent_entry = ctk.CTkEntry(self, textvariable=sv.diam_percent, font=(default_font, entry_font_size, "bold"), justify=justify, fg_color=entry_fg_color, text_color=color_outer , width=entry_width, state=tk.DISABLED)
        self.diam_percent_entry.grid(row=4, column=0, padx=(20,30), pady=5)
        self.caliper_length_entry = ctk.CTkEntry(self, textvariable=sv.caliper_length, font=(default_font, entry_font_size, "bold"), justify=justify, fg_color=entry_fg_color, text_color=color_gray, width=entry_width, state=tk.DISABLED)
        self.caliper_length_entry.grid(row=4, column=1, padx=padx, pady=5)
        self.temperature_entry = ctk.CTkEntry(self, textvariable=sv.temperature, font=(default_font, entry_font_size, "bold"), justify=justify, fg_color=entry_fg_color, text_color=color_gray, width=entry_width, state=tk.DISABLED)
        self.temperature_entry.grid(row=4, column=2, padx=padx, pady=5)

        # Label and Recessed Entry for Pressure countdown (s)
        ctk.CTkLabel(self, text="Pressure countdown (s):", anchor="center", font=(default_font, default_font_size)).grid(row=1, column=3, padx=padx, pady=0, sticky=tk.EW)
        self.countdown_entry = ctk.CTkEntry(self, textvariable=sv.countdown, font=(default_font, entry_font_size, "bold"), justify=justify, width=entry_width, fg_color=entry_fg_color, text_color=color_gray, state=tk.DISABLED)
        self.countdown_entry.grid(row=2, column=3, padx=padx, pady=5)

        # Label and Recessed Entry for Time (s)
        ctk.CTkLabel(self, text="Time (hh:mm:ss):", anchor="center", font=(default_font, default_font_size)).grid(row=3, column=3, padx=padx, pady=0, sticky=tk.EW)
        self.time_entry = ctk.CTkEntry(self, textvariable=sv.time_string, font=(default_font, entry_font_size, "bold"), justify=justify, width=entry_width, fg_color=entry_fg_color, text_color=color_gray,  state=tk.DISABLED)
        self.time_entry.grid(row=4, column=3,padx=padx, pady=5)



class ImageDimensionsPane(ToolbarPane):
    def __init__(self, parent, model_vars: VtState):
        super().__init__(parent, height=175, width=150)
        self.parent = parent
        self.model_vars = model_vars
        sv = model_vars.toolbar.image_dim
        imagedimensions_entry_width = 100

        #self.pack(side=tk.LEFT, anchor=tk.N, padx=3, fill=tk.Y)

        self.frame_label = ctk.CTkLabel(self, text="Image dimensions", font=(default_font, 16, 'bold'), fg_color=frame_label_color, height=frame_label_height, text_color='white').grid(row=0, column=0, columnspan=2,padx=1,pady=1, sticky="nsew")


        make_entry = make_entry_factory(self)
        ctk.CTkLabel(self, text="Camera width:",  font=(default_font, default_font_size)).grid(row=1, column=0, sticky=tk.E)
        self.cam_width_entry = make_entry(
            ctk.CTkEntry,
            textvariable=sv.cam_width,
            font=(default_font, default_font_size),
            fg_color = entry_disabled_color,
            width=imagedimensions_entry_width,
            row=1,
            column=1,
            disabled=True,
        )
        ctk.CTkLabel(self, text="Camera height:", font=(default_font, default_font_size)).grid(row=2, column=0, sticky=tk.E)
        self.cam_height_entry = make_entry(
            ctk.CTkEntry,
            textvariable=sv.cam_height,
            font=(default_font, default_font_size),
            fg_color = entry_disabled_color,
            width=imagedimensions_entry_width,
            row=2,
            column=1,
            disabled=True,
        )
        ctk.CTkLabel(self, text="FOV width:",  font=(default_font, default_font_size)).grid(row=3, column=0, sticky=tk.E)
        self.fov_width_entry = make_entry(
            ctk.CTkEntry,
            textvariable=sv.fov_width,
            font=(default_font, default_font_size),
            fg_color = entry_disabled_color,
            width=imagedimensions_entry_width,
            row=3,
            column=1,
            disabled=True,
        )
        ctk.CTkLabel(self, text="FOV height:",  font=(default_font, default_font_size)).grid(row=4, column=0, sticky=tk.E)
        self.fov_height_entry = make_entry(
            ctk.CTkEntry,
            textvariable=sv.fov_height,
            font=(default_font, default_font_size),
            fg_color = entry_disabled_color,
            width=imagedimensions_entry_width,
            row=4,
            column=1,
            disabled=True,
        )


class ServoSettingsPane(ToolbarPane):
    def __init__(self, parent, model_vars: VtState):
        super().__init__(parent, height=175, width=150)
        self.parent = parent
        self.model_vars = model_vars
        sv = model_vars.toolbar.servo

        #self.pack(side=tk.LEFT, anchor=tk.N, padx=3, fill=tk.Y)

        make_entry = make_entry_factory(self)
        self.dev_options = ["", "Dev0", "Dev1", "Dev2"]
        self.ao_options = ["", "ao0", "ao1", "ao2"]

        # Add a label to display PyDAQmx availability


        self.pydaqmx_status_label = ctk.CTkLabel(self, text=f"PyDAQmx Available: {is_pydaqmx_available}", font=(default_font, default_font_size))
        self.pydaqmx_status_label.grid(row=0, column=0, columnspan=2)

        # Device option menu
        ctk.CTkLabel(self, text="Device", font=(default_font, default_font_size)).grid(row=1, column=0, sticky=tk.E, )
        self.dev_entry = make_entry(
            ttk.OptionMenu,
            args=(
                sv.device,
                sv.device.get(), #self.dev_options[0],
                *self.dev_options,
            ),
            row=1,
            column=1,
        )

        # AO channel option menu
        ctk.CTkLabel(self, text="ao channel:", font=(default_font, default_font_size)).grid(row=2, column=0, sticky=tk.E)
        self.ao_entry = make_entry(
            ttk.OptionMenu,
            args=(
                sv.ao_channel,
                sv.ao_channel.get(),
                *self.ao_options,
            ),
            row=2,
            column=1,
        )

        # Add traces to the StringVar instances
        #sv.device.trace_add("write", lambda *args: self.model_vars.pressure_controller.on_option_changed())
        try:
            sv.ao_channel.trace_add("write", lambda *args: self.model_vars.pressure_controller.on_option_changed())
        except:
            pass

        # Create a single tooltip instance for the container
        tooltip = ToolTip(self)

        # Bind tooltips to the buttons
        tooltips = {
            self.dev_entry: "Select your NI device.",
            self.ao_entry: "Set the analogue output channel.",
        }

        for widget, text in tooltips.items():
            tooltip.register(widget, text)


class PressureControlPane(ToolbarPane):
    def __init__(self, parent, model_vars: VtState):
        super().__init__(parent, height=400, width=400)
        self.parent = parent
        self.model_vars = model_vars
        sv = model_vars.toolbar.pressure_protocol

        self.pack(side=tk.LEFT, anchor=tk.N, padx=5, pady=5, fill=tk.Y)
        self.frame_label = ctk.CTkLabel(self, text="Pressure control (mmHg)", font=(default_font, 16, 'bold'), fg_color=frame_label_color, height=frame_label_height, text_color='white').grid(row=0, column=0, columnspan=4,padx=1,pady=1, sticky="nsew")

        justify = 'center'
        BUTTON_HEIGHT = 30
        BUTTON_WIDTH = 30

        padx=(5,1)
        pady=0

        # Scale for pressure increment

        self.pressure_connect_img = self.resize_img(os.path.join(images_folder, 'Connect Button Black.png'),  BUTTON_WIDTH, BUTTON_HEIGHT)
        self.pressure_connect_button = ctk.CTkButton(self, image=self.pressure_connect_img, text="", height=BUTTON_HEIGHT, width=BUTTON_WIDTH)
        self.pressure_connect_button.grid(row=1, column=0, padx=padx, pady=(8,0))
        self.pressure_connect_button.image = self.pressure_connect_img  # Keep a reference

        self.pressure_settings_img = self.resize_img(os.path.join(images_folder, 'Settings Button Black.png'),  BUTTON_WIDTH, BUTTON_HEIGHT)
        self.pressure_settings_button = ctk.CTkButton(self, image=self.pressure_settings_img, text="", height=BUTTON_HEIGHT, width=BUTTON_HEIGHT)
        self.pressure_settings_button.grid(row=1, column=1, padx=(5,5), pady=(8,0))
        self.pressure_settings_button.image = self.pressure_settings_img  # Keep a reference

        self.pressure_start_img = self.resize_img(os.path.join(images_folder, 'Pressure Step Button-01.png'),  BUTTON_WIDTH, BUTTON_HEIGHT)
        self.start_protocol_button = ctk.CTkButton(self, image=self.pressure_start_img, text="", height=BUTTON_HEIGHT, width=BUTTON_WIDTH, fg_color="#BDC3C7", state=tk.DISABLED)
        self.start_protocol_button.grid(row=1, column=2, padx=padx, pady=(8,0))
        self.start_protocol_button.image = self.pressure_start_img  # Keep a reference

        self.pressure_stop_img = self.resize_img(os.path.join(images_folder, 'Pressure Step Button.png'),  BUTTON_WIDTH, BUTTON_HEIGHT)
        self.pressure_stop_img.image = self.pressure_stop_img  # Keep a reference

        self.set_pressure_img = self.resize_img(os.path.join(images_folder, 'Pressure Start Button-01.png'),  BUTTON_WIDTH, BUTTON_HEIGHT)
        self.set_pressure_button = ctk.CTkButton(self, image=self.set_pressure_img, text="", height=BUTTON_HEIGHT, width=BUTTON_WIDTH, fg_color="#BDC3C7", state=tk.DISABLED)
        self.set_pressure_button.grid(row=1, column=3, padx=padx, pady=(8,0))
        self.set_pressure_button.image = self.set_pressure_img  # Keep a reference



        ctk.CTkLabel(self, text="Manual control:", font=(default_font, default_font_size)).grid(row=2, column=0, columnspan=4, sticky=tk.W)

        # Recessed Entry
        self.outer_diam_entry = ctk.CTkEntry(self, font=(default_font, 20), textvariable=sv.set_pressure, justify=justify, width=100, fg_color=entry_disabled_color, state=tk.DISABLED)
        self.outer_diam_entry.grid(row=3, column=1, columnspan=2)  # Span two columns

        self.minus_img = self.resize_img(os.path.join(images_folder, 'Subtract Button Black.png'), BUTTON_WIDTH, BUTTON_HEIGHT)
        self.minus_button = ctk.CTkButton(self, image=self.minus_img, text="", height=BUTTON_HEIGHT, width=BUTTON_WIDTH)
        self.minus_button.grid(row=3, column=0, padx=padx, pady=pady)
        self.minus_button.image = self.minus_img  # Keep a reference

        self.add_img = self.resize_img(os.path.join(images_folder, 'Add Button Black.png'), BUTTON_WIDTH, BUTTON_HEIGHT)
        self.add_button = ctk.CTkButton(self, image=self.add_img, text="", height=BUTTON_HEIGHT, width=BUTTON_WIDTH,)
        self.add_button.grid(row=3, column=3, padx=(5,5), pady=pady)
        self.add_button.image = self.add_img  # Keep a reference

        ctk.CTkLabel(self, text="Increment change:", font=(default_font, default_font_size)).grid(row=4, column=0, columnspan=4, sticky=tk.W)

        self.pressure_increment_entry = ctk.CTkSlider(self, from_=1, to=20, variable=sv.pressure_increment, width=120)
        self.pressure_increment_entry.grid(row=5, column=0, padx=padx, columnspan=2)  # Span two columns

        self.slider_value_entry = ctk.CTkEntry(self, textvariable=sv.pressure_increment, justify=justify, width=40,font=(default_font,20), fg_color=entry_disabled_color, state=tk.DISABLED)
        self.slider_value_entry.grid(row=5, column=2, padx=padx, columnspan=2, sticky="w")  # Span two columns

        self.model_vars.app.auto_pressure.trace_add(
            "write", lambda *args: self.start_protocol_button_state_callback()
        )


        # Button for setting pressure
        #self.set_pressure_button = ctk.CTkButton(self, text="Set Pressure")
        #self.set_pressure_button.grid(row=2, column=1, sticky=tk.W)

        # Buttons for starting and stopping the pressure protocol
        #self.start_protocol_button = ctk.CTkButton(self, text="Start Protocol")
        #self.start_protocol_button.grid(row=2, column=2, sticky=tk.E)
        
        # Create a single tooltip instance for the container
        tooltip = ToolTip(self)

        # Bind tooltips to the buttons
        tooltips = {
            self.pressure_connect_button: "Connect your NI board for pressure control.",
            self.start_protocol_button: "Start pressure ramp experiment.",
            self.set_pressure_button: "Set pressure to indicated value.",
            self.pressure_settings_button: "Open pressure protocol settings.",
            self.outer_diam_entry: "Click -/+ buttons to change desired pressure.",
            self.pressure_increment_entry: "Slide to increase pressure increment.",
        }

        for widget, text in tooltips.items():
            tooltip.register(widget, text)


    def start_protocol_button_state_callback(self):
        running = self.model_vars.app.auto_pressure.get()
        if running:
            self.start_protocol_button.configure(image=self.pressure_stop_img)
            self.set_pressure_button.configure(state=tk.DISABLED, fg_color="#BDC3C7")
        else:
            self.start_protocol_button.configure(image=self.pressure_start_img)
            self.set_pressure_button.configure(state=tk.NORMAL, fg_color="white")

    def resize_img(self, img_path, width=50, height=50):  # Match BUTTON_WIDTH and BUTTON_HEIGHT
        img = Image.open(img_path)
        resized_image = img.resize((width, height), Image.LANCZOS)
        tk_image = ctk.CTkImage(resized_image, size=(width, height))  # Ensure proper scaling
        return tk_image

    def set_lock_state(self, state=tk.DISABLED):
        pass
        #self.start_protocol_button.configure(state=state)
        #self.set_pressure_entry.configure(state=state)
        #self.set_pressure_button.configure(state=state)

    def set_unlock_state(self, state=tk.NORMAL):
        pass
        #self.start_protocol_button.configure(state=state)
        #self.set_pressure_entry.configure(state=state)
        #self.set_pressure_button.configure(state=state)

    def enable_buttons(self):
        self.start_protocol_button.configure(state=tk.NORMAL)
        self.set_pressure_button.configure(state=tk.NORMAL)


    def toggle_protocol_button(self):
        current_state = self.model_vars.app.auto_pressure.get()
        self.model_vars.app.auto_pressure.set(not current_state)
        running  = self.model_vars.app.auto_pressure.set(not current_state)
        if running:
            self.start_protocol_button.configure(image=self.pressure_stop_img)
            # Reset the variables here!!!
            try:
                self.model.pressure_controller.reset_protocol()
            except:
                pass

        else:
            self.start_protocol_button.configure(image=self.pressure_start_img)

# Specify a larger font
large_font = ('Helvetica', 14)

class PressureProtocolPane(ToolbarPane):
    def __init__(self, parent, model_vars: VtState):
        super().__init__(parent)
        self.parent = parent
        self.model_vars = model_vars
        sv = model_vars.toolbar.pressure_protocol
        pressure_entry_width = 50
        padx = 10

        self.frame_label = ctk.CTkLabel(self, text="Pressure protocol settings", font=(default_font, 16, 'bold'), fg_color=frame_label_color, height=frame_label_height, text_color='white').grid(row=0, column=0, columnspan=2,padx=1,pady=1, sticky="nsew")

        # Adjusted the layout to not specify height and width here
        

        make_entry = make_entry_factory(self)
        ctk.CTkLabel(self, text="Start (mmHg):", font=(default_font, default_font_size)).grid(row=1, column=0, sticky=tk.E, padx=10, pady=5)
        self.pressure_start_entry = make_entry(
            ctk.CTkEntry,
            textvariable=sv.pressure_start,
            font=(default_font, default_font_size),
            row=1,
            width=pressure_entry_width,
            fg_color = "white",
            padx=padx,
            disabled=False,
        )
        ctk.CTkLabel(self, text="Stop (mmHg):", font=(default_font, default_font_size)).grid(row=2, column=0, sticky=tk.E, padx=10, pady=5)
        self.pressure_stop_entry = make_entry(
            ctk.CTkEntry,
            textvariable=sv.pressure_stop,
            font=(default_font, default_font_size),
            row=2,
            width=pressure_entry_width,
            fg_color = "white",
            padx=padx,
            disabled=False,
        )
        ctk.CTkLabel(self, text="Intvl (mmHg):", font=(default_font, default_font_size)).grid(row=3, column=0, sticky=tk.E, padx=10, pady=5)
        self.pressure_intvl_entry = make_entry(
            ctk.CTkEntry,
            textvariable=sv.pressure_intvl,
            font=(default_font, default_font_size),
            row=3,
            width=pressure_entry_width,
            fg_color = "white",
            padx=padx,
            disabled=False,
        )
        ctk.CTkLabel(self, text="Intvl (s):",font=(default_font, default_font_size)).grid(row=4, column=0, sticky=tk.E, padx=10, pady=5)
        self.time_intvl_entry = make_entry(
            ctk.CTkEntry,
            textvariable=sv.time_intvl,
            font=(default_font, default_font_size),
            row=4,
            width=pressure_entry_width,
            fg_color = "white",
            padx=padx,
            disabled=False,
        )

        self.hold_pressure_entry = ctk.CTkCheckBox(self, text="Hold final pressure", font=(default_font, default_font_size), variable=sv.hold_pressure, checkbox_height=20, checkbox_width=20)
        self.hold_pressure_entry.grid(row=5, column=0, padx=10, pady=(0,10), columnspan=2, sticky=tk.NS)

        # Make the container expandable
        for i in range(4):  # Assuming 4 rows
            self.grid_rowconfigure(i, weight=1)
        self.grid_columnconfigure(0, weight=1)


        # Create a single tooltip instance for the container
        tooltip = ToolTip(self)

        # Bind tooltips to the buttons
        tooltips = {
            self.pressure_start_entry: "Set the initial pressure for your ramp experiment.",
            self.pressure_stop_entry: "Set the final pressure for your ramp experiment.",
            self.pressure_intvl_entry: "Set the increment for ramp experiment.",
            self.time_intvl_entry: "Set the interval between pressure steps (in seconds).",
            self.hold_pressure_entry: "Enable/disable to end on final pressure increment (disable to reset pressure at end of experiment).",
        }

        for widget, text in tooltips.items():
            tooltip.register(widget, text)



    def set_lock_state(self, state=tk.DISABLED):
        self.pressure_start_entry.configure(state=state)
        self.pressure_stop_entry.configure(state=state)
        self.pressure_intvl_entry.configure(state=state)
        self.time_intvl_entry.configure(state=state)
        #self.countdown_entry.configure(state=state)

    def set_lock_state_running(self, state=tk.DISABLED):
        self.pressure_start_entry.configure(state=state)
        self.pressure_stop_entry.configure(state=state)
        self.pressure_intvl_entry.configure(state=state)
        self.time_intvl_entry.configure(state=state)
        #self.countdown_entry.configure(state=state)

    def set_unlock_state(self, state=tk.NORMAL):
        self.pressure_start_entry.configure(state=state)
        self.pressure_stop_entry.configure(state=state)
        self.pressure_intvl_entry.configure(state=state)
        self.time_intvl_entry.configure(state=state)
        #self.countdown_entry.configure(state=state)



class StartStopPane(ToolbarPane):
    def __init__(self, parent, model_vars: VtState):
        super().__init__(parent)
        self.parent = parent
        self.model_vars = model_vars
        sv = model_vars.toolbar.start_stop
        BUTTON_WIDTH = 60
        BUTTON_HEIGHT = 60
        # Set explicit width and height
        #self.configure(width=800, height=175)
        #self.pack_propagate(False)  # Prevents shrinking to fit children
        self.frame_label = ctk.CTkLabel(self, text="Start / Stop", font=(default_font, 16, 'bold'), fg_color=frame_label_color, height=frame_label_height, text_color='white').grid(row=0, column=0, columnspan=4,padx=1,pady=1, sticky="nsew")

        # Pack this frame inside the parent
        self.pack(side=tk.LEFT, anchor=tk.CENTER, padx=5, pady=5, fill=tk.Y)

        self.camera_on_img = self.resize_img(os.path.join(images_folder, 'Camera_button_on.png'), BUTTON_WIDTH, BUTTON_HEIGHT)
        self.camera_off_img = self.resize_img(os.path.join(images_folder, 'Camera_button_off.png'), BUTTON_WIDTH, BUTTON_HEIGHT)

        self.start_button =ctk.CTkButton(self, image=self.camera_on_img, width=50, text="")#, compound='top')#text="Snapshot",
        self.start_button.grid(row=1, column=0, padx=5, pady=8)#, sticky="nsew")

        self.tracking_on_img = self.resize_img(os.path.join(images_folder, 'Tracking_button_on.png'), BUTTON_WIDTH, BUTTON_HEIGHT)
        self.tracking_off_img = self.resize_img(os.path.join(images_folder, 'Tracking_button_off.png'), BUTTON_WIDTH, BUTTON_HEIGHT)

        self.track_button = ctk.CTkButton(self, image=self.tracking_on_img, width=50, text="")#, compound='top')#text="Snapshot",
        self.track_button.grid(row=1, column=1, padx=5, pady=8)#, sticky="nsew")

        self.snapshot_image = self.resize_img(os.path.join(images_folder, 'Snapshot_Icon.png'), BUTTON_WIDTH, BUTTON_HEIGHT)

        self.snapshot_button = ctk.CTkButton(self, image=self.snapshot_image, width=50, text="")#, compound='top')#text="Snapshot",
        self.snapshot_button.grid(row=1, column=2, padx=5, pady=8)#, sticky="nsew")

        self.record_button = ctk.CTkSwitch(self, variable=sv.record, text="Record Camera", font=(default_font, default_font_size), switch_height=20, switch_width=40, border_width=2, border_color="#203C57")
        self.record_button.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky="NS")

        self.model_vars.app.acquiring.trace_add(
            "write", lambda *args: self.start_button_state_callback()
        )

        self.model_vars.app.tracking.trace_add(
            "write", lambda *args: self.track_button_state_callback()
        )

        # Create a single tooltip instance for the container
        tooltip = ToolTip(self)

        # Bind tooltips to the buttons
        tooltips = {
            self.start_button: "Start/stop the camera display.",
            self.track_button: "Start/stop diameter tracking.",
            self.snapshot_button: "Take a snapshot.",
            self.record_button: "Enable/disable video recording.",
        }

        for widget, text in tooltips.items():
            tooltip.register(widget, text)


    def resize_img(self, img_path, width=50, height=50):  # Match BUTTON_WIDTH and BUTTON_HEIGHT
        img = Image.open(img_path)
        resized_image = img.resize((width, height), Image.LANCZOS)
        tk_image = ctk.CTkImage(resized_image, size=(width, height))  # Ensure proper scaling
        return tk_image

    def record_button_state_callback(self):
        recording = self.model_vars.toolbar.start_stop.record.get()
        self.model_vars.toolbar.start_stop.record.set(True)


    def start_button_state_callback(self):
        running = self.model_vars.app.acquiring.get()
        if self.model_vars.camera.camera_name == "Image from file":
            self.start_button.configure(image=self.camera_on_img)
        elif running:
            self.start_button.configure(image=self.camera_off_img)
        else:
            self.start_button.configure(image=self.camera_on_img)

    def track_button_state_callback(self):
        tracking = self.model_vars.app.tracking.get()
        if tracking:
            self.track_button.configure(image=self.tracking_off_img)
        else:
            self.track_button.configure(image=self.tracking_on_img)



class ToolbarView(ctk.CTkFrame):
    def __init__(self, parent, state, set_camera_callback):
        super().__init__(parent)
        # NOTE(cmo): The underscore is more to avoid shadowing from a parent
        # class than privatising the variable

        super().__init__(parent)
        self._state = state
        self.panes: List[ToolbarPane] = []
        self.set_camera_callback = set_camera_callback

        print(f"AcquisitionSettingsPane received set_camera_callback: {self.set_camera_callback}")  # ✅ Debug print


        # Initialize and pack other panes with side='left' to align them to the left
        self.acq = AcquisitionSettingsPane(self, state, self.set_camera_callback)
        self.panes.append(self.acq)
        self.acq.pack(side='left', fill='y')

        self.analysis = AnalysisSettingsPane(self, state)
        self.panes.append(self.analysis)
        self.analysis.pack(side='left', fill='y')

        #self.graph = GraphSettingsPane(self, state)
        #self.panes.append(self.graph)
        #self.graph.pack(side='left', fill='y')

        self.caliper_roi = CaliperROIPane(self, state)
        self.panes.append(self.caliper_roi)
        self.caliper_roi.pack(side='left', fill='y')

        if is_pydaqmx_available:
            self.pressure_control_settings = PressureControlPane(self, state)
            self.panes.append(self.pressure_control_settings)
            self.pressure_control_settings.pack(side='left', fill='y')


        self.start_stop = StartStopPane(self, state)
        self.panes.append(self.start_stop)
        self.start_stop.pack(side='left', fill='y')

        # Pack the DataAcquisitionPane last with side='right' to anchor it to the right side of the container
        self.data_acq = DataAcquisitionPane(self, state)
        self.panes.append(self.data_acq)
        self.data_acq.pack(side='right', fill='y')

        # Initialise, but do not add to the toolbar
        # Must also comment out the pack command in the Class.
        self.source = SourcePane(self, state)
        self.plotting = PlottingPane(self, state)
        self.image_dim = ImageDimensionsPane(self, state)  # Initialise, but do not add to the toolbar
        self.servo_settings = ServoSettingsPane(self, state)
        self.pressure_protocol_settings = PressureProtocolPane(self, state)
        self.graph = GraphSettingsPane(self, state)

        self.setup_view_blockers()

    def get_image_dimensions_pane(self):
        return self.image_dim

    def setup_view_blockers(self):
        def callback(*args):
            if self._state.app.acquiring.get():
                self.set_acquire_state()
            elif not self._state.cam_show.canvas_draw_state.user_drawing.get():
                # NOTE(cmo): Don't unlock the toolbar for a temporary pause
                # whilst the user is drawing on the canvas
                self.set_edit_state()

        self._state.app.acquiring.trace_add("write", callback)

    def set_edit_state(self):
        for pane in self.panes:
            pane.set_edit_state()

    def set_acquire_state(self):
        for pane in self.panes:
            # Skip setting the state for PressureProtocolPane
            if not isinstance(pane, PressureProtocolPane):
                pane.set_acquire_state()


class Menus:
    def __init__(self, root):
        self.root = root
        self.menu_bar = tk.Menu(root)
        file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.file_menu = file_menu
        file_menu.add_command(
            label="New file...",
        )
        file_menu.add_separator()
        file_menu.add_command(
            label="Analyze file...",
        )
        file_menu.add_separator()
        file_menu.add_command(
            label="Load settings...",
        )
        file_menu.add_command(
            label="Save settings...",
        )
        file_menu.add_separator()
        file_menu.add_command(
            label="Exit",
        )

        settings_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.settings_menu = settings_menu

        self.settings_menu.add_command(label="File details")
        self.settings_menu.add_separator()
        self.settings_menu.add_command(label="Image Dimensions")
        self.settings_menu.add_separator()
        self.settings_menu.add_command(label="Graph Axes")
        self.settings_menu.add_command(label="Show/Hide Traces")

        if is_pydaqmx_available:
            self.settings_menu.add_separator()
            self.settings_menu.add_command(label="DAQ Setup")
            self.settings_menu.add_command(label="Configure Pressure Protocol")

        notepad_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.notepad_menu = notepad_menu
        notepad_menu.add_command(label="Open notepad")

        help_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.help_menu = help_menu
        help_menu.add_command(label="The VasoTracker Band", command=self.play_band)
        help_menu.add_command(label="Boogie woogie")
        help_menu.add_separator()
        help_menu.add_command(label="Register")
        help_menu.add_command(label="User Guide")
        help_menu.add_command(label="About")
        help_menu.add_command(label="Update")
        # Add Pacman Launch Option
        help_menu.add_separator()
        help_menu.add_command(label="Waka Waka", command=self.launch_pacman)
        # Add Pacman Launch Option
        help_menu.add_separator()
        help_menu.add_command(label="Peow Peow", command=self.launch_invaders)

        self.menu_bar.add_cascade(label="File", menu=file_menu)
        self.menu_bar.add_cascade(label="Settings", menu=settings_menu)
        self.menu_bar.add_cascade(label="Notepad", menu=notepad_menu)
        self.menu_bar.add_cascade(label="Help", menu=help_menu)
        self.root.configure(menu=self.menu_bar)

    def launch_pacman(self):
        self.launch_game("pacman", "pacman.py")

    def launch_invaders(self):
        self.launch_game("space-invaders", "spaceinvaders.py")
    
    def launch_game(self, game_name, script_name):
        try:
            game_dir = get_resource_path(os.path.join(game_name))
            script_path = os.path.join(game_dir, script_name)

            print(f"Launching {game_name} from:")
            print(" - Script:", script_path)
            print(" - Dir:", game_dir)

            original_cwd = os.getcwd()  # Store current directory

            os.chdir(game_dir)  # Change working directory so relative asset paths work
            sys.path.insert(0, game_dir)  # Allow local imports like `import board`

            runpy.run_path(script_path, run_name="__main__")

        except Exception as e:
            print(f"Error launching {game_name}: {e}")

        finally:
            os.chdir(original_cwd)  # Always restore original working directory
    
    def play_band(self):
        """Play a random MP3 song from the music/ folder in a background thread."""
        def play_music_background():
            # Initialize pygame mixer (if not already initialized)
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            # Prepare list of MP3 files in the music directory
            music_dir = os.path.join(os.path.dirname(__file__), "music")
            songs = [f for f in os.listdir(music_dir) if f.lower().endswith(".mp3")]
            if not songs:
                print("No MP3 files found in the music/ directory.")
                return  # No music to play
            # Choose a random song and load it
            song_file = random.choice(songs)
            song_path = os.path.join(music_dir, song_file)
            pygame.mixer.music.load(song_path)
            pygame.mixer.music.play()
            # Keep thread alive until the music finishes playing
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)  # small delay to avoid busy-waiting
            # Optional: uninitialize mixer to release audio device
            pygame.mixer.quit()
        
        # Start the background thread for music playback (daemon=True so it exits with the app)
        threading.Thread(target=play_music_background, daemon=True).start()
'''
GraphState, MeasureStore, GraphPaneState
VtState: graph, measure
'''

import numpy as np  # Make sure you have numpy imported if you're using it

class GraphFrame(ttk.Frame):
    def __init__(self, parent, state: VtState):
        super().__init__(parent)

        self.parent = parent
        self.state_vars = state

        self.grid_propagate(False)

        # Get the default axis limits
        settings = state.toolbar.graph
        self.xlim = (settings.x_min.get(), settings.x_max.get())
        self.ylim_id = (settings.y_min_id.get(), settings.y_max_id.get())
        self.ylim_od = (settings.y_min_od.get(), settings.y_max_od.get())

        self.setup_widgets()
        self.update_lims()
        self.get_blit_area()

        state.graph.dirty.trace_add("write", lambda *args: self.draw())
        state.graph.clear.trace_add("write", lambda *args: self.clear_graph())
        settings.dirty.trace_add("write", lambda *args: self.update_lims_callback())

        # Dirty variable for updating the graph axis limits.
        settings.limits_dirty.trace_add("write", lambda *args: self.update_lims_fromfile_callback())

    def get_blit_area(self):
        self.ax1_bg = self.figure.canvas.copy_from_bbox(self.ax1.bbox)
        self.ax2_bg = self.figure.canvas.copy_from_bbox(self.ax2.bbox)

    def setup_widgets(self):
        axis_color = VasoTracker_Blue #"black"
        graph_color = "white"
        graph_text_color = VasoTracker_Blue #"black"
        # Create a figure with two subplots (ax1 and ax2) stacked vertically
        self.figure, (self.ax1, self.ax2) = plt.subplots(2, 1)
        self.figure.patch.set_facecolor("white")

        # Create separate axes for markers
        self.ax1_markers = self.ax1.twinx()
        self.ax2_markers = self.ax2.twinx()

        # Set subplot backgrounds
        self.ax1.set_facecolor(graph_color)
        self.ax2.set_facecolor(graph_color)
        self.ax1_markers.set_facecolor(axis_color)
        self.ax2_markers.set_facecolor(axis_color)

        # Make axis lines, ticks, and labels white
        for ax in [self.ax1, self.ax2, self.ax1_markers, self.ax2_markers]:
            ax.spines['bottom'].set_color(axis_color)
            ax.spines['top'].set_color(axis_color)
            ax.spines['left'].set_color(axis_color)
            ax.spines['right'].set_color(axis_color)
            ax.xaxis.label.set_color(graph_text_color)
            ax.yaxis.label.set_color(graph_text_color)
            ax.tick_params(colors=axis_color, which='both')

            # Set font size and style for axis labels
            ax.xaxis.label.set_fontsize(16)  # Change font size for x-axis label
            ax.yaxis.label.set_fontsize(16)  # Change font size for y-axis label
            ax.xaxis.label.set_fontname(default_font)  # Change font for x-axis label
            ax.yaxis.label.set_fontname(default_font)  # Change font for y-axis label

            # Set font size for ticks
            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                tick.set_fontsize(12)  # Set font size for tick labels
                tick.set_fontname(default_font)  # Set font for tick labels



        # Initialize empty plots for dynamic updating
        (self.od_avg,) = self.ax1.plot([], [], label='OD Avg')
        (self.id_avg,) = self.ax2.plot([], [], label='ID Avg')
        (self.markers,) = self.ax1_markers.plot([], [], label='Markers')
        (self.markers,) = self.ax2_markers.plot([], [])
        # Repeat for `self.ax2_markers` if necessary

        # Assuming NUM_LINES is defined and represents the number of dynamic lines
        self.od_lines = [self.ax1.plot([], [])[0] for _ in range(NUM_LINES)]
        self.id_lines = [self.ax2.plot([], [])[0] for _ in range(NUM_LINES)]

        self.ax1_vline = self.ax1.axvline(1, c='k')
        self.ax2_vline = self.ax2.axvline(1, c='k')

        self.ax1.set_ylabel("Outer Diameter (OD)")
        self.ax2.set_xlabel("Time (s or frames)")
        self.ax2.set_ylabel("Inner Diameter (ID)")

        # Create the canvas and pack it to fill available space
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Create and pack the Navigation Toolbar
        self.toolbar = CustomVTToolbar(self.canvas, self, self)  # 'self' is passed twice: once as parent, once as graph_frame reference
        self.toolbar.update()
        self.toolbar.pack(side=tk.TOP, fill=tk.X, expand=False)

    def draw(self):
        state = self.state_vars.graph
        if state.clear.get():
            self.clear_graph()

        Cblue = (0, 0, 200) #Blue outer
        Cgreen = (0,125, 0) #Dark green inner

        # Convert RGB to hexadecimal
        hex_color_Cblue = '#{:02x}{:02x}{:02x}'.format(C1[0], C1[1], C1[2])
        hex_color_Cgreen = '#{:02x}{:02x}{:02x}'.format(C2[0], C2[1], C2[2])

        if state.dirty.get():
            plot_mask = [b.get() for b in self.state_vars.toolbar.plotting.line_show]

            # Clear existing vertical lines and annotations
            self.clear_markers()

            # self.figure.canvas.restore_region(self.ax1_bg)
            # self.figure.canvas.restore_region(self.ax2_bg)
            self.od_avg.set_xdata(state.od_avg.x)
            self.od_avg.set_ydata(state.od_avg.y)
            self.od_avg.set_color(hex_color_Cblue)
            self.ax1.draw_artist(self.od_avg)

            self.id_avg.set_xdata(state.id_avg.x)
            self.id_avg.set_ydata(state.id_avg.y)
            self.id_avg.set_color(hex_color_Cgreen)
            self.ax2.draw_artist(self.id_avg)

            self.markers.set_xdata(state.markers.x)
            self.markers.set_ydata(state.markers.y)

            for i, plot in enumerate(plot_mask):
                if not plot:
                    self.od_lines[i].set_xdata([])
                    self.od_lines[i].set_ydata([])
                    self.id_lines[i].set_xdata([])
                    self.id_lines[i].set_ydata([])
                    continue

                self.od_lines[i].set_xdata(state.od_lines[i].x)
                self.od_lines[i].set_ydata(state.od_lines[i].y)
                self.od_lines[i].set_color(f"C{i}")
                self.ax1.draw_artist(self.od_lines[i])

                self.id_lines[i].set_xdata(state.id_lines[i].x)
                self.id_lines[i].set_ydata(state.id_lines[i].y)
                self.id_lines[i].set_color(f"C{i}")
                self.ax2.draw_artist(self.id_lines[i])

            
            #marker_coords = [state.od_avg.x[0], state.od_avg.x[len(state.od_avg.x) // 2], state.od_avg.x[-1]]
            #
            #for i, x in enumerate(marker_coords):
            #    color = 'red'
            #
            #    # Create Line2D objects for vertical lines
            #    line1 = Line2D([x, x], [self.ylim_od[0], self.ylim_od[1]], color=color, linestyle='--')
            #    line2 = Line2D([x, x], [self.ylim_id[0], self.ylim_id[1]], color=color, linestyle='--')
            #
            #    # Add the lines to the axes
            #    self.ax1_markers.add_line(line1)
            #    self.ax2_markers.add_line(line2)
            #

            # Create Line2D objects for markers on both axes
            self.od_markers_line = Line2D([], [], color='red', marker='o', markersize=5, linewidth=1, label='Markers', linestyle='-')
            self.id_markers_line = Line2D([], [], color='red', marker='o', markersize=5, linewidth=1, label='Markers', linestyle='-')

            # Add the marker lines to their respective axes
            self.ax1_markers.add_line(self.od_markers_line)
            self.ax2_markers.add_line(self.id_markers_line)

            # Update marker positions based on state.markers.y
            marker_x = []
            od_marker_y = []
            id_marker_y = []

            # Create Line2D objects for markers
            count = 1
            for x, y in zip(state.markers.x, state.markers.y):
                if y == 1:
                    color = 'green'
                    marker_line_od = Line2D([x, x], [self.ylim_od[0], self.ylim_od[1]], color=color, marker='o', markersize=5, linewidth=1)
                    marker_line_id = Line2D([x, x], [self.ylim_id[0], self.ylim_id[1]], color=color, marker='o', markersize=5, linewidth=1)

                    # Add the lines to the axes
                    self.ax1_markers.add_line(marker_line_od)
                    self.ax2_markers.add_line(marker_line_id)

                    # Add labels
                    self.ax1_markers.annotate(
                        f"{int(count)}",  # Convert x to integer for label
                        (x, self.ylim_od[1]),
                        xytext=(0, 5),
                        textcoords='offset points',
                        color=color,
                        ha='center',
                        va='center'
                    )

                    self.ax2_markers.annotate(
                        f"{int(count)}",  # Convert x to integer for label
                        (x, self.ylim_id[1]),
                        xytext=(0, 5),
                        textcoords='offset points',
                        color=color,
                        ha='center',
                        va='center'
                    )
                    count += 1

            if state.vertical_indicator is not None:
                self.ax1_vline.set_xdata([state.vertical_indicator])
                self.ax2_vline.set_xdata([state.vertical_indicator])


            self.figure.canvas.draw()

            # self.figure.canvas.blit(self.ax1.bbox)
            # self.figure.canvas.blit(self.ax2.bbox)

            state.dirty.set(False)

    def clear_markers(self):
        # Remove existing vertical lines and text annotations
        for line in self.ax1_markers.lines + self.ax2_markers.lines:
            line.remove()

        for text in self.ax1_markers.texts + self.ax2_markers.texts:
            text.remove()

    def update_lims(self):
        settings = self.state_vars.toolbar.graph
        self.xlim = (settings.x_min.get(), settings.x_max.get())
        self.ylim_id = (settings.y_min_id.get(), settings.y_max_id.get())
        self.ylim_od = (settings.y_min_od.get(), settings.y_max_od.get())

        self.ax1.set_xlim(*self.xlim)
        self.ax1.set_ylim(*self.ylim_od)
        self.ax2.set_xlim(*self.xlim)
        self.ax2.set_ylim(*self.ylim_id)

        self.ax1_markers.set_ylim(*self.ylim_od)
        self.ax2_markers.set_ylim(*self.ylim_id)

        # After setting up your initial plot and axes limits
        self.figure.canvas.draw()  # Draw the canvas with the initial plot
        self.toolbar.update()  # Update the navigation toolbar to reflect the current state
        #self.toolbar.push_current()  # Push the current view to the stack to set it as the 'home' position


    def update_lims_callback(self):
        settings = self.state_vars.toolbar.graph
        if settings.dirty.get():
            self.update_lims()
            settings.dirty.set(False)

    def update_lims_fromfile_callback(self):
        # Retrieve the current settings
        settings = self.state_vars.toolbar.graph
        if settings.limits_dirty.get():
            # Process outer diameter data
            xdata, ydata = self.od_avg.get_data()
            xlim_min_od = xdata[0]
            xlim_max_od = xdata[-1]

            ylim_min_od = np.floor(np.min(ydata) / 50) * 50
            ylim_max_od = np.ceil(np.max(ydata) / 50) * 50

            # Set the new limits for outer diameter
            settings.x_min.set(xlim_min_od)
            settings.x_max.set(xlim_max_od)
            settings.y_min_od.set(ylim_min_od)
            settings.y_max_od.set(ylim_max_od)

            # Update the xlim and ylim attributes for outer diameter
            self.xlim = (xlim_min_od, xlim_max_od)
            self.ylim_od = (settings.y_min_od.get(), settings.y_max_od.get())

            # Set the new limits on the axes for outer diameter
            self.ax1.set_xlim(*self.xlim)
            self.ax1.set_ylim(*self.ylim_od)
            self.ax1_markers.set_ylim(*self.ylim_od)

            # Process inner diameter data
            try:
                xdata, ydata = self.id_avg.get_data()
                ylim_min_id = np.floor(np.min(ydata) / 50) * 50
                ylim_max_id = np.ceil(np.max(ydata) / 50) * 50

                # Set the new limits for inner diameter
                settings.y_min_id.set(ylim_min_id)
                settings.y_max_id.set(ylim_max_id)

                # Update the ylim attributes for inner diameter
                self.ylim_id = (settings.y_min_id.get(), settings.y_max_id.get())

                # Set the new limits on the axes for inner diameter
                self.ax2.set_xlim(*self.xlim)
                self.ax2.set_ylim(*self.ylim_id)
                self.ax2_markers.set_ylim(*self.ylim_id)
            except Exception as e:
                print("Could not update inner diameter limits due to:", e)
                pass

            # Redraw the figure canvas
            self.figure.canvas.draw()

            # Reset the limits dirty flag
            settings.limits_dirty.set(False)



    def clear_graph(self):

        state = self.state_vars.graph
        # Clear the lines and data in both subplots (ax1 and ax2)
        self.od_avg.set_xdata([])
        self.od_avg.set_ydata([])
        self.id_avg.set_xdata([])
        self.id_avg.set_ydata([])

        for i in range(NUM_LINES):
            self.od_lines[i].set_xdata([])
            self.od_lines[i].set_ydata([])
            self.id_lines[i].set_xdata([])
            self.id_lines[i].set_ydata([])

        # Clear the data stored in state variables
        state.od_avg.x = []
        state.od_avg.y = []
        state.id_avg.x = []
        state.id_avg.y = []

        for i in range(NUM_LINES):
            state.od_lines[i].x = []
            state.od_lines[i].y = []
            state.id_lines[i].x = []
            state.id_lines[i].y = []

        (self.od_avg,) = self.ax1.plot([], [])
        (self.id_avg,) = self.ax2.plot([], [])

        # Redraw the cleared graph
        self.figure.canvas.draw()


class TableFrame(ttk.Frame):
    def __init__(self, parent, state: VtState):
        super().__init__(parent)
        self.parent = parent
        self.state_vars = state

        self.grid_propagate(False)

        self.width = self.winfo_width()
        self.height = self.winfo_height()

        self.setup_widgets()

        self.state_vars.table.dirty.trace_add(
            "write", lambda *args: self.add_row_callback()
        )

        self.state_vars.table.clear.trace_add(
            "write", lambda *args: self.clear_table()
        )

    def setup_widgets(self):
        sv = self.state_vars.table

        padx = 8

        # Create a style instance
        style = ttk.Style()
        style.configure("Treeview.Heading", font=(default_font, 10, "bold"), foreground=VasoTracker_Blue)
        style.configure("Treeview", font=(default_font, 10), foreground=VasoTracker_Blue)

        table_controls = ctk.CTkFrame(self)
        self.table_controls = table_controls
        table_controls.grid(
            row=0, column=0, columnspan=5, sticky=tk.N + tk.S + tk.E + tk.W
        )
        #ctk.CTkLabel(table_controls, text="Label:").grid(row=0, column=0)
        self.label_entry = ctk.CTkEntry(table_controls, width=200, textvariable=sv.label, font=(default_font, default_font_size), fg_color="white")
        self.label_entry.grid(row=0, column=1)
        self.add_button = ctk.CTkButton(table_controls, text="Add", font=(default_font, default_font_size), width=80, text_color="black")
        self.add_button.grid(row=0, column=2, padx=padx)

        ctk.CTkLabel(table_controls, text="Ref Diameter:", font=(default_font, default_font_size)).grid(
            row=0, column=4, padx=(20, 0)
        )
        self.ref_diam_entry = ctk.CTkEntry(
            table_controls, width=60, textvariable=sv.ref_diam, font=(default_font, default_font_size), fg_color=entry_disabled_color
            )
        self.ref_diam_entry.grid(row=0, column=5)
        self.ref_diam_entry.configure(state=tk.DISABLED)

        self.ref_button = ctk.CTkButton(table_controls, text="Set ref", font=(default_font, default_font_size), width=80, text_color="black")
        self.ref_button.grid(row=0, column=6, padx=padx)
        
        self.table = ttk.Treeview(self, show="headings")
        self.table["columns"] = sv.headers()

        self.table.column("#0", width=25)
        self.table.column("#", width=25)
        self.table.column("Time", width=75, stretch=False)
        self.table.column("Frame", width=50, stretch=False)
        self.table.column("Label", width=200)
        self.table.column("OD", width=50)
        self.table.column("%OD ref", width=75)
        self.table.column("ID", width=50)
        self.table.column("Caliper", width=75)
        self.table.column("Pavg", width=50)
        self.table.column("P1", width=50)
        self.table.column("P2", width=50)
        self.table.column("Temp", width=50)

        self.table.heading("#1", text="#")
        self.table.heading("#2", text="Time")
        self.table.heading("#3", text="Frame")
        self.table.heading("#4", text="Label")
        self.table.heading("#5", text="OD")
        self.table.heading("#6", text="%OD ref")
        self.table.heading("#7", text="ID")
        self.table.heading("#8", text="Caliper")
        self.table.heading("#9", text="Pavg")
        self.table.heading("#10", text="P1")
        self.table.heading("#11", text="P2")
        self.table.heading("#12", text="Temp")

        # Create a horizontal scrollbar and link it to the table
        h_scrollbar = ttk.Scrollbar(self, orient="horizontal", command=self.table.xview)
        self.table.configure(xscrollcommand=h_scrollbar.set)
        h_scrollbar.grid(row=2, column=0, sticky=tk.E + tk.W)

        v_scrollbar = ttk.Scrollbar(self)
        v_scrollbar.grid(row=1, column=2, sticky=tk.N + tk.S)
        v_scrollbar.configure(command=self.table.yview)
        self.table.grid(row=1, column=0, sticky=tk.N + tk.S + tk.E + tk.W)
        self.table.configure(yscrollcommand=v_scrollbar.set)
        self.grid_rowconfigure(0, weight=1, minsize=30)
        self.grid_rowconfigure(1, weight=9)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)  # Make the table column expandable

    def add_row(self, row: List[str]):
        self.table.insert(
            "",
            "end",
            values=row,
        )
        self.table.yview_moveto(1)

    def add_row_callback(self):
        table = self.state_vars.table
        if table.dirty.get():
            for row in table.rows_to_add:
                self.add_row(row)
            table.rows_to_add.clear()

            table.dirty.set(False)

    def clear_table(self):
        for item in self.table.get_children():
            self.table.delete(item)
        self.state_vars.table.clear.set(False)

def resize_image_to_fit(im: Image, width: int, height: int):
    curr_width, curr_height = im.size
    width_ratio = width / curr_width
    height_ratio = height / curr_height
    resize_ratio = min(width_ratio, height_ratio)

    return im.resize((int(curr_width * resize_ratio), int(curr_height * resize_ratio)))


class CameraFrame(ctk.CTkFrame):
    def __init__(self, parent, state: VtState):
        super().__init__(parent)
        self.parent = parent
        self.state_vars = state

        self.grid_propagate(False)

        # self.label_text = StringVar(value="y x z")
        # ctk.CTkLabel(self, textvariable=self.label_text).pack()

        # def update_text(event):
        #     self.label_text.set(f"{event.width} x {event.height}")

        # self.bind("<Configure>", update_text)

        self.setup_widgets()

        self.state_vars.cam_show.dirty.trace_add("write", self.show_image_callback)

        self.state_vars.cam_show.slider_length_dirty.trace_add("write", self.update_slider_length) # For updating the length of the slider on image import.

        self.state_vars.cam_show.slider_dirty.trace_add("write", self.update_slider)

        self.state_vars.cam_show.slider_toggle_dirty.trace_add("write", self.toggle_slider_state)

        self.state_vars.cam_show.slider_change_state.trace_add("write", self.toggle_slider_state)

    def setup_widgets(self):
        # Fix: Ensure number_of_steps is an integer
        self.slider = tk.Scale(
            self,
            from_=0,
            to=99,  # Equivalent to 100-1
            orient=tk.HORIZONTAL,  # Fix: Correct parameter name
            length=200,  # Controls slider width
            resolution=1,  # Controls step size (1 ensures integer steps)
            command=self.update_image_from_slider
        )
        self.slider.pack(fill=tk.X, side=tk.BOTTOM)
        self.slider.configure(state="disabled")  # Initially disable the slider

        # Canvas setup
        self.canvas = tk.Canvas(self, background="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas_image_id = None

    def update_image_from_slider(self, *args):
        """Updates the image based on the slider position."""
        # Fix: Check if slider should be enabled before changing state
        if self.slider.cget("state") == "disabled":
            self.slider.configure(state="normal")

        current_value = self.slider.get()
        self.state_vars.cam_show.slider_position_manual = current_value

        '''
        When loading from a file only show the vertical indicator on the graph when we are not tracking and not acquiring i.e., only after the analysis has ran.
        '''
        if self.state_vars.camera.camera_name == "Image from file":
            if self.state_vars.app.acquiring.get() and self.state_vars.app.tracking_file.get() and not self.state_vars.app.tracking.get():
                self.state_vars.graph.vertical_indicator = (current_value - self.state_vars.camera.max_frame_count + 1)
            else:
                self.state_vars.graph.vertical_indicator = None
        else:
            self.state_vars.graph.vertical_indicator = None

        '''
        If we are running the analyis on an image from file, do NOT show the graph being plotted. After the analysis has run the graph is plotted.
        The else statement ensures the vertical line indicator on the graph is also plotted and we can see it.
        '''
        if (self.state_vars.camera.camera_name == "Image from file" and self.state_vars.app.tracking.get()):
            self.state_vars.graph.dirty.set(False)
        else:
            self.state_vars.graph.dirty.set(True)

        self.state_vars.cam_show.dirty.set(True)
        
        # Optional: Force UI update
        self.canvas.update_idletasks()

    def updateValue(self, value):
        current_value = self.slider.get()
        current_value = int(value)


    def update_slider_length(self, *args):
        l = self.state_vars.toolbar.image_dim.file_length.get()
        # Calculate the number of intervals, aiming for a maximum of 10
        raw_tick_interval = l / 10
        # Round the interval to the nearest 100, but ensure it's at least 100
        tick_interval = max(100, np.ceil(raw_tick_interval / 100) * 100)

        self.slider.config(from_=0, to=l-1)
        self.slider.config(tickinterval=tick_interval)
        self.state_vars.cam_show.slider_length_dirty.set(False)


    def update_slider(self, *args):
        #self.slider.config(state='normal')
        l = self.state_vars.camera.frame_count
        self.slider.set(l)
        #print ("l == ",l)
        self.state_vars.cam_show.slider_position = l
        self.state_vars.cam_show.slider_dirty.set(False)

        #self.slider.config(state='disabled')
        #print("Slider is disabled")

    def toggle_slider(self, *args):
        if self.slider.cget('state') == 'disabled':
            self.slider.configure(state='normal', command=self.update_slider)  # Enable the slider and set the command callback
        else:
            self.slider.configure(state='disabled', command="")  # Disable the slider and remove the command callback

    def toggle_slider_state(self, *args):
        if self.state_vars.camera.camera_name == "Image from file":
            self.slider.configure(state='normal')
        elif self.slider.cget('state') == 'disabled':
            self.slider.configure(state='normal')  # Enable the slider and set the command callback
        else:
            self.slider.configure(state='disabled')  # Disable the slider and remove the command callback



    def show_rois(self):
        state = self.state_vars.cam_show.canvas_draw_state
        for roi in list(state.multi_roi.values()) + [state.roi]:
            if roi is not None and roi.dirty:
                if roi.handle is None:
                    roi.handle = self.canvas.create_rectangle(*roi.fixed_corners(), outline=VasoTracker_Green_hex, fill="",  width=3)
                else:
                    self.canvas.coords(roi.handle, *roi.fixed_corners())
                roi.dirty = False
                if self.canvas_image_id is not None:
                    self.canvas.lift(roi.handle, self.canvas_image_id)

    def show_calipers(self):
        state = self.state_vars.cam_show.canvas_draw_state
        for cal in list(state.autocaliper.values()) + [state.caliper]:
            if cal is not None and cal.dirty:
                if cal.handle is None:
                    cal.handle = self.canvas.create_line(
                        cal.x1,
                        cal.y1,
                        cal.x2,
                        cal.y2,
                        width=3,
                        fill=VasoTracker_Green_hex,
                    )
                else:
                    self.canvas.coords(cal.handle, cal.x1, cal.y1, cal.x2, cal.y2)
                cal.dirty = False
                if self.canvas_image_id is not None:
                    self.canvas.lift(cal.handle, self.canvas_image_id)

    def cleanup_rois(self):
        state = self.state_vars.cam_show.canvas_draw_state
        for roi in state.roi_cleanup:
            if roi.handle is not None:
                self.canvas.delete(roi.handle)
        state.roi_cleanup.clear()

    def show_image(self):
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        # NOTE(cmo): Edge case while the app is setting up
        if min(width, height) < 20:
            return
        state = self.state_vars.cam_show
        im_data = state.im_data
        if im_data is None:
            return

        image = Image.fromarray(im_data)
        resized = resize_image_to_fit(image, width, height)
        state.im_presented_size = (resized.height, resized.width)
        x_centre = width // 2
        y_centre = height // 2
        state.im_centre = (y_centre, x_centre)

        tk_image = ImageTk.PhotoImage(resized)
        # NOTE(cmo): Need to assign this image to the class or it gets gc'd
        self.tk_image = tk_image
        if self.canvas_image_id is not None:
            # NOTE(cmo): Update the image if it exists, rather than adding a new one.
            self.canvas.itemconfigure(self.canvas_image_id, image=tk_image)
            self.canvas.coords(self.canvas_image_id, x_centre, y_centre)
        else:
            self.canvas_image_id = self.canvas.create_image(
                x_centre,
                y_centre,
                anchor=tk.CENTER,
                image=tk_image,
            )

    def show_image_callback(self, *args):
        im_dirty = self.state_vars.cam_show.dirty
        if not im_dirty.get():
            return

        self.show_image()
        self.show_rois()
        self.show_calipers()
        self.cleanup_rois()

        im_dirty.set(False)


class View(ctk.CTkFrame):
    def __init__(
        self, root: ctk.CTk, state: VtState, set_camera_callback, shutdown_callbacks: List[Callable[[], None]]
    ):
        super().__init__(root)
        self.root = root
        root.iconbitmap(os.path.join(images_folder, 'vt_icon.ICO')) #(Path(__file__).parent / "images" / "VasoTracker_Icon.ICO") #
        root.wm_title(f"VasoTracker {__version__}")

        # Maximize the window without covering the taskbar
        root.state('zoomed')

        self.state_vars = state
        self.menus = Menus(root)
        self.toolbar = ToolbarView(self, state, set_camera_callback)
        self.graph = GraphFrame(self, state)
        self.table = TableFrame(self, state)
        self.camera = CameraFrame(self, state)
        self.status_bar = tk.Label(
            self,
            text="Thank you for using VasoTracker. To support us, please cite the latest VasoTracker release (click here for the paper).",
            relief="sunken",
            anchor="w",
        )

        # Add a link to the status bar along the bottom
        def callback(event):
            webbrowser.open_new(r"https://doi.org/10.1101/2025.04.23.648411")
        self.status_bar.bind("<Button-1>", callback)

        self.pack(fill=tk.BOTH, expand=False)

        self.toolbar.grid(
            row=0,
            column=0,
            rowspan=1,
            columnspan=3,
            sticky=tk.N + tk.S + tk.E + tk.W,
            padx=2,
            pady=2,
        )
        self.graph.grid(
            row=1,
            column=0,
            rowspan=2,
            columnspan=2,
            sticky=tk.N + tk.S + tk.E + tk.W,
            padx=2,
            pady=2,
        )
        self.table.grid(
            row=1,
            column=2,
            rowspan=1,
            columnspan=1,
            sticky=tk.N + tk.S + tk.E + tk.W,
            padx=2,
            pady=2,
        )
        self.camera.grid(
            row=2,
            column=2,
            rowspan=1,
            columnspan=1,
            sticky=tk.N + tk.S + tk.E + tk.W,
            padx=2,
            pady=2,
        )
        self.status_bar.grid(
            row=3,
            column=0,
            columnspan=3,
            sticky=tk.W + tk.E,
        )
        
        screen_height = self.root.winfo_screenheight()

        # Formula: Toolbar height is 12% of the screen height, but clamped between 140px and 220px
        toolbar_height = max(225, min(int(screen_height * 0.25), 250))

        self.grid_rowconfigure(0, weight=0, minsize=toolbar_height)
        # Apply the minimum size dynamically
 

        self.grid_rowconfigure(1, weight=3, uniform="row")
        self.grid_rowconfigure(2, weight=6, uniform="row")

        self.grid_columnconfigure(0, weight=1, uniform="column")
        self.grid_columnconfigure(1, weight=1, uniform="column")
        self.grid_columnconfigure(2, weight=1, uniform="column")

        # NOTE(cmo): Pop the window to the top (unhide it)
        root.deiconify()

        def set_toolbar_min_height():
            root.update_idletasks()  # Forces layout update
            self.grid_rowconfigure(0, weight=2, minsize=self.toolbar.winfo_height(), uniform="row")

        # NOTE(cmo): Can't do this until the widget is drawn.
        root.after(100, set_toolbar_min_height)

        self.shutdown_callbacks = shutdown_callbacks
        root.protocol("WM_DELETE_WINDOW", lambda *args: self.shutdown_app())

        self.setup_message_handlers()
        # NOTE(cmo): Check for messages to display in case one was posted before
        # we set up our handler
        self.handle_message_callback()

    def shutdown_app(self, force: bool = False):
        if force or tmb.askokcancel("Quit", "Are you sure?"):
            plt.close("all")
            for cb in self.shutdown_callbacks:
                cb()

            # NOTE(cmo): Give the app a chance for other threads to shutdown
            time.sleep(0.05)
            root.withdraw()
            root.quit()
            # NOTE(cmo): Running quit before withdraw prevents errors about
            # remaining `after` callbacks -- even when no more were queued.
            root.destroy()

    def setup_message_handlers(self):
        self.state_vars.message.dirty.trace_add(
            "write", lambda *args: self.handle_message_callback()
        )

    def handle_message_callback(self):
        message = self.state_vars.message
        dirty = message.dirty
        if not dirty.get():
            return

        if message.type == MessageType.Info:
            box_fn = tmb.showinfo
        elif message.type == MessageType.Warning:
            box_fn = tmb.showwarning
        elif message.type == MessageType.Error:
            box_fn = tmb.showerror

        box_fn(title=message.title, message=message.message)
        dirty.set(False)


class CameraInteractionMode(IntEnum):
    Default = auto()
    SetRoi = auto()
    SetCaliper = auto()
    AddMultiRoi = auto()
    AddAutoCaliper = auto()


class CameraController:
    def __init__(self, model: Model, view: View):
        self.model = model
        self.view = view
        self.state = model.state.cam_show
        self.mode = CameraInteractionMode.Default
        self.was_acquiring = False

        canvas = self.view.camera.canvas
        canvas.bind("<ButtonPress-1>", self.handle_press)
        canvas.bind("<B1-Motion>", self.handle_motion)
        canvas.bind("<ButtonRelease-1>", self.handle_release)

    def handle_press(self, event):
        if self.mode == CameraInteractionMode.Default:
            return

        state = self.state.canvas_draw_state
        state.user_drawing.set(True)
        self.was_acquiring = self.model.state.app.acquiring.get()
        if self.was_acquiring:
            self.model.state.app.acquiring.set(False)

        x, y = event.x, event.y
        if self.mode == CameraInteractionMode.SetRoi:
            if state.roi is not None:
                state.roi_cleanup.append(state.roi)
            state.roi = Roi(x, x, y, y, dirty=True)
        elif self.mode == CameraInteractionMode.SetCaliper:
            if state.caliper is not None:
                state.roi_cleanup.append(state.caliper)
            state.caliper = Roi(x, x, y, y, dirty=True)
        elif self.mode == CameraInteractionMode.AddMultiRoi:
            multi = state.multi_roi
            multi[f"ROI{len(multi)}"] = Roi(x, x, y, y, dirty=True)
        elif self.mode == CameraInteractionMode.AddAutoCaliper:
            caliper = state.autocaliper
            caliper[f"Caliper{len(caliper)}"] = Roi(x, x, y, y, dirty=True)
        else:
            raise ValueError("Unhandled mode!")

        self.state.dirty.set(True)

    def handle_motion(self, event, set_dirty=True):
        if self.mode == CameraInteractionMode.Default:
            return

        x, y = event.x, event.y
        state = self.state.canvas_draw_state
        if self.mode == CameraInteractionMode.SetRoi:
            roi_to_update = state.roi
        elif self.mode == CameraInteractionMode.SetCaliper:
            roi_to_update = state.caliper
        elif self.mode == CameraInteractionMode.AddMultiRoi:
            multi = state.multi_roi
            idx = len(multi) - 1
            roi_to_update = multi[f"ROI{idx}"]
        elif self.mode == CameraInteractionMode.AddAutoCaliper:
            caliper = state.autocaliper
            idx = len(caliper) - 1
            roi_to_update = caliper[f"Caliper{idx}"]
        else:
            raise ValueError("Unhandled mode!")

        roi_to_update.x2 = x
        roi_to_update.y2 = y
        if set_dirty:
            roi_to_update.dirty = True
            self.state.dirty.set(True)

    def handle_release(self, event):
        if self.mode == CameraInteractionMode.Default:
            return

        state = self.state.canvas_draw_state
        # NOTE(cmo): Update for potential cursor movement since last update
        # cycle.
        self.handle_motion(event, set_dirty=False)

        def image_space_coords(x, y):
            # NOTE(cmo): if an image isn't currently being displayed, the model
            # will ignore the additions anyway
            if self.state.im_data is None:
                return 1, 1
            image_top = self.state.im_centre[0] - self.state.im_presented_size[0] // 2
            image_left = self.state.im_centre[1] - self.state.im_presented_size[1] // 2
            ratio = self.state.im_data.shape[0] / self.state.im_presented_size[0]
            return int((x - image_left) * ratio), int((y - image_top) * ratio)

        if self.mode == CameraInteractionMode.SetRoi:
            roi = state.roi
            self.model.set_roi(
                *image_space_coords(roi.x1, roi.y1),
                *image_space_coords(roi.x2, roi.y2),
            )
            state.roi_cleanup.append(roi)
            state.roi = None

        elif self.mode == CameraInteractionMode.SetCaliper:
            cal = state.caliper
            self.model.set_caliper(
                *image_space_coords(cal.x1, cal.y1),
                *image_space_coords(cal.x2, cal.y2),
            )
            state.roi_cleanup.append(cal)
            state.caliper = None
        elif self.mode == CameraInteractionMode.AddMultiRoi:

            def path_from_roi(roi):
                path1 = MplPath(
                    [
                        [roi.x1, roi.y1],
                        [roi.x2, roi.y1],
                        [roi.x2, roi.y2],
                        [roi.x1, roi.y2],
                        [roi.x1, roi.y1],
                    ]
                )
                return path1

            idx = len(state.multi_roi) - 1
            key = f"ROI{idx}"
            new_addition = state.multi_roi[key]
            x1, y1 = (*image_space_coords(new_addition.x1, new_addition.y1),)
            x2, y2 = (*image_space_coords(new_addition.x2, new_addition.y2),)
            im_space = Roi(x1=x1, y1=y1, x2=x2, y2=y2)
            new_path = path_from_roi(im_space)
            # NOTE(cmo): Existing ones will be in the raster roi data
            to_check = self.state.raster_draw_state.multi_roi
            if len(to_check) < NUM_ROIS:
                intersections = []
                for rr in to_check.values():
                    path_to_check = path_from_roi(rr)
                    intersections.append(
                        new_path.intersects_path(path_to_check, filled=False)
                    )
                # NOTE(cmo): Add if no intersection
                if not any(intersections):
                    self.model.add_multi_roi(
                        im_space.x1,
                        im_space.y1,
                        im_space.x2,
                        im_space.y2,
                    )
            # NOTE(cmo): Clean up canvas state
            state.roi_cleanup.append(new_addition)
            del state.multi_roi[key]
        elif self.mode == CameraInteractionMode.AddAutoCaliper:
            #to_check = self.state.raster_draw_state.autocaliper
            #if len(to_check) < NUM_LINES:
            key = f"Caliper{len(state.autocaliper)-1}"
            cal = state.autocaliper[key]
            self.model.add_auto_caliper(
                *image_space_coords(cal.x1, cal.y1),
                *image_space_coords(cal.x2, cal.y2),
            )
            # NOTE(cmo): Cleanup canvas state
            state.roi_cleanup.append(cal)
            del state.autocaliper[key]
        else:
            raise ValueError("Unhandled mode!")

        self.state.dirty.set(True)
        self.mode = CameraInteractionMode.Default
        self.state.canvas_draw_state.user_drawing.set(False)
        if self.was_acquiring:
            self.model.state.app.acquiring.set(True)
            self.was_acquiring = False





class Controller:
    def __init__(self, root, mmc):
        self.model = Model(mmc, set_timeout=root.after)
        shutdown_callbacks = []
        shutdown_callbacks.append(self.model.get_shutdown_callback())
        self.view = View(root, self.model.state, self.set_camera, shutdown_callbacks=shutdown_callbacks)
        self.camera_controller = CameraController(self.model, self.view)

        # Instantiate the PressureController
        if is_pydaqmx_available:
            self.pressure_controller = PressureController(self.model, self.view, utilities.VT_Pressure.is_pydaqmx_available())
        else:
            self.pressure_controller = None
        self.model.set_pressure_controller(self.pressure_controller)
        self.model.state.pressure_controller = self.pressure_controller

        # Instantiate the ArduinoController
        self.arduino_controller = ArduinoController(self)
        self.model.set_arduino_controller(self.arduino_controller)
        self.model.state.arduino_controller = self.arduino_controller


        self.bind_buttons()
        self.bind_checkboxes()
        self.bind_menu_items()

        self.output_path = None

        #output_path = self.get_output_filename()
        #self.model.setup_output_files(output_path=output_path)

        if self.model.configure.registration.register_flag == 0:
            # Prompt user to register
            # On successful registration:
            splash = VasoTrackerSplashScreen(root, self.update_settings)
            splash.splash_win.focus_force()

        self.model.process_updates()

    def get_output_filename(self):
        # Create a folder with the current date
        now = datetime.now()
        folder_name = now.strftime("%Y%m%d")
        #main_folder_path = os.path.join("Results", folder_name)

        #
        # Use the Documents folder as the base
        documents_folder = os.path.expanduser("~/Documents")
        main_folder_path = os.path.join(documents_folder, "Results", folder_name)

        #Create the main folder if it doesn't exist
        os.makedirs(main_folder_path, exist_ok=True)

        # Initialize the filename and counter
        savename = now.strftime("%Y%m%d")
        counter = 1

        while True:
            # Generate the subfolder name with _ExpXX suffix
            subfolder_name = f"{savename}_Exp{counter:02d}"

            # Generate the full path for the subfolder
            subfolder_path = os.path.join(main_folder_path, subfolder_name)

            # Generate the filename with .csv extension
            filename = f"{subfolder_name}.csv"

            # Check if the subfolder already exists
            if os.path.exists(subfolder_path):
                counter += 1  # Increment the counter
            else:
                # Create the subfolder
                os.makedirs(subfolder_path)
                break  # Exit the loop if the subfolder is created

        # Ask for the filename within the subfolder
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialdir=subfolder_path,
            initialfile=filename,
        )

        if not file_path:
            if (
                tmb.askquestion(
                    "No save file selected",
                    "Do you want to quit VasoTracker?",
                    icon="warning",
                )
                == "yes"
            ):
                self.view.shutdown_app(force=True)

        return file_path





    def bind_buttons(self):
        tb = self.view.toolbar

        #tb.acq.camera_entry.bind("<Configure>", lambda *args: self.set_camera())

        '''
        tb.acq.res_entry.bind(
            "<Configure>", lambda *args: self.set_camera_resolution()
        )
        tb.acq.fov_entry.bind("<Configure>", lambda *args: self.set_camera_fov())
        '''

        tb.graph.set_button.configure(command=self.set_graph_lims)
        tb.graph.default_button.configure(command=self.model.set_default_graph_lims)

        tb.caliper_roi.roi_button.configure(command=self.delete_all_rois)
        tb.caliper_roi.caliper_button.configure(command=self.delete_all_rois)


        # Single caliper/roi buttons
        tb.caliper_roi.draw_roi_button.configure(command=self.roi_manual_draw)
        tb.caliper_roi.draw_caliper_button.configure(command=self.caliper_manual_draw)
        tb.caliper_roi.delete_roi_caliper_button.configure(command=self.caliper_manual_delete)

        # Multi caliper/roi buttons
        tb.caliper_roi.auto_add_button.configure(command=self.caliper_auto_add)
        tb.caliper_roi.auto_delete_button.configure(command=self.caliper_auto_delete)
        tb.caliper_roi.auto_delete_all_button.configure(command=self.caliper_auto_delete_all)
        
        tb.caliper_roi.showtraces_button.configure(command=self.show_plotting_popup)
        '''
        for i in range(NUM_LINES):
            tb.plotting.line_buttons[i].configure(command=partial(self.toggle_line, i))
        '''

        if is_pydaqmx_available:
            tb.pressure_control_settings.start_protocol_button.configure(command=self.servo_start)
            #tb.pressure_protocol_settings.stop_protocol_button.configure(command=self.servo_stop)
            tb.pressure_control_settings.add_button.configure(command=self.increase_pressure)
            tb.pressure_control_settings.minus_button.configure(command=self.decrease_pressure)

        tb.start_stop.start_button.configure(command=self.start_acq)
        tb.start_stop.track_button.configure(command=self.start_tracking)
        #tb.start_stop.record_button.configure(command=self.record_data)
        tb.start_stop.snapshot_button.configure(command=self.take_snapshot)
        self.view.table.add_button.configure(command=self.add_table_row)
        self.view.table.ref_button.configure(command=self.set_ref_diameter)

        if is_pydaqmx_available:
            tb.pressure_control_settings.set_pressure_button.configure(command=self.update_set_pressure)
            tb.pressure_control_settings.pressure_connect_button.configure(command=self.open_pressure_settings)
            tb.pressure_control_settings.pressure_settings_button.configure(command=self.open_pressure_protocol_settings)


    def bind_checkboxes(self):
        """Bind checkboxes that need to have callbacks when clicked, rather than
        just boolean state updates."""
        tb = self.view.toolbar

        tb.acq.default_settings.configure(command=self.acq_set_default)

    def bind_menu_items(self):
        menu = self.view.menus
        file = menu.file_menu
        file.entryconfig(
            file.index("New file..."), command=self.menu_new_file
        )

        file.entryconfig(
            file.index("Analyze file..."), command=self.menu_analyze_file
        )

        file.entryconfig(
            file.index("Load settings..."), command=self.menu_load_settings
        )
        file.entryconfig(
            file.index("Save settings..."), command=self.menu_save_settings
        )
        file.entryconfig(file.index("Exit"), command=self.menu_exit)

        help_menu = menu.help_menu
        help_menu.entryconfig(
            help_menu.index("Boogie woogie"), command=self.menu_rock_and_or_roll
        )
        help_menu.entryconfig(
            help_menu.index("Register"), command=self.menu_register
        )

        help_menu.entryconfig(
            help_menu.index("User Guide"), command=self.menu_user_guide
        )
        #help_menu.entryconfig(help_menu.index("Contact"), command=self.menu_contact)
        help_menu.entryconfig(help_menu.index("About"), command=self.menu_about)
        help_menu.entryconfig(help_menu.index("Update"), command=self.menu_update)

        # Create the "Settings" dropdown menu
        file_details_menu = menu.settings_menu
        file_details_menu.entryconfig(
            file_details_menu.index("File details"), command=self.show_file_details_popup
        )

        # Create the "Settings" dropdown menu
        settings_menu = menu.settings_menu
        settings_menu.entryconfig(
            settings_menu.index("Image Dimensions"), command=self.show_image_dimensions_popup
        )

        # Create the "Plotting" dropdown menu
        settings_menu = menu.settings_menu
        settings_menu.entryconfig(
            settings_menu.index("Graph Axes"), command=self.show_axes_popup
        )


        # Create the "Plotting" dropdown menu
        settings_menu = menu.settings_menu
        settings_menu.entryconfig(
            settings_menu.index("Show/Hide Traces"), command=self.show_plotting_popup
        )
        if is_pydaqmx_available:
            # Create the "DAQ Setup" dropdown menu
            settings_menu = menu.settings_menu
            settings_menu.entryconfig(
                settings_menu.index("DAQ Setup"), command=self.show_daq_settings
            )

            # Create the "Pressure Protocol" dropdown menu
            settings_menu = menu.settings_menu
            settings_menu.entryconfig(
                settings_menu.index("Configure Pressure Protocol"), command=self.show_pressure_settings
            )

        # Create the "Notepad"
        notepad_menu = menu.notepad_menu
        notepad_menu.entryconfig(
            notepad_menu.index("Open notepad"), command=self.show_notepad
        )




    def set_camera(self, cam_name=None):
        print("setting the camera...")
        if cam_name is None:
            cam_name = self.model.state.toolbar.acq.camera.get()
        
        print("Camera name:", cam_name)
        self.model.set_camera(cam_name)

    def set_camera_fov(self):
        tb = self.model.state.toolbar
        fov = tb.acq.fov.get()
        w = tb.image_dim.cam_width.get()
        h = tb.image_dim.cam_height.get()
        if fov == "w x h":
            args = (0, 0, w, h)
        elif fov == "w/2 x h/2":
            args = (w / 4, h / 4, w / 2, h / 2)
        self.model.set_camera_fov(*args)

    def set_camera_resolution(self):
        tb = self.model.state.toolbar
        new_res = tb.acq.res.get()
        if new_res == ELLIPSIS:
            return
        x_s, y_s = new_res.split("x")
        self.model.set_camera_resolution(int(x_s), int(y_s))

    def set_ref_diameter(self):
        self.model.set_ref_diameter()

    def set_graph_lims(self):
        self.view.graph.update_lims()

    def roi_manual_draw(self):
        self.camera_controller.mode = CameraInteractionMode.SetRoi

    def caliper_manual_draw(self):
        self.camera_controller.mode = CameraInteractionMode.SetCaliper
        

    def caliper_manual_delete(self):
        try:
            self.model.delete_caliper()
        except:
            pass

        try:
            self.model.delete_roi()
        except:
            pass


    def caliper_auto_add(self):
        if self.model.state.toolbar.caliper_roi.roi_flag.get() == "Caliper":
            self.camera_controller.mode = CameraInteractionMode.AddAutoCaliper
        elif self.model.state.toolbar.caliper_roi.roi_flag.get() == "ROI":
            self.camera_controller.mode = CameraInteractionMode.AddMultiRoi
      

    def caliper_auto_delete(self):
        if self.model.state.toolbar.caliper_roi.roi_flag.get() == "Caliper":
            self.model.delete_most_recent_autocaliper()
        elif self.model.state.toolbar.caliper_roi.roi_flag.get() == "ROI":
            self.model.delete_most_recent_multi_roi()

    def delete_all_rois(self):
        self.model.delete_all_autocaliper()
        self.model.delete_all_multi_roi()

    def caliper_auto_delete_all(self):
        if self.model.state.toolbar.caliper_roi.roi_flag.get() == "Caliper":
            self.model.delete_all_autocaliper()
        elif self.model.state.toolbar.caliper_roi.roi_flag.get() == "ROI":
            self.model.delete_all_multi_roi()


    def roi_single_draw(self):
        self.camera_controller.mode = CameraInteractionMode.SetRoi

    def roi_single_delete(self):
        self.model.delete_roi()

    def roi_multi_add(self):
        self.camera_controller.mode = CameraInteractionMode.AddMultiRoi

    def roi_multi_delete(self):
        self.model.delete_most_recent_multi_roi()

    def toggle_line(self, i: int):
        state = self.model.state.toolbar.plotting.line_show[i]
        prev_state = state.get()
        new_state = not prev_state
        button_state = tk.SUNKEN if new_state else tk.RAISED

        #self.view.toolbar.plotting.line_buttons[i].configure(relief=button_state)
        state.set(new_state)
        self.model.state.graph.dirty.set(True)
        
        # Update button states in both PlottingPane instances
        self.view.toolbar.plotting.update_button_states()
        try: #If it exists
            self.menu_plotting_pane.update_button_states()  # Replace with actual reference
        except:
            pass
        
    def servo_start(self):
        current_state = self.model.state.app.auto_pressure.get()
        if current_state == 0:
            if tmb.askokcancel("Start Pressure Protocol", "Are you sure?"):
                start_time = time.time()
                self.model.state.toolbar.pressure_protocol.protocol_start_time.set(start_time)
                self.model.state.toolbar.pressure_protocol.pressure_protocol_flag.set(1)
                #self.view.toolbar.pressure_control_settings.toggle_protocol_button()
                self.model.state.app.auto_pressure.set(not current_state)
        else:
            if tmb.askokcancel("Stop Pressure Protocol", "Are you sure?"):
                self.model.state.toolbar.pressure_protocol.pressure_protocol_flag.set(0)
                #self.view.toolbar.pressure_control_settings.toggle_protocol_button()
                self.model.state.app.auto_pressure.set(not current_state)
                self.model.pressure_controller.reset_protocol()

    def servo_stop(self):
        self.model.state.toolbar.pressure_protocol.pressure_protocol_flag.set(0)

    def decrease_pressure(self):
        increment = self.model.state.toolbar.pressure_protocol.pressure_increment.get()
        current_pressure = self.model.state.toolbar.pressure_protocol.set_pressure.get()
        new_pressure = current_pressure - increment
        if new_pressure < 0:
            new_pressure = 0
        self.model.state.toolbar.pressure_protocol.set_pressure.set(new_pressure)

    def increase_pressure(self):
        increment = self.model.state.toolbar.pressure_protocol.pressure_increment.get()
        current_pressure = self.model.state.toolbar.pressure_protocol.set_pressure.get()
        new_pressure = current_pressure + increment
        if new_pressure > 200:
            new_pressure = 200
        self.model.state.toolbar.pressure_protocol.set_pressure.set(new_pressure)

    def start_acq(self):
        current_state = self.model.state.app.acquiring.get()
        if self.model.state.camera.camera_name == "Image from file":
            self.model.state.app.acquiring.set(True)
            self.model.state.app.tracking.set(False)
        else:
            if self.model.state.camera == None:
                tmb.showwarning(
                    title="Warning",
                    message="You need to select your camera to show images!",
                )
            else:
                self.model.state.app.acquiring.set(not current_state)
            current_state = self.model.state.app.acquiring.get()
            if current_state == 0:
                self.model.state.app.tracking.set(current_state)



    def start_tracking(self):
        if self.model.state.camera.camera_name == "Image from file":
            '''
            This ensure that when a file is loaded in tracking is set when the button is pressed.
            '''
            self.setup_files()
            if self.model.state.app.tracking.get():
                self.model.state.app.tracking.set(False)
                self.model.state.app.acquiring.set(False)
                self.model.state.app.file_analysed.set(0)
            else:
                #Rerun the analysis in the While acq code.
                self.model.state.app.tracking.set(True)
                self.model.state.app.acquiring.set(True)
                self.model.state.app.file_analysed.set(0)
                self.reset_model_variables()
        else:
            if self.model.state.camera == None:
                tmb.showwarning(
                    title="Warning",
                    message="You need to select your camera to show images!",
                )
            else:
                if not self.output_path:
                    tmb.showwarning(
                        title="Warning",
                        message="You need to set up an output file (File -> New File).",
                    )
                else:
                    current_state = self.model.state.app.acquiring.get()

                    if current_state == 0:
                        self.model.state.app.acquiring.set(not current_state)
                        current_time = time.time()
                        self.model.start_time = current_time
                    current_state = self.model.state.app.tracking.get()
                    self.model.state.app.tracking.set(not current_state)

    def start_tracking_file(self):
        self.model.state.app.tracking_file.set(True)



    def take_snapshot(self):
        im_data = self.model.state.cam_show.im_data
        if im_data is not None:
            self.model.save_snapshot(im_data, subdir=None)



    def open_browser_kofi(self):
        webbrowser.open_new(r"https://ko-fi.com/vasotracker")

    def acq_set_default(self):
        if self.model.state.toolbar.acq.default_settings.get():
            self.model.set_default_acq_settings()

    def add_table_row(self):
        self.model.add_table_row()

    def update_set_pressure(self):
        new_pressure_value = self.model.state.toolbar.pressure_protocol.set_pressure.get()
        self.pressure_controller.adjust_pressure(new_pressure_value, update_table=True)

    def open_pressure_settings(self):
        self.view.toolbar.pressure_control_settings.start_protocol_button.configure(state=tk.NORMAL)
        self.view.toolbar.pressure_control_settings.start_protocol_button.configure(fg_color='white')
        self.view.toolbar.pressure_control_settings.set_pressure_button.configure(state=tk.NORMAL)
        self.view.toolbar.pressure_control_settings.set_pressure_button.configure(fg_color='white')

        self.show_daq_settings()

    def open_pressure_protocol_settings(self):
        self.show_pressure_settings()

    def disable_buttons_on_file_load(self):
        self.view.toolbar.start_stop.start_button.configure(state="disabled", fg_color=entry_disabled_color)
        pass

    def reset_model_variables(self):
        self.model.start_time = 0.0
        self.model.prev_update = 0.0
        self.model.time_elapsed = 0.0
        self.model.frame_count = 0
        self.model.state.cam_show.slider_position_manual = 0
        self.model.state.camera.reinitialize()
        self.model.state.frames_elapsed = 0

    def menu_analyze_file(self):
        # TODO: Probably need to reset everything here.
        # TODO: reset buttons to enabled after analysis ran
        self.model.state.toolbar.acq.camera.set("...")
        if tmb.askyesno("Load image file", message="Load file to analyze. Are you sure?"):
            self.model.state.app.file_analysed.set(0) # Reset this
            self.model.state.app.tracking.set(False)
            self.model.setup_default_ui_state_loadfile()
            #self.model.state.camera.camera_name.set()
            self.model.state.toolbar.acq.camera.set("Image from file")
            self.set_camera("Image from file")
            #self.model.state.cam_show.slider_change_state.set(True)
            self.output_path = None
            self.output_path = self.get_output_filename()
            #self.disable_buttons_on_file_load()
            self.setup_files()
            self.start_acq()
            #self.start_tracking()
            self.start_tracking_file()
            self.model.state.cam_show.slider_change_state.set(True)

    def setup_files(self):
        if self.output_path:
            self.model.setup_output_files(output_path=self.output_path)
            self.model.state.table.clear.set(True)
            self.model.state.graph.clear.set(True)
            self.reset_model_variables()


    def menu_new_file(self):
        if tmb.askokcancel("New experiment...", "Are you sure?"):
            self.model.state.app.tracking.set(False)
            self.model.state.app.tracking.set(False)

            self.output_path = None
            self.output_path = self.get_output_filename()
            if self.output_path:
                self.model.setup_output_files(output_path=self.output_path)

                self.model.state.table.clear.set(True)
                self.model.state.graph.clear.set(True)
            #TODO: Clear all data!!!

    def menu_load_settings(self):
        settings_filename = filedialog.askopenfilename(
            defaultextension=".toml",
            filetypes=(("toml files", "*.toml"), ("all files", "*.*")),
            initialfile="settings.toml",
            initialdir=os.getcwd(),
        )
        try:
            new_config = Config.from_file(settings_filename)
        except:
            traceback.print_exc()
            tmb.showerror(
                "Failed to load config",
                "Failed to load config file, continuing with previous settings",
            )
            return

        try:
            self.model.load_config(new_config)
        except:
            traceback.print_exc()
            tmb.showerror(
                "Critical error loading settings",
                "More details printed to console. App will now close.",
            )
            self.view.shutdown_app(force=True)
    
    def update_settings(self, flag_name, value):
        config = self.model.to_config()
        # Use setattr to update the flag
        setattr(self.model.configure.registration, flag_name, value)
        #self.model.configure.registration.set_values(self.state)
        #self.model.configure.save(override_path=self.model.config_path)

        """Get the path to a resource, whether it's bundled with PyInstaller or not."""
        base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
        self.model.configure.save(os.path.join(base_path, self.model.config_path))

    
    def menu_save_settings(self):
        now = datetime.now()
        savename = now.strftime("%Y%m%d") + "_Settings"
        path = filedialog.asksaveasfilename(
            defaultextension=".toml", initialfile=savename, initialdir=os.getcwd
        )
        if path == "":
            return
        print("Saving settings to: ", path)
        self.model.to_config().save(override_path=path)

    def menu_exit(self):
        self.view.shutdown_app()

    def menu_rock_and_or_roll(self):
        tmb.showinfo(
            "We like dancing in the shower",
            "Whether in the lab or in the shower, these songs make us boogie...",
        )
        webbrowser.open_new(
            r"https://open.spotify.com/playlist/5isnlNKb6Xtm975J9rxxT0?si=U5qpBEeHTKW9S0mLe70rKQ"
        )

    def menu_register(self):
        webbrowser.open_new(r"https://forms.office.com/e/Ke9mjE6CQg")

    def menu_user_guide(self):
        webbrowser.open_new(
            r"https://vasotracker.com/resources/"
        )

    def menu_about(self):
        webbrowser.open_new(r"http://www.vasotracker.com/about/")

    def menu_update(self):
        webbrowser.open_new(
            r"https://github.com/VasoTracker/VasoTracker-2/releases"
        )



    def show_file_details_popup(self):
        popup = tk.Toplevel(root)
        popup.title("File details")

        # Set the window icon to be the same as the main window
        icon_path = os.path.join(images_folder, 'vt_icon.ICO')#Path(__file__).parent / "images" / "VasoTracker_Icon.ICO"
        popup.iconbitmap(icon_path)

        # Ensure the popup window is non-resizable (if needed)
        popup.resizable(False, False)

        #popup.geometry("400x300")  # Set the size of the popup window
        #popup.grab_set()  # Make the popup window modal

        # Create a placeholder frame for ImageDimensionsPane using grid()
        frame = tk.Frame(popup)
        frame.grid(row=1, column=0, columnspan=2, sticky="nsew")

        # Create an instance of the ImageDimensionsPane within the frame
        image_dimensions_pane = SourcePane(frame, self.model.state)
        image_dimensions_pane.grid(sticky="nsew")

    def show_image_dimensions_popup(self):
        popup = tk.Toplevel(root)
        popup.title("Image Dimensions")

        # Set the window icon to be the same as the main window
        icon_path = os.path.join(images_folder, 'vt_icon.ICO')#Path(__file__).parent / "images" / "VasoTracker_Icon.ICO"
        popup.iconbitmap(icon_path)

        # Ensure the popup window is non-resizable (if needed)
        popup.resizable(False, False)

        #popup.geometry("400x300")  # Set the size of the popup window
        #popup.grab_set()  # Make the popup window modal

        # Create a placeholder frame for ImageDimensionsPane using grid()
        frame = tk.Frame(popup)
        frame.grid(row=1, column=0, columnspan=2, sticky="nsew")

        # Create an instance of the ImageDimensionsPane within the frame
        image_dimensions_pane = ImageDimensionsPane(frame, self.model.state)
        image_dimensions_pane.grid(sticky="nsew")


    def show_axes_popup(self):
        popup = tk.Toplevel(root)
        popup.title("Graph Axes:")

        # Set the window icon to be the same as the main window
        icon_path = os.path.join(images_folder, 'vt_icon.ICO')#Path(__file__).parent / "images" / "VasoTracker_Icon.ICO"
        popup.iconbitmap(icon_path)

        # Ensure the popup window is non-resizable (if needed)
        popup.resizable(False, False)

        #popup.grab_set()  # Make the popup window modal

        # Add a descriptive label
        label = tk.Label(popup)
        label.pack()

        # Create a placeholder frame for PlottingFrame using grid()
        frame = tk.Frame(popup)
        frame.pack()

        # Create an instance of the PlottingFrame within the frame
        self.graph_axis_pane = GraphSettingsPane(frame, self.model.state)
        self.graph_axis_pane.grid(sticky="nsew")

        # Link the buttons in the popup to their functionality
        self.graph_axis_pane.set_button.configure(command=self.set_graph_lims)
        self.graph_axis_pane.default_button.configure(command=self.model.set_default_graph_lims)


    def show_plotting_popup(self):
        popup = tk.Toplevel(root)
        popup.title("Show traces:")

        # Set the window icon to be the same as the main window
        icon_path = os.path.join(images_folder, 'vt_icon.ICO')#Path(__file__).parent / "images" / "VasoTracker_Icon.ICO"
        popup.iconbitmap(icon_path)

        # Ensure the popup window is non-resizable (if needed)
        popup.resizable(False, False)

        #popup.grab_set()  # Make the popup window modal

        # Add a descriptive label
        label = tk.Label(popup)
        label.pack()

        # Create a placeholder frame for PlottingFrame using grid()
        frame = tk.Frame(popup)
        frame.pack()

        # Create an instance of the PlottingFrame within the frame
        self.menu_plotting_pane = PlottingPane(frame, self.model.state)
        self.menu_plotting_pane.grid(sticky="nsew")
        
        for i in range(NUM_LINES):
           self.menu_plotting_pane.show_buttons[i].configure(command=partial(self.toggle_line, i))
        
        # Update the button states to reflect the current model state
        self.menu_plotting_pane.update_button_states()
        


    def show_daq_settings(self):
        # Create the popup window
        popup = tk.Toplevel(root)
        popup.title("NI DAQ Settings:")

        # Set the window icon to be the same as the main window
        icon_path = os.path.join(images_folder, 'vt_icon.ICO')  # Path to the icon file
        popup.iconbitmap(icon_path)
        
        # Ensure the popup window is non-resizable
        popup.resizable(False, False)

        # Add a descriptive label
        label = ctk.CTkLabel(popup, text="Configure the National Instruments DAQ settings:", font=(default_font, default_font_size))
        label.pack()

        # Create a placeholder frame for PlottingFrame using grid()
        frame = tk.Frame(popup)
        frame.pack()

        # Create an instance of the DAQ Settings within the frame
        self.menu_plotting_pane = ServoSettingsPane(frame, self.model.state)
        self.menu_plotting_pane.grid(sticky="nsew")

        # Ensure all the widgets are updated before showing the window
        popup.update_idletasks()

        # Make the popup window stay on top
        popup.attributes('-topmost', True)

        # Now show the window after everything is fully generated
        popup.deiconify()


    def show_pressure_settings(self):
        popup = tk.Toplevel(root)
        popup.title("Pressure Protocol Settings:")

        # Set the window icon to be the same as the main window
        icon_path = os.path.join(images_folder, 'vt_icon.ICO')#Path(__file__).parent / "images" / "VasoTracker_Icon.ICO"
        popup.iconbitmap(icon_path)

        # Ensure the popup window is non-resizable (if needed)
        popup.resizable(False, False)

        #popup.grab_set()  # Make the popup window modal

        # Add a descriptive label
        label = tk.Label(popup)
        label.pack()

        # Create a placeholder frame for PlottingFrame using grid()
        frame = tk.Frame(popup)
        frame.pack()

        # Create an instance of the DAAQ Setings within the frame
        self.menu_plotting_pane = PressureProtocolPane(frame, self.model.state)
        self.menu_plotting_pane.grid(sticky="nsew")

    def show_notepad(self):
        if self.model.notepad_path:
            popup = tk.Toplevel(root)
            popup.title("Notepad")

            # Set the window icon to be the same as the main window
            icon_path = os.path.join(images_folder, 'vt_icon.ICO')#Path(__file__).parent / "images" / "VasoTracker_Icon.ICO"
            popup.iconbitmap(icon_path)

            #popup.grab_set()  # Make the popup window modal

            # Create a scrolled text area that expands and fills the available space
            self.text_area = scrolledtext.ScrolledText(popup, wrap=tk.WORD)
            self.text_area.pack(padx=10, pady=10, expand=True, fill=tk.BOTH)

            # Check if a file already exists at the specified path
            notepad_path = self.model.notepad_path
            if os.path.isfile(notepad_path):
                # Read the content of the existing file and insert it into the text area
                with open(notepad_path, "r") as file:
                    content = file.read()
                self.text_area.insert(tk.END, content)
            else:
                # Prepopulate the text area with a header if the file doesn't exist
                header = "We love VasoTracker, you love VasoTracker. Show your love and cite the papers. Even the ones you didn't read.\n" \
                        "------------------------------------\n" \
                        f"- Notes for experiment: {os.path.basename(self.output_path)}\n" \
                        "------------------------------------\n\n"
                self.text_area.insert(tk.END, header)

            # Prepopulate the text area with a header
            header = f"\n{self.model.state.toolbar.data_acq.time_string.get()}: "
            self.text_area.insert(tk.END, header)

            # Set the cursor position to the end of the text
            self.text_area.insert(tk.END, "")
            # Scroll the widget to the bottom
            self.text_area.yview_moveto(1.0)
            self.text_area.focus_set()

            # Bind the text area to the auto_save function on any text change
            self.text_area.bind("<KeyRelease>", self.auto_save)

    def auto_save(self, event):
        """Automatically save the contents of the text area to a file."""
        with open(self.model.notepad_path, "w") as file:
            file.write(self.text_area.get("1.0", tk.END))







import tkinter as tk
from tkinter import font
import os
import sys
import webbrowser
from multiprocessing import freeze_support
from PIL import Image, ImageTk

def show_splash():
    """Show the initial splash screen while the app loads."""
    global rootsplash, splash_image_tk

    rootsplash = tk.Toplevel()
    rootsplash.overrideredirect(True)  # Remove title bar

    width, height = rootsplash.winfo_screenwidth(), rootsplash.winfo_screenheight()

    image_file = os.path.join(images_folder, 'Splash.gif')

    with Image.open(image_file) as image:
        new_width, new_height = int(width * 0.5), int(height * 0.5)
        image = image.resize((new_width, new_height), Image.LANCZOS)
        splash_image_tk = ImageTk.PhotoImage(image)

    x_offset, y_offset = (width - new_width) // 2, (height - new_height) // 2
    rootsplash.geometry(f"{new_width}x{new_height}+{x_offset}+{y_offset}")

    canvas = tk.Canvas(rootsplash, width=new_width, height=new_height, highlightthickness=0)
    canvas.pack()
    canvas.create_image(0, 0, anchor="nw", image=splash_image_tk)

    rootsplash.update()


def show_registration_screen(controller):
    """Show the VasoTracker registration splash screen and keep it on top."""
    global registration_screen

    registration_screen = VasoTrackerSplashScreen(root, controller.update_settings)  # Pass update_settings
    registration_screen.splash_win.lift()
    registration_screen.splash_win.attributes("-topmost", True)
    registration_screen.splash_win.focus_force()


    # Ensure main window remains hidden until registration is complete
    root.wait_window(registration_screen.splash_win)

    # Now that registration is done, show the main window
    root.deiconify()

def initialize_controller():
    """Initialize the main application and check for registration."""
    global app  
    app = Controller(root, mmc)  # Initialize the controller (WITHOUT launching VasoTrackerSplashScreen)

    rootsplash.destroy()  # Remove the loading splash screen

    '''
    # **Check if registration is required**
    if app.model.configure.registration.register_flag == 0:
        show_registration_screen(app)  # Show registration screen
    else:
        root.deiconify()  # If already registered, show the main window immediately
    '''

if __name__ == "__main__":
    freeze_support()

    # **Create Main Window but Keep it Hidden**
    root = tk.Tk()
    root.withdraw()  # Keep it hidden until registration is resolved
    root.iconbitmap(os.path.join(images_folder, 'vt_icon.ICO'))
    root.state("zoomed")

    ctk.set_default_color_theme(gui_json_path)

    # **Show Splash Screen**
    show_splash()

    # **Find MicroManager Path**
    mm_path = find_micromanager()
    print(mm_path)

    if mm_path is None:
        tmb.showinfo("Warning", "MicroManager not installed. Please download and install then relaunch VasoTracker.")
        webbrowser.open_new("https://download.micro-manager.org/nightly/2.0/Windows/")
        root.destroy()
        sys.exit()
    else:
        mmc = CMMCorePlus(adapter_paths=[mm_path, SYS32_PATH, BASLER_PATH, BASLER_PATH2])

    if not is_pydaqmx_available:
        tmb.showinfo("Warning", "niDAQmx not found. Please install to enable automatic pressure control.")

    # **Schedule Controller Initialization on the Main Thread (No Freezing)**
    root.after(2000, initialize_controller)  # Start loading the app after splash screen

    root.mainloop()  # Enter main event loop























'''






##################################################
## Splash screen
##################################################

rootsplash = tk.Tk()
rootsplash.overrideredirect(True)
width, height = rootsplash.winfo_screenwidth(), rootsplash.winfo_screenheight()

#Load in the splash screen image
image_file = os.path.join(images_folder, 'Splash.gif')

with Image.open(image_file) as image:
    image2 = ImageTk.PhotoImage(file=image_file)
    # Scale to half screen, centered
    imagewidth, imageheight = image2.width(), image2.height()
    newimagewidth, newimageheight = int(np.floor(width*0.5)),  int(np.floor(height*0.5))
    image = image.resize((newimagewidth,newimageheight), Image.LANCZOS)
    image = ImageTk.PhotoImage(image)

# Create and show for 3 seconds
rootsplash.geometry('%dx%d+%d+%d' % (newimagewidth, newimageheight, width/2 - newimagewidth/2, height/2 - newimageheight/2))
canvas = tk.Canvas(rootsplash, height=height, width=width, bg="darkgrey")
canvas.create_image(width/2 - newimagewidth/2, height/2 - newimageheight/2, image=image)
canvas.pack()
rootsplash.after(2000, rootsplash.destroy)
rootsplash.mainloop()


if __name__ == "__main__":
    freeze_support()

    root = ctk.CTk()
    root.iconbitmap(os.path.join(images_folder, 'vt_icon.ICO'))
    #root.withdraw()

    ctk.set_default_color_theme("C:\\Users\\Calum\\Documents\\GitHub\\VasoTracker-2_GUI_new\\vasotracker 2.0\\VasoTrackerblue.json")

    mm_path = find_micromanager()

    print(mm_path)

    if mm_path is None:
        tmb.showinfo("Warning", "MicroManager not installed. Please download and install then relaunch VasoTracker.")
        webbrowser.open_new(
            r"https://download.micro-manager.org/nightly/2.0/Windows/"
        )
        root.destroy()
        sys.exit()
    else:
        mmc = CMMCorePlus(adapter_paths=[mm_path, SYS32_PATH, BASLER_PATH, BASLER_PATH2])

    if not is_pydaqmx_available:
        tmb.showinfo("Warning", "niDAQmx not found. Please install to enable automatic pressure control.")

    # Get the text font used by text entry widgets and text boxes
    text_font = font.nametofont("TkTextFont")
    # Set the text font size
    text_font.configure(size=default_font_size)

    app = Controller(root, mmc)

    # Set threshold for a "low-resolution" screen
    LOW_RES_THRESHOLD = 1000  # Adjust as needed
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    if screen_width < LOW_RES_THRESHOLD:
        tmb.showwarning(
            "Low Resolution Warning",
            "Your screen resolution is low. Some toolbar elements may not be fully visible. "
            "Try increasing the resolution or resizing the window for a better experience."
            )
    
    def maximize_window():
        root.state("zoomed")
    root.geometry(f'{screen_width}x{screen_height}')  
    root.after(100, maximize_window)

    ###TODO: Need a way to wait until the pop-up information box is removed to load GUI. Otherwise it loads it the wrong size.

    root.mainloop()

'''
import os
import traceback
import numpy as np
import cv2

import skimage
from . import CameraBase
from pymmcore_plus import CMMCorePlus, find_micromanager
import tkinter.messagebox as tmb

import tifffile as tf
import tkinter as tk
from tkinter import filedialog
import sys
import threading


# The following is so that the required resources are included in the PyInstaller build.
# Utility functions
def get_resource_path(relative_path):
    """Get the path to a resource, whether it's bundled with PyInstaller or not.

    Frozen: next to the exe (sys._MEIPASS). From source: the .cfg files live in
    the vasotracker_2 folder, i.e. the parent of this cameras/ package - not the
    current working directory.
    """
    base_path = getattr(
        sys, "_MEIPASS",
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    return os.path.join(base_path, relative_path)



class Basler(CameraBase, camera_name="Basler"):
    device_label = "BaslerCamera"
    module_name = "BaslerPylon"
    device_name = "BaslerCamera"

    def __init__(self, mmc: CMMCorePlus, state, config):
        super().__init__(mmc, state, config)

        config_path = get_resource_path("Basler.cfg")
        self.mmc.loadSystemConfiguration(config_path)
        #self.mmc.setConfig("FrameRate", "4Hz")
        exposure = state.toolbar.acq.exposure.get()
        self.set_exposure(exposure)


class ThorlabsDcc(CameraBase, camera_name="DCC1545M"):
    device_label = "ThorCam"
    module_name = "ThorlabsUSBCamera"
    device_name = "ThorCam"

    def __init__(self, mmc: CMMCorePlus, state, config):
        super().__init__(mmc, state, config)

        self.load_device()
        self.set_property("HardwareGain", 1)
        pix_clock = state.toolbar.acq.pixel_clock.get()
        self.set_property('PixelClockMHz', pix_clock)
        self.set_property('PixelType', '8bit')
        exposure = state.toolbar.acq.exposure.get()
        self.set_exposure(exposure)

class ThorlabsCS165MU(CameraBase, camera_name="CS165MU"):
    device_label = "TSICam"
    module_name = "TSI"
    device_name = "TSICam"
    '''
    def __init__(self, mmc: CMMCorePlus, state, config):
        super().__init__(mmc, state, config)

        self.load_device()
        exposure = state.toolbar.acq.exposure.get()
        self.set_exposure(exposure)
    '''
    def __init__(self, mmc: CMMCorePlus, state, config):
        super().__init__(mmc, state, config)

        try:
            self.load_device()
            print(f"Device {self.device_label} loaded successfully.")
        except Exception as e:
            tmb.showinfo("Device Error", f"Failed to load the device {self.device_label}.")
            print(f"Error: Failed to load device {self.device_label}: {str(e)}")
            return  # Optionally return or handle the error further

        try:
            exposure = state.toolbar.acq.exposure.get()
            self.set_exposure(exposure)
            print("Exposure set successfully.")
        except Exception as e:
            tmb.showinfo("Exposure Error", "Failed to set exposure on the device.")
            print(f"Error: Failed to set exposure: {str(e)}")
            return  # Optionally return or handle the error further


    def get_image(self):
        # 10-bit sensor. Keep the original, hardware-verified 10->8 bit
        # conversion (divide by 4) and bypass CameraBase's bit-depth
        # normalisation: that relies on the adapter reporting its bit depth
        # correctly, which we cannot verify without this camera attached.
        # Output is byte-identical to previous releases.
        return (np.asarray(self.mmc.getLastImage()) / 4).astype(np.uint8)


class DemoCamera(CameraBase, camera_name="Demo"):
    """Micro-Manager's built-in synthetic camera - for testing the live
    acquisition path without any hardware attached."""
    device_label = "DCam"
    module_name = "DemoCamera"
    device_name = "DCam"

    pixel_type = "8bit"

    def __init__(self, mmc: CMMCorePlus, state, config):
        super().__init__(mmc, state, config)

        self.load_device()
        try:
            self.set_property("PixelType", self.pixel_type)
        except Exception:
            traceback.print_exc()
        exposure = state.toolbar.acq.exposure.get()
        self.set_exposure(exposure)


class DemoCamera16(DemoCamera, camera_name="Demo 16-bit"):
    """Demo camera in 16-bit mode - exercises the high bit-depth frame
    normalisation exactly like a real >8-bit camera would."""
    pixel_type = "16bit"


'''
class DmtTis(CameraBase, camera_name="DMT/TIS"):
    device_label = "TIS_DCAM"
    module_name = "TIScam"
    device_name = "TIS_DCAM"

    def __init__(self, mmc: CMMCorePlus, state, config):
        super().__init__(mmc, state, config)

        self.load_device()

        try:
            self.set_property("Property Gain_Auto", "Off")
            self.set_property("Exposure Auto", "Off")
        except:
            pass

        exposure = state.toolbar.acq.exposure.get()
        self.set_exposure(exposure)

        try:
            self.set_property("Property Gain", config.TIS_DCAM.property_gain)
        except:
            traceback.print_exc()
'''


'''
class OpenCvCamera(CameraBase, camera_name="OpenCV"):
    device_label = "OpenCVgrabber"
    module_name = "OpenCVgrabber"
    device_name = "OpenCVgrabber"

    def __init__(self, mmc: CMMCorePlus, state, config):
        super().__init__(mmc, state, config)

        self.mmc.loadSystemConfiguration("OpenCV.cfg")
        self.set_property("PixelType", "8bit")
        exposure = state.toolbar.acq.exposure.get()
        self.set_exposure(exposure)

    def set_resolution(self, width, height):
        self.mmc.setProperty('OpenCVgrabber', 'Resolution', f"{width}x{height}")
'''

'''
class JoyceCamera(CameraBase, camera_name="Joyce"):
    device_label = "OpenCVgrabber"
    module_name = "OpenCVgrabber"
    device_name = "OpenCVgrabber"

    def __init__(self, mmc: CMMCorePlus, state, config):
        super().__init__(mmc, state, config)

        self.mmc.loadSystemConfiguration("OpenCV.cfg")
        self.set_property("PixelType", "8bit")
        self.set_property("Resolution", "1280x720")
        exposure = state.toolbar.acq.exposure.get()
        self.set_exposure(exposure)

    def set_resolution(self, resolution):
        self.mmc.setProperty('OpenCVgrabber', 'Resolution', resolution)
'''

class MManagerCamera(CameraBase, camera_name="MMConfig"):

    def __init__(self, mmc: CMMCorePlus, state, config):
        super().__init__(mmc, state, config)

        config_loaded = False
        config_path = self._resolve_config_path(state)
        try:
            print(f"Current Working Directory: {os.getcwd()}")
            print(f"Looking for file here: {config_path}")
            self.mmc.loadSystemConfiguration(config_path)
            config_loaded = True
            print("Configuration loaded successfully.")
        except FileNotFoundError:
            tmb.showinfo("Configuration Error", f"MMConfig.cfg not found at {config_path}!")
            print(f"Error: Configuration file not found at {config_path}.")
        except Exception as e:
            tmb.showinfo("Configuration Error", "An error occurred while loading the configuration.")
            print(f"An unexpected error occurred: {str(e)}")
        finally:
            print("Configuration loading attempt completed.")


        if config_loaded:
            camera = self.mmc.getLoadedDevicesOfType(2)
            self.device_label = camera
            self.mmc.getDevicePropertyNames(camera[0])
            #self.set_property("PixelType", "8bit")
            exposure = state.toolbar.acq.exposure.get()
            self.set_exposure(exposure)

    @staticmethod
    def _resolve_config_path(state):
        """Pick the Micro-Manager .cfg to load, in order:
        1. the path chosen in the Acquisition Settings pane (settings.toml), if set
        2. MMConfig.cfg sitting next to the active Micro-Manager install
        3. the placeholder bundled with VasoTracker
        """
        try:
            chosen = state.toolbar.acq.mm_config_file.get().strip()
        except Exception:
            chosen = ""
        if chosen and os.path.isfile(chosen):
            print(f"MMConfig: using configured path: {chosen}")
            return chosen
        if chosen:
            print(f"MMConfig: configured path missing ({chosen}), falling back")

        try:
            mm_path = find_micromanager()
            if mm_path:
                candidate = os.path.join(mm_path, "MMConfig.cfg")
                if os.path.isfile(candidate):
                    print(f"MMConfig: using Micro-Manager install config: {candidate}")
                    return candidate
        except Exception:
            traceback.print_exc()

        bundled = get_resource_path("MMConfig.cfg")
        print(f"MMConfig: using bundled config: {bundled}")
        return bundled


    def get_image(self):
        image = super().get_image()
        if image.dtype == np.uint16:
            # Convert 16-bit to 8-bit by scaling down
            image = ((image / 65535) * 255).astype(np.uint8)
        return image


class OpenCVCamera(CameraBase, camera_name="OpenCV"):
    """Native OpenCV camera - works with any USB camera/webcam without Micro-Manager."""

    def __init__(self, mmc: CMMCorePlus, state, config):
        super().__init__(mmc, state, config)
        self.cap = None
        self.camera_index = 0  # Default to first camera
        self.width = 640
        self.height = 480
        self._last_frame = None
        self.running = False

    def start_acquisition(self):
        # Use DirectShow backend on Windows for faster camera initialization
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            tmb.showinfo(
                "Camera Error",
                f"Could not open camera {self.camera_index}.\n"
                "Is it in use by another program?",
            )
            # Raising lets the app revert to the unlocked, no-acquisition
            # state instead of locking the toolbar for a dead camera.
            raise RuntimeError(f"Could not open camera {self.camera_index}")
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # Get actual resolution (may differ from requested)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.running = True

    def stop_acquisition(self):
        self.running = False

    def shutdown(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def get_image(self):
        if self.cap is None or not self.cap.isOpened():
            return self._last_frame if self._last_frame is not None else np.zeros((self.height, self.width), dtype=np.uint8)
        ret, frame = self.cap.read()
        if ret:
            # Convert BGR to grayscale 8-bit
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            self._last_raw = gray.astype(np.uint8)
            # Route through the shared contrast window (identity by default)
            self._last_frame = self._normalise_image(self._last_raw)
            return self._last_frame
        return self._last_frame if self._last_frame is not None else np.zeros((self.height, self.width), dtype=np.uint8)

    def get_raw_frame(self):
        return getattr(self, "_last_raw", None)

    def image_ready(self):
        return self.cap is not None and self.cap.isOpened()

    def is_buffer_empty(self):
        return 1 if self.image_ready() else 0

    def get_camera_dims(self):
        return self.width, self.height

    def set_exposure(self, exposure):
        if self.cap is not None:
            # OpenCV exposure is camera-dependent, may not work on all cameras
            self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)

    def set_resolution(self, width, height):
        self.width = width
        self.height = height
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def load_device(self):
        pass  # No MM device to load

    def reset(self):
        self.shutdown()

    def set_fov(self, x, y, xSize, ySize):
        pass  # ROI not supported via OpenCV

    def set_pixel_clock(self, pix_clock):
        pass  # Not applicable


'''
class ProxyCamera(CameraBase, camera_name="SampleData"):

    def __init__(self, mmc: CMMCorePlus, state, config):
        super().__init__(mmc, state, config)

        self.frame_count = 0
        self.max_frame_count = self.config.proxy_camera.max_frame
        self.path_template = os.getcwd() + self.config.proxy_camera.path_template

    def get_image(self):
        print("We are trying to get the image here...")
        resolved_path = self.path_template.format(self.frame_count % self.max_frame_count)

        try:
            #print("resolved path: ", resolved_path)
            image = skimage.io.imread(resolved_path)
        except FileNotFoundError:
            image = np.zeros((1, 1))
        return image.astype(np.uint8)

    def next_position(self):
        self.frame_count += 1

    def image_ready(self):
        return True

    def start_acquisition(self):
        pass

    def stop_acquisition(self):
        pass

    def shutdown(self):
        pass

    def set_resolution(self, width, height):
        raise NotImplementedError("set_resolution is not implemented by ProxyCamera")

    def set_fov(self, x, y, xSize, ySize):
        raise NotImplementedError("set_fov is not implemented by ProxyCamera")

    def set_pixel_clock(self, pix_clock):
        raise NotImplementedError("set_pixel_clock is not implemented by ProxyCamera")

    def set_exposure(self, exposure):
        pass

    def get_camera_dims(self):
        im = self.get_image()
        height, width = im.shape
        return width, height
'''


class _TiffFrameSource:
    """Frame access for a multi-page TIFF (one page per frame)."""

    def __init__(self, path):
        self.path = path
        try:
            with tf.TiffFile(path) as t:
                self.n_frames = len(t.pages)
        except Exception:
            self.n_frames = 0

    def frame(self, idx):
        """Frame ``idx`` as a 2-D grayscale array at native bit depth, or None."""
        try:
            with tf.TiffFile(self.path) as t:
                if not 0 <= idx < len(t.pages):
                    return None
                image = t.pages[idx].asarray()
        except Exception:
            return None
        if image.ndim == 3:
            code = cv2.COLOR_RGBA2GRAY if image.shape[-1] == 4 else cv2.COLOR_RGB2GRAY
            image = cv2.cvtColor(image, code)
        return image

    def close(self):
        pass


class _VideoFrameSource:
    """Frame access for a video file (AVI/MP4/MOV/...) via OpenCV.

    One shared VideoCapture guarded by a lock, so reads from the tracking
    thread and the preview / contrast calls on the UI thread are serialised.
    Reads are sequential where possible (always frame-accurate); a short
    forward gap is closed by reading through it, and only a backward jump or
    a long forward jump issues a library seek.
    """

    # Close a forward gap of up to this many frames by decoding through it
    # rather than seeking - keeps scrubbing frame-accurate on long-GOP codecs.
    _MAX_SEQUENTIAL_SKIP = 48

    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()
        self._cap = cv2.VideoCapture(path)
        opened = self._cap is not None and self._cap.isOpened()
        n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
        self.n_frames = max(n, 0)
        self._next = 0  # frame index the capture will return next

    def frame(self, idx):
        with self._lock:
            if self._cap is None or not self._cap.isOpened():
                return None
            if idx < 0 or (self.n_frames and idx >= self.n_frames):
                return None
            if idx < self._next or idx > self._next + self._MAX_SEQUENTIAL_SKIP:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                self._next = idx
            frame = None
            while self._next <= idx:
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    self._next = -1
                    return None
                self._next += 1
        if frame is None:
            return None
        if frame.ndim == 3:
            code = cv2.COLOR_BGRA2GRAY if frame.shape[-1] == 4 else cv2.COLOR_BGR2GRAY
            frame = cv2.cvtColor(frame, code)
        return frame

    def close(self):
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None


# Extensions routed to the video reader; everything else is treated as a TIFF.
_VIDEO_EXTS = (".avi", ".mp4", ".mov", ".mkv", ".m4v", ".wmv", ".mpg", ".mpeg")


class SavedDataCamera(CameraBase, camera_name="Image from file"):
    def __init__(self, mmc: CMMCorePlus, state, config):
        super().__init__(mmc, state, config)

        self.frame_count = 0
        self.path_to_tiff = self.get_tiff_file_path()
        self._src = self._open_source(self.path_to_tiff)
        self.max_frame_count = self.get_num_frames()
        self.config.proxy_camera.max_frame = self.max_frame_count
        self.camera_stopped = False
        self.last_frame = None
        self.last_frame_idx = None
        self._compute_intensity_scaling()

        if self.path_to_tiff and self.max_frame_count == 0:
            kind = ("video" if isinstance(self._src, _VideoFrameSource)
                    else "multi-frame TIFF")
            tmb.showerror(
                "Could not open file",
                f"'{os.path.basename(self.path_to_tiff)}' could not be read as a "
                f"{kind} (no frames found).\n\nFor video, make sure it is a "
                "format OpenCV can decode (most AVI/MP4/MOV files are)."
            )

    @staticmethod
    def _open_source(path):
        """Pick a frame reader from the file extension (TIFF pages vs video)."""
        ext = os.path.splitext(path)[1].lower() if path else ""
        if ext in _VIDEO_EXTS:
            return _VideoFrameSource(path)
        return _TiffFrameSource(path)

    def reinitialize(self):
        self.frame_count = 0
        self.camera_stopped = False
        self.last_frame = None


    def _gray_frame(self, idx):
        """One frame as grayscale at native bit depth, or None if unreadable."""
        src = getattr(self, "_src", None)
        if src is None:
            return None
        return src.frame(int(idx))

    def _sample_gray_pages(self):
        """Grayscale frames sampled from the start, middle and end of the file."""
        n = self.get_num_frames()
        if n == 0:
            return None
        samples = []
        for idx in sorted({0, n // 2, n - 1}):
            f = self._gray_frame(idx)
            if f is not None:
                samples.append(f)
        return samples or None

    def _compute_intensity_scaling(self):
        """Work out the intensity window mapped to 0-255, once per file.

        8-bit files default to the identity window (0-255) so results are
        untouched unless the user adjusts contrast. Higher bit-depth cameras
        rarely use their full range (a 12-bit sensor tops out at 4095), so
        those default to the file's actual dynamic range (robust percentiles
        over sampled frames). Computed once so all frames share the same
        window and real intensity changes over time are preserved.
        """
        self._scale_lo = None
        self._scale_hi = None
        self._data_min = 0.0
        self._data_max = 255.0
        try:
            samples = self._sample_gray_pages()
        except (FileNotFoundError, tf.TiffFileError):
            return
        if not samples:
            return
        if all(s.dtype == np.uint8 for s in samples):
            self._scale_lo, self._scale_hi = 0.0, 255.0
            return
        stack = np.concatenate([s.ravel() for s in samples])
        self._data_min = float(stack.min())
        self._data_max = float(stack.max())
        lo = float(np.percentile(stack, 0.5))
        hi = float(np.percentile(stack, 99.7))
        if hi <= lo:
            lo, hi = self._data_min, self._data_max
        if hi <= lo:
            hi = lo + 1.0
        self._scale_lo = lo
        self._scale_hi = hi
        print(f"High bit-depth file: scaling intensities {lo:.0f}-{hi:.0f} to 0-255")

    def reset_contrast(self):
        """Back to the load-time default (identity for 8-bit files)."""
        self._compute_intensity_scaling()

    def get_raw_frame(self):
        """Currently shown frame, before the contrast window (histogram)."""
        try:
            frame = int(self.state.cam_show.slider_position_manual)
        except Exception:
            frame = 0
        n = self.get_num_frames()
        if n == 0:
            return None
        frame = min(max(frame, 0), n - 1)
        return self._gray_frame(frame)

    def auto_contrast(self):
        """Percentile-stretch the intensity window (any bit depth)."""
        try:
            samples = self._sample_gray_pages()
        except (FileNotFoundError, tf.TiffFileError):
            return
        if not samples:
            return
        stack = np.concatenate([s.ravel() for s in samples])
        lo = float(np.percentile(stack, 0.5))
        hi = float(np.percentile(stack, 99.7))
        if hi <= lo:
            lo, hi = float(stack.min()), float(stack.max())
        if hi <= lo:
            hi = lo + 1.0
        self._scale_lo = lo
        self._scale_hi = hi

    def _read_page(self, idx):
        """Read one frame as 8-bit grayscale through the intensity window,
        or None if the frame cannot be read."""
        image = self._gray_frame(idx)
        if image is None:
            return None
        lo = getattr(self, "_scale_lo", None)
        hi = getattr(self, "_scale_hi", None)

        if image.dtype != np.uint8:
            if lo is None or hi is None:
                lo, hi = float(image.min()), float(max(image.max(), image.min() + 1))
            image = np.clip((image.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0) * 255.0
        elif lo is not None and hi is not None and (lo != 0.0 or hi != 255.0):
            # 8-bit with a user-adjusted window; the identity default is
            # deliberately skipped so untouched files stay bit-identical.
            image = np.clip((image.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0) * 255.0

        return image.astype(np.uint8)

    def _process_frame(self, idx):
        """Read a frame with the user's file image processing applied:
        temporal averaging over the preceding frames and gaussian smoothing.
        Both default to off (1 frame / sigma 0). Returns None if the frame
        cannot be read."""
        try:
            analysis = self.state.toolbar.analysis
            n_avg = max(1, int(analysis.temporal_frames.get()))
            sigma = float(analysis.gauss_sigma.get())
        except Exception:
            n_avg, sigma = 1, 0.0

        if n_avg > 1:
            first = max(0, idx - n_avg + 1)
            frames = [self._read_page(k) for k in range(first, idx + 1)]
            frames = [f for f in frames if f is not None]
            if not frames:
                return None
            image = np.mean(frames, axis=0).astype(np.uint8)
        else:
            image = self._read_page(idx)
            if image is None:
                return None

        if sigma > 0:
            image = cv2.GaussianBlur(image, (0, 0), sigma)
        return image

    def _fallback_frame(self):
        if self.last_frame is not None:
            return self.last_frame
        return np.zeros((1, 1), dtype=np.uint8)

    def get_image(self):
        if self.camera_stopped:
            return self._fallback_frame()

        n = self.get_num_frames()
        if n and self.frame_count < n:
            image = self._process_frame(self.frame_count)
        else:
            image = None
            self.camera_stopped = True

        if image is None:
            return self._fallback_frame().astype(np.uint8)
        self.last_frame = image
        return image.astype(np.uint8)

    def get_specific_frame(self, frame):
        if self.camera_stopped:
            return self._fallback_frame()

        try:
            frame = int(frame)
        except (TypeError, ValueError):
            return np.zeros((1, 1), dtype=np.uint8)

        n = self.get_num_frames()
        if n and 0 <= frame < n:
            image = self._process_frame(frame)
        elif n and frame >= n:
            image = None
            self.camera_stopped = True
        else:
            image = self._process_frame(max(frame, 0))

        if image is None:
            return self._fallback_frame().astype(np.uint8)
        self.last_frame = image
        return image.astype(np.uint8)


    def get_num_frames(self):
        src = getattr(self, "_src", None)
        return int(src.n_frames) if src is not None else 0


    def get_tiff_file_path(self):
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        sample_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "SampleData")
        initial_dir = sample_data_dir if os.path.isdir(sample_data_dir) else os.getcwd()
        video_glob = " ".join("*" + e for e in _VIDEO_EXTS)
        file_path = filedialog.askopenfilename(
            title="Select a multi-frame TIFF or a video file",
            filetypes=[
                ("Image / video files", "*.tif *.tiff " + video_glob),
                ("Multi-frame TIFF", "*.tif *.tiff"),
                ("Video", video_glob),
                ("All files", "*.*"),
            ],
            initialdir=initial_dir,
        )
        return file_path

    def next_position(self, state):
        if state is True:
            self.frame_count += 1
        else:
            pass

    def image_ready(self):
        return True

    def start_acquisition(self):
        pass

    def stop_acquisition(self):
        pass

    def shutdown(self):
        src = getattr(self, "_src", None)
        if src is not None:
            try:
                src.close()
            except Exception:
                pass
            self._src = None

    def set_resolution(self, width, height):
        raise NotImplementedError("set_resolution is not implemented by ProxyCamera")

    def set_fov(self, x, y, xSize, ySize):
        raise NotImplementedError("set_fov is not implemented by ProxyCamera")

    def set_pixel_clock(self, pix_clock):
        raise NotImplementedError("set_pixel_clock is not implemented by ProxyCamera")

    def set_exposure(self, exposure):
        pass

    def get_camera_dims(self):
        im = self.get_image()
        height, width, = im.shape
        length = self.config.proxy_camera.max_frame
        print("Image shape: ", height, width, length)
        return width, height, length

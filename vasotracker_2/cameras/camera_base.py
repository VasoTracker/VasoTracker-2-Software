
import cv2
import numpy as np
from pymmcore_plus import CMMCorePlus


class CameraBase:

    _registry = {}

    def __init_subclass__(cls, camera_name: str) -> None:
        cls.camera_name = camera_name
        cls._registry[camera_name.lower()] = cls

    def __init__(self, mmc: CMMCorePlus, state, config):
        self.mmc = mmc
        self.state = state
        self.config = config
        self.running = False



    def set_exposure(self, exposure):
        print(f"Exposure type before conversion: {type(exposure)}")  

        # Convert numpy int32 to standard Python int
        if isinstance(exposure, np.integer):  
            exposure = int(exposure)

        print(f"Exposure type after conversion: {type(exposure)}")  

        self.mmc.setExposure(exposure)



    def set_pixel_clock(self, pix_clock):
        self.mmc.setProperty(self.device_label, 'PixelClockMHz', pix_clock)

    def set_resolution(self, width, height):
        raise NotImplementedError("set_resolution not implemented for current camera.")

    def set_fov(self, x, y, xSize, ySize):
        if self.running:
            self.mmc.stopSequenceAcquisition()

        error = None
        try:
            self.mmc.setROI(x, y, xSize, ySize)
            self.mmc.startContinuousSequenceAcquisition(0)
        except:
            self.mmc.startContinuousSequenceAcquisition(0)
            error = NotImplementedError("set_fov not implemented for current camera.")

        if not self.running:
            self.mmc.stopSequenceAcquisition()

        if error is not None:
            raise error

    def load_device(self):
        self.mmc.loadDevice(self.device_label, self.module_name, self.device_name)
        self.mmc.initializeDevice(self.device_label)
        self.mmc.setCameraDevice(self.device_label)

    def set_property(self, prop, value):
        self.mmc.setProperty(self.device_label, prop, value)

    def reset(self):
        self.mmc.reset()

        self.mmc.setCircularBufferMemoryFootprint(12800)

    def image_ready(self):
        return self.mmc.getRemainingImageCount() > 0 or self.mmc.isSequenceRunning()

    def get_image(self):
        return self._normalise_image(self.mmc.getLastImage())

    def contrast_range(self):
        """(min, max) bounds for the adjustable intensity window."""
        data_min = getattr(self, "_data_min", None)
        if data_min is not None:
            return float(data_min), float(self._data_max)
        try:
            bits = int(self.mmc.getImageBitDepth())
        except Exception:
            bits = 8
        max_val = float(2 ** bits - 1) if 0 < bits <= 16 else 65535.0
        return 0.0, max_val

    def auto_contrast(self):
        """Percentile-stretch the intensity window from the latest frame."""
        try:
            image = np.asarray(self.mmc.getLastImage())
        except Exception:
            return
        if image.ndim == 3:
            code = cv2.COLOR_BGRA2GRAY if image.shape[-1] == 4 else cv2.COLOR_BGR2GRAY
            image = cv2.cvtColor(image, code)
        lo = float(np.percentile(image, 0.5))
        hi = float(np.percentile(image, 99.7))
        if hi <= lo:
            lo, hi = float(image.min()), float(image.max())
        if hi <= lo:
            hi = lo + 1.0
        self._scale_lo = lo
        self._scale_hi = hi

    def reset_contrast(self):
        """Back to the default mapping (full sensor bit depth)."""
        self._scale_lo = None
        self._scale_hi = None

    def get_raw_frame(self):
        """Latest frame as grayscale BEFORE the contrast window is applied.
        Used by the contrast dialog's histogram."""
        try:
            image = np.asarray(self.mmc.getLastImage())
        except Exception:
            return None
        if image.ndim == 3:
            code = cv2.COLOR_BGRA2GRAY if image.shape[-1] == 4 else cv2.COLOR_BGR2GRAY
            image = cv2.cvtColor(image, code)
        return image

    def _normalise_image(self, image):
        """The analysis pipeline expects 8-bit grayscale. Cameras configured
        without an 8-bit preset (e.g. OpenCVGrabber in RGB mode) deliver
        colour frames, and high bit-depth cameras deliver uint16 - both used
        to crash the frame processing on every frame, which looks like the
        app simply not responding to play. A user-adjusted contrast window
        (see the Contrast dialog) overrides the default full-range mapping."""
        image = np.asarray(image)
        if image.ndim == 3:
            code = cv2.COLOR_BGRA2GRAY if image.shape[-1] == 4 else cv2.COLOR_BGR2GRAY
            image = cv2.cvtColor(image, code)

        lo = getattr(self, "_scale_lo", None)
        hi = getattr(self, "_scale_hi", None)
        if image.dtype != np.uint8:
            if lo is None or hi is None:
                # No window yet: measure it from the first usable frame. A
                # 16-bit camera imaging only ~1000 grey levels would show
                # ~4 levels if scaled by the theoretical sensor maximum.
                # Once set, the window stays fixed (stable analysis
                # intensities) until changed via the Contrast dialog.
                p_lo = float(np.percentile(image, 0.5))
                p_hi = float(np.percentile(image, 99.7))
                if p_hi - p_lo > 2.0:
                    self._scale_lo = lo = p_lo
                    self._scale_hi = hi = p_hi
                    print(f"Live camera: auto intensity window {lo:.0f}-{hi:.0f} -> 0-255")
                else:
                    # Flat frame (shutter closed?): show it via the sensor
                    # range and try again on the next frame.
                    lo, hi = self.contrast_range()
            image = np.clip((image.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0) * 255.0
            image = image.astype(np.uint8)
        elif lo is not None and hi is not None and (lo != 0.0 or hi != 255.0):
            # 8-bit camera with a user-adjusted window; the identity default
            # is skipped so untouched cameras stay bit-identical.
            image = np.clip((image.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0) * 255.0
            image = image.astype(np.uint8)
        return image

    def next_position(self):
        """Override this for custom per-frame behaviour, like FakeCamera needs"""
        pass

    def start_acquisition(self):
        self.mmc.startContinuousSequenceAcquisition(0)
        self.running = True

    def stop_acquisition(self):
        self.running = False
        self.mmc.stopSequenceAcquisition()

    def shutdown(self):
        try:
            self.mmc.stopSequenceAcquisition()
            self.mmc.close()
        except:
            self.mmc.reset()

    def get_camera_dims(self):
        x_dim = self.mmc.getImageWidth()
        y_dim = self.mmc.getImageHeight()
        return x_dim, y_dim

    def is_buffer_empty(self):
        try:
            # Assuming `getBufferCount` is a method that returns the number of images in the buffer
            count = self.mmc.getRemainingImageCount()
            return count
        except AttributeError:
            # If `getBufferCount` method does not exist, you might need to handle it differently
            print("The method `getBufferCount` is not available.")
            return False  # Or handle the error as appropriate for your application


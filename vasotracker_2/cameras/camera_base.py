
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
        return self.mmc.getLastImage()

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


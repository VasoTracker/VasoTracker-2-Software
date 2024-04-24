import os
import traceback
import numpy as np

import skimage
from . import CameraBase
from pymmcore_plus import CMMCorePlus, find_micromanager
import tkinter.messagebox as tmb

import tifffile as tf
import tkinter as tk
from tkinter import filedialog
import sys


# The following is so that the required resources are included in the PyInstaller build.
# Utility functions
def get_resource_path(relative_path):
    """Get the path to a resource, whether it's bundled with PyInstaller or not."""
    base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
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

    def __init__(self, mmc: CMMCorePlus, state, config):
        super().__init__(mmc, state, config)

        self.load_device()
        exposure = state.toolbar.acq.exposure.get()
        self.set_exposure(exposure)

    def get_image(self):
        high_bit_depth = super().get_image()
        return (high_bit_depth / 4).astype(np.uint8)

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
        config_path = get_resource_path("MMConfig.cfg")
        try:
            print("CWD: ", os.getcwd())
            self.mmc.loadSystemConfiguration(config_path)
            config_loaded = True
            print("bahahaha1")
        except:
            tmb.showinfo("Warning", "MMConfig.cfg not found in home directory!")
        print("bahahaha")

        
        if config_loaded:
            camera = self.mmc.getLoadedDevicesOfType(2)
            self.device_label = camera
            self.mmc.getDevicePropertyNames(camera[0])
            #self.set_property("PixelType", "8bit")
            exposure = state.toolbar.acq.exposure.get()
            self.set_exposure(exposure)

        

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
        

class SavedDataCamera(CameraBase, camera_name="Image from file"):
    def __init__(self, mmc: CMMCorePlus, state, config):
        super().__init__(mmc, state, config)

        self.frame_count = 0
        self.path_to_tiff = self.get_tiff_file_path()
        self.max_frame_count = self.get_num_frames()
        self.config.proxy_camera.max_frame = self.max_frame_count
        self.camera_stopped = False
        self.last_frame = None

    def reinitialize(self):
        self.frame_count = 0
        self.camera_stopped = False
        self.last_frame = None


    def get_image(self):
        if self.camera_stopped:
            if self.last_frame is not None:
                return self.last_frame
            else:
                return np.zeros((1, 1)) 
        
        try:
            with tf.TiffFile(self.path_to_tiff) as tif:
                if self.frame_count < len(tif.pages):
                    image = tif.pages[self.frame_count].asarray()
                else:
                    image = self.last_frame  # Return the last frame
                    self.camera_stopped = True
        except (FileNotFoundError, tf.TiffFileError):
            image = np.zeros((1, 1))

        #self.frame_count = (self.frame_count + 1) % self.max_frame_count
        return image.astype(np.uint8)

    def get_specific_frame(self, frame):
        if self.camera_stopped:
            if self.last_frame is not None:
                return self.last_frame
            else:
                return np.zeros((1, 1)) 
            
        if not isinstance(frame, int):
            return np.zeros((1, 1))  # Return a default blank image


        try:
            with tf.TiffFile(self.path_to_tiff) as tif:
                if self.frame_count < len(tif.pages):
                    image = tif.pages[frame].asarray()
                else:
                    image = self.last_frame  # Return the last frame
                    self.camera_stopped = True
        except (FileNotFoundError, tf.TiffFileError):
            image = np.zeros((1, 1))

        #self.frame_count = (self.frame_count + 1) % self.max_frame_count
        return image.astype(np.uint8)    


    def get_num_frames(self):
        try:
            with tf.TiffFile(self.path_to_tiff) as tif:
                return len(tif.pages)
        except (FileNotFoundError, tf.TiffFileError):
            return 0


    def get_tiff_file_path(self):
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        file_path = filedialog.askopenfilename(title="Select Multi-frame TIFF File", filetypes=[("TIFF files", "*.tiff *.tif")])
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
        height, width, = im.shape
        length = self.config.proxy_camera.max_frame
        print("Image shape: ", height, width, length)
        return width, height, length
    
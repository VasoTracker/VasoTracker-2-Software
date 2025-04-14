##################################################
## VasoTracker 2.0 - Part of the Pressure Myograph Software
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

pydaqmx_available = True


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
    
    def end_protocol(self):
        try:
            self.view.toolbar.pressure_control_settings.toggle_protocol_button()
        except Exception as e:
            print(f"Error in end_protocol: {e}")



    def initialize_pressure_system(self):
        if self.pydaqmx_available:
            self.task = PyDAQmx.Task()
            self.set_dev()

        servo_settings = self.model.state.toolbar.servo
        device = servo_settings.device.get()
        ao_channel = servo_settings.ao_channel.get()

        print(f"The device is {device}, and the aochannel is {ao_channel}")

    def on_option_changed(self, *args):
        servo_settings = self.model.state.toolbar.servo
        device = servo_settings.device.get()
        ao_channel = servo_settings.ao_channel.get()
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
            self.update_pressure()
            self.next_pressure_update_time += self.pressure_time_interval# * self.multiplier

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

        time.sleep(2)
        servo_settings = self.model.state.toolbar.servo
        device = servo_settings.device.get()
        ao_channel = servo_settings.ao_channel.get()



        # Clear any existing task to avoid conflicts
        if self.task is not None:
            self.task.ClearTask()

        try:
            self.task = PyDAQmx.Task()
            self.task.CreateAOVoltageChan(f"/{device}/{ao_channel}", "", -10.0, 10.0, PyDAQmx.DAQmx_Val_Volts, None)
            self.task.StartTask()
            # Assuming 'set_pressure_entry' is part of the view
            self.view.toolbar.pressure_protocol_settings.set_unlock_state()  # Enable the entry
            self.view.toolbar.pressure_protocol_settings.set_unlock_state() 
            return True  # Device successfully set
        except Exception as e:
            print("Failed to connect to NI device:", e)
            self.view.toolbar.pressure_protocol_settings.set_lock_state()  # Disable the entry
            self.view.toolbar.pressure_protocol_settings.set_lock_state() 
            # Temporarily remove the trace callback if necessary
            try:
                servo_settings.device.trace_remove(...)
            except:
                pass
            try:
                servo_settings.ao_channel.trace_remove(...)
            except:
                pass
            tmb.showinfo("Warning", "Cannot connect to NI device:\n - Ensure the device is connected via USB.\n - Check the device name in the NI Device Monitor Software.")

            servo_settings.device.set("")
            servo_settings.ao_channel.set("")
            servo_settings.device.trace_add("write", ...)
            servo_settings.ao_channel.trace_add("write", ...)
            return False  # Failed to set the device
            
        

    def adjust_pressure(self, pressure_value, update_table=True):
        if not self.pydaqmx_available:
            return

        # Validate and adjust pressure value to be within the acceptable range
        pressure_value = max(min(200, pressure_value), 0)

        pressure_protocol_settings = self.model.state.toolbar.pressure_protocol
        pressure_protocol_settings.set_pressure.set(pressure_value)

        # Update the pressure using PyDAQmx
        try:
            # This line writes the analog value to the DAQ device to set the pressure
            # The pressure_value is divided by 100, assuming it's being scaled to the DAQ device's range
            self.task.WriteAnalogScalarF64(1, 10.0, pressure_value / 100, None)
        except Exception as e:
            # Handle exceptions, possibly log or show an error message
            print("Exception occurred while setting pressure:", e)

        # Optionally update the table
        # If update_table is True, this will update the UI to reflect the new pressure
        if update_table:
            # Assuming this method updates a UI element to show the current pressure
            self.model.state.table.label.set(f"Set pressure = {pressure_value} mmHg")
            self.model.add_table_row()

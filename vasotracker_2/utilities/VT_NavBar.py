##################################################
## VasoTracker 2 - Blood Vessel Diameter Measurement Software
##
## Author: Calum Wilson, Matthew D Lee, and Chris Osborne
## License: BSD 3-Clause License (See main file for details)
## Website: www.vasostracker.com
##
##################################################


import os, sys
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from PIL import Image, ImageTk

def get_resource_path(relative_path):
    """Get the path to a resource, whether it's bundled with PyInstaller or not."""
    base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
    return os.path.join(base_path, relative_path)

# Resource paths
images_folder = get_resource_path("images\\")
sample_data_path = get_resource_path("SampleData\\")

class CustomVTToolbar(NavigationToolbar2Tk):
    def __init__(self, canvas, parent, graph_frame, *args, **kwargs):
        super().__init__(canvas, parent, *args, **kwargs)
        self.graph_frame = graph_frame  # Store the reference to GraphFrame

        self.home_img = self.resize_img(os.path.join(images_folder, 'Home Symbol Button.png'))

        self.add_img = self.resize_img(os.path.join(images_folder, 'Add Button.png'))
        self.remove_img = self.resize_img(os.path.join(images_folder, 'Remove Button.png'))

        # Remove unwanted default buttons
        self.remove_button_by_command('pan')
        self.remove_button_by_command('configure_subplots')
        self.remove_button_by_command('home')

        # Add a separator
        self.add_separator()

        # Define predefined time ranges (in seconds)
        self.time_ranges = [-60, -120, -300, -600, -1800, -3600, -7200]

        # Add custom image buttons for adjusting the x-axis time range
        self.add_label("X-Axis Zoom:")
        self.add_image_button(self.add_img, self.next_time_range)
        self.add_image_button(self.remove_img, self.prev_time_range)

        self.add_separator()

        # Add custom buttons for setting y-axis limits
        self.add_label("Y-Axis Zoom:")
        self.add_image_button(self.add_img, lambda: self.adjust_y_axis(-100))
        self.add_image_button(self.remove_img, lambda: self.adjust_y_axis(+100))

        self.add_separator()

        # Add a new custom button
        self.add_image_button(self.home_img, self.custom_home_function)

    def add_custom_button(self, text, command):
        """Add a custom button with text label to the toolbar."""
        button = tk.Button(self, text=text, command=command)
        button.pack(side=tk.LEFT, padx=2)

    def custom_home_function(self):
        # Your custom 'Home' button functionality here
        print("Custom Home Function Called")
        self.graph_frame.state_vars.toolbar.graph.limits_dirty.set(True)
        self.graph_frame.update_lims_fromfile_callback()

    def add_label(self, text):
        """Add a label widget to the toolbar."""
        label = ttk.Label(self, text=text)
        label.pack(side=tk.LEFT, padx=(5, 5))

    def remove_button_by_command(self, command_name):
        """Remove a toolbar button based on its command function name."""
        for text, button in self._buttons.items():
            if command_name in str(button.config('command')):
                button.pack_forget()
                break

    def add_separator(self):
        """Add a visual separator to the toolbar."""
        separator = ttk.Label(self, text="|", font=("Arial", 12))
        separator.pack(side=tk.LEFT, padx=(5, 5))


    def find_closest_time_range(self, current_start, direction):
        """Find the closest predefined time range to the current start time."""
        closest_range = None
        min_difference = float('inf')  # Initialize with positive infinity

        for time_range in self.time_ranges:
            if direction == 'next' and time_range > current_start:
                difference = time_range - current_start
            elif direction == 'prev' and time_range < current_start:
                difference = current_start - time_range
            else:
                continue  # Skip time ranges in the opposite direction

            if difference < min_difference:
                min_difference = difference
                closest_range = time_range

        return closest_range


    def next_time_range(self):
        """Go to the next predefined time range and update the plot."""
        current_start = self.graph_frame.state_vars.toolbar.graph.x_min.get()
        closest_range = self.find_closest_time_range(current_start, direction='next')

        if closest_range is not None:
            new_start = closest_range
            new_end = 0  # Set upper limit to 0

            self.graph_frame.state_vars.toolbar.graph.x_min.set(new_start)
            self.graph_frame.state_vars.toolbar.graph.x_max.set(new_end)
            self.graph_frame.update_lims()  # Update the plot with the new settings

    def prev_time_range(self):
        """Go to the previous predefined time range and update the plot."""
        current_start = self.graph_frame.state_vars.toolbar.graph.x_min.get()
        closest_range = self.find_closest_time_range(current_start, direction='prev')

        if closest_range is not None:
            new_start = closest_range
            new_end = 0  # Set upper limit to 0

            self.graph_frame.state_vars.toolbar.graph.x_min.set(new_start)
            self.graph_frame.state_vars.toolbar.graph.x_max.set(new_end)
            self.graph_frame.update_lims()  # Update the plot with the new settings






    def adjust_y_axis(self, increment):
        """Adjust the upper limit of the y-axis by the specified increment and update the plot."""
        settings = self.graph_frame.state_vars.toolbar.graph
        current_value = settings.y_max_od.get()

        # If the current value is 50 and we're adding, only add 50
        if increment > 0 and current_value == 50:
            new_value = current_value + 50
        else:
            # Calculate the new value after adding or subtracting the increment
            new_value = max(50, current_value + increment)  # Ensure minimum value is 50

        # Ensure the upper limit doesn't drop below 0
        new_value = max(0, new_value)

        settings.y_max_od.set(new_value)
        settings.y_min_od.set(0)

        settings.y_max_id.set(new_value)
        settings.y_min_id.set(0)


        self.graph_frame.update_lims()  # Update the plot with the new settings

    def add_image_button(self, img, command):
        """Add a custom button to the toolbar."""
        style = ttk.Style()
        style.configure("ImageButton.TButton", padding=0, background="white")  # Customize button appearance

        button = ttk.Button(self, image=img, command=command, style="ImageButton.TButton")
        button.pack(side=tk.LEFT, padx=2)

    def resize_img(self, img_path):
        img = Image.open(img_path)
        resized_image = img.resize((25, 25), Image.LANCZOS)
        tk_image = ImageTk.PhotoImage(resized_image)
        return tk_image


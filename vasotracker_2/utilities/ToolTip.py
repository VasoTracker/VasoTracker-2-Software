##################################################
## VasoTracker 2 - Blood Vessel Diameter Measurement Software
##
## Author: Calum Wilson, Matthew D Lee, and Chris Osborne
## License: BSD 3-Clause License (See main file for details)
## Website: www.vasostracker.com
##
##################################################


from tkinter import *

class ToolTip:
    """Class to create and manage tooltips for tkinter widgets."""

    def __init__(self, container):
        """
        Initialize the tooltip.
        :param container: The parent container (e.g., frame) where the tooltip should be displayed.
        """
        self.container = container
        self.tipwindow = None
        self.text = ""

    def showtip(self, text):
        """Display text in a tooltip window at the bottom of the container."""
        self.text = text
        if self.tipwindow or not self.text:
            return

        # Calculate position for the tooltip at the bottom of the container
        x = self.container.winfo_rootx()
        y = self.container.winfo_rooty() + self.container.winfo_height() + 5  # Slight offset below the container

        # Create the tooltip window
        self.tipwindow = Toplevel(self.container)
        self.tipwindow.wm_overrideredirect(1)  # Remove window decorations
        self.tipwindow.wm_geometry(f"+{x}+{y}")
        self.tipwindow.wm_attributes("-topmost", 1)  # Ensure it's always on top

        # Create and configure the tooltip label
        label = Label(
            self.tipwindow,
            text=self.text,
            justify=LEFT,
            background="#ffffe0",  # Light yellow background
            relief=SOLID,
            borderwidth=1,
            font=("tahoma", "8", "normal"),
        )
        label.pack(ipadx=1, ipady=1)

    def hidetip(self):
        """Hide the tooltip."""
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None

    def register(self, widget, text):
        """Bind tooltip to a widget."""
        def enter(event):
            self.showtip(text)

        def leave(event):
            self.hidetip()

        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)

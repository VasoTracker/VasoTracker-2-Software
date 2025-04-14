##################################################
## VasoTracker 2.0 - Part of the Pressure Myograph Software
##
## Author: Calum Wilson, Matthew D Lee, and Chris Osborne
## License: BSD 3-Clause License (See main file for details)
## Website: www.vasostracker.com
##
##################################################


# Tkinter imports
import tkinter as tk
from tkinter import *
import tkinter.simpledialog as tkSimpleDialog
import tkinter.messagebox as tmb
import tkinter.filedialog as tkFileDialog
from tkinter import ttk 
from tkinter.font import Font
import customtkinter as ctk
import webbrowser
from PIL import Image, ImageTk #convert cv2 image to tkinter
E = tk.E
W = tk.W
N = tk.N
S = tk.S
ypadding = 1.5 #ypadding just to save time - used for both x and y

import sys
import os
import re

import json

# The following is so that the required resources are included in the PyInstaller build.
# Utility functions
def get_resource_path(relative_path):
    """Get the path to a resource, whether it's bundled with PyInstaller or not."""
    base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
    return os.path.join(base_path, relative_path)

# Resource paths
images_folder = get_resource_path("images\\")
sample_data_path = get_resource_path("SampleData\\")


##################################################
## Base Application using init method to launch splash screen and main GUI application 
##################################################
class VasoTrackerSplashScreen(ctk.CTkFrame):

    #Initialisation function
    def __init__(self, master, update_settings_callback, *args, **kwargs):
        self.master = master #this is root
        self.update_settings_callback = update_settings_callback
        # Registration status
        self.registered = False


        # Set up the main splash screen
        self.mainWidgets()


    def mainWidgets(self):    
    # Set up a new top level window for the splash screen
        self.splash_win= Toplevel(self.master)
        self.splash_win.title("Let us know you use VasoTracker")
        self.splash_win.iconbitmap(os.path.join(images_folder, 'vt_icon.ICO'))#('images\VasoTracker_Icon.ICO')
        #self.splash_win.geometry("700x200")
        self.splash_win.config(bg='#0B5A81')
    # make the top right close button minimize (iconify) the main window
        self.splash_win.protocol("WM_DELETE_WINDOW", self.disable_event)
        #self.splash_win.overrideredirect(True)
        self.splash_win.protocol("WM_DELETE_WINDOW", lambda: None)
        self.splash_win.resizable(False, False)  # Disable Resizing (Maximize/Minimize)
        self.splash_win.overrideredirect(True)  # Remove Title Bar (No Controls)

            # Block interaction with anything underneath
        self.splash_win.grab_set()  # Makes this window modal (blocks input to other windows)



    # Load in the splash screen image
        self.image_file = os.path.join(images_folder, 'Small_Splash.gif')#"images\Small_Splash.gif" 
        self.image = Image.open(self.image_file)
        self.image2 = PhotoImage(file=self.image_file)
        self.imagewidth, self.imageheight = self.image2.width(), self.image2.height()
        copy_image = self.image2

    # Add the various frames
        top_frame = Frame(self.splash_win, bd=0, width = self.imagewidth, height = self.imageheight, bg='red', relief=SOLID, padx=0, pady=0)
        #top_frame.grid_propagate(1)
        top_middle_Frame = Frame(self.splash_win, bd=0,width = self.imagewidth, bg='#CCCCCC',   relief=SOLID, padx=0, pady=0)
        top_middle_Frame.grid_propagate(1)
        middle_frame = Frame(self.splash_win, bd=0,width = self.imagewidth, bg='#CCCCCC',   relief=SOLID, padx=0, pady=0)
        middle_frame.grid_propagate(1)
        bottom_frame = Frame(self.splash_win, bd=2, width = self.imagewidth, bg='#CCCCCC',relief=SOLID, padx=0, pady=0)



    # Add the splash screen image to the top frame
    # We are using a tkinter canvas


        canvas = tk.Canvas(top_frame, width=self.imagewidth, height=self.imageheight)
        canvas.grid(row=0, column=0, sticky=NSEW, pady=0, padx=0)
        canvas.create_image(self.imagewidth/2, self.imageheight/2, anchor=CENTER, image=self.image2)
        canvas.grid()

    # Add a title to the upper middle frame
        label = Label(top_middle_Frame, text = "Help Keep VasoTracker Alive", background="gray")
        label.config(font =("Arial", 14, "  bold"))
        label.grid(sticky="we")

    # Add text to the lower middle frame
        string1 = "\n" + \
                "VasoTracker was initially funded by grants from Wellcome Trust and the British Heart Foundation." + "\n" + "\n" + \
                "As of October 2021, VasoTracker is not funded and we are exploring new support options. " + \
                "So please help us and let us know that you are using this software. " + "\n" + "\n" + \
                "It's a very short form - So if you do not let us know and we find out that you use VasoTracker, we will secretly call you names."  + "\n" + "\n" + \
                "Also, please cite the VasoTracker 2.0 paper."+ "\n"

        label2 = Label(middle_frame, text = string1, wraplength=self.imagewidth, anchor="e", justify=LEFT)
        label2.config(font =("Arial", 10))
        label2.grid(sticky="nswe")

        # Font for the buttons
        button_font = Font(family="Arial", size=11, weight="bold")

        # Create buttons with auto-adjusted width
        register_text = 'Click Here to Support VasoTracker'
        go_away_text = 'Click Here to Make This Box Go Away'
        
        register_btn_width = self.get_button_width(register_text, button_font)
        go_away_btn_width = self.get_button_width(go_away_text, button_font)

        register_btn = Button(bottom_frame, text=register_text, font=button_font, relief=SOLID, cursor='hand2', command=self.insert_record, width=register_btn_width)
        go_away_btn = Button(bottom_frame, text=go_away_text, font=button_font, relief=SOLID, cursor='hand2', command=self.OnClose, width=go_away_btn_width)

        # Place the buttons in the bottom frame
        register_btn.grid(row=0, column=1, pady=5, padx=0, sticky="EW")
        go_away_btn.grid(row=1, column=1, pady=5, padx=0, sticky="EW")


    # Sort the layout of the frames.
        top_frame.grid(row=0, column=0, pady=0, padx=0, sticky=E+W)
        top_middle_Frame.grid(row=1, column=0, pady=0, padx=0,sticky=E+W)
        middle_frame.grid(row=2, column=0, pady=0, padx=0, sticky=N+S+E+W)
        bottom_frame.grid(row=3, column=0, pady=0, padx=0, sticky=E+W)

        top_frame.columnconfigure(0, weight=1)
        top_middle_Frame.columnconfigure(0, weight=1)
        middle_frame.columnconfigure(0, weight=1)
        bottom_frame.columnconfigure(0, weight=1)  # Left spacer column
        bottom_frame.columnconfigure(1, weight=2)  # Middle column where buttons will be placed
        bottom_frame.columnconfigure(2, weight=1)  # Right spacer column

        top_frame.rowconfigure(0, weight=1)
        top_middle_Frame.rowconfigure(0, weight=1)
        middle_frame.rowconfigure(0, weight=1)
        bottom_frame.rowconfigure(0, weight=1)
        bottom_frame.rowconfigure(0, weight=1)

        self.center(self.splash_win)
        self.splash_win.update()
        
    def OnClose(self):
        self.update_settings_callback('neveragain_flag', 1)
        self.splash_win.destroy()
  
    def disable_event(self):
        tmb.showerror("Warning", "To make the box go away permanently, just send us some info about your use of VasoTracker!")
        self.splash_win.destroy()

    def get_button_width(self, text, font):
        text_width = font.measure(text)
        # Conversion factor (pixels to text units). Adjust as needed for accuracy
        conversion_factor = 7.5
        return int(text_width // conversion_factor)

    # Function for inserting the records
    def insert_record(self):
        self.update_settings_callback('register_flag', 1)
        webbrowser.open_new(
            r"https://forms.office.com/e/Ke9mjE6CQg"
        )
        
            
    def center(self, win):
        # centers a tkinter window
        win.update_idletasks()
        width = win.winfo_width()
        frm_width = win.winfo_rootx() - win.winfo_x()
        win_width = width + 2 * frm_width
        height = win.winfo_height()
        titlebar_height = win.winfo_rooty() - win.winfo_y()
        win_height = height + titlebar_height + frm_width
        x = win.winfo_screenwidth() // 2 - win_width // 2
        y = win.winfo_screenheight() // 2 - win_height // 2
        win.geometry('{}x{}+{}+{}'.format(width, height, x, y))
        win.deiconify()
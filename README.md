# VasoTracker 2.3 - Blood Vessel Diameter Tracking Software (online and offline analysis)

The VasoTracker 2.3 software is a comprehensive software solution designed for the acquisition and analysis of blood vessel imaging data. It supports both live and pre-recorded video analysis, making it adaptable for various experimental set ups. It was initially developed for pressure myography, but it works for many other types of imaging!

![til](https://github.com/VasoTracker/VasoTracker-2-Software/blob/main/VasoTracker%20GUI.gif)



## Table of Contents
- [What's New in v2.3](#whats-new-in-v23)
- [What's New in v2.2](#whats-new-in-v22)
- [Key Features](#key-features)
- [Installation Instructions](#vasotracker-installation-instructions)
  - [Executable File](#option-1-installing-and-running-from-the-executable-file)
  - [From Source](#option-2-installing-from-source-using-anaconda)
- [License](#license)
- [Issues](#issues)

---

## What's New in v2.3 (August 2026)

* **Open any TIFF:** RGB and high bit-depth (10/12/16-bit) recordings now load and analyse directly. Intensities are automatically windowed to the data actually present, so 16-bit files no longer appear black - no manual 8-bit conversion needed.
* **Smart file loading:** On loading a file, VasoTracker auto-detects the vessel orientation (90&deg; mode), fluorescence vs transmitted light, and a suitable smoothing factor. Every decision is shown in the console and adjustable as usual.
* **More robust diameter tracking:** Edge-corrected smoothing removes boundary artefacts that biased measurements on small images; scanlines now cross-check each other (with support for slanted vessels), and one-frame flow artefacts (bubbles, debris) are automatically re-detected against the recent trace instead of spiking the record.
* **Brightness & Contrast tool:** A Fiji-style contrast dialog (Settings > Contrast) with a live histogram, black/white level sliders, and a data-autoscaled range - plus a scaling preview when connecting a >8-bit camera, so the raw-to-8-bit conversion is a visible choice.
* **Image processing for files:** Optional Gaussian smoothing, temporal frame averaging, and display colormaps for prerecorded recordings (Settings > Image Processing).
* **Demo cameras:** Micro-Manager's synthetic camera ("Demo", "Demo 16-bit") is available in the camera dropdown for testing a full acquisition setup without hardware.
* **Reliability:** A failed camera connection (e.g. webcam in use elsewhere) no longer locks the toolbar; re-running a file analysis fully clears the previous results; the toolbar scrolls on small screens instead of cutting off panes; packaged builds bundle all required system libraries (fixes startup crashes on machines without Anaconda).

---

## What's New in v2.2 (January 2026)

* **Native OpenCV Camera Support:** Use any USB camera or webcam directly without configuration - just select "OpenCV" from the camera dropdown.
* **Automatic Micro-Manager Installation:** No more manual prerequisites - Micro-Manager components are automatically downloaded on first run.
* **Python 3.11:** Updated to Python 3.11 for improved performance and compatibility.
* **Background Arduino Polling:** Improved responsiveness when using Arduino-based pressure and temperature monitoring.
* **Large File Handling:** Automatic file splitting for long recordings to prevent oversized TIFF files.

---

## Key Features

* **Software Base:** Now using μManager 2.0.
* **Programming Language:** Updated to Python 3.11 for better performance and compatibility.
* **Live Data Acquisition:** Allows for the real-time display of pressurized arteries mounted in the VasoTracker vessel chamber.
* **Diameter Measurement:** Real-time measurement and display of both outer and inner artery diameters.
* **Multiple Tracking Algorithms:** Allow accurate tracking of brightfield or fluorscence imaging data.
* **Environmental Monitoring:** Continuously tracks bath temperature and intraluminal pressure.
* **Data Recording:** Live recording and graphing of artery diameters.
* **Experimental Tracking:** Facilitates the tracking of experimental manipulations, such as drug additions.
* **Advanced Tracking Options:** Includes multi-line diameter tracking and the ability to specify regions of interest (ROI).
* **Data Analysis:** Implements line averaging and statistical filtering to refine results.
* **Pressure Control:** Integrates with National Instruments DAQ boards, enabling automatic control of Living Systems PS-200 pressure servo systems.
* **Video Output:** Supports exporting data to .tiff files for further analysis.

---

## VasoTracker Installation Instructions

VasoTracker can be installed using either the standalone executable file for straightforward setup or from the source code for more advanced customization options. Below are the steps for both methods.

### Option 1: Installing and Running from the Executable File

***This method is recommended for most users.***

#### Steps:

1. **Download the latest VasoTracker release:**
   - Visit the [VasoTracker Releases Page](https://github.com/VasoTracker/VasoTracker-2-Software/releases) and download the latest zip file for your operating system.

2. **Extract the Zip File:**
   - Locate the downloaded zip file on your computer.
   - Right-click the file and select "Extract All..." or use your preferred extraction software.
   - Choose a destination folder to extract the files and confirm the action.

3. **Run VasoTracker:**
   - Navigate to the extracted folder.
   - Double-click the executable file to start the application.
   - **On first run**, VasoTracker will automatically download and install the required Micro-Manager components. This may take a few minutes.

#### Using Webcams and USB Cameras

VasoTracker supports Basler and Thorlabs cameras out of the box. For webcams and USB cameras, you have two options:

**Option 1: Native OpenCV (Recommended)**
- Simply select **"OpenCV"** from the camera dropdown
- Works immediately with any USB camera or webcam
- No configuration required

**Option 2: Micro-Manager Configuration (Advanced)**

For cameras requiring specific Micro-Manager device adapters:

1. **Open Micro-Manager** (automatically installed by VasoTracker):
   ```
   C:\Users\<YourName>\AppData\Local\pymmcore-plus\pymmcore-plus\mm\<version>\ImageJ.exe
   ```

2. **Create a hardware configuration:**
   - Go to **Devices > Hardware Configuration Wizard**
   - Add your camera device
   - Complete the wizard and save the configuration

3. **Save the config file:**
   - Save as `MMConfig.cfg` in your VasoTracker folder:
     - Packaged app: inside the `_internal` folder next to `vasotracker_x.y.z.exe` (replace the bundled `MMConfig.cfg`)
     - Running from source: the `vasotracker_2` folder
     - (When you select "MMConfig" as the camera, the console prints the exact path it is looking in.)

4. **Select your camera in VasoTracker:**
   - Choose "MMConfig" as your camera type

### Option 2: Installing from Source Using Anaconda

***For users who need more control over the installation environment or wish to contribute to the software development.***

#### Steps:

1. **Clone the Repository:**
   - Clone the VasoTracker repository to your local machine using:
     ```
     git clone https://github.com/VasoTracker/VasoTracker-2-Software.git
     ```

2. **Set Up the Anaconda Environment:**
   - Navigate to the directory where you cloned the repository.
   - Use the provided `environment.yml` file to create the VasoTracker Anaconda environment:
     ```
     conda env create -f environment.yml
     ```

3. **Activate the Environment:**
   - Activate the newly created environment:
     ```
     conda activate vasotracker2
     ```

4. **Install Micro-Manager Components:**
   - Install the required Micro-Manager device adapters:
     ```
     mmcore install
     ```

5. **Run VasoTracker:**
   - Navigate to the vasotracker_2 folder and run:
     ```
     cd vasotracker_2
     python vasotracker_2.py
     ```

This approach ensures you have a development environment configured with all necessary dependencies, allowing you to modify or use VasoTracker immediately.

---

## License

Distributed under the terms of the [The 3-Clause BSD License]

"VasoTracker" is free and open source software

---

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[μManager 2.0]: https://micro-manager.org/
[The 3-Clause BSD License]: http://opensource.org/licenses/BSD-3-Clause

#### Added Fund

Sometimes, a little bit of Pac-Man or Space Invaders is required. We included these games courtesy of:

   - [Whoever created PacManCode](https://pacmancode.com/)
   - [Lee Rob on GitHub](https://github.com/leerob/space-invaders)

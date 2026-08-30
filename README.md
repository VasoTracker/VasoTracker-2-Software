# VasoTracker 2.4 - Blood Vessel Diameter Tracking Software (online and offline analysis)

The VasoTracker 2.4 software is a comprehensive software solution designed for the acquisition and analysis of blood vessel imaging data. It supports both live and pre-recorded video analysis, making it adaptable for various experimental set ups. It was initially developed for pressure myography, but it works for many other types of imaging!

![til](https://github.com/VasoTracker/VasoTracker-2-Software/blob/main/VasoTracker%20GUI.gif)



## Table of Contents
- [What's New in v2.4.1](#whats-new-in-v241)
- [What's New in v2.4](#whats-new-in-v24)
- [What's New in v2.3](#whats-new-in-v23)
- [What's New in v2.2](#whats-new-in-v22)
- [Key Features](#key-features)
- [Installation Instructions](#vasotracker-installation-instructions)
  - [Executable File](#option-1-installing-and-running-from-the-executable-file)
  - [From Source](#option-2-installing-from-source-using-anaconda)
- [License](#license)
- [Issues](#issues)

---

## What's New in v2.4.1 (August 2026)

* **Recorded TIFFs open in classic ImageJ again.** v2.4.0 wrote recordings as BigTIFF, which ImageJ's built-in reader (without Bio-Formats) rejects as an unsupported format. Recordings are now standard TIFF - they still roll over to a new file below 4 GB, and the lossless Deflate compression is unchanged and ImageJ-readable.

---

## What's New in v2.4 (August 2026)

* **Micro-Manager: install it yourself.** VasoTracker no longer downloads or installs Micro-Manager. It needs one specific nightly build (currently `20260828`, device interface 75); if a compatible install isn't found it names the exact build, lists any incompatible ones it did find, and opens the download page. Install that nightly the normal way (default location) and it's picked up automatically. Supports current Micro-Manager nightlies.
* **Choose your Micro-Manager config.** Selecting the "MMConfig" camera opens a chooser: pick from the `.cfg` files found next to your Micro-Manager install (or browse to one), test it with a live preview, then use it. The choice is remembered.
* **Recording is a proper time-lapse.** Fixed the frame-interval logic (previously it only saved on exact whole-second boundaries, so most settings saved nothing); files now flush to disk continuously and survive a crash; default interval lowered to 10 s. New "incl. tracked overlay" toggle - turn it off to save only the raw stack + CSV. Recorded TIFFs are now compressed (lossless, ~2-3x smaller) and carry a `Frame` number matching the new `Frame` column in the results CSV.
* **Analyse video files, not just TIFFs.** "Analyse File" now accepts `.avi`, `.mp4`, `.mov` and similar alongside multi-frame TIFFs - useful for ultrasound / flow-mediated-dilation clips. Frames are read straight from the file; no conversion step.
* **New vascular-ultrasound / FMD analysis.** Ticking the "US" checkbox now opens a chooser: **Vascular wall tracking (FMD)** or **Standard ultrasound (legacy)**. The FMD option is a B-mode wall model built for flow-mediated dilation - it locates the bright near/far wall reflections either side of the anechoic lumen, reports **inner diameter = the lumen (intima-intima)** and outer = wall-to-wall, averages along the vessel to beat speckle, and seeds each frame from the last so the vessel can drift without the ROI clipping it. On test clips it recovers the dilation curve the legacy ultrasound mode missed entirely.
* **Snapshot works again.** The Snapshot button had been calling a removed function since January and did nothing; restored, with filenames that don't overwrite each other.
* **Runs from any working directory.** Fixed icon/image/config resource paths that only resolved when the app was launched from inside its own folder.
* **Settings file no longer churns.** Saving settings was silently corrupting three values (`integration`, inner-diameter axis max, default pressure) on every write.

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

3. **Install Micro-Manager:**
   - VasoTracker needs the **Micro-Manager 2.0 nightly build dated `20260828`** (Windows). Download it from the [Micro-Manager nightly archive](https://download.micro-manager.org/nightly/2.0/Windows/) and run the installer (the default location is fine).
   - A different Micro-Manager version will not work - VasoTracker's imaging core is tied to one specific build. If it starts and can't find a compatible install, it names the exact one to get.

4. **Run VasoTracker:**
   - Navigate to the extracted folder.
   - Double-click the executable file to start the application.

#### Using Webcams and USB Cameras

VasoTracker supports Basler and Thorlabs cameras out of the box. For webcams and USB cameras, you have two options:

**Option 1: Native OpenCV (Recommended)**
- Simply select **"OpenCV"** from the camera dropdown
- Works immediately with any USB camera or webcam
- No configuration required

**Option 2: Micro-Manager Configuration (Advanced)**

For cameras requiring specific Micro-Manager device adapters:

1. **Open Micro-Manager** (the nightly you installed above):
   ```
   C:\Program Files\Micro-Manager-2.0\ImageJ.exe
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

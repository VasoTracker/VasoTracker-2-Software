# VasoTracker 2.0 - Pressure Myograph and Blood Vessel Diameter Tracking Software (online and offline analysis)

The VasoTracker 2.0 software is a comprehensive software solution designed for the acquisition and analysis of imaging data from blood vessels mounted in a pressure myograph. It supports both live and pre-recorded video analysis, making it adaptable for various experimental set ups.

## Table of Contents
- [Key Features](#key-features-of-vasotracker-20)
- [Installation Instructions](#vasotracker-installation-instructions)
  - [Executable File](#option-1-installing-and-running-from-the-executable-file)
  - [From Source](#option-2-installing-from-source-using-anaconda)
- [License](#license)
- [Issues](#issues)

---

## Key Features of VasoTracker 2.0

* **Software Base:** Now using μManager 2.0.
* **Programming Language:** Updated to Python 3.X for better performance and compatibility.
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

#### Prerequisites

1. **Install uManager:**
   - Visit the [µManager Downloads Page](https://micro-manager.org/wiki/Download_Micro-Manager_Latest_Release) and download the latest **nightly build** of µManager for your operating system.
   - Follow the provided instructions to install µManager on your computer.

#### Steps:

1. **Download the latest VasoTracker release:**
   - Visit the [VasoTracker Releases Page](https://github.com/VasoTracker/VasoTracker-2/releases) and download the latest zip file for your operating system.

2. **Extract the Zip File:**
   - Locate the downloaded zip file on your computer.
   - Right-click the file and select "Extract All..." or use your preferred extraction software.
   - Choose a destination folder to extract the files and confirm the action.

3. **Run VasoTracker:**
   - Navigate to the extracted folder.
   - Double-click the executable file to start the application.

### Option 2: Installing from Source Using Anaconda

***For users who need more control over the installation environment or wish to contribute to the software development.***

#### Steps:

1. **Clone the Repository:**
   - Clone the VasoTracker repository to your local machine using:
     ```
     git clone URL-to-VasoTracker-repository
     ```

2. **Set Up the Anaconda Environment:**
   - Navigate to the directory where you cloned the repository.
   - Use the provided `environment.yml` file to create the VasoTracker Anaconda environment. This file includes the necessary Python version and all dependencies:
     ```
     conda env create -f environment.yml
     ```
   - This command will download Python and all the dependencies listed in the `environment.yml` file, setting up a self-contained environment tailored for VasoTracker.

3. **Activate the Environment:**
   - Activate the newly created environment to switch to it:
     ```
     conda activate vasotracker2
     ```

4. **Run or Modify VasoTracker:**
   - You are now ready to run or modify the VasoTracker software. Navigate to the folder and run or edit the code as needed.
   - To run vasotracker from source, use the following command:
     ```
     python vasotracker2.py
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

Sometimes, a little bit of Pac-Man or Space Invaders is required. We included these games courtesy of the people

   - [Whoever created PacManCode](https://pacmancode.com/)
   - [Lee Rob on GitHub](https://github.com/leerob/space-invaders)
##################################################
## VasoTracker 2 - Blood Vessel Diameter Measurement Software
##
## Author: Calum Wilson, Matthew D Lee, and Chris Osborne
## License: BSD 3-Clause License (See main file for details)
## Website: www.vasostracker.com
##
##################################################


__version__ = "2.3.2"

# Micro-Manager compatibility pins - keep these two in sync (MICROMANAGER.md).
# MM_DEVICE_INTERFACE must equal the device interface version of the pymmcore
# pinned in environment.yml (the 4th field of pymmcore.__version__).
# MM_COMPATIBLE_NIGHTLY is the newest Micro-Manager Windows nightly built
# against that interface; it is what the app auto-installs.
MM_DEVICE_INTERFACE = 74
MM_COMPATIBLE_NIGHTLY = "20251231"

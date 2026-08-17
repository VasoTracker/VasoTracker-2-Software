# -*- mode: python ; coding: utf-8 -*-

# -*- mode: python ; coding: utf-8 -*-
import sys
sys.setrecursionlimit(sys.getrecursionlimit() * 5)

import os
import sys

# Ensure the current directory of the spec file is the working directory
spec_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.insert(0, spec_dir)


import version
from version import __version__

# Build-time check: the Micro-Manager nightly the app auto-installs must be
# built against the same device interface as the pymmcore we are bundling.
# When pymmcore is upgraded past a device interface bump, this fails the
# build until the pins in version.py are updated (see MICROMANAGER.md) -
# instead of shipping an app that installs an incompatible Micro-Manager.
import pymmcore as _pymmcore_check
_bundled_div = int(str(_pymmcore_check.__version__).split(".")[3])
if _bundled_div != version.MM_DEVICE_INTERFACE:
    raise SystemExit(
        f"pymmcore in the build env speaks device interface {_bundled_div}, but "
        f"version.py pins Micro-Manager for interface {version.MM_DEVICE_INTERFACE}. "
        "Update MM_DEVICE_INTERFACE and MM_COMPATIBLE_NIGHTLY in version.py "
        "per MICROMANAGER.md before building."
    )
del _pymmcore_check

# Build-time sanity check: the build env must provide pyserial, not the
# unrelated "serial" package (both install a module named `serial`). A build
# with the wrong one ships an Arduino controller that dies with
# "module 'serial' has no attribute 'Serial'".
import serial as _serial_check
if not hasattr(_serial_check, "Serial"):
    raise SystemExit(
        "Build environment has the wrong 'serial' module (not pyserial). "
        "Fix with: pip uninstall serial && pip install pyserial"
    )
del _serial_check

added_files = [("music", "music"), ("images", "images"), ("SampleData", "SampleData"), ('settings.toml', '.'), ('MMConfig.cfg', '.'), ('Basler.cfg', '.'), ('VasoTrackerblue.json', '.'), ('pacman', 'pacman'), ('space-invaders', 'space-invaders')]

# Conda keeps the C libraries behind Python's stdlib extension modules in
# Library\bin, which PyInstaller misses. Without them the frozen app dies at
# startup with "DLL load failed" (_ctypes needs ffi, pyexpat needs libexpat,
# _ssl needs libssl/libcrypto, etc). Bundle them all explicitly.
import glob
_env_bin = os.path.join(sys.prefix, "Library", "bin")
_conda_dll_patterns = [
    "ffi*.dll",            # _ctypes
    "*expat*.dll",         # pyexpat
    "libssl*.dll",         # _ssl
    "libcrypto*.dll",      # _ssl, _hashlib
    "sqlite3*.dll",        # _sqlite3
    "*bz2*.dll",           # _bz2
    "liblzma*.dll",        # _lzma
    "zlib*.dll",           # zlib users
    "tcl86*.dll",          # tkinter
    "tk86*.dll",           # tkinter
]
conda_binaries = []
for _pat in _conda_dll_patterns:
    conda_binaries += [(p, ".") for p in glob.glob(os.path.join(_env_bin, _pat))]
if not any("ffi" in os.path.basename(p).lower() for p, _ in conda_binaries):
    raise SystemExit(f"No ffi*.dll found in {_env_bin} - frozen _ctypes would fail to load.")

a = Analysis(
    ['vasotracker_2.py'],
    pathex=[],
    binaries=conda_binaries,
    datas=added_files,
    hiddenimports=['PyDAQmx', 'scipy'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=f"vasotracker_{__version__}",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    uac_admin=True,
    icon='D:\\OneDrive - University of Strathclyde\\Documents\\GitHub\\VasoTracker-2-Software\\vasotracker_2\\images\\vt_icon.ico',
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=f"vasotracker_{__version__}",
)

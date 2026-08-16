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

a = Analysis(
    ['vasotracker_2.py'],
    pathex=[],
    binaries=[],
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

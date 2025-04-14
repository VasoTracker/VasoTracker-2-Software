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
    uac_admin=False,
    icon='C:/Users/Calum/Documents/GitHub/VasoTracker-2/software/vasotracker 2.0/images/vt_icon.ico',
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

# Micro-Manager, pymmcore, and the Device Interface Version

A note for maintainers (written August 2026, after this bit us twice).

## The moving parts

VasoTracker talks to cameras through three layers, each versioned
separately:

1. **pymmcore** (Python package, bundled into VasoTracker) - a compiled
   copy of Micro-Manager's C++ core (MMCore). Its version string ends in
   the **device interface version (DIV)** it speaks, e.g.
   `11.10.0.74.1` -> DIV **74**.
2. **pymmcore-plus** (Python package, bundled) - convenience layer on
   top: finds/installs Micro-Manager, `CMMCorePlus`, etc.
3. **Micro-Manager itself** (installed on the user's machine) - supplies
   the **device adapters** (the DLLs that drive actual cameras). Every
   nightly build is compiled against one specific DIV.

**The rule: the adapters' DIV must exactly match pymmcore's DIV.**
There is no backwards compatibility. An MMConfig file or device adapter
built for DIV 71, 72 or 75 will not load into a pymmcore that speaks 74.

## How VasoTracker handles it

- On startup, `find_micromanager()` (pymmcore-plus) looks for an
  installation with a matching DIV - it checks its own managed folder
  (`%LOCALAPPDATA%\pymmcore-plus\...\mm\`) first, then Program Files.
  Incompatible installations are ignored (this is why an installed
  Micro-Manager 1.4 or a too-new nightly is "invisible" to VasoTracker).
- If none is found, VasoTracker auto-installs one. **This is pinned to
  the newest nightly known to match our bundled DIV** - see
  `KNOWN_COMPATIBLE_MM_NIGHTLY` in `vasotracker_2.py` (currently
  `20251231`, the newest DIV-74 Windows nightly).

## Why the pin exists (what went wrong)

pymmcore-plus's `install()` defaults to `release="latest-compatible"`,
which sounds right but resolves against a **table of interface versions
hard-coded at the time the package was published**. When our bundled
pymmcore's DIV is the newest in that table, it assumes the latest
nightly is still compatible and downloads `MMSetup_x64_latest.exe`.

That assumption expires the day Micro-Manager bumps its interface. In
August 2026 the nightlies moved past DIV 74, and from then on every
fresh user's auto-install fetched an incompatible Micro-Manager: the
app then reported "Could not find a compatible Micro-Manager
installation ... required by pymmcore 74" on every launch even though
an install was present. (First field report: 17 Aug 2026. The same
class of mismatch - config files built for DIV 70/71/72 against our
DIV-71-era release - was reported by the Pittsburgh lab in Dec 2025.)

## Maintenance procedure

When upgrading pymmcore / pymmcore-plus in `environment.yml`:

1. Check the new DIV: `python -c "import pymmcore; print(pymmcore.CMMCore().getAPIVersionInfo())"`
2. If the DIV changed, find the newest Windows nightly built against it
   at <https://download.micro-manager.org/nightly/2.0/Windows/>
   (the nightly's DIV bump dates are in pymmcore-plus's
   `pymmcore_plus/install.py` INTERFACES table - or just test one).
3. Update `KNOWN_COMPATIBLE_MM_NIGHTLY` in `vasotracker_2.py`.
4. Rebuild the executable and test on a machine (or fresh user account)
   without an existing `%LOCALAPPDATA%\pymmcore-plus` folder, so the
   auto-install path actually runs.

If the pinned nightly ever disappears from the download server, the
code falls back to pymmcore-plus's own resolution (and prints the
pinned failure to the console). The startup failure dialog tells users
which nightly date to install manually.

## Helping a user whose install is broken

Symptoms: "Could not find a compatible Micro-Manager installation for
the device interface required by pymmcore NN" at startup, possibly
followed by a failed download/install.

1. Have them delete `%LOCALAPPDATA%\pymmcore-plus\pymmcore-plus\mm\`
   (stale incompatible auto-installs live there and are re-checked on
   every launch).
2. Relaunch VasoTracker - the pinned auto-install should fetch the
   right build.
3. If it cannot (offline/firewalled), have them install the nightly
   named in the error dialog manually from
   <https://download.micro-manager.org/nightly/2.0/Windows/> - the
   default install location (Program Files) is found automatically.

Installer exit code 5 means the (Inno Setup) installer aborted during
the file-copy phase - in silent mode that is almost always a locked
file in the target folder. The classic sequence: an earlier launch
already installed an (incompatible) build to the same folder, the user
launches again ("nothing happened"), and the re-install races a
previous installer / antivirus / indexer over the same files. Deleting
the folder in step 1 clears it; the pinned install prevents the
reinstall loop from happening at all.

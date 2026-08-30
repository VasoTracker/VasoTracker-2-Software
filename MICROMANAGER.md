# Micro-Manager, pymmcore, and the Device Interface Version

A note for maintainers (written August 2026, after this bit us twice).

## The moving parts

VasoTracker talks to cameras through three layers, each versioned
separately:

1. **pymmcore** (Python package, bundled into VasoTracker) - a compiled
   copy of Micro-Manager's C++ core (MMCore). Its version string ends in
   the **device interface version (DIV)** it speaks, e.g.
   `12.5.0.75.0` -> DIV **75**.
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
  Micro-Manager 1.4 or a wrong-DIV nightly is "invisible" to VasoTracker).
- **VasoTracker does not install Micro-Manager.** If no compatible
  installation is found, it shows a dialog naming the exact nightly to
  install (`MM_COMPATIBLE_NIGHTLY` in `version.py`, currently `20260828`,
  a DIV-75 Windows nightly) and the DIV it needs, lists any incompatible
  installs it did find, opens the nightly download page, and exits. The
  user installs the named nightly the normal way (default location) and
  relaunches; it is then found in Program Files.

## Why the pin exists (what went wrong)

VasoTracker used to auto-install via pymmcore-plus's `install()`, whose
`release="latest-compatible"` default resolves against a **table of
interface versions hard-coded when the package was published**. Once our
bundled pymmcore's DIV was the newest in that table it assumed the latest
nightly still matched and downloaded `MMSetup_x64_latest.exe`.

That assumption expired the day Micro-Manager bumped its interface. In
August 2026 the nightlies moved past DIV 74 and every fresh user's
auto-install then fetched an incompatible Micro-Manager: the app reported
"Could not find a compatible Micro-Manager installation ... required by
pymmcore 74" on every launch even though an install was present. (First
field report: 17 Aug 2026. The same class of mismatch - config files
built for DIV 70/71/72 against our DIV-71-era release - was reported by
the Pittsburgh lab in Dec 2025.) Auto-install was removed in response;
the pinned nightly name is now guidance shown to the user.

## Maintenance procedure

When upgrading pymmcore / pymmcore-plus in `environment.yml`:

1. Check the new DIV: `python -c "import pymmcore; print(pymmcore.CMMCore().getAPIVersionInfo())"`
2. If the DIV changed, find the newest Windows nightly built against it
   at <https://download.micro-manager.org/nightly/2.0/Windows/>
   (the nightly's DIV bump dates are in pymmcore-plus's
   `pymmcore_plus/install.py` INTERFACES table - or just test one).
3. Update `MM_DEVICE_INTERFACE` and `MM_COMPATIBLE_NIGHTLY` in `version.py`.
   The build (`vasotracker_2.spec`) fails if the bundled pymmcore's DIV
   does not match `MM_DEVICE_INTERFACE`.
4. Update the required-nightly date in `README.md`.
5. Rebuild and test on a machine with only that nightly installed.

Existing users must install the newly-named nightly when they take a
VasoTracker update that bumped the DIV - the startup dialog tells them
which one and why.

## Helping a user whose install is broken

Symptom: the "Micro-Manager required" dialog at startup, listing the
nightly to install (and possibly an incompatible install it found).

1. Have them install the exact nightly named in the dialog from
   <https://download.micro-manager.org/nightly/2.0/Windows/> - default
   location (Program Files) is found automatically. A wrong-DIV nightly
   or a Micro-Manager 1.4 does not count; it must be that build.
2. Multiple Micro-Manager installs can coexist - they do not need to
   uninstall the incompatible one, just add the right one.
3. If a stale/partial auto-install from an old VasoTracker is still
   being picked up (`%LOCALAPPDATA%\pymmcore-plus\pymmcore-plus\mm\`),
   deleting that folder removes it from the search.

The old auto-install could hit Inno Setup exit code 5 (a locked file in
the target folder when a previous install/AV/indexer was still touching
it). Removing auto-install removed that failure mode.

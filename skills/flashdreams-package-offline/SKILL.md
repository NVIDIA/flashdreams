---
name: flashdreams-package-offline
description: Package an installed FlashDreams V2 application as an offline executable folder and diagnose preload or bundled-cache failures. Use for native-window or WebRTC distribution without Python, uv, or network access on the target. Ordinary demo launching, generic Python packaging, and website builds are outside this workflow.
license: Apache-2.0
metadata:
  author: "FlashDreams Team <flashdreams@nvidia.com>"
---

# Package FlashDreams for offline use

## Establish the target and working launch

Locate the user's checkout of `https://github.com/NVIDIA/flashdreams` and run
commands from its root. If only a plan or command is requested, do not start
downloads, compilation or a bundle build. Record the application preset, runtime
mode, application arguments, source revision, target OS, GPU and driver.

The packager supports `native-window` and `webrtc`. It produces a host-native
folder, not a cross-platform binary: build Windows on Windows and Linux on Linux.
The target still needs a compatible NVIDIA GPU and driver. Select a preset that
actually supports that hardware rather than substituting a fast preset by name.

First verify the normal V2 launch with its documented dependencies. Install the
integration and any app extras, then the packager group while retaining them:

```bash
uv sync --package flashdreams-omnidreams --inexact
uv sync --group offline-exe --inexact
uv run --no-sync python tools/package-as-offline-exe/package_as_offline_exe.py --help
```

For Interactive Drive, include its `interactive-drive` extra in the first sync.
These are environment-changing operations and may compile native packages.
Preload and real packaging require the selected model's resources and GPU.

## Build the folder

Use `:::` to separate packager options from the complete application command.
Within that command, runtime arguments precede `--` and app arguments follow it.

```bash
uv run --no-sync python tools/package-as-offline-exe/package_as_offline_exe.py \
  --output artifacts/taxi-omnidreams-bundle \
  ::: flashdreams-run-v2 crazy-robotaxi-omnidreams --mode native-window
```

The output directory must not already exist. Choose a new destination rather
than deleting an existing bundle. Do not pass `--preload-application`: the
packager manages preload and the check that attempts one model-loop step.

Useful options, before `:::`:

- `--preload-timeout SECONDS`: bound startup when downloads/builds are slow.
- `--hidden-import MODULE`: include a dynamically imported module known to be missing.
- `--collect-executable NAME`: include a required executable from `PATH`.
- `--skip-runtime-validation`: skip the model-loop step check while still
  initializing the application. Even an enabled check may produce no frames if
  the app is waiting for input; verify actual generation separately.
- `--noconsole`: hide the Windows console; runtime logs remain in the user's temp area.

Check the current tool's help before adding options. If preparation finds a
download, compilation or native load after initialization, fix the application:
complete one-time preparation within `IApplication.init`, waiting for any
preparation workers before returning. The preparation guard applies after `init`
returns, including to workers started there. Later steps consume prepared state.
Do not fix offline failures by depending on a developer's warm cache or allowing
downloads on the target.

## Validate distribution, not just the build

Inspect `INSTALLER_OUTPUT.txt` and any `PREPARATION_ISSUES.txt`. A successful
PyInstaller step alone is insufficient. Distribute the complete raw folder:
the executable, `_internal/`, `cache/`, generated assets and supplied notices.
Copying just the executable loses required libraries and prepared resources.

The bundle's cache is a read-only seed. On first launch it is copied to a writable
per-user location, normally `~/flashdreams/unpacked/<application-slug>/cache`
on Linux or `%USERPROFILE%\flashdreams\unpacked\<application-slug>\cache` on Windows.
`--unpack-dir ABSOLUTE_PATH` selects the parent of `<application-slug>/cache`;
`FLASHDREAMS_RUNTIME_CACHE_DIR` selects the exact writable cache directory.
The command-line option takes precedence. Prepared native extensions still load
from the original bundled cache, so the writable copy is not a replacement bundle.

Test the complete folder on a clean, compatible target with networking disabled,
empty user caches, and no development environment. Exercise the requested output
mode, generation, restart, and first-launch cache creation. Confirm redistributable
model/asset/dependency terms before distributing. Do not include credentials or
unrelated files from the build machine in the deliverable.

Report the build command, source revision, OS/GPU/driver, bundle path, validation
logs and any skipped runtime checks. If only command construction was checked,
say so; do not label the bundle offline-validated.

## References

- [Offline program packager](https://flashdreams.org/main/documentation/tools/offline_program_packager/)
- [CLI and preload](https://flashdreams.org/main/documentation/cli/)

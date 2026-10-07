# Package a FlashDreams application for offline use

`package_as_offline_exe.py` prepares an installed FlashDreams v2 application
and creates a single executable with its offline cache in one folder.
Packaging is supported for applications running in `native-window` or `webrtc`
mode.

## Usage

From the repository root, use `:::` to separate this tool's options from the
complete `flashdreams-run-v2` command:

```bash
uv run python tools/package-as-offline-exe/package_as_offline_exe.py \
  --output artifacts/interactive-drive-omnidreams-bundle \
  ::: flashdreams-run-v2 interactive-drive-omnidreams-fast-perf \
  --mode native-window -- --game-mode
```

The command after `:::` must begin with `flashdreams-run-v2`, and its effective
mode must be `native-window` or `webrtc`.

To package the Interactive Drive browser experience, select `webrtc` normally:

```bash
uv run python tools/package-as-offline-exe/package_as_offline_exe.py \
  --output artifacts/interactive-drive-omnidreams-webrtc-bundle \
  ::: flashdreams-run-v2 interactive-drive-omnidreams \
  --mode webrtc --host 0.0.0.0 --port 8089
```

Use the runtime command's `--` separator when the application needs its own
arguments:

```bash
uv run python tools/package-as-offline-exe/package_as_offline_exe.py \
  --output artifacts/interactive-drive-omnidreams-bundle \
  ::: flashdreams-run-v2 interactive-drive-omnidreams \
  --mode native-window \
  -- --game-mode
```

If `--output` is omitted, the default is
`artifacts/<application-slug>-bundle`. The output directory must not already
exist.

To package the fast-perf Interactive Drive game mode:

```powershell
uv run python tools/package-as-offline-exe/package_as_offline_exe.py `
  --output artifacts/interactive-drive-omnidreams-fast-perf-onefile-1700 `
  ::: flashdreams-run-v2 interactive-drive-omnidreams-fast-perf `
  --mode native-window -- --game-mode
```

The output is still a directory. `cache/` holds the downloaded model weights
and prepared kernels. When a onefile bundle would exceed 1,700 files, the
packager stores the largest cache directories in `cache-extras.zip`. The
launcher extracts that archive into the writable runtime cache on first launch;
subsequent launches reuse it. Preload may also produce sibling `artifacts/`
assets. Set `FLASHDREAMS_RUNTIME_CACHE_DIR` to a path outside the bundle if
overriding the default per-user cache.
PyInstaller extracts the embedded runtime files to a temporary directory on
each launch, which adds startup time.

On a Windows build of `interactive-drive-omnidreams-fast-perf` on 2026-10-07,
the completed bundle contained **98 files**: a 2.28 GiB EXE, a 0.47 GiB cache
archive, and 93 files in the 21.79 GiB `cache/`. A one-step native-window
game-mode run used the default per-user cache, restored the PhysX tree, rendered
five frames, and exited successfully in 136.1 seconds. This timing is
host-specific and includes model startup.

## Runtime and application arguments

Arguments before the runtime command's `--` are FlashDreams runtime arguments;
arguments after it are application arguments:

```bash
uv run python tools/package-as-offline-exe/package_as_offline_exe.py \
  --output artifacts/interactive-drive-omnidreams-bundle \
  ::: flashdreams-run-v2 interactive-drive-omnidreams \
  --mode native-window --timeout 120 \
  -- --game-mode
```

Do not pass `--preload-application`; the packager controls preparation so it
can validate and populate the bundle's caches.

The generated launcher embeds the supplied application arguments to this installer program inside the executable.

The packager runs initialization and one model block. Preparation findings are
written to `PREPARATION_ISSUES.txt` beside `INSTALLER_OUTPUT.txt`.

## Options

```text
--output PATH
    Destination bundle directory.

--preload-timeout SECONDS
    Fail if initialization and preload do not finish within this time.

--skip-runtime-validation
    Skip the one-block runtime check. Initialization still runs.

--hidden-import MODULE
    Add a PyInstaller hidden import. Repeat for multiple modules.

--collect-executable NAME
    Include an executable found on PATH. Repeat for multiple executables.
```

For the authoritative command-line help, run:

```bash
uv run python tools/package-as-offline-exe/package_as_offline_exe.py --help
```

## Generated bundle

The generated directory has this shape:

```text
<output>/
|-- <application-slug>[.exe]
|-- cache/
|-- cache-extras.zip        # when needed to meet the file limit
|-- artifacts/             # when preload produces application assets
|-- INSTALLER_OUTPUT.txt
|-- PREPARATION_ISSUES.txt  # only when preload warnings are found
`-- README.md
```

The executable contains the Python runtime, application code, dependencies,
native libraries, Ninja, and packaged GPU assets. The bundle has at most
1,700 files; `cache-extras.zip` holds cache files moved out of the directory
tree to meet that limit.

Notes on generated files:

- `cache/` is the offline seed containing resources and build artifacts
  prepared during initialization.
- `INSTALLER_OUTPUT.txt` contains the application preload/validation and
  PyInstaller build output, with the exit code for each completed step.
- `PREPARATION_ISSUES.txt` contains preload warnings and their application call
  stacks when validation finds preparation work outside `IApplication.init`.
- The generated `README.md` explains how to launch a generated bundle.
- `<application-slug>[.exe]` is the executable that can be launched to run the application.
    - Windows builds use the `.exe` suffix; Linux builds do not.

The launcher sets its working directory to the bundle directory when started,
so relative application paths resolve beside the executable.

## Runtime cache behavior

The packaged `cache/` directory remains read-only. On first launch of the executable,
the launcher copies it into a writable per-user cache:

- Windows: `%LOCALAPPDATA%\FlashDreams\<application-slug>\cache`
- Linux: `$XDG_CACHE_HOME/flashdreams/<application-slug>`, or
  `~/.cache/flashdreams/<application-slug>` when `XDG_CACHE_HOME` is unset

Set `FLASHDREAMS_RUNTIME_CACHE_DIR` if a custom cache location is desired.

Keep the executable with its sibling `cache/`, any `cache-extras.zip`, and any
`artifacts/` assets. Set `FLASHDREAMS_RUNTIME_CACHE_DIR` to move the writable
cache.

Destination machine still needs a compatible NVIDIA driver to run the packaged application.

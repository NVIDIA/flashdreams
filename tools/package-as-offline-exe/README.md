# Package a FlashDreams application for offline use

`package_as_offline_exe.py` prepares an installed FlashDreams v2 application
and creates an isolated executable with all its dependencies in one folder.
Packaging is supported for applications running in `native-window` or `webrtc`
mode.

## Usage

From the repository root, use `:::` to separate this tool's options from the
complete `flashdreams-run-v2` command:

```bash
uv run python tools/package-as-offline-exe/package_as_offline_exe.py \
  --output artifacts/interactive-drive-omnidreams-bundle \
  ::: flashdreams-run-v2 interactive-drive-omnidreams \
  --mode native-window
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
|-- data/
|-- cache/
|-- INSTALLER_OUTPUT.txt
|-- PREPARATION_ISSUES.txt  # only when preload warnings are found
`-- README.md
```

Notes on generated files:

- `data/` contains the Python runtime, application code, dependencies, native
  libraries, and packaged GPU assets.
- `cache/` is the offline seed containing resources and build artifacts
  prepared during initialization.
- `INSTALLER_OUTPUT.txt` contains the application preload/validation and
  PyInstaller build output, with the exit code for each completed step.
- `PREPARATION_ISSUES.txt` contains preload warnings and their application call
  stacks when validation finds preparation work outside `IApplication.init`.
- The generated `README.md` explains how to launch a generated bundle.
- `<application-slug>[.exe]` is the executable that can be launched to run the application.
    - Windows builds use the `.exe` suffix; Linux builds do not.

## Runtime cache behavior

The packaged `cache/` directory remains read-only. On first launch of the executable,
the launcher copies it into a writable per-user cache:

- Windows: `%LOCALAPPDATA%\FlashDreams\<application-slug>\cache`
- Linux: `$XDG_CACHE_HOME/flashdreams/<application-slug>`, or
  `~/.cache/flashdreams/<application-slug>` when `XDG_CACHE_HOME` is unset

Set `FLASHDREAMS_RUNTIME_CACHE_DIR` if a custom cache location is desired.

Do not move or distribute only the executable. It depends on the sibling
`data/` and `cache/` directories. Set `FLASHDREAMS_RUNTIME_CACHE_DIR` to move
the writable cache.

Destination machine still needs a compatible NVIDIA driver to run the packaged application.

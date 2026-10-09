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

If `--output` is omitted, the default is
`artifacts/<application-slug>-bundle`. The output directory must not already
exist.

The output is still a directory. `cache/` holds the downloaded model weights
and prepared kernels. When a onefile bundle would exceed 1,700 files, the
packager stores the largest cache directories in `cache-extras.zip`. The
launcher extracts that archive into the writable runtime cache on first launch;
subsequent launches reuse it. Preload may also produce sibling `artifacts/`
assets. Pass `--unpack-dir` to the packaged executable to change where its
cache is extracted.
PyInstaller extracts the embedded runtime files to a temporary directory on
each launch, which adds startup time.


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
the launcher copies it into a writable runtime cache:

- Windows: `%USERPROFILE%\flashdreams\unpacked\<application-slug>\cache`
- Linux: `~/flashdreams/unpacked/<application-slug>/cache`

Pass `--unpack-dir PATH` to the packaged executable to use
`PATH/<application-slug>/cache` instead. `PATH` must be absolute. For example,
`interactive-drive-omnidreams-fast-perf.exe --unpack-dir D:\FlashDreamsData`.
`FLASHDREAMS_RUNTIME_CACHE_DIR` remains available to set the exact cache
directory; the command-line option takes precedence. PyInstaller's onefile
runtime files still extract to the OS temp directory on every launch, before
the application can process `--unpack-dir`. On Windows, cache paths use the
extended-length path prefix so deep Hugging Face snapshots can be copied.

Keep the executable with its sibling `cache/`, any `cache-extras.zip`, and any
`artifacts/` assets. Pass `--unpack-dir` to move the writable cache.

Destination machine still needs a compatible NVIDIA driver to run the packaged application.

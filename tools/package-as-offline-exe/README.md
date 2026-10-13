# Package a FlashDreams application for offline use

`package_as_offline_exe.py` prepares an installed FlashDreams v2 application
and creates a raw folder containing an executable, its runtime, and offline cache.
Packaging is supported for applications running in `native-window` or `webrtc`
mode.

## Usage

From the repository root, use `:::` to separate this tool's options from the
complete `flashdreams-run-v2` command:

```bash
uv run python tools/package-as-offline-exe/package_as_offline_exe.py \
  --output artifacts/taxi-omnidreams-bundle \
  ::: flashdreams-run-v2 crazy-robotaxi-omnidreams-fast-perf-responsive \
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

If `--output` is omitted, the default is
`artifacts/<application-slug>-bundle`. The output directory must not already
exist.

The executable embeds Python bytecode, which is read directly without extraction.
`_internal/` contains the Python runtime, dependencies, and native libraries.
The standard-library archive is also read in place. `cache/` holds downloaded
model weights and prepared kernels. Preload may also produce sibling `artifacts/`
assets. The packager uses PyInstaller's directory mode and does not extract
runtime files to a temporary directory at launch or archive the cache.
Deploy the complete folder as raw files.
Pass `--unpack-dir` to change where the writable data cache is copied.


The packager runs initialization and one model block. Preparation findings are
written to `PREPARATION_ISSUES.txt` beside `INSTALLER_OUTPUT.txt`.

## Options

```text
--output PATH
    Destination bundle directory.

--working-dir PATH
    Defaults to the output directory (which must not already exist, ensuring nothing extra is copied if this option is used). Create if missing and copy its contents into the bundle before preload.

--noconsole
    Build without a console window on Windows (ignored on Linux).
    Console-free launchers write stdout and stderr to runtime.log in the writable runtime cache.

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
|-- _internal/
|-- cache/
|-- artifacts/             # when preload produces application assets
|-- INSTALLER_OUTPUT.txt
|-- PREPARATION_ISSUES.txt  # only when preload warnings are found
`-- README.md
```

The executable contains embedded Python bytecode. `_internal/` contains the
Python runtime, supporting source files, dependencies, native libraries, Ninja,
and packaged GPU assets.
Native extensions are compiled during packaging and reused at runtime. The
destination machine does not need CUDA Toolkit or Visual Studio Build Tools
for those extensions, but it still needs a compatible NVIDIA GPU and driver.
If a prepared extension is missing, rebuild the bundle on the packaging host.

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
directory; the command-line option takes precedence. Python and runtime DLLs
load directly from `_internal/`. Prepared C++, Triton, and PhysX extensions load from
the bundled `cache/`, even when the writable cache is relocated.
On Windows, cache paths use the
extended-length path prefix so deep Hugging Face snapshots can be copied.

Keep the executable with its sibling `_internal/`, `cache/`, and any
`artifacts/` assets. Pass `--unpack-dir` to move the writable data cache.

Destination machine still needs a compatible NVIDIA driver to run the packaged application.

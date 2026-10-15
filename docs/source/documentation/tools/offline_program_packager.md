---
title: 'Offline Program Packager'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

`tools/package-as-offline-exe/package_as_offline_exe.py` preloads an installed
FlashDreams v2 application and creates a host-native executable with its Python
runtime, dependencies, native libraries, and prepared caches in one raw folder.
The target machine does not need Python, `uv`, or network access.

The packager supports `native-window` and `webrtc` applications. It is not a
cross-compiler: build a Windows bundle on Windows and a Linux bundle on Linux.
The target still needs a compatible NVIDIA driver and GPU.

## Package an application

Run the tool from the repository root. Use `:::` to separate packager options
from the complete `flashdreams-run-v2` command:

```bash
uv run python tools/package-as-offline-exe/package_as_offline_exe.py \
  --output artifacts/taxi-omnidreams-bundle \
  ::: flashdreams-run-v2 crazy-robotaxi-omnidreams-fast-perf \
  --mode native-window --fullscreen
```

Do not pass `--preload-application`; the packager controls preparation and
validation. If `--output` is omitted, the destination is
`artifacts/<application-slug>-bundle`. The destination must not already exist.

## Options

| Option | Purpose |
| --- | --- |
| `--output PATH` | Set the destination bundle directory. |
| `--preload-timeout SECONDS` | Fail if initialization and preload exceed the limit. |
| `--skip-runtime-validation` | Initialize the application without validating one model block. |
| `--hidden-import MODULE` | Add a PyInstaller hidden import; repeat as needed. |
| `--collect-executable NAME` | Include an executable found on `PATH`; repeat as needed. |
| `--noconsole` | Build without a console window on Windows (ignored on Linux). `stdout` and `stderr` write to `%TEMP%\flashdreams\<application-slug>\runtime.log`. |
| 
| `--working-dir PATH` | Defaults to the output directory (which must not already exist. Creates dir if missing and copy its contents into the bundle before preload. |

Use `--help` for the authoritative option list:

```bash

uv run python tools/package-as-offline-exe/package_as_offline_exe.py --help

```

## Bundle contents

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

The executable embeds Python bytecode, which is read without extraction.
`_internal/` contains the executable runtime. `cache/` contains the resources and
build artifacts prepared during initialization. `INSTALLER_OUTPUT.txt` records
preload, validation, and PyInstaller output. `PREPARATION_ISSUES.txt` records
work that the application attempted after its preparation phase.

Distribute the complete folder, not only the executable: the launcher depends
on its sibling `_internal/` and `cache/` directories. The packager uses
PyInstaller's directory mode. Embedded bytecode and the standard-library archive
are read directly. Python and DLLs are not extracted into temporary storage at
launch, and the cache is not archived.
Deploy the complete folder as raw files.

## Runtime cache

The bundled cache is a read-only seed. On first launch, the application copies
it to a writable per-user cache:

- Windows: `%USERPROFILE%\flashdreams\unpacked\<application-slug>\cache`
- Linux: `~/flashdreams/unpacked/<application-slug>/cache`

Pass `--unpack-dir PATH` to use `PATH/<application-slug>/cache`, or set
`FLASHDREAMS_RUNTIME_CACHE_DIR` to choose the exact writable cache location.
The command-line option takes precedence; paths must be absolute.
Prepared C++, Triton, and PhysX extensions load from the original bundled `cache/`;
Python and runtime libraries load from `_internal/`.

Before distributing a bundle, test it on a clean target machine with networking
disabled and empty user caches. Also confirm that every bundled model, asset,
executable, and dependency permits redistribution.

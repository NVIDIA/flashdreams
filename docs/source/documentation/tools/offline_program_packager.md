---
title: 'Offline Program Packager'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

`tools/package-as-offline-exe/package_as_offline_exe.py` preloads an installed
FlashDreams v2 application and creates a host-native executable with its Python
runtime, dependencies, native libraries, and prepared caches in one folder.
The target machine does not need Python, `uv`, or network access.

The packager supports `native-window` and `webrtc` applications. It is not a
cross-compiler: build a Windows bundle on Windows and a Linux bundle on Linux.
The target still needs a compatible NVIDIA driver and GPU.

## Package an application

Run the tool from the repository root. Use `:::` to separate packager options
from the complete `flashdreams-run-v2` command:

```bash

uv run python tools/package-as-offline-exe/package_as_offline_exe.py \
  --output artifacts/interactive-drive-omnidreams-bundle \
  ::: flashdreams-run-v2 interactive-drive-omnidreams \
  --mode native-window

```

For a browser application, package its normal WebRTC command:

```bash

uv run python tools/package-as-offline-exe/package_as_offline_exe.py \
  --output artifacts/interactive-drive-omnidreams-webrtc-bundle \
  ::: flashdreams-run-v2 interactive-drive-omnidreams \
  --mode webrtc --host 0.0.0.0 --port 8089

```

Arguments before the runtime command's `--` belong to FlashDreams. Arguments
after it belong to the application and are embedded in the generated launcher:

```bash

uv run python tools/package-as-offline-exe/package_as_offline_exe.py \
  ::: flashdreams-run-v2 interactive-drive-omnidreams \
  --mode native-window --timeout 120 \
  -- --game-mode

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

Use `--help` for the authoritative option list:

```bash

uv run python tools/package-as-offline-exe/package_as_offline_exe.py --help

```

## Bundle contents

```text

<output>/
|-- <application-slug>[.exe]
|-- data/
|-- cache/
|-- INSTALLER_OUTPUT.txt
|-- PREPARATION_ISSUES.txt  # only when preload warnings are found
`-- README.md

```

`data/` contains the executable runtime. `cache/` contains the resources and
build artifacts prepared during initialization. `INSTALLER_OUTPUT.txt` records
preload, validation, and PyInstaller output. `PREPARATION_ISSUES.txt` records
work that the application attempted after its preparation phase.

Distribute the complete folder, not only the executable: the launcher depends
on its sibling `data/` and `cache/` directories.

## Runtime cache

The bundled cache is a read-only seed. On first launch, the application copies
it to a writable per-user cache:

- Windows: `%LOCALAPPDATA%\FlashDreams\<application-slug>\cache`
- Linux: `$XDG_CACHE_HOME/flashdreams/<application-slug>`, or
  `~/.cache/flashdreams/<application-slug>` when `XDG_CACHE_HOME` is unset

Set `FLASHDREAMS_RUNTIME_CACHE_DIR` to choose another writable cache location.

Before distributing a bundle, test it on a clean target machine with networking
disabled and empty user caches. Also confirm that every bundled model, asset,
executable, and dependency permits redistribution.

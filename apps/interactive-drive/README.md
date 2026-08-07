<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# interactive-drive app

Application-owned input semantics and chrome for the interactive driving demo.

`InteractiveDriveApplication` owns the window/event loop while a worker thread
owns one reusable model runtime. Each selected scene gets a fresh model session.
Compatibility requires the adapter's input mapping to consume `driver_command`
and its output schema to produce decoded RGB `VideoStepResult` values.

Panel widgets stack through the shared `PanelOverlay`, so the app composes the
chrome it wants instead of inheriting one class that owns every widget:

- `overlays/panel.py` reserves the chrome column and opens the shared row layout
- `overlays/header.py` scene and variant bars, post-processing toggle
- `overlays/speed.py` speed digit
- `overlays/controls.py` steering wheel and pedals
- `overlays/bev.py` top-down minimap, processed off the render thread
- `overlays/theme.py` shared palette

Widgets read their values through callables rather than holding engine state,
so nothing here imports a model and the package stays reusable.

## Running

List installed compatible adapters or select one directly:

```bash
uv run --package flashdreams-app-interactive-drive --extra all-models \
  flashdreams-interactive-drive --list-models
uv run --package flashdreams-app-interactive-drive --extra omnidreams \
  flashdreams-interactive-drive --model-id omnidreams
```

The OmniDreams compatibility command builds the same route by default:

```bash
uv run omnidreams-demo local-window
```

Use `--presenter-backend legacy` only for the temporary full-HUD compatibility
fallback. `Tab` and `Backspace` cycle discovered scene sessions, `R` resets the
current session, and `X` exits.

## Status

OmniDreams and LingBot register plug-compatible driving adapters. OmniDreams
keeps model-specific scene parsing, vehicle simulation, HD-map rendering, and
pipeline state inside its integration. The old interactive-drive loop,
presenters, and bridge remain only for explicit legacy fallback until manual
feature-parity checks are complete.

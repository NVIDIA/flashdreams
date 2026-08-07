<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# interactive-drive app

Chrome for the interactive driving demo, drawn over the shared local-window
presenter in `flashdreams.serving.presentation`.

Each widget is a `HudOverlay` and they stack through `CompositeOverlay`, so a
demo composes the chrome it wants instead of inheriting one class that owns
every widget:

- `overlays/panel.py` reserves the chrome column and opens the shared row layout
- `overlays/header.py` scene and variant bars, post-processing toggle
- `overlays/speed.py` speed digit
- `overlays/controls.py` steering wheel and pedals
- `overlays/bev.py` top-down minimap, processed off the render thread
- `overlays/theme.py` shared palette

Widgets read their values through callables rather than holding engine state,
so nothing here imports a model and the package stays reusable.

## Running

The demo still launches through the OmniDreams integration, which owns the
engine:

```bash
uv run --package flashdreams-omnidreams \
  interactive-drive --synthetic-scene --presenter-backend local-window
```

## Status

This is the first piece of `<ROOT>/apps`. The engine — main loop, ego-vehicle
simulation, scene loading, and the presenter bridge — still lives in
`integrations/omnidreams/omnidreams/interactive_drive/` and moves here as it
stops depending on OmniDreams specifics. Until then the integration imports
this package, not the other way round.

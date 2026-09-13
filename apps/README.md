<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Reusable applications built on the v2 API. Each directory is a standalone
package that depends on `flashdreams` and stays model-agnostic: an app runs
against a stub network, and binding a real model is the adapter's job, in
`integrations_v2/<model>/apps/<demo>/adapter.py`.

## The layout

```text
apps/<app_slug>/
  <app_slug>/          # the app implementation
  tests/               # validate the app against a stub, not a real model
  pyproject.toml
  README.md            # purpose, controls, example command line, options
```

Tests live in `apps/<app_slug>/tests/`, beside the package rather than inside
it, and must not import from `integrations_v2/`. A check that needs a real
model or its adapter belongs in `integrations_v2/<model>/tests/`.

## What is here

- `t2v` — text-to-video: prompt in, frames out. The reference app to copy.
- `cam2v` — interactive camera-to-video.
- `action2v` — world models driven by direct action input.
- `v2v` — video-to-video, used for super-resolution.
- `interactive_drive` — the interactive driving demo, keyboard and wheel input.
- `crazy_robotaxi` — the driving game built on `omnidreams_game_engine`.
- `omnidreams_game_engine` — reusable simulation, authored-map, physics, and
  conditioning components. It owns no runtime loop, so it is a library the
  other apps build on rather than something you launch.

`omnidreams_game_engine` is not the OmniDreams model. That is
`integrations_v2/omnidreams`. The names are close; the packages are unrelated.

Each app's own README covers how to launch it and what the controls are. To
bind a model to one of these, read `integrations_v2/README.md`.

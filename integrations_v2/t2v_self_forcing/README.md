<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Self-Forcing text-to-video

The first real model on the v2 API. Self-Forcing distilled Wan 2.1 1.3B, which
the `flashdreams-self-forcing` package already configures for the v1 runner; this
package is the application around it and holds no model code of its own.

The shared command line, the `HF_TOKEN` first run, and how these integrations are
built and tested are in the
[t2v guide](../../flashdreams/flashdreams/t2v_v2/README.md).

```bash
flashdreams-run-v2 t2v-self-forcing --output-path clip.mp4 \
    -- --prompt "A cat surfing" --total-blocks 7 --no-compile
```

## The model

Streaming, so `--total-blocks` is how long the clip is. The first block decodes 9
frames and every block after it 12, at 16 frames per second, so seven blocks is
about four and a half seconds.

832x480, `tchw`. It overrides none of the `T2VApplication` hooks.

## Tests

```bash
uv sync --package flashdreams-t2v-self-forcing --group test --inexact
uv run --no-sync pytest integrations_v2/t2v_self_forcing -m ci_cpu -v
```

Alongside this integration's own defaults and compile override, these carry the
one stand-in run that reaches a real MP4 on behalf of all five integrations, each
being the same factory over the same shared layer.

```bash
T2V_SELF_FORCING_REAL_MODEL_RUN=1 uv run --no-sync pytest \
    integrations_v2/t2v_self_forcing -m ci_gpu -s --basetemp="$HOME/t2v-out"
vlc "$HOME"/t2v-out/*current/clip.mp4
```

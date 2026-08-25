<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Wan 2.1 text-to-video

Wan 2.1 1.3B at 480p, which the `flashdreams-wan21` package already configures
for the v1 runner; this package is the application around it and holds no model
code of its own.

The shared command line, the `HF_TOKEN` first run, and how these integrations are
built and tested are in the
[t2v guide](../../flashdreams/flashdreams/t2v_v2/README.md).

```bash
flashdreams-run-v2 t2v-wan21 --output-path clip.mp4 \
    -- --prompt "A cat surfing" --no-compile
```

## The model

Bidirectional rather than streaming: it attends over the whole clip at once and
generates it in a single block. A run is therefore one step, and the clip is
however long the checkpoint generates rather than however long it is asked for.

`--total-blocks` is 1 and has to be, so this integration overrides
`_validate_total_blocks` to refuse a rollout rather than quietly generate a
second, unrelated clip. One block decodes 81 frames at 16 frames per second, so
a clip is about five seconds.

832x480, `tchw`. The rollout length is the one default this package states rather
than reads, a runner config for a model that does not roll out carrying no block
count.

## Tests

```bash
uv sync --package flashdreams-t2v-wan21 --group test --inexact
uv run --no-sync pytest integrations_v2/t2v_wan21 -m ci_cpu -v
```

```bash
T2V_WAN21_REAL_MODEL_RUN=1 uv run --no-sync pytest \
    integrations_v2/t2v_wan21 -m ci_gpu -s --basetemp="$HOME/t2v-out"
vlc "$HOME"/t2v-out/*current/clip.mp4
```

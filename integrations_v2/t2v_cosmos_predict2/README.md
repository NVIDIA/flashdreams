<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Cosmos Predict2 text-to-video

Cosmos Predict2 2B at 720p, which the `flashdreams-cosmos-predict2` package
already configures for the v1 runner; this package is the application around it
and holds no model code of its own.

The shared command line, the `HF_TOKEN` first run, and how these integrations are
built and tested are in the
[t2v guide](../../flashdreams/flashdreams/t2v_v2/README.md).

```bash
flashdreams-run-v2 t2v-cosmos-predict2 --output-path clip.mp4 \
    -- --prompt "A cat surfing" --no-compile
```

## The model

Like Wan 2.1 and unlike the streaming models here, this one generates its whole
clip in a single block, so a run is one step. It is also the largest and slowest
model here.

`--total-blocks` is 1 and has to be, so this integration overrides
`_validate_total_blocks` to refuse a rollout. One block decodes 93 frames at 16
frames per second, so a clip is about six seconds.

1280x720, `tchw`, and the only 720p model of the five. It plays back at 16
frames per second like the rest, but it is generating four times the pixels per
frame, so the rate it generates them at is not comparable with theirs.

## Tests

```bash
uv sync --package flashdreams-t2v-cosmos-predict2 --group test --inexact
uv run --no-sync pytest integrations_v2/t2v_cosmos_predict2 -m ci_cpu -v
```

```bash
T2V_COSMOS_PREDICT2_REAL_MODEL_RUN=1 uv run --no-sync pytest \
    integrations_v2/t2v_cosmos_predict2 -m ci_gpu -s --basetemp="$HOME/t2v-out"
vlc "$HOME"/t2v-out/*current/clip.mp4
```

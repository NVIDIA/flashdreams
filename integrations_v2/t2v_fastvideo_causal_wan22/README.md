<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# FastVideo CausalWan 2.2 text-to-video

FastVideo CausalWan 2.2 14B, which the `flashdreams-fastvideo-causal-wan22`
package already configures for the v1 runner; this package is the application
around it and holds no model code of its own.

The shared command line, the `HF_TOKEN` first run, and how these integrations are
built and tested are in the
[t2v guide](../../flashdreams/flashdreams/t2v_v2/README.md).

```bash
flashdreams-run-v2 t2v-fastvideo-causal-wan22 --output-path clip.mp4 \
    -- --prompt "A cat surfing" --total-blocks 7 --no-compile
```

## The model

Streaming, so `--total-blocks` is how long the clip is. The first block decodes 9
frames and every block after it 12, at 16 frames per second, so seven blocks is
about four and a half seconds.

832x480, `tchw`. It denoises with two transformers rather than one, a high-noise
branch and a low-noise branch, so it holds two 14B checkpoints and wants a GPU to
match. That is also why it overrides `_apply_compile_override`: the shared
override reaches one branch and would leave the other out of step. `--no-compile`
is worth doubly as much here, compilation costing minutes per transformer.

Both checkpoints download on the first run, which is a large download even by the
standards of the other models here.

Image-to-video is not wired up in the v1 package this wraps, so there is nothing
here for it either.

## Tests

```bash
uv sync --package flashdreams-t2v-fastvideo-causal-wan22 --group test --inexact
uv run --no-sync pytest integrations_v2/t2v_fastvideo_causal_wan22 -m ci_cpu -v
```

Those include the check that `--no-compile` reaches both transformers.

```bash
T2V_FASTVIDEO_CAUSAL_WAN22_REAL_MODEL_RUN=1 uv run --no-sync pytest \
    integrations_v2/t2v_fastvideo_causal_wan22 -m ci_gpu -s \
    --basetemp="$HOME/t2v-out"
vlc "$HOME"/t2v-out/*current/clip.mp4
```

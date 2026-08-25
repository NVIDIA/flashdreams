<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Causal-Forcing text-to-video

Causal-Forcing Wan 2.1 1.3B in its chunkwise configuration, which the
`flashdreams-causal-forcing` package already configures for the v1 runner; this
package is the application around it and holds no model code of its own.

The shared command line, the `HF_TOKEN` first run, and how these integrations are
built and tested are in the
[t2v guide](../../flashdreams/flashdreams/t2v_v2/README.md).

```bash
flashdreams-run-v2 t2v-causal-forcing --output-path clip.mp4 \
    -- --prompt "A cat surfing" --total-blocks 7 --no-compile
```

## The model

Streaming, so `--total-blocks` is how long the clip is. The first block decodes 9
frames and every block after it 12, at 16 frames per second, so seven blocks is
about four and a half seconds.

832x480, `tchw`. It overrides none of the `T2VApplication` hooks.

The framewise configuration this integration also ships generates one latent
frame a block rather than three. It is not what this application runs, since a
block of one frame is a great deal of overhead per frame.

## Tests

```bash
uv sync --package flashdreams-t2v-causal-forcing --group test --inexact
uv run --no-sync pytest integrations_v2/t2v_causal_forcing -m ci_cpu -v
```

```bash
T2V_CAUSAL_FORCING_REAL_MODEL_RUN=1 uv run --no-sync pytest \
    integrations_v2/t2v_causal_forcing -m ci_gpu -s --basetemp="$HOME/t2v-out"
vlc "$HOME"/t2v-out/*current/clip.mp4
```

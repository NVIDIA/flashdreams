<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Self-Forcing text-to-video

The first real model on the v2 API. A prompt goes in, a clip comes out as an
MP4 file. The model is Self-Forcing distilled Wan 2.1 1.3B, which the
`flashdreams-self-forcing` package already configures for the v1 runner; this
package is the application around it and holds no model code of its own.

## Generate a clip

```bash
flashdreams-run-v2 t2v-self-forcing --output-path clip.mp4 \
    -- --prompt "A cat surfing" --total-blocks 7 --no-compile
```

Arguments after `--` go to the application, and
`flashdreams-run-v2 t2v-self-forcing -- --help` lists them.

`--total-blocks` is how many autoregressive blocks to generate, and the run ends
when the session has generated them. The first block decodes 9 frames and every
block after it 12, at 16 frames per second, so seven blocks is about four and a
half seconds.

`--no-compile` is worth it for a short clip. Compilation is on in the model's
own config; it costs minutes on the first run and saves milliseconds a block.

## What a run looks like in code

```python
from flashdreams.runtime_v2.mp4_client_window import Mp4ClientWindow
from flashdreams.runtime_v2.session_runner import run_session
from t2v_self_forcing import SelfForcingT2VApplication

app = SelfForcingT2VApplication()
app.init(["--prompt", "A cat surfing", "--total-blocks", "7"])
try:
    run_session(
        app.create_session(app.session_desc()), Mp4ClientWindow("clip.mp4")
    )
finally:
    app.close()
```

## What it generates

832x480 frames at 16fps, laid out `tchw`, as `[-1, 1]` floats on the GPU. Those
numbers are the checkpoint's and are not written down here: this package is a
factory over
[`T2VApplication`](../../flashdreams/flashdreams/t2v_v2/README.md), which reads
them off the runner config the `flashdreams-self-forcing` package already
ships. `app.session_desc()` is what they add up to.

Something else can be asked for, with `--pixel-width`, `--pixel-height`, and
`--fps` before the `--`, since they describe the session rather than the model.
A session refuses a description it cannot honour rather than quietly generating
something else, so each dimension has to be a multiple of 8, which is what one
latent covers.

## First run

The checkpoint is fetched from Hugging Face the first time, which is tens of
gigabytes including the text encoder, so set a token and expect to wait:

```bash
export HF_TOKEN=<your-hf-token>
```

## Tests

```bash
uv sync --package flashdreams-t2v-self-forcing --group test --inexact
uv run --no-sync pytest integrations_v2/t2v_self_forcing -m ci_cpu -v
```

Those run against the stand-in model in `flashdreams.t2v_v2.testing`, and cover
what is specific to this integration: that the defaults come off the runner
config, and that turning compilation off reaches the setting this model's config
keeps. They also cover a run through an integration to a real MP4, on behalf of
all five of them, since each is the same factory over the same shared layer. How
a text-to-video application behaves in general, and how one rollout is driven,
are covered once in `flashdreams/test_v2`.

The run that uses the real model needs a GPU, and skips unless its environment
variable is set, so asking for it is deliberate:

```bash
T2V_SELF_FORCING_REAL_MODEL_RUN=1 uv run --no-sync pytest \
    integrations_v2/t2v_self_forcing -m ci_gpu -s --basetemp="$HOME/t2v-out"
vlc "$HOME"/t2v-out/*current/clip.mp4
```

Both go through `flashdreams.t2v_v2.testing`: the run above through
`check_t2v_model_impl`, and this one through
`check_real_model_generates_a_clip`, which wraps it with the numbers and checks
every integration's clip is measured against.

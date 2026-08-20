<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Cosmos Predict2 text-to-video

A prompt goes in, a clip comes out as an MP4 file. The model is Cosmos Predict2
2B at 720p, which the `flashdreams-cosmos-predict2` package already configures
for the v1 runner; this package is the application around it and holds no model
code of its own.

Like Wan 2.1 and unlike the streaming models here, this one generates its whole
clip in a single block, so a run is one step and the clip is however long the
checkpoint generates.

## Generate a clip

```bash
flashdreams-run-v2 t2v-cosmos-predict2 --output-path clip.mp4 \
    -- --prompt "A cat surfing" --no-compile
```

Arguments after `--` go to the application, and
`flashdreams-run-v2 t2v-cosmos-predict2 -- --help` lists them.

`--total-blocks` is 1 and has to be, and asking for more is refused rather than
quietly generating a second clip: a second block would not continue the first.
One block decodes 93 frames at 16 frames per second, so a clip is about six
seconds.

`--no-compile` is worth it for a single clip. Compilation is on in the model's
own config; it costs minutes on the first run and saves milliseconds, which is
the wrong trade for one block.

This is the largest and slowest model here at 720p, and it denoises for many
more steps within its one block than the distilled models do.

## What a run looks like in code

```python
from flashdreams.runtime_v2.mp4_client_window import Mp4ClientWindow
from flashdreams.runtime_v2.session_runner import run_session
from t2v_cosmos_predict2 import CosmosPredict2T2VApplication

app = CosmosPredict2T2VApplication()
app.init(["--prompt", "A cat surfing"])
try:
    run_session(
        app.create_session(app.session_desc()), Mp4ClientWindow("clip.mp4")
    )
finally:
    app.close()
```

## What it generates

1280x720 frames at 16fps, laid out `tchw`, as `[-1, 1]` floats on the GPU. Those
numbers are the checkpoint's and are not written down here: this package is a
factory over
[`T2VApplication`](../../flashdreams/flashdreams/t2v_v2/README.md), which reads
them off the runner config the `flashdreams-cosmos-predict2` package already
ships, block count included. `app.session_desc()` is what they add up to.

Something else can be asked for, with `--pixel-width`, `--pixel-height`, and
`--fps`. A session refuses a description it cannot honour rather than quietly
generating something else, so each dimension has to be a multiple of 8, which
is what one latent covers.

## First run

The checkpoint is fetched from Hugging Face the first time, which is tens of
gigabytes including the text encoder, so set a token and expect to wait:

```bash
export HF_TOKEN=<your-hf-token>
```

## Tests

```bash
uv sync --package flashdreams-t2v-cosmos-predict2 --group test --inexact
uv run --no-sync pytest integrations_v2/t2v_cosmos_predict2 -m ci_cpu -v
```

Those run against the stand-in model in `flashdreams.t2v_v2.testing`, and cover
what is specific to this integration: that the defaults come off the runner
config, and that a rollout is refused. How a text-to-video application behaves
in general is covered once, in `flashdreams/test_v2`, and a run of one reaching a
file is covered once in the Self-Forcing integration, since every integration
here is the same factory over the same shared layer.

The run that uses the real model needs a GPU, and skips unless its environment
variable is set, so asking for it is deliberate:

```bash
T2V_COSMOS_PREDICT2_REAL_MODEL_RUN=1 uv run --no-sync pytest \
    integrations_v2/t2v_cosmos_predict2 -m ci_gpu -s --basetemp="$HOME/t2v-out"
vlc "$HOME"/t2v-out/*current/clip.mp4
```

That run goes through
`flashdreams.t2v_v2.testing.check_real_model_generates_a_clip`, which is where
every integration's clip and the checks made of it come from.

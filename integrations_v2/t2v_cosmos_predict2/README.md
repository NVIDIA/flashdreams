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
flashdreams-run-v2 t2v-cosmos-predict2 --output-path clip.mp4 --steps 1 \
    -- --prompt "A cat surfing" --no-compile
```

Arguments after `--` go to the application, and
`flashdreams-run-v2 t2v-cosmos-predict2 -- --help` lists them.

`--steps` has to be 1, and asking for more is refused rather than quietly
generating a second clip: a second block would not continue the first. One block
decodes 93 frames at 16 frames per second, so a clip is about six seconds.

`--no-compile` is worth it for a single clip. Compilation is on in the model's
own config; it costs minutes on the first run and saves milliseconds, which is
the wrong trade for one block.

This is the largest and slowest model here at 720p, and it denoises for many
more steps within its one block than the distilled models do.

## What a run looks like in code

```python
from flashdreams.runtime_v2.batch_runner import run_batch
from flashdreams.runtime_v2.mp4_output_sink import Mp4OutputSink
from t2v_cosmos_predict2 import CosmosPredict2T2VApplication

app = CosmosPredict2T2VApplication()
app.init(["--prompt", "A cat surfing"])
try:
    run_batch(
        app.create_session(app.session_desc()),
        Mp4OutputSink("clip.mp4"),
        steps=1,
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
config, that a rollout is refused, and that a run reaches a file. How a
text-to-video application behaves in general is covered once, in
`flashdreams/test_v2`.

The run that uses the real model needs a GPU, and skips unless its environment
variable is set, so asking for it is deliberate. It also needs torchvision,
which this model's text encoder loads through and which no package here
declares, so sync a CUDA group as well as the test one:

```bash
uv sync --package flashdreams-t2v-cosmos-predict2 --group test --group cuda13 --inexact
T2V_COSMOS_PREDICT2_REAL_MODEL_RUN=1 uv run --no-sync pytest \
    integrations_v2/t2v_cosmos_predict2 -m ci_gpu -s --basetemp="$HOME/t2v-out"
vlc "$HOME"/t2v-out/*/clip.mp4
```

Without torchvision the run skips saying so, rather than loading the model for a
while and then failing inside transformers.

Both go through `flashdreams.t2v_v2.testing.check_t2v_model_impl`,
the shared check a text-to-video integration runs to cover the batch path.

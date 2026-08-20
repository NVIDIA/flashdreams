<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Wan 2.1 text-to-video

A prompt goes in, a clip comes out as an MP4 file. The model is Wan 2.1 1.3B at
480p, which the `flashdreams-wan21` package already configures for the v1
runner; this package is the application around it and holds no model code of its
own.

Unlike the streaming models here, this one is bidirectional: it attends over the
whole clip at once and generates it in a single block. A run is therefore one
step, and the clip is however long the checkpoint generates rather than however
long it is asked for.

## Generate a clip

```bash
flashdreams-run-v2 t2v-wan21 --output-path clip.mp4 \
    -- --prompt "A cat surfing" --no-compile
```

Arguments after `--` go to the application, and
`flashdreams-run-v2 t2v-wan21 -- --help` lists them.

`--total-blocks` is 1 and has to be, and asking for more is refused rather than
quietly generating a second clip: a second block would not continue the first.
One block decodes 81 frames at 16 frames per second, so a clip is about five
seconds.

`--no-compile` is worth it for a single clip. Compilation is on by default; it
costs minutes on the first run and saves milliseconds, which is the wrong trade
for one block.

## What a run looks like in code

```python
from flashdreams.runtime_v2.mp4_client_window import Mp4ClientWindow
from flashdreams.runtime_v2.session_runner import run_session
from t2v_wan21 import Wan21T2VApplication

app = Wan21T2VApplication()
app.init(["--prompt", "A cat surfing"])
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
them off the runner config the `flashdreams-wan21` package already ships.
`app.session_desc()` is what they add up to.

The rollout length is the one thing this package states rather than reads. A
runner config for a model that does not roll out carries no block count, so
`--total-blocks` defaults to 1 here and refuses anything else.

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
uv sync --package flashdreams-t2v-wan21 --group test --inexact
uv run --no-sync pytest integrations_v2/t2v_wan21 -m ci_cpu -v
```

Those run against the stand-in model in `flashdreams.t2v_v2.testing`, and cover
what is specific to this integration: that the defaults come off the runner
config, that a rollout is refused, and that a run reaches a file. How a
text-to-video application behaves in general is covered once, in
`flashdreams/test_v2`.

The run that uses the real model needs a GPU, and skips unless its environment
variable is set, so asking for it is deliberate:

```bash
T2V_WAN21_REAL_MODEL_RUN=1 uv run --no-sync pytest \
    integrations_v2/t2v_wan21 -m ci_gpu -s --basetemp="$HOME/t2v-out"
vlc "$HOME"/t2v-out/*current/clip.mp4
```

Both go through `flashdreams.t2v_v2.testing.check_t2v_model_impl`,
the shared check a text-to-video integration runs to cover the batch path.

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Self-Forcing text-to-video

The first real model on the v2 API. A prompt goes in, a clip comes out as an
MP4 file. The model is Self-Forcing distilled Wan 2.1 1.3B, which the
`flashdreams-self-forcing` package already configures for the v1 runner; this
package is the application around it and holds no model code of its own.

## What a run looks like

```python
from flashdreams.runtime_v2.batch_runner import run_batch
from flashdreams.runtime_v2.mp4_output_sink import Mp4OutputSink
from t2v_self_forcing import SelfForcingT2VApplication, default_session_desc

app = SelfForcingT2VApplication()
app.init(["--prompt", "A cat surfing"])
try:
    run_batch(
        app.create_session(default_session_desc()),
        Mp4OutputSink("clip.mp4"),
        steps=7,
    )
finally:
    app.close()
```

`steps` is how many autoregressive blocks to generate. The model streams, so a
run is as long as the caller asks for: the first block decodes 9 frames and
every block after it 12, at 16 frames per second, so seven blocks is about four
and a half seconds.

Arguments are `--prompt` (required), `--device` (default `cuda`), and
`--compile` / `--no-compile`. Compilation is on in the model's own config: it
costs minutes on the first run and saves milliseconds a block, so a short clip
is quicker without it.

## What it generates

832x480 frames, laid out `tchw`, as `[-1, 1]` floats on the GPU. That is what
`default_session_desc()` describes, and a session refuses a description it
cannot honour rather than quietly generating something else. Frame sizes other
than the trained one are accepted as long as each dimension is a multiple of 8,
which is what one latent covers.

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

Those run against a stand-in model, and cover everything except what the real
one generates. The run that uses the real model needs a GPU, and skips unless
its environment variable is set, so asking for it is deliberate:

```bash
T2V_SELF_FORCING_REAL_MODEL_RUN=1 uv run --no-sync pytest \
    integrations_v2/t2v_self_forcing -m ci_gpu -s --basetemp="$HOME/t2v-out"
vlc "$HOME"/t2v-out/*current/clip.mp4
```

Both go through `flashdreams.testing_v2.t2v_conformance.check_t2v_model_impl`,
the shared check a text-to-video integration runs to cover the batch path.

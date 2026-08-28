<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Benchmarking models on the v2 API

[`configs/v2_model_benchmarks.json`](../../../configs/v2_model_benchmarks.json)
runs every model on the v2 API at the same seed, writing a clip and its runtime
metrics for each. Use it to compare the models against each other, or a change
against a baseline of a previous run. The text-to-video models share one prompt
so their clips can be put side by side; a camera-to-video model is conditioned
on a first frame that its prompt belongs to, so it is compared against runs of
itself rather than against the text-to-video clips.

The v1 demo suites that run through `flashdreams-run` are a separate workflow,
in [the local benchmarks guide](../../../docs/source/developer_guides/local_benchmarks.rst).

## Set up

```bash
uv sync --package flashdreams --group cuda13 --extra runners --inexact
export HF_TOKEN=<your-hf-token>
```

That is the harness's own environment; each scenario syncs its model when it
runs. Every scenario needs a GPU and host FFmpeg, and the first run of a model
downloads tens of gigabytes.

```bash
uv run --no-sync flashdreams-benchmark --list-scenarios \
    --scenario-file configs/v2_model_benchmarks.json
```

Three streaming text-to-video models and LingBot World have a ten-second
scenario and a one-minute one; Wan 2.1 and Cosmos Predict2 generate their whole
clip in one block, so each has one scenario at the length it does generate.

LingBot World runs from its bundled example conditioning with the camera overlay
off, so the clip holds the model's own frames rather than live timing text drawn
over them. An MP4 run has nobody to send it keys, so the camera holds still while
the scene moves: the throughput is measured at a resting camera.

## Score the clips with PAI-Bench, if you want scores

Scoring runs as a scenario generates, so this is a decision to make before a
sweep rather than after. PAI-Bench is evaluator-only and carries its own license
terms, so it goes in a separate environment you have reviewed for your use case:

```bash
uv venv --python 3.12 ~/.venvs/flashdreams-paibench
export PAI_BENCH_PYTHON="$HOME/.venvs/flashdreams-paibench/bin/python"

uv pip install --python "$PAI_BENCH_PYTHON" \
    torch torchvision \
    opencv-python-headless \
    omegaconf \
    openai-clip \
    "pyiqa>=0.1.15,<0.1.16" \
    "setuptools<81"

"$PAI_BENCH_PYTHON" -c "import clip, cv2, omegaconf, pyiqa, torch; print('PAI-Bench environment OK')"
```

Dropping `--quality-profile` and `--pai-bench-python` from the commands below is
how a run skips scoring.

## Check the sweep without running it

```bash
uv run --no-sync flashdreams-benchmark --dry-run \
    --scenario-file configs/v2_model_benchmarks.json \
    --scenario t2v-self-forcing-quality-10s \
    --output-dir artifacts/benchmarks/v2-model-dry-run
```

Renders the commands and the report skeleton without launching anything, which
is the cheap way to confirm a change to the scenario file.

## Generate the baseline

```bash
uv run --no-sync flashdreams-benchmark \
    --scenario-file configs/v2_model_benchmarks.json \
    --scenario t2v-self-forcing-quality-10s \
    --scenario t2v-causal-forcing-quality-10s \
    --scenario t2v-fastvideo-causal-wan22-quality-10s \
    --scenario t2v-wan21-native-clip \
    --scenario t2v-cosmos-predict2-native-clip \
    --scenario cam2v-lingbot-quality-10s \
    --scenario t2v-self-forcing-one-minute \
    --scenario t2v-causal-forcing-one-minute \
    --scenario t2v-fastvideo-causal-wan22-one-minute \
    --scenario cam2v-lingbot-one-minute \
    --keep-going \
    --output-dir artifacts/benchmarks/v2-model-baseline \
    --quality-profile pai-bench-long \
    --pai-bench-python "$PAI_BENCH_PYTHON"
```

Scenarios are named rather than using `--all`, which would also pull in the v1
scenarios, and the short clips come first so there is something to watch before
the one-minute rollouts start. `--keep-going` stops one model failing from
abandoning the models after it. `artifacts/` is git-ignored.

## Compare a candidate

The same command, with a directory of its own and the baseline to compare
against:

```bash
uv run --no-sync flashdreams-benchmark \
    --scenario-file configs/v2_model_benchmarks.json \
    --scenario t2v-self-forcing-quality-10s \
    --scenario t2v-causal-forcing-quality-10s \
    --scenario t2v-fastvideo-causal-wan22-quality-10s \
    --scenario t2v-wan21-native-clip \
    --scenario t2v-cosmos-predict2-native-clip \
    --scenario cam2v-lingbot-quality-10s \
    --scenario t2v-self-forcing-one-minute \
    --scenario t2v-causal-forcing-one-minute \
    --scenario t2v-fastvideo-causal-wan22-one-minute \
    --scenario cam2v-lingbot-one-minute \
    --keep-going \
    --output-dir artifacts/benchmarks/v2-model-candidate \
    --quality-baseline-dir artifacts/benchmarks/v2-model-baseline \
    --quality-profile pai-bench-long \
    --pai-bench-python "$PAI_BENCH_PYTHON"
```

Clips are matched by scenario id, so a comparison only says something when both
runs used the same prompt, seed, and model configuration, which is why the suite
fixes all three. It is non-gating: the `quality_*` scores land in the report,
higher being better, without changing whether a scenario passed. The one-minute
scenarios skip the pixel comparison and report PAI-Bench and runtime alone,
a minute of autoregressive generation drifting enough that comparing pixels
reports noise.

## What a run writes

`report.html` with a page per model, `metrics.csv` beside `metrics.ndjson`, and
under `scenarios/<id>/` the MP4, the stats JSON, and `command.log`, which is
where that scenario's output goes so a long sweep can be watched without model
logs filling the terminal.

## Adding a model

Copy the scenarios closest to the new model's behaviour, point `--project` and
the application slug at its integration, and keep the prompt and the seed: a
comparison where each model gets its own prompt compares prompts. Keep
`--stats-path` too, which is what records the runtime metrics the report reads.
Tag a scenario `one-minute` or `pai-bench` for PAI-Bench to score it.

A model conditioned on more than a prompt cannot take the shared one, since the
prompt belongs with its first frame. Fix that conditioning instead, the way
LingBot World pins `--example-data --example-idx 0`, and the scenario still
compares against runs of itself even though it no longer lines up with the
text-to-video clips. Turn off anything that draws over the model output, or the
clip records the overlay's timing text as well as the frames.

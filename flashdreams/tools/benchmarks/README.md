<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Benchmarking models on the v2 API

[`configs/v2_model_benchmarks.json`](../../../configs/v2_model_benchmarks.json)
runs the same prompt and seed through every model on the v2 API, writing a clip
and its runtime metrics for each. Use it to compare the models against each
other, or a change against a baseline of a previous run.

[`configs/v2_webrtc_benchmarks.json`](../../../configs/v2_webrtc_benchmarks.json)
contains command-backed presentation benchmarks. Its Lingbot scenario runs a
process-isolated ABBA comparison with a loopback aiortc receiver, retains raw
model/window/WebRTC records, and applies explicit FPS, model-compute, and
publish-wait thresholds. Every run must also complete the configured model-step
count, drain its receiver tail, preserve strictly increasing timestamps, and
report zero sender drops and zero missing or extra receiver frames. WebRTC keeps
its normal capacity-two FIFO queue for unsent frames, evicts the oldest queued
frame on overflow, and records frames dropped when the receiver lags. A frame
already handed to aiortc is committed and excluded from that capacity.
Run it directly with:

```bash
uv run --project integrations_v2/cam2v_lingbot --no-sync \
  python -m tools.benchmarks.v2_webrtc_ab \
    --config configs/v2_webrtc_benchmarks.json \
    --benchmark cam2v-lingbot-hud-ab \
    --repo-root . \
    --output-dir artifacts/benchmarks/cam2v-lingbot-hud-ab
```

The resulting webrtc_ab.json is machine-readable, webrtc_ab.md is the
compact report, and `runs/<run-id>/` retains requests, raw events, and command
logs. Sender snapshots include synchronous write-side materialization and
queue state. Steady model FPS is the sum of measured output frames
divided by the sum of every measured model step's wall time; no measured step is
dropped from the rate. The ABBA variants compare the SlangPy UI presentation
path with the default blit path; they are not a pure overlay microbenchmark and
do not compare high-priority versus default-priority stream scheduling. Loopback
aiortc measures the server through decoder materialization; it does not include
browser DOM/display compositing or real-network behavior.
`runtime_presentation_publish_wait_s` is model-thread wait for presentation-manager
capacity, not CUDA composition or transfer time.

Each case starts a fresh Python worker, but the harness does not clear or
isolate persistent TorchInductor, Triton, or CUDA caches. Control those cache
directories outside the harness when cold-cache behavior matters.

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

Three streaming models have a ten-second scenario and a one-minute one; Wan 2.1
and Cosmos Predict2 generate their whole clip in one block, so each has one
scenario at the length it does generate.

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
    --scenario t2v-self-forcing-one-minute \
    --scenario t2v-causal-forcing-one-minute \
    --scenario t2v-fastvideo-causal-wan22-one-minute \
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
    --scenario t2v-self-forcing-one-minute \
    --scenario t2v-causal-forcing-one-minute \
    --scenario t2v-fastvideo-causal-wan22-one-minute \
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

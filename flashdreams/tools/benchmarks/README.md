<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Benchmarking models on the v2 API

Everything needed to run the shipped v2 model comparison: generating a baseline,
comparing a candidate against it, and scoring clips with PAI-Bench. The v1 demo
suites that run through `flashdreams-run` are a separate workflow, documented in
[the local benchmarks guide](../../../docs/source/developer_guides/local_benchmarks.rst),
and nothing here needs it.

The suite is
[`configs/v2_model_benchmarks.json`](../../../configs/v2_model_benchmarks.json),
a scenario file the harness loads with `--scenario-file`. Every model in it runs
the same prompt through `flashdreams-run-v2` at the size it was trained for,
seeded, so the clips can be put next to each other and the difference is the
model.

## What it runs

| Scenario | Blocks | Clip |
| --- | --- | --- |
| `t2v-self-forcing-quality-10s` | 14 | 165 frames, 10.3s |
| `t2v-causal-forcing-quality-10s` | 14 | 165 frames, 10.3s |
| `t2v-fastvideo-causal-wan22-quality-10s` | 14 | 165 frames, 10.3s |
| `t2v-wan21-native-clip` | 1 | 81 frames, 5.1s |
| `t2v-cosmos-predict2-native-clip` | 1 | 93 frames, 5.8s |
| `t2v-self-forcing-one-minute` | 81 | 969 frames, 60.6s |
| `t2v-causal-forcing-one-minute` | 81 | 969 frames, 60.6s |
| `t2v-fastvideo-causal-wan22-one-minute` | 81 | 969 frames, 60.6s |

The three streaming models generate as long a clip as they are asked for, so
each has two scenarios: ten seconds to look at, and a minute for PAI-Bench to
score. Wan 2.1 and Cosmos Predict2 generate their whole clip in one
bidirectional block and reach neither length, so each has one scenario at the
length it does generate. They are still tagged `pai-bench`, which is what makes
them eligible for scoring, because a short clip scored is worth more than a
model missing from the comparison.

## Set up the environment

Sync the harness's own environment, which is what the `--no-sync` in the
commands below then trusts. The models are not part of it: each scenario syncs
its own integration when it runs, so there is nothing to install per model here.

```bash
uv sync --package flashdreams --group cuda13 --extra runners --inexact
```

`runners` brings the MP4 and OpenCV dependencies the harness reads generated
clips through, and `--inexact` leaves the rest of a working tree alone rather
than pruning it to this one package.

Every scenario needs a GPU, host FFmpeg to write the MP4, and an `HF_TOKEN`,
and the first run of a model downloads its checkpoint, which for these is tens
of gigabytes.

Each scenario also runs `uv run --project integrations_v2/t2v_<model>`, and that
sync is exact: it uninstalls whatever is not in that model's closure. So the
environment churns between scenarios, and on a machine with a cold uv cache that
means downloads between them as well as at the start. It is also why a model
needing a package must declare it rather than have it synced in by hand, since
the next scenario in the sweep would prune it.

## Generate the baseline

A dry run first, which renders every command and writes the report skeleton
without launching anything. It is the cheap way to confirm a change to the
scenario file:

```bash
uv run --no-sync flashdreams-benchmark --dry-run \
    --scenario-file configs/v2_model_benchmarks.json \
    --scenario t2v-self-forcing-quality-10s \
    --output-dir artifacts/benchmarks/v2-model-dry-run
```

Then the sweep that becomes the baseline. Scenarios are named by id because
`--all` would also pull in the built-in v1 scenarios, and the short clips come
first so there is something to watch before the one-minute rollouts start:

```bash
export HF_TOKEN=<your-hf-token>
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
    --output-dir artifacts/benchmarks/v2-model-baseline
```

`--keep-going` is what stops one model failing from abandoning the models after
it. `--list-scenarios` with the same `--scenario-file` prints the ids and their
tags, which is the table above without reading the JSON. Without `--output-dir`
a run lands in `artifacts/benchmarks/<timestamp>`; naming it instead is what
makes it findable later as the baseline. Either way `artifacts/` is git-ignored.

## Compare a candidate

The same command, plus the baseline to compare against and a directory of its
own:

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
    --quality-baseline-dir artifacts/benchmarks/v2-model-baseline \
    --output-dir artifacts/benchmarks/v2-model-candidate
```

`--quality-baseline-dir` takes a previous run root, or a flat directory of
`<scenario-id>.mp4` files. Clips are matched by scenario id, so a comparison
only says something when both runs used the same prompt, seed, and model
configuration — which is why the suite fixes all three rather than leaving them
to the command line.

The comparison is non-gating: it does not change a scenario's pass or fail
status. It writes its numbers under
`scenarios/<id>/quality/baseline-clip-compare/` and surfaces them in the report,
where `quality_score` is a 0-1 blend of similarity to the baseline with
no-reference sanity checks, `quality_similarity_score` is closeness to the
baseline alone, `quality_visual_sanity_score` guards against blank, flat,
striped, or unstable clips, and `quality_temporal_score` is a frame-to-frame
stability proxy. Higher is better for all four. `quality_ssim_score`,
`quality_rmse`, `quality_mean_abs`, and `quality_psnr_db` support them, with
RMSE and mean absolute difference in 8-bit pixel units and PSNR in dB. As rough
bands for local debugging rather than pass/fail: RMSE below about 15 is a small
pixel-space difference, above about 40 is large visual drift, PSNR above about
30 dB is close, and below about 20 dB is a large difference.

The three one-minute scenarios set `quality_baseline_compare` to false, because a
minute of autoregressive generation drifts enough that a pixel comparison
reports noise. They still report runtime performance, PAI-Bench scores, and
review links to both clips. The five short scenarios compare the whole frame, so
`--quality-compare-region` does not need passing. `--quality-sample-count` and
`--quality-frame-indices` choose which frames are compared, and
`--quality-compute-flip` adds FLIP metrics when `flip-evaluator` is installed,
warning and carrying on when it is not.

## Score clips with PAI-Bench

PAI-Bench and its dependencies are not FlashDreams dependencies: they are
evaluator-only and carry their own license terms, so they go in a separate
environment you have reviewed for your use case. This is the one the local runs
used:

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

If the virtual environment already exists, `uv venv` asks before replacing it.

Then add the profile to a baseline or candidate run, selecting only the
scenarios it applies to: the three one-minute rollouts and the two single-block
clips. The profile ignores anything not tagged `one-minute` or `pai-bench`, so
including the 10-second scenarios costs their generation time and scores
nothing.

```bash
uv run --no-sync flashdreams-benchmark \
    --scenario-file configs/v2_model_benchmarks.json \
    --scenario t2v-wan21-native-clip \
    --scenario t2v-cosmos-predict2-native-clip \
    --scenario t2v-self-forcing-one-minute \
    --scenario t2v-causal-forcing-one-minute \
    --scenario t2v-fastvideo-causal-wan22-one-minute \
    --keep-going \
    --quality-profile pai-bench-long \
    --pai-bench-python "$PAI_BENCH_PYTHON" \
    --output-dir artifacts/benchmarks/v2-model-baseline
```

`pai-bench-long` splits each clip into segments, scores them with public
PAI-Bench-G, and reports a 0-100 `pai_bench_long_*` average. Its dimensions are
`aesthetic_quality`, `background_consistency`, `imaging_quality`, and
`motion_smoothness`; segments default to ten seconds, which
`--pai-bench-segment-duration-s` changes. `--quality-profile pai-bench-g` scores
the whole MP4 without segmenting and adds `overall_consistency` and
`subject_consistency`, the second of which can hit Torch Hub rate limits.
Passing `--pai-bench-dimension` narrows the set for local triage and changes
what is reported, so those results should not be compared against standard runs.

PAI-Bench itself stays an external checkout. The adapter clones
`https://github.com/SHI-Labs/physical-ai-bench.git` at the revision pinned in
[`pai_bench_profile.py`](pai_bench_profile.py) into
`.cache/flashdreams/evaluators/physical-ai-bench` when that path is missing, so
copying `artifacts/` around does not carry the evaluator with it.
`--pai-bench-root` points at a checkout of your own and `--no-pai-bench-fetch`
stops it fetching one that already exists.

The default `--pai-bench-runner local` runs the public entrypoint with the
evaluator Python and injects an OpenCV-backed `decord` shim, which is what
avoids the upstream `decord` wheel failing to resolve on aarch64. Use
`--pai-bench-runner upstream` only when you deliberately want the older
execution mode on a machine whose PAI-Bench environment works.

Before launching, the adapter imports the requested dimension modules with the
evaluator Python and logs the result to
`scenarios/<id>/quality/<profile>/pai_bench_preflight.log`. A missing `clip`
import there means `--pai-bench-python` is not pointing at the environment built
above. Staged and segmented copies of the MP4 are deleted when scoring
finishes; `--pai-bench-keep-staged-videos` keeps them when debugging what the
evaluator was fed.

## What a run writes

A run directory holds `report.html`, a `reports/<model>.html` page for each
model, `manifest.json`, `environment.json`, and `metrics.ndjson` alongside
`metrics.csv`. Under `scenarios/<id>/` are the MP4, the stats JSON, and
`command.log`, which is where a scenario's stdout and stderr go so a long run
can be watched without model logs filling the terminal.

The generated frame rate in the report is post-warmup generated frames over
post-warmup runtime, not the rate the MP4 plays at. The streaming scenarios
declare one warmup block so the first block's compile and cache costs stay out
of that figure; the single-block scenarios declare none, having only the one.

## What a scenario command does

The 10-second and single-block scenarios run through
[`strict_run.py`](strict_run.py) with `--entrypoint flashdreams-run-v2`, which
launches the v2 CLI under deterministic CUDA and PyTorch settings. The
one-minute scenarios call `flashdreams-run-v2` directly, because determinism
costs time and time is what those runs measure; they are seeded all the same, so
a clip is still repeatable. They also allow themselves six hours rather than
two.

Every scenario passes `--stats-path` as well as `--output-path`. That is what
gets [`BenchmarkStatsOutputSink`](../../flashdreams/runtime_v2/benchmark_stats_sink.py)
into the run beside the MP4 writer, recording each step's own measurements in
the artifact the harness reads. Without it a scenario generates a clip that says
nothing about what it cost. `--seed 1` is passed to every model, and reaches the
model's config rather than the session; see
[the t2v layer's README](../../flashdreams/t2v_v2/README.md).

## Adding a model

Copy the pair of scenarios closest to the model's behaviour, point `--project`
and the application slug at the new integration, and keep the prompt, the seed,
and the frame arithmetic in the description. The prompt is shared deliberately:
a comparison where each model gets its own prompt compares prompts. Tag a
scenario `one-minute` or `pai-bench` if PAI-Bench should score it. The file is
where the rest of the models go as they move onto the v2 API, not only the
text-to-video ones.

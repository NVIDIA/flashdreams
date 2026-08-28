<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# `omnidreams`

Omnidreams integration package for `flashdreams`.

## Hugging Face assets

Omnidreams resolves public Omni Dreams assets from the `nvidia` Hugging Face
org:

- `nvidia/omni-dreams-models` for checkpoints.
- `nvidia/omni-dreams-samples` for bundled example data.
- `nvidia/omni-dreams-scenes` for WebRTC scenes.

Set `HF_TOKEN` to a token with access to these repos before running or
importing FlashDreams:

```bash
export HF_TOKEN=<YOUR-HF-TOKEN>
```

## Run batch evaluation

The `omnidreams-eval` CLI automates a fixed-split evaluation flow for
OmniDreams scene batches:

1. Discover Hugging Face scene assets and write a JSONL manifest.
2. Plan byte- or count-capped batches.
3. Stage one batch into local scratch storage.
4. Run FlashDreams generation for the staged cases.
5. Validate generated artifacts and runner logs.
6. Stage/run DrivingGen FVD-lite and WorldLens consistency evaluators.
7. Write a JSON and Markdown summary report.
8. Optionally compare the summary against a checked-in metric baseline.

The high-level workflow is:

```bash
RUN=/trees/$USER/od-runs/od-26.01
SCRATCH=/local_nvme/$USER/omnidreams-eval-scratch
BASELINE=integrations_v2/omnidreams/impl/eval_baselines/od-26.01-worldlens-40-v1.json

uv run --package flashdreams-omnidreams omnidreams-eval discover \
  --output "$RUN/manifest.jsonl"

uv run --package flashdreams-omnidreams omnidreams-eval plan-batches \
  --manifest "$RUN/manifest.jsonl" \
  --output "$RUN/batches.json" \
  --batch-size 20

uv run --package flashdreams-omnidreams omnidreams-eval stage-batch \
  --manifest "$RUN/manifest.jsonl" \
  --batch-plan "$RUN/batches.json" \
  --batch-id batch-00000 \
  --scratch-root "$SCRATCH" \
  --output "$RUN/staged/batch-00000.jsonl"

uv run --package flashdreams-omnidreams omnidreams-eval generate \
  --staged-manifest "$RUN/staged/batch-00000.jsonl" \
  --run-root "$RUN"

uv run --package flashdreams-omnidreams omnidreams-eval validate-generation \
  --run-root "$RUN" \
  --output "$RUN/validation.json"

uv run --package flashdreams-omnidreams omnidreams-eval summarize-run \
  --run-root "$RUN"

uv run --package flashdreams-omnidreams omnidreams-eval check-baseline \
  --summary "$RUN/evaluation-summary.json" \
  --baseline "$BASELINE" \
  --output-json "$RUN/baseline-check.json"
```

External evaluator setup is intentionally separate from FlashDreams generation,
because DrivingGen and WorldLens have their own dependencies and checkpoint
caches. Use `setup-evaluator` for DrivingGen and `setup-worldlens` for
WorldLens, then run the corresponding `prepare-*` and evaluator commands. The
adapter modules pin the upstream GitHub URLs and revisions used today; moving
those pins into shared config or a maintained fork is a reasonable follow-up if
the evaluator stack becomes long-lived.

Runtime depends mostly on FlashDreams generation and evaluator environment
setup. On a workstation-class GPU such as an RTX 6000 Pro, 20-scene batches are
intended to be practical, while the full Hugging Face scene set should be run in
batches to avoid staging all 1-2 GB scenes at once. Evaluator setup can also
download model checkpoints and may take several minutes on first use.

Interpret the report as follows:

- Validation checks generation completeness, frame counts, runner schedules,
  and missing artifacts. Any validation failure should be inspected before
  trusting evaluator metrics.
- DrivingGen FVD-lite is a regression metric. Lower is better only when
  comparing the same fixed scene split across model versions. Do not compare
  `batch-00000` directly against `batch-00001` as a quality claim.
- DrivingGen reference-vs-reference FVD is diagnostic only; it measures split
  diversity, not OmniDreams quality.
- WorldLens temporal and subject consistency are roughly higher-is-better, with
  1.0 as an idealized upper bound. They are useful standalone video-consistency
  signals, but they do not directly measure closed-loop simulator quality,
  path correctness, off-road behavior, or collisions.
- `check-baseline` compares a run summary against a JSON file containing the
  accepted metric envelope. Keep generated clips as run artifacts, not in the
  baseline JSON; the baseline should contain only expected metric values and
  tolerances.

## Run the driving applications

The native-v2 `interactive-drive` application launches the single-process
driving demo. Refer to the
[interactive guide](../../apps/interactive_drive/README.md) for
controls and runtime requirements. The model-neutral application code lives in
`apps/`; this integration supplies its OmniDreams model adapter.

Validated & Supported GPU: RTX 6000 Pro Blackwell
Validated & Supported OS: Ubuntu 26.04
Validated & Supported Controllers: Playstation Dualshock - Fanatec Driving-sim wheel

Execute the following:
```bash
# We are Assuming `uv` is installed

# Token For Asset Repos
export HF_TOKEN=<YOUR-HF-TOKEN>

# Enable long paths to avoid breaking third-party source checkouts
git config --system core.longpaths true

# Sync dependencies
uv sync --package flashdreams-omnidreams --extra interactive-drive
uv run --package flashdreams-omnidreams python integrations_v2/omnidreams/impl/omnidreams_singleview/tools/sync_thirdparty.py sync

# Prepare to run tuned for performance
uv run --package flashdreams-omnidreams omnidreams-prepare --perf

# Setup controllers if not using keyboard as control scheme & NOT runing headless
uv run --package flashdreams-omnidreams interactive-drive-configuration

# Run in a browser. The first run can take a while to autotune.
uv run --package flashdreams-omnidreams flashdreams-run-v2 \
    interactive-drive-omnidreams --mode webrtc --host 0.0.0.0 --port 8089

# Or write a bounded rollout to MP4.
mkdir -p outputs
uv run --package flashdreams-omnidreams flashdreams-run-v2 \
    interactive-drive-omnidreams --mode mp4 \
    --output-path outputs/interactive-drive.mp4 -- --total-blocks 10
```

## Native DiT defaults

NVIDIA OmniDreams native DiT acceleration remains gated by the pipeline config's
`native_dit_acceleration` policy (`disabled`, `auto`, or `required`). When that
native path is enabled, the default compute profile is the FP8 KV-cache backend
with cuDNN attention:

- `native_dit_backend="fp8_kvcache_cudnn"`
- `native_dit_attention_backend="auto"` (currently resolves to cuDNN)

Set `native_dit_attention_backend="sparge"`, `"sage3"`, or `"sage3_fp8"`
explicitly to opt into Sparge/SageAttention-3 experiments. Use
`native_dit_sparge_hybrid_period > 1` with `"sparge"` to enable the FP8
Sparge/SageAttention-3 hybrid schedule when the extension and GPU support it.

## Run tests

Run tests from the workspace root. Sync the OmniDreams `dev` extra, which
provides the interactive-drive test dependencies, together with the workspace
`test` group, which provides pytest and its shared plugins:

```bash
uv sync --package flashdreams-omnidreams --extra dev --group test
```

Run all tests that participate in CPU or GPU CI with:

```bash
uv run --package flashdreams-omnidreams --extra dev --group test pytest \
  integrations_v2/omnidreams/tests \
  -m "not manual" -v
```

Use the tier markers to run a narrower suite:

```bash
# CPU-safe tests
uv run --package flashdreams-omnidreams --extra dev --group test pytest \
  integrations_v2/omnidreams/tests -m ci_cpu -v

# Tests that require CUDA, libGL, or cv2
uv run --package flashdreams-omnidreams --extra dev --group test pytest \
  integrations_v2/omnidreams/tests -m ci_gpu -v
```

Heavy, credential-dependent, or environment-specific tests use the `manual`
marker. For example, run the end-to-end streaming pipeline test on a suitable
GPU with access to the required checkpoints:

```bash
uv run --package flashdreams-omnidreams --extra dev --group test pytest \
  integrations_v2/omnidreams/tests/test_omnidreams_pipeline.py::test_omnidreams_streaming_inference \
  -p no:manual_marker -m manual -v -s
```

The native CUDA extension build smoke test is opt-in because it performs a
clean extension build:

```bash
OMNIDREAMS_SINGLEVIEW_RUN_NATIVE_BUILD_TEST=1 \
uv run --package flashdreams-omnidreams --extra dev --group test pytest \
  integrations_v2/omnidreams/tests/test_omnidreams_singleview_native.py::test_cuda_native_extension_builds \
  -m ci_gpu -v -s
```

Keep `--extra dev --group test` on `uv run`: it synchronizes the shared `.venv`
before launching pytest, and omitted selections may be removed.

## Run benchmarks

The OmniDreams benchmarks are manual, GPU-only pytest tests. Run them from the
workspace root on a supported NVIDIA GPU. First sync the OmniDreams package and
the workspace `test` dependency group, which provides both `pytest` and
`pytest-benchmark`:

```bash
uv sync --package flashdreams-omnidreams --group test
```

Run the complete benchmark suite with:

```bash
uv run --package flashdreams-omnidreams --group test pytest \
  integrations_v2/omnidreams/benchmarks \
  -p no:manual_marker -m manual --benchmark-only -v
```

To run a narrower benchmark, replace the benchmark directory in that command
with one of these files:

- `test_modules.py` benchmarks the DiT block and self-attention with the
  `omnidreams_torch`, `optimized_cudnn`, and `optimized_fa2` implementations.
  Cross-attention is unaffected by the SDPA selector and is benchmarked once
  per projection backend. The backend-independent MLP is benchmarked once.
- `test_network.py` benchmarks one steady-state DiT evaluation with the
  `omnidreams_torch`, `optimized_cudnn`, `optimized_fa2`, and native `cuda`
  implementations. It uses production tensor geometry with random weights, so
  checkpoint loading and startup are excluded.
- `test_pipeline.py` benchmarks steady-state generation and finalization with
  the `omnidreams_torch`, `optimized_cudnn`, `optimized_fa2`, and native `cuda`
  implementations at the runner's production 704x1280 resolution and scheduler
  configuration.

The selected optimized configurations use row-scaled FP8 projections. `optimized_cudnn`
uses PyTorch's cuDNN SDPA backend with a BF16 self-attention cache, while
`optimized_fa2` uses Triton FlashAttention2 (FA2) with an E4M3 cache. Text and
cross-view attention remain BF16.

The optimized cases require an NVIDIA GPU with compute capability 9.0 or newer;
they skip cleanly on older GPUs.

Keep `--group test` on both commands. A plain
`uv sync --project integrations_v2/omnidreams` installs only the integration's
runtime dependencies, so a later `uv run pytest` cannot find the benchmark
test tools. The benchmarks manage their warmup and measured rounds internally;
when publishing results, also record the commit, GPU and software stack, model
configuration, and any fallback warnings.

## Run Interactive Drive directly

From the repository root on a CUDA machine:

```bash
export HF_TOKEN=<your-hf-token>

mkdir -p outputs
uv sync --python 3.12 --package flashdreams-omnidreams --no-dev
```

Generate an MP4 using a local USDZ scene:

```bash
uv run --package flashdreams-omnidreams flashdreams-run-v2 \
  interactive-drive-omnidreams --mode mp4 \
  --output-path outputs/omnidreams.mp4 \
  --backpressure-mode block --presentation-mode on_demand -- \
  --scene /path/to/scene.usdz --no-ui --total-blocks 10
```

Serve the same application through WebRTC:

```bash
uv run --package flashdreams-omnidreams flashdreams-run-v2 \
  interactive-drive-omnidreams --mode webrtc --host 0.0.0.0 --port 8089 -- \
  --scene /path/to/scene.usdz
```

Open the URL printed by `flashdreams-run-v2`. Omit `--scene` to download the
default scene. Application-specific arguments go after `--`.

## Run gRPC server

From the workspace root, run:

```bash
uv run --package flashdreams-omnidreams torchrun --nproc_per_node 1 -m omnidreams.impl.grpc.server --pipeline_config_name omnidreams-perf --host 0.0.0.0 --port 50051
```

The server implements `omnidreams.impl.grpc.protos.video_model.WorldModelService`
and listens on `0.0.0.0:50051` by default. Clients provide the static map,
camera specs, initial frames, prompt, rig trajectory, and dynamic actor state
through the gRPC API. Use `--record_dir <dir>` to save replayable session logs,
and add `--enable_profiling --profile_output <path>` when collecting timing
data. For distributed/context-parallel launches, increase `--nproc_per_node`;
the world size must be compatible with the selected pipeline config's camera
count.

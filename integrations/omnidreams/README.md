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

The high-level workflow is:

```bash
RUN=/trees/$USER/od-runs/od-26.01
SCRATCH=/local_nvme/$USER/omnidreams-eval-scratch

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

uv run --package flashdreams-omnidreams omnidreams-eval validate-generated \
  --run-root "$RUN" \
  --output "$RUN/validation.json"

uv run --package flashdreams-omnidreams omnidreams-eval summarize-run \
  --run-root "$RUN"
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

## Run interactive-drive (desktop demo)

The `omnidreams.interactive_drive` subpackage ships a single-process
driving demo with a Ludus OpenGL raster backend and a PyTorch world-model
backend ([see its README](omnidreams/interactive_drive/README.md) for the
full guide). From the flashdreams workspace root:

```bash
uv sync --package flashdreams-omnidreams --extra interactive-drive
uv run --package flashdreams-omnidreams interactive-drive
```

The `interactive-drive` extra adds `slangpy` (the Vulkan-backed local
windowing runtime); server users running only `omnidreams.webrtc` or
`omnidreams.grpc` can skip it. The default scene auto-stages from
`nvidia/omni-dreams-scenes` on first launch when `HF_TOKEN` is set; use
`omnidreams-prepare` for explicit staging of arbitrary scene UUIDs
or to pre-warm the ~14 GB Cosmos-Reason1 text encoder.

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

## Run WebRTC server

From the workspace root, run:

```bash
uv run --package flashdreams-omnidreams torchrun --nproc_per_node 1 -m omnidreams.webrtc.server --pipeline_config_name omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf --scene-uuid 0d404ff7-2b66-498c-b047-1ed8cded60d4 --port 8089
```

When `--scene_dir` is omitted, the server downloads the selected scene from the
configured Hugging Face org, extracts its `clipgt-<uuid>[-<variant>].usdz`
archive, and stages it under `FLASHDREAMS_CACHE_DIR` (or `~/.cache/flashdreams`).
If `--scene-uuid` is omitted too, the server uses the default WebRTC scene.
Weather variants ship as sibling archives; pass `--scene-variant rain` (or
`snow`) to serve one (default is the clear-weather scene). The runtime seeds
from the scene's first ground-truth camera frame
(`clipgt/frames/<camera>/<ts>.jpeg`, falling back to `clipgt/first_image.*`) and
the weather-matched `clipgt/prompt<N>.txt` (falling back to `clipgt/prompt.txt`).
Pass `--scene_dir <path>` to use a pre-staged local scene instead.

## Run gRPC server

From the workspace root, run:

```bash
uv run --package flashdreams-omnidreams torchrun --nproc_per_node 1 -m omnidreams.grpc.server --pipeline_config_name omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf --host 0.0.0.0 --port 50051
```

The server implements `omnidreams.grpc.protos.video_model.WorldModelService`
and listens on `0.0.0.0:50051` by default. Clients provide the static map,
camera specs, initial frames, prompt, rig trajectory, and dynamic actor state
through the gRPC API. Use `--record_dir <dir>` to save replayable session logs,
and add `--enable_profiling --profile_output <path>` when collecting timing
data. For distributed/context-parallel launches, increase `--nproc_per_node`;
the world size must be compatible with the selected pipeline config's camera
count.

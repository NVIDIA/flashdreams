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

# flashdreams-lingbot

LingBot-World v1/v2 streaming camera-control I2V integration + a minimal
WebRTC demo server, packaged as a [`flashdreams`](../..) plugin. Both model
versions use the same pipeline and serving code; v2 is selected by its
checkpoint-backed config slug.

This is a worked example of the
[Add a new method](https://nvidia.github.io/flashdreams/main/developer_guides/new_integration.html)
developer-guide flow, extended with a per-plugin runtime server.

## Shipped slugs

| slug | description |
| --- | --- |
| `lingbot-world-fast` | Lingbot World Fast streaming camera-control I2V (Wan VAE decoder, 4-step). |
| `lingbot-world-fast-taehv-window15-sink3` | LightTAE decoder swap with `window_size_t=15` + `sink_size_t=3` for tighter interactive streaming. |
| `lingbot-world-v2-14b-causal-fast` | LingBot-World v2 14B causal-fast using the shared LingBot pipeline (Wan VAE decoder, 4-step). |
| `lingbot-world-v2-14b-causal-fast-taehv-window15-sink3` | v2 checkpoint with the same LightTAE/window/sink interactive preset. |

## Install

The plugin is registered as a `uv` workspace member in the repo-root
`pyproject.toml`, so a single `uv sync` from the repo root pulls it in:

```bash
uv sync
```

Standalone (outside the workspace) also works:

```bash
uv pip install -e integrations/lingbot
```

## HuggingFace setup

Checkpoints are auto-downloaded from HuggingFace at first run. Set an
auth token first.

```bash
# huggingface token.
export HF_TOKEN=<your-hf-token>

# (optional) override the cache location.
export HF_HOME=~/.cache/huggingface  # default
```

## Run

Once installed, the slugs are discovered automatically by `flashdreams-run`:

```bash
# List every registered runner (this plugin's slugs appear under "lingbot-world-*").
uv run flashdreams-run --help

# Per-runner help: every overridable field is a CLI flag.
uv run flashdreams-run lingbot-world-fast --help

# Single-GPU demo with the bundled example assets (lazy-downloaded
# from the upstream LingBot-World GitHub examples folder on first run).
uv run flashdreams-run lingbot-world-fast --example-data True --total-blocks 21

# The v2 model is the same runtime with the v2 checkpoint config.
uv run flashdreams-run lingbot-world-v2-14b-causal-fast \
    --example-data True --total-blocks 21

# Custom inputs (production layout).
uv run flashdreams-run lingbot-world-fast \
    --image-path /path/to/first_frame.jpg \
    --pose-path /path/to/poses.npy \
    --intrinsic-path /path/to/intrinsics.npy \
    --prompt "your text prompt here" --total-blocks 21
```

Multi-GPU via context-parallelism (Wan 2.1 CP assumes `cp_size == world_size`):

```bash
# e.g. 4GPUs
uv run torchrun --nproc_per_node=4 --no-python flashdreams-run \
    lingbot-world-fast --example-data True --total-blocks 21
```

## Programmatic access

Access via runner.
```python
from lingbot.config import RUNNER_LINGBOT_WORLD_FAST as runner_config
from flashdreams.infra.config import derive_config

cfg = derive_config(runner_config, prompt="A cinematic flythrough.", example_data=True)
runner = cfg.setup()
runner.run()
```

To use v2, import
`RUNNER_LINGBOT_WORLD_V2_14B_CAUSAL_FAST` from the same `lingbot.config`
module; no separate package or alternate serving path is required.

Access via pipeline.
```python
import torch
from lingbot.config import PIPELINE_LINGBOT_WORLD_FAST as pipeline_config
from lingbot.encoder.camctrl import CamCtrlInput

pipeline = pipeline_config.setup().to("cuda").eval()
sp = pipeline.decoder.spatial_compression_ratio

cache = pipeline.initialize_cache(
    text=["A cinematic flythrough."],
    image=first_frames_t,         # [T=1, C, H, W] in [-1, 1] (batch_shape=())
    height=464 // sp,             # latent height for DiT
    width=832 // sp,              # latent width for DiT
)

total_blocks: int = 21
generated_chunks: list[torch.Tensor] = []
for i in range(total_blocks):
    camctrl_input = CamCtrlInput(
        intrinsics=...,           # [T_chunk, 4] (fx, fy, cx, cy)
        poses=...,                # [T_chunk, 4, 4] camera-to-world
        world_scale=...,
    )
    video_chunk = pipeline.generate(autoregressive_index=i, cache=cache, input=camctrl_input)
    pipeline.finalize(autoregressive_index=i, cache=cache)  # update KV cache
    generated_chunks.append(video_chunk.cpu())              # each chunk is [T, C, H, W]
```

## Three-stage disaggregated inference

LingBot can load the encoder, DiT, and decoder on three independent GPUs:

```text
GPU 0: UMT5 + image/VAE/camera encoder
         │ Mooncake GPU-memory transfer
GPU 1: scheduler + DiT + session-pinned KV cache
         │ Mooncake GPU-memory transfer
GPU 2: streaming VAE or LightTAE decoder
```

This is pipeline-stage disaggregation, analogous to SGLang's independently
scheduled prefill and decode pools but split at diffusion-native boundaries.
The evolving autoregressive KV cache stays on the DiT worker. Only one-shot
conditioning, per-step encoder features, and clean latents cross stage
boundaries. Moving the KV cache every chunk would add a much larger transfer
and break session affinity.

Install the optional Mooncake transport:

```bash
uv sync --package flashdreams-lingbot --extra dev --extra disagg

# The container/host runtime must also provide the RDMA userspace libraries.
apt-get install libibverbs1 ibverbs-providers librdmacm1 ibverbs-utils
```

Run the reproducible three-GPU benchmark:

```bash
uv run --package flashdreams-lingbot torchrun \
  --standalone --nproc_per_node=3 \
  -m lingbot.disagg.benchmark \
  --model lingbot-world-fast-taehv-window15-sink3 \
  --warmup-blocks 6 --measured-blocks 5 \
  --bandwidth-probe-mib 256 --bandwidth-probe-iters 8 \
  --output-dir outputs/lingbot_disagg
```

Use all eight GPUs for concurrent sessions by keeping one encoder and one
decoder worker and assigning the other six GPUs to session-affine DiT workers:

```bash
uv run --package flashdreams-lingbot torchrun \
  --standalone --nproc_per_node=8 \
  -m lingbot.disagg.benchmark_replicated \
  --dit-replicas 6 \
  --model lingbot-world-fast-taehv-window15-sink3 \
  --warmup-blocks 6 --measured-blocks 5 \
  --bandwidth-probe-mib 256 --bandwidth-probe-iters 8 \
  --output-dir outputs/lingbot_disagg_1e6d1d
```

The allocation is derived from the tracked 1:1:1 stage service times. It is a
throughput topology: each DiT replica owns a distinct session and KV cache.
It does not split one session's DiT computation over six GPUs.

To minimize one session's latency instead, make ranks 1–6 one context-parallel
DiT group. Rank 1 receives the Mooncake handoff, broadcasts the input within
the NCCL subgroup, and sends the gathered clean latent to rank 7:

```bash
TORCHINDUCTOR_COMPILE_THREADS=4 \
uv run --package flashdreams-lingbot torchrun \
  --standalone --nproc_per_node=8 \
  -m lingbot.disagg.benchmark_cp \
  --cp-ranks 6 --cp-method ring \
  --model lingbot-world-fast-taehv-window15-sink3 \
  --warmup-blocks 6 --measured-blocks 5 \
  --bandwidth-probe-mib 256 --bandwidth-probe-iters 8 \
  --output-dir outputs/lingbot_disagg_cp6
```

CP6 must use ring attention for this 40-head model. Ulysses requires the head
count to be divisible by the context-parallel size, so CP6 Ulysses is rejected
(`40 % 6 != 0`). A CP4 Ulysses comparison uses six processes and leaves two
GPUs idle:

```bash
TORCHINDUCTOR_COMPILE_THREADS=4 \
uv run --package flashdreams-lingbot torchrun \
  --standalone --nproc_per_node=6 \
  -m lingbot.disagg.benchmark_cp \
  --cp-ranks 4 --cp-method ulysses \
  --model lingbot-world-fast-taehv-window15-sink3 \
  --warmup-blocks 6 --measured-blocks 5 \
  --output-dir outputs/lingbot_disagg_cp4
```

Cap Inductor compilation parallelism when several compiled DiT ranks share a
node. The default 32 workers per rank becomes 192 workers at CP6 and can
overcommit the host during cold compilation.

On the tested H100 node, CP6 ring was the minimum-latency allocation: 743.27 ms
median per 12-frame chunk and 15.90 generated FPS, a 3.01× latency speedup over
CP1. CP4 Ulysses reached 754.41 ms and 15.70 FPS. CP6 was 1.5% faster; CP4 had
higher scaling efficiency and left two GPUs available for other work.

The complete pipeline also fits on one H100 80 GB. At 832×464, the measured
single-GPU aggregated run reached **5.56 FPS** and **2157.51 ms median / 2166.25
ms p90** latency per 12-frame chunk. Initialization peaked at **66.55 GiB**
allocated HBM; rollout peaked at **59.36 GiB**. See the
[single-H100 report](docs/benchmark_h100_aggregated_cp1/README.md).

Running one complete CP1 pipeline and one session independently on each of
eight H100s reached **43.44 aggregate FPS**, **5.54 median FPS per session**,
and **2163.64 ms median** chunk latency. Rollout peak allocation was **59.35
GiB per GPU / 474.84 GiB node-wide**. See the
[eight-replica report](docs/benchmark_h100_aggregated_8xcp1/README.md).

For the eight-GPU aggregated baseline, put the complete pipeline on every GPU
and use all ranks as the DiT context-parallel WORLD group:

```bash
TORCHINDUCTOR_COMPILE_THREADS=4 \
uv run --package flashdreams-lingbot torchrun \
  --standalone --nproc_per_node=8 \
  -m lingbot.disagg.benchmark_aggregated \
  --cp-method ulysses \
  --model lingbot-world-fast-taehv-window15-sink3 \
  --pixel-width 832 --pixel-height 448 \
  --warmup-blocks 6 --measured-blocks 5 \
  --bandwidth-probe-mib 256 --bandwidth-probe-iters 8 \
  --output-dir outputs/lingbot_aggregated_cp8
```

The normal 832×464 grid has 4,524 tokens and cannot divide over CP8. The
nearest valid height is 448, which produces 4,368 tokens. On eight H100s, the
aggregated CP8 Ulysses run reached 393.33 ms median latency and 29.50 generated
FPS. It is the fastest tested single-session topology, but it replicates
encoder and decoder work and gives up independent stage placement. The
[aggregated report](docs/benchmark_h100_aggregated_cp8/README.md) and
[comparison chart](docs/aggregated_vs_disaggregated.svg) include the
resolution-normalized throughput and node-wide HBM tradeoff.

Validate the data plane without loading checkpoints:

```bash
uv run --package flashdreams-lingbot torchrun \
  --standalone --nproc_per_node=3 \
  -m lingbot.disagg.benchmark --transport-only \
  --bandwidth-probe-mib 256 --bandwidth-probe-iters 8
```

Both benchmarks write `benchmark.json` and a Markdown summary. They report:

- median and p90 encoder, DiT, finalize, decoder, and end-to-end chunk latency;
- generated FPS after excluding warmup;
- payload bytes, sender registration time, transfer time, full handoff time,
  and effective GB/s for encoder → DiT and DiT → decoder;
- a reusable 256 MiB transfer probe to distinguish link bandwidth from
  small-payload setup overhead;
- per-stage peak GPU memory and the exact software/hardware environment.

Start with the concise
[disaggregation experiment summary](docs/disaggregation_experiment_summary.md)
for the deployment decision, headline data, limitations, and comparison chart.
The [full H100 experiment record](docs/disaggregated_inference_experiment.md)
contains the tested stack, measurement method, Slurm reproduction procedures,
stage breakdowns, and chronological optimization findings.

Mooncake is explicitly initialized with its `rdma` protocol. On a single node,
the engine may select a topology-local GPU path; the measured effective GB/s
is therefore authoritative for that allocation, while the protocol name alone
must not be presented as proof that traffic traversed an InfiniBand NIC.
Cross-node deployment additionally needs routable stage hostnames, RDMA-capable
NICs, GPUDirect RDMA, and a control plane that forwards the opaque
`TensorTransferTicket` between stage services.

The design follows the
[LightX2V three-stage disaggregation study](https://light-ai.top/LightX2V-BLOG/posts/Disaggregation/):
control-plane messages carry only tensor metadata and registered destination
addresses; Mooncake moves the tensor payload directly between device buffers.

## Run (WebRTC interactive demo)

The `lingbot.webrtc` subpackage exposes a minimal WebRTC server that
binds the integration pipeline to keyboard input over a DataChannel and streams the
generated video back to the browser.

- `GET /request_session` serves a standalone viewer page (HTML/CSS/JS files on disk, not inlined in Python).
- `POST /api/webrtc/offer` performs SDP offer/answer signaling.
- Runtime/model/config preloading during server startup (before handling requests).
- A single active WebRTC session per server process.
- Action-bound control flow:
  1. browser sends an action (`keydown`, `keyup`, or `step`) over DataChannel,
  2. server runs one Lingbot AR inference chunk,
  3. server enqueues chunk frames to the WebRTC track and emits `chunk_done`.

From repository root:

```bash
uv run --package flashdreams-lingbot python -m lingbot.webrtc.server \
    --host 0.0.0.0 --port 8089 --config_name lingbot-world-fast-taehv-window15-sink3

# 4 GPUs
uv run --package flashdreams-lingbot \
  python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
  -m lingbot.webrtc.server \
  --host 0.0.0.0 --port 8089 \
  --config_name lingbot-world-fast-taehv-window15-sink3
```

Then open:

- [http://localhost:8089/request_session](http://localhost:8089/request_session)
- [http://localhost:8089/healthz](http://localhost:8089/healthz) (`runtime_ready` indicates preload completion)

### Runtime requirements

- CUDA-capable GPU.
- `HF_TOKEN` exported. The selected `robbyant/lingbot-world-fast` (v1) or
  `robbyant/lingbot-world-v2-14b-causal-fast` (v2) checkpoint is pulled from
  HuggingFace on first run and cached under `$HF_HOME`.
- ~200 GB free disk for the model + HF cache.
- Example assets (`image.jpg`, `intrinsics.npy`, `poses.npy`, and a prompt when
  available) for both v1 and v2 models auto-download from the canonical
  [`Robbyant/lingbot-world-v2`](https://github.com/Robbyant/lingbot-world-v2/tree/main/examples)
  examples folder into `$FLASHDREAMS_CACHE_DIR/example_data/lingbot_world/<NN>/` on
  first launch (`<NN>` is the `--example-idx`: `00` through `05`). Examples
  `03` and `04` use an empty prompt because they do not provide their own
  upstream `prompt.txt`.

### DataChannel message format

Browser -> server:

```json
{
  "type": "action",
  "action": {
    "event": "keydown",
    "key": "w"
  }
}
```

- Supported `event` values:
  - `keydown` / `keyup` (requires `key` in `w,a,s,d,q,e,i,j,k,l`)
  - `step` (no key required; generates a chunk using current key state)
- Key mapping:
  - `w/s`: forward/backward
  - `a/d` (or `j/l`): yaw left/right
  - `q/e`: strafe left/right
  - `i/k`: pitch up/down
- If multiple key events arrive before the next chunk starts, the server
  aggregates them and applies latest-pressed precedence per component
  (forward/backward, turn, strafe, pitch).

Server -> browser:

```json
{
  "type": "chunk_done",
  "chunk_index": 3,
  "num_frames": 12,
  "enqueued_frames": 12
}
```

Text-driven events work with both v1 and v2 through the same DataChannel:

```json
{
  "type": "event",
  "event_id": "portal",
  "state": "trigger"
}
```

- `trigger`, `hold`, or `on` activates an advertised event.
- `clear`, `release`, `off`, or `none` restores the base prompt context.
- The initial-scene payload advertises `capabilities.text_events`,
  `event_catalog`, and `active_event_id`; successful updates receive an
  `event_ack`.

## Tests

```bash
uv run --extra dev pytest integrations/lingbot/tests
```

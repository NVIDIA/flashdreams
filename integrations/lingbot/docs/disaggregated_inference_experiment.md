<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LingBot three-stage disaggregation experiment

## Status

**Useful opt-in.** The experiment validates that encoder, DiT, and decoder
components can run on separate H100 GPUs and exchange their real LingBot tensor
payloads through Mooncake's RDMA transport. It does not establish output-quality
equivalence with the aggregated runner, cross-node performance, or concurrent
multi-session capacity, so it is not a new default serving path.

The machine-readable measurements are in
[`benchmark_h100_3stage/benchmark.json`](benchmark_h100_3stage/benchmark.json);
the generated compact table is in
[`benchmark_h100_3stage/README.md`](benchmark_h100_3stage/README.md).

## Question and design

The experiment asked whether the monolithic LingBot inference pipeline could be
split into independently placed services without making inter-GPU transfer the
new bottleneck. The tested topology was:

```text
rank 0 / GPU 0                     rank 1 / GPU 1                 rank 2 / GPU 2
text + image + VAE + camera  ───▶  scheduler + DiT + KV cache ───▶ LightTAE
             14.36 MiB / block          0.55 MiB / block
```

The evolving autoregressive KV cache stays on the DiT worker. Session
conditioning crosses the encoder-to-DiT boundary once; each block then sends
the I2V latent, mask, and Plücker features to DiT and sends only the clean latent
to the decoder. Mooncake uses receiver-allocated, registered VRAM and a batched
synchronous write, so tensor contents are not serialized through the Python
control plane or staged through host memory.

This is diffusion pipeline-stage disaggregation, analogous in scheduling intent
to separating prefill and decode pools in an LLM server, but the boundaries and
resident state are diffusion-native.

## Tested configuration

| Item | Value |
| --- | --- |
| Date | 2026-07-29 |
| Slurm allocation | Job `14621292`, node `pool0-01299` |
| Repository base | `e580e27d408b3cf8bd8a549f990c361b94d3379f`; the implementation under test was the worktree change recorded with this report |
| Container | `flashdreams-base-v0.3-20260429-af40a4f.sqsh` |
| GPUs used | 3 × NVIDIA H100 80 GB HBM3, one process per GPU |
| GPU topology | GPU 0/1/2 connected by NVLink; node exposed nine `mlx5` HCAs |
| Driver | Not captured by the original harness; the reproduction checklist below captures it |
| Python / PyTorch | Python 3.12, PyTorch `2.12.1+cu130` |
| CUDA / cuDNN | CUDA 13.0, cuDNN 92000 |
| Mooncake | `mooncake-transfer-engine-cuda13==0.3.12.post1` |
| Model | `lingbot-world-fast-taehv-window15-sink3` |
| DiT checkpoint | `robbyant/lingbot-world-fast`, `diffusion_pytorch_model.safetensors.index.json` |
| Precision | BF16 model and image tensors; FP32 camera tensors |
| Scheduler | Four-step distilled flow matching, CFG 1.0, seed 42 |
| Streaming layout | 3 latent frames per chunk, temporal window 15, sink 3 |
| Decoder | LightTAE / TAEHV |
| Input | Upstream LingBot example `00`: `image.jpg`, `intrinsics.npy`, `poses.npy`, and `prompt.txt` |
| Resolution / target playback | 832 × 464, 16 FPS |
| Process state | Fresh `torchrun` process with a persistent, previously populated Triton cache |
| Measurement policy | 6 warmup blocks, then 5 measured blocks; 8 iterations per 256 MiB link probe |

The exact prompt was:

> The video presents a soaring journey through a fantasy jungle. The wind
> whips past the rider's blue hands gripping the reins, causing the leather
> straps to vibrate. The ancient gothic castle approaches steadily, its stone
> details becoming clearer against the backdrop of floating islands and distant
> waterfalls.

## Method

1. Each rank constructed only its stage-local weights. Model construction
   happened before `torch.distributed` initialization because LingBot otherwise
   interprets a three-rank process group as three-way context parallelism.
2. The ranks initialized a Gloo control group and independent Mooncake P2P
   endpoints with `protocol=rdma`.
3. Mooncake discovered nine `mlx5` HCAs, installed its RDMA transport, and
   completed RDMA-ready handshakes.
4. Before model measurements, each edge ran one connection warmup followed by
   eight synchronous 256 MiB transfers through reusable registered buffers.
5. The model ran eleven autoregressive blocks. Blocks 0–5 were excluded because
   they cover initial model/compiler warmup and the second DiT cache-shape
   transition at block 5. Blocks 6–10 were measured.
6. CUDA events measured stage compute. Host wall clocks measured synchronous
   transfer calls, complete handoffs, and barrier-to-barrier chunk latency.
7. Throughput was computed from the 60 generated measured frames divided by the
   sum of the five measured chunk latencies. It is generated throughput, not the
   configured 16 FPS playback target.

Warmup must remain at six blocks or more for this preset. Shorter trials observed
one-time DiT compilation at block 5 and produced misleading 13–24 second
outliers in the measured set.

## Findings

### Steady-state compute and latency

| Component | Median | P90 | Share of median chunk latency |
| --- | ---: | ---: | ---: |
| Encoder compute | 1.08 ms | 1.14 ms | 0.05% |
| Encoder → DiT full handoff | 25.38 ms | 25.88 ms | 1.14% |
| DiT denoise | 1734.84 ms | 1755.00 ms | 77.67% |
| DiT cache finalization | 444.79 ms | 446.72 ms | 19.91% |
| DiT → decoder full handoff | 12.05 ms | 14.73 ms | 0.54% |
| Decoder compute | 7.14 ms | 7.15 ms | 0.32% |
| End-to-end chunk | 2233.57 ms | 2250.79 ms | 100% |

The five measured blocks generated 12 frames each. Total generated throughput
was **5.36 FPS**. DiT denoising plus cache finalization consumed 97.58% of median
chunk latency, so additional DiT replicas—not encoder or decoder replicas—are
the first resource-scaling lever for concurrent sessions.

### Transfer behavior

| Edge | Real payload | Copy median | Payload bandwidth median | Full handoff median | 256 MiB probe median |
| --- | ---: | ---: | ---: | ---: | ---: |
| Encoder → DiT | 14.36 MiB | 1.35 ms | 11.12 GB/s | 25.38 ms | 41.35 GB/s |
| DiT → decoder | 0.55 MiB | 1.49 ms | 0.39 GB/s | 12.05 ms | 41.00 GB/s |

The two transfer-engine copy calls used 0.13% of median chunk latency. Complete
handoffs—including receiver allocation and registration, sender registration,
metadata broadcasts, barriers, and the copy—used 1.68%. The small clean-latent
payload cannot saturate the link, which explains its low payload GB/s despite
the 41 GB/s large-buffer probe.

The full handoff gap is mostly setup and synchronization rather than byte
movement. Production serving should pool and reuse registered destination
buffers and replace global barriers/object broadcasts with request-scoped
control messages.

### Peak allocated GPU memory

| Stage GPU | Peak allocated memory | Share of three-stage total |
| --- | ---: | ---: |
| Encoder | 13.72 GiB | 18.98% |
| DiT | 56.29 GiB | 77.86% |
| Decoder | 2.29 GiB | 3.17% |

Disaggregation makes the imbalance explicit: encoder and decoder GPUs have
substantial unused capacity while the DiT GPU owns most weights and resident
state. A production scheduler should allow encoder and decoder workers to serve
multiple session-affine DiT workers.

## How to reproduce

The commands below assume the repository is at
`/home/gtong/lustre/flashdreams-dist` and the cluster helper is
`/home/gtong/work/srun.sh`.

### 1. Allocate or reuse a compute node

Run on the login node:

```bash
squeue -u gtong
cd /home/gtong/work
export FLASHDREAMS_HOST_DIR=/home/gtong/lustre/flashdreams-dist
./srun.sh
```

If `squeue` already shows a running job, attach to that exact allocation:

```bash
./srun.sh 1 <job-id>
```

Do not run model loading or package synchronization on the login node.

### 2. Verify the mounted checkout and hardware

Inside the container shell:

```bash
hostname
pwd
git rev-parse HEAD
nvidia-smi -L
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
nvidia-smi topo -m
ibv_devices
```

`pwd` must be `/workspace/flashdreams`, the Git revision must match the intended
checkout, and at least three GPUs must be visible. Preserve the driver and
topology output with the benchmark artifacts.

### 3. Install RDMA userspace support and the optional transport

The writable container used for the experiment needed:

```bash
apt-get update
apt-get install -y libibverbs1 ibverbs-providers librdmacm1 ibverbs-utils
uv sync --package flashdreams-lingbot --extra dev --extra disagg
```

The optional `disagg` extra installs the CUDA 13 Mooncake wheel. A missing
`libibverbs.so.1` means the OS packages above are absent.

### 4. Validate the transport before loading checkpoints

```bash
uv run --package flashdreams-lingbot torchrun \
  --standalone --nproc_per_node=3 \
  -m lingbot.disagg.benchmark \
  --transport-only \
  --bandwidth-probe-mib 256 \
  --bandwidth-probe-iters 8
```

Confirm that Mooncake logs `installTransport, type=rdma`, discovers the expected
HCAs, completes RDMA-ready handshakes, and reports finite bandwidth in both
directions. Protocol configuration alone is not evidence that bytes traversed
the intended RDMA path.

### 5. Run the model benchmark

Keep the default persistent `TRITON_CACHE_DIR` supplied by `srun.sh`, then run:

```bash
uv run --package flashdreams-lingbot torchrun \
  --standalone --nproc_per_node=3 \
  -m lingbot.disagg.benchmark \
  --model lingbot-world-fast-taehv-window15-sink3 \
  --example-idx 0 \
  --pixel-width 832 \
  --pixel-height 464 \
  --fps 16 \
  --warmup-blocks 6 \
  --measured-blocks 5 \
  --bandwidth-probe-mib 256 \
  --bandwidth-probe-iters 8 \
  --output-dir outputs/lingbot_disagg_h100
```

The first run downloads checkpoints and example assets and may compile kernels.
For a warm-cache steady-state comparison, rerun in a fresh `torchrun` process
without clearing `TRITON_CACHE_DIR`. For a cold-start study, use a new explicit
cache directory and report startup/compile latency separately; do not mix it
into the steady-state rows.

### 6. Inspect and preserve results

```bash
sed -n '1,240p' outputs/lingbot_disagg_h100/README.md
jq '.environment, .summary' outputs/lingbot_disagg_h100/benchmark.json
jq '.records[] | select(.warmup == false)' \
  outputs/lingbot_disagg_h100/benchmark.json
```

The output directory contains the generated Markdown summary and raw per-block
JSON. Check that exactly five records have `warmup=false`, that every measured
block emits 12 frames, and that no measured latency contains a compile outlier.

### 7. Run focused CPU validation and release the node

```bash
uv run --package flashdreams-lingbot pytest \
  flashdreams/tests/test_pipeline_stages.py \
  flashdreams/tests/test_transfer.py \
  integrations/lingbot/tests/test_disagg_stages.py

uv lock --check
exit
```

## Acceptance and limitations

| Decision | Status | Evidence / missing evidence |
| --- | --- | --- |
| Three independent GPU stages | Useful opt-in | Real LingBot rollout completed with stage-local weights and state |
| Mooncake RDMA data plane | Useful opt-in | RDMA transport/handshakes observed; 41 GB/s single-node probes |
| Default serving path | Deferred | Needs scheduler, bounded queues, cancellation, buffer pooling, and failure recovery |
| Output-quality equivalence | Deferred | No matched-seed aggregated-vs-disaggregated decoded comparison yet |
| Cross-node efficiency | Deferred | Only one eight-H100 node was measured |
| Throughput improvement | Deferred | Single-session serial latency was measured; no aggregated baseline or concurrent-session load test |

Results apply to this H100/CUDA 13/Mooncake stack and should not be generalized
to other GPU, NIC, driver, topology, or model configurations without repeating
the experiment.

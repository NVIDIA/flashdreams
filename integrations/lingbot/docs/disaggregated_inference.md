<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LingBot disaggregated inference design

## Stage contract

The monolithic `initialize_cache → generate → finalize` lifecycle becomes:

1. The encoder worker creates session conditioning and its streaming VAE/camera
   cache. It transfers text/image embeddings to the DiT worker once.
2. For every autoregressive block, the encoder worker transfers the I2V latent,
   injection mask, and Plücker features to the DiT worker.
3. The DiT worker denoises and finalizes its resident autoregressive KV cache.
   It transfers only the clean, unpatchified latent to the decoder.
4. The decoder worker advances its own streaming cache and produces pixels.

The DiT worker must have session affinity. Its KV cache is not a standardized
LLM prefix cache, is mutated by `finalize`, and is deliberately excluded from
the transfer protocol.

## Data plane

`MooncakeTensorTransport` uses receiver-owned buffers:

1. The sender publishes tensor names, shapes, dtypes, and byte counts.
2. The receiver allocates contiguous device tensors and registers their VRAM.
3. The receiver returns an opaque ticket containing its Mooncake session and
   registered addresses.
4. The sender registers its source allocations and performs one batched
   synchronous write.
5. The control plane waits for completion before dispatching the consumer.

No tensor is serialized into the Python control message and there is no
device-to-host-to-device staging in FlashDreams.

## Scheduling

The three-stage baseline is a fixed 1 encoder : 1 DiT : 1 decoder topology.
The eight-GPU benchmark also implements a 1 encoder : 6 DiT : 1 decoder wave
scheduler. It starts with one replica per stage, then assigns each remaining
GPU to the stage with the largest measured service-time-per-replica:

```text
stage capacity = replicas / median service time
system capacity = minimum stage capacity
```

The tracked baseline assigns all five additional GPUs to DiT because denoising
and cache finalization account for 97.58% of one-session latency. Each DiT
replica owns a distinct session and resident KV cache; the shared encoder feeds
six inputs, the DiTs execute concurrently, and the shared decoder drains six
latents. This scales concurrent-session throughput rather than one session's
latency. A production service still needs asynchronous bounded queues, request
IDs, cancellation, and pooled registered buffers in place of benchmark-wide
barriers.

The single-session topology uses the same physical allocation but groups ranks
1–6 into one context-parallel DiT:

```text
GPU 0 encoder ── Mooncake ──▶ GPU 1 CP leader ══ NCCL CP6 ══ GPUs 2–6
GPU 7 decoder ◀─ Mooncake ─── GPU 1 gathered clean latent
```

All six DiT ranks cooperate on every denoising step and retain their local
shard of the session cache. This can reduce one session's latency, unlike six
independent DiT replicas. The leader distributes full per-block input tensors
within the subgroup; the DiT's existing context-parallel path shards token
sequences and reconstructs its output before the decoder handoff.

LingBot's 14B DiT has 40 attention heads. CP6 therefore uses ring attention:
Ulysses requires `num_heads % cp_size == 0`, and `40 % 6 != 0`. CP4 Ulysses is
a compatible alternative, but it uses only six of the allocated GPUs after
reserving separate encoder and decoder ranks. Ring rotation must use subgroup
local ranks because the DiT subgroup is global ranks 1–6.

## Measurement rules

- Discard compilation, cache-fill, and connection warmup blocks.
- Report median and p90 compute and end-to-end chunk latency.
- Report generated frames divided by measured wall time, not target playback
  FPS.
- Record sender memory registration separately from the Mooncake transfer.
- Report both real handoff payload bandwidth and a large-buffer link probe.
- Record the selected GPU/NIC topology. `protocol=rdma` is configuration, not
  by itself evidence that an InfiniBand port carried the bytes.
- Compare decoded output against the aggregated pipeline with matched prompt,
  first frame, camera path, checkpoint, seed, and decoder before recommending
  the path as a default.

## H100 validation

The implementation was exercised in Slurm job `14621292` on `pool0-01299`
with three NVIDIA H100 80 GB HBM3 GPUs. Mooncake discovered the node's nine
`mlx5` HCAs, installed its RDMA transport, and completed RDMA ready handshakes.
After six warmup blocks, five measured LingBot blocks produced:

- 5.36 generated FPS and 2233.57 ms median / 2250.79 ms p90 chunk latency;
- 1734.84 ms DiT denoise and 444.79 ms DiT cache finalization;
- 25.38 ms encoder → DiT handoff for 14.36 MiB and 12.05 ms DiT → decoder
  handoff for 0.55 MiB;
- 0.13% of median latency in the two synchronous RDMA copy calls, or 1.68%
  including allocation, metadata exchange, and synchronization;
- 41.35 GB/s encoder → DiT and 41.00 GB/s DiT → decoder on reusable 256 MiB
  RDMA probes;
- 13.72 / 56.29 / 2.29 GiB peak allocated memory for encoder / DiT / decoder.

The complete per-block measurements and reproduction command are in the
[H100 benchmark report](benchmark_h100_3stage/README.md). This path remains
opt-in: the stage boundaries and tensor round trips are covered by CPU tests
and the real LingBot rollout completed, but a matched-seed decoded-output
comparison with the original aggregated runner is still required before
making it the default serving path.

The eight-GPU follow-up ran in Slurm job `14628860` on `pool0-00205`. Five
measured six-session waves produced **27.20 aggregate generated FPS**, a
**5.07× throughput gain** over the tracked 1:1:1 result and **1.90× higher
throughput per allocated GPU**. Median wave latency was 2657.06 ms, or 1.19×
the single-session baseline latency; median per-session throughput was
4.53 FPS. The twelve reusable 256 MiB Mooncake probes measured 41.22 GB/s
median across all stage edges. Peak allocations were 18.77 GiB on the shared
encoder, 56.34–56.51 GiB on each DiT, and 2.65 GiB on the shared decoder.
See the [eight-GPU report](benchmark_h100_1e6d1d/README.md) and the
[wall-time and memory chart](disaggregated_inference_breakdown.svg).

The minimum-single-session experiment ran in Slurm job `14646820` on
`pool0-01260`. One CP6 ring DiT group on GPUs 1–6 reduced median chunk latency
to **743.27 ms** and raised one-session throughput to **15.90 generated FPS**:
a **3.01× end-to-end speedup** and **3.19× DiT speedup** over CP1. The CP4
Ulysses comparison reached 754.41 ms and 15.70 FPS. CP6 was 1.5% faster and
used all eight GPUs; CP4 achieved higher per-DiT-GPU scaling efficiency (78.2%
versus 53.2%) and left two GPUs idle. See the
[CP6 report](benchmark_h100_cp6_single_session/README.md),
[CP4 report](benchmark_h100_cp4_single_session/README.md), and
[single-session wall-time and memory chart](disaggregated_inference_single_session.svg).

The
[full experiment record](disaggregated_inference_experiment.md)
documents the tested stack, warmup behavior, stage and transfer breakdowns,
Slurm reproduction procedure, interpretation, and deferred validation.

## References

- [LightX2V: Breaking the Memory and Throughput Bottlenecks of Diffusion Model Inference](https://light-ai.top/LightX2V-BLOG/posts/Disaggregation/)
- [Mooncake Transfer Engine](https://github.com/kvcache-ai/Mooncake)
- [NIXL architecture](https://github.com/ai-dynamo/nixl/blob/main/docs/nixl.md)

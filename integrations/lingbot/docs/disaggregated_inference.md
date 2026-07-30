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

The current benchmark is a fixed 1 encoder : 1 DiT : 1 decoder topology. A
production scheduler should replicate the DiT stage according to measured
service times:

```text
stage capacity = replicas / median service time
system capacity = minimum stage capacity
```

As in LightX2V, the DiT is expected to dominate. Encoder and decoder workers
can be shared across multiple session-pinned DiT workers once bounded queues,
request IDs, cancellation, and buffer lifetime management are added.

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

The
[full experiment record](disaggregated_inference_experiment.md)
documents the tested stack, warmup behavior, stage and transfer breakdowns,
Slurm reproduction procedure, interpretation, and deferred validation.

## References

- [LightX2V: Breaking the Memory and Throughput Bottlenecks of Diffusion Model Inference](https://light-ai.top/LightX2V-BLOG/posts/Disaggregation/)
- [Mooncake Transfer Engine](https://github.com/kvcache-ai/Mooncake)
- [NIXL architecture](https://github.com/ai-dynamo/nixl/blob/main/docs/nixl.md)

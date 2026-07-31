<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LingBot three-group, two-stage DiT benchmark

## Result

This experiment used GPU 0 for the shared encoder and decoder, three two-rank
pipeline-parallel DiT groups on GPUs 1–6, and left GPU 7 spare. Each DiT rank
owns 20 of the model's 40 transformer blocks and only the KV cache for those
blocks.

| Metric | Batch 1 per group | Batch 2 per group | Full-DiT replica reference |
| --- | ---: | ---: | ---: |
| Sessions | 3 | 6 | 7 |
| Aggregate generated FPS | 16.80 | **17.24** | **35.15** |
| Generated FPS per session | **5.60** | 2.87 | 5.02 |
| Median wave latency | **2140.73 ms** | 4177.56 ms | 2358.51 ms |
| P90 wave latency | **2147.37 ms** | 4183.50 ms | 2454.10 ms |
| Maximum DiT-rank required HBM | **28.70 GiB** | 39.55 GiB | 56.51 GiB |
| Node-wide required HBM | **188.44 GiB** | 256.91 GiB | 415.02 GiB |
| Median 256 MiB intra-pair P2P | 345.43 GB/s | 344.64 GB/s | — |

Pipeline sharding reduces the largest DiT-rank footprint by 30.0% at batch two
and 49.2% at batch one relative to the full-DiT replica. The batch-two layout
uses 42.82 GiB per session; six independent full pipelines require 399.30 GiB,
so the equal-session memory saving is 35.7%.

It does not improve node throughput. Batch two is 51.0% slower than the
seven-replica reference. It gains only 2.6% aggregate FPS over batch one while
almost doubling latency. The measured 344.64 GB/s P2P bandwidth and 0.34 ms
input fanout rule out NVLink as the bottleneck. The current fixed-batch
execution does not overlap stage-0 work for one session with stage-1 work for
another, so it pays a pipeline bubble at every diffusion step.

Use this implementation to reduce the minimum GPU-memory requirement. A
throughput version needs double-buffered session microbatches and per-session
cache gather/scatter before GPU 7 is likely to help. Encoder plus decoder work
was only about 46 ms of the 4.18-second batch-two wave, so moving I/O to the
spare GPU would not change the conclusion.

## Reproduction

The measured runs used Slurm job `14799647` on `pool0-01062`, eight H100 80 GB
GPUs with NV18 connectivity between every pair, revision
`0c2d48a8249577fb617bb5280208dd77409d9b1a` plus this worktree change, PyTorch
2.12.1+cu130, CUDA 13.0, and Mooncake RDMA. Install the missing RDMA userspace
libraries in the allocation's writable container layer if needed:

```bash
apt-get update
apt-get install -y libibverbs1 ibverbs-providers rdma-core
```

From the mounted repository checkout, run the batch-two experiment:

```bash
env GLOG_minloglevel=2 \
  TORCHINDUCTOR_COMPILE_THREADS=1 \
  TORCH_NCCL_SHOW_EAGER_INIT_P2P_SERIALIZATION_WARNING=0 \
uv run --no-sync --package flashdreams-lingbot torchrun \
  --standalone --nproc_per_node=8 \
  -m lingbot.disagg.benchmark_pipeline \
  --sessions-per-group 2 \
  --compile-network \
  --warmup-blocks 6 \
  --measured-blocks 5 \
  --bandwidth-probe-iters 10 \
  --output-dir outputs/lingbot_disagg_pipeline_3x2_microbatch2
```

Repeat with `--sessions-per-group 1` and a different output directory for the
control. Six warmup blocks are required because block 5 changes the cache shape
and otherwise puts compilation in the measurement window. Verify that
`nvidia-smi topo -m` reports NV18, Mooncake logs `installTransport, type=rdma`,
and the result contains three P2P pairs and five measured records.

The exact extracted values are in [summary.json](summary.json). Full per-wave
records remain in the generated `outputs/` benchmark directories.

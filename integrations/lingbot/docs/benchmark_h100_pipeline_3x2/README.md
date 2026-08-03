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

| Metric | Batch 1 | Fixed batch 2, rerun | Double-buffered batch 2 | Full-DiT replica reference |
| --- | ---: | ---: | ---: | ---: |
| Sessions | 3 | 6 | 6 | 7 |
| Aggregate generated FPS | 16.80 | 16.77 | **21.16** | **35.15** |
| Generated FPS per session | **5.60** | 2.79 | **3.53** | 5.02 |
| Median wave latency | **2140.73 ms** | 4286.95 ms | **3401.46 ms** | 2358.51 ms |
| P90 wave latency | **2147.37 ms** | 4313.83 ms | **3406.93 ms** | 2454.10 ms |
| Median DiT denoise | 1647.70 ms | 3307.89 ms | **2615.76 ms** | 1718.78 ms |
| Median finalization | 408.37 ms | 819.46 ms | **646.98 ms** | 426.45 ms |
| Maximum DiT-rank required HBM | **28.70 GiB** | 39.63 GiB | **39.47 GiB** | 56.51 GiB |
| Node-wide required HBM | **188.44 GiB** | 257.29 GiB | **256.21 GiB** | 415.02 GiB |
| Median 256 MiB intra-pair P2P | 345.43 GB/s | 345.33 GB/s | 344.40 GB/s | — |

Pipeline sharding reduces the largest DiT-rank footprint by 30.1% with double
buffering and 49.2% at batch one relative to the full-DiT replica. The
double-buffered layout uses 42.70 GiB per session; six independent full
pipelines require 399.30 GiB, so the equal-session memory saving is 35.8%.

Double buffering is a material improvement over the matched fixed-batch run:
aggregate throughput rises **26.2%**, median wave latency falls **20.7%**, DiT
denoise time falls **20.9%**, and finalization falls **21.0%**. Maximum DiT-rank
HBM changes by only -0.16 GiB. The implementation keeps two separate batch-one
session caches, sends session 0 to stage 1, computes session 1 on stage 0 while
stage 1 processes session 0, then drains both outputs. The unchanged 344–345
GB/s P2P result confirms that the gain comes from compute overlap, not a
transfer change.

This still does not beat full-DiT replicas for maximum H100-node throughput.
The double-buffered layout is 39.8% below the seven-replica reference, while
reducing the largest DiT-rank footprint by 30.1%. It is therefore useful when
the full DiT and its session cache do not fit on a smaller GPU, or when lower
per-rank HBM matters more than maximum throughput. Each four-step denoise and
the final cache-update forward still pays one fill and one drain; a deeper
pipeline or more slots would increase bubbles and is not automatically better.

## Reproduction

The double-buffer and matched fixed runs used Slurm job `15002340` on
`pool0-01858`, eight H100 80 GB GPUs with NV18 connectivity between every pair,
revision `542190c0ae4134cf5e0da24342687a5328e93ac9` plus this worktree change,
PyTorch 2.12.1+cu130, CUDA 13.0, and Mooncake RDMA. Install the missing RDMA
userspace libraries in the allocation's writable container layer if needed:

```bash
apt-get update
apt-get install -y libibverbs1 ibverbs-providers rdma-core
```

From the mounted repository checkout, run the double-buffered experiment:

```bash
env GLOG_minloglevel=2 \
  TORCHINDUCTOR_COMPILE_THREADS=1 \
  TORCH_NCCL_SHOW_EAGER_INIT_P2P_SERIALIZATION_WARNING=0 \
uv run --no-sync --package flashdreams-lingbot torchrun \
  --standalone --nproc_per_node=8 \
  -m lingbot.disagg.benchmark_pipeline \
  --sessions-per-group 2 \
  --double-buffered \
  --compile-network \
  --warmup-blocks 6 \
  --measured-blocks 5 \
  --bandwidth-probe-iters 10 \
  --output-dir outputs/lingbot_disagg_pipeline_3x2_double_buffered
```

Remove `--double-buffered`, retain `--sessions-per-group 2`, and use
`outputs/lingbot_disagg_pipeline_3x2_fixed_rerun` for the matched control. Six
warmup blocks are required because block 5 changes the cache shape and otherwise
puts compilation in the measurement window. Verify that `nvidia-smi topo -m`
reports NV18, Mooncake logs `installTransport, type=rdma`, and each result
contains three P2P pairs, five measured records, and 72 decoded frames in every
measured wave.

The exact extracted values are in [summary.json](summary.json). Full per-wave
records remain in the generated `outputs/` benchmark directories.

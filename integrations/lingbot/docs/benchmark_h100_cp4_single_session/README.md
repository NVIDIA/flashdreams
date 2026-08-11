# LingBot CP4 single-session disaggregation benchmark

## Result

Topology: **1 encoder : 1 DiT group with CP4 (ulysses) : 1 decoder**.
The 4 DiT ranks cooperate on one autoregressive session.

| Metric | Median | P90 |
| --- | ---: | ---: |
| End-to-end chunk latency | 754.41 ms | 786.46 ms |
| Encoder compute | 0.85 ms | 0.92 ms |
| Encoder → CP leader handoff | 31.70 ms | 32.61 ms |
| CP input fanout | 0.76 ms | 1.54 ms |
| CP DiT critical path | 696.76 ms | 725.92 ms |
| CP leader → decoder handoff | 12.38 ms | 13.01 ms |
| Decoder compute | 7.07 ms | 7.09 ms |
| 256 MiB Mooncake probes | 41.98 GB/s | — |
| 256 MiB-equivalent NCCL broadcast | 307.31 GB/s | 309.31 GB/s |
| 256 MiB-equivalent NCCL all-gather | 389.11 GB/s | 391.99 GB/s |

- Single-session throughput: **15.70 generated FPS**
- Latency speedup versus tracked CP1 baseline: **2.96×**
- DiT critical-path speedup: **3.13×**
- CP scaling efficiency: **78.2%**

The headline excludes 6 warmup blocks and measures
5 blocks. It accelerates one session; it does not represent
independent concurrent sessions.

## Peak allocated memory

| Role | Peak |
| --- | ---: |
| Encoder | 13.72 GiB |
| CP DiT ranks | 40.52–40.64 GiB each |
| Decoder | 2.37 GiB |

## Reproduction

```bash
env TORCHINDUCTOR_COMPILE_THREADS=4 uv run --package flashdreams-lingbot torchrun --standalone --nproc_per_node=6 -m lingbot.disagg.benchmark_cp --cp-ranks 4 --cp-method ulysses --model lingbot-world-fast-taehv-window15-sink3 --warmup-blocks 6 --measured-blocks 5 --bandwidth-probe-mib 256 --bandwidth-probe-iters 8 --output-dir integrations/lingbot/docs/benchmark_h100_cp4_single_session
```

- Repository revision: `66bcd32ece1d03b3362d71a7340691a0687a4069` (modified worktree)
- Slurm: job `14646820` on `pool0-01260`
- GPU: `NVIDIA H100 80GB HBM3` × 6
- Model: `lingbot-world-fast-taehv-window15-sink3`

# LingBot CP6 single-session disaggregation benchmark

## Result

Topology: **1 encoder : 1 DiT group with CP6 (ring) : 1 decoder**.
The six DiT ranks cooperate on one autoregressive session.

| Metric | Median | P90 |
| --- | ---: | ---: |
| End-to-end chunk latency | 743.27 ms | 780.60 ms |
| Encoder compute | 0.90 ms | 1.12 ms |
| Encoder → CP leader handoff | 30.33 ms | 36.77 ms |
| CP input fanout | 0.89 ms | 1.56 ms |
| CP DiT critical path | 683.16 ms | 713.95 ms |
| CP leader → decoder handoff | 12.59 ms | 15.92 ms |
| Decoder compute | 7.13 ms | 7.17 ms |
| 256 MiB Mooncake probes | 42.44 GB/s | — |
| 256 MiB-equivalent NCCL broadcast | 283.71 GB/s | 286.38 GB/s |
| 256 MiB-equivalent NCCL all-gather | 231.72 GB/s | 232.51 GB/s |

- Single-session throughput: **15.90 generated FPS**
- Latency speedup versus tracked CP1 baseline: **3.01×**
- DiT critical-path speedup: **3.19×**
- CP scaling efficiency: **53.2%**

The headline excludes 6 warmup blocks and measures
5 blocks. It accelerates one session; it does not represent
independent concurrent sessions.

## Peak allocated memory

| Role | Peak |
| --- | ---: |
| Encoder | 13.72 GiB |
| CP DiT ranks | 39.18–39.18 GiB each |
| Decoder | 2.29 GiB |

## Reproduction

```bash
env TORCHINDUCTOR_COMPILE_THREADS=4 uv run --package flashdreams-lingbot torchrun --standalone --nproc_per_node=8 -m lingbot.disagg.benchmark_cp --cp-ranks 6 --cp-method ring --model lingbot-world-fast-taehv-window15-sink3 --warmup-blocks 6 --measured-blocks 5 --bandwidth-probe-mib 256 --bandwidth-probe-iters 8 --output-dir integrations/lingbot/docs/benchmark_h100_cp6_single_session
```

- Repository revision: `66bcd32ece1d03b3362d71a7340691a0687a4069` (modified worktree)
- Slurm: job `14646820` on `pool0-01260`
- GPU: `NVIDIA H100 80GB HBM3` × 8
- Model: `lingbot-world-fast-taehv-window15-sink3`

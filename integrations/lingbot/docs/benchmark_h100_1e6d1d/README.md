# LingBot replicated-DiT disaggregation benchmark

## Result

Topology: **1 encoder : 6 DiT : 1 decoder**.
Each DiT worker owns one concurrent session and its resident autoregressive KV cache.

| Metric | Median | P90 |
| --- | ---: | ---: |
| Six-session wave latency | 2657.06 ms | 2671.08 ms |
| Encoder wave | 4.91 ms | 5.21 ms |
| DiT critical path | 2185.88 ms | 2209.48 ms |
| Decoder wave | 42.26 ms | 42.34 ms |
| Encoder → DiT handoff, each | 33.74 ms | 39.20 ms |
| DiT → decoder handoff, each | 31.51 ms | 41.58 ms |
| 256 MiB RDMA probes, all edges | 41.22 GB/s | 42.27 GB/s |

- Aggregate throughput: **27.20 generated FPS**
- Per-session throughput: **4.53 generated FPS**
- Throughput versus tracked 1:1:1 baseline: **5.07×**
- Wave latency versus one-session baseline latency: **1.19×**
- GPU-normalized throughput versus the three-GPU baseline: **1.90×**

The headline excludes 6 warmup waves and measures
5 waves. It represents six concurrent, session-affine
rollouts, not acceleration of one autoregressive session.

## Peak allocated memory

| Role | Peak |
| --- | ---: |
| Shared encoder | 18.77 GiB |
| DiT workers | 56.34–56.51 GiB each |
| Shared decoder | 2.65 GiB |

## Reproduction

```bash
uv run --package flashdreams-lingbot torchrun --standalone --nproc_per_node=8 -m lingbot.disagg.benchmark_replicated --dit-replicas 6 --model lingbot-world-fast-taehv-window15-sink3 --warmup-blocks 6 --measured-blocks 5 --bandwidth-probe-mib 256 --bandwidth-probe-iters 8 --output-dir integrations/lingbot/docs/benchmark_h100_1e6d1d
```

- Repository revision: `08d4c6c159321221c9a2d213c5ebb1359f443ef0` (modified worktree)
- Slurm: job `14628860` on `pool0-00205`
- GPU: `NVIDIA H100 80GB HBM3` × 8
- Model: `lingbot-world-fast-taehv-window15-sink3`

For the allocation method, component and memory chart, raw-result checks, and
the observed Mooncake deregistration warning, see the
[full experiment record](../disaggregated_inference_experiment.md).

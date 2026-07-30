# LingBot three-stage disaggregation benchmark

For the full tested configuration, methodology, findings, Slurm setup, and
limitations, see the
[experiment report](../disaggregated_inference_experiment.md).

## Result

| Metric | Median | P90 |
| --- | ---: | ---: |
| End-to-end chunk latency | 2233.57 ms | 2250.79 ms |
| Encoder compute | 1.08 ms | 1.14 ms |
| DiT denoise | 1734.84 ms | 1755.00 ms |
| DiT cache finalize | 444.79 ms | 446.72 ms |
| Decoder compute | 7.14 ms | 7.15 ms |
| Encoder → DiT handoff | 25.38 ms | 25.88 ms |
| DiT → decoder handoff | 12.05 ms | 14.73 ms |
| Encoder → DiT payload bandwidth | 11.12 GB/s | 11.45 GB/s |
| DiT → decoder payload bandwidth | 0.39 GB/s | 0.52 GB/s |
| 256 MiB encoder → DiT probe | 41.35 GB/s | 41.48 GB/s |
| 256 MiB DiT → decoder probe | 41.00 GB/s | 41.31 GB/s |

Steady-state throughput: **5.36 generated FPS**.

The headline excludes 6 warmup block(s). Mooncake was
configured with the RDMA protocol. Effective payload bandwidth includes the
synchronous transfer call but excludes receiver allocation and control-plane
ticket exchange; handoff timing in `benchmark.json` includes those costs.
The real payloads were 14.36 MiB
(encoder → DiT) and 0.55 MiB
(DiT → decoder). The two synchronous copy calls account for
0.13% of median
chunk latency; complete allocation, metadata, synchronization, and copy
handoffs account for 1.68%.

## Reproduction

```bash
uv run --package flashdreams-lingbot torchrun --standalone --nproc_per_node=3 \
  -m lingbot.disagg.benchmark \
  --model lingbot-world-fast-taehv-window15-sink3 \
  --example-idx 0 --pixel-width 832 --pixel-height 464 --fps 16 \
  --warmup-blocks 6 --measured-blocks 5 \
  --bandwidth-probe-mib 256 --bandwidth-probe-iters 8 \
  --output-dir integrations/lingbot/docs/benchmark_h100_3stage
```

- Repository base: `e580e27d408b3cf8bd8a549f990c361b94d3379f`; the implementation was the worktree change recorded with this report.
- Slurm: job `14621292` on `pool0-01299`
- GPU: `NVIDIA H100 80GB HBM3` × 3
- Resolution: `832x464`
- Model: `lingbot-world-fast-taehv-window15-sink3`

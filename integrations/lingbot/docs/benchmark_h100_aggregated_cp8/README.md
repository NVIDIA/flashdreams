# LingBot aggregated CP8 benchmark

All ranks own the complete encoder, DiT, and decoder pipeline. The DiT token
axis is context-parallel across WORLD with ulysses attention. Encoder
and decoder work is replicated on every rank; there are no RDMA stage
boundaries in this topology.

## Result

| Metric | Median | P90 |
| --- | ---: | ---: |
| End-to-end chunk latency | 393.33 ms | 434.08 ms |
| Encoder critical-rank compute | 0.84 ms | 1.03 ms |
| DiT critical-rank denoise | 309.14 ms | 349.85 ms |
| Decoder critical-rank compute | 6.88 ms | 6.91 ms |
| DiT cache finalize | 76.06 ms | 76.76 ms |
| NCCL broadcast probe | 266.83 GB/s | 268.38 GB/s |
| NCCL all-gather probe | 360.38 GB/s | 361.75 GB/s |

- Generated throughput: **29.50 FPS**
- DiT token throughput: **10739 token/s**
- Peak allocated HBM: **40.88–40.88 GiB per rank**, **327.03 GiB node total**
- Steady allocated HBM after rollout: **310.37 GiB node total**

## Comparison with disaggregated CP

| Metric | Disaggregated CP6 | Aggregated CP8 | Change |
| --- | ---: | ---: | ---: |
| Median chunk latency | 743.27 ms | 393.33 ms | 1.89× faster |
| Generated FPS | 15.90 | 29.50 | 1.86× |
| DiT token throughput | 5995 token/s | 10739 token/s | 1.79× |
| Node peak allocated HBM | 251.07 GiB | 327.03 GiB | 1.30× |

The resolutions differ because the tracked 832×464 grid has 4,524 tokens,
which is not divisible by eight. CP8 uses 832×448 and 4,368 tokens (3.45%
fewer). Token throughput is therefore the fairest compute-rate comparison.

## Reproduction

```bash
env TORCHINDUCTOR_COMPILE_THREADS=4 uv run --package flashdreams-lingbot torchrun --standalone --nproc_per_node=8 -m lingbot.disagg.benchmark_aggregated --cp-method ulysses --model lingbot-world-fast-taehv-window15-sink3 --example-idx 0 --pixel-width 832 --pixel-height 448 --fps 16 --warmup-blocks 6 --measured-blocks 5 --bandwidth-probe-mib 256 --bandwidth-probe-iters 8 --comparison-json integrations/lingbot/docs/benchmark_h100_cp6_single_session/benchmark.json --output-dir outputs/lingbot_aggregated_cp8
```

- Repository revision: `bb67d2868babea53cda7a9027831f26ee948293f` (modified worktree)
- Slurm: job `14652956` on `pool0-01714`
- GPU: `NVIDIA H100 80GB HBM3` × 8
- Resolution: `832x448`
- Warmup / measured blocks: 6 / 5

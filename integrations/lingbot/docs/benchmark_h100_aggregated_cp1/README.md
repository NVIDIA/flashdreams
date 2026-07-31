# LingBot fully aggregated single-H100 benchmark

The complete LingBot encoder, DiT, and LightTAE decoder ran in one process on
one H100 80 GB. There are no RDMA stage boundaries and CP1 has no inter-GPU
attention communication.

## Result

| Metric | Median | P90 |
| --- | ---: | ---: |
| End-to-end 12-frame chunk latency | **2157.51 ms** | **2166.25 ms** |
| Encoder | 0.88 ms | 0.94 ms |
| DiT denoise | 1726.27 ms | 1733.16 ms |
| DiT cache finalize | 424.76 ms | 425.28 ms |
| Decoder | 7.39 ms | 7.49 ms |

- Generated throughput: **5.56 FPS**
- Initialization peak allocated HBM: **66.55 GiB**
- Measured-rollout peak allocated HBM: **59.36 GiB**
- Steady allocated HBM after rollout: **57.15 GiB**

The five measured chunks were 2155.13–2171.30 ms and each emitted 12 frames.
Six preceding chunks were excluded for model compilation, autotuning, cache
fill, and the block-5 cache-shape transition. Inductor rejected several Triton
autotuning candidates that exceeded H100 shared-memory resources and selected
valid fallback kernels; the pipeline did not encounter an HBM out-of-memory
failure.

Compared with the tracked three-GPU, stage-disaggregated CP1 result at the same
832×464 shape, aggregation improved FPS from 5.36 to 5.56 and reduced median
latency from 2233.57 to 2157.51 ms. Removing stage handoffs therefore saved
76.05 ms, or 3.4%, but did not materially change the DiT-dominated latency.

## Reproduction

```bash
./srun.sh

cd /path/to/flashdreams
mkdir -p outputs/lingbot_aggregated_cp1

CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_COMPILE_THREADS=4 GLOG_minloglevel=2 \
uv run --package flashdreams-lingbot torchrun \
  --standalone --nproc_per_node=1 \
  -m lingbot.disagg.benchmark_aggregated \
  --model lingbot-world-fast-taehv-window15-sink3 \
  --example-idx 0 \
  --pixel-width 832 --pixel-height 464 --fps 16 \
  --cp-method ulysses \
  --warmup-blocks 6 --measured-blocks 5 \
  --bandwidth-probe-mib 256 --bandwidth-probe-iters 8 \
  --comparison-json outputs/no-comparison.json \
  --output-dir outputs/lingbot_aggregated_cp1
```

Environment:

- Repository base: `0c2d48a8249577fb617bb5280208dd77409d9b1a`, plus the CP1 benchmark-harness change
- Slurm: job `14790020`, node `pool0-01083`
- GPU: one NVIDIA H100 80 GB HBM3
- PyTorch 2.12.1+cu130, CUDA 13.0, cuDNN 9.2, driver 535.216.03
- BF16, seed 42, four diffusion steps, window 15, sink 3
- Checkpoint: `robbyant/lingbot-world-fast`

The machine-readable result, including all warmup and measured records, is in
[`benchmark.json`](benchmark.json).

# LingBot eight independent aggregated workers

Eight H100s each ran one complete CP1 encoder + DiT + LightTAE decoder pipeline
and one independent session. All workers completed six warmup chunks, waited at
a shared barrier, and then ran five measured 12-frame chunks concurrently.

## Result

| Metric | Value |
| --- | ---: |
| Aggregate generated FPS | **43.44** |
| Per-session FPS, median / p90 | **5.54 / 5.59** |
| Chunk latency, median / p90 | **2163.64 / 2205.53 ms** |
| Shared measurement wall time | 11.050 s |
| Measurement start skew | 0.32 ms |
| Measured sessions / chunks / frames | 8 / 40 / 480 |
| Rollout peak allocated HBM per GPU | **59.35 GiB** |
| Initialization peak allocated HBM per GPU | **66.55 GiB** |
| Steady allocated HBM per GPU | **57.15 GiB** |
| Rollout peak allocated HBM, node total | **474.84 GiB** |
| Initialization peak allocated HBM, node total | **532.38 GiB** |
| Steady allocated HBM, node total | **457.16 GiB** |

The measured aggregate is 97.7% of eight times the tracked 5.56 FPS
single-H100 result. The small gap includes the 0.32 ms start skew and 329.85 ms
finish skew across workers; summing the independently measured worker rates
gives 44.34 FPS, while the stricter shared-window calculation gives the 43.44
FPS headline.

## Serving comparison

| Eight-GPU topology | Sessions | Aggregate FPS | FPS/session | Median latency | Rollout peak node HBM |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 I/O + 7 independent DiTs, pooled async | 7 | 35.15 | 5.02 | 2358.51 ms wave | 415.02 GiB |
| **8 independent full pipelines** | **8** | **43.44** | **5.54 median** | **2163.64 ms** | **474.84 GiB** |
| One aggregated CP8 pipeline | 1 | 29.50 | 29.50 | 393.33 ms | 327.03 GiB |

Eight full replicas deliver 23.6% more aggregate FPS than 1 I/O + 7 DiTs and
10.4% more median FPS per session, while using 14.4% more rollout peak node
HBM. CP8 remains the single-session latency topology; the independent replicas
are the highest-throughput measured topology when eight full pipelines fit.

## Reproduction

From an eight-GPU Slurm node:

```bash
GLOG_minloglevel=2 \
uv run --package flashdreams-lingbot python \
  -m lingbot.disagg.benchmark_independent \
  --replicas 8 \
  --model lingbot-world-fast-taehv-window15-sink3 \
  --example-idx 0 \
  --pixel-width 832 --pixel-height 464 --fps 16 \
  --warmup-blocks 6 --measured-blocks 5 \
  --compile-threads-per-replica 4 \
  --timeout-s 3600 \
  --output-dir outputs/lingbot_aggregated_8xcp1
```

The coordinator assigns one visible GPU to each subprocess. Every subprocess
runs ``benchmark_aggregated`` with one ``torchrun`` rank, creates a readiness
file after warmup, and waits for the common release file. Aggregate FPS is 480
frames divided by the wall time from the earliest worker start to the latest
worker finish.

Environment:

- Repository base: `0c2d48a8249577fb617bb5280208dd77409d9b1a`, plus the benchmark worktree changes
- Slurm: job `14793417`, node `pool0-01151`
- GPU: 8 × NVIDIA H100 80 GB HBM3
- Resolution: 832×464; BF16; seed 42; four diffusion steps; window 15; sink 3
- PyTorch 2.12.1+cu130, CUDA 13.0, cuDNN 9.2, driver 535.216.03

The complete per-worker environment, timing records, stage breakdown, and
memory arrays are in [`benchmark.json`](benchmark.json). Worker logs remain in
the untracked output directory.

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LingBot 1-I/O : 7-DiT optimization benchmark

## Result

The optimized eight-H100 layout co-locates encoder and decoder on GPU 0 and
uses GPUs 1–7 as independent, session-affine DiT workers. Five measured
seven-session waves reached **35.15 aggregate generated FPS** and **5.02 FPS
per session**.

| Metric | 1E:6DiT:1D tracked | 1IO:7DiT synchronous | 1IO:7DiT optimized |
| --- | ---: | ---: | ---: |
| Aggregate FPS | 27.20 | 31.57 | **35.15** |
| FPS per session | 4.53 | 4.51 | **5.02** |
| Median wave | 2657.06 ms | 2671.78 ms | **2358.51 ms** |
| P90 wave | 2671.08 ms | 2705.95 ms | **2454.10 ms** |
| Median 256 MiB RDMA probe | 41.22 GB/s | 42.28 GB/s | **41.97 GB/s** |
| I/O GPU peak HBM | encoder 18.77 + decoder 2.65 GiB on separate GPUs | 20.51 GiB | 20.59 GiB |
| DiT peak HBM, each | 56.34–56.51 GiB | 56.29 GiB | 56.29–56.51 GiB |

The optimized path is **29.2% faster** than the tracked six-DiT topology and
**11.4% faster** than the same co-located seven-DiT topology using synchronous,
per-request allocation and registration. It is 10.9% above the earlier
unvalidated linear projection of 31.7 FPS.

This is aggregate capacity for seven concurrent sessions. It does not make one
autoregressive session run at 35 FPS. Each individual session measured 5.02
generated FPS; the aggregated CP8 baseline remains the one-session latency
choice at 29.50 FPS.

## Wall time and memory

![Optimized LingBot wall-time and HBM breakdown](../disaggregated_inference_optimized.svg)

| Component | Median | P90 | Interpretation |
| --- | ---: | ---: | --- |
| Encoder compute, seven inputs | 5.91 ms | 6.08 ms | Sequential on co-located I/O GPU |
| DiT denoise, worker sample | 1718.78 ms | 1816.21 ms | Seven workers execute concurrently |
| DiT cache finalization, worker sample | 426.45 ms | 436.32 ms | Overlapped with clean-latent RDMA |
| DiT critical path | 2178.39 ms | 2256.77 ms | Slowest worker per wave |
| Decoder compute, seven outputs | 49.06 ms | 49.11 ms | Sequential on co-located I/O GPU |
| End-to-end seven-session wave | 2358.51 ms | 2454.10 ms | Barrier-to-barrier service time |

The encoder and decoder together use only 20.59 GiB, so co-location releases
the eighth GPU for another DiT while leaving substantial HBM headroom. Each
DiT still uses about 56.3 GiB because its weights and autoregressive cache stay
resident and are never transferred.

The asynchronous JSON fields named `transfer_ms` and `handoff_ms` cover the
whole interval from submission until the deliberately delayed wait. They are
**in-flight residency windows**, not isolated copy latency, and are
non-additive because useful compute happens inside those windows. In
particular, the 0.55 MiB DiT-to-decoder wait occurs after roughly 426 ms of
cache finalization. Use the reusable 256 MiB probe—not payload divided by that
residency window—to characterize the link.

## What changed

| Optimization | Implementation | Validation |
| --- | --- | --- |
| Co-locate encoder and decoder | `--co-locate-io`; both stage weights/caches live on rank 0 | Full seven-session rollout |
| Seven independent DiTs | ranks 1–7, one resident cache per session | Full rollout; 35.15 aggregate FPS |
| Async Mooncake | non-blocking batch writes plus explicit wait handles; CUDA event replaces device-wide synchronization | Full rollout; clean completion |
| Receiver/ticket pooling | fixed-shape `RegisteredTensorPool` buckets and stable remote tickets | CPU reuse test and full rollout |
| Finalization overlap | submit clean latent before `finalize`, wait before decoder | Full rollout |
| Session-aware routing | sticky placement by pool, queue prediction, shape/CP compatibility, HBM, rack/NIC locality, and verified RDMA | CPU policy tests |
| Direct CP input shards | patchify once on encoder and transfer each rank's token shard directly | Full CP6 model measurements completed, but handoff grew to 174.23 ms and FPS fell to 12.70; experimental only |
| NIXL transport | interchangeable `NixlTensorTransport` behind the same descriptor/ticket/handle contract | Fake-agent CPU round trip; NIXL was absent from the allocated image |
| DiT microbatch admission | groups independent sessions only when worker, shape, and CP size match | CPU scheduler test; fused model/cache execution remains deferred |
| Separate service pools | `aggregated-cp8` latency pool and `io-plus-7-dit` throughput pool | CPU scheduler test and measured topology comparison |

The synchronous control completed at 31.57 FPS but emitted repeated Mooncake
`remote access error`, `local access violation`, rail-pause, and rail-recovery
messages while short-lived registrations were recycled. The pooled run emitted
none of those errors. Pooling is therefore a buffer-lifetime correctness fix as
well as a performance optimization; the synchronous control is retained only
as diagnostic evidence, not as a production-safe configuration.

## Reproduction

Allocate or reuse one eight-GPU node:

```bash
squeue -u "$USER"
cd /home/gtong/work
export FLASHDREAMS_HOST_DIR=/home/gtong/lustre/flashdreams-dist
./srun.sh
# Reattach instead of allocating again:
./srun.sh 1 <existing-job-id>
```

Inside the node, verify the exact mounted checkout and fabric:

```bash
cd /lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_computervision/users/gtong/flashdreams-dist
git rev-parse --show-toplevel
git rev-parse HEAD
nvidia-smi -L
nvidia-smi topo -m
ibv_devices
```

The experiment used Slurm job `14761875`, node `pool0-01924`, eight NVIDIA
H100 80 GB HBM3 GPUs, driver 535.216.03, PyTorch 2.12.1+cu130, CUDA 13.0,
Mooncake 0.3.12.post1, and base revision
`b762d079245681e1db70f1ffc5728753ce2a90b8` plus this worktree change.
The container needed RDMA userspace libraries:

```bash
apt-get update
apt-get install -y libibverbs1 ibverbs-providers rdma-core
```

Run focused CPU validation:

```bash
uv run --no-sync --package flashdreams-lingbot pytest -q \
  flashdreams/tests/test_transfer.py \
  integrations/lingbot/tests/test_disagg_stages.py \
  integrations/lingbot/tests/test_disagg_scheduler.py
```

Probe every stage edge before model loading:

```bash
env TORCHINDUCTOR_COMPILE_THREADS=1 \
uv run --no-sync --package flashdreams-lingbot torchrun \
  --standalone --nproc_per_node=8 \
  -m lingbot.disagg.benchmark_replicated \
  --dit-replicas 7 \
  --co-locate-io \
  --pooled-async \
  --transport mooncake \
  --transport-only \
  --bandwidth-probe-mib 256 \
  --bandwidth-probe-iters 8
```

Check the logs for Mooncake's RDMA transport installation and RDMA-ready
handshakes. Reject TCP fallback. Then run the model:

```bash
env GLOG_minloglevel=2 TORCHINDUCTOR_COMPILE_THREADS=1 \
uv run --no-sync --package flashdreams-lingbot torchrun \
  --standalone --nproc_per_node=8 \
  -m lingbot.disagg.benchmark_replicated \
  --dit-replicas 7 \
  --co-locate-io \
  --pooled-async \
  --transport mooncake \
  --warmup-blocks 6 \
  --measured-blocks 5 \
  --bandwidth-probe-mib 256 \
  --bandwidth-probe-iters 8 \
  --output-dir outputs/lingbot_disagg_1io7dit_optimized
```

Inspect the raw measurements:

```bash
jq '.environment, .summary' \
  outputs/lingbot_disagg_1io7dit_optimized/benchmark.json
jq '.records[] | select(.warmup == false)' \
  outputs/lingbot_disagg_1io7dit_optimized/benchmark.json
```

Keep six warmup waves: block 5 changes the cache shape and can otherwise put
compilation in the measured set. The measured output must contain five waves,
seven DiT records and 84 decoded frames per wave.

To exercise the NIXL adapter after installing a compatible NIXL release, use
`--transport nixl` first with `--transport-only`, confirm `Backend UCX was
instantiated`, and only then load the model. This allocation did not contain
NIXL, so no real NIXL bandwidth or model result is claimed.

## Deployment decision

Use the disaggregated 1IO:7DiT pool when traffic contains enough simultaneous
interactive sessions to keep the seven DiTs busy, independent stage scaling or
fault isolation matters, and the stage path has verified RDMA. Use aggregated
CP8 when one session needs minimum latency, concurrency is low, or a simple
single-pod deployment is more valuable than stage elasticity. Maintain both
fixed pools for mixed SLAs; model loading, compilation, and cache warmup make
hot-repartitioning an eight-GPU node too expensive.

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Optional LiDAR generation

> Prepared-scene LiDAR rendering uses independently written Apache-2.0 geometry
> and projection implementations. See the [test specification](tests/specs/lidar_geometry_projection.md).

LiDAR is generated alongside RGB from the driven trajectory. **Use RGB-only
for responsive interaction on one GPU**; joint RGB+LiDAR is much slower.
Both modes use the same version 2 scene bundle, which includes `lidar.json`.

Prepared-scene geometry starts at the format's compatibility spacing of 2 cm.
If the map plus the largest actor frame would exceed the point caps, preparation
doubles the spacing until it fits and logs the effective value. The selected
spacing stays fixed throughout the rollout, retains every source primitive and
actor, and keeps the 60-million-point scene cap. Coarser sampling can change
LiDAR returns; 2 cm is a compatibility default, not a measured speed/quality
optimum. Invalid geometry still fails validation; retries are bounded to reject
inputs that cannot fit even at extremely coarse spacing.

## Download model and codec

Complete the [demo setup](README.md#install). Enable Interactive Drive LiDAR
with `--lidar`. The application downloads **and loads** the matching V1.2 codec
from `lidar_vae` in the same Sim-Auto HF repository/revision as the transformer.
Both assets use normal HF authentication and caching during initialization;
no separate codec download or `LIDAR_CHECKPOINT` variable is needed.
The native codec uses vendored vLLM-Omni PyTorch FlexAttention, without NATTEN.

The V1.2 checkpoint is
[`lidar_vae/diffusion_pytorch_model.safetensors`](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Auto/blob/ccde59bc545540cbbb38389a9c18e2347ba679bf/lidar_vae/diffusion_pytorch_model.safetensors)
in `nvidia/Cosmos3-Nano-Sim-Auto`. The repository required private access when
verified; the QA account needs access and a normal HF login:

```bash
uv run --no-sync hf auth login
```

The default model URL uses `main`. For reproducible QA, pin the model URL; the
codec automatically follows the same revision:

```bash
export COSMOS_INTERACTIVE_CHECKPOINT="https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Auto/resolve/ccde59bc545540cbbb38389a9c18e2347ba679bf/model.safetensors.index.json"
```

These model weights use the model card's OpenMDW-1.1 terms, separately from the
Apache/MIT inference source. At the pinned revision the 596,033,216-byte codec
file has SHA-256 `8fc88321323327fcb885d9ff9642f1317c81a96c1758ee99cac7b98f41e557f3`.
All 295 network tensors and both latent statistics exactly match the earlier
`v1p2-iter_000030000.pt` checkpoint used for GPU validation. No conversion or
training checkout is needed.

A standalone CPU check exercises the same default checkpoint selection:

```bash
uv run --no-sync python - <<'PYCODE'
from cosmos_interactive.checkpoint_load import resolve_checkpoint_source
from cosmos_interactive.lidar import LidarTokenizer
from cosmos3_nano_sim_auto.checkpoint import resolve_lidar_checkpoint
from cosmos3_nano_sim_auto.config import DEFAULT_CHECKPOINT
source = resolve_checkpoint_source(None, default=DEFAULT_CHECKPOINT)
checkpoint = resolve_lidar_checkpoint(source)
LidarTokenizer.from_checkpoint(checkpoint, version="v1p2", device="cpu")
print("LiDAR V1.2 checkpoint loaded:", checkpoint)
PYCODE
```

A local transformer snapshot uses its sibling
`lidar_vae/diffusion_pytorch_model.safetensors` automatically. For other exports
or custom codecs, pass `--lidar-checkpoint /path/to/codec.safetensors`; this also
enables Interactive Drive LiDAR. An explicit missing path fails rather than
selecting different weights. V1.1 requires an explicit codec and
`--lidar-tokenizer-version v1p1`.

For recorded batch generation, `--lidar-sample` enables LiDAR with matching RGB
`--sample` inputs; the codec is selected automatically unless overridden with
`--lidar-checkpoint`. The resolved path is also used by the decode worker.

## Run in a browser

From the repository root:

```bash
uv run --no-sync flashdreams-run-v2 interactive-drive-cosmos3-nano-sim-auto \
  --mode webrtc --host 0.0.0.0 --port 8080 \
  --backpressure-mode block --presentation-mode continuous -- \
  --camera camera_front_wide_120fov \
  --world-model-seed 0 --start-frame 0 --compile-denoise off \
  --lidar \
  --culled-lidar-renderer \
  --lidar-view point-cloud
```

Range culling is enabled by default with LiDAR; the explicit flag above makes
the setting visible. Use `--no-culled-lidar-renderer` for the baseline renderer.
The default retains the complete source map and the 2 cm
compatibility spacing, subject to the capacity fallback above. Each sweep
projects only points within its possible 100 m return range, padded for sensor
motion. Near occluders are retained before the 5 m training-validity threshold.
The moving-shutter solve converges each point independently; it can differ from
the baseline renderer's global-mean stopping rule. The additional
`--lidar-codec-tf32 --lidar-contiguous-linears` settings remain **Useful opt-in**
options. TF32 trades matmul precision
for speed; contiguous codec projections retain FP32 and change layout/kernel
dispatch. Use
`--no-culled-lidar-renderer --no-lidar-codec-tf32 --no-lidar-contiguous-linears`
for the previous baseline configuration.

The Slurm launcher's `LIDAR=1` preset already selects these three options. With
four GPUs per node, the 32-GPU launch is:

```bash
sbatch -A <account> -p <partition> --nodes 8 \
  --export ALL,MODE=drive,LIDAR=1,NPROC=4 \
  integrations_v2/cosmos3_nano_sim_auto/probes/launch.sbatch
```

Up to four ranks render a chunk's four sweeps, preferably on ranks without RGB
cameras, while camera ranks prepare RGB controls. One owner gathers and encodes
the LiDAR scans. Joint inference consumes completed controls for the current
supplied poses; how much preparation is hidden depends on the hardware and
collective timing. Validate overlap on the target 32-GPU mesh before treating
rendering as absent from the critical path.

Single-GPU validation on RTX PRO 6000 Blackwell, Torch 2.12.1+cu130, used the
complete residential scene, one front-wide camera plus LiDAR, HF V1.2 codec,
seed 0, start frame 0, neutral coasting and `--compile-denoise off`. Each profile
ran in a fresh process with warm checkpoint/compiler caches for 17 chunks;
the first four were excluded, leaving 13 measured 12-frame chunks. Nsight
profiling was collected separately from these synchronized timing runs:

| Configuration | LiDAR render median | Total chunk median / p90 |
| --- | --- | --- |
| Baseline | 1.444 s | 4.396 / 4.537 s |
| Local culling, FP32 codec | 0.287 s | 3.267 / 3.347 s |
| Local culling, TF32 and contiguous codec projections | 0.281 s | 3.055 / 3.124 s |

All three produced 193-frame RGB+LiDAR videos. On six matched stationary/moving
source poses, local culling matched the full-map per-point solver's output maps
exactly. A separate same-latent codec comparison measured TF32/contiguous versus
baseline: worst chunk p99 range difference 2.06 cm, maximum difference 28.42 cm,
and 77 validity flips among 4,838,400 cells (0.0016%). These are one-hardware
control/codec checks and a coasting video smoke test; broader field quality and
32-GPU throughput/overlap remain unvalidated.

For seven views, replace the `--camera` value with the seven-camera list in
[README.md](README.md#run-in-a-browser). Open `http://<server-address>:8080/`.
Choose range or intensity at launch with `--lidar-view range` or
`--lidar-view intensity`; HDMAP displays the control scan.
Output timestamps use 30 FPS for RGB and 10 Hz for LiDAR;
inference throughput depends on the hardware.

The codec accepts local `.pt` and `.safetensors` checkpoints. Use `.pt` files
only from trusted sources because loading their metadata requires unpickling.

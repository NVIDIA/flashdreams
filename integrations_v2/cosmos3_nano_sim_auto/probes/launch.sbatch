#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Interactive Drive (MODE=drive, default) or a recorded-trajectory video
# (MODE=batch) on --nodes x 4 GPUs. Submit from the repository root after
# installation in the README; built-in scenes download automatically. Arguments
# after the script go to the runner after its `--`; append --no-ui to hide the HUD.
# Set LIDAR=1 (and a LiDAR SCENE) for RGB+LiDAR
# Interactive Drive with the matching HF codec; LIDAR_CHECKPOINT overrides it:
#
#   sbatch -A <account> -p <partition> --nodes 4 integrations_v2/cosmos3_nano_sim_auto/probes/launch.sbatch
#   sbatch -A <account> -p <partition> --nodes 2 --export ALL,MODE=batch \
#     integrations_v2/cosmos3_nano_sim_auto/probes/launch.sbatch --frames 193
#
# Add --gpus-per-node=4 where the cluster schedules GPUs as GRES. The first
# launch compiles for about 12 minutes; later ones reuse .cache/inductor.
# Seven views on GB200, steady FPS per camera:
#   GPUs          1    2    4    8   16   32
#   batch       4.6  7.4 12.1 18.7 19.2 20.0
#   drive         -    - 12.5 17.7 20.8 23.1   (RGB+LiDAR at 16: 16.3)
#SBATCH --job-name=cosmos3-nano-sim-auto
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail

export COSMOS3_NANO_SIM_AUTO_TENSOR_PARALLEL=1
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$PWD/.cache/inductor}"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
SCENE="${SCENE:-clipgt-residential-iter1500}"
CAMERAS=camera_front_wide_120fov,camera_cross_right_120fov,camera_rear_right_70fov,camera_rear_tele_30fov,camera_rear_left_70fov,camera_cross_left_120fov,camera_front_tele_30fov
FAST=(--parallel-decode --decoder-compile --compile-denoise default)

case "${MODE:-drive}" in
  drive)
    FAST+=(--compile-control-encode)
    if [ "${LIDAR:-0}" = 1 ] || [ -n "${LIDAR_CHECKPOINT:-}" ]; then
      FAST+=(--lidar --lidar-tokenizer-version v1p2
             --culled-lidar-renderer --lidar-codec-tf32 --lidar-contiguous-linears)
      if [ -n "${LIDAR_CHECKPOINT:-}" ]; then
        FAST+=(--lidar-checkpoint "$LIDAR_CHECKPOINT")
      fi
    elif [ "$SLURM_NNODES" -ge 4 ]; then  # 16+ GPUs: half the GPUs draw the map
      : "${NPROC:=2}"
      FAST+=(--prefetch-controls --prefetch-controls-gpu-offset "$NPROC")
    fi
    echo "Interactive Drive: http://$MASTER_ADDR:8080/"
    RUN=(interactive-drive-cosmos3-nano-sim-auto --mode webrtc --host 0.0.0.0 --port 8080
         --backpressure-mode block --presentation-mode continuous --
         --scene "$SCENE" --camera "$CAMERAS" --world-model-seed 0 --start-frame 0) ;;
  batch)
    RUN=(cosmos3-nano-sim-auto --output-path "${OUTPUT:-mv-$SLURM_JOB_ID.mp4}" --timeout unbound --
         --scene "$SCENE" --views 7) ;;
  *) echo "MODE must be drive or batch" >&2; exit 2 ;;
esac
export NPROC="${NPROC:-4}"

srun --kill-on-bad-exit=1 bash -c 'exec uv run --no-sync torchrun --nnodes "$SLURM_NNODES" \
  --nproc_per_node "$NPROC" --node_rank "$SLURM_NODEID" --master_addr "$MASTER_ADDR" \
  --master_port 29500 --no-python flashdreams-run-v2 "$@"' _ "${RUN[@]}" "${FAST[@]}" "$@"

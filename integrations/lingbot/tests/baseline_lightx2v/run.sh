#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
REPO_DIR="${SCRIPT_DIR}/LightX2V"
REPO_URL="https://github.com/ModelTC/LightX2V.git"
PIN_COMMIT="caf1056d042931a50bf14c95d1501d3a5be926fa"
PATCH_FILE="${SCRIPT_DIR}/changes.patch"
VENV_PYTHON="${SCRIPT_DIR}/.venv/bin/python"

MODEL_DIR="${SCRIPT_DIR}/lingbot-world-base-cam"
MODEL_FAST_DIR="${MODEL_DIR}/lingbot_world_fast"
RESULT_PATH="${REPO_DIR}/save_results/output_lightx2v_lingbot_fast_i2v.mp4"
RUN_CONFIG_JSON="${SCRIPT_DIR}/lingbot_fast_i2v.runtime.json"
RESULT_PATH_4GPU="${REPO_DIR}/save_results/output_lightx2v_lingbot_fast_i2v_4gpu.mp4"
RUN_CONFIG_JSON_4GPU="${SCRIPT_DIR}/lingbot_fast_i2v_4gpu.runtime.json"
LOG_DIR="${SCRIPT_DIR}/logs"
EXAMPLES_DIR="${SCRIPT_DIR}/lingbot-world-examples/00"
EXAMPLES_BASE_URL="https://raw.githubusercontent.com/Robbyant/lingbot-world/main/examples/00"
PROMPT_PATH="${EXAMPLES_DIR}/prompt.txt"

if [[ ! -d "${REPO_DIR}/.git" ]]; then
    echo "[setup] cloning ${REPO_URL} -> ${REPO_DIR}"
    git clone "${REPO_URL}" "${REPO_DIR}"
else
    echo "[setup] repo already present at ${REPO_DIR}, skipping clone"
fi

cd "${REPO_DIR}"

CURRENT_COMMIT="$(git rev-parse HEAD)"
if [[ "${CURRENT_COMMIT}" != "${PIN_COMMIT}" ]]; then
    echo "[setup] checking out pinned commit ${PIN_COMMIT}"
    git checkout "${PIN_COMMIT}"
else
    echo "[setup] already at pinned commit ${PIN_COMMIT}, skipping checkout"
fi

if [[ -f "${PATCH_FILE}" ]]; then
    if git apply --reverse --check "${PATCH_FILE}" >/dev/null 2>&1; then
        echo "[setup] patch already applied, skipping"
    elif git apply --check "${PATCH_FILE}" >/dev/null 2>&1; then
        echo "[setup] applying ${PATCH_FILE}"
        git apply "${PATCH_FILE}"
    else
        echo "[setup] ERROR: ${PATCH_FILE} neither applies cleanly nor is already applied." >&2
        exit 1
    fi
else
    echo "[setup] no patch file at ${PATCH_FILE}, skipping"
fi

echo "[setup] ensuring Python deps via uv sync (isolated venv)"
( cd "${SCRIPT_DIR}" && uv sync )

if [[ ! -d "${MODEL_DIR}" ]] || [[ -z "$(ls -A "${MODEL_DIR}" 2>/dev/null)" ]]; then
    echo "[setup] downloading lingbot-world-base-cam"
    "${SCRIPT_DIR}/.venv/bin/hf" download \
        robbyant/lingbot-world-base-cam \
        --local-dir "${MODEL_DIR}"
else
    echo "[setup] lingbot-world-base-cam exists, skipping download"
fi

if [[ ! -d "${MODEL_FAST_DIR}" ]] || [[ -z "$(ls -A "${MODEL_FAST_DIR}" 2>/dev/null)" ]]; then
    echo "[setup] downloading lingbot-world-fast"
    "${SCRIPT_DIR}/.venv/bin/hf" download \
        robbyant/lingbot-world-fast \
        --local-dir "${MODEL_FAST_DIR}"
else
    echo "[setup] lingbot-world-fast exists, skipping download"
fi

mkdir -p "${EXAMPLES_DIR}"
for file in image.jpg action.npy intrinsics.npy poses.npy prompt.txt; do
    if [[ ! -f "${EXAMPLES_DIR}/${file}" ]]; then
        echo "[setup] downloading examples/00/${file}"
        curl -fsSL "${EXAMPLES_BASE_URL}/${file}" -o "${EXAMPLES_DIR}/${file}"
    fi
done

PROMPT="$(tr -d '\r' < "${PROMPT_PATH}")"
IMAGE_PATH="${LINGBOT_IMAGE_PATH:-${EXAMPLES_DIR}/image.jpg}"
ACTION_PATH="${LINGBOT_ACTION_PATH:-${EXAMPLES_DIR}}"

mkdir -p "${REPO_DIR}/save_results"
mkdir -p "${LOG_DIR}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_PATH_1GPU="${LOG_DIR}/run_1gpu_${RUN_TS}.log"
LOG_PATH_4GPU="${LOG_DIR}/run_4gpu_${RUN_TS}.log"

echo "[setup] generating runtime config with torch_sdpa attention"
"${VENV_PYTHON}" - <<PY
import json
from pathlib import Path

src = Path("${REPO_DIR}/configs/lingbot_fast/lingbot_fast_i2v.json")
dst = Path("${RUN_CONFIG_JSON}")
cfg = json.loads(src.read_text())
cfg["self_attn_1_type"] = "torch_sdpa"
cfg["cross_attn_1_type"] = "torch_sdpa"
cfg["cross_attn_2_type"] = "torch_sdpa"
# Native torch-compile support is not fully wired for this LingBot fast
# model variant in the pinned LightX2V commit (no stable graph-selection path),
# so keep compile disabled for this harness.
cfg["compile"] = False
cfg["target_height"] = 480
cfg["target_width"] = 832
cfg["target_video_length"] = 237
dst.write_text(json.dumps(cfg, indent=4) + "\n")
cfg_4gpu = dict(cfg)
cfg_4gpu["parallel"] = {
    "cfg_p_size": 1,
    "seq_p_size": 4,
    "seq_p_attn_type": "ulysses",
}
dst_4gpu = Path("${RUN_CONFIG_JSON_4GPU}")
dst_4gpu.write_text(json.dumps(cfg_4gpu, indent=4) + "\n")
print(f"[setup] wrote {dst}")
print(f"[setup] wrote {dst_4gpu}")
PY

echo "[run] starting LightX2V LingBot fast i2v baseline [1 GPU]"
lightx2v_path="${REPO_DIR}" \
model_path="${MODEL_DIR}" \
PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}" \
MOONCAKE_CONFIG_PATH="${REPO_DIR}/configs/mooncake_config.json" \
TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTORCH_ALLOC_CONF=expandable_segments:True \
DTYPE=BF16 \
SENSITIVE_LAYER_DTYPE=None \
PROFILING_DEBUG_LEVEL=2 \
"${VENV_PYTHON}" "${SCRIPT_DIR}/run_lingbot_fast_i2v_minimal.py" \
    --model_cls lingbot_world_fast \
    --task i2v \
    --model_path "${MODEL_DIR}" \
    --dit_original_ckpt "${MODEL_FAST_DIR}" \
    --config_json "${RUN_CONFIG_JSON}" \
    --prompt "${PROMPT}" \
    --negative_prompt "" \
    --image_path "${IMAGE_PATH}" \
    --action_path "${ACTION_PATH}" \
    --save_result_path "${RESULT_PATH}" \
    2>&1 | tee "${LOG_PATH_1GPU}"

echo "[run] starting LightX2V LingBot fast i2v baseline [4 GPUs]"
lightx2v_path="${REPO_DIR}" \
model_path="${MODEL_DIR}" \
PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}" \
MOONCAKE_CONFIG_PATH="${REPO_DIR}/configs/mooncake_config.json" \
TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTORCH_ALLOC_CONF=expandable_segments:True \
DTYPE=BF16 \
SENSITIVE_LAYER_DTYPE=None \
PROFILING_DEBUG_LEVEL=2 \
"${VENV_PYTHON}" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
    "${SCRIPT_DIR}/run_lingbot_fast_i2v_minimal.py" \
    --model_cls lingbot_world_fast \
    --task i2v \
    --model_path "${MODEL_DIR}" \
    --dit_original_ckpt "${MODEL_FAST_DIR}" \
    --config_json "${RUN_CONFIG_JSON_4GPU}" \
    --prompt "${PROMPT}" \
    --negative_prompt "" \
    --image_path "${IMAGE_PATH}" \
    --action_path "${ACTION_PATH}" \
    --save_result_path "${RESULT_PATH_4GPU}" \
    2>&1 | tee "${LOG_PATH_4GPU}"

echo "[run] logs saved to:"
echo "  - ${LOG_PATH_1GPU}"
echo "  - ${LOG_PATH_4GPU}"

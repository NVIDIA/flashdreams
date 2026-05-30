#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Native-only bench loop over every image in ``data_local/`` (or a
# user-supplied IMAGES_DIR). For each image, runs the in-tree
# HY-WorldPlay WAN-5B I2V plugin, writes an mp4 + per-AR-step stats
# JSON, and at the end aggregates all stats into one ``bench_all.md``
# perf table -- the PR-attachable summary.
#
# Per-image prompts: drop a sidecar ``<stem>.txt`` next to each image
# (e.g. ``data_local/cat_surf.txt`` next to ``data_local/cat_surf.jpg``).
# Its first non-empty line is used as the prompt for that image. When
# no sidecar exists, the global ``PROMPT`` env var (or its built-in
# default) is used.
#
# Required env setup before first invocation:
#   1. uv installed   : curl -LsSf https://astral.sh/uv/install.sh | sh
#   2. HF auth token  : export HF_TOKEN=<your-token>   (read access to
#                       Wan-AI/Wan2.2-TI2V-5B-Diffusers + tencent/HY-WorldPlay)
#   3. Workspace sync : ( cd integrations/hy_worldplay/tests/parity_check && uv sync )
#
# Vendor wrapper is intentionally not run here. For the side-by-side
# native-vs-vendor comparison use ``bench.sh`` (which still drives
# upstream's ``wan/generate.py`` via ``run.sh``).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

IMAGES_DIR="${IMAGES_DIR:-${REPO_ROOT}/data_local}"
HF_MODELS_DIR="${HF_MODELS_DIR:-${SCRIPT_DIR}/HY-WorldPlay/hf_models}"
CKPT_PATH="${CKPT_PATH:-${HF_MODELS_DIR}/wan_distilled_model/model.pt}"

NUM_CHUNK="${NUM_CHUNK:-2}"
# ``w-N`` is N motion steps; with the initial-frame identity pose
# prepended this produces ``N + 1`` latents. The rollout consumes
# ``num_chunk * 4`` latents, so ``POSE`` must equal ``num_chunk * 4 - 1``.
POSE="${POSE:-w-7}"
SEED="${SEED:-0}"
PROMPT="${PROMPT:-First-person view walking around ancient Athens, with Greek architecture and marble structures}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/bench_batch}"

# ----------------------------------------------------------------- preflight
if ! command -v uv >/dev/null 2>&1; then
    echo "[bench_batch] ERROR: uv not on PATH." >&2
    echo "        Install via: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi
if [[ ! -d "${IMAGES_DIR}" ]]; then
    echo "[bench_batch] ERROR: IMAGES_DIR=${IMAGES_DIR} does not exist." >&2
    exit 1
fi

# ------------------------------------------------------ distilled checkpoint
# Download HY-WorldPlay's distilled wan_distilled_model on demand. Skip
# silently when already present so re-runs are no-op-fast. ``--ckpt-path``
# is technically optional (a missing checkpoint leaves the HY conditioners
# zero-init and the pipeline produces a base Wan 2.2 TI2V-5B identity
# rollout, useless for a perf vs identity comparison), so the harness
# treats the download as a hard prerequisite.
if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "[bench_batch] distilled checkpoint missing at ${CKPT_PATH}; downloading"
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "[bench_batch] ERROR: HF_TOKEN not set; cannot fetch tencent/HY-WorldPlay." >&2
        echo "        export HF_TOKEN=<your-token> first." >&2
        exit 1
    fi
    ( cd "${SCRIPT_DIR}" && uv run hf download tencent/HY-WorldPlay \
        --include "wan_distilled_model/*" \
        --local-dir "${HF_MODELS_DIR}" )
fi

mkdir -p "${OUTPUT_DIR}"

# ----------------------------------------------------------------- discover
# Collect every common image extension. ``-iname`` covers ``.JPG`` vs
# ``.jpg`` etc. ``sort`` so the per-image rows in the final summary stay
# in a deterministic order.
mapfile -t IMAGES < <(
    find "${IMAGES_DIR}" -maxdepth 1 -type f \
        \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) \
        | sort
)
if [[ "${#IMAGES[@]}" -eq 0 ]]; then
    echo "[bench_batch] ERROR: no .jpg/.jpeg/.png files in ${IMAGES_DIR}." >&2
    exit 1
fi
echo "[bench_batch] discovered ${#IMAGES[@]} image(s) under ${IMAGES_DIR}"

# ------------------------------------------------------------------ per-image
for img in "${IMAGES[@]}"; do
    stem="$(basename "${img}")"
    stem="${stem%.*}"
    out="${OUTPUT_DIR}/${stem}"
    mkdir -p "${out}"

    # Per-image prompt sidecar (``<stem>.txt`` next to the image);
    # fall back to the global ``PROMPT`` env var when absent.
    prompt_file="${IMAGES_DIR}/${stem}.txt"
    if [[ -f "${prompt_file}" ]]; then
        # First non-empty line, trimmed.
        image_prompt="$(awk 'NF { sub(/^[[:space:]]+/, ""); sub(/[[:space:]]+$/, ""); print; exit }' "${prompt_file}")"
        prompt_src="sidecar ${stem}.txt"
    else
        image_prompt="${PROMPT}"
        prompt_src="global PROMPT default"
    fi

    echo "[bench_batch] -> ${stem} (num_chunk=${NUM_CHUNK}, pose=${POSE}, seed=${SEED}, prompt: ${prompt_src})"
    ( cd "${SCRIPT_DIR}" && uv run flashdreams-run hy-worldplay-wan-i2v-5b \
        --image-path "${img}" \
        --ckpt-path "${CKPT_PATH}" \
        --prompt "${image_prompt}" \
        --num-chunk "${NUM_CHUNK}" \
        --pose "${POSE}" \
        --seed "${SEED}" \
        --output-dir "${out}" )
done

# ----------------------------------------------------------------- aggregate
echo "[bench_batch] aggregating -> ${OUTPUT_DIR}/bench_all.md"
( cd "${SCRIPT_DIR}" && uv run python "${SCRIPT_DIR}/bench_batch_summary.py" \
    --output-dir "${OUTPUT_DIR}" \
    --report "${OUTPUT_DIR}/bench_all.md" \
    --num-chunk "${NUM_CHUNK}" \
    --pose "${POSE}" \
    --seed "${SEED}" )

echo "[bench_batch] done. per-image mp4s + stats under ${OUTPUT_DIR}/<image>/"
echo "              summary: ${OUTPUT_DIR}/bench_all.md"

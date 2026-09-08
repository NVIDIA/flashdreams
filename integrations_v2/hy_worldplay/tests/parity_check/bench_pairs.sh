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

# Drive ``bench.sh`` once per image in ``IMAGES_DIR``. Each invocation
# produces a native + vendor MP4 pair plus a per-image ``bench.md``
# (perf + parity); at the end this script concatenates every per-image
# ``bench.md`` into a single ``bench_all.md`` for direct PR attachment.
#
# Use this instead of ``bench_batch.sh`` when you want both legs per
# image (the parity-check ask). ``bench_batch.sh`` stays native-only
# for the fast "perf across many images" path.
#
# Per-image prompts: drop a sidecar ``<stem>.txt`` next to each image
# (e.g. ``data_local/cat_surf.txt`` next to ``data_local/cat_surf.jpg``).
# Its first non-empty line is used as the prompt for that image's
# bench.sh invocation. When no sidecar exists, the global ``PROMPT``
# env var (or bench.sh's built-in default) is used.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

IMAGES_DIR="${IMAGES_DIR:-${REPO_ROOT}/data_local}"
# Match bench.sh's VRAM-constrained defaults: ``num_chunk=2`` is the
# largest that fits both legs in 44 GiB on RTX 6000 Ada. 8 DiT
# forwards per side; on a >=80 GiB GPU bump to 8 with WARMUP_CHUNKS=5.
NUM_CHUNK="${NUM_CHUNK:-2}"
POSE="${POSE:-w-7}"
SEED="${SEED:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/bench_pairs}"

if [[ ! -d "${IMAGES_DIR}" ]]; then
    echo "[bench_pairs] ERROR: IMAGES_DIR=${IMAGES_DIR} does not exist." >&2
    exit 1
fi

# Sort for stable ordering in the final aggregated report.
mapfile -t IMAGES < <(
    find "${IMAGES_DIR}" -maxdepth 1 -type f \
        \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) \
        | sort
)
if [[ "${#IMAGES[@]}" -eq 0 ]]; then
    echo "[bench_pairs] ERROR: no .jpg/.jpeg/.png files in ${IMAGES_DIR}." >&2
    exit 1
fi
echo "[bench_pairs] discovered ${#IMAGES[@]} image(s) under ${IMAGES_DIR}"

mkdir -p "${OUTPUT_DIR}"

# ---------------------------------------------------------------- per-image
for img in "${IMAGES[@]}"; do
    stem="$(basename "${img}")"
    stem="${stem%.*}"
    echo
    echo "================== ${stem} =================="

    # Per-image prompt sidecar. If ``<stem>.txt`` is next to the image,
    # use its first non-empty line; otherwise pass through ``PROMPT``
    # (or let bench.sh fall back to its own default if ``PROMPT`` is
    # unset).
    prompt_file="${IMAGES_DIR}/${stem}.txt"
    if [[ -f "${prompt_file}" ]]; then
        image_prompt="$(awk 'NF { sub(/^[[:space:]]+/, ""); sub(/[[:space:]]+$/, ""); print; exit }' "${prompt_file}")"
        echo "[bench_pairs] using prompt from ${prompt_file}"
    else
        image_prompt="${PROMPT:-}"
    fi

    # Only set PROMPT in the child env when we have a concrete value;
    # otherwise let bench.sh apply its built-in default.
    if [[ -n "${image_prompt}" ]]; then
        PROMPT="${image_prompt}" IMAGE_PATH="${img}" \
            NUM_CHUNK="${NUM_CHUNK}" POSE="${POSE}" SEED="${SEED}" \
            OUTPUT_DIR="${OUTPUT_DIR}/${stem}" \
            bash "${SCRIPT_DIR}/bench.sh"
    else
        IMAGE_PATH="${img}" NUM_CHUNK="${NUM_CHUNK}" POSE="${POSE}" SEED="${SEED}" \
            OUTPUT_DIR="${OUTPUT_DIR}/${stem}" \
            bash "${SCRIPT_DIR}/bench.sh"
    fi
done

# ----------------------------------------------------------------- aggregate
# Concatenate every per-image ``bench.md`` (in sorted-stem order) into
# one top-level report. ``bench.md`` already carries its own H1 header
# per image, so a plain ``cat`` produces a navigable doc.
echo
echo "[bench_pairs] aggregating per-image bench.md -> ${OUTPUT_DIR}/bench_all.md"
{
    echo "# HY-WorldPlay WAN-5B I2V: native vs vendor bench across ${#IMAGES[@]} images"
    echo
    echo "Settings: \`num_chunk=${NUM_CHUNK}\`, \`pose=${POSE}\`, \`seed=${SEED}\`."
    echo
    for img in "${IMAGES[@]}"; do
        stem="$(basename "${img}")"
        stem="${stem%.*}"
        per_image_md="${OUTPUT_DIR}/${stem}/bench.md"
        if [[ -f "${per_image_md}" ]]; then
            echo "---"
            echo
            cat "${per_image_md}"
            echo
        else
            echo "---"
            echo
            echo "## ${stem}: bench.md missing (run failed?)"
            echo
        fi
    done
} > "${OUTPUT_DIR}/bench_all.md"

echo "[bench_pairs] done. per-image artifacts under ${OUTPUT_DIR}/<stem>/"
echo "              aggregated summary: ${OUTPUT_DIR}/bench_all.md"

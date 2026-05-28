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

# Native plugin vs upstream ``wan/generate.py`` bench, matched first
# frame and seed.
#
# Collects the two MP4s and a per-side stats JSON into one tree, then
# emits ``bench.md`` summarising perf (elapsed, peak GPU mem, native
# per-chunk ms) and parity (mean / max |Delta| between the two MP4s)
# for direct attachment to the integration PR.
#
# The vendor leg shells out to ``run.sh`` (which invokes upstream's
# ``wan/generate.py`` directly via torchrun); the plugin no longer
# contains a vendor wrapper. The vendor stats JSON is synthesised here
# from wall-clock timing -- upstream's script does not emit one.
#
# Pose strings:
#   Both impls share the same convention -- ``w-N`` produces ``N + 1``
#   latents (one identity pose for the input frame plus N motions).
#   The only difference is strictness: native asserts
#   ``len(pose_json) == num_chunk * 4``; upstream truncates past
#   ``num_chunk * CHUNK_SIZE`` silently. Pass the same ``POSE`` to both
#   legs; pick ``N == num_chunk * 4 - 1``.
#
# Noise alignment:
#   ``HY_VENDOR_NOISE_MODE=1`` on the native leg pre-draws diffusion
# noise from the rollout seed in vendor's ``prepare_latents`` layout so
# native and vendor consume bit-identical noise tensors per chunk. This
# is the regime the 15.65 / 255 headline parity number was measured in;
# set ``HY_VENDOR_NOISE_MODE=0`` to compare under each side's private
# RNG instead.
#
# Vendor baseline:
#   ``USE_KV_CACHE_TRUE=1`` swaps vendor's default single-forward-pass
# code path for the cache-prefill path (``use_kv_cache=True``) -- the
# regime our native runner mirrors. This is the phase 2b.6 acceptance
# baseline (the 15.65/255 measurement). Setting ``USE_KV_CACHE_TRUE=0``
# compares against vendor's default single-pass path instead, which
# uses a different inference scheme and produces a much larger drift
# (~27/255 on the cat_surf.jpg smoke).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

HY_REPO_DIR="${HY_REPO_DIR:-${SCRIPT_DIR}/HY-WorldPlay}"
HF_MODELS_DIR="${HF_MODELS_DIR:-${HY_REPO_DIR}/hf_models}"
CKPT_PATH="${CKPT_PATH:-${HF_MODELS_DIR}/wan_distilled_model/model.pt}"

IMAGE_PATH="${IMAGE_PATH:-${REPO_ROOT}/data_local/cat_surf.jpg}"
# ``num_chunk=6`` gives 6 chunks * 4 denoising steps = 24 DiT forwards
# per side. After discarding the first 5 chunks (20 forwards) per
# manager's "exclude first 5 AR steps" spec, 4 DiT samples remain --
# the minimum for a "median" statistic but the largest we can fit:
# vendor OOMs on 44 GiB at ``num_chunk>=7`` (the compiled-block cache
# + KV memory pool exceed VRAM there). For more samples on a larger
# GPU, bump ``NUM_CHUNK`` and ``POSE``.
NUM_CHUNK="${NUM_CHUNK:-6}"
# Native pose. ``num_chunk * 4 - 1`` motion steps (the parser prepends
# an identity for the input frame).
POSE="${POSE:-w-23}"
SEED="${SEED:-0}"
PROMPT="${PROMPT:-First-person view walking around ancient Athens, with Greek architecture and marble structures}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/bench}"
HY_VENDOR_NOISE_MODE="${HY_VENDOR_NOISE_MODE:-1}"
USE_KV_CACHE_TRUE="${USE_KV_CACHE_TRUE:-1}"
# DiT-only timer + matched cuDNN SDPA on vendor: Option A
# methodology for the post-warmup median comparison. Vendor
# ``torch.compile`` was dropped from this matrix -- vendor's KV cache
# uses dynamic shapes per chunk that Inductor can't trace cleanly.
# Native ships ``compile_network=True`` regardless.
HY_DIT_PROFILE="${HY_DIT_PROFILE:-1}"
HY_VENDOR_CUDNN_SDPA="${HY_VENDOR_CUDNN_SDPA:-1}"
# Chunks at the start of the rollout to drop from the DiT median
# (Inductor autotune + CUDA-graph capture land in the first few).
WARMUP_CHUNKS="${WARMUP_CHUNKS:-5}"

NATIVE_OUT="${OUTPUT_DIR}/native"
VENDOR_OUT="${OUTPUT_DIR}/vendor"

RUNNER_NAME="hy-worldplay-wan-i2v-5b"

if [[ ! -f "${IMAGE_PATH}" ]]; then
    echo "[bench] ERROR: IMAGE_PATH=${IMAGE_PATH} does not exist." >&2
    exit 1
fi
# Absolutize path-like env vars before handing them to ``run.sh`` /
# ``flashdreams-run``; both ``cd`` into other directories before they
# resolve the path, so a relative path would otherwise look under the
# wrong root.
IMAGE_PATH="$(readlink -f "${IMAGE_PATH}")"
CKPT_PATH="$(readlink -m "${CKPT_PATH}")"
OUTPUT_DIR="$(readlink -m "${OUTPUT_DIR}")"
NATIVE_OUT="${OUTPUT_DIR}/native"
VENDOR_OUT="${OUTPUT_DIR}/vendor"
# ``run.sh`` (invoked below for the vendor leg) clones the upstream
# tree and downloads HY-WorldPlay's checkpoints on demand, so we don't
# preflight-check ``HY_REPO_DIR`` or ``CKPT_PATH`` here -- the first
# bench invocation is allowed to bootstrap both.

mkdir -p "${NATIVE_OUT}" "${VENDOR_OUT}"

## -------------------------------------------------------------- vendor leg
# Drive upstream's ``wan/generate.py`` via the canonical entry point
# (``run.sh``). ``USE_KV_CACHE_TRUE=1`` selects the cache-prefill code
# path our native runner mirrors. Time it ourselves to synthesise the
# vendor stats JSON.
#
# Vendor's memory-prefill KV pool grows per chunk + cuDNN attention
# holds a workspace; together they OOM on 44 GiB at ``num_chunk>=8``
# without the expandable allocator. ``expandable_segments`` defrags
# the allocator in-process; cheap to leave on for both legs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Clear prior outputs so the ``find`` below only sees this run's fresh
# upstream-pattern mp4 (``<pose>_<prompt>.mp4``) and doesn't trip on
# the already-renamed ``${RUNNER_NAME}.mp4`` from a previous bench.
rm -f "${VENDOR_OUT}/${RUNNER_NAME}.mp4" "${VENDOR_OUT}"/*.mp4 \
      "${VENDOR_OUT}/stats_${RUNNER_NAME}.json"
echo "[bench] running VENDOR leg via run.sh (USE_KV_CACHE_TRUE=${USE_KV_CACHE_TRUE}, CUDNN_SDPA=${HY_VENDOR_CUDNN_SDPA}) -> ${VENDOR_OUT}"
_vendor_start_s="$(date +%s)"
NUM_CHUNK="${NUM_CHUNK}" POSE="${POSE}" SEED="${SEED}" \
    PROMPT="${PROMPT}" IMAGE_PATH="${IMAGE_PATH}" \
    OUTPUT_DIR="${VENDOR_OUT}" \
    USE_KV_CACHE_TRUE="${USE_KV_CACHE_TRUE}" \
    HY_DIT_PROFILE="${HY_DIT_PROFILE}" \
    HY_VENDOR_CUDNN_SDPA="${HY_VENDOR_CUDNN_SDPA}" \
    HY_DIT_OUTPUT_JSON="${VENDOR_OUT}/stats_dit_vendor.json" \
    bash "${SCRIPT_DIR}/run.sh"
_vendor_elapsed_s=$(( $(date +%s) - _vendor_start_s ))

# ``wan/generate.py`` writes ``<pose>_<sanitized_prompt>.mp4`` under
# ``OUTPUT_DIR``; rename to the runner-name stem so bench_summary.py
# can pick it up without grokking upstream's filename scheme.
_vendor_mp4="$(find "${VENDOR_OUT}" -maxdepth 1 -name '*.mp4' -print -quit)"
if [[ -z "${_vendor_mp4}" ]]; then
    echo "[bench] ERROR: no vendor mp4 was written under ${VENDOR_OUT}." >&2
    exit 1
fi
mv "${_vendor_mp4}" "${VENDOR_OUT}/${RUNNER_NAME}.mp4"

# Synthesised stats JSON: wall-clock elapsed only, no per-chunk timing
# (upstream's ``predict`` is opaque to this harness).
cat > "${VENDOR_OUT}/stats_${RUNNER_NAME}.json" <<EOF
{
  "runner_name": "${RUNNER_NAME}",
  "backend": "vendor",
  "num_chunk": ${NUM_CHUNK},
  "pose": "${POSE}",
  "elapsed_s": ${_vendor_elapsed_s}
}
EOF

## -------------------------------------------------------------- native leg
echo "[bench] running NATIVE leg -> ${NATIVE_OUT} (HY_VENDOR_NOISE_MODE=${HY_VENDOR_NOISE_MODE}, DIT_PROFILE=${HY_DIT_PROFILE})"
( cd "${SCRIPT_DIR}" && \
    HY_VENDOR_NOISE_MODE="${HY_VENDOR_NOISE_MODE}" \
    HY_DIT_PROFILE="${HY_DIT_PROFILE}" \
    uv run flashdreams-run "${RUNNER_NAME}" \
    --image-path "${IMAGE_PATH}" \
    --ckpt-path "${CKPT_PATH}" \
    --prompt "${PROMPT}" \
    --num-chunk "${NUM_CHUNK}" \
    --pose "${POSE}" \
    --seed "${SEED}" \
    --output-dir "${NATIVE_OUT}" )

## ---------------------------------------------------------------- summary
echo "[bench] summarising -> ${OUTPUT_DIR}/bench.md"
( cd "${SCRIPT_DIR}" && uv run python "${SCRIPT_DIR}/bench_summary.py" \
    --native-dir "${NATIVE_OUT}" \
    --vendor-dir "${VENDOR_OUT}" \
    --image-path "${IMAGE_PATH}" \
    --pose "${POSE}" \
    --num-chunk "${NUM_CHUNK}" \
    --seed "${SEED}" \
    --warmup-chunks "${WARMUP_CHUNKS}" \
    --output "${OUTPUT_DIR}/bench.md" )

echo "[bench] done."
echo "        native mp4 : ${NATIVE_OUT}/${RUNNER_NAME}.mp4"
echo "        vendor mp4 : ${VENDOR_OUT}/${RUNNER_NAME}.mp4"
echo "        summary    : ${OUTPUT_DIR}/bench.md"

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

# Pull FlashVSR, apply local patch, and run the benchmark.
# Idempotent: re-running skips clone / checkout / downloads / patch when
# already in place, and just re-runs the benchmark.

set -euo pipefail

# Resolve the directory containing this script so the script can be invoked
# from anywhere. ``../changes.patch`` in the original was implicit; here we
# anchor everything to ``SCRIPT_DIR``.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/FlashVSR"
PATCH_FILE="${SCRIPT_DIR}/changes.patch"
REPO_URL="https://github.com/OpenImagingLab/FlashVSR.git"
# Latest main HEAD as of 2026-05-13. Bump when upstream lands fixes.
PIN_COMMIT="b527c6f285fb30df530f5febc8b45764a789c961"
HF_REPO="JunhaoZhuang/FlashVSR-v1.1"
HF_LOCAL_DIR="examples/WanVSR/FlashVSR-v1.1"

# Default input video for the benchmark. Override with ``INPUT_PATH=...``.
INPUT_PATH="${INPUT_PATH:-${REPO_DIR}/examples/WanVSR/inputs/example4.mp4}"

# ``data_local/docker_interactive.sh`` pins
# ``UV_PROJECT_ENVIRONMENT=/tmp/venv/flashdreams`` on the docker session so
# every workspace ``uv sync`` lands in one shared venv. The parity-check
# is the opposite shape -- an isolated venv pinned to upstream FlashVSR's
# deps -- and running ``uv sync`` from here against the shared venv
# uninstalls every workspace integration (flash-onmidreams,
# flash-lingbot, ..., flashdreams-flashvsr) that this directory's
# pyproject.toml doesn't declare. Override the inherited variable to the
# default ``${SCRIPT_DIR}/.venv`` path so:
#   - ``uv sync`` / ``uv run`` here manage ${SCRIPT_DIR}/.venv, leaving
#     the workspace venv intact
#   - ``uv pip install`` (which does NOT respect UV_PROJECT_ENVIRONMENT
#     and only auto-detects ``.venv/`` adjacent to ``cwd``) lands in the
#     same venv that ``uv sync`` just populated
# The parity tests and the benchmark all run from this same venv -- the
# pyproject's ``flashdreams-flashvsr = { path = "../.." }`` editable
# source makes the candidate side (``flashvsr.transformer``,
# ``flashvsr.decoder.network``) importable here, replacing the previous
# flow where we flipped back to the workspace venv mid-script just for
# the TC decoder parity test.
export UV_PROJECT_ENVIRONMENT="${SCRIPT_DIR}/.venv"

# ---------------------------------------------------------------- clone + pin
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

# ------------------------------------------------------------------- patching
# Apply ``changes.patch`` BEFORE the editable install: the setup.py hunk
# strips ``import pkg_resources`` (setuptools 81+ no longer ships it), so
# ``uv pip install -e ./FlashVSR`` further down only works once the patch
# has landed. Reset the worktree to the pinned commit + drop untracked
# files (e.g. a benchmark.py from a prior run) first so the apply is
# idempotent across patch revisions; the HF download dir is gitignored
# upstream and survives ``git clean -fd``.
if [[ -f "${PATCH_FILE}" ]]; then
    echo "[setup] resetting upstream tree to ${PIN_COMMIT} before applying patch"
    git reset --hard "${PIN_COMMIT}"
    git clean -fd
    echo "[setup] applying ${PATCH_FILE}"
    git apply "${PATCH_FILE}"
else
    echo "[setup] no patch file at ${PATCH_FILE}, skipping"
fi

# --------------------------------------------------------------- HF downloads
# Upstream README uses ``git lfs clone``; we use ``huggingface-cli download``
# instead so the user doesn't need a separate git-lfs install (matches the
# self_forcing parity-check flow). Idempotent: HF CLI no-ops when the local
# dir already mirrors the repo.
if [[ ! -f "${HF_LOCAL_DIR}/diffusion_pytorch_model_streaming_dmd.safetensors" ]]; then
    echo "[setup] downloading ${HF_REPO} -> ${HF_LOCAL_DIR}"
    uv run huggingface-cli download "${HF_REPO}" \
        --local-dir-use-symlinks False \
        --local-dir "${HF_LOCAL_DIR}"
else
    echo "[setup] ${HF_LOCAL_DIR} already populated, skipping HF download"
fi

# ------------------------------------------------------------------- pip deps
# Materialize the isolated venv defined by ``${SCRIPT_DIR}/pyproject.toml``.
# ``uv sync`` is no-op-fast when the venv is already in sync. Run it from
# ${SCRIPT_DIR} so uv finds *this* project's pyproject (not flashdreams').
# All subsequent ``uv run`` calls (from inside ${REPO_DIR}) walk up and
# resolve to the same ``${SCRIPT_DIR}/.venv``.
echo "[setup] ensuring Python deps via uv sync (isolated venv)"
( cd "${SCRIPT_DIR}" && uv sync )

# Layer the upstream FlashVSR repo into the isolated venv as an editable
# install: this is what registers the ``diffsynth`` package the upstream
# scripts (and our patched ``benchmark.py``) import. Idempotent: ``uv pip
# install -e`` is a no-op when the package is already wired up. The
# setup.py hunk applied above is what makes this work on setuptools
# 81+; without it the ``import pkg_resources`` line errors out at build
# time.
#
# ``--no-deps`` is load-bearing: upstream's ``setup.py`` reads
# ``requirements.txt`` straight into ``install_requires`` (see the
# applied patch -- we replaced ``pkg_resources.parse_requirements``
# with a plain line-strip but kept the file-driven dep list, matching
# upstream behavior), and that requirements.txt pins
# ``torch==2.6.0+cu124`` plus other legacy versions that conflict with
# the parity-check venv's ``torch==2.12.0`` (pulled transitively by
# ``flashdreams``). Honoring those deps would either fail to resolve
# the ``+cu124`` index (uv: "Because there is no version of
# torch==2.6.0+cu124 and diffsynth==1.1.7 depends on
# torch==2.6.0+cu124...") or downgrade torch and break every
# CUDA-typed extension we just installed (``block-sparse-attn``,
# transformer-engine, etc.). Mirrors the ``uv pip install --no-deps
# ./flashdreams`` pattern in ``.github/workflows/doc.yml``.
# Everything the parity test + benchmark actually need lives in this
# directory's ``pyproject.toml`` (or comes in transitively via
# ``flashdreams``); add missing packages there as they surface rather
# than relaxing ``--no-deps`` and re-introducing upstream's pins.
echo "[setup] uv pip install --no-deps -e ${REPO_DIR} (registers diffsynth in venv)"
( cd "${SCRIPT_DIR}" && uv pip install --no-deps -e "${REPO_DIR}" )

# ------------------------------------------------------------------ block-sparse-attn
# Upstream's Locality-Constrained Sparse Attention (LCSA) requires the
# Block-Sparse-Attention CUDA extension. ``uv sync`` above already built
# and installed it via the ``[tool.uv.extra-build-dependencies]`` seed
# in ``pyproject.toml`` (setuptools / wheel / packaging / psutil / ninja
# + a runtime-matched torch). Verify the import succeeded so a silent
# build failure (e.g. missing ``nvcc`` on ``PATH``) surfaces here
# instead of as an opaque ImportError deep inside ``benchmark.py``.
#
# Smoke-import ``torch`` BEFORE ``block_sparse_attn``: the compiled
# ``block_sparse_attn_cuda.so`` has NEEDED entries for ``libc10.so`` /
# ``libtorch_cpu.so`` / ``libtorch_cuda.so`` but no RPATH/RUNPATH, and
# upstream's ``block_sparse_attn_interface.py`` does
# ``import block_sparse_attn_cuda`` *before* ``import torch`` so the
# "torch loads its own libs first" trick the wheel relies on doesn't
# kick in for a bare ``import block_sparse_attn``. Importing torch in
# this preamble matches every real caller (``benchmark.py`` ->
# ``diffsynth.__init__`` pulls torch first via the WanModel module, and
# both ``test_tcdecoder_parity.py`` / ``test_dit_parity.py`` import torch
# at the top of the test file), so a green import here actually mirrors
# what the downstream paths will see. Without the torch preload the
# check yields a misleading "libc10.so: cannot open shared object file"
# even though the build succeeded.
if ! ( cd "${SCRIPT_DIR}" && \
    uv run python -c "import torch; import block_sparse_attn" \
    >/dev/null 2>&1 ); then
    cat <<EOF >&2
[setup] ERROR: ``block_sparse_attn`` is not importable after ``uv sync``.
        Common causes: ``nvcc`` not on PATH, CUDA toolkit version mismatch,
        or the build was cancelled. Re-run with verbose output to inspect
        the build log:

            ( cd "${SCRIPT_DIR}" && uv sync -v )

        For the actual error, re-run the smoke check directly:

            ( cd "${SCRIPT_DIR}" && uv run python -c \\
                "import torch; import block_sparse_attn" )
EOF
    exit 1
fi

# ----------------------------------------------------------------- benchmark
# ``benchmark.py`` lives next to the upstream inference scripts under
# ``examples/WanVSR/``; run from that directory so its relative
# ``./FlashVSR-v1.1`` checkpoint paths resolve. ``FORCE_CUDNN_ATTN=1``
# routes the (non-sparse) cross-attn path through cuDNN's SDPA backend,
# matching the default flashdreams runs and what the self_forcing
# parity-check uses.
echo "[run] starting benchmark with INPUT_PATH=${INPUT_PATH}"
cd "${REPO_DIR}/examples/WanVSR"
FORCE_CUDNN_ATTN=1 uv run python benchmark.py --input "${INPUT_PATH}"

# -------------------------------------------------------- parity tests
# Both parity tests live next to this script and run from the same
# parity-check venv:
#   - ``test_tcdecoder_parity.py``: loads the upstream
#     ``examples/WanVSR/utils/TCDecoder.py`` we just cloned and asserts
#     chunk-by-chunk parity against
#     ``flashvsr.decoder.network.FlashVSR_TAEHV``.
#   - ``test_dit_parity.py``: loads upstream's
#     ``diffsynth.models.wan_video_dit.WanModel`` plus the streaming
#     wrapper ``diffsynth.pipelines.flashvsr_tiny_long.model_fn_wan_video``
#     and asserts parity against the live ``flashvsr.transformer.FlashVSRTransformer``.
# Both candidates are made importable by the
# ``flashdreams-flashvsr = { path = "../.." }`` editable source declared
# in this directory's pyproject (``uv sync`` above wires it in). The
# legacy upstream side resolves via the ``uv pip install --no-deps -e
# ./FlashVSR`` step earlier in this script, which registers
# ``diffsynth`` (and the patched ``wan_video_dit.py``) in the same venv.
# Point ``FLASHVSR_WEIGHTS_ROOT`` at the parity-check's HF download so
# both tests load the same checkpoints the benchmark below will load.
# ``set -e`` aborts the run on parity failure; nothing below should
# silently overwrite a broken model with a green benchmark.
echo "[run] running parity tests from ${SCRIPT_DIR}"
(
    cd "${SCRIPT_DIR}"
    FLASHVSR_WEIGHTS_ROOT="${REPO_DIR}/examples/WanVSR" \
        uv run pytest \
            test_tcdecoder_parity.py \
            test_dit_parity.py -v
)

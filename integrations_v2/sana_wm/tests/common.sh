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

SANA_TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SANA_TEST_REPO_ROOT="$(cd "${SANA_TEST_DIR}/../../.." && pwd)"
SANA_TEST_REPO_URL="https://github.com/NVlabs/Sana.git"
SANA_TEST_PIN_COMMIT="6298508"

sana_test_abspath() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s/%s\n' "${PWD}" "$1" ;;
    esac
}

sana_test_is_true() {
    case "${1,,}" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

sana_test_apply_patch_once() {
    local patch_file="$1"
    if git apply --reverse --check "${patch_file}" >/dev/null 2>&1; then
        echo "[setup] patch already applied, skipping ${patch_file}"
    elif git apply --check "${patch_file}" >/dev/null 2>&1; then
        echo "[setup] applying ${patch_file}"
        git apply "${patch_file}"
    else
        echo "[setup] refreshing stale patched targets for ${patch_file}"
        while IFS=$'\t' read -r _added _deleted path; do
            [[ -n "${path}" ]] || continue
            git checkout -- "${path}"
        done < <(git apply --numstat "${patch_file}")
        if git apply --check "${patch_file}" >/dev/null 2>&1; then
            echo "[setup] applying ${patch_file}"
            git apply "${patch_file}"
        else
            echo "[setup] ERROR: ${patch_file} neither cleanly applies nor is already applied." >&2
            exit 1
        fi
    fi
}

sana_test_prepare_sana_repo() {
    local sana_repo="$1"
    shift
    if [[ ! -d "${sana_repo}/.git" ]]; then
        echo "[setup] cloning ${SANA_TEST_REPO_URL} -> ${sana_repo}"
        git clone "${SANA_TEST_REPO_URL}" "${sana_repo}"
    else
        echo "[setup] repo already present at ${sana_repo}, skipping clone"
    fi

    cd "${sana_repo}"
    local current_commit
    current_commit="$(git rev-parse --short HEAD)"
    if [[ "${current_commit}" != "${SANA_TEST_PIN_COMMIT}" ]]; then
        echo "[setup] checking out pinned commit ${SANA_TEST_PIN_COMMIT}"
        git checkout "${SANA_TEST_PIN_COMMIT}"
    else
        echo "[setup] already at pinned commit ${SANA_TEST_PIN_COMMIT}, skipping checkout"
    fi

    local patch_file
    for patch_file in "$@"; do
        sana_test_apply_patch_once "${patch_file}"
    done
}

sana_test_sync_venv() {
    echo "[setup] ensuring Python deps via uv sync (isolated venv)"
    ( cd "${SANA_TEST_DIR}" && uv sync )
}

sana_test_upstream_pythonpath() {
    local sana_repo="$1"
    local upstream_pythonpath="${SANA_TEST_DIR}/compat:${sana_repo}"
    if [[ -n "${PYTHONPATH:-}" ]]; then
        upstream_pythonpath="${upstream_pythonpath}:${PYTHONPATH}"
    fi
    printf '%s\n' "${upstream_pythonpath}"
}

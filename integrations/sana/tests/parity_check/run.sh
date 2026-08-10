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

# Dispatch to the bidirectional or streaming SANA-WM parity runner.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SANA_WM_VARIANT="${SANA_WM_VARIANT:-bidirectional}"

case "${SANA_WM_VARIANT}" in
    bidirectional)
        exec bash "${SCRIPT_DIR}/run_bidirectional.sh" "$@"
        ;;
    streaming)
        exec bash "${SCRIPT_DIR}/run_streaming.sh" "$@"
        ;;
    *)
        echo "[run] ERROR: SANA_WM_VARIANT must be bidirectional or streaming; got ${SANA_WM_VARIANT}" >&2
        exit 1
        ;;
esac

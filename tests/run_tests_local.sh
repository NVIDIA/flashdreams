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
#
# INTERNAL helper: discover flashdreams test files and invoke pytest.
#
# It assumes flashdreams and its integrations are already installed in the active
# Python environment. CWD is changed to the repo root so the discovery globs
# resolve correctly regardless of where the caller invoked us from.
#
# Usage:
#   ./tests/run_tests_local.sh [TEST_TARGET...]
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

uv run --extra dev pytest -m "not manual" "$@"

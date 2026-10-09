#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

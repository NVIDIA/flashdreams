#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# Run all checks: ruff lint, ruff format, pyright, pytest.
#
# Usage:
#   scripts/check.sh         # check-only; fails on any issue
#   scripts/check.sh --fix   # auto-fix ruff lint + format, then run full check
#
# Runs from the interactive-drive sample root regardless of cwd. Uses `uv run
# --package` so dev/ui/world-model extras resolve against the
# flashdreams-interactive-drive workspace member even if the shared venv at
# the flashdreams workspace root is active.
set -euo pipefail

# Resolve the sample root (one level above this script) and the flashdreams
# workspace root (four levels above the sample root). uv needs the
# workspace root explicitly because the enclosing
# ``integrations/omnidreams/pyproject.toml`` is not a workspace root, so
# uv's own walk-up stops there and discovers the sample as a stray
# single-project workspace (astral-sh/uv#16640).
SAMPLE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE_ROOT="$(cd "$SAMPLE_ROOT/../../../.." && pwd)"

cd "$SAMPLE_ROOT"

UV_RUN=(uv --project "$WORKSPACE_ROOT" run --package omnidreams-interactive-drive --extra dev)

if [[ "${1:-}" == "--fix" ]]; then
  "${UV_RUN[@]}" ruff check --fix .
  "${UV_RUN[@]}" ruff format .
fi

"${UV_RUN[@]}" ruff check .
"${UV_RUN[@]}" ruff format --check .
"${UV_RUN[@]}" pyright
"${UV_RUN[@]}" pytest --durations=20

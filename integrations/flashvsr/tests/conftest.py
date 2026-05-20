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

"""Pytest collection config for the FlashVSR test tree.

Drops ``parity_check/`` from default recursive collection. Those tests
import ``diffsynth`` (upstream FlashVSR's vendored package), which only
exists inside the parity-check's isolated venv at
``tests/parity_check/.venv/`` -- so running e.g.
``uv run --extra dev pytest integrations/flashvsr/tests`` from the
workspace venv would otherwise ``ModuleNotFoundError`` at collection
time. ``parity_check/run.sh`` invokes the parity tests with explicit
file paths from inside ``tests/parity_check/``; explicit args bypass
``collect_ignore_glob`` (which is only consulted during recursive
directory collection), so the documented invocation keeps working.
"""

from __future__ import annotations

import importlib.util

collect_ignore_glob = ["parity_check"]

# Skip test modules that transitively import the compiled CUDA extension
# (block_sparse_attn_cuda) when it is not available -- e.g. on CPU-only CI
# runners where BLOCK_SPARSE_ATTN_SKIP_CUDA_BUILD=TRUE.
if importlib.util.find_spec("block_sparse_attn_cuda") is None:
    collect_ignore_glob += [
        "test_flashvsr.py",
        "test_projector_cuda_graph.py",
        "test_projector_benchmark.py",
        "test_color_corrector_benchmark.py",
    ]

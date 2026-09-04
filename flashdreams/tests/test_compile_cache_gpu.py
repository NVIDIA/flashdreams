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

"""GPU check that ``compile_module`` writes a reusable FX-graph cache."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.ci_gpu


def _compile_once() -> None:
    """Compile a tiny MLP through ``compile_module`` and print elapsed seconds."""
    import torch.nn as nn

    from flashdreams.infra.compile import compile_module

    class TinyMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(128, 256),
                nn.GELU(),
                nn.Linear(256, 128),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    device = torch.device("cuda")
    module = TinyMLP().to(device).eval()
    x = torch.randn(16, 128, device=device)
    started = time.perf_counter()
    compiled = compile_module(module)
    with torch.inference_mode():
        compiled(x)
    torch.cuda.synchronize()
    print(f"elapsed_s={time.perf_counter() - started:.6f}", flush=True)


def _run_worker(cache_dir: Path) -> float:
    env = os.environ.copy()
    env["FLASHDREAMS_CACHE_DIR"] = str(cache_dir)
    env.pop("TORCHINDUCTOR_CACHE_DIR", None)
    result = subprocess.run(
        [sys.executable, str(Path(__file__))],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )
    for line in result.stdout.splitlines():
        if line.startswith("elapsed_s="):
            return float(line.split("=", 1)[1])
    raise AssertionError(
        "worker did not print elapsed_s=; "
        f"stdout={result.stdout!r} stderr={result.stderr[-2000:]!r}"
    )


def test_compile_module_reuses_inductor_cache_across_processes(
    tmp_path: Path,
) -> None:
    """A second process should reuse FX-graph artifacts written by the first."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required.")

    cache_dir = tmp_path / "flashdreams"
    inductor_dir = cache_dir / "torchinductor"
    fxgraph_dir = inductor_dir / "fxgraph"

    cold_s = _run_worker(cache_dir)
    assert fxgraph_dir.is_dir(), f"missing FX-graph cache at {fxgraph_dir}"
    cold_files = {path.relative_to(inductor_dir) for path in inductor_dir.rglob("*") if path.is_file()}
    assert cold_files, "inductor cache directory is empty"

    warm_s = _run_worker(cache_dir)
    warm_files = {path.relative_to(inductor_dir) for path in inductor_dir.rglob("*") if path.is_file()}
    assert cold_files <= warm_files, "warm compile dropped artifacts from the cold cache"
    # Wall-clock speedup is logged, not gated: a loaded runner can hide a real
    # cache hit behind CUDA/import noise and fail a 2x threshold.
    print(f"compile_module cache probe cold_s={cold_s:.3f} warm_s={warm_s:.3f}")


if __name__ == "__main__":
    _compile_once()

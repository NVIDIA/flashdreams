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

"""GPU check that ``compile_module`` reuses Inductor artifacts across processes."""

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
    """A second process should skip Inductor autotune when the cache is warm."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required.")

    cache_dir = tmp_path / "flashdreams"
    cold_s = _run_worker(cache_dir)
    warm_s = _run_worker(cache_dir)
    inductor_dir = cache_dir / "torchinductor"
    assert inductor_dir.is_dir(), f"missing inductor cache at {inductor_dir}"
    assert any(inductor_dir.rglob("*")), "inductor cache directory is empty"
    assert warm_s < cold_s * 0.5, (
        f"warm compile {warm_s:.3f}s was not faster than half of cold {cold_s:.3f}s"
    )


if __name__ == "__main__":
    _compile_once()

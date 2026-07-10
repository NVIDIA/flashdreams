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

"""Same-seed bitwise-reproducibility GPU test for the omnidreams runner.

This pins the property the team cares about for trustworthy results:

    Two runs of the distilled omnidreams runner with the *same seed*,
    under PyTorch's strict-determinism configuration, produce a
    **byte-identical** MP4.

Unlike ``test_omnidreams_strict_determinism.py`` (which compares against a
committed "golden" sha256 and so must be re-gilded on every toolchain/recipe
bump), this test compares the two runs *against each other*. It therefore
needs no golden file and stays green across toolchain changes as long as the
runner remains run-to-run reproducible -- exactly the invariant we want a
periodic GPU job to guard.

PyTorch determinism flags
-------------------------
The two knobs below are set in the runner *subprocess* before the first CUDA
context is created (setting them after ``import torch`` is too late for the
cuBLAS workspace knob). They are the combination the PyTorch reproducibility
notes recommend for bitwise-reproducible matmul reductions:

* ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` -- cuBLAS must allocate a deterministic
  workspace for matmul reductions to be reproducible.
* ``torch.use_deterministic_algorithms(True, warn_only=True)`` -- route ops to
  deterministic kernels. ``warn_only=True`` because a few inference kernels in
  the AR loop (FlashAttention SDPA, some Triton-autotuned paths) have no strict
  implementation and would otherwise raise; we accept the warning and let the
  byte-equality assertion measure the residual drift empirically.
* ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` -- stable allocator
  behavior across the two runs.

See https://pytorch.org/docs/stable/notes/randomness.html

Why it is opt-in (``ci_gpu`` + env gate)
-----------------------------------------
The test needs a real GPU, downloads the distilled checkpoint + an example
HDMap clip from HF on first run, and runs two full rollouts, so it is too heavy
for the per-PR ``ci_gpu`` job. It carries the ``ci_gpu`` tier marker (it is a
GPU test) but **skips unless ``OMNIDREAMS_REPRO_RUN`` is set**, so the per-PR
``pytest -m ci_gpu`` run collects it and skips it in milliseconds. The nightly
job in ``.github/workflows/determinism.yml`` sets ``OMNIDREAMS_REPRO_RUN=1`` to
actually execute it.

(We deliberately do not use the ``manual`` marker: the ``pytest-manual-marker``
plugin xfails every ``manual`` test at setup, so a ``manual`` test never runs in
automation -- the wrong semantics for a guard we want a periodic job to run.)

Run it locally with::

    OMNIDREAMS_REPRO_RUN=1 pytest -s \
        integrations/omnidreams/tests/test_omnidreams_same_seed_reproducibility.py

Tunable via env vars (all optional, sensible defaults):

* ``OMNIDREAMS_REPRO_RUN``          -- set to ``1`` to actually run the test
* ``OMNIDREAMS_REPRO_SEED``         -- seed to test (default 1)
* ``OMNIDREAMS_REPRO_TOTAL_BLOCKS`` -- AR chunks; bigger catches accumulation
                                        drift but costs wall time (default 4)
* ``OMNIDREAMS_REPRO_CLIP_UUID``    -- example-data clip uuid (default: runner's)
* ``OMNIDREAMS_REPRO_RUN_TIMEOUT_SECONDS`` -- per-run timeout (default 2700)
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

pytestmark = pytest.mark.ci_gpu

# Distilled single-view recipe: 2-step flow-match, CFG off, ~1 s per 4 blocks.
_RECIPE = "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae"
_DEFAULT_SEED = int(os.environ.get("OMNIDREAMS_REPRO_SEED", "1"))
_DEFAULT_TOTAL_BLOCKS = int(os.environ.get("OMNIDREAMS_REPRO_TOTAL_BLOCKS", "4"))
_DEFAULT_RUN_TIMEOUT_SECONDS = int(
    os.environ.get("OMNIDREAMS_REPRO_RUN_TIMEOUT_SECONDS", "2700")
)

# Inline bootstrap run by the subprocess *before* importing the CLI, so the
# deterministic-algorithms flag is in force before the first CUDA context.
# Mirrors internal/scripts/diversity_probe/metrics/_strict_flashdreams_run.py
# but kept inline so the test carries no dependency on internal scripts.
_STRICT_BOOTSTRAP = """
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
torch.use_deterministic_algorithms(True, warn_only=True)
from flashdreams.scripts.cli import entrypoint
entrypoint()
"""


@dataclass(frozen=True)
class _RunResult:
    mp4_path: Path
    returncode: int
    stdout: str
    stderr: str


def _run_once(*, seed: int, total_blocks: int, output_dir: Path) -> _RunResult:
    """Invoke the runner once in a strict-determinism subprocess."""
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    # Must be present before the first CUDA context; we spawn fresh, so here is
    # the right place (the bootstrap also sets the cuBLAS knob as a backstop).
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    cmd = [
        sys.executable,
        "-c",
        _STRICT_BOOTSTRAP,
        _RECIPE,
        "--example-data",
        "True",
        "--total-blocks",
        str(total_blocks),
        "--output-dir",
        str(output_dir),
        "--pipeline.diffusion-model.seed",
        str(seed),
    ]
    clip_uuid = os.environ.get("OMNIDREAMS_REPRO_CLIP_UUID")
    if clip_uuid:
        cmd += ["--example-data-uuid", clip_uuid]

    completed = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=_DEFAULT_RUN_TIMEOUT_SECONDS,
    )
    return _RunResult(
        mp4_path=output_dir / f"{_RECIPE}.mp4",
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_mp4(result: _RunResult, *, label: str) -> Path:
    """Fail with the subprocess output if the run did not write an MP4."""
    if result.returncode != 0:
        stdout_tail = "\n".join(result.stdout.splitlines()[-40:])
        pytest.fail(
            f"omnidreams runner ({label}) exited rc={result.returncode}.\n"
            f"\n--- stderr ---\n{result.stderr}\n"
            f"\n--- stdout (last 40 lines) ---\n{stdout_tail}\n"
        )
    assert result.mp4_path.exists(), (
        f"runner ({label}) returned rc=0 but wrote no MP4 at {result.mp4_path}. "
        f"dir contents: {sorted(p.name for p in result.mp4_path.parent.iterdir())}"
    )
    return result.mp4_path


def test_same_seed_is_bitwise_reproducible(tmp_path: Path) -> None:
    """Two same-seed runs of the distilled runner produce identical MP4 bytes."""
    if not os.environ.get("OMNIDREAMS_REPRO_RUN"):
        pytest.skip(
            "Opt-in GPU test. Set OMNIDREAMS_REPRO_RUN=1 to run "
            "(heavy: downloads checkpoint + 2 rollouts). The nightly "
            "determinism workflow sets it."
        )

    import torch

    if not torch.cuda.is_available():
        pytest.skip("Same-seed reproducibility test requires CUDA.")

    seed = _DEFAULT_SEED
    total_blocks = _DEFAULT_TOTAL_BLOCKS

    run_a = _run_once(
        seed=seed, total_blocks=total_blocks, output_dir=tmp_path / "run_a"
    )
    mp4_a = _require_mp4(run_a, label="run A")

    run_b = _run_once(
        seed=seed, total_blocks=total_blocks, output_dir=tmp_path / "run_b"
    )
    mp4_b = _require_mp4(run_b, label="run B")

    sha_a, sha_b = _sha256(mp4_a), _sha256(mp4_b)
    size_a, size_b = mp4_a.stat().st_size, mp4_b.stat().st_size

    assert sha_a == sha_b, (
        f"Same-seed runs of {_RECIPE!r} are NOT bitwise-identical "
        f"(seed={seed}, total_blocks={total_blocks}).\n"
        f"  run A : sha256={sha_a} size={size_a} bytes ({mp4_a})\n"
        f"  run B : sha256={sha_b} size={size_b} bytes ({mp4_b})\n"
        f"\nThis means a nondeterministic kernel slipped into the inference "
        f"path. Investigate recent changes to attention / VAE / scheduler "
        f"kernels, autotune, or CUDA-graph capture. The two MP4s above are "
        f"kept under the pytest tmp dir for diffing."
    )

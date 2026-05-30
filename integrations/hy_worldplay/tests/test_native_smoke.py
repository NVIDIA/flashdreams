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

"""GPU smoke tests for the native-pipeline path.

Skipped on environments without CUDA or without a fixture image
(``HY_WORLDPLAY_FIXTURE_IMAGE``). Run locally via::

    HY_WORLDPLAY_FIXTURE_IMAGE=/path/to/first_frame.jpg \\
      uv run pytest integrations/hy_worldplay/tests/test_native_smoke.py -v

The native pipeline pulls the base Wan 2.2 TI2V-5B weights from
``Wan-AI/Wan2.2-TI2V-5B-Diffusers`` via Hugging Face (set ``HF_TOKEN``
once). Without ``ckpt_path``, HY conditioners stay zero-init identity;
set it to HY-WorldPlay's distilled ``model.pt`` to exercise the full
distilled stack. CI hooks should set the env var before invoking pytest
with the ``ci_gpu`` marker.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.ci_gpu


def _fixture_image_or_skip() -> Path:
    """Resolve the first-frame image fixture from the env var.

    Skips the test (rather than failing) when the fixture isn't present
    so a clean dev box that hasn't wired the env var doesn't see a hard
    failure. The ``ci_gpu`` marker is the contractual signal "run only
    when the GPU fixtures are wired up".
    """
    fixture_image = os.environ.get("HY_WORLDPLAY_FIXTURE_IMAGE")
    if not fixture_image:
        pytest.skip(
            "HY_WORLDPLAY_FIXTURE_IMAGE must be set to a first-frame "
            "image path to run the native-pipeline GPU smoke test."
        )
    assert fixture_image is not None
    image = Path(fixture_image)
    if not image.exists():
        pytest.skip(f"native-pipeline GPU smoke fixture not found: {image}")
    return image


def test_native_pipeline_end_to_end_single_chunk(tmp_path: Path) -> None:
    """End-to-end: the static HY pipeline produces a valid mp4 for ``num_chunk=1`` on a single GPU.

    Asserts the native rollout reaches the persistence step without
    raising and the resulting mp4 is non-empty. With ``ckpt_path``
    unset, HY conditioners stay zero-init identity so this is parity-
    safe against the base Wan 2.2 TI2V-5B output.
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip("native-pipeline GPU smoke requires CUDA")

    image = _fixture_image_or_skip()

    from dataclasses import replace

    from hy_worldplay.config import RUNNER_HY_WORLDPLAY_WAN_I2V_5B

    cfg = replace(
        RUNNER_HY_WORLDPLAY_WAN_I2V_5B,
        image_path=image,
        num_chunk=1,
        # ``num_chunk * 4 - 1 = 3`` motions; the pose parser prepends an
        # identity pose for the input frame, so the total is 4 latents.
        pose="w-3",
        output_dir=tmp_path,
    )
    cfg.setup().run()

    out_path = tmp_path / "hy-worldplay-wan-i2v-5b.mp4"
    assert out_path.exists(), f"native-pipeline did not write {out_path}"
    assert out_path.stat().st_size > 1024, (
        f"native-pipeline mp4 is suspiciously small: {out_path.stat().st_size} bytes"
    )

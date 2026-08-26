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

"""Opt-in GPU coverage for the real LingBot-VA checkpoint.

The production-path test downloads about 23 GiB unless a resolved snapshot is
provided and writes a short MP4, metrics JSON, and actions array::

    LINGBOT_VA_REAL_MODEL_RUN=1 \
    LINGBOT_VA_INPUT_DIR=/path/to/robotwin-images \
    uv run --no-sync pytest integrations_v2/lingbot_va -m ci_gpu -s

The separate compile gate is intentionally explicit because cold Inductor
autotuning can take minutes::

    LINGBOT_VA_REAL_MODEL_COMPILE_RUN=1 \
    LINGBOT_VA_INPUT_DIR=/path/to/robotwin-images \
    uv run --no-sync pytest integrations_v2/lingbot_va -m ci_gpu -s -k compile
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from lingbot_va.constants import DEFAULT_CHECKPOINT_ROOT
from lingbot_va.engine import (
    LingbotVAEngine,
    LingbotVAEngineConfig,
    LingbotVAEngineState,
)
from lingbot_va_v2.app import LingbotVAApplication

from flashdreams.runtime_v2.application_runner import ApplicationRunner
from flashdreams.runtime_v2.metrics_output_sink import MetricsOutputSink
from flashdreams.runtime_v2.mp4_client_window import Mp4ClientWindow
from flashdreams.runtime_v2.tensor_artifact_output_sink import (
    TensorArtifactOutputSink,
)
from flashdreams.t2v_v2.testing import real_model_run_skip_reason

pytestmark = pytest.mark.ci_gpu

_CHECKPOINT_REVISION = "8c9dea8abbc5c91cc9e18bc3264b8915083bbe70"
_RUN_SKIP = real_model_run_skip_reason("LINGBOT_VA_REAL_MODEL_RUN")


def _compile_skip_reason() -> str | None:
    """Return why the expensive real compile test cannot run here."""
    if not os.environ.get("LINGBOT_VA_REAL_MODEL_COMPILE_RUN"):
        return "set LINGBOT_VA_REAL_MODEL_COMPILE_RUN=1 to test torch.compile"
    if not torch.cuda.is_available():
        return "the model needs a GPU"
    return None


_COMPILE_SKIP = _compile_skip_reason()


def _checkpoint_root() -> str:
    """Return an optional local snapshot override or the official repository."""
    return os.environ.get("LINGBOT_VA_CHECKPOINT_ROOT", DEFAULT_CHECKPOINT_ROOT)


def _input_dir() -> Path:
    """Return the explicitly supplied three-camera input directory."""
    value = os.environ.get("LINGBOT_VA_INPUT_DIR")
    if value is None:
        raise RuntimeError(
            "Set LINGBOT_VA_INPUT_DIR to a directory containing the three "
            "Robotwin camera PNGs."
        )
    return Path(value)


@pytest.mark.skipif(_RUN_SKIP is not None, reason=_RUN_SKIP or "")
def test_real_model_v2_offload_writes_video_actions_and_metrics(
    tmp_path: Path,
) -> None:
    application = LingbotVAApplication()
    video_path = tmp_path / "clip.mp4"
    metrics_path = tmp_path / "metrics.json"
    ApplicationRunner(
        application,
        Mp4ClientWindow(video_path),
        metrics_output_sink=MetricsOutputSink(metrics_path),
        model_output_sinks=[TensorArtifactOutputSink(tmp_path)],
    ).run(
        application.session_desc(),
        [
            "--checkpoint-root",
            _checkpoint_root(),
            "--checkpoint-revision",
            os.environ.get(
                "LINGBOT_VA_CHECKPOINT_REVISION",
                _CHECKPOINT_REVISION,
            ),
            "--input-image-dir",
            str(_input_dir()),
            "--num-chunks",
            "2",
            "--enable-offload",
            "--no-compile",
        ],
    )

    actions = np.load(tmp_path / "actions.npy", allow_pickle=False)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    samples = {sample["name"]: sample for sample in metrics["samples"]}

    assert video_path.stat().st_size > 0
    assert actions.shape == (64, 16)
    assert actions.dtype == np.float32
    assert np.isfinite(actions).all()
    assert not np.array_equal(actions[:32], actions[32:])
    assert metrics["steps"] == [
        {
            "step_index": 0,
            "frame_count": 13,
            "sample_count": 6,
        }
    ]
    assert {name: sample["unit"] for name, sample in samples.items()} == {
        "prompt_encode_s": "s",
        "observation_encode_s": "s",
        "denoise_s": "s",
        "decode_s": "s",
        "total_s": "s",
        "peak_allocated_bytes": "bytes",
    }
    assert samples["peak_allocated_bytes"]["value"] > 0
    print(f"\nwrote {video_path}, {tmp_path / 'actions.npy'}, {metrics_path}")


@pytest.mark.skipif(_COMPILE_SKIP is not None, reason=_COMPILE_SKIP or "")
def test_real_model_compiled_engine_returns_finite_outputs() -> None:
    engine = LingbotVAEngine(
        LingbotVAEngineConfig(
            checkpoint_root=_checkpoint_root(),
            checkpoint_revision=os.environ.get(
                "LINGBOT_VA_CHECKPOINT_REVISION",
                _CHECKPOINT_REVISION,
            ),
            input_image_dir=_input_dir(),
            num_chunks=1,
            device="cuda:0",
            compile_network=True,
            video_inference_steps=1,
            action_inference_steps=1,
        )
    )
    try:
        output = engine.run()
        assert output.video.shape == (5, 3, 256, 320)
        assert output.actions.shape == (32, 16)
        assert torch.isfinite(output.video).all()
        assert torch.isfinite(output.actions).all()
        assert output.metrics["peak_allocated_bytes"] > 0
    finally:
        engine.close()

    assert engine.state is LingbotVAEngineState.CLOSED

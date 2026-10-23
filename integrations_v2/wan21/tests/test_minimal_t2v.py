# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU check for the separately packaged Minimal T2V adapter."""

from importlib.metadata import EntryPoint
from pathlib import Path
from unittest.mock import Mock

import pytest
import tomli
import torch
from minimal_t2v import MinimalT2VApplication
from wan21.config import PIPELINE_WAN21_T2V_1PT3B_480P

from flashdreams.runtime_v2.user_input_events import UserInputEvents

pytestmark = pytest.mark.ci_cpu


def test_registered_minimal_t2v_adapter() -> None:
    metadata = tomli.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    assert "wan21.apps.minimal_t2v" in metadata["tool"]["setuptools"]["packages"]
    target = metadata["project"]["entry-points"]["flashdreams.applications_v2"][
        "minimal-t2v-wan21"
    ]
    assert target == "wan21.apps.minimal_t2v.adapter:create_app"
    app = EntryPoint(
        name="minimal-t2v-wan21", value=target, group="flashdreams.applications_v2"
    ).load()()
    assert isinstance(app, MinimalT2VApplication)
    assert app.pipeline is None
    assert app.pipeline_config is PIPELINE_WAN21_T2V_1PT3B_480P
    assert app.total_blocks == 1
    desc = app.session_desc()
    assert (desc.video_width, desc.video_height) == (832, 480)
    assert desc.frames_per_second_for_step == 16

    pipeline = Mock()
    pipeline.to.return_value = pipeline
    pipeline.eval.return_value = pipeline
    pipeline.decoder.spatial_compression_ratio = 8
    pipeline.generate.return_value = torch.zeros(1, 3, 480, 832)
    pipeline.finalize.return_value = {}
    app.pipeline_config = Mock()
    app.pipeline_config.setup.return_value = pipeline
    app.init(["--default-prompt", "A cat surfing", "--device", "cpu"])
    session = app.create_session(desc)
    session.init()
    result = session.model_loop.step(0, UserInputEvents([]))[0]
    pipeline.initialize_cache.assert_called_once_with(
        text=["A cat surfing"], height=60, width=104
    )
    assert result.read_output().shape == (1, 3, 480, 832)
    assert session.model_loop.step(1, UserInputEvents([])) == []
    session.model_loop.close()
    app.close()

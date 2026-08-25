# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU ownership checks for the model-owned TI2V adapter."""

from pathlib import Path

import pytest
import tomli as tomllib
from wan22.config import WAN22_TI2V_5B_DIT_DIFFUSERS_PATH

pytestmark = pytest.mark.ci_cpu


def test_transformer_uses_the_published_sharded_checkpoint_index() -> None:
    assert WAN22_TI2V_5B_DIT_DIFFUSERS_PATH.endswith(
        "transformer/diffusion_pytorch_model.safetensors.index.json"
    )


def test_model_package_registers_only_the_nested_v2_adapter() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    manifest = tomllib.loads((project_dir / "pyproject.toml").read_text())
    assert "flashdreams-t2v" in manifest["project"]["dependencies"]
    assert "flashdreams.applications" not in manifest["project"].get("entry-points", {})
    assert manifest["project"]["entry-points"]["flashdreams.applications_v2"] == {
        "ti2v-wan22": "wan22.apps.ti2v.adapter:create_app"
    }
    assert (project_dir / "config.py").is_file()
    assert (project_dir / "ti2v" / "adapter.py").is_file()

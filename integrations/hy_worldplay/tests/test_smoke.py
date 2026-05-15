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

"""Cheap import-time checks for the ``hy_worldplay`` plugin.

These tests deliberately avoid touching the upstream HY-WorldPlay tree
or any GPU code; they only exercise the dataclass surface and the CLI
wiring so that ``uv run pytest integrations/hy_worldplay/tests`` is
fast and CPU-only.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hy_worldplay.config import RUNNER_CONFIGS, RUNNER_HY_WORLDPLAY_WAN_I2V_5B
from hy_worldplay.runner import (
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_PROMPT,
    HyWorldPlayWanI2VRunnerConfig,
)


def test_runners_dict_is_non_empty() -> None:
    """Plugin must expose at least one runner."""
    assert RUNNER_CONFIGS, "RUNNER_CONFIGS is empty"


def test_runner_keyed_by_runner_name() -> None:
    """Dict key must mirror ``cfg.runner_name`` (matches the
    self_forcing / wan21 conventions)."""
    drifted = {
        slug: cfg.runner_name
        for slug, cfg in RUNNER_CONFIGS.items()
        if slug != cfg.runner_name
    }
    assert not drifted, f"slug != runner_name: {drifted}"


def test_runners_have_descriptions() -> None:
    """Every shipped runner needs a non-empty CLI description."""
    empty = [
        slug for slug, cfg in RUNNER_CONFIGS.items() if not cfg.description.strip()
    ]
    assert not empty, f"runners missing description: {empty}"


def test_default_prompts_are_nonempty() -> None:
    """Sanity: default prompts shouldn't drift to empty strings."""
    assert DEFAULT_PROMPT.strip(), "DEFAULT_PROMPT is empty"
    assert DEFAULT_NEGATIVE_PROMPT.strip(), "DEFAULT_NEGATIVE_PROMPT is empty"


def test_default_pose_string_well_formed() -> None:
    """Pose string ``num_chunk * 4`` invariant from upstream's
    ``WanRunner.predict`` -> ``pose_to_input`` assertion."""
    cfg = RUNNER_HY_WORLDPLAY_WAN_I2V_5B
    # ``"w-16"`` -> 16 latents; default num_chunk=4 -> 4*4=16 latents.
    parts = cfg.pose.split("-")
    assert len(parts) == 2, f"unexpected default pose: {cfg.pose!r}"
    assert int(parts[1]) == cfg.num_chunk * 4, (
        f"default pose '{cfg.pose}' ({parts[1]} latents) does not match "
        f"num_chunk={cfg.num_chunk} * 4 = {cfg.num_chunk * 4} latents"
    )


def test_setup_without_required_paths_raises() -> None:
    """Constructing the runner without the three required paths should
    fail loudly rather than try to import upstream and segfault."""
    cfg = HyWorldPlayWanI2VRunnerConfig()
    assert cfg.ar_model_path is None
    assert cfg.ckpt_path is None
    assert cfg.hy_worldplay_repo_root is None
    with pytest.raises(ValueError, match="ar-model-path"):
        cfg.setup()


def test_missing_repo_root_raises_filenotfound() -> None:
    """Pointing at a non-existent repo root should give a clear error
    rather than a cryptic ``ImportError``."""
    cfg = HyWorldPlayWanI2VRunnerConfig(
        ar_model_path=Path("/nonexistent/wan_transformer"),
        ckpt_path=Path("/nonexistent/model.pt"),
        hy_worldplay_repo_root=Path("/nonexistent/HY-WorldPlay"),
    )
    with pytest.raises(FileNotFoundError, match="HY-WorldPlay tree not found"):
        cfg.setup()


def test_cli_module_imports() -> None:
    """The CLI module must be importable without side effects (no
    upstream / GPU imports at import time)."""
    import hy_worldplay.cli  # noqa: F401

    assert callable(hy_worldplay.cli.entrypoint)
    assert callable(hy_worldplay.cli.main)

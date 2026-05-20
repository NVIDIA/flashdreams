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
or any GPU code; they only exercise the dataclass surface and the
``flashdreams-run`` registration wiring so that
``uv run pytest integrations/hy_worldplay/tests`` is fast and CPU-only.
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

from flashdreams.infra.runner import RunnerConfig

pytestmark = pytest.mark.ci_cpu


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


# Reference strings copied verbatim from upstream
# ``HY-WorldPlay/wan/generate.py`` (``--input`` / ``--negative_prompt``
# argparse defaults). Pinned here so the parity-check delta against
# upstream cannot regress without a test failure first; bump in
# lockstep when upstream rotates its example prompt.
#
# This test exists because phase-1 development hit a 36%-of-total
# parity drift (mean |Δ| 5.35 -> 3.41 on uint8 RGB) that turned out to
# be a single trailing ``.`` on ``DEFAULT_PROMPT`` -- UMT5 tokenises
# the period as an extra token, which shifts the text embedding and
# perturbs every diffusion step. Cheap test, expensive bug to find.
_UPSTREAM_INPUT_DEFAULT = (
    "First-person view walking around ancient Athens, "
    "with Greek architecture and marble structures"
)
_UPSTREAM_NEGATIVE_PROMPT_DEFAULT = (
    "色调艳丽,过曝,静态,细节模糊不清,字幕,风格,作品,画作,画面,静止,整体发灰,"
    "最差质量,低质量,JPEG压缩残留,丑陋的,残缺的,多余的手指,画得不好的手部,"
    "画得不好的脸部,畸形的,毁容的,形态畸形的肢体,手指融合,静止不动的画面,"
    "杂乱的背景,三条腿,背景人很多,倒着走"
)


def test_default_prompt_byte_matches_upstream() -> None:
    """Parity guard: ``DEFAULT_PROMPT`` must byte-match upstream's
    ``--input`` argparse default."""
    assert DEFAULT_PROMPT == _UPSTREAM_INPUT_DEFAULT, (
        "DEFAULT_PROMPT drifted from upstream wan/generate.py --input "
        "default. UMT5 tokenises trailing punctuation, whitespace, and "
        "unicode-look-alikes as extra tokens -> any drift here directly "
        "shifts the text embedding and the parity check.\n"
        f"plugin   : {DEFAULT_PROMPT!r}\n"
        f"upstream : {_UPSTREAM_INPUT_DEFAULT!r}"
    )


def test_default_negative_prompt_byte_matches_upstream() -> None:
    """Parity guard: ``DEFAULT_NEGATIVE_PROMPT`` must byte-match
    upstream's ``--negative_prompt`` argparse default."""
    assert DEFAULT_NEGATIVE_PROMPT == _UPSTREAM_NEGATIVE_PROMPT_DEFAULT, (
        "DEFAULT_NEGATIVE_PROMPT drifted from upstream wan/generate.py "
        "--negative_prompt default. Same risk as the positive prompt: "
        "even invisible whitespace changes the tokenisation.\n"
        f"plugin   len={len(DEFAULT_NEGATIVE_PROMPT)} \n"
        f"upstream len={len(_UPSTREAM_NEGATIVE_PROMPT_DEFAULT)}"
    )


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
    cfg = HyWorldPlayWanI2VRunnerConfig(runner_name="hy-worldplay-wan-i2v-5b")
    assert cfg.ar_model_path is None
    assert cfg.ckpt_path is None
    assert cfg.hy_worldplay_repo_root is None
    with pytest.raises(ValueError, match="ar-model-path"):
        cfg.setup()


def test_missing_repo_root_raises_filenotfound() -> None:
    """Pointing at a non-existent repo root should give a clear error
    rather than a cryptic ``ImportError``."""
    cfg = HyWorldPlayWanI2VRunnerConfig(
        runner_name="hy-worldplay-wan-i2v-5b",
        ar_model_path=Path("/nonexistent/wan_transformer"),
        ckpt_path=Path("/nonexistent/model.pt"),
        hy_worldplay_repo_root=Path("/nonexistent/HY-WorldPlay"),
    )
    with pytest.raises(FileNotFoundError, match="HY-WorldPlay tree not found"):
        cfg.setup()


def test_runner_config_is_runner_config_subclass() -> None:
    """``HyWorldPlayWanI2VRunnerConfig`` must subclass
    :class:`flashdreams.infra.runner.RunnerConfig` so the
    ``flashdreams.runner_configs`` entry-point discovery layer (which
    ``isinstance``-checks against ``RunnerConfig``) accepts it."""
    assert isinstance(RUNNER_HY_WORLDPLAY_WAN_I2V_5B, RunnerConfig)


def test_pipeline_is_none() -> None:
    """Phase-1 wrapper has no flashdreams ``StreamInferencePipeline`` to
    drive (it delegates to upstream's ``WanRunner.predict()``); the
    base ``RunnerConfig.pipeline`` field must therefore stay ``None``
    so ``Runner.__init__`` (when the config is later promoted onto the
    base ``Runner`` ABC in phase 2b) skips pipeline construction."""
    assert RUNNER_HY_WORLDPLAY_WAN_I2V_5B.pipeline is None


def test_entry_point_registered() -> None:
    """The plugin's ``pyproject.toml`` must publish the runner under the
    ``flashdreams.runner_configs`` entry-point group so
    ``flashdreams-run`` discovers it. Importing the entry point exercises
    the same code path ``flashdreams.plugins.registry.discover_runners``
    uses at CLI startup -- a missing or misnamed entry would surface here
    rather than as a confusing "no such subcommand" later.

    Requires the plugin to be installed (``uv sync`` from repo root, or
    ``uv pip install -e integrations/hy_worldplay``); skipped if not
    installed so the test still works in editable checkouts that didn't
    sync yet.
    """
    import sys

    if sys.version_info < (3, 10):
        from importlib_metadata import entry_points  # type: ignore[import-not-found]
    else:
        from importlib.metadata import entry_points

    eps = {
        ep.name: ep
        for ep in entry_points(group="flashdreams.runner_configs")
        if ep.value.startswith("hy_worldplay.")
    }
    if not eps:
        pytest.skip(
            "flashdreams-hy-worldplay not installed (no "
            "flashdreams.runner_configs entry point registered). Run "
            "``uv sync`` or ``uv pip install -e integrations/hy_worldplay``."
        )

    assert "hy-worldplay-wan-i2v-5b" in eps, (
        f"expected entry-point 'hy-worldplay-wan-i2v-5b', got {list(eps)}"
    )
    loaded = eps["hy-worldplay-wan-i2v-5b"].load()
    assert isinstance(loaded, RunnerConfig)
    assert loaded.runner_name == "hy-worldplay-wan-i2v-5b"

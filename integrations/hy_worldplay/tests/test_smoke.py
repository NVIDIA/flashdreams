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

"""Cheap import-time smoke checks for the ``hy_worldplay`` plugin."""

from __future__ import annotations

from pathlib import Path

import pytest
from hy_worldplay.config import (
    PIPELINE_HY_WORLDPLAY_WAN_I2V_5B,
    RUNNER_CONFIGS,
    RUNNER_HY_WORLDPLAY_WAN_I2V_5B,
)
from hy_worldplay.runner import DEFAULT_PROMPT, HyWorldPlayWanI2VRunnerConfig

from flashdreams.infra.runner import RunnerConfig

pytestmark = pytest.mark.ci_cpu


def test_runners_dict_is_non_empty() -> None:
    """Plugin must expose at least one runner."""
    assert RUNNER_CONFIGS, "RUNNER_CONFIGS is empty"


def test_runner_keyed_by_runner_name() -> None:
    """Each ``RUNNER_CONFIGS`` key must match its ``cfg.runner_name``."""
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


def test_default_prompt_is_nonempty() -> None:
    """Default prompt must be non-empty."""
    assert DEFAULT_PROMPT.strip(), "DEFAULT_PROMPT is empty"


_UPSTREAM_INPUT_DEFAULT = (
    "First-person view walking around ancient Athens, "
    "with Greek architecture and marble structures"
)
"""Pinned verbatim from upstream ``HY-WorldPlay/wan/generate.py``
(``--input`` argparse default). UMT5 tokenises trailing punctuation /
whitespace as extra tokens, so any byte drift here shifts the text
embedding and breaks parity -- bump in lockstep when upstream rotates
its example prompt."""


def test_default_prompt_byte_matches_upstream() -> None:
    """``DEFAULT_PROMPT`` must byte-match upstream's ``--input`` argparse default."""
    assert DEFAULT_PROMPT == _UPSTREAM_INPUT_DEFAULT, (
        "DEFAULT_PROMPT drifted from upstream wan/generate.py --input "
        "default. UMT5 tokenises trailing punctuation, whitespace, and "
        "unicode-look-alikes as extra tokens -> any drift here directly "
        "shifts the text embedding and the parity check.\n"
        f"plugin   : {DEFAULT_PROMPT!r}\n"
        f"upstream : {_UPSTREAM_INPUT_DEFAULT!r}"
    )


def test_default_pose_string_well_formed() -> None:
    """Default pose must satisfy the parser's ``N + 1 == num_chunk * 4`` latent-count invariant.

    ``_pose.py`` prepends an identity pose for the input frame, so a
    ``w-N`` motion script produces ``N + 1`` latents; the rollout
    consumes ``num_chunk * 4``.
    """
    cfg = RUNNER_HY_WORLDPLAY_WAN_I2V_5B
    parts = cfg.pose.split("-")
    assert len(parts) == 2, f"unexpected default pose: {cfg.pose!r}"
    expected_motions = cfg.num_chunk * 4 - 1
    assert int(parts[1]) == expected_motions, (
        f"default pose '{cfg.pose}' has {parts[1]} motions; expected "
        f"{expected_motions} (num_chunk={cfg.num_chunk} * 4 - 1)"
    )


def test_runner_config_is_runner_config_subclass() -> None:
    """Runner config must subclass :class:`RunnerConfig` so entry-point discovery accepts it."""
    assert isinstance(RUNNER_HY_WORLDPLAY_WAN_I2V_5B, RunnerConfig)


def test_runner_target_routes_to_runner() -> None:
    """``_target`` must resolve to :class:`HyWorldPlayWanI2VRunner`."""
    from hy_worldplay.runner import HyWorldPlayWanI2VRunner

    assert RUNNER_HY_WORLDPLAY_WAN_I2V_5B._target is HyWorldPlayWanI2VRunner


def test_pipeline_name_matches_runner_name() -> None:
    """The static pipeline's ``name`` must match the runner slug."""
    assert PIPELINE_HY_WORLDPLAY_WAN_I2V_5B.name == "hy-worldplay-wan-i2v-5b"
    assert (
        RUNNER_HY_WORLDPLAY_WAN_I2V_5B.runner_name
        == PIPELINE_HY_WORLDPLAY_WAN_I2V_5B.name
    )


def test_static_pipeline_is_fully_hy_swapped() -> None:
    """Encoder / transformer / network must be the HY subclasses, with PRoPE blocks on."""
    from hy_worldplay._action import (
        HyWorldPlayWan21TransformerConfig,
        HyWorldPlayWanCtrlEncoderConfig,
        HyWorldPlayWanDiTNetworkConfig,
    )

    pipeline = PIPELINE_HY_WORLDPLAY_WAN_I2V_5B
    assert isinstance(pipeline.encoder, HyWorldPlayWanCtrlEncoderConfig)
    transformer = pipeline.diffusion_model.transformer
    assert isinstance(transformer, HyWorldPlayWan21TransformerConfig)
    assert isinstance(transformer.network, HyWorldPlayWanDiTNetworkConfig)
    assert transformer.network.use_prope_blocks is True
    # HY-WorldPlay autoregressive WAN-5B uses 4-latent chunks
    # (upstream's ``pred_latent_size=4``); not the base recipe's 21.
    assert transformer.len_t == 4
    assert transformer.window_size_t == 4
    # Distilled WAN-5B bakes CFG into the checkpoint.
    assert transformer.guidance_scale == 1.0
    # Wan 2.2 TI2V-5B knobs propagate through the swap.
    assert transformer.stamp_image_latent is True
    assert transformer.ti2v_first_frame_per_token_timestep is True
    assert transformer.network.in_dim == 48
    assert transformer.network.out_dim == 48
    assert transformer.network.dim == 3072


def test_static_pipeline_swaps_scheduler_to_euler_distilled() -> None:
    """The static HY pipeline uses the 4-step Euler distilled grid, not UniPC."""
    from wan22.config import PIPELINE_WAN22_TI2V_5B

    from flashdreams.infra.diffusion.scheduler import (
        FlowMatchEulerDiscreteSchedulerConfig,
        FlowMatchUniPCSchedulerConfig,
    )

    # Base recipe stays on UniPC so non-HY callers are unaffected.
    assert isinstance(
        PIPELINE_WAN22_TI2V_5B.diffusion_model.scheduler,
        FlowMatchUniPCSchedulerConfig,
    )

    sched = PIPELINE_HY_WORLDPLAY_WAN_I2V_5B.diffusion_model.scheduler
    assert isinstance(sched, FlowMatchEulerDiscreteSchedulerConfig)
    assert sched.num_inference_steps == 4
    assert sched.fixed_timesteps == (1000.0, 960.0, 888.8889, 727.2728, 0.0)


def test_static_pipeline_is_distinct_from_base() -> None:
    """The HY swap must not mutate the shared :data:`PIPELINE_WAN22_TI2V_5B` singleton."""
    from wan22.config import PIPELINE_WAN22_TI2V_5B

    assert PIPELINE_HY_WORLDPLAY_WAN_I2V_5B is not PIPELINE_WAN22_TI2V_5B
    # And the base recipe still has the stock (non-HY) encoder / transformer.
    from flashdreams.recipes.wan.autoencoder.i2v import WanI2VCtrlEncoderConfig
    from flashdreams.recipes.wan.transformer.wan21 import Wan21TransformerConfig

    assert type(PIPELINE_WAN22_TI2V_5B.encoder) is WanI2VCtrlEncoderConfig
    assert (
        type(PIPELINE_WAN22_TI2V_5B.diffusion_model.transformer)
        is Wan21TransformerConfig
    )


def test_runner_uses_base_checkpoint_without_ckpt_path() -> None:
    """No ``ckpt_path`` keeps the base 5B diffusers safetensors + remap on the static pipeline."""
    from wan22.config import (
        WAN22_TI2V_5B_DIT_DIFFUSERS_PATH,
        wan22_ti2v_5b_dit_state_dict_transform,
    )

    from flashdreams.recipes.wan.transformer.wan21 import Wan21TransformerConfig

    transformer = PIPELINE_HY_WORLDPLAY_WAN_I2V_5B.diffusion_model.transformer
    assert isinstance(transformer, Wan21TransformerConfig)
    assert transformer.checkpoint_path == WAN22_TI2V_5B_DIT_DIFFUSERS_PATH
    assert transformer.state_dict_transform is wan22_ti2v_5b_dit_state_dict_transform
    assert RUNNER_HY_WORLDPLAY_WAN_I2V_5B.ckpt_path is None


def test_runner_config_default_paths_are_unset() -> None:
    """Per-user paths must default to ``None`` so the config is portable."""
    cfg = HyWorldPlayWanI2VRunnerConfig(
        runner_name="hy-worldplay-wan-i2v-5b",
        pipeline=PIPELINE_HY_WORLDPLAY_WAN_I2V_5B,
    )
    assert cfg.image_path is None
    assert cfg.ckpt_path is None


def test_runner_config_accepts_ckpt_path() -> None:
    """Setting ``ckpt_path`` on the runner config is plain dataclass assignment, no side-effects."""
    from flashdreams.recipes.wan.transformer.wan21 import Wan21TransformerConfig

    distilled_path = Path("/some/distilled/model.pt")
    cfg = HyWorldPlayWanI2VRunnerConfig(
        runner_name="hy-worldplay-wan-i2v-5b",
        pipeline=PIPELINE_HY_WORLDPLAY_WAN_I2V_5B,
        ckpt_path=distilled_path,
    )
    assert cfg.ckpt_path == distilled_path
    # The static pipeline's checkpoint slot is *not* mutated by the
    # dataclass assignment; the runner threads ``ckpt_path`` into the
    # transformer config at construction time (see
    # ``HyWorldPlayWanI2VRunner.__init__``).
    transformer = cfg.pipeline.diffusion_model.transformer
    assert isinstance(transformer, Wan21TransformerConfig)
    from wan22.config import WAN22_TI2V_5B_DIT_DIFFUSERS_PATH

    assert transformer.checkpoint_path == WAN22_TI2V_5B_DIT_DIFFUSERS_PATH


def test_entry_point_registered() -> None:
    """Runner must be registered under the ``flashdreams.runner_configs`` entry-point group.

    Skipped when the plugin isn't installed (``uv sync`` /
    ``uv pip install -e integrations/hy_worldplay``) so editable
    checkouts that haven't synced yet still run the rest of the suite.
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

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
from hy_worldplay._vendor_pipeline import (
    VENDOR_WRAPPER_RECIPE_NAME,
    _NoopPipeline,
    _NoopPipelineConfig,
)
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


def test_default_prompts_are_nonempty() -> None:
    """Default prompts must be non-empty."""
    assert DEFAULT_PROMPT.strip(), "DEFAULT_PROMPT is empty"
    assert DEFAULT_NEGATIVE_PROMPT.strip(), "DEFAULT_NEGATIVE_PROMPT is empty"


# Pinned verbatim from upstream ``HY-WorldPlay/wan/generate.py``
# (``--input`` / ``--negative_prompt`` argparse defaults). UMT5
# tokenises trailing punctuation / whitespace as extra tokens, so any
# byte drift here shifts the text embedding and breaks parity -- bump
# in lockstep when upstream rotates its example prompt.
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
    """Assert ``DEFAULT_PROMPT`` byte-matches upstream's ``--input`` argparse default."""
    assert DEFAULT_PROMPT == _UPSTREAM_INPUT_DEFAULT, (
        "DEFAULT_PROMPT drifted from upstream wan/generate.py --input "
        "default. UMT5 tokenises trailing punctuation, whitespace, and "
        "unicode-look-alikes as extra tokens -> any drift here directly "
        "shifts the text embedding and the parity check.\n"
        f"plugin   : {DEFAULT_PROMPT!r}\n"
        f"upstream : {_UPSTREAM_INPUT_DEFAULT!r}"
    )


def test_default_negative_prompt_byte_matches_upstream() -> None:
    """Assert ``DEFAULT_NEGATIVE_PROMPT`` byte-matches upstream's ``--negative_prompt`` default."""
    assert DEFAULT_NEGATIVE_PROMPT == _UPSTREAM_NEGATIVE_PROMPT_DEFAULT, (
        "DEFAULT_NEGATIVE_PROMPT drifted from upstream wan/generate.py "
        "--negative_prompt default. Same risk as the positive prompt: "
        "even invisible whitespace changes the tokenisation.\n"
        f"plugin   len={len(DEFAULT_NEGATIVE_PROMPT)} \n"
        f"upstream len={len(_UPSTREAM_NEGATIVE_PROMPT_DEFAULT)}"
    )


def test_default_pose_string_well_formed() -> None:
    """Default pose must satisfy upstream's ``num_chunk * 4`` latent-count invariant."""
    cfg = RUNNER_HY_WORLDPLAY_WAN_I2V_5B
    # ``"w-16"`` -> 16 latents; default num_chunk=4 -> 4*4=16 latents.
    parts = cfg.pose.split("-")
    assert len(parts) == 2, f"unexpected default pose: {cfg.pose!r}"
    assert int(parts[1]) == cfg.num_chunk * 4, (
        f"default pose '{cfg.pose}' ({parts[1]} latents) does not match "
        f"num_chunk={cfg.num_chunk} * 4 = {cfg.num_chunk * 4} latents"
    )


def test_setup_without_required_paths_raises() -> None:
    """Vendor-wrapper ``setup()`` without the required paths must raise ``ValueError``."""
    cfg = HyWorldPlayWanI2VRunnerConfig(
        runner_name="hy-worldplay-wan-i2v-5b",
        use_native_pipeline=False,
    )
    assert cfg.ar_model_path is None
    assert cfg.ckpt_path is None
    assert cfg.hy_worldplay_repo_root is None
    with pytest.raises(ValueError, match="ar-model-path"):
        cfg.setup()


def test_missing_repo_root_raises_filenotfound() -> None:
    """Vendor-wrapper ``setup()`` with a missing repo root must raise ``FileNotFoundError``."""
    cfg = HyWorldPlayWanI2VRunnerConfig(
        runner_name="hy-worldplay-wan-i2v-5b",
        use_native_pipeline=False,
        ar_model_path=Path("/nonexistent/wan_transformer"),
        ckpt_path=Path("/nonexistent/model.pt"),
        hy_worldplay_repo_root=Path("/nonexistent/HY-WorldPlay"),
    )
    with pytest.raises(FileNotFoundError, match="HY-WorldPlay tree not found"):
        cfg.setup()


def test_runner_config_is_runner_config_subclass() -> None:
    """Runner config must subclass :class:`RunnerConfig` so entry-point discovery accepts it."""
    assert isinstance(RUNNER_HY_WORLDPLAY_WAN_I2V_5B, RunnerConfig)


def test_default_runner_uses_native_pipeline() -> None:
    """Default runner config routes through the native runner + real ``WanInferencePipelineConfig``."""
    from flashdreams.recipes.wan.config import WanInferencePipelineConfig
    from hy_worldplay._native_runner import HyWorldPlayWanI2VNativeRunner

    cfg = RUNNER_HY_WORLDPLAY_WAN_I2V_5B
    assert cfg.use_native_pipeline is True
    assert isinstance(cfg.pipeline, WanInferencePipelineConfig), (
        f"expected WanInferencePipelineConfig, got {type(cfg.pipeline).__name__}"
    )
    assert cfg.pipeline.recipe_name == "wan22-ti2v-5b"
    assert cfg._target is HyWorldPlayWanI2VNativeRunner


def test_vendor_wrapper_still_available_via_explicit_optout() -> None:
    """``use_native_pipeline=False`` keeps the vendor-wrapper :class:`_NoopPipelineConfig` path."""
    cfg = HyWorldPlayWanI2VRunnerConfig(
        runner_name="hy-worldplay-wan-i2v-5b",
        use_native_pipeline=False,
    )
    assert isinstance(cfg.pipeline, _NoopPipelineConfig)
    assert cfg.pipeline.recipe_name == VENDOR_WRAPPER_RECIPE_NAME
    pipeline = cfg.pipeline.setup()
    assert isinstance(pipeline, _NoopPipeline)


def test_use_native_pipeline_routes_to_wan_pipeline() -> None:
    """``use_native_pipeline`` selects the native vs vendor-wrapper pipeline + ``_target`` pair."""
    from flashdreams.recipes.wan.config import WanInferencePipelineConfig
    from hy_worldplay._native_runner import HyWorldPlayWanI2VNativeRunner
    from hy_worldplay.runner import HyWorldPlayWanI2VRunner

    # Default config: native path is the production target.
    default_cfg = HyWorldPlayWanI2VRunnerConfig(
        runner_name="hy-worldplay-wan-i2v-5b",
    )
    assert default_cfg.use_native_pipeline is True
    assert isinstance(default_cfg.pipeline, WanInferencePipelineConfig), (
        f"expected WanInferencePipelineConfig, got "
        f"{type(default_cfg.pipeline).__name__}"
    )
    assert default_cfg.pipeline.recipe_name == "wan22-ti2v-5b"
    assert default_cfg._target is HyWorldPlayWanI2VNativeRunner

    # Explicit opt-out preserves the vendor wrapper path.
    wrapper_cfg = HyWorldPlayWanI2VRunnerConfig(
        runner_name="hy-worldplay-wan-i2v-5b",
        use_native_pipeline=False,
    )
    assert isinstance(wrapper_cfg.pipeline, _NoopPipelineConfig)
    assert wrapper_cfg._target is HyWorldPlayWanI2VRunner

    # Explicit opt-in (== default) lands on the native target.
    native_cfg = HyWorldPlayWanI2VRunnerConfig(
        runner_name="hy-worldplay-wan-i2v-5b",
        use_native_pipeline=True,
    )
    assert isinstance(native_cfg.pipeline, WanInferencePipelineConfig)
    assert native_cfg.pipeline.recipe_name == "wan22-ti2v-5b"
    assert native_cfg._target is HyWorldPlayWanI2VNativeRunner


def test_use_native_pipeline_deepcopies_singleton() -> None:
    """Each native-mode config owns a distinct pipeline copy so mutations cannot leak between them."""
    from flashdreams.recipes.wan import PIPELINE_WAN22_TI2V_5B

    cfg_a = HyWorldPlayWanI2VRunnerConfig(
        runner_name="hy-worldplay-wan-i2v-5b",
        use_native_pipeline=True,
    )
    cfg_b = HyWorldPlayWanI2VRunnerConfig(
        runner_name="hy-worldplay-wan-i2v-5b",
        use_native_pipeline=True,
    )
    assert cfg_a.pipeline is not cfg_b.pipeline
    assert cfg_a.pipeline is not PIPELINE_WAN22_TI2V_5B


def test_use_native_pipeline_swaps_scheduler_to_euler_distilled() -> None:
    """Native mode swaps the base recipe's UniPC scheduler for upstream's distilled Euler schedule."""
    from flashdreams.infra.diffusion.scheduler import (
        FlowMatchEulerDiscreteSchedulerConfig,
        FlowMatchUniPCSchedulerConfig,
    )
    from flashdreams.recipes.wan import PIPELINE_WAN22_TI2V_5B

    # Base recipe stays on UniPC so non-HY callers of PIPELINE_WAN22_TI2V_5B
    # keep their existing scheduler.
    assert isinstance(
        PIPELINE_WAN22_TI2V_5B.diffusion_model.scheduler,
        FlowMatchUniPCSchedulerConfig,
    )

    cfg = HyWorldPlayWanI2VRunnerConfig(
        runner_name="hy-worldplay-wan-i2v-5b",
        use_native_pipeline=True,
    )
    sched = cfg.pipeline.diffusion_model.scheduler
    assert isinstance(sched, FlowMatchEulerDiscreteSchedulerConfig), (
        f"expected FlowMatchEulerDiscreteSchedulerConfig, got "
        f"{type(sched).__name__}"
    )
    assert sched.num_inference_steps == 4
    assert sched.fixed_timesteps == (1000.0, 960.0, 888.8889, 727.2728, 0.0)


def test_use_native_pipeline_respects_user_override() -> None:
    """A user-supplied ``pipeline=`` override must not be clobbered by the ``use_native_pipeline`` swap."""
    from flashdreams.recipes.wan import PIPELINE_WAN22_TI2V_5B
    from flashdreams.infra.config import derive_config

    custom = derive_config(PIPELINE_WAN22_TI2V_5B, recipe_name="custom-recipe")
    cfg = HyWorldPlayWanI2VRunnerConfig(
        runner_name="hy-worldplay-wan-i2v-5b",
        use_native_pipeline=True,
        pipeline=custom,
    )
    assert cfg.pipeline is custom
    assert cfg.pipeline.recipe_name == "custom-recipe"


def test_use_action_conditioning_off_by_default() -> None:
    """``use_action_conditioning`` defaults off; the encoder / transformer pair stays stock."""
    from flashdreams.recipes.wan.autoencoder.i2v import WanI2VCtrlEncoderConfig
    from flashdreams.recipes.wan.transformer.wan21 import Wan21TransformerConfig

    cfg = HyWorldPlayWanI2VRunnerConfig(
        runner_name="hy-worldplay-wan-i2v-5b",
        use_native_pipeline=True,
    )
    assert type(cfg.pipeline.encoder) is WanI2VCtrlEncoderConfig
    assert type(cfg.pipeline.diffusion_model.transformer) is Wan21TransformerConfig


def test_use_action_conditioning_swaps_encoder_and_transformer() -> None:
    """``use_action_conditioning`` swaps in the HY encoder / transformer / network triple."""
    from hy_worldplay._action import (
        HyWorldPlayWan21TransformerConfig,
        HyWorldPlayWanCtrlEncoderConfig,
        HyWorldPlayWanDiTNetworkConfig,
    )

    cfg = HyWorldPlayWanI2VRunnerConfig(
        runner_name="hy-worldplay-wan-i2v-5b",
        use_native_pipeline=True,
        use_action_conditioning=True,
    )
    assert isinstance(cfg.pipeline.encoder, HyWorldPlayWanCtrlEncoderConfig)
    transformer = cfg.pipeline.diffusion_model.transformer
    assert isinstance(transformer, HyWorldPlayWan21TransformerConfig)
    assert isinstance(transformer.network, HyWorldPlayWanDiTNetworkConfig)
    # Wan 2.2 TI2V 5B knobs must propagate through the swap. ``len_t``
    # is overridden from the base recipe's 21 down to 4 to match
    # upstream's ``pred_latent_size=4`` per-AR-step chunk; without that
    # override the native and vendor paths produce different frame
    # counts and are not parity-comparable.
    assert transformer.len_t == 4
    assert transformer.window_size_t == 4
    assert transformer.stamp_image_latent is True
    assert transformer.ti2v_first_frame_per_token_timestep is True
    # Distilled WAN-5B bakes CFG into the checkpoint, so the swap pins
    # ``guidance_scale=1.0`` to skip the uncond forward in
    # ``Wan21Transformer.predict_flow``. The base (non-distilled)
    # WAN-5B recipe stays at 5.0 because it still needs explicit CFG.
    assert transformer.guidance_scale == 1.0
    assert transformer.network.in_dim == 48
    assert transformer.network.out_dim == 48
    assert transformer.network.dim == 3072


def test_use_action_conditioning_requires_native_pipeline() -> None:
    """``use_action_conditioning`` without ``use_native_pipeline`` leaves the vendor wrapper in place."""
    cfg = HyWorldPlayWanI2VRunnerConfig(
        runner_name="hy-worldplay-wan-i2v-5b",
        use_native_pipeline=False,
        use_action_conditioning=True,
    )
    assert isinstance(cfg.pipeline, _NoopPipelineConfig)


def test_use_camera_conditioning_off_by_default() -> None:
    """``use_camera_conditioning`` defaults off; PRoPE blocks stay disabled even when action is on."""
    from hy_worldplay._action import HyWorldPlayWanDiTNetworkConfig

    only_action = HyWorldPlayWanI2VRunnerConfig(
        runner_name="hy-worldplay-wan-i2v-5b",
        use_native_pipeline=True,
        use_action_conditioning=True,
    )
    network = only_action.pipeline.diffusion_model.transformer.network
    assert isinstance(network, HyWorldPlayWanDiTNetworkConfig)
    assert network.use_prope_blocks is False


def test_use_camera_conditioning_flips_prope_blocks_flag() -> None:
    """``use_camera_conditioning=True`` triggers the HY subclass swap and flips ``use_prope_blocks``."""
    from hy_worldplay._action import (
        HyWorldPlayWan21TransformerConfig,
        HyWorldPlayWanCtrlEncoderConfig,
        HyWorldPlayWanDiTNetworkConfig,
    )

    cfg = HyWorldPlayWanI2VRunnerConfig(
        runner_name="hy-worldplay-wan-i2v-5b",
        use_native_pipeline=True,
        use_camera_conditioning=True,
    )
    assert isinstance(cfg.pipeline.encoder, HyWorldPlayWanCtrlEncoderConfig)
    transformer = cfg.pipeline.diffusion_model.transformer
    assert isinstance(transformer, HyWorldPlayWan21TransformerConfig)
    assert isinstance(transformer.network, HyWorldPlayWanDiTNetworkConfig)
    assert transformer.network.use_prope_blocks is True


def test_use_camera_conditioning_composes_with_action() -> None:
    """Action + camera flags together yield a single combined config, not nested swaps."""
    from hy_worldplay._action import (
        HyWorldPlayWan21TransformerConfig,
        HyWorldPlayWanCtrlEncoderConfig,
        HyWorldPlayWanDiTNetworkConfig,
    )

    cfg = HyWorldPlayWanI2VRunnerConfig(
        runner_name="hy-worldplay-wan-i2v-5b",
        use_native_pipeline=True,
        use_action_conditioning=True,
        use_camera_conditioning=True,
    )
    assert isinstance(cfg.pipeline.encoder, HyWorldPlayWanCtrlEncoderConfig)
    transformer = cfg.pipeline.diffusion_model.transformer
    assert isinstance(transformer, HyWorldPlayWan21TransformerConfig)
    network = transformer.network
    assert isinstance(network, HyWorldPlayWanDiTNetworkConfig)
    assert network.use_prope_blocks is True


def test_use_camera_conditioning_requires_native_pipeline() -> None:
    """``use_camera_conditioning`` without ``use_native_pipeline`` is a no-op."""
    cfg = HyWorldPlayWanI2VRunnerConfig(
        runner_name="hy-worldplay-wan-i2v-5b",
        use_native_pipeline=False,
        use_camera_conditioning=True,
    )
    assert isinstance(cfg.pipeline, _NoopPipelineConfig)


def test_use_action_conditioning_respects_user_encoder_override() -> None:
    """User-supplied encoder overrides must not be clobbered by the action swap."""
    from flashdreams.recipes.wan import PIPELINE_WAN22_TI2V_5B
    from flashdreams.recipes.wan.autoencoder.i2v import WanI2VCtrlEncoderConfig
    from flashdreams.infra.config import derive_config

    # Subclass the encoder so the swap's
    # ``type(...) is WanI2VCtrlEncoderConfig`` guard rejects it.
    class _CustomEncoderConfig(WanI2VCtrlEncoderConfig):
        pass

    custom = derive_config(PIPELINE_WAN22_TI2V_5B, recipe_name="custom-recipe")
    custom.encoder = _CustomEncoderConfig(encoder=custom.encoder.encoder)

    cfg = HyWorldPlayWanI2VRunnerConfig(
        runner_name="hy-worldplay-wan-i2v-5b",
        use_native_pipeline=True,
        use_action_conditioning=True,
        pipeline=custom,
    )
    assert isinstance(cfg.pipeline.encoder, _CustomEncoderConfig)


def test_distilled_checkpoint_routing_off_by_default() -> None:
    """Without ``ckpt_path``, the transformer keeps the base 5B safetensors checkpoint + remap."""
    from flashdreams.recipes.wan.config import (
        WAN22_TI2V_5B_DIT_DIFFUSERS_PATH,
        wan22_ti2v_5b_dit_state_dict_transform,
    )

    cfg = HyWorldPlayWanI2VRunnerConfig(
        runner_name="hy-worldplay-wan-i2v-5b",
        use_native_pipeline=True,
        use_action_conditioning=True,
    )
    transformer = cfg.pipeline.diffusion_model.transformer
    assert transformer.checkpoint_path == WAN22_TI2V_5B_DIT_DIFFUSERS_PATH
    assert (
        transformer.state_dict_transform is wan22_ti2v_5b_dit_state_dict_transform
    )


def test_distilled_checkpoint_routing_swaps_when_ckpt_path_set() -> None:
    """Setting ``ckpt_path`` re-routes the transformer to the distilled ``.pt`` + HY remap."""
    from hy_worldplay._checkpoint import hy_worldplay_distilled_state_dict_transform

    distilled_path = Path("/some/distilled/model.pt")
    cfg = HyWorldPlayWanI2VRunnerConfig(
        runner_name="hy-worldplay-wan-i2v-5b",
        use_native_pipeline=True,
        use_action_conditioning=True,
        use_camera_conditioning=True,
        ckpt_path=distilled_path,
    )
    transformer = cfg.pipeline.diffusion_model.transformer
    assert transformer.checkpoint_path == str(distilled_path)
    assert (
        transformer.state_dict_transform
        is hy_worldplay_distilled_state_dict_transform
    )


def test_distilled_checkpoint_routing_skipped_without_conditioners() -> None:
    """``ckpt_path`` without action / camera conditioning keeps the base 5B safetensors load.

    The distilled ``.pt`` is only the right source of truth once the
    action / PRoPE deltas are actually being consumed; the bit-stable
    native baseline keeps the diffusers checkpoint otherwise.
    """
    from flashdreams.recipes.wan.config import WAN22_TI2V_5B_DIT_DIFFUSERS_PATH

    cfg = HyWorldPlayWanI2VRunnerConfig(
        runner_name="hy-worldplay-wan-i2v-5b",
        use_native_pipeline=True,
        ckpt_path=Path("/some/distilled/model.pt"),
    )
    transformer = cfg.pipeline.diffusion_model.transformer
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

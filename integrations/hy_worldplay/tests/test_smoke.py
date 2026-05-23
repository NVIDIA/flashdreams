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
    """Constructing the *vendor-wrapper* runner without the three
    required paths should fail loudly rather than try to import
    upstream and segfault. Phase 2b.6.2 default-flip note: now that
    ``use_native_pipeline`` defaults to ``True`` (native runner
    path), this test must explicitly opt back into the wrapper to
    exercise its path validation.
    """
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
    """Pointing at a non-existent repo root should give a clear error
    rather than a cryptic ``ImportError``. Vendor-wrapper-specific (the
    native runner does not consume ``hy_worldplay_repo_root``); see
    ``test_setup_without_required_paths_raises`` for the default-flip
    rationale on the explicit ``use_native_pipeline=False`` opt-in.
    """
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
    """``HyWorldPlayWanI2VRunnerConfig`` must subclass
    :class:`flashdreams.infra.runner.RunnerConfig` so the
    ``flashdreams.runner_configs`` entry-point discovery layer (which
    ``isinstance``-checks against ``RunnerConfig``) accepts it."""
    assert isinstance(RUNNER_HY_WORLDPLAY_WAN_I2V_5B, RunnerConfig)


def test_default_runner_uses_native_pipeline() -> None:
    """Phase 2b.6.2 default-flip: the shipped runner config defaults to the
    native :class:`HyWorldPlayWanI2VNativeRunner` path with a real
    :class:`WanInferencePipelineConfig` from
    :data:`flashdreams.recipes.wan.PIPELINE_WAN22_TI2V_5B`.

    Previously (phases 2b.1 - 2b.6) the default was the inert
    :class:`_NoopPipelineConfig` stand-in routed through upstream's
    :class:`wan.generate.WanRunner`. The phase 2b.6.2 parity close
    moves the native path to production default; callers who need
    bit-exact match against upstream's ``use_kv_cache=False`` default
    explicitly opt back into the wrapper with
    ``use_native_pipeline=False``.
    """
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
    """``use_native_pipeline=False`` keeps the phase-1 vendor wrapper path.

    The :class:`_NoopPipelineConfig` stand-in is still constructable
    so downstream callers who need bit-exact match against upstream's
    ``use_kv_cache=False`` default can fall back to it via the
    explicit opt-out. The setup() call returns the matching no-op
    pipeline instance.
    """
    cfg = HyWorldPlayWanI2VRunnerConfig(
        runner_name="hy-worldplay-wan-i2v-5b",
        use_native_pipeline=False,
    )
    assert isinstance(cfg.pipeline, _NoopPipelineConfig)
    assert cfg.pipeline.recipe_name == VENDOR_WRAPPER_RECIPE_NAME
    pipeline = cfg.pipeline.setup()
    assert isinstance(pipeline, _NoopPipeline)


def test_use_native_pipeline_routes_to_wan_pipeline() -> None:
    """Phase 2b.1 feature flag: ``use_native_pipeline=True`` swaps the
    pipeline slot from :class:`_NoopPipelineConfig` to a fresh copy of
    :data:`flashdreams.recipes.wan.PIPELINE_WAN22_TI2V_5B` and swaps
    ``_target`` to :class:`HyWorldPlayWanI2VNativeRunner`.

    Phase 2b.6.2 closing step: the default flipped to ``True``, so the
    native :class:`HyWorldPlayWanI2VNativeRunner` is the production
    target. ``use_native_pipeline=False`` is preserved as an explicit
    opt-in for the phase-1 vendor wrapper (callers who specifically
    need bit-exact match against upstream's ``use_kv_cache=False``
    default).
    """
    from flashdreams.recipes.wan.config import WanInferencePipelineConfig
    from hy_worldplay._native_runner import HyWorldPlayWanI2VNativeRunner
    from hy_worldplay.runner import HyWorldPlayWanI2VRunner

    # Default config: native path is now the production target.
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

    # Explicit opt-out preserves the phase-1 vendor wrapper path.
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
    """Two native-mode configs must own distinct pipeline-config
    instances so per-rank seed offsets / ``derive_config`` mutations on
    one config cannot leak into another or into the module-level
    :data:`PIPELINE_WAN22_TI2V_5B` singleton."""
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
    """Phase 2b.2: native mode must swap the base recipe's UniPC
    scheduler for the upstream-distilled Euler schedule.

    Guards against the pipeline silently falling back to UniPC: at
    sub-PR 2b.1 the native runner produced visibly drifted output
    against the vendor wrapper because the schedulers diverged
    (UniPC 40-step vs Euler 4-step). 2b.2 closes that gap; this test
    pins both the scheduler class and the exact 5-entry timestep
    schedule lifted from upstream
    ``wan/inference/pipeline_wan_w_mem_relative_rope.py``.
    """
    from flashdreams.infra.diffusion.scheduler import (
        FlowMatchEulerDiscreteSchedulerConfig,
        FlowMatchUniPCSchedulerConfig,
    )
    from flashdreams.recipes.wan import PIPELINE_WAN22_TI2V_5B

    # Base recipe stays neutral so non-HY callers of PIPELINE_WAN22_TI2V_5B
    # keep their existing UniPC scheduler.
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
    """A user-supplied ``pipeline=`` override must not be clobbered by
    the ``use_native_pipeline`` swap. Lets power users plug in custom
    pipeline configs (e.g. a derived ``PIPELINE_WAN22_TI2V_5B`` with a
    different scheduler) without round-tripping through the flag."""
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
    """Phase 2b.3 feature flag defaults off so the 2b.1/2b.2 baseline is untouched.

    Without ``use_action_conditioning=True`` the deep-copied
    ``PIPELINE_WAN22_TI2V_5B`` must keep its stock
    :class:`WanI2VCtrlEncoderConfig` / :class:`Wan21TransformerConfig`
    pair, so a runner that only opts into ``use_native_pipeline`` sees
    the same encoder + transformer it saw at 2b.2 even after 2b.3 lands.
    """
    from flashdreams.recipes.wan.autoencoder.i2v import WanI2VCtrlEncoderConfig
    from flashdreams.recipes.wan.transformer.wan21 import Wan21TransformerConfig

    cfg = HyWorldPlayWanI2VRunnerConfig(
        runner_name="hy-worldplay-wan-i2v-5b",
        use_native_pipeline=True,
    )
    assert type(cfg.pipeline.encoder) is WanI2VCtrlEncoderConfig
    assert type(cfg.pipeline.diffusion_model.transformer) is Wan21TransformerConfig


def test_use_action_conditioning_swaps_encoder_and_transformer() -> None:
    """Phase 2b.3 wiring: action flag swaps in the HY encoder / transformer / network.

    The encoder becomes :class:`HyWorldPlayWanCtrlEncoderConfig`, the
    transformer becomes :class:`HyWorldPlayWan21TransformerConfig`, and
    the nested network becomes :class:`HyWorldPlayWanDiTNetworkConfig`.
    All other transformer fields (``len_t``, ``stamp_image_latent``,
    ``ti2v_first_frame_per_token_timestep``, ...) must be propagated so
    the subclassed transformer behaves identically apart from the action
    plumbing.
    """
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
    # Critical Wan 2.2 TI2V 5B knobs must be propagated, not silently
    # reset to subclass defaults. ``len_t`` is the one exception: the
    # HY-WorldPlay swap overrides it from the base recipe's 21 down to
    # 4 to match upstream's ``pred_latent_size=4`` per-AR-step chunk
    # size (see ``_swap_in_action_conditioning_configs`` for the
    # rationale). Without this override, native vs vendor produce
    # different total frame counts and are not parity-comparable.
    assert transformer.len_t == 4
    assert transformer.window_size_t == 4
    assert transformer.stamp_image_latent is True
    assert transformer.ti2v_first_frame_per_token_timestep is True
    # HY-WorldPlay distilled WAN-5B bakes CFG into the checkpoint; the
    # swap pins ``guidance_scale=1.0`` so ``Wan21Transformer.predict_flow``
    # skips the uncond forward + ``flow_uncond + s * (flow_cond -
    # flow_uncond)`` combine (which upstream
    # ``pipeline_wan_w_mem_relative_rope.py`` also skips on its
    # few-step distilled path). The base WAN-5B recipe stays at ``5.0``
    # because the non-distilled model needs explicit CFG.
    assert transformer.guidance_scale == 1.0
    assert transformer.network.in_dim == 48
    assert transformer.network.out_dim == 48
    assert transformer.network.dim == 3072


def test_use_action_conditioning_requires_native_pipeline() -> None:
    """``use_action_conditioning`` without ``use_native_pipeline`` is a no-op.

    The vendor-wrapper path drives upstream's runner end-to-end, so the
    action-aware encoder / transformer swap has nothing to attach to; we
    leave the inert :class:`_NoopPipelineConfig` in place and let the
    vendor wrapper carry through its own action conditioning. Phase
    2b.6.2 default-flip note: the test must opt out of
    ``use_native_pipeline`` explicitly (the default is now ``True``).
    """
    cfg = HyWorldPlayWanI2VRunnerConfig(
        runner_name="hy-worldplay-wan-i2v-5b",
        use_native_pipeline=False,
        use_action_conditioning=True,
    )
    assert isinstance(cfg.pipeline, _NoopPipelineConfig)


def test_use_camera_conditioning_off_by_default() -> None:
    """Phase 2b.4 feature flag defaults off; PRoPE blocks stay disabled.

    With only ``use_native_pipeline`` (and even with
    ``use_action_conditioning``) set, the deep-copied network config must
    keep ``use_prope_blocks=False`` so the DiT keeps building stock
    :class:`Block` instances. Guards against accidentally promoting PRoPE
    blocks (which need bound camera data) when callers only want action
    conditioning.
    """
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
    """Phase 2b.4 wiring: camera flag enables the dual-branch block path.

    ``use_camera_conditioning=True`` must (a) trigger the encoder /
    transformer / network swap (the camera tensors ride on the same
    :class:`HyWorldPlayCtrl` payload as the action labels, so there's
    one subclass tree, not two) and (b) flip ``use_prope_blocks=True``
    on the resulting :class:`HyWorldPlayWanDiTNetworkConfig` so
    :meth:`HyWorldPlayWanDiTNetwork._build_block` emits
    :class:`HyWorldPlayPRoPEBlock` instances at network construction.
    """
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
    """Action + camera flags together must yield a single combined config.

    The camera swap reuses the action subclass tree, so flipping both
    flags should not produce duplicate / nested swaps -- exactly one
    :class:`HyWorldPlayWanCtrlEncoderConfig` instance with
    ``use_prope_blocks=True`` on its network sibling.
    """
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
    """``use_camera_conditioning`` without ``use_native_pipeline`` is a no-op.

    Phase 2b.6.2 default-flip note: the test must opt out of
    ``use_native_pipeline`` explicitly (default is now ``True``).
    """
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

    # Build a custom pipeline with a custom (subclassed) encoder so the
    # swap's ``type(...) is WanI2VCtrlEncoderConfig`` guard rejects it.
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
    """Phase 2b.5b routing: without ``ckpt_path`` the transformer keeps
    the base 5B diffusers safetensors checkpoint + remap.

    Used by the swap-config smoke tests above (which intentionally
    don't supply a ``ckpt_path``); also covers the case where a user
    enables conditioners without yet downloading the distilled
    checkpoint. The conditioners stay as zero-init identities so
    output continues to match the base recipe.
    """
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
    """Phase 2b.5b routing: setting ``ckpt_path`` re-routes the transformer
    load to the distilled ``.pt`` and swaps the state-dict transform to
    the HY-specific remap.

    The actual remap function is exercised end-to-end in
    ``test_checkpoint.py::test_distilled_checkpoint_loads_strict``;
    here we just verify the runner's ``__post_init__`` plumbs the
    user-supplied path + transform through to the transformer config.
    """
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
    """Phase 2b.5b routing: even with ``ckpt_path`` set, the transformer
    keeps the base 5B safetensors when neither action nor camera
    conditioning is enabled.

    Conservative gate: 2b.1's bit-stable native baseline (no
    conditioners) must keep loading the same diffusers checkpoint
    it loaded at 2b.1 -- the distilled ``.pt`` is only the right
    source of truth once the action / PRoPE deltas are actually
    being consumed.
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

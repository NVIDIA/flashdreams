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

"""CPU lifecycle tests for the OmniDreams inference session."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
import torch
from flashdreams.infra.decoder import StreamingVideoDecoder
from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler.fm_euler import (
    FlowMatchEulerDiscreteSchedulerConfig,
)
from flashdreams.recipes.taehv import TeahvVAEDecoder, TeahvVAEDecoderConfig
from flashdreams.recipes.taehv.impl import TAEHVCache
from omnidreams.encoder.pixel_shuffle import (
    PixelShuffleVAEEncoderCache,
    PixelShuffleVAEEncoderConfig,
)
from omnidreams.pipeline import OmnidreamsPipeline, OmnidreamsPipelineConfig
from omnidreams.runtime.inference_session import (
    InferenceGlobalCondition,
    InferenceInput,
    InferenceSession,
    InferenceUserCondition,
)
from omnidreams.transformer import CosmosTransformerConfig
from omnidreams.transformer.impl.network import CosmosDiTNetworkConfig
from pydantic import ValidationError
from torch import Tensor

pytestmark = pytest.mark.ci_cpu


# ---------------------- Mock Pipeline  ---------------------- #

# OmnidreamsPipeline requires a Wan or TAEHV decoder. This lightweight TAEHV
# subclass preserves that concrete contract without downloading decoder weights;
# the pipeline, HDMap encoder, transformer, scheduler, and caches remain real.


@dataclass(kw_only=True)
class _CPUDecoderConfig(TeahvVAEDecoderConfig):
    """Configure the checkpoint-free decoder used by the CPU pipeline fixture."""

    _target: type[_CPUDecoder] = field(default_factory=lambda: _CPUDecoder)


class _CPUDecoder(TeahvVAEDecoder):
    """Preserve the Taehv pipeline contract without loading decoder weights."""

    def __init__(self, config: TeahvVAEDecoderConfig) -> None:
        """Initialize only the streaming decoder interface."""
        StreamingVideoDecoder.__init__(self, config)

    def initialize_autoregressive_cache(self) -> TAEHVCache:
        """Return an empty Taehv-compatible cache."""
        return TAEHVCache()

    def forward(
        self,
        input: Tensor,
        autoregressive_index: int = 0,
        cache: TAEHVCache | None = None,
    ) -> Tensor:
        """Expose three latent channels as a cheap decoded video."""
        del autoregressive_index, cache
        # Decoded pixel quality is outside this session test; retaining RGB channels
        # keeps output assertions representative without running a pretrained VAE.
        return input[..., :3, :, :]


@pytest.fixture
def pipeline() -> OmnidreamsPipeline:
    """Set up the actual OmniDreams pipeline with tiny CPU components."""
    config = OmnidreamsPipelineConfig(
        name="test-omnidreams-inference-session",
        # Conditions arrive as precomputed embeddings, so one-shot text/image
        # encoders are unnecessary. PixelShuffle remains the real per-step path.
        text_encoder=None,
        image_encoder=None,
        encoder=PixelShuffleVAEEncoderConfig(),
        decoder=_CPUDecoderConfig(),
        diffusion_model=DiffusionModelConfig(
            transformer=CosmosTransformerConfig(
                network=CosmosDiTNetworkConfig(
                    # An 8x8 RGB HDMap becomes a 192-channel 1x1 control latent.
                    # Zero blocks retain patching, conditioning, and final-layer
                    # execution while keeping the CPU fixture small.
                    in_channels=16,
                    out_channels=16,
                    patch_spatial=1,
                    patch_temporal=1,
                    model_channels=12,
                    num_blocks=0,
                    num_heads=1,
                    mlp_ratio=1.0,
                    concat_padding_mask=False,
                    use_adaln_lora=False,
                    use_crossattn_projection=False,
                    crossattn_emb_channels=4,
                    additional_concat_ch=192,
                ),
                # Keep ci_cpu on eager, random-init code paths with no downloads.
                dtype=torch.float32,
                checkpoint_path=None,
                batch_shape=(1,),
                num_views=1,
                len_t=1,
                h_extrapolation_ratio=1.0,
                w_extrapolation_ratio=1.0,
                window_size_t=1,
                sink_size_t=0,
                compile_network=False,
                use_cuda_graph=False,
                skip_finalize_kv_cache=True,
            ),
            # One Euler step is enough to exercise generation orchestration.
            scheduler=FlowMatchEulerDiscreteSchedulerConfig(
                num_inference_steps=1,
                fixed_timesteps=(1000.0, 0.0),
            ),
            seed=0,
        ),
    )

    pipeline = config.setup()
    assert type(pipeline) is OmnidreamsPipeline
    return pipeline


@pytest.fixture
def session(pipeline: OmnidreamsPipeline) -> InferenceSession:
    """Construct an inference session from the actual pipeline."""
    return InferenceSession(pipeline)


# ------------------- Condition Factories  ------------------- #


def _user_condition(
    value: float,
    *,
    num_frames: int = 1,
    height: int = 8,
    width: int = 8,
) -> InferenceUserCondition:
    return InferenceUserCondition(
        hdmap=torch.full((1, 1, num_frames, 3, height, width), value)
    )


def _global_condition(
    value: float,
    *,
    include_negative: bool = False,
    latent_height: int = 1,
    latent_width: int = 1,
) -> InferenceGlobalCondition:
    condition = InferenceGlobalCondition(
        text_embeddings=torch.full((1, 1, 2, 4), value),
        image_embeddings=torch.full(
            (1, 1, 1, 16, latent_height, latent_width), value + 1
        ),
    )
    if include_negative:
        condition["negative_text_embeddings"] = torch.full((1, 1, 2, 4), value + 2)
    return condition


# ------------------- Session Conditioning ------------------- #


def test_step_runs_actual_pipeline_with_global_conditions(
    session: InferenceSession,
    pipeline: OmnidreamsPipeline,
) -> None:
    """Verify the first step initializes and runs the actual pipeline."""
    user_condition = _user_condition(2.0)
    global_condition = _global_condition(3.0, include_negative=True)

    output = session.step(
        InferenceInput(
            user_condition=user_condition,
            global_condition=global_condition,
        )
    )

    # Exact type equality prevents a test double from silently replacing the
    # integration pipeline while preserving isinstance compatibility.
    assert type(pipeline) is OmnidreamsPipeline
    assert session.cache is not None
    # Pipeline caches record the last generated index; the session index points
    # to the next step that will be generated.
    assert session.cache.autoregressive_index == 0
    assert isinstance(session.cache.encoder_cache, PixelShuffleVAEEncoderCache)
    assert session.cache.encoder_cache.autoregressive_index == 0
    assert session.autoregressive_index == 1
    assert output.value.shape == (1, 1, 1, 3, 1, 1)
    assert torch.isfinite(output.value).all()
    assert output.start_timestamp == pytest.approx(0.0)
    assert output.fps == pytest.approx(30.0)
    assert output.frame_present_time == pytest.approx(1.0 / 30.0)


def test_step_reuses_actual_pipeline_cache_with_different_user_conditions(
    session: InferenceSession,
) -> None:
    """Verify later steps use new HDMaps while retaining rollout state."""
    first_output = session.step(
        InferenceInput(
            user_condition=_user_condition(1.0),
            global_condition=_global_condition(2.0),
        )
    )
    cache = session.cache
    second_output = session.step(
        InferenceInput(user_condition=_user_condition(7.0, num_frames=4))
    )

    # The second user condition advances the same rollout cache rather than
    # rebuilding global text/image conditioning.
    assert session.cache is cache
    assert cache is not None
    assert cache.autoregressive_index == 1
    assert isinstance(cache.encoder_cache, PixelShuffleVAEEncoderCache)
    assert cache.encoder_cache.autoregressive_index == 1
    assert session.autoregressive_index == 2
    assert first_output.value.shape == second_output.value.shape
    assert torch.isfinite(first_output.value).all()
    assert torch.isfinite(second_output.value).all()
    assert first_output.start_timestamp == pytest.approx(0.0)
    assert second_output.start_timestamp == pytest.approx(
        first_output.value.shape[2] * first_output.frame_present_time
    )
    assert second_output.fps == first_output.fps


def test_step_uses_different_global_conditions_after_reset(
    session: InferenceSession,
) -> None:
    """Verify reset creates an actual pipeline cache from new embeddings."""
    first_output = session.step(
        InferenceInput(
            user_condition=_user_condition(2.0),
            global_condition=_global_condition(1.0),
        )
    )
    first_cache = session.cache
    assert first_cache is not None
    first_image = first_cache.transformer_cache.image.clone()

    # Reset releases both the cache and its fixed pixel resolution, so a new
    # rollout may use a different aligned HDMap/image-latent size.
    session.reset()
    second_output = session.step(
        InferenceInput(
            user_condition=_user_condition(8.0, height=16),
            global_condition=_global_condition(
                9.0,
                include_negative=True,
                latent_height=2,
            ),
        )
    )

    second_cache = session.cache
    assert second_cache is not None
    assert second_cache is not first_cache
    assert not torch.equal(second_cache.transformer_cache.image, first_image)
    assert not torch.equal(second_output.value, first_output.value)
    assert second_output.start_timestamp == pytest.approx(0.0)
    assert session.autoregressive_index == 1


def test_step_requires_global_conditions_for_new_rollout(
    session: InferenceSession,
) -> None:
    """Verify a new rollout rejects an HDMap without embedding conditions."""
    with pytest.raises(ValueError, match="global_condition is required"):
        session.step(InferenceInput(user_condition=_user_condition(1.0)))


def test_step_rejects_global_conditions_during_active_rollout(
    session: InferenceSession,
) -> None:
    """Verify an active rollout rejects replacement embedding conditions."""
    session.step(
        InferenceInput(
            user_condition=_user_condition(1.0),
            global_condition=_global_condition(2.0),
        )
    )
    cache = session.cache

    with pytest.raises(ValueError, match="can only be supplied on the first step"):
        session.step(
            InferenceInput(
                user_condition=_user_condition(3.0, num_frames=4),
                global_condition=_global_condition(4.0),
            )
        )

    # Rejection happens before pipeline generation and leaves both indices intact.
    assert session.cache is cache
    assert cache is not None
    assert cache.autoregressive_index == 0
    assert session.autoregressive_index == 1


# ---------------- Pydantic Schema Validation ---------------- #


@pytest.mark.parametrize(
    "missing_field",
    ["hdmap", "text_embeddings", "image_embeddings"],
)
def test_step_validates_omnidreams_condition_fields(
    session: InferenceSession,
    missing_field: str,
) -> None:
    """Verify Pydantic rejects missing required OmniDreams conditions."""
    inference_input: Any = {
        "user_condition": {"hdmap": torch.zeros(1)},
        "global_condition": {
            "text_embeddings": torch.zeros(1),
            "image_embeddings": torch.zeros(1),
        },
    }
    container = (
        inference_input["user_condition"]
        if missing_field == "hdmap"
        else inference_input["global_condition"]
    )
    del container[missing_field]

    with pytest.raises(ValidationError) as exc_info:
        session.step(inference_input)

    assert any(
        error["loc"][-1:] == (missing_field,) for error in exc_info.value.errors()
    )


@pytest.mark.parametrize(
    ("field_name", "expected_rank"),
    [
        ("hdmap", 6),
        ("text_embeddings", 4),
        ("negative_text_embeddings", 4),
        ("image_embeddings", 6),
    ],
)
def test_step_validates_omnidreams_condition_tensor_ranks(
    session: InferenceSession,
    field_name: str,
    expected_rank: int,
) -> None:
    """Verify Pydantic rejects condition tensors with the wrong rank."""
    # Begin with a fully valid input and replace one field so the reported
    # Pydantic location identifies only the dimension under test.
    user_condition = dict(_user_condition(1.0))
    global_condition = dict(_global_condition(2.0, include_negative=True))
    condition = user_condition if field_name == "hdmap" else global_condition
    condition[field_name] = torch.zeros((1,) * (expected_rank - 1))
    inference_input: Any = {
        "user_condition": user_condition,
        "global_condition": global_condition,
    }

    with pytest.raises(ValidationError) as exc_info:
        session.step(inference_input)

    matching_errors = [
        error for error in exc_info.value.errors() if error["loc"][-1:] == (field_name,)
    ]
    assert len(matching_errors) == 1
    assert f"rank-{expected_rank}" in matching_errors[0]["msg"]
    assert session.cache is None
    assert session.autoregressive_index == 0


@pytest.mark.parametrize(
    ("field_name", "invalid_tensor", "expected_message"),
    [
        (
            "hdmap",
            torch.zeros(1, 1, 1, 4, 8, 8),
            "[B, V, T, 3, H, W]",
        ),
        (
            "hdmap",
            torch.zeros(1, 1, 0, 3, 8, 8),
            "every axis",
        ),
        (
            "text_embeddings",
            torch.zeros(1, 1, 0, 4),
            "every axis",
        ),
        (
            "negative_text_embeddings",
            torch.zeros(1, 1, 2, 0),
            "every axis",
        ),
        (
            "image_embeddings",
            torch.zeros(1, 1, 2, 16, 1, 1),
            "[B, V, 1, Cl, Hl, Wl]",
        ),
        (
            "image_embeddings",
            torch.zeros(1, 1, 1, 16, 0, 1),
            "every axis",
        ),
    ],
)
def test_step_validates_omnidreams_condition_tensor_shapes(
    session: InferenceSession,
    field_name: str,
    invalid_tensor: Tensor,
    expected_message: str,
) -> None:
    """Verify Pydantic rejects fixed-axis and empty condition shapes."""
    # Keep every other field valid to isolate fixed-axis and empty-axis checks.
    user_condition = dict(_user_condition(1.0))
    global_condition = dict(_global_condition(2.0, include_negative=True))
    condition = user_condition if field_name == "hdmap" else global_condition
    condition[field_name] = invalid_tensor
    inference_input: Any = {
        "user_condition": user_condition,
        "global_condition": global_condition,
    }

    with pytest.raises(ValidationError) as exc_info:
        session.step(inference_input)

    matching_errors = [
        error for error in exc_info.value.errors() if error["loc"][-1:] == (field_name,)
    ]
    assert len(matching_errors) == 1
    assert expected_message in matching_errors[0]["msg"]
    assert session.cache is None
    assert session.autoregressive_index == 0


@pytest.mark.parametrize(
    ("field_name", "invalid_tensor", "expected_message"),
    [
        (
            "hdmap",
            torch.zeros(1, 2, 1, 3, 8, 8),
            "share [B, V] dimensions",
        ),
        (
            "text_embeddings",
            torch.zeros(2, 1, 2, 4),
            "share [B, V] dimensions",
        ),
        (
            "image_embeddings",
            torch.zeros(1, 2, 1, 16, 1, 1),
            "share [B, V] dimensions",
        ),
        (
            "negative_text_embeddings",
            torch.zeros(1, 1, 3, 4),
            "shape to match text_embeddings",
        ),
    ],
)
def test_step_validates_condition_shape_relationships(
    session: InferenceSession,
    field_name: str,
    invalid_tensor: Tensor,
    expected_message: str,
) -> None:
    """Verify Pydantic validates shapes shared by multiple conditions."""
    # These tensors are individually valid; only their shared dimensions differ.
    user_condition = dict(_user_condition(1.0))
    global_condition = dict(_global_condition(2.0, include_negative=True))
    condition = user_condition if field_name == "hdmap" else global_condition
    condition[field_name] = invalid_tensor
    inference_input: Any = {
        "user_condition": user_condition,
        "global_condition": global_condition,
    }

    with pytest.raises(ValidationError) as exc_info:
        session.step(inference_input)

    assert expected_message in str(exc_info.value)
    assert session.cache is None
    assert session.autoregressive_index == 0


# ------------ Pipeline-aware Pydantic Validation ------------ #


def test_step_validates_hdmap_resolution_alignment_with_pipeline(
    session: InferenceSession,
) -> None:
    """Verify Pydantic applies the pipeline's pixel-alignment check."""
    # Width 9 violates the fixture's 8x VAE alignment while retaining rank/layout.
    inference_input = InferenceInput(
        user_condition=_user_condition(1.0, width=9),
        global_condition=_global_condition(2.0),
    )

    with pytest.raises(ValidationError) as exc_info:
        session.step(inference_input)

    assert "must be divisible by 8" in str(exc_info.value)
    assert session.cache is None
    assert session.autoregressive_index == 0


def test_step_validates_image_embedding_resolution_against_hdmap(
    session: InferenceSession,
) -> None:
    """Verify Pydantic relates image latent and HDMap pixel resolutions."""
    # A 16x8 HDMap requires a 2x1 latent, but the default global condition is 1x1.
    inference_input = InferenceInput(
        user_condition=_user_condition(1.0, height=16),
        global_condition=_global_condition(2.0),
    )

    with pytest.raises(ValidationError) as exc_info:
        session.step(inference_input)

    assert "expected image_embeddings latent resolution (2, 1)" in str(exc_info.value)
    assert session.cache is None
    assert session.autoregressive_index == 0


def test_step_validates_first_hdmap_frame_count_with_pipeline(
    session: InferenceSession,
) -> None:
    """Verify Pydantic checks the first AR step's HDMap frame count."""
    # len_t=1 produces one pixel frame at AR 0 despite the steady-state 4x ratio.
    inference_input = InferenceInput(
        user_condition=_user_condition(1.0, num_frames=4),
        global_condition=_global_condition(2.0),
    )

    with pytest.raises(ValidationError) as exc_info:
        session.step(inference_input)

    assert "expected hdmap T=1 at autoregressive index 0; got T=4" in str(
        exc_info.value
    )
    assert session.cache is None
    assert session.autoregressive_index == 0


def test_step_validates_later_hdmap_frame_count_with_pipeline(
    session: InferenceSession,
) -> None:
    """Verify Pydantic checks later AR steps using the pipeline index."""
    session.step(
        InferenceInput(
            user_condition=_user_condition(1.0),
            global_condition=_global_condition(2.0),
        )
    )
    cache = session.cache

    # Steady-state PixelShuffle/TAEHV geometry requires four input frames.
    with pytest.raises(ValidationError) as exc_info:
        session.step(InferenceInput(user_condition=_user_condition(3.0)))

    assert "expected hdmap T=4 at autoregressive index 1; got T=1" in str(
        exc_info.value
    )
    assert session.cache is cache
    assert cache is not None
    assert cache.autoregressive_index == 0
    assert session.autoregressive_index == 1


def test_step_validates_hdmap_resolution_is_stable_during_rollout(
    session: InferenceSession,
) -> None:
    """Verify Pydantic rejects aligned resolution changes within a rollout."""
    session.step(
        InferenceInput(
            user_condition=_user_condition(1.0),
            global_condition=_global_condition(2.0),
        )
    )
    cache = session.cache

    # 16x8 is independently aligned; only changing the active rollout size is invalid.
    with pytest.raises(ValidationError) as exc_info:
        session.step(
            InferenceInput(user_condition=_user_condition(3.0, num_frames=4, height=16))
        )

    assert "expected hdmap resolution (8, 8)" in str(exc_info.value)
    assert session.cache is cache
    assert cache is not None
    assert cache.autoregressive_index == 0
    assert session.autoregressive_index == 1

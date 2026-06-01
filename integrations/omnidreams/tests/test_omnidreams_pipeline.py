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

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from omnidreams import transformer as omnidreams_transformer_module
from omnidreams.config import (
    OMNIDREAMS_CONFIGS,
    SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE,
)
from omnidreams.constants import NEGATIVE_PROMPT
from omnidreams.pipeline import (
    OmnidreamsPipeline,
    _validate_pixel_resolution_alignment,
)
from omnidreams.transformer import (
    CosmosTransformer,
    CosmosTransformerConfig,
)
from omnidreams.transformer.impl.context_parallel import (
    HierarchicalCPGroups,
)

# Mixed markers: most tests are ci_cpu; streaming_inference is manual.
# Per-function markers used below.
from flashdreams.infra.diffusion.scheduler.fm_unipc import (
    FlowMatchUniPCSchedulerConfig,
)
from flashdreams.infra.pipeline import StreamInferencePipeline


def _make_uninitialized_omnidreams_pipeline() -> OmnidreamsPipeline:
    pipeline = OmnidreamsPipeline.__new__(OmnidreamsPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.diffusion_model = SimpleNamespace(
        device=torch.device("cpu"),
        transformer=SimpleNamespace(config=CosmosTransformerConfig()),
    )
    pipeline.V_group = None
    return pipeline


@pytest.mark.ci_cpu
def test_validate_pixel_resolution_alignment_requires_vae_and_patch_multiple() -> None:
    _validate_pixel_resolution_alignment(
        640,
        1168,
        spatial_compression_ratio=8,
        patch_spatial=2,
    )
    with pytest.raises(ValueError, match="divisible by 16"):
        _validate_pixel_resolution_alignment(
            640,
            1164,
            spatial_compression_ratio=8,
            patch_spatial=2,
        )


@pytest.mark.ci_cpu
def test_omnidreams_initialize_cache_from_embeddings_negative_text_optional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_contexts: list[dict[str, Any]] = []

    def capture_initialize_cache(
        self: StreamInferencePipeline,
        *,
        transformer_context: dict[str, Any] | None = None,
        encoder_context: dict[str, Any] | None = None,
        decoder_context: dict[str, Any] | None = None,
    ) -> object:
        del self, encoder_context, decoder_context
        assert transformer_context is not None
        captured_contexts.append(transformer_context)
        return object()

    monkeypatch.setattr(
        StreamInferencePipeline,
        "initialize_cache",
        capture_initialize_cache,
    )

    pipeline = _make_uninitialized_omnidreams_pipeline()
    text_embeddings = torch.randn(1, 1, 2, 3)
    image_embeddings = torch.randn(1, 1, 1, 2, 2, 2)
    negative_text_embeddings = torch.randn(1, 1, 2, 3)

    pipeline.initialize_cache_from_embeddings(
        text_embeddings=text_embeddings,
        image_embeddings=image_embeddings,
    )
    assert "negative_text_embeddings" not in captured_contexts[-1]

    pipeline.initialize_cache_from_embeddings(
        text_embeddings=text_embeddings,
        image_embeddings=image_embeddings,
        negative_text_embeddings=negative_text_embeddings,
    )
    assert captured_contexts[-1]["negative_text_embeddings"] is negative_text_embeddings


@pytest.mark.ci_cpu
def test_omnidreams_initialize_cache_encodes_cfg_negative_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded_prompts: list[list[str]] = []
    captured_embeddings: dict[str, Any] = {}

    class FakeTextEncoder:
        def __call__(self, prompts: list[str]) -> torch.Tensor:
            encoded_prompts.append(prompts)
            return torch.full((len(prompts), 2, 3), float(len(encoded_prompts)))

    class FakeImageEncoder:
        def __call__(self, image: torch.Tensor) -> torch.Tensor:
            del image
            return torch.ones(1, 1, 1, 2, 2, 2)

    def capture_initialize_cache_from_embeddings(
        self: OmnidreamsPipeline,
        *,
        text_embeddings: torch.Tensor,
        image_embeddings: torch.Tensor,
        negative_text_embeddings: torch.Tensor | None = None,
        view_names: list[str] | None = None,
    ) -> object:
        del self, view_names
        captured_embeddings["text_embeddings"] = text_embeddings
        captured_embeddings["image_embeddings"] = image_embeddings
        captured_embeddings["negative_text_embeddings"] = negative_text_embeddings
        return object()

    def skip_validate_image_resolution(
        self: OmnidreamsPipeline, image: torch.Tensor
    ) -> None:
        del self, image

    monkeypatch.setattr(
        OmnidreamsPipeline,
        "_validate_image_resolution",
        skip_validate_image_resolution,
    )
    monkeypatch.setattr(
        OmnidreamsPipeline,
        "initialize_cache_from_embeddings",
        capture_initialize_cache_from_embeddings,
    )

    pipeline = _make_uninitialized_omnidreams_pipeline()
    pipeline.text_encoder = cast(Any, FakeTextEncoder())
    pipeline.image_encoder = cast(Any, FakeImageEncoder())
    pipeline.diffusion_model.transformer.config = CosmosTransformerConfig(
        guidance_scale=3.0
    )

    pipeline.initialize_cache(
        text=[["positive prompt"]],
        image=torch.randn(1, 1, 1, 3, 4, 4),
    )

    assert encoded_prompts == [["positive prompt"], [NEGATIVE_PROMPT]]
    assert captured_embeddings["negative_text_embeddings"] is not None


@pytest.mark.ci_cpu
def test_bidirectional_transformer_requires_and_wires_negative_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``CosmosTransformer.initialize_autoregressive_cache`` must reject
    missing ``negative_text_embeddings`` under CFG (``guidance_scale > 1``)
    and, when provided, must thread them into the uncond ``network_cache``.

    We exercise the real method end-to-end and only stub out the heavy
    leaves (``RotaryPositionEmbedding3D``, the network's
    ``initialize_cache``, and patchify) so the test stays CPU-only and
    free of irrelevant config plumbing.
    """

    class FakeNetwork:
        # ``cfg.network.{model_channels,num_heads,enable_cross_view_attn}``
        # are read directly by ``initialize_autoregressive_cache``.
        model_channels = 4
        num_heads = 2
        enable_cross_view_attn = False

        def __init__(self) -> None:
            self.cache_kwargs: list[dict[str, Any]] = []

        def initialize_cache(self, **kwargs: Any) -> object:
            self.cache_kwargs.append(kwargs)
            return object()

    class FakeRopeAdapter:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def set_context_parallel_group(self, cp_group: Any = None) -> None:
            del cp_group

    monkeypatch.setattr(
        omnidreams_transformer_module,
        "RotaryPositionEmbedding3D",
        FakeRopeAdapter,
    )

    transformer = CosmosTransformer.__new__(CosmosTransformer)
    torch.nn.Module.__init__(transformer)
    fake_network = FakeNetwork()
    cfg = SimpleNamespace(
        guidance_scale=3.0,
        requires_negative_text_embeddings=True,
        network=SimpleNamespace(
            patch_temporal=1,
            patch_spatial=1,
            model_channels=fake_network.model_channels,
            num_heads=fake_network.num_heads,
            enable_cross_view_attn=fake_network.enable_cross_view_attn,
        ),
        len_t=1,
        window_size_t=1,
        sink_size_t=0,
        h_extrapolation_ratio=1.0,
        w_extrapolation_ratio=1.0,
        dtype=torch.float32,
        num_views=1,
    )
    transformer.config = cast(Any, cfg)
    transformer.cp_groups = HierarchicalCPGroups(rank=0)
    transformer.network = cast(Any, fake_network)
    transformer._output_height = None
    transformer._output_width = None
    transformer._optimized_dit_executor = None
    # ``Transformer.device`` is a property reading from ``self.parameters()``;
    # register a placeholder so it resolves to CPU instead of asserting.
    transformer.register_parameter(
        "_test_device_anchor", torch.nn.Parameter(torch.empty(0, device="cpu"))
    )
    transformer._use_cuda_graph = False
    monkeypatch.setattr(
        transformer,
        "patchify_and_maybe_split_cp",
        lambda x: x,
        raising=False,
    )

    text_embeddings = torch.randn(1, 1, 2, 3)
    image_embeddings = torch.randn(1, 1, 1, 2, 2, 2)
    negative_text_embeddings = torch.randn(1, 1, 2, 3)

    with pytest.raises(AssertionError, match="requires negative_text_embeddings"):
        transformer.initialize_autoregressive_cache(
            height=2,
            width=2,
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings,
        )

    cache = transformer.initialize_autoregressive_cache(
        height=2,
        width=2,
        text_embeddings=text_embeddings,
        image_embeddings=image_embeddings,
        negative_text_embeddings=negative_text_embeddings,
    )

    assert cache.network_cache_uncond is not None
    assert fake_network.cache_kwargs[-1]["context"] is negative_text_embeddings


@pytest.mark.ci_cpu
def test_cosmos_transformer_patchify_casts_to_transformer_dtype() -> None:
    class FakeNetwork:
        def __init__(self) -> None:
            self.patchify_input: torch.Tensor | None = None

        def patchify_and_maybe_split_cp(
            self, x: torch.Tensor, **_kwargs: Any
        ) -> torch.Tensor:
            self.patchify_input = x
            return x

    transformer = CosmosTransformer.__new__(CosmosTransformer)
    torch.nn.Module.__init__(transformer)
    fake_network = FakeNetwork()
    transformer.config = cast(Any, SimpleNamespace(dtype=torch.bfloat16))
    transformer.cp_groups = HierarchicalCPGroups(rank=0)
    transformer.network = cast(Any, fake_network)
    transformer.flatten_thw = False
    transformer.register_parameter(
        "_test_device_anchor", torch.nn.Parameter(torch.empty(0, device="cpu"))
    )

    input_tensor = torch.zeros(1, 1, 1, 1, 2, 2, dtype=torch.float16)
    output = transformer.patchify_and_maybe_split_cp(input_tensor)

    assert fake_network.patchify_input is not None
    assert fake_network.patchify_input.dtype is torch.bfloat16
    assert output.dtype is torch.bfloat16


@pytest.mark.ci_cpu
def test_cosmos_transformer_checkpoint_path_none_keeps_random_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeNetwork(torch.nn.Module):
        def __init__(self, config: object) -> None:
            super().__init__()
            self.config = config
            self.updated_after_load = False

        def set_context_parallel_group(self, **kwargs: object) -> None:
            del kwargs

        def load_state_dict(
            self,
            state_dict: object,
            *args: object,
            **kwargs: object,
        ) -> object:
            del state_dict, args, kwargs
            raise AssertionError("load_state_dict should not run without a checkpoint")

        def update_parameters_after_loading_checkpoint(self) -> None:
            self.updated_after_load = True

    monkeypatch.setattr(
        omnidreams_transformer_module,
        "CosmosDiTNetwork",
        FakeNetwork,
    )

    transformer = CosmosTransformer(
        CosmosTransformerConfig(
            checkpoint_path=None,
            compile_network=False,
            use_cuda_graph=False,
        )
    )

    assert isinstance(transformer.network, FakeNetwork)
    assert transformer.network.updated_after_load is True


@pytest.mark.ci_cpu
@pytest.mark.parametrize(
    (
        "config_name",
        "expected_len_t",
        "expected_window_size_t",
        "expected_skip_finalize_kv_cache",
    ),
    [
        (
            "omnidreams-sv-35steps-chunk2-loc24-cosmos2-2b-res720p-30fps-hdmap-vae-mads1m",
            2,
            24,
            False,
        ),
        (
            "omnidreams-sv-35steps-chunk48-loc48-cosmos2-2b-res720p-30fps-hdmap-vae-mads1m",
            48,
            48,
            True,
        ),
    ],
)
def test_omnidreams_teacher_configs_wire_cfg_negative_text(
    config_name: str,
    expected_len_t: int,
    expected_window_size_t: int,
    expected_skip_finalize_kv_cache: bool,
) -> None:
    pipeline_config = OMNIDREAMS_CONFIGS[config_name]
    transformer_config = pipeline_config.diffusion_model.transformer

    assert isinstance(transformer_config, CosmosTransformerConfig)
    assert transformer_config.guidance_scale > 1.0
    assert transformer_config.requires_negative_text_embeddings
    assert transformer_config.len_t == expected_len_t
    assert transformer_config.window_size_t == expected_window_size_t
    assert transformer_config.skip_finalize_kv_cache is expected_skip_finalize_kv_cache

    scheduler_config = pipeline_config.diffusion_model.scheduler
    assert isinstance(scheduler_config, FlowMatchUniPCSchedulerConfig)
    assert scheduler_config.num_inference_steps == 35
    assert scheduler_config.shift == 5.0


@pytest.mark.manual
def test_omnidreams_streaming_inference():
    num_views = 1
    # Must match the omnidreams checkpoint training resolution
    height = 704
    width = 1280

    device = torch.device("cuda")
    dtype = torch.bfloat16

    image = torch.randn(1, num_views, 1, 3, height, width, device=device, dtype=dtype)
    text = [["Hello, world!"] * num_views]

    config = SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE
    pipeline = config.setup().to(device)
    assert isinstance(pipeline, OmnidreamsPipeline)
    cache = pipeline.initialize_cache(text=text, image=image)

    autoregressive_index = 0
    num_frames = pipeline.get_num_frames(autoregressive_index)
    hdmap = torch.randn(
        1, num_views, num_frames, 3, height, width, device=device, dtype=dtype
    )
    decoded_video = pipeline.generate(autoregressive_index, hdmap=hdmap, cache=cache)
    pipeline.finalize(autoregressive_index, cache=cache)
    assert decoded_video.shape == hdmap.shape

    autoregressive_index = 1
    num_frames = pipeline.get_num_frames(autoregressive_index)
    hdmap = torch.randn(
        1, num_views, num_frames, 3, height, width, device=device, dtype=dtype
    )
    decoded_video = pipeline.generate(autoregressive_index, hdmap=hdmap, cache=cache)
    pipeline.finalize(autoregressive_index, cache=cache)
    assert decoded_video.shape == hdmap.shape


if __name__ == "__main__":
    test_omnidreams_streaming_inference()

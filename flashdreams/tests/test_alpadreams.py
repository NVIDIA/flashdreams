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

from flashdreams.infra.diffusion.scheduler.fm_unipc import (
    FlowMatchUniPCSchedulerConfig,
)
from flashdreams.infra.pipeline import StreamInferencePipeline
from flashdreams.recipes.alpadreams.config import (
    ALPADREAMS_CONFIG_BUILDERS,
    build_sv_2steps_chunk2_loc6_lightvae_lighttae,
)
from flashdreams.recipes.alpadreams.constants import NEGATIVE_PROMPT
from flashdreams.recipes.alpadreams.pipeline import AlpadreamsPipeline
from flashdreams.recipes.alpadreams.transformer import (
    CosmosTransformer,
    CosmosTransformerCache,
    CosmosTransformerConfig,
)
from flashdreams.recipes.alpadreams.transformer.impl.context_parallel import (
    HierarchicalCPGroups,
)


def _make_uninitialized_alpadreams_pipeline() -> AlpadreamsPipeline:
    pipeline = AlpadreamsPipeline.__new__(AlpadreamsPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.diffusion_model = SimpleNamespace(
        device=torch.device("cpu"),
        transformer=SimpleNamespace(config=CosmosTransformerConfig()),
    )
    pipeline.V_group = None
    return pipeline


def test_alpadreams_initialize_cache_from_embeddings_negative_text_optional(
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

    pipeline = _make_uninitialized_alpadreams_pipeline()
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


def test_alpadreams_initialize_cache_encodes_cfg_negative_prompt(
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
        self: AlpadreamsPipeline,
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

    monkeypatch.setattr(
        AlpadreamsPipeline,
        "initialize_cache_from_embeddings",
        capture_initialize_cache_from_embeddings,
    )

    pipeline = _make_uninitialized_alpadreams_pipeline()
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


def test_bidirectional_transformer_requires_and_wires_negative_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeNetwork:
        def __init__(self) -> None:
            self.cache_kwargs: list[dict[str, Any]] = []

        def initialize_cache(self, **kwargs: Any) -> object:
            self.cache_kwargs.append(kwargs)
            return object()

    def fake_parent_initialize_cache(
        self: CosmosTransformer,
        *,
        text_embeddings: torch.Tensor,
        image_embeddings: torch.Tensor,
        view_names: list[str] | None = None,
        **unused: Any,
    ) -> CosmosTransformerCache:
        del self, text_embeddings, view_names, unused
        return CosmosTransformerCache(
            network_cache=cast(Any, object()),
            rope_adapter=cast(Any, object()),
            image=image_embeddings,
            mask_first_block=torch.ones(1),
            mask_other_blocks=torch.zeros(1),
        )

    monkeypatch.setattr(
        CosmosTransformer,
        "initialize_autoregressive_cache",
        fake_parent_initialize_cache,
    )

    transformer = CosmosTransformer.__new__(CosmosTransformer)
    torch.nn.Module.__init__(transformer)
    transformer.config = SimpleNamespace(
        guidance_scale=3.0,
        _pH=2,
        _pW=2,
        _pT=1,
        window_size_t=1,
        sink_size_t=0,
    )
    transformer.cp_groups = HierarchicalCPGroups(rank=0)
    fake_network = FakeNetwork()
    transformer.network = fake_network
    transformer._use_cuda_graph = False

    text_embeddings = torch.randn(1, 1, 2, 3)
    image_embeddings = torch.randn(1, 1, 1, 2, 2, 2)
    negative_text_embeddings = torch.randn(1, 1, 2, 3)

    with pytest.raises(AssertionError, match="negative_text_embeddings is required"):
        transformer.initialize_autoregressive_cache(
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings,
        )

    cache = transformer.initialize_autoregressive_cache(
        text_embeddings=text_embeddings,
        image_embeddings=image_embeddings,
        negative_text_embeddings=negative_text_embeddings,
    )

    assert cache.network_cache_uncond is not None
    assert fake_network.cache_kwargs[-1]["context"] is negative_text_embeddings


@pytest.mark.parametrize(
    (
        "config_name",
        "expected_len_t",
        "expected_window_size_t",
        "expected_skip_finalize_kv_cache",
    ),
    [
        (
            "sv_35steps_chunk2_loc24_cosmos2_2B_res720p_30fps_hdmap_vae_mads1m",
            2,
            24,
            False,
        ),
        (
            "sv_35steps_chunk48_loc48_cosmos2_2B_res720p_30fps_hdmap_vae_mads1m",
            48,
            48,
            True,
        ),
    ],
)
def test_alpadreams_teacher_config_builders_wire_cfg_negative_text(
    config_name: str,
    expected_len_t: int,
    expected_window_size_t: int,
    expected_skip_finalize_kv_cache: bool,
) -> None:
    pipeline_config = ALPADREAMS_CONFIG_BUILDERS[config_name](
        compile_network=False,
        use_cuda_graph=False,
    )
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


def test_alpadreams_streaming_inference():
    num_views = 1
    # Must match the alpadreams checkpoint training resolution
    height = 704
    width = 1280

    device = torch.device("cuda")
    dtype = torch.bfloat16

    image = torch.randn(1, num_views, 1, 3, height, width, device=device, dtype=dtype)
    text = [["Hello, world!"] * num_views]

    config = build_sv_2steps_chunk2_loc6_lightvae_lighttae()
    pipeline = config.setup().to(device)
    assert isinstance(pipeline, AlpadreamsPipeline)
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
    test_alpadreams_streaming_inference()

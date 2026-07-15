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

"""CPU tests for FlashVSR post-processing orchestration."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import flashvsr.postprocess as flashvsr_postprocess
import pytest
import torch
from flashvsr.pipeline import FlashVSRPipeline
from flashvsr.postprocess import FlashVSRPostProcessorConfig

from flashdreams.infra.postprocess import (
    VideoChunk,
    VideoSpec,
)
from flashdreams.infra.postprocess.base import concatenate_video_chunks

pytestmark = pytest.mark.ci_cpu


class _FakeFlashVSRPipelineConfig:
    def __init__(self, kwargs: dict[str, Any], created: list["_FakeFlashVSRPipeline"]):
        self.kwargs = kwargs
        self._created = created

    def setup(self) -> "_FakeFlashVSRPipeline":
        pipeline = _FakeFlashVSRPipeline(self.kwargs)
        self._created.append(pipeline)
        return pipeline


class _FakeFlashVSRPipeline:
    def __init__(self, kwargs: dict[str, Any]) -> None:
        self.kwargs = kwargs
        self.device = torch.device("cpu")
        self.diffusion_model = SimpleNamespace(dtype=torch.float32)
        self.inputs: list[tuple[int, torch.Tensor]] = []
        self.finalized: list[int] = []
        self.cache_initializations = 0
        self.cache_resets = 0
        self.cache_ids: list[int] = []

    def to(self, device: str) -> "_FakeFlashVSRPipeline":
        self.device = torch.device(device)
        return self

    def eval(self) -> "_FakeFlashVSRPipeline":
        return self

    def initialize_cache(self) -> SimpleNamespace:
        self.cache_initializations += 1
        return SimpleNamespace()

    def reset_cache_in_place(self, cache: SimpleNamespace) -> None:
        self.cache_resets += 1
        cache.was_reset = True

    def generate(
        self,
        autoregressive_index: int,
        cache: SimpleNamespace,
        input: torch.Tensor,
    ) -> torch.Tensor:
        self.inputs.append((autoregressive_index, input.detach().cpu()))
        self.cache_ids.append(id(cache))
        return input.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)

    def finalize(self, autoregressive_index: int, cache: SimpleNamespace) -> None:
        self.finalized.append(autoregressive_index)


def _install_fake_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> list[_FakeFlashVSRPipeline]:
    created: list[_FakeFlashVSRPipeline] = []

    def fake_builder(**kwargs: Any) -> _FakeFlashVSRPipelineConfig:
        return _FakeFlashVSRPipelineConfig(kwargs, created)

    monkeypatch.setattr(flashvsr_postprocess, "_build_flashvsr_pipeline", fake_builder)
    return created


def test_flashvsr_postprocess_coalesces_chunks_and_flushes_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = _install_fake_builder(monkeypatch)
    config = FlashVSRPostProcessorConfig(
        device="cpu",
        chunk_size=8,
        scale=2,
        sparse_ratio=1.5,
        attention_mode="sparse",
        tail_policy="replicate_pad",
        dtype="float32",
    )
    session = config.setup().start(VideoSpec(height=4, width=6, fps=12))
    video = torch.linspace(-1.0, 1.0, steps=7 * 3 * 4 * 6).reshape(7, 3, 4, 6)

    assert session.process(VideoChunk(tensor=video[:3], layout="tchw")) == []
    ready = session.process(VideoChunk(tensor=video[3:], layout="tchw"))
    tail = session.flush()
    result = concatenate_video_chunks(
        [*ready, *tail],
        layout="tchw",
    )

    assert result.shape == (7, 3, 8, 12)
    assert len(created) == 1
    pipeline = created[0]
    assert [idx for idx, _ in pipeline.inputs] == [0, 1]
    assert [clip.shape[2] for _, clip in pipeline.inputs] == [5, 8]
    assert pipeline.finalized == [0, 1]
    assert pipeline.kwargs["input_H"] == 4
    assert pipeline.kwargs["input_W"] == 6
    assert pipeline.kwargs["scale"] == 2
    assert pipeline.kwargs["sparse_ratio"] == 1.5
    assert pipeline.kwargs["compile_network"] is True
    assert pipeline.kwargs["use_cuda_graph"] is True
    assert pipeline.kwargs["attention_mode"] == "sparse"


def test_flashvsr_postprocess_can_drop_short_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = _install_fake_builder(monkeypatch)
    config = FlashVSRPostProcessorConfig(
        device="cpu",
        chunk_size=8,
        tail_policy="drop",
        dtype="float32",
    )
    session = config.setup().start(VideoSpec(height=4, width=4, fps=24))

    ready = session.process(VideoChunk(tensor=torch.zeros((4, 3, 4, 4)), layout="tchw"))

    assert ready == []
    assert session.flush() == []
    assert len(created) == 1
    assert created[0].inputs == []


def test_flashvsr_postprocess_does_not_finalize_when_generate_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = _install_fake_builder(monkeypatch)

    def failing_generate(
        self: _FakeFlashVSRPipeline,
        autoregressive_index: int,
        cache: SimpleNamespace,
        input: torch.Tensor,
    ) -> torch.Tensor:
        self.inputs.append((autoregressive_index, input.detach().cpu()))
        raise RuntimeError("simulated FlashVSR failure")

    monkeypatch.setattr(_FakeFlashVSRPipeline, "generate", failing_generate)

    config = FlashVSRPostProcessorConfig(device="cpu", chunk_size=8, dtype="float32")
    session = config.setup().start(VideoSpec(height=4, width=4, fps=24))
    video = torch.zeros((8, 3, 4, 4))

    with pytest.raises(RuntimeError, match="simulated FlashVSR failure"):
        session.process(VideoChunk(tensor=video, layout="tchw"))

    pipeline = created[0]
    assert pipeline.finalized == []
    assert len(pipeline.inputs) == 1


def test_flashvsr_postprocess_rejects_multi_view_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_builder(monkeypatch)
    config = FlashVSRPostProcessorConfig(device="cpu", dtype="float32")
    session = config.setup().start(VideoSpec(height=4, width=4, fps=24))
    multi_view = torch.zeros((1, 2, 5, 3, 4, 4))

    with pytest.raises(ValueError, match="one stream at a time"):
        session.process(VideoChunk(tensor=multi_view, layout="bvtchw"))


def test_flashvsr_postprocessor_declares_distributed_execution() -> None:
    sparse = FlashVSRPostProcessorConfig(attention_mode="sparse")
    full = FlashVSRPostProcessorConfig(attention_mode="full")

    assert not sparse.requires_all_ranks(world_size=1)
    assert full.requires_all_ranks(world_size=2)
    with pytest.raises(ValueError, match="does not support multi-GPU"):
        sparse.validate_execution(world_size=2)


def test_flashvsr_distributed_prepare_warms_both_shapes_and_resets_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = _install_fake_builder(monkeypatch)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        flashvsr_postprocess, "_resolve_postprocess_device", lambda _: "cpu"
    )
    config = FlashVSRPostProcessorConfig(
        device="cpu",
        chunk_size=8,
        attention_mode="full",
        compile_network=True,
        dtype="float32",
    )
    session = config.setup().start(VideoSpec(height=4, width=4, fps=24))

    session.prepare()
    output = session.process(VideoChunk(tensor=torch.ones((5, 3, 4, 4)), layout="tchw"))

    assert len(created) == 1
    pipeline = created[0]
    assert [idx for idx, _ in pipeline.inputs] == [0, 1, 2, 3, 4, 0]
    assert [clip.shape[2] for _, clip in pipeline.inputs] == [5, 8, 8, 8, 8, 5]
    assert pipeline.finalized == [0, 1, 2, 3, 4, 0]
    assert pipeline.cache_initializations == 1
    assert pipeline.cache_resets == 1
    assert len(set(pipeline.cache_ids)) == 1
    assert len(output) == 1


def test_flashvsr_pipeline_resets_nested_caches_in_place() -> None:
    class _Resettable:
        def __init__(self) -> None:
            self.calls = 0

        def reset(self) -> None:
            self.calls += 1

    encoder = _Resettable()
    transformer = _Resettable()
    decoder = _Resettable()
    cache = SimpleNamespace(
        encoder_cache=encoder,
        transformer_cache=transformer,
        decoder_cache=decoder,
        final_state=object(),
        autoregressive_index=4,
        event_profiler=object(),
    )
    identities = tuple(id(item) for item in (encoder, transformer, decoder))

    FlashVSRPipeline.reset_cache_in_place(cast(Any, cache))

    assert tuple(id(item) for item in (encoder, transformer, decoder)) == identities
    assert (encoder.calls, transformer.calls, decoder.calls) == (1, 1, 1)
    assert cache.final_state is None
    assert cache.autoregressive_index is None
    assert cache.event_profiler is None


def test_flashvsr_postprocessor_reports_aligned_output_spec() -> None:
    config = FlashVSRPostProcessorConfig(scale=2)

    output = config.output_spec(VideoSpec(height=416, width=640, fps=24))

    assert output == VideoSpec(height=768, width=1280, fps=24)


def test_flashvsr_postprocess_preserves_input_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_builder(monkeypatch)
    config = FlashVSRPostProcessorConfig(device="cpu", chunk_size=8, dtype="float32")
    session = config.setup().start(VideoSpec(height=4, width=4, fps=24))

    output = session.process(
        VideoChunk(
            tensor=torch.zeros((5, 3, 4, 4)),
            layout="tchw",
            metadata={"autoregressive_index": 7},
        )
    )

    assert output[0].metadata == {
        "source": "flashvsr",
        "input_chunks": ({"autoregressive_index": 7},),
    }

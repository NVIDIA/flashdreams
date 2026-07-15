# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for runner-owned streaming post-processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Literal

import pytest

from flashdreams.infra.postprocess import (
    VideoPostprocessChainConfig,
    VideoPostProcessorConfig,
    create_runner_postprocess_stream,
)

pytestmark = pytest.mark.ci_cpu


@dataclass(kw_only=True)
class _DistributedProcessorConfig(VideoPostProcessorConfig):
    attention_mode: Literal["sparse", "full"] = "sparse"
    _target: type[object] = field(default_factory=lambda: object)

    def requires_all_ranks(self, *, world_size: int) -> bool:
        return world_size > 1 and self.attention_mode == "full"

    def validate_execution(self, *, world_size: int) -> None:
        if world_size > 1 and self.attention_mode == "sparse":
            raise ValueError("sparse processor does not support multi-GPU")


@dataclass(kw_only=True)
class _RankZeroProcessorConfig(VideoPostProcessorConfig):
    _target: type[object] = field(default_factory=lambda: object)


def _config(postprocess: VideoPostprocessChainConfig, *, layout: str | None = "tchw"):
    return SimpleNamespace(
        postprocess=postprocess,
        postprocess_output_layout=layout,
        postprocess_per_view=False,
        fps=16,
    )


def test_disabled_runner_postprocess_does_not_require_layout() -> None:
    stream = create_runner_postprocess_stream(
        _config(VideoPostprocessChainConfig(), layout=None), world_size=1
    )

    assert stream is None


def test_enabled_runner_postprocess_requires_layout() -> None:
    config = _config(
        VideoPostprocessChainConfig(
            processors=(_DistributedProcessorConfig(attention_mode="full"),)
        ),
        layout=None,
    )

    with pytest.raises(ValueError, match="postprocess_output_layout"):
        create_runner_postprocess_stream(config, world_size=1)


def test_mixed_chain_validates_every_processor_under_multi_gpu() -> None:
    config = _config(
        VideoPostprocessChainConfig(
            processors=(
                _DistributedProcessorConfig(attention_mode="full"),
                _DistributedProcessorConfig(attention_mode="sparse"),
            )
        )
    )

    with pytest.raises(ValueError, match="sparse processor"):
        create_runner_postprocess_stream(config, world_size=2)


def test_full_attention_chain_requires_all_ranks() -> None:
    chain = VideoPostprocessChainConfig(
        processors=(_DistributedProcessorConfig(attention_mode="full"),)
    )

    assert chain.requires_all_ranks(world_size=2)
    assert not chain.requires_all_ranks(world_size=1)


def test_rank_zero_processor_is_skipped_on_nonzero_rank() -> None:
    config = _config(
        VideoPostprocessChainConfig(processors=(_RankZeroProcessorConfig(),))
    )

    stream = create_runner_postprocess_stream(
        config,
        world_size=2,
        is_rank_zero=False,
    )

    assert stream is None


def test_all_rank_processor_stream_does_not_collect_on_nonzero_rank() -> None:
    config = _config(
        VideoPostprocessChainConfig(
            processors=(_DistributedProcessorConfig(attention_mode="full"),)
        )
    )

    stream = create_runner_postprocess_stream(
        config,
        world_size=2,
        is_rank_zero=False,
    )

    assert stream is not None
    assert not stream.collect_output

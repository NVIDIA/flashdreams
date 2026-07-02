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

"""CPU tests for distributed post-processing orchestration on :class:`Runner`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
from unittest.mock import MagicMock, patch

import pytest
import torch

from flashdreams.infra.pipeline import StreamInferencePipelineConfig
from flashdreams.infra.postprocess import (
    VideoPostProcessorConfig,
    VideoPostprocessChainConfig,
)
from flashdreams.infra.runner import Runner, RunnerConfig

pytestmark = pytest.mark.ci_cpu


@dataclass(kw_only=True)
class _FlashVSRLikePostProcessorConfig(VideoPostProcessorConfig):
    attention_mode: Literal["sparse", "full"] = "sparse"
    _target: type[object] = field(default_factory=lambda: object)

    def uses_context_parallelism(self) -> bool:
        return self.attention_mode == "full"


@dataclass(kw_only=True)
class _MinimalRunnerConfig(RunnerConfig):
    runner_name: str = "test-runner"
    pipeline: StreamInferencePipelineConfig = field(
        default_factory=lambda: MagicMock(spec=StreamInferencePipelineConfig)
    )


class _MinimalRunner(Runner[_MinimalRunnerConfig, MagicMock]):
    def run(self) -> None:
        raise NotImplementedError


def _runner(
    *,
    is_rank_zero: bool,
    world_size: int,
    postprocess: VideoPostprocessChainConfig,
) -> _MinimalRunner:
    config = _MinimalRunnerConfig(postprocess=postprocess)
    runner = _MinimalRunner.__new__(_MinimalRunner)
    runner.config = config
    runner.local_rank = 0 if is_rank_zero else 1
    runner.world_size = world_size
    runner.global_rank = 0 if is_rank_zero else 1
    runner.is_rank_zero = is_rank_zero
    runner.pipeline = MagicMock()
    return runner


def test_apply_output_postprocess_disabled_returns_input() -> None:
    runner = _runner(
        is_rank_zero=True,
        world_size=1,
        postprocess=VideoPostprocessChainConfig(),
    )
    video = torch.zeros(3, 3, 4, 5)

    result = runner.apply_output_postprocess(video, layout="tchw")

    assert result is video


@patch.object(_MinimalRunner, "postprocess_video_tensor", return_value=torch.ones(2, 3, 4, 5))
def test_apply_output_postprocess_rank_zero_only(mock_postprocess: MagicMock) -> None:
    runner = _runner(
        is_rank_zero=True,
        world_size=1,
        postprocess=VideoPostprocessChainConfig(
            processors=(_FlashVSRLikePostProcessorConfig(attention_mode="sparse"),)
        ),
    )
    video = torch.zeros(3, 3, 4, 5)

    result = runner.apply_output_postprocess(video, layout="tchw")

    mock_postprocess.assert_called_once()
    assert result is not None
    assert torch.equal(result, torch.ones(2, 3, 4, 5))


@patch.object(_MinimalRunner, "postprocess_video_tensor", return_value=torch.ones(2, 3, 4, 5))
def test_apply_output_postprocess_non_zero_skips_rank_zero_only(
    mock_postprocess: MagicMock,
) -> None:
    runner = _runner(
        is_rank_zero=False,
        world_size=1,
        postprocess=VideoPostprocessChainConfig(
            processors=(_FlashVSRLikePostProcessorConfig(attention_mode="sparse"),)
        ),
    )

    result = runner.apply_output_postprocess(torch.zeros(3, 3, 4, 5), layout="tchw")

    mock_postprocess.assert_not_called()
    assert result is None


@patch.object(_MinimalRunner, "postprocess_video_tensor", return_value=torch.ones(2, 3, 4, 5))
def test_apply_output_postprocess_full_attn_runs_on_all_ranks(
    mock_postprocess: MagicMock,
) -> None:
    postprocess = VideoPostprocessChainConfig(
        processors=(_FlashVSRLikePostProcessorConfig(attention_mode="full"),)
    )

    rank_zero = _runner(is_rank_zero=True, world_size=2, postprocess=postprocess)
    rank_one = _runner(is_rank_zero=False, world_size=2, postprocess=postprocess)
    video = torch.zeros(3, 3, 4, 5)

    rank_zero_result = rank_zero.apply_output_postprocess(video, layout="tchw")
    rank_one_result = rank_one.apply_output_postprocess(video, layout="tchw")

    assert mock_postprocess.call_count == 2
    assert rank_zero_result is not None
    assert rank_one_result is None


def test_apply_output_postprocess_rejects_sparse_under_multi_gpu() -> None:
    runner = _runner(
        is_rank_zero=True,
        world_size=2,
        postprocess=VideoPostprocessChainConfig(
            processors=(_FlashVSRLikePostProcessorConfig(attention_mode="sparse"),)
        ),
    )

    with pytest.raises(ValueError, match="flashvsr-v1.1-full-attn"):
        runner.apply_output_postprocess(torch.zeros(3, 3, 4, 5), layout="tchw")

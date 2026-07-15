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

"""``Runner`` ABC + ``RunnerConfig`` base: CLI-side driver around a pipeline."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Generic, TypeVar, cast

import torch
import tyro

from flashdreams.core.distributed import init as init_distributed
from flashdreams.core.io.disk import preflight_runtime_write_paths
from flashdreams.infra.config import InstantiateConfig, derive_config
from flashdreams.infra.pipeline import (
    StreamInferencePipeline,
    StreamInferencePipelineConfig,
)
from flashdreams.infra.postprocess import (
    VideoPostprocessChainConfig,
    VideoPostprocessStream,
    VideoTensorLayout,
    create_runner_postprocess_stream,
)


def _is_torchrun_env() -> bool:
    """Return ``True`` iff torchrun set the rendezvous env vars (``RANK`` + ``WORLD_SIZE``)."""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


@dataclass(kw_only=True)
class RunnerConfig(InstantiateConfig):
    """Base config every integration runner extends with its own I/O fields."""

    _target: type["Runner"] = field(default_factory=lambda: Runner)

    runner_name: str
    """Registry key and ``flashdreams-run`` subcommand name. By convention
    mirrors the wrapped pipeline's ``name`` slug."""

    description: Annotated[str, tyro.conf.Suppress] = ""
    """One-line subcommand description shown next to the slug in
    ``flashdreams-run --help``. ``tyro.conf.Suppress`` hides it from
    per-runner ``--help`` (it's metadata, not a knob); a non-empty
    value is enforced for in-tree runners by the registry test."""

    pipeline: StreamInferencePipelineConfig
    """Wrapped pipeline config; the runner instantiates and drives it."""

    postprocess: VideoPostprocessChainConfig = field(
        default_factory=VideoPostprocessChainConfig
    )
    """Optional video post-processing chain for decoded output chunks."""

    postprocess_output_layout: VideoTensorLayout | None = None
    """Decoded output layout; required when :attr:`postprocess` is enabled."""

    postprocess_per_view: bool = False
    """Attach one post-processing session per view for multi-view outputs."""

    output_dir: Path = Path("outputs")
    """Directory the runner writes outputs into. Created on demand."""

    device: str = "cuda"
    """PyTorch device string passed to ``pipeline.to(device)``. Under
    ``torchrun`` the runner overrides this with ``cuda:LOCAL_RANK`` so
    each rank pins its own GPU."""

    offset_seed_by_global_rank: bool = True
    """Offset ``pipeline.diffusion_model.seed`` by ``global_rank`` when
    running distributed. Defaults to ``True`` so each rank draws from a
    distinct RNG stream while preserving deterministic replay per rank."""


RunnerConfigT = TypeVar("RunnerConfigT", bound=RunnerConfig)
PipelineT = TypeVar("PipelineT", bound=StreamInferencePipeline[Any, Any, Any])
"""Pipeline type parameter for :class:`Runner`. The bound's three cache
slots are ``Any`` so integration pipelines parameterized with their own cache
subclasses pass ty's invariant generic check."""


class Runner(ABC, Generic[RunnerConfigT, PipelineT]):
    """Uniform end-to-end driver around a :class:`StreamInferencePipeline`.

    Subclasses own the integration-specific :class:`RunnerConfig` and the
    body of :meth:`run`. The base ``__init__`` bridges the launcher to
    ``torch.distributed`` (so integration transformers auto-detect their CP
    size from the ``WORLD`` group), pins ``cuda:LOCAL_RANK``, and builds
    the pipeline -- subclasses don't reimplement construction.

    Multi-GPU contract:
        Launch with ``torchrun --nproc_per_node=N --no-python flashdreams-run <slug>``.
        ``Runner.__init__`` initializes ``torch.distributed`` (if it
        isn't already) before pipeline construction so context-parallel
        transformers shard tokens across ``WORLD``. Subclasses gate
        per-rollout I/O on :attr:`is_rank_zero`; compute (``generate`` /
        ``finalize``) runs on every rank. Runner-level post-processing operates
        on decoded output chunks, and context-parallel processors execute on
        every rank before rank-zero persistence.
    """

    config: RunnerConfigT
    pipeline: PipelineT

    def __init__(self, config: RunnerConfigT) -> None:
        # Bridge ``torchrun`` -> ``torch.distributed`` *before*
        # ``config.pipeline.setup()`` so integration transformers can pick up
        # the CP world size at construction time. Idempotent: skipped
        # when distributed is already initialized (long-lived servers
        # that init once and call us repeatedly).
        if _is_torchrun_env() and not torch.distributed.is_initialized():
            init_distributed()

        if torch.distributed.is_initialized():
            self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            self.world_size = torch.distributed.get_world_size()
            self.global_rank = torch.distributed.get_rank()
            device = f"cuda:{self.local_rank}"
        else:
            self.local_rank = 0
            self.world_size = 1
            self.global_rank = 0
            device = config.device
        self.is_rank_zero = self.global_rank == 0
        preflight_runtime_write_paths(output_dir=config.output_dir)

        # Keep per-rank RNG streams distinct under torchrun without mutating
        # the caller's literal config.
        effective_config = config
        base_seed = config.pipeline.diffusion_model.seed
        if (
            config.offset_seed_by_global_rank
            and base_seed is not None
            and self.global_rank != 0
        ):
            effective_config = cast(
                RunnerConfigT,
                derive_config(
                    config,
                    pipeline=dict(
                        diffusion_model=dict(seed=base_seed + self.global_rank),
                    ),
                ),
            )  # ty:ignore[redundant-cast]
        self.config = effective_config

        pipeline = self.config.pipeline.setup()
        self.pipeline = pipeline.to(device=device).eval()

    def create_postprocess_stream(
        self,
        *,
        fps: float | None = None,
        move_to_cpu: bool = True,
    ) -> VideoPostprocessStream:
        """Create one stateful post-processing stream for a rollout."""
        stream = create_runner_postprocess_stream(
            self.config,
            world_size=self.world_size,
            is_rank_zero=self.is_rank_zero,
            fps=fps,
        )
        if stream is not None:
            stream.collect_output = self.is_rank_zero
            stream.move_to_cpu = move_to_cpu
            return stream

        layout = self.config.postprocess_output_layout
        if layout is None:
            raise ValueError("Runner output collection requires an output layout.")
        return VideoPostprocessStream(
            postprocess=VideoPostprocessChainConfig(),
            output_layout=layout,
            fps=fps,
            collect_output=self.is_rank_zero,
            move_to_cpu=move_to_cpu,
        )

    @abstractmethod
    def run(self) -> None:
        """Generate and persist one rollout's outputs.

        Implementations:

        1. Resolve integration-specific I/O (load image, decode prompt file, ...).
        2. Build the per-rollout cache via ``self.pipeline.initialize_cache(...)``.
        3. Loop ``generate`` + ``finalize`` for the configured number of AR steps.
        4. Persist outputs under ``self.config.output_dir`` and log the
           absolute paths.
        """

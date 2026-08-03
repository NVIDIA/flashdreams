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

"""Wan 2.1 transformer adapter with Plücker camera control for Lingbot World."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable
from typing import cast, overload

import torch
from torch import Tensor
from torch.distributed import ProcessGroup

from flashdreams.recipes.wan.transformer.wan21 import (
    Wan21Transformer,
    Wan21TransformerCache,
    Wan21TransformerConfig,
)
from lingbot.encoder.camctrl import I2VCamCtrlEmbeddings

from .impl.network import (
    LingbotWorldDiTNetwork,
    LingbotWorldDiTNetwork14BConfig,
    LingbotWorldDiTNetworkCache,
    LingbotWorldDiTNetworkConfig,
)

LINGBOT_WORLD_MIN_CHECKPOINT_FREE_GB = 200.0
"""First-run storage budget documented for LingBot-World model caches."""


@dataclass(kw_only=True)
class LingbotWorldTransformerCache(Wan21TransformerCache):
    """Long-lived AR cache for the Lingbot World transformer.

    Narrows :class:`Wan21TransformerCache`'s network-cache slots to the
    Plücker-aware Lingbot variant. Inherits ``rope_adapter`` /
    ``rope_freqs`` / ``autoregressive_index`` from the parent and the
    same ``start`` / ``finalize`` lifecycle.
    """

    network_cache: LingbotWorldDiTNetworkCache
    """Conditional per-block KV / cross-attn cache."""

    network_cache_uncond: LingbotWorldDiTNetworkCache | None = None
    """Unconditional per-block caches; ``None`` disables CFG."""


@dataclass(kw_only=True)
class LingbotWorldTransformerConfig(Wan21TransformerConfig):
    """Config for the Lingbot World transformer.

    Bakes in the temporal layout (``len_t``, ``window_size_t``,
    ``sink_size_t``); per-rollout spatial layout (``height``, ``width``)
    is supplied to
    :meth:`Wan21Transformer.initialize_autoregressive_cache`. CP size is
    auto-detected from ``torch.distributed.get_world_size()`` (see
    :class:`Wan21TransformerConfig`).
    """

    _target: type[LingbotWorldTransformer] = field(
        default_factory=lambda: LingbotWorldTransformer
    )

    network: LingbotWorldDiTNetworkConfig = field(
        default_factory=LingbotWorldDiTNetwork14BConfig
    )
    checkpoint_min_free_gb: float | None = LINGBOT_WORLD_MIN_CHECKPOINT_FREE_GB


class LingbotWorldTransformer(Wan21Transformer):
    """Lingbot World DiT (Wan 2.1 + per-block Plücker camera control)."""

    def __init__(self, config: LingbotWorldTransformerConfig) -> None:
        super().__init__(config)
        self._pipeline_double_buffer_call: (
            Callable[..., tuple[Tensor, Tensor]] | None
        ) = None

    def set_context_parallel_group(self, cp_group: ProcessGroup | None) -> None:
        """Bind a DiT-only context-parallel group before cache construction.

        The normal runner uses the global process group that exists when the
        transformer is constructed. A disaggregated deployment constructs the
        stage-local weights first, then creates a subgroup containing only DiT
        ranks. Rebinding is safe until a rollout cache or CUDA graph has bound
        shapes and storage.

        Args:
            cp_group: DiT-only process group, or ``None`` to disable CP.

        Raises:
            RuntimeError: A rollout cache has already been initialized.
        """
        if self._output_height is not None or self._output_width is not None:
            raise RuntimeError(
                "Context parallelism must be configured before initializing "
                "a LingBot rollout cache."
            )
        self._cp_group = cp_group
        self._cp_size = cp_group.size() if cp_group is not None else 1
        network = getattr(self.network, "_orig_mod", self.network)
        network.set_context_parallel_group(cp_group)
        if self._use_cuda_graph:
            self._cuda_graph_dispatch.reset()

    def configure_pipeline_parallel(
        self,
        *,
        stage_index: int,
        stage_count: int,
        group: ProcessGroup,
        ranks: tuple[int, ...],
    ) -> None:
        """Partition DiT layers and bind a fixed NCCL pipeline group.

        Args:
            stage_index: Zero-based position inside the pipeline group.
            stage_count: Number of pipeline stages.
            group: NCCL process group containing the pipeline ranks.
            ranks: Global ranks ordered from input to output stage.

        Raises:
            RuntimeError: A rollout cache exists or CUDA graphs are enabled.
        """
        if self._output_height is not None or self._output_width is not None:
            raise RuntimeError(
                "Pipeline parallelism must be configured before cache initialization."
            )
        if self._use_cuda_graph:
            raise RuntimeError(
                "Pipeline-parallel DiT stages do not support CUDA graph capture."
            )
        network = getattr(self.network, "_orig_mod", self.network)
        assert isinstance(network, LingbotWorldDiTNetwork)
        network.configure_pipeline_parallel(
            stage_index=stage_index,
            stage_count=stage_count,
            group=group,
            ranks=ranks,
        )
        double_buffer_call = network.forward_pipeline_double_buffered
        if self.config.compile_network:
            double_buffer_call = torch.compile(
                double_buffer_call,
                mode="max-autotune-no-cudagraphs",
            )
        self._pipeline_double_buffer_call = cast(
            Callable[..., tuple[Tensor, Tensor]],
            double_buffer_call,
        )

    @torch.no_grad()
    def replace_text_embeddings(
        self,
        cache: LingbotWorldTransformerCache,
        text_embeddings: Tensor,
    ) -> None:
        """Swap the rollout's conditional cross-attention text context."""
        network = getattr(self.network, "_orig_mod", self.network)
        assert isinstance(network, LingbotWorldDiTNetwork)
        network.replace_text_embeddings(cache.network_cache, text_embeddings)
        if self._use_cuda_graph:
            self._cuda_graph_dispatch.reset()

    def predict_flow(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: LingbotWorldTransformerCache,
        input: I2VCamCtrlEmbeddings,
    ) -> Tensor:
        return super().predict_flow(
            noisy_latent=noisy_latent,
            timestep=timestep,
            cache=cache,
            input=input.i2v,
            network_extra_kwargs={"plucker": input.plucker},
        )

    def predict_flow_double_buffered(
        self,
        *,
        noisy_latents: tuple[Tensor, Tensor],
        timesteps: tuple[Tensor, Tensor],
        caches: tuple[LingbotWorldTransformerCache, LingbotWorldTransformerCache],
        inputs: tuple[I2VCamCtrlEmbeddings, I2VCamCtrlEmbeddings],
    ) -> tuple[Tensor, Tensor]:
        """Predict two session flows with a fill-and-drain pipeline schedule.

        Args:
            noisy_latents: Per-session noisy latent tokens.
            timesteps: Per-session diffusion timesteps.
            caches: Per-session transformer caches.
            inputs: Per-session patchified I2V and camera-control payloads.

        Returns:
            Per-session flow predictions in input order.

        Raises:
            RuntimeError: Pipeline parallelism is not configured or CFG is enabled.
        """
        if self._pipeline_double_buffer_call is None:
            raise RuntimeError("Double buffering requires pipeline parallelism.")
        if any(cache.network_cache_uncond is not None for cache in caches):
            raise RuntimeError(
                "Double-buffered LingBot inference does not support CFG."
            )

        autoregressive_indices = tuple(cache.autoregressive_index for cache in caches)
        if autoregressive_indices[0] != autoregressive_indices[1]:
            raise RuntimeError(
                "Double-buffered sessions must have the same autoregressive index."
            )
        autoregressive_index = autoregressive_indices[0]
        if autoregressive_index < 0:
            raise RuntimeError("Call cache.start() before predicting flow.")

        network_inputs = tuple(
            self._build_network_input(noisy_latent, input.i2v)
            for noisy_latent, input in zip(noisy_latents, inputs)
        )
        network_timesteps = tuple(
            self._maybe_build_per_token_timestep(
                timestep=timestep,
                input=input.i2v,
                autoregressive_index=autoregressive_index,
            )
            for timestep, input in zip(timesteps, inputs)
        )
        rope_freqs = tuple(cache.rope_freqs for cache in caches)
        if any(value is None for value in rope_freqs):
            raise RuntimeError("cache.start() must populate RoPE frequencies.")

        return self._pipeline_double_buffer_call(
            pluckers=(inputs[0].plucker, inputs[1].plucker),
            xs=(network_inputs[0], network_inputs[1]),
            timesteps=(network_timesteps[0], network_timesteps[1]),
            caches=(caches[0].network_cache, caches[1].network_cache),
            rope_freqs=cast(tuple[Tensor, Tensor], rope_freqs),
            current_chunk_idx=autoregressive_index,
            eager_mode=False,
        )

    @overload
    def patchify_and_maybe_split_cp(self, x: Tensor) -> Tensor: ...
    @overload
    def patchify_and_maybe_split_cp(
        self, x: I2VCamCtrlEmbeddings
    ) -> I2VCamCtrlEmbeddings: ...
    def patchify_and_maybe_split_cp(
        self, x: Tensor | I2VCamCtrlEmbeddings
    ) -> Tensor | I2VCamCtrlEmbeddings:
        """Patchify and (optionally) split for context parallelism."""
        if isinstance(x, I2VCamCtrlEmbeddings):
            if x._is_patchified:
                return x
            return I2VCamCtrlEmbeddings(
                i2v=super().patchify_and_maybe_split_cp(x.i2v),
                plucker=super().patchify_and_maybe_split_cp(x.plucker),
                _is_patchified=True,
            )
        return super().patchify_and_maybe_split_cp(x)

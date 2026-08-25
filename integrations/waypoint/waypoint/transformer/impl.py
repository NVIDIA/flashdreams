# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FlashDreams transformer adapter for one-action Waypoint denoising."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from flashdreams.infra.diffusion.transformer import (
    Transformer,
    TransformerAutoregressiveCache,
    TransformerConfig,
)
from waypoint.checkpoint import load_waypoint_state_dict
from waypoint.controls import WaypointControl, make_control_context
from waypoint.spec import WaypointModelSpec
from waypoint.transformer.cache import WaypointKVCache
from waypoint.transformer.network import WaypointDiT, WaypointDiTConfig


@dataclass(kw_only=True)
class WaypointTransformerCache(TransformerAutoregressiveCache):
    """Long-lived sparse history for an autoregressive Waypoint rollout."""

    kv_cache: WaypointKVCache = field(default_factory=WaypointKVCache)
    """Per-block sparse causal K/V history."""

    batch_size: int
    """Batch size fixed when the rollout starts."""

    autoregressive_index: int = -1
    """Current latent action index; ``-1`` before the first ``start`` call."""

    def start(self, autoregressive_index: int) -> None:
        """Mark the action whose repeated denoise passes may replace K/V state.

        Args:
            autoregressive_index: Zero-indexed latent action to generate.

        Raises:
            ValueError: The action index is negative or skips rollout history.
        """
        if autoregressive_index < 0:
            raise ValueError(
                f"autoregressive_index must be non-negative, got {autoregressive_index}"
            )
        if self.autoregressive_index >= 0 and autoregressive_index not in (
            self.autoregressive_index,
            self.autoregressive_index + 1,
        ):
            raise ValueError(
                "Waypoint actions must be generated in order or re-evaluated in place; "
                f"got {autoregressive_index} after {self.autoregressive_index}"
            )
        self.autoregressive_index = autoregressive_index
        self.kv_cache.set_frozen(True)


@dataclass(kw_only=True)
class WaypointTransformerConfig(TransformerConfig):
    """Construction config for the one-action Waypoint transformer adapter."""

    _target: type["WaypointTransformer"] = field(
        default_factory=lambda: WaypointTransformer
    )

    network: WaypointDiTConfig = field(default_factory=WaypointDiTConfig)
    """Native checkpoint-compatible Waypoint DiT."""

    dtype: torch.dtype = torch.bfloat16
    """Network parameter and activation dtype."""

    checkpoint_path: str | None = None
    """Raw Waypoint safetensors path; ``None`` retains random initialization."""


class WaypointTransformer(Transformer[WaypointTransformerCache]):
    """Adapt Waypoint's internal patchifier to FlashDreams flow prediction.

    Waypoint owns its spatial patchifier, so FlashDreams sees a one-frame
    latent action rather than pre-patchified tokens.  This keeps the external
    streaming layout conventional while preserving the checkpoint's learned
    convolutional patch embedding exactly.
    """

    config: WaypointTransformerConfig
    network: WaypointDiT

    def __init__(self, config: WaypointTransformerConfig) -> None:
        super().__init__(config)
        self.config = config
        self.network = config.network.setup().to(dtype=config.dtype)
        self.network.eval()
        if config.checkpoint_path is not None:
            from flashdreams.core.checkpoint.load import load_checkpoint

            state_dict = load_checkpoint(config.checkpoint_path)
            if not isinstance(state_dict, dict):
                raise RuntimeError(
                    "Waypoint checkpoint loader did not return a state dict"
                )
            load_waypoint_state_dict(self.network, state_dict, spec=self.spec)
        self._batch_size: int | None = None

    @property
    def spec(self) -> WaypointModelSpec:
        """Return the immutable architecture contract of the owned DiT."""
        return self.network.spec

    @property
    def latent_shape(self) -> tuple[int, ...]:
        """Return the internal one-action latent layout ``[B, 1, C, H, W]``."""
        if self._batch_size is None:
            raise RuntimeError(
                "latent_shape requires initialize_autoregressive_cache(batch_size=...)"
            )
        return self.spec.latent_shape(self._batch_size)

    def initialize_autoregressive_cache(
        self, *, batch_size: int, **context: Any
    ) -> WaypointTransformerCache:
        """Allocate sparse history for a fixed-batch Waypoint rollout.

        Args:
            batch_size: Number of actions generated together.
            context: Rejected when non-empty; Waypoint 1.5 has no one-shot context.

        Returns:
            Empty K/V history ready for ``cache.start(0)``.

        Raises:
            ValueError: The batch size is invalid or one-shot context was supplied.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if context:
            raise ValueError(
                "Waypoint 1.5 has no text/image context encoder; unexpected "
                f"cache context keys: {sorted(context)}"
            )
        self._batch_size = batch_size
        return WaypointTransformerCache(
            batch_size=batch_size,
            kv_cache=WaypointKVCache(use_fixed_attention=True),
        )

    def patchify_and_maybe_split_cp(self, x: Any) -> Any:
        """Convert external video latents to Waypoint's internal frame-first layout.

        Args:
            x: Video latent in ``[B, C, T, H, W]`` layout or a non-tensor control.

        Returns:
            Tensor latents in ``[B, T, C, H, W]`` layout; controls are unchanged.
        """
        if not isinstance(x, Tensor):
            return x
        if x.ndim != 5:
            raise ValueError(f"Waypoint latent must have rank 5, got {x.ndim}")
        return x.permute(0, 2, 1, 3, 4).contiguous()

    def unpatchify_and_maybe_gather_cp(self, x: Tensor) -> Tensor:
        """Convert Waypoint's internal frame-first latent layout to video layout.

        Args:
            x: Internal latent in ``[B, T, C, H, W]`` layout.

        Returns:
            Video latent in ``[B, C, T, H, W]`` layout.
        """
        if x.ndim != 5:
            raise ValueError(f"Waypoint latent must have rank 5, got {x.ndim}")
        return x.permute(0, 2, 1, 3, 4).contiguous()

    def predict_flow(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: WaypointTransformerCache,
        input: WaypointControl | None = None,
    ) -> Tensor:
        """Predict one flow field using the current action's control event.

        Args:
            noisy_latent: Internal noisy action in ``[B, 1, C, H, W]`` layout.
            timestep: Scalar sigma supplied by the fixed Euler scheduler.
            cache: Per-rollout K/V history after ``cache.start``.
            input: Optional public keyboard/mouse control for this action.

        Returns:
            Flow prediction with the same layout as ``noisy_latent``.

        Raises:
            RuntimeError: Called before an action has been selected with ``start``.
            TypeError: ``input`` is not a :class:`WaypointControl`.
        """
        if cache.autoregressive_index < 0:
            raise RuntimeError("cache.start(autoregressive_index) must run first")
        if input is not None and not isinstance(input, WaypointControl):
            raise TypeError(
                f"Waypoint input must be WaypointControl or None, got {type(input)}"
            )
        sigma = (
            timestep.reshape(1)
            .expand(cache.batch_size)
            .to(device=noisy_latent.device, dtype=noisy_latent.dtype)
        )
        if input is None:
            return self.network(
                noisy_latent,
                sigma=sigma,
                frame_index=cache.autoregressive_index,
                kv_cache=cache.kv_cache,
            )
        control = make_control_context(
            input,
            frame_index=cache.autoregressive_index,
            batch_size=cache.batch_size,
            dtype=noisy_latent.dtype,
            device=noisy_latent.device,
            spec=self.spec,
        )
        return self.network(
            noisy_latent,
            sigma=sigma,
            frame_index=cache.autoregressive_index,
            kv_cache=cache.kv_cache,
            button=control["button"],
            mouse=control["mouse"],
            scroll=control["scroll"],
        )

    def finalize_kv_cache(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: WaypointTransformerCache,
        input: WaypointControl | None = None,
    ) -> None:
        """Commit the clean action's K/V entries after its Euler solve.

        Denoising repeatedly replaces a provisional current-frame slot. Only
        this final sigma-zero pass may persist that slot into long-term history;
        otherwise the next action has no visual world state to condition on.
        """
        cache.kv_cache.set_frozen(False)
        try:
            _ = self.predict_flow(noisy_latent, timestep, cache, input)
        finally:
            cache.kv_cache.set_frozen(True)

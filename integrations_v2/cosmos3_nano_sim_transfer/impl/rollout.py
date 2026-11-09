# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-view Transfer rollout on ``cosmos_interactive``'s bounded stream cache.

One latent per step, no opening frames: every RGB latent is generated from
noise, conditioned on the prompt and on a control latent per step. Per latent
``k`` the stream cache holds one block of two halves, ``[c_k | r_k]``: the
control's keys and values, then the clean RGB's. A step runs:

1. the control pass: control ``k`` reads the prompt, every cached block before
   it and itself, and writes ``c_k``. It runs only once ``r_(k-1)`` is in the
   cache, so a control reads strictly earlier clean RGB
   (``causal_control_with_rgb_history``);
2. the denoise passes: the noisy latent reads the prompt, the earlier blocks,
   ``c_k`` and itself, and writes nothing;
3. the clean replay: the denoised latent reads the same keys and writes ``r_k``.

Each pass sees exactly the keys it is handed, so no attention mask is needed:
selecting the history slice is the mask. The cache pins the first
``sink_frames`` generated latents and rolls the rest of a ``window_frames``
window, as cosmos-framework counts it (``2 * sinks`` pinned entries, then
``2 * (window - sinks - 1)`` recent entries, plus ``c_k`` for a target), so
memory is bounded by the window whatever the clip length.

Control and RGB of a latent share their positions; temporal positions start
at ``text tokens + temporal_modality_margin`` and advance ``base_fps / fps``
per latent. Noise follows cosmos-framework's Transfer seeding: initial noise
``seed + k`` in the ``[C, 1, H, W]`` latent layout and the checkpoint dtype,
re-noise ``seed + k * 1_000_003 + (step + 1) * 9_176`` in float32.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from cosmos_interactive._network import CosmosMoTNetwork
from cosmos_interactive._patchify import patchify_latent, unpatchify_latent
from cosmos_interactive._rope import text_mrope_ids, vae_mrope_ids
from cosmos_interactive._sampler import fixed_step_denoise
from torch import Tensor

from ..config import HistoryProfile, TransferModelConfig

SDE_FRAME_STRIDE = 1_000_003
"""Re-noise seed stride per latent."""

SDE_STEP_STRIDE = 9_176
"""Re-noise seed stride per sampler step (counted from one)."""

LATENT_HW = (30, 52)
"""Latent grid of an 832x480 frame (16x spatial compression)."""


def initial_noise(
    seed: int,
    latent: int,
    shape: tuple[int, ...],
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Starting noise of latent ``latent``: ``seed + latent`` drawn in ``shape`` and ``dtype``.

    The draw depends on the device's generator: CUDA and CPU give different
    numbers for the same seed, so reproduce a run on its own device type.
    """
    generator = torch.Generator(device=device).manual_seed(seed + latent)
    return torch.empty(shape, device=device, dtype=dtype).normal_(generator=generator)


def sde_noise(
    seed: int, latent: int, step_idx: int, numel: int, *, device: torch.device | str
) -> Tensor:
    """Re-noise of latent ``latent`` after sampler step ``step_idx``, float32, ``[numel]``."""
    generator = torch.Generator(device=device).manual_seed(
        seed + latent * SDE_FRAME_STRIDE + (step_idx + 1) * SDE_STEP_STRIDE
    )
    return torch.empty(numel, device=device, dtype=torch.float32).normal_(
        generator=generator
    )


def text_positions(num_text: int) -> Tensor:
    """``[3, num_text]`` prompt positions (all three axes equal, from zero)."""
    return text_mrope_ids(num_text, 0.0)[0]


def vision_positions(
    latent: int,
    *,
    num_text: int,
    grid_hw: tuple[int, int],
    fps: float,
    model: TransferModelConfig,
) -> Tensor:
    """``[3, gh * gw]`` positions of latent ``latent`` (shared by its control and RGB)."""
    ids, _ = vae_mrope_ids(
        1,
        grid_hw[0],
        grid_hw[1],
        float(num_text) + model.temporal_modality_margin,
        reset_spatial_indices=model.reset_spatial_ids,
        fps=fps if model.enable_fps_modulation else None,
        base_fps=model.base_fps,
        temporal_compression_factor=model.temporal_compression_factor,
        start_frame_offset=latent,
    )
    return ids


class TransferRollout:
    """Generate a clip a latent at a time from a stream of control latents.

    One instance serves many sessions: :meth:`start` resets the cache and
    reseeds without reallocating it.

    Args:
        network: The loaded Cosmos3 network.
        model: Settings of the export the network was loaded from.
        history: Window and sinks (see :class:`~cosmos3_nano_sim_transfer.config.HistoryProfile`).
        latent_hw: Latent grid per frame.
        fps: Frame rate of the clip, for temporal positions.
        native_gqa_attention: Let SDPA expand grouped-query heads itself.
        noise_dtype: Dtype of the initial noise draw (the checkpoint's).
    """

    def __init__(
        self,
        network: CosmosMoTNetwork,
        model: TransferModelConfig,
        *,
        history: HistoryProfile,
        latent_hw: tuple[int, int] = LATENT_HW,
        fps: float = 30.0,
        native_gqa_attention: bool = True,
        noise_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        patch = model.latent_patch_size
        if latent_hw[0] % patch or latent_hw[1] % patch:
            raise ValueError(
                f"latent grid {latent_hw} is not divisible by patch {patch}"
            )
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        self.net = network
        self.model = model
        self.history = history
        self.fps = float(fps)
        self.native_gqa_attention = bool(native_gqa_attention)
        self.noise_dtype = noise_dtype
        self.device = network.embed_tokens.weight.device
        self.dtype = network.config.dtype
        self.channels = network.config.spec.vfm.vae_latent_channels
        self.patch = patch
        self.latent_hw = (int(latent_hw[0]), int(latent_hw[1]))
        self.grid_hw = (self.latent_hw[0] // patch, self.latent_hw[1] // patch)
        self.tokens_per_latent = self.grid_hw[0] * self.grid_hw[1]
        self.layers = list(network.layers)
        # One block per latent holds its control, then its clean RGB.
        self.cache = network.init_stream_cache(
            chunk_grid=(1, *self.grid_hw),
            window_frames=history.window_frames - history.sink_frames,
            sink_frames=history.sink_frames,
            tokens_per_frame=2 * self.tokens_per_latent,
        )
        self.text_ids: Tensor | None = None
        self.seed = 0
        self.next_latent = 0
        self._finished = False

    @property
    def num_text(self) -> int:
        """Prompt length in tokens (0 before :meth:`start`)."""
        return 0 if self.text_ids is None else int(self.text_ids.numel())

    @property
    def cached_tokens(self) -> int:
        """Generation tokens held in the cache (bounded by the window)."""
        return int(self.cache.gen[0]._n_cached)

    @torch.no_grad()
    def start(self, text_ids: Tensor, *, seed: int) -> None:
        """Begin a session: reset the cache, reseed, and prefill a changed prompt.

        Args:
            text_ids: ``[N]`` prompt token ids (see
                :func:`~cosmos3_nano_sim_transfer.impl.prompt.prompt_token_ids`).
            seed: Noise seed of the session.

        Raises:
            ValueError: The prompt is empty.
        """
        ids = torch.as_tensor(text_ids, dtype=torch.long).reshape(-1).cpu()
        if ids.numel() == 0:
            raise ValueError("the prompt has no tokens")
        self.cache.reset()
        if self.text_ids is None or not torch.equal(ids, self.text_ids):
            self.net.prefill_text_stream(
                ids,
                text_positions(int(ids.numel())),
                self.cache,
                native_gqa_attention=self.native_gqa_attention,
            )
            self.text_ids = ids
        self.seed = int(seed)
        self.next_latent = 0
        self._finished = False

    def positions(self, latent: int) -> Tensor:
        """``[3, tokens_per_latent]`` positions of latent ``latent``."""
        return vision_positions(
            latent,
            num_text=self.num_text,
            grid_hw=self.grid_hw,
            fps=self.fps,
            model=self.model,
        )

    def initial_noise(self, latent: int) -> Tensor:
        """Starting noise of latent ``latent`` in this session, ``[C, 1, H, W]``."""
        return initial_noise(
            self.seed,
            latent,
            (self.channels, 1, *self.latent_hw),
            device=self.device,
            dtype=self.noise_dtype,
        )

    def sde_noise_fn(self, latent: int, numel: int) -> Callable[[int], Tensor]:
        """The sampler's re-noise hook for latent ``latent``."""

        def noise(step_idx: int) -> Tensor:
            return sde_noise(self.seed, latent, step_idx, numel, device=self.device)

        return noise

    def _tokens(self, latent: Tensor) -> Tensor:
        """``[C, 1, H, W]`` -> ``[tokens_per_latent, p * p * C]``."""
        if tuple(latent.shape) != (self.channels, 1, *self.latent_hw):
            raise ValueError(
                f"expected a [{self.channels}, 1, {self.latent_hw[0]}, "
                f"{self.latent_hw[1]}] latent, got {tuple(latent.shape)}"
            )
        return patchify_latent(latent, self.patch)

    def _write_clean(
        self, tokens: Tensor, cos: Tensor, sin: Tensor, start: int
    ) -> None:
        """Run clean tokens (no timestep) against ``cache[:start]`` and store their K/V at ``start``."""
        h = self.net.vae2llm(tokens.to(device=self.device, dtype=self.dtype))[None]
        end = start + self.tokens_per_latent
        for layer, text_k, text_v, gen in zip(
            self.layers,
            self.cache.text_k,
            self.cache.text_v,
            self.cache.gen,
            strict=True,
        ):
            history_k = gen._k[:, :, :start] if start else None
            history_v = gen._v[:, :, :start] if start else None
            h, k, v = layer.gen_compute_with_kv(
                h,
                cos,
                sin,
                text_k,
                text_v,
                history_k,
                history_v,
                None,
                native_gqa_attention=self.native_gqa_attention,
            )
            gen._k[:, :, start:end].copy_(k)
            gen._v[:, :, start:end].copy_(v)

    def _flow(
        self, latent: Tensor, timestep: Tensor, cos: Tensor, sin: Tensor, end: int
    ) -> Tensor:
        """Flow of a noisy ``[C, 1, H, W]`` latent against ``cache[:end]``; nothing is written.

        One scalar timestep embedding is expanded over the tokens (rather than
        embedding a per-token copy), as the parity-checked prototype did.
        """
        tokens = self._tokens(latent)
        h = self.net.vae2llm(tokens.to(device=self.device, dtype=self.dtype))
        h = h + self.net.embed_uniform_timestep(timestep, h.shape[0]).to(self.dtype)
        h = h[None]
        for layer, text_k, text_v, gen in zip(
            self.layers,
            self.cache.text_k,
            self.cache.text_v,
            self.cache.gen,
            strict=True,
        ):
            h = layer.gen_compute(
                h,
                cos,
                sin,
                text_k,
                text_v,
                gen._k[:, :, :end],
                gen._v[:, :, :end],
                None,
                native_gqa_attention=self.native_gqa_attention,
            )
        flow = self.net.llm2vae(self.net.norm_moe_gen(h))[0]
        return unpatchify_latent(
            flow,
            channels=self.channels,
            grid_t=1,
            grid_h=self.grid_hw[0],
            grid_w=self.grid_hw[1],
            patch_spatial=self.patch,
        )

    @torch.no_grad()
    def step(
        self,
        control_latent: Tensor,
        *,
        last: bool = False,
        noise: Tensor | None = None,
        replay: Tensor | None = None,
    ) -> Tensor:
        """Generate the next latent from its control latent.

        Args:
            control_latent: ``[C, 1, H, W]`` encoded control of this latent.
                Encode it only now, after the previous latent is committed.
            last: Skip the clean replay; the session ends with this latent.
            noise: Replace the seeded starting noise (``[C, 1, H, W]``).
            replay: Commit this ``[C, 1, H, W]`` clean latent to the cache
                instead of the generated one (teacher forcing: a parity probe
                replays a reference run's own latents to check each step on
                the reference's exact history).

        Returns:
            The clean latent, ``[C, 1, H, W]`` float32.

        Raises:
            RuntimeError: No session is open, or the last latent was generated.
        """
        if self.text_ids is None or self._finished:
            raise RuntimeError("call start() before step() (and again after last=True)")
        # Validate before touching the cache.
        control_tokens = self._tokens(control_latent)
        replay_tokens = None if replay is None else self._tokens(replay)
        k = self.next_latent
        cos, sin = self.net._mrope(self.positions(k))
        for gen in self.cache.gen:
            gen.before_update(k)
        control_start = int(self.cache.gen[0]._current_write_bounds()[0])
        rgb_start = control_start + self.tokens_per_latent

        self._write_clean(control_tokens, cos, sin, control_start)

        shape = (self.channels, 1, *self.latent_hw)
        start = self.initial_noise(k) if noise is None else noise.to(self.device)
        flat = start.reshape(-1).float()
        denoised = fixed_step_denoise(
            lambda x, t: self._flow(x.reshape(shape), t, cos, sin, rgb_start).reshape(
                -1
            ),
            flat,
            t_list=list(self.model.t_list),
            sample_type=self.model.sample_type,
            num_train_timesteps=self.model.num_train_timesteps,
            sde_noise_fn=self.sde_noise_fn(k, flat.numel()),
            fp32_math=True,
        ).reshape(shape)

        if last:
            self._finished = True
        else:
            clean = self._tokens(denoised) if replay_tokens is None else replay_tokens
            self._write_clean(clean, cos, sin, rgb_start)
        for gen in self.cache.gen:
            gen.after_update(k)
        self.next_latent += 1
        return denoised

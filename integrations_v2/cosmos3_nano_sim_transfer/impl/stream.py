# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming Transfer in-process: control frames in, RGB frames out, a latent at a time.

:class:`TransferStream` is what a live host drives: :meth:`~TransferStream.start`
a session with a prompt and a seed, then :meth:`~TransferStream.step` it with
the control frames of each latent (uint8 RGB, one frame first, then four) and
get as many generated frames back. Control frames of any size are resized to
832x480 the way cosmos-framework resizes that kind of control. It holds the
network, the control encoder, the decoder and the bounded stream cache, so a
host keeps one instance for as long as it runs and starts a session per clip.

The calls take the shapes a host already uses for a control-to-RGB model that
generates one latent per step: ``start(prompt=, seed=)`` returning a string,
``step(frames) -> frames`` on ``[T, H, W, 3]`` uint8 arrays, ``last_timing``
(milliseconds per stage) and ``close()``.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from cosmos_interactive._network import CosmosMoTNetwork
from torch import Tensor

from ..config import HistoryProfile, TransferModelConfig, fetch_release
from .control import ControlEncoder, resize_control, to_model_range
from .decode import LatentDecoder, to_uint8
from .fast import ACCELERATIONS, make_rollout
from .prompt import HEIGHT, WIDTH, prompt_token_ids
from .rollout import LATENT_HW, TransferRollout

NOMINAL_SECONDS = 20
"""Clip length a live session's prompt states when the caller names none."""


def nominal_clip_frames(fps: float, seconds: float = NOMINAL_SECONDS) -> int:
    """Frames of a ``seconds`` clip at ``fps`` as a request counts them: 601 at 30 fps.

    The duration sentence of the prompt is ``frames / fps`` seconds, so an
    open-ended session states a fixed nominal length instead.
    """
    if fps <= 0 or seconds <= 0:
        raise ValueError(f"need fps > 0 and seconds > 0, got {fps}, {seconds}")
    return int(round(seconds * fps)) + 1


class LatentEncoder(Protocol):
    """Encodes the control frames of one latent (see :class:`ControlEncoder`)."""

    def reset(self) -> None: ...

    def encode(self, frames: Tensor) -> Tensor: ...


class FrameDecoder(Protocol):
    """Decodes one clean latent to pixel frames (see :class:`LatentDecoder`)."""

    def reset(self) -> None: ...

    def decode(self, latent: Tensor) -> Tensor: ...


class TransferStream:
    """Cosmos3-Nano-Sim-Transfer as a streaming model, one latent per step.

    Build it with :meth:`load`; the constructor takes the parts already built
    (and lets tests pass small ones).

    Args:
        network: The loaded Cosmos3 network.
        model: Settings of the export the network was loaded from.
        encoder: Control encoder (the Wan 2.2 streaming encoder).
        decoder: Latent decoder (Wan 2.2 or TAEHV-super).
        tokenizer_path: The release's ``text_tokenizer`` folder.
        hint: Kind of control video, one the checkpoint was trained on
            (``model.hints``: ``seg``, ``depth``, ``edge`` or ``blur`` for the
            release). It selects how frames of another size are resized.
        emphasize_control_in_prompt: Append Framework's control-adherence
            sentence. Match the reference request; off for the parity baseline.
        prompt_aspect_ratio: Optional aspect-ratio label injected into JSON
            prompts; match the reference request. This does not resize controls.
        history: Default history of a session: a preset name, ``W/S`` or a
            :class:`HistoryProfile`.
        fps: Frame rate of the control frames, for positions and the prompt.
        nominal_frames: Clip length the prompt states when :meth:`start` names
            none; :func:`nominal_clip_frames` (20 s) by default.
        latent_hw: Latent grid per frame (the encoder's output size).
        native_gqa_attention: Let SDPA expand grouped-query heads itself.
        noise_dtype: Dtype of the initial noise draw (the checkpoint's).
        accel: How the rollout runs, one of
            :data:`~cosmos3_nano_sim_transfer.impl.fast.ACCELERATIONS`: ``eager``,
            ``graphs`` (fixed K/V buffers and CUDA graphs) or ``compile``
            (graphs over compiled layers). The fast modes need CUDA and a
            prompt of at most ``text_cache_max_len`` tokens; run
            :meth:`warm_up` once before the first session so the graphs are
            captured then.
        t_list: Sampler schedule replacing the export's (see
            :meth:`TransferModelConfig.with_schedule`); each sigma is one
            denoise pass per latent, besides the control pass and the clean
            replay. ``None``: the export's four steps.
    """

    width: int = WIDTH
    """Width of the generated frames."""

    height: int = HEIGHT
    """Height of the generated frames."""

    def __init__(
        self,
        network: CosmosMoTNetwork,
        model: TransferModelConfig,
        *,
        encoder: LatentEncoder,
        decoder: FrameDecoder,
        tokenizer_path: str | Path,
        hint: str = "seg",
        emphasize_control_in_prompt: bool = False,
        prompt_aspect_ratio: str | None = None,
        history: str | HistoryProfile = "long-rollout",
        fps: float = 30.0,
        nominal_frames: int | None = None,
        latent_hw: tuple[int, int] = LATENT_HW,
        native_gqa_attention: bool = True,
        noise_dtype: torch.dtype = torch.bfloat16,
        accel: str = "eager",
        t_list: Sequence[float] | None = None,
    ) -> None:
        if accel not in ACCELERATIONS:
            raise ValueError(
                f"unknown acceleration {accel!r}; use one of {ACCELERATIONS}"
            )
        model = model.with_schedule(t_list=t_list)
        if hint not in model.hints:
            raise ValueError(
                f"control {hint!r} is not one the checkpoint was trained on; "
                f"use one of {model.hints}"
            )
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        self.network: CosmosMoTNetwork | None = network
        self.model = model
        self.encoder: LatentEncoder | None = encoder
        self.decoder: FrameDecoder | None = decoder
        self.tokenizer_path = Path(tokenizer_path)
        self.hint = hint
        self.emphasize_control_in_prompt = emphasize_control_in_prompt
        self.prompt_aspect_ratio = prompt_aspect_ratio
        self.default_history = model.history(history)
        self.fps = float(fps)
        self.nominal_frames = (
            nominal_clip_frames(self.fps)
            if nominal_frames is None
            else int(nominal_frames)
        )
        self.latent_hw = latent_hw
        self.native_gqa_attention = native_gqa_attention
        self.noise_dtype = noise_dtype
        self.device = network.embed_tokens.weight.device
        if accel != "eager" and self.device.type != "cuda":
            raise ValueError(f"accel {accel!r} needs a CUDA device, not {self.device}")
        self.accel = accel
        self.temporal_compression = model.temporal_compression_factor
        self.rollout: TransferRollout | None = None
        self.prompt: str | None = None
        self.seed = 0
        self.prompt_frames = self.nominal_frames
        self.latents = 0
        """Latents generated in the current session."""
        self._ended = False
        self.load_s: dict[str, float] = {}
        """Seconds :meth:`load` spent per part."""
        self._timing: dict[str, float] = {}

    @classmethod
    def load(
        cls,
        release: str | Path | None = None,
        *,
        hint: str = "seg",
        emphasize_control_in_prompt: bool = False,
        prompt_aspect_ratio: str | None = None,
        history: str | HistoryProfile = "long-rollout",
        fps: float = 30.0,
        nominal_frames: int | None = None,
        decoder: str = "wan2.2",
        vae: str | Path | None = None,
        taehv: str | Path | None = None,
        device: torch.device | str = "cuda",
        accel: str = "eager",
        steps: int | None = None,
        t_list: Sequence[float] | None = None,
    ) -> TransferStream:
        """Load the release, the Wan 2.2 encoder and a decoder onto ``device``.

        Args:
            release: Valid local export folder; the pinned release is downloaded
                (see :func:`~cosmos3_nano_sim_transfer.config.fetch_release`)
                only when ``None``.
            hint: See :class:`TransferStream`.
            emphasize_control_in_prompt: See :class:`TransferStream`.
            prompt_aspect_ratio: See :class:`TransferStream`.
            history: See :class:`TransferStream`.
            fps: See :class:`TransferStream`.
            nominal_frames: See :class:`TransferStream`.
            decoder: ``wan2.2`` (the decoder cosmos-framework uses) or
                ``taehv-super`` (much faster, an approximation of it).
            vae: Wan 2.2 VAE weights (``Wan2.2_VAE.pth``) for the encoder and
                the ``wan2.2`` decoder; the FlashDreams default when ``None``.
            taehv: TAEHV-super weights; the FlashDreams default when ``None``.
            device: Where to run.
            accel: See :class:`TransferStream`. The control encoder runs
                eagerly in every mode: a CUDA graph of its four-frame calls
                saved about 2 ms a latent and is captured again in every
                session.
            steps: Sampler steps per latent: the export's schedule with
                intermediate sigmas dropped (2: ``1, 0.625``; see
                :func:`~cosmos3_nano_sim_transfer.config.skip_schedule`).
            t_list: A sampler schedule of its own; give ``steps`` or
                ``t_list``, not both. Both are checked before the weights load.
        """
        from cosmos_interactive.checkpoint_load import load_exported_checkpoint

        if accel not in ACCELERATIONS:
            raise ValueError(
                f"unknown acceleration {accel!r}; use one of {ACCELERATIONS}"
            )
        timings: dict[str, float] = {}
        started = time.perf_counter()
        root = fetch_release(release)
        model = TransferModelConfig.from_export(root).with_schedule(
            steps=steps, t_list=t_list
        )
        timings["release"] = time.perf_counter() - started
        started = time.perf_counter()
        network = load_exported_checkpoint(root, device=str(device)).eval()
        timings["network"] = time.perf_counter() - started
        started = time.perf_counter()
        encoder = ControlEncoder(
            checkpoint_path=None if vae is None else str(vae), device=device
        )
        weights = vae if decoder == "wan2.2" else taehv
        frame_decoder = LatentDecoder(
            decoder,
            checkpoint_path=None if weights is None else str(weights),
            device=device,
        )
        timings["encoder_decoder"] = time.perf_counter() - started
        stream = cls(
            network,
            model,
            encoder=encoder,
            decoder=frame_decoder,
            tokenizer_path=root / "text_tokenizer",
            hint=hint,
            emphasize_control_in_prompt=emphasize_control_in_prompt,
            prompt_aspect_ratio=prompt_aspect_ratio,
            history=history,
            fps=fps,
            nominal_frames=nominal_frames,
            accel=accel,
        )
        stream.load_s = timings
        return stream

    @property
    def t_list(self) -> tuple[float, ...]:
        """Sampler schedule of every session (the export's unless replaced)."""
        return self.model.t_list

    @property
    def history(self) -> HistoryProfile:
        """History of the current session (the default before the first)."""
        return self.default_history if self.rollout is None else self.rollout.history

    @property
    def last_timing(self) -> dict[str, float]:
        """Milliseconds of the last step: ``encode_ms`` (resize and encode the
        control), ``model_ms`` (control pass, denoising and committing the clean
        latent to the cache), ``decode_ms``, ``to_host_ms`` (:meth:`step` only)
        and ``step_ms`` (all of it)."""
        return dict(self._timing)

    def _parts(self) -> tuple[CosmosMoTNetwork, LatentEncoder, FrameDecoder]:
        if self.network is None or self.encoder is None or self.decoder is None:
            raise RuntimeError("the stream is closed")
        return self.network, self.encoder, self.decoder

    def _sync(self) -> float:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    @torch.no_grad()
    def start(
        self,
        prompt: str,
        seed: int,
        *,
        history: str | HistoryProfile | None = None,
        num_frames: int | None = None,
    ) -> str:
        """Begin a session: empty history, this prompt and noise seed.

        Args:
            prompt: Plain text or JSON scene description; metadata is added
                using the corresponding Framework format.
            seed: Noise seed of the session.
            history: History of this session; the stream's default when
                ``None``. Another window reallocates the cache.
            num_frames: Clip length the prompt states; the stream's
                ``nominal_frames`` when ``None``. A batch run passes its clip
                length, as a cosmos-framework request of that clip does.

        Returns:
            ``new`` when the prompt was run into a new or changed text cache,
            ``reset`` when the previous session's prompt cache was reused.
            Both draw the same frames.

        Raises:
            ValueError: Invalid frame count, frame rate or control hint.
        """
        network, encoder, decoder = self._parts()
        profile = (
            self.default_history if history is None else self.model.history(history)
        )
        frames = self.nominal_frames if num_frames is None else int(num_frames)
        text_ids = prompt_token_ids(
            prompt,
            num_frames=frames,
            fps=self.fps,
            tokenizer_path=self.tokenizer_path,
            hint=self.hint,
            emphasize_control_in_prompt=self.emphasize_control_in_prompt,
            aspect_ratio=self.prompt_aspect_ratio,
        )
        rollout = self.rollout
        if rollout is None or rollout.history != profile:
            # Drop the old cache before allocating the next one.
            self.rollout = rollout = None
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            rollout = make_rollout(
                self.accel,
                network,
                self.model,
                history=profile,
                latent_hw=self.latent_hw,
                fps=self.fps,
                native_gqa_attention=self.native_gqa_attention,
                noise_dtype=self.noise_dtype,
            )
            self.rollout = rollout
        reused = rollout.text_ids is not None and torch.equal(
            rollout.text_ids, torch.as_tensor(text_ids).reshape(-1).cpu()
        )
        rollout.start(text_ids, seed=seed)
        encoder.reset()
        decoder.reset()
        self.prompt, self.seed, self.prompt_frames = prompt, int(seed), frames
        self.latents = 0
        self._ended = False
        self._timing = {}
        return "reset" if reused else "new"

    @torch.no_grad()
    def warm_up(
        self,
        prompt: str,
        seed: int = 0,
        *,
        history: str | HistoryProfile | None = None,
        latents: int = 0,
    ) -> dict[str, float]:
        """Run throwaway sessions on black control frames so the real sessions
        do not pay for kernel loading, graph capture or compile.

        The eager rollout runs one session of ``max(latents, 3)`` latents; the
        fast rollouts run as many sessions and latents as capture every graph
        (:attr:`~cosmos3_nano_sim_transfer.impl.fast.FastTransferRollout.warmup_sessions`
        and ``warmup_latents``: two sessions of a full window plus one latent at
        the defaults), and at least ``latents`` latents. Use the prompt of the
        sessions to come: a prompt of another length moves the slices the
        graphs read, and they are captured again. :meth:`start` must be called
        afterwards.

        Returns:
            ``sessions``, ``latents`` (per session), ``seconds`` and ``graphs``.
        """
        started = time.perf_counter()
        self.start(prompt, seed, history=history)
        rollout = self.rollout
        assert rollout is not None
        count = max(int(latents), int(getattr(rollout, "warmup_latents", 3)))
        sessions = int(getattr(rollout, "warmup_sessions", 1))
        for session in range(sessions):
            if session:
                self.start(prompt, seed, history=history)
            for k in range(count):
                frames = torch.zeros(
                    (self.frames_for(k), self.height, self.width, 3),
                    dtype=torch.uint8,
                    device=self.device,
                )
                self.step_tensor(frames)
        self._ended = True
        self._sync()
        return {
            "sessions": float(sessions),
            "latents": float(count),
            "seconds": time.perf_counter() - started,
            "graphs": float(self.graphs_captured),
        }

    @property
    def graphs_captured(self) -> int:
        """Pass kinds of the rollout that replay a captured CUDA graph."""
        return int(getattr(self.rollout, "graphs_captured", 0))

    def reset(self) -> str:
        """Start the current session over: same prompt, seed and history."""
        if self.prompt is None:
            raise RuntimeError("call start() before reset()")
        return self.start(
            self.prompt, self.seed, history=self.history, num_frames=self.prompt_frames
        )

    def frames_for(self, latent: int) -> int:
        """Control frames latent ``latent`` takes: one first, then four."""
        return 1 if latent == 0 else self.temporal_compression

    def _control(self, control: np.ndarray | Tensor) -> Tensor:
        """Validate ``[T, H, W, 3]`` uint8 frames; return them as a tensor."""
        frames = (
            torch.from_numpy(np.ascontiguousarray(control))
            if isinstance(control, np.ndarray)
            else control
        )
        expected = self.frames_for(self.latents)
        if (
            frames.ndim != 4
            or frames.shape[0] != expected
            or frames.shape[3] != 3
            or frames.dtype != torch.uint8
        ):
            raise ValueError(
                f"latent {self.latents} takes [{expected}, H, W, 3] uint8 control "
                f"frames, got {tuple(frames.shape)} {frames.dtype}"
            )
        return frames

    @torch.no_grad()
    def step_tensor(
        self, control: np.ndarray | Tensor, *, last: bool = False
    ) -> Tensor:
        """Generate the frames of the next latent, left on the device.

        Args:
            control: ``[T, H, W, 3]`` uint8 control frames: one for the first
                latent of a session, four after it.
            last: Skip committing this latent to the cache; the session ends
                with it (a batch run knows its last latent).

        Returns:
            ``[T, 3, 480, 832]`` uint8 frames on the stream's device.

        Raises:
            RuntimeError: No session is open, or the stream is closed.
            ValueError: The control frames do not fit this latent.
        """
        _, encoder, decoder = self._parts()
        if self.rollout is None or self.prompt is None:
            raise RuntimeError("call start() before step()")
        if self._ended:
            raise RuntimeError("the session ended with its last latent; call start()")
        frames = self._control(control)
        t0 = self._sync()
        pixels = frames.to(self.device).permute(0, 3, 1, 2).contiguous()
        latent = encoder.encode(to_model_range(resize_control(pixels, self.hint)))
        t1 = self._sync()
        clean = self.rollout.step(latent, last=last)
        t2 = self._sync()
        out = to_uint8(decoder.decode(clean))
        t3 = self._sync()
        self.latents += 1
        self._ended = last
        self._timing = {
            "encode_ms": 1000 * (t1 - t0),
            "model_ms": 1000 * (t2 - t1),
            "decode_ms": 1000 * (t3 - t2),
            "step_ms": 1000 * (t3 - t0),
        }
        return out

    def step(self, control: np.ndarray | Tensor, *, last: bool = False) -> np.ndarray:
        """Generate the frames of the next latent as ``[T, 480, 832, 3]`` uint8 on the host.

        Args:
            control: ``[T, H, W, 3]`` uint8 control frames (numpy or torch, on
                any device): one for the first latent of a session, four after.
            last: See :meth:`step_tensor`.
        """
        started = time.perf_counter()
        out = self.step_tensor(control, last=last)
        copied = time.perf_counter()
        frames = out.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        done = time.perf_counter()
        self._timing["to_host_ms"] = 1000 * (done - copied)
        self._timing["step_ms"] = 1000 * (done - started)
        return frames

    @property
    def cached_tokens(self) -> int:
        """Generation tokens held in the cache (bounded by the window)."""
        return 0 if self.rollout is None else self.rollout.cached_tokens

    def close(self) -> None:
        """Release the network, the cache, the encoder and the decoder."""
        self.rollout = None
        self.network = None
        self.encoder = None
        self.decoder = None
        self.prompt = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


__all__ = ["NOMINAL_SECONDS", "TransferStream", "nominal_clip_frames"]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Latents back to pixels, eleven cameras at a time.

The cameras go on the batch axis so each gets its own decoder
history. The decoder also drops 3 warm-up frames on its first call, so a clip
comes back the length it went in; :func:`decoded_pixel_frames` is where that
arithmetic lives.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from torch import Tensor

from cosmos3_nano_sim_auto.impl.distributed.parallel import gather_views, owned_views
from flashdreams.core.distributed.parallel import ParallelContext

TIME_UPSCALE = 4
"""Pixel frames per latent frame: ``2 ** len(decoder_time_upscale)`` for TAEHV-super."""

FRAMES_TO_TRIM = 3
"""Warm-up frames the first decode drops: ``2 ** sum(decoder_time_upscale) - 1``."""

DecodeStep = Callable[[Tensor], Tensor]
"""``[V, t, C, H, W]`` normalized latent -> ``[V, t*4, 3, H*16, W*16]`` pixels in ``[-1, 1]``."""


def decoded_pixel_frames(
    latent_frames: int,
    *,
    time_upscale: int = TIME_UPSCALE,
    frames_to_trim: int = FRAMES_TO_TRIM,
) -> int:
    """Pixel frames a decoder produces from ``latent_frames``, warm-up included.

    Only the first chunk of a stream pays the trim, so this is the count over a
    whole decoder lifetime rather than per call.
    """
    if latent_frames < 1:
        raise ValueError(f"latent_frames must be >= 1, got {latent_frames}.")
    frames = latent_frames * time_upscale - frames_to_trim
    if frames < 1:
        raise ValueError(
            f"{latent_frames} latent frames decode to nothing once the decoder's "
            f"{frames_to_trim} warm-up frames are trimmed."
        )
    return frames


def multiview_chunk_decoder(
    decode_step: DecodeStep,
    *,
    len_t: int = 1,
    num_views: int | None = None,
    time_upscale: int = TIME_UPSCALE,
    frames_to_trim: int = FRAMES_TO_TRIM,
) -> Callable[[Tensor], Tensor]:
    """Wrap a batched decode step into a streaming multiview decoder.

    The returned callable takes ``[V, C, T, H, W]`` and gives
    ``[V, T_pix, 3, H*16, W*16]`` in ``[-1, 1]``. Call it once per rollout chunk
    to stream, or once with the whole clip; the temporal cache lives in
    ``decode_step``, so the results are the same either way and a new decoder is
    needed on reset.

    Args:
        decode_step: A batched, cache-carrying decode. From
            :func:`make_multiview_chunk_decoder`, or a stand-in.
        len_t: Latent frames per call into ``decode_step``. Every call is this
            length, whatever the rollout's chunking was.
        num_views: Cameras this decoder is for. The cache takes its batch from
            the first call either way, so this only moves a mismatch complaint
            from the second call to the first.

    Raises:
        ValueError: the latent is the wrong rank, the view count is not the one
            the decoder is pinned to, ``T`` is not a multiple of ``len_t``, or
            the frames coming back disagree with :func:`decoded_pixel_frames`.
    """
    if len_t < 1:
        raise ValueError(f"len_t must be >= 1, got {len_t}.")
    if num_views is not None and num_views < 1:
        raise ValueError(f"num_views must be >= 1, got {num_views}.")
    state: dict[str, int] = {
        "views": num_views or 0,
        "latent_frames": 0,
        "pixel_frames": 0,
    }

    def decode(latent: Tensor) -> Tensor:
        if latent.ndim != 5:
            raise ValueError(
                f"expected a [V, C, T, H, W] latent, got {tuple(latent.shape)}."
            )
        views, _channels, frames, _height, _width = latent.shape
        if frames % len_t:
            raise ValueError(
                f"{frames} latent frames is not a multiple of len_t={len_t}; the decoder "
                "cache is sized from the first call and every call has to match it."
            )
        if state["views"] and views != state["views"]:
            raise ValueError(
                f"this decoder is pinned to {state['views']} views and cannot switch to "
                f"{views}. The cache holds one row per view; build a new decoder."
            )
        state["views"] = views

        pixels = [
            decode_step(latent[:, :, start : start + len_t].permute(0, 2, 1, 3, 4))
            for start in range(0, frames, len_t)
        ]
        decoded = torch.cat(pixels, dim=1)

        state["latent_frames"] += frames
        state["pixel_frames"] += int(decoded.shape[1])
        expected = decoded_pixel_frames(
            state["latent_frames"],
            time_upscale=time_upscale,
            frames_to_trim=frames_to_trim,
        )
        if state["pixel_frames"] != expected:
            raise ValueError(
                f"{state['latent_frames']} latent frames should have decoded to {expected} "
                f"pixel frames, but the decoder returned {state['pixel_frames']}. Either the "
                "warm-up trim is not the 3 frames TAEHV-super takes, or a call was dropped."
            )
        return decoded

    return decode


def make_multiview_chunk_decoder(
    *,
    num_views: int | None = None,
    len_t: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    checkpoint_path: str | None = None,
    denormalize: bool = False,
    device: str | torch.device = "cuda",
    compile: bool = False,
) -> Callable[[Tensor], Tensor]:
    """Build the real TAEHV-super decoder, batched across cameras.

    Args:
        num_views: cameras to expect, so a wrong-width latent is caught on the
            first call. See :func:`multiview_chunk_decoder`.
        denormalize: ``False``, because the rollout emits latents in the
            mean/std-normalized space this decoder reads directly. The stock
            factory infers the same value from the checkpoint's name, but the
            rollout's correctness should not rest on a filename heuristic.
        compile: run the decoder under ``torch.compile``.
    """
    # Imported here rather than at module scope: this reaches the whole VAE
    # stack, and the arithmetic above is worth having without it.
    from cosmos_interactive._taehv_super import SuperTAEHV
    from cosmos_interactive._vae import TAEW2_2_SUPER_CHECKPOINT

    from flashdreams.recipes.wan.autoencoder.vae import (
        _WAN22_TI2V_5B_LATENT_MEAN,
        _WAN22_TI2V_5B_LATENT_STD,
    )

    taehv = SuperTAEHV(
        checkpoint_path=checkpoint_path or TAEW2_2_SUPER_CHECKPOINT,
        use_cuda_graph=False,
        use_compile=compile,
    )
    taehv = taehv.to(device=device, dtype=dtype).eval()
    resident = next(taehv.parameters()).device
    cache = taehv.prepare_cache()
    mean = torch.tensor(_WAN22_TI2V_5B_LATENT_MEAN, dtype=dtype, device=resident).view(
        1, 1, -1, 1, 1
    )
    std = torch.tensor(_WAN22_TI2V_5B_LATENT_STD, dtype=dtype, device=resident).view(
        1, 1, -1, 1, 1
    )

    def decode_step(z: Tensor) -> Tensor:
        z = z.to(device=resident, dtype=dtype)
        if denormalize:
            z = z * std + mean
        with torch.no_grad():
            # TAEHV emits [0, 1]; the rest of the package speaks [-1, 1].
            return taehv.decode(z, cache=cache).mul(2).sub(1).float()

    return multiview_chunk_decoder(decode_step, len_t=len_t, num_views=num_views)


def decoded_views(
    latent: Tensor,
    decode: Callable[[Tensor], Tensor] | None,
    *,
    parallel: ParallelContext | None,
) -> Tensor:
    """``[V, C, t, H, W]`` decoded to ``[V, T, 3, 16H, 16W]``, across the mesh if asked.

    Without ``parallel`` this is a plain ``decode(latent)``. With it, each rank
    decodes the cameras :func:`owned_views` gives it (the same round-robin as
    the encode, so the decoder's per-camera temporal cache always sees the same
    camera) and :func:`gather_views` hands rank 0 every camera in rig order.
    Other ranks get their own slice back, which the caller discards.
    """
    if parallel is None or parallel.world_size == 1:
        assert decode is not None
        return decode(latent)
    num_views = int(latent.shape[0])
    owned = owned_views(parallel, num_views)
    if decode is None:
        # Fewer cameras than ranks: this rank owns none and takes no part.
        assert len(owned) == 0
        return latent.new_empty(0)
    local = decode(latent[list(owned)])
    views = gather_views(list(local), num_views=num_views, parallel=parallel)
    return local if views is None else torch.stack(views)


def write_view_videos(
    pixels: Tensor,
    directory: str | Path,
    *,
    cameras: Sequence[str],
    fps: float = 30.0,
) -> list[Path]:
    """Write one mp4 per camera, named for it, and return the paths in rig order.

    Named by camera key rather than by index, for the reason
    :attr:`~cosmos3_nano_sim_auto.inputs.MultiviewSample.physical_camera_indexes`
    exists: a file called ``view_03.mp4`` invites being reloaded against the
    wrong camera, and nothing about the picture would give that away.
    """
    from flashdreams.infra.runner_io import write_video_tensor

    if pixels.ndim != 5:
        raise ValueError(f"expected [V, T, 3, H, W] pixels, got {tuple(pixels.shape)}.")
    if len(cameras) != pixels.shape[0]:
        raise ValueError(f"{pixels.shape[0]} views but {len(cameras)} camera names.")

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    return [
        write_video_tensor(view, destination / f"{camera}.mp4", fps=fps, layout="tchw")
        for camera, view in zip(cameras, pixels, strict=True)
    ]

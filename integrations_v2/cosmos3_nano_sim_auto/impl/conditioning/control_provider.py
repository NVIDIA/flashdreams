# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Supply HD-map conditioning tokens for requested latent-frame ranges.

Providers preserve view-major token order and the causal video encoder's
history. :class:`EncodedControls` encodes a local clip once up front;
:class:`RenderedControls` renders live poses and retains encoder state between
calls. Both return ``[V*t*S, patch_latent_dim]`` tokens.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch
from torch import Tensor

from cosmos3_nano_sim_auto.impl.packing import ClipGeometry
from cosmos3_nano_sim_auto.inputs import (
    WAN_TEMPORAL_COMPRESSION,
    _encoded_views,
    control_tokens,
)

if TYPE_CHECKING:
    from flashdreams.core.distributed.parallel import ParallelContext

__all__ = [
    "ControlProvider",
    "EncodedControls",
    "RenderedControls",
    "RigRenderFn",
    "pixel_span_for_latents",
    "rendered_clips",
]

RigRenderFn = Callable[..., Tensor]
"""Render every camera in the rig over pixel frames ``[start, end)``.

Called as ``render(start, end)``, or ``render(start, end, views=[...])`` for
just those cameras in that order when the provider splits cameras across ranks.

``[V*T, H, W, C]`` uint8, view-outer and frame-inner: camera 0's frames, then
camera 1's. That is what ludus's ``LudusCudaTimestampedContext.render`` returns
when the query tensors are built in that order, and it is the order
:func:`~cosmos3_nano_sim_auto.inputs.control_tokens` wants. ``C`` may be 4; the alpha
channel is dropped.

Eleven cameras over one chunk is 176 images, well inside ludus's batch limit of
2048, so a chunk is one call and the frames never leave the GPU. Whoever
supplies this closes over the ego poses -- the provider deliberately knows
nothing about driving.
"""


@runtime_checkable
class ControlProvider(Protocol):
    """Control patches for a range of latent frames, on demand."""

    def tokens(self, start: int, end: int) -> Tensor:
        """``[V*(end-start)*S, patch_latent_dim]`` for latent frames ``[start, end)``.

        Called for the prefill's opening window, then once per chunk if the
        cache is too small to hold the rest. Always for a range that carries on
        from the last, so a provider may rely on being asked in order;
        :class:`RenderedControls` refuses anything else.

        Must be on the network's device, since the rollout embeds what it is
        handed without moving it.

        Raises:
            ValueError: the range is empty, reaches past what the provider can
                supply, or does not follow on from the previous one.
        """
        ...


class EncodedControls:
    """A control track encoded up front, handed out a range at a time.

    Local videos are encoded once during prefill, preserving encoder history.
    Requests slice that encoded track, so memory grows with clip length.
    """

    def __init__(self, tokens: Tensor, geometry: ClipGeometry) -> None:
        """
        Args:
            tokens: ``[V*T*S, patch_latent_dim]`` for the whole clip, from
                :func:`~cosmos3_nano_sim_auto.inputs.control_tokens`.
            geometry: what the shape is checked against, and what says how to
                find a frame range inside it.

        Raises:
            ValueError: the token count disagrees with the geometry.
        """
        if geometry.frames_per_view is None:
            raise ValueError("Encoded controls require a finite clip")
        expected = (
            geometry.num_views * geometry.frames_per_view * geometry.spatial_tokens
        )
        if tokens.ndim != 2 or tokens.shape[0] != expected:
            raise ValueError(
                f"expected [{expected}, width] control tokens for "
                f"{geometry.num_views} view(s) x {geometry.frames_per_view} frames "
                f"x {geometry.spatial_tokens} spatial tokens, got {tuple(tokens.shape)}."
            )
        self._tokens = tokens
        self._geometry = geometry
        self._frames = geometry.frames_per_view

    def tokens(self, start: int, end: int) -> Tensor:
        """The patches for frames ``[start, end)``, still view-outer.

        A frame range is a stride within each view's block rather than a slice
        off the front, since the whole track for view 0 comes before view 1.
        """
        geometry = self._geometry
        if not 0 <= start < end <= self._frames:
            raise ValueError(
                f"Control frames [{start}, {end}) run outside the clip's "
                f"{geometry.frames_per_view}."
            )
        if (start, end) == (0, geometry.frames_per_view):
            return self._tokens
        per_view = self._frames, geometry.spatial_tokens
        return (
            self._tokens.reshape(geometry.num_views, *per_view, -1)[:, start:end]
            .reshape(-1, self._tokens.shape[-1])
            .contiguous()
        )


def pixel_span_for_latents(start: int, end: int) -> tuple[int, int]:
    """The pixel frames latent frames ``[start, end)`` are encoded from.

    Latent frame 0 is pixel frame 0 alone; every later latent frame covers the
    four pixel frames ending at ``4 * frame``. Consecutive ranges therefore tile
    the pixel timeline exactly, with no overlap and nothing skipped.

    The counts come out as the encoder demands them: a span from 0 is ``1 + 4n``
    pixel frames, and every later span is ``4n``, which is what
    ``WanVAEEncoder.get_output_temporal_size`` asserts for the opening call and
    for the rest. So these spans keep a streaming encoder fed with legal shapes.

    Raises:
        ValueError: the range is empty or negative.
    """
    if start < 0 or start >= end:
        raise ValueError(f"[{start}, {end}) is not a frame range.")
    first = 0 if start == 0 else WAN_TEMPORAL_COMPRESSION * start - 3
    return first, WAN_TEMPORAL_COMPRESSION * (end - 1) + 1


class RenderedControls:
    """Control rendered and encoded a chunk at a time, for a run of no fixed length.

    ``render`` produces the rig's pixel frames for a span, this encodes and
    patchifies them, and the rollout sees the same patches a file would have
    given it. Nothing holds a whole track, so the length of the run costs no
    memory, and the frames can come from a live renderer.

    The result matches encoding the whole clip at once, because each camera gets
    a stream from :func:`~cosmos3_nano_sim_auto.inputs.make_control_encode_streams`
    that holds a ``WanVAECache``. A span is encoded against the footage before
    it rather than an empty state, and the spans from
    :func:`pixel_span_for_latents` tile the pixel timeline without a gap.

    That state makes this order-dependent, where :class:`EncodedControls` is
    not. Ranges must start at 0 and each carry on from the last, which
    :meth:`tokens` enforces, and a stream cannot be rewound: a new run needs
    new streams.
    """

    def __init__(
        self,
        render: RigRenderFn | None,
        encode_streams: Sequence[Callable[[Tensor], Tensor]],
        *,
        geometry: ClipGeometry,
        latent_patch_size: int,
        parallel: ParallelContext | None = None,
        worker: Any | None = None,
    ) -> None:
        """
        Args:
            render: see :data:`RigRenderFn`.
            encode_streams: One stateful ``clip[T, 3, H, W] -> latent[C, T_lat,
                H_lat, W_lat]`` per camera, in the rig's order, from
                :func:`~cosmos3_nano_sim_auto.inputs.make_control_encode_streams`. Not
                :func:`~cosmos3_nano_sim_auto.inputs.make_control_encode_fn`, which
                starts from an empty state on every call.
            geometry: The rig's shape, giving the number of cameras to expect
                back and the patch grid.
            latent_patch_size: The checkpoint's ``spec.vfm.latent_patch_size``.
            parallel: With a mesh, camera ``v`` is encoded only on rank
                ``v % world_size`` and broadcast, so a chunk's encode costs one
                camera per rank instead of every camera on every rank. Each
                camera keeps one owner for the whole run, which is what its
                stream's temporal state needs.
            worker: A :class:`~cosmos3_nano_sim_auto.impl.conditioning.control_worker.ControlWorker`
                that draws and encodes this rank's cameras in its own process.
                With one, ``render`` and ``encode_streams`` are never called:
                :meth:`prefetch` asks the worker for a range ahead of the
                :meth:`tokens` call that wants it, so the control for chunk n+1
                is made while the model denoises chunk n. The map for a chunk
                depends on the clip and the car, never on the model's pixels,
                so drawing early changes nothing but when.

        Raises:
            ValueError: there is not exactly one stream per camera.
        """
        if len(encode_streams) != geometry.num_views:
            raise ValueError(
                f"expected one encode stream per camera, got "
                f"{len(encode_streams)} for {geometry.num_views} view(s)."
            )
        if render is None and worker is None:
            raise ValueError("Rendered controls require a renderer or worker")
        self._render = render
        self._streams = list(encode_streams)
        self._geometry = geometry
        self._latent_patch_size = latent_patch_size
        self._parallel = parallel
        self._worker = worker
        self._cursor = 0

    def tokens(self, start: int, end: int) -> Tensor:
        """Render, encode and patchify the control for frames ``[start, end)``.

        Raises:
            ValueError: the range is empty, or does not carry on from the last
                one -- the encode streams have already consumed the footage up
                to :attr:`next_frame` and cannot go back over it.
        """
        if start < 0 or start >= end:
            raise ValueError(f"[{start}, {end}) is not a frame range.")
        if start != self._cursor:
            raise ValueError(
                f"Control frames must carry on from {self._cursor}, got a range "
                f"starting at {start}. The encode streams hold the temporal "
                f"state for frames [0, {self._cursor}) and cannot be rewound."
            )
        owned = self._owned()
        first, last = pixel_span_for_latents(start, end)
        if self._worker is not None:
            encoded, pixel_frames = self._worker.take(start, end)
            latents = self._gathered(encoded, pixel_frames=pixel_frames)
        else:
            assert self._render is not None
            frames = (
                self._render(first, last)
                if self._parallel is None
                else self._render(first, last, views=owned)
            )
            latents = self._encoded(frames, pixel_frames=last - first, owned=owned)
        if latents.shape[2] != end - start:
            raise ValueError(
                f"encoding {last - first} pixel frames gave {latents.shape[2]} "
                f"latent frame(s), expected {end - start} for [{start}, {end})."
            )
        self._cursor = end
        return control_tokens(latents, latent_patch_size=self._latent_patch_size)

    def prefetch(self, start: int, end: int) -> None:
        """Have the worker start on ``[start, end)`` now, for a :meth:`tokens` later.

        Nothing happens without a worker. Asking again for the range in flight
        is a no-op, so a caller may say it every chunk; the worker refuses a
        different one, since its encoder has consumed the footage.

        Raises:
            ValueError: the range is empty or does not carry on from
                :attr:`next_frame`.
        """
        if self._worker is None:
            return
        if start < 0 or start >= end:
            raise ValueError(f"[{start}, {end}) is not a frame range.")
        if start != self._cursor:
            raise ValueError(
                f"Control frames must carry on from {self._cursor}, got a prefetch "
                f"starting at {start}."
            )
        self._worker.request(start, end)

    @property
    def next_frame(self) -> int:
        """The latent frame the next range has to start at."""
        return self._cursor

    def _owned(self) -> list[int]:
        """The cameras this rank draws and encodes: all of them without a mesh."""
        world = 1 if self._parallel is None else self._parallel.world_size
        rank = 0 if self._parallel is None else self._parallel.rank
        return list(range(rank, self._geometry.num_views, world))

    def _encoded(
        self, frames: Tensor, *, pixel_frames: int, owned: list[int]
    ) -> Tensor:
        """``[len(owned)*T, H, W, C]`` uint8 -> ``[V, C, T_lat, H_lat, W_lat]``.

        One camera through its own stream, because the encoder's temporal state
        is per track and eleven cameras are eleven tracks. Sharing a stream would
        run the front camera's footage into the rear camera's. ``frames`` holds
        only the cameras this rank owns; the rest arrive by broadcast.
        """
        clips = rendered_clips(frames, pixel_frames=pixel_frames, count=len(owned))
        streams = iter(self._streams[view] for view in owned)
        return _encoded_views(
            clips,
            lambda clip: next(streams)(clip),
            num_views=self._geometry.num_views,
            frames=-(-pixel_frames // WAN_TEMPORAL_COMPRESSION),
            parallel=self._parallel,
        )

    def _gathered(self, latents: list[Tensor], *, pixel_frames: int) -> Tensor:
        """Latents already encoded (by the worker) -> ``[V, C, T_lat, H_lat, W_lat]`` across ranks."""
        return _encoded_views(
            latents,
            lambda latent: latent,
            num_views=self._geometry.num_views,
            frames=-(-pixel_frames // WAN_TEMPORAL_COMPRESSION),
            parallel=self._parallel,
        )


def rendered_clips(frames: Tensor, *, pixel_frames: int, count: int) -> list[Tensor]:
    """``[count*T, H, W, C]`` uint8 as a rasterizer hands it over -> ``count`` clips ``[T, 3, H, W]`` in ``[-1, 1]``.

    Alpha is dropped: the control renders are opaque and the encoder has three
    input channels.

    Raises:
        ValueError: the frame count or channel count is not what ``count`` cameras over
            ``pixel_frames`` frames would give.
    """
    if frames.ndim != 4 or frames.shape[0] != count * pixel_frames:
        raise ValueError(
            f"expected [{count * pixel_frames}, H, W, C] rendered frames for "
            f"{count} view(s) x {pixel_frames} pixel frames, got "
            f"{tuple(frames.shape)}."
        )
    if count and frames.shape[-1] < 3:
        raise ValueError(f"expected at least 3 channels, got {frames.shape[-1]}.")
    clips = (
        frames[..., :3]
        .reshape(count, pixel_frames, *frames.shape[1:3], 3)
        .permute(0, 1, 4, 2, 3)
        .to(torch.float32)
        .div(127.5)
        .sub(1.0)
    )
    return list(clips)

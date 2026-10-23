# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""From a multiview sample to the tokens the rollout starts from.

Everything downstream works in token space; this is where the shapes stop being
arbitrary. It reads the packaged sample, encodes each camera's control clip into
the 48-channel Wan 2.2 latent space, patchifies it, and derives the
:class:`~cosmos3_nano_sim_auto.impl.packing.ClipGeometry` the rest of the package is cut to.

A sample is either ``text2video``, where the model invents the opening, or
``image2video``, where :func:`condition_latents` encodes real opening frames.
The checkpoint was validated the second way.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, cast

import torch
import torch.distributed as dist
from cosmos_interactive._patchify import (
    pad_latent_bottom,
    pad_latent_right,
    patchify_latent,
)
from cosmos_interactive._tokenizer import DEFAULT_TOKENIZER, tokenize_prompt
from torch import Tensor

from cosmos3_nano_sim_auto.impl.conditioning.captions import (
    AV_JOINT_CAMERA_LIDAR_TRANSFER_SYSTEM_PROMPT,
    AV_MULTIVIEW_TRANSFER_SYSTEM_PROMPT,
    CAMERA_CAPTION_PREFIXES,
    caption_token_ids,
    format_view_caption,
    multiview_caption,
    view_caption_payloads,
)
from cosmos3_nano_sim_auto.impl.distributed.parallel import owned_views
from cosmos3_nano_sim_auto.impl.packing import ClipGeometry
from flashdreams.core.distributed.parallel import ParallelContext

CAMERA_VIEW_IDS = {
    camera: index for index, camera in enumerate(CAMERA_CAPTION_PREFIXES)
}
"""Canonical recorded-scene embedding rows 0..10; row 11 is reserved for LiDAR."""


def camera_view_ids(cameras: Sequence[str]) -> tuple[int, ...] | None:
    """Resolve physical IDs without treating a subset index as a rig identity."""
    if any(camera not in CAMERA_VIEW_IDS for camera in cameras):
        return None
    return tuple(CAMERA_VIEW_IDS[camera] for camera in cameras)


WAN_SPATIAL_COMPRESSION = 16
"""Pixels per latent cell on each spatial axis, for the Wan 2.2 48-channel VAE."""

WAN_TEMPORAL_COMPRESSION = 4
"""Pixel frames per latent frame after the first, which stands alone."""

VIEW_SIZES: dict[str, tuple[int, int]] = {"480": (832, 480), "720": (1280, 720)}
"""``resolution`` field -> one view's ``(pixel_width, pixel_height)`` at 16:9.

Matches the upstream 480p and 720p multiview tables. Unknown resolutions
raise rather than guessing; 720p pads the odd latent height only for patching.
"""

CONDITION_FRAMES: dict[str, int] = {"text2video": 0, "image2video": 1}
"""Latent frames given rather than generated, per ``model_mode``."""


def latent_frames_for_pixels(pixel_frames: int) -> int:
    """Latent frames a clip of ``pixel_frames`` encodes to.

    The first latent frame covers one pixel frame and every later one covers
    :data:`WAN_TEMPORAL_COMPRESSION`, so the count is exact only when
    ``pixel_frames - 1`` divides evenly. Requiring that rather than rounding is
    deliberate: a partial trailing group would leave the control track and the
    target a frame out of step, which the mask cannot express and would not
    complain about.
    """
    if pixel_frames < 1:
        raise ValueError(f"pixel_frames must be >= 1, got {pixel_frames}.")
    if (pixel_frames - 1) % WAN_TEMPORAL_COMPRESSION:
        raise ValueError(
            f"pixel_frames must be 1 + a multiple of {WAN_TEMPORAL_COMPRESSION}, "
            f"got {pixel_frames}. The nearest usable counts are "
            f"{pixel_frames - (pixel_frames - 1) % WAN_TEMPORAL_COMPRESSION} and "
            f"{pixel_frames + WAN_TEMPORAL_COMPRESSION - (pixel_frames - 1) % WAN_TEMPORAL_COMPRESSION}."
        )
    return 1 + (pixel_frames - 1) // WAN_TEMPORAL_COMPRESSION


def pixel_frames_for_latents(latent_frames: int) -> int:
    """Inverse of :func:`latent_frames_for_pixels`."""
    if latent_frames < 1:
        raise ValueError(f"latent_frames must be >= 1, got {latent_frames}.")
    return WAN_TEMPORAL_COMPRESSION * (latent_frames - 1) + 1


def patch_grid(
    pixel_height: int, pixel_width: int, *, latent_patch_size: int
) -> tuple[int, int]:
    """One view's ``(patch_h, patch_w)`` token grid for a frame of that pixel size."""
    if latent_patch_size < 1:
        raise ValueError(f"latent_patch_size must be >= 1, got {latent_patch_size}.")
    if min(pixel_height, pixel_width) < 1 or (
        pixel_height % WAN_SPATIAL_COMPRESSION or pixel_width % WAN_SPATIAL_COMPRESSION
    ):
        raise ValueError(
            f"view size {pixel_width}x{pixel_height} must divide by "
            f"{WAN_SPATIAL_COMPRESSION} on both axes for VAE compression."
        )
    cell = WAN_SPATIAL_COMPRESSION * latent_patch_size
    return (pixel_height + cell - 1) // cell, (pixel_width + cell - 1) // cell


@dataclass(frozen=True)
class MultiviewSample:
    """One record of an ``omni_multiview`` input file.

    Attributes:
        name: the record's own name, used to label outputs.
        model_mode: ``"text2video"`` or ``"image2video"``.
        prompt: the caption, shared by every view.
        resolution: key into :data:`VIEW_SIZES`.
        cameras: camera keys in the file's order, which is the rig order views
            are numbered by.
        control_paths: one control clip per camera, aligned with
            :attr:`cameras`.
        view_captions: one caption per camera, same order, for samples from real
            recorded-scene data. Empty for the vendored sample and hand-written records,
            which carry a single :attr:`prompt` instead. Whichever is present
            decides how the prompt is built; see :func:`prompt_token_ids`.
        image_paths: one video clip per camera, same order, when the sample is
            ``image2video``. Their opening frames are given to the model rather
            than invented.
        frame_range: the ``(start, end)`` source frames the captions describe,
            for samples from real recorded-scene data. All that ties the words to the
            pixels, so a clip cut anywhere else is captioned wrong. ``None`` for
            records that describe whatever they are given.
    """

    name: str
    model_mode: str
    prompt: str
    resolution: str
    cameras: tuple[str, ...]
    control_paths: tuple[str, ...]
    view_captions: tuple[Any, ...] = ()
    image_paths: tuple[str, ...] = ()
    frame_range: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        # Without the footage, the geometry would still reserve frames as
        # supplied and nothing would write them, leaving the model to denoise
        # against noise it was told was real.
        if self.model_mode == "image2video" and not self.image_paths:
            raise ValueError(
                f"{self.name!r} is image2video but carries no image_path; "
                "the opening frames need real footage to encode."
            )

    @classmethod
    def load(cls, path: str | Path, *, record: int = 0) -> MultiviewSample:
        """Read one record from a JSON or JSON Lines ``omni_multiview`` input.

        Args:
            path: explicit local input file.
            record: which record, for the JSON Lines files that carry several.
                They differ only in sampling settings, so the default is usually
                what you want.

        Raises:
            ValueError: the file is empty, has no views, or holds fewer records
                than ``record`` asks for.
        """
        path = Path(path)
        text = path.read_text()
        if path.suffix == ".jsonl":
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            records = [json.loads(text)]
        if not records:
            raise ValueError(f"{path} holds no records.")
        if not 0 <= record < len(records):
            raise ValueError(
                f"{path} holds {len(records)} record(s); asked for index {record}."
            )
        selected = records[record]
        if "prepared_scene" in selected:
            from cosmos3_nano_sim_auto.impl.conditioning.world_scenario.prepared_scene import (
                asset,
            )

            if selected["prepared_scene"] != "scene.json":
                raise ValueError("Unsupported prepared scene reference")
            for view in selected["multiview"]["views"]:
                for key in ("control_path", "image_path"):
                    view[key] = str(asset(path.parent, view[key]))
        return cls.from_record(selected, name=path.stem)

    @classmethod
    def from_record(
        cls, record: dict[str, Any], *, name: str = "sample"
    ) -> MultiviewSample:
        """Build from an already-parsed record."""
        views = record.get("multiview", {}).get("views", [])
        if not views:
            raise ValueError("the record carries no multiview views.")
        missing = [
            key
            for key in ("camera_key", "control_path")
            if any(key not in v for v in views)
        ]
        if missing:
            raise ValueError(f"every view needs {missing}; some are missing it.")
        # All or nothing: a record where only some views carry a caption would
        # build a prompt with holes in it, and the model would read the holes as
        # descriptions of those cameras rather than as absences.
        captions = [view["caption"] for view in views if "caption" in view]
        if captions and len(captions) != len(views):
            raise ValueError(
                f"{len(captions)} of {len(views)} views carry a caption; "
                "a multiview prompt needs one per camera or none at all."
            )
        images = [str(view["image_path"]) for view in views if "image_path" in view]
        if images and len(images) != len(views):
            raise ValueError(
                f"{len(images)} of {len(views)} views carry an image_path; "
                "conditioning is synchronised across the rig, so it is all or none."
            )
        span = record.get("frame_range")
        if span is not None and len(tuple(span)) != 2:
            raise ValueError(f"frame_range must be a (start, end) pair, got {span!r}.")
        return cls(
            name=str(record.get("name", name)),
            model_mode=str(record["model_mode"]),
            prompt=str(record.get("prompt", "")),
            resolution=str(record.get("resolution", "480")),
            cameras=tuple(str(view["camera_key"]) for view in views),
            control_paths=tuple(str(view["control_path"]) for view in views),
            view_captions=tuple(captions),
            image_paths=tuple(images),
            frame_range=None if span is None else (int(span[0]), int(span[1])),
        )

    def offset_for(
        self, pixel_frames: int | None, *, outlive_caption: bool = False
    ) -> int:
        """The source frame the clip should start at, checked against the caption.

        Args:
            pixel_frames: pixel frames per view the run will generate, or
                ``None`` for a run with no end, which outlives any caption and
                so is only allowed to start where one does.
            outlive_caption: Allow a run longer than the caption describes.
                A caption chunk covers 200 source frames and training always
                cut the clip to that span, so this is off by default. It exists
                to measure how badly a stale caption ages, which a run of
                unknown length has to know.

        Returns:
            :attr:`frame_range`'s start, or 0 when the sample carries no range.

        Raises:
            ValueError: ``pixel_frames`` runs past the end of the range the
                captions describe, and ``outlive_caption`` is not set.
        """
        if self.frame_range is None:
            return 0
        start, end = self.frame_range
        if pixel_frames is None:
            return start
        if pixel_frames > end - start and not outlive_caption:
            raise ValueError(
                f"{pixel_frames} frames asked for, but {self.name!r}'s captions "
                f"describe source frames {start}-{end}, which is {end - start}. "
                "Generating past the range makes the prompt describe frames that "
                "are not in the clip; use --frames at or below it, or pass "
                "--outlive-caption to do it anyway."
            )
        return start

    @property
    def num_views(self) -> int:
        return len(self.cameras)

    def trimmed_to(self, num_views: int) -> MultiviewSample:
        """The first ``num_views`` cameras, with everything else cut to match.

        Trimming keeps rig order, so view N stays the camera it was and its
        positions stay where the checkpoint expects them. A method rather than a
        ``dataclasses.replace`` at each call site, because three tuples have to
        be cut together: leave the captions long and they run off the end of the
        cameras, drop them entirely and the prompt quietly reverts to the plain
        string.
        """
        if not 1 <= num_views <= self.num_views:
            raise ValueError(
                f"num_views must lie in [1, {self.num_views}], got {num_views}."
            )
        return dataclasses.replace(
            self,
            cameras=self.cameras[:num_views],
            control_paths=self.control_paths[:num_views],
            view_captions=self.view_captions[:num_views],
            image_paths=self.image_paths[:num_views],
        )

    def for_rig(self, cameras: Sequence[str]) -> MultiviewSample:
        """The cameras named, in the order named, everything else cut to match.

        :meth:`trimmed_to` takes a prefix, which is the rig ``--views N`` asks
        for. An application that names its cameras can ask for any of them in
        any order, and view N is then whichever it named Nth, so reordering is
        the point here rather than a side effect of the cut.

        Raises:
            ValueError: a name is not one of this sample's cameras, or one is
                named twice, which would feed the model one view at two
                positions and leave another unconditioned.
        """
        wanted = list(cameras)
        repeated = sorted({name for name in wanted if wanted.count(name) > 1})
        if repeated:
            raise ValueError(f"the rig names {repeated} more than once.")
        missing = [name for name in wanted if name not in self.cameras]
        if missing:
            raise ValueError(
                f"{missing} are not cameras of {self.name!r}, which carries "
                f"{list(self.cameras)}."
            )
        picked = [self.cameras.index(name) for name in wanted]

        def cut(values: tuple[Any, ...]) -> tuple[Any, ...]:
            # Absent rather than short: an empty captions tuple is what tells
            # the prompt builder there are none, and indexing into it here
            # would raise instead.
            return tuple(values[index] for index in picked) if values else ()

        return dataclasses.replace(
            self,
            cameras=tuple(self.cameras[index] for index in picked),
            control_paths=tuple(self.control_paths[index] for index in picked),
            view_captions=cut(self.view_captions),
            image_paths=cut(self.image_paths),
        )

    @property
    def physical_camera_indexes(self) -> tuple[int, ...]:
        """The ``_cN`` index each control path carries, in rig order.

        Rig order is not physical camera order. The sample lists cameras in the
        model's view order, while clip names use physical camera indices.
        The two orders are a permutation of each other:
        ``(0, 2, 4, 5, 3, 1, 6, 7, 8, 10, 9)`` for the 11-view sample, so six of
        the eleven disagree.

        Pairing clips to views by position, or naming downloads by ``cN`` and
        reloading them by index, therefore gives six cameras another camera's
        road, and the result still decodes and still looks like driving. Pair by
        :attr:`cameras`; use this property to interpret source clip names.

        Raises:
            ValueError: a control path does not end in ``_cN.mp4``.
        """
        indexes: list[int] = []
        for camera, path in zip(self.cameras, self.control_paths, strict=True):
            match = re.search(r"_c(\d+)\.mp4$", path)
            if match is None:
                raise ValueError(
                    f"control path for {camera} does not end in _cN.mp4: {path}"
                )
            indexes.append(int(match.group(1)))
        return tuple(indexes)

    @property
    def view_size(self) -> tuple[int, int]:
        """One view's ``(pixel_width, pixel_height)``."""
        if self.resolution not in VIEW_SIZES:
            raise ValueError(
                f"resolution {self.resolution!r} is not one this demo knows; "
                f"expected one of {sorted(VIEW_SIZES)}."
            )
        return VIEW_SIZES[self.resolution]

    @property
    def condition_frames(self) -> int:
        """Latent frames given rather than generated, from :attr:`model_mode`."""
        if self.model_mode not in CONDITION_FRAMES:
            raise ValueError(
                f"model_mode {self.model_mode!r} is not one this demo knows; "
                f"expected one of {sorted(CONDITION_FRAMES)}."
            )
        return CONDITION_FRAMES[self.model_mode]

    def geometry(
        self,
        *,
        pixel_frames: int | None,
        latent_patch_size: int,
        frames_per_chunk: int,
        fps: float = 30.0,
        condition_frames: int | None = None,
    ) -> ClipGeometry:
        """The clip geometry this sample rolls out at.

        Args:
            pixel_frames: pixel frames per view to generate, or ``None`` for a
                run with no end.
            latent_patch_size: the checkpoint's ``spec.vfm.latent_patch_size``.
            frames_per_chunk: latent frames per autoregressive step.
            fps: playback rate. Only the ``"decomposed"`` scope's timestamps
                read it, and the default ``"all_views"`` scope does not.
            condition_frames: Override :attr:`condition_frames`, which the mode
                otherwise fixes at 0 or 1. Training used two a tenth of the
                time, so two works without a third ``model_mode``.

        Raises:
            ValueError: the override asks for opening frames the sample has no
                footage to supply.
        """
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}.")
        frames = self.condition_frames if condition_frames is None else condition_frames
        if frames and not self.image_paths:
            raise ValueError(
                f"{frames} conditioning frame(s) asked for, but the sample carries no "
                "image_path to encode them from."
            )
        width, height = self.view_size
        patch_h, patch_w = patch_grid(
            height, width, latent_patch_size=latent_patch_size
        )
        return ClipGeometry(
            num_views=self.num_views,
            frames_per_view=(
                None if pixel_frames is None else latent_frames_for_pixels(pixel_frames)
            ),
            patch_h=patch_h,
            patch_w=patch_w,
            frames_per_chunk=frames_per_chunk,
            condition_frames=frames,
            seconds_per_frame=WAN_TEMPORAL_COMPRESSION / fps,
            latent_size=(
                height // WAN_SPATIAL_COMPRESSION,
                width // WAN_SPATIAL_COMPRESSION,
            ),
            rig_view_ids=camera_view_ids(self.cameras),
        )


class Prompt(NamedTuple):
    """A tokenized prompt and, if it is one caption per camera, its runs.

    ``view_text_tokens`` is ``None`` for a single caption covering the rig, and
    otherwise the length of each camera's caption in rig order. It travels with
    the ids everywhere, because the ids alone do not say where one caption ends
    and the next begins, and everything downstream needs to know: the prompt's
    self-attention keeps each caption a document of its own, the mRoPE grid lays
    them all from zero, and the attention mask scopes each to its camera.
    """

    text_ids: Tensor
    view_text_tokens: tuple[int, ...] | None


def prompt_token_ids(
    sample: MultiviewSample,
    *,
    pixel_frames: int,
    fps: float = 30.0,
    tokenizer_path: str = DEFAULT_TOKENIZER,
    per_view_captions: bool = True,
    lidar: bool = False,
    parallel: ParallelContext | None = None,
) -> Prompt:
    """The prompt as token ids, in whichever of two shapes fits.

    A sample with :attr:`~MultiviewSample.view_captions` gets what training
    used: a JSON object per camera holding that camera's caption alongside the
    clip's duration, fps, resolution and aspect ratio. See
    :mod:`cosmos3_nano_sim_auto.impl.conditioning.captions`.

    A sample with only a single :attr:`~MultiviewSample.prompt` gets that
    sentence, pointed at each camera in turn, plus the model's resolution suffix. That
    keeps the vendored sample and the synthetic path working without the data
    bucket, but it is not what the checkpoint was fit to, so a real sample
    should not use it.

    Args:
        pixel_frames: frames per view being generated. Only the caption payload
            reads it, to state the duration the way training did.
        fps: playback rate the duration is computed against.
        per_view_captions: whether the checkpoint tokenizes one caption per
            camera. This is a property of the checkpoint and not a choice: the
            delivered one was trained that way, and a merged prompt read by
            every camera is a layout it never saw. ``False`` is the older
            layout, kept because the two are worth being able to compare.
        lidar: Select the joint camera/LiDAR system message. RGB generation
            uses the multiview system message even when selecting one camera.
        parallel: Tokenize once on rank zero and share the token ids.
    """
    if parallel is not None and parallel.world_size > 1:
        payload = [
            prompt_token_ids(
                sample,
                pixel_frames=pixel_frames,
                fps=fps,
                tokenizer_path=tokenizer_path,
                per_view_captions=per_view_captions,
                lidar=lidar,
            )
            if parallel.is_main
            else None
        ]
        dist.broadcast_object_list(payload, src=0, device=_object_device(parallel))
        assert payload[0] is not None
        return payload[0]
    width, height = sample.view_size
    system_prompt = (
        AV_JOINT_CAMERA_LIDAR_TRANSFER_SYSTEM_PROMPT
        if lidar
        else AV_MULTIVIEW_TRANSFER_SYSTEM_PROMPT
    )
    if not per_view_captions:
        if sample.view_captions:
            payload = multiview_caption(
                sample.view_captions,
                sample.cameras,
                num_frames=pixel_frames,
                fps=fps,
                pixel_height=height,
                pixel_width=width,
            )
            return Prompt(
                caption_token_ids(
                    payload,
                    tokenizer_path=tokenizer_path,
                    system_prompt=system_prompt,
                ),
                None,
            )
        return Prompt(
            tokenize_prompt(
                sample.prompt,
                height=height,
                width=width,
                tokenizer_path=tokenizer_path,
                system_prompt=system_prompt,
            ),
            None,
        )

    if sample.view_captions:
        runs = [
            caption_token_ids(
                payload,
                tokenizer_path=tokenizer_path,
                system_prompt=system_prompt,
            )
            for payload in view_caption_payloads(
                sample.view_captions,
                sample.cameras,
                num_frames=pixel_frames,
                fps=fps,
                pixel_height=height,
                pixel_width=width,
            )
        ]
    else:
        runs = [
            tokenize_prompt(
                format_view_caption(sample.prompt, camera),
                height=height,
                width=width,
                tokenizer_path=tokenizer_path,
                system_prompt=system_prompt,
            )
            for camera in sample.cameras
        ]
    return Prompt(torch.cat(runs), tuple(int(run.shape[0]) for run in runs))


def _load_control_encoder(
    *,
    dtype: torch.dtype,
    checkpoint_path: str | None,
    device: str | torch.device,
    compile: bool = False,
) -> tuple[Any, torch.device]:
    """The Wan 2.2 encoder behind both the control track and the supplied frames.

    ``compile`` runs the encoder under ``torch.compile``. Its CUDA graph stays
    off: the wrapper captures the one cache it was captured with, and the
    streams below each carry their own.
    """
    # Imported here rather than at module scope: it reaches the whole VAE stack,
    # and the geometry, parsing and patchify above are worth having in a CPU
    # environment that has none of it.
    from cosmos_interactive._vae import wan22_vae_encoder_config

    encoder = (
        wan22_vae_encoder_config(
            dtype=dtype, checkpoint_path=checkpoint_path, use_compile=compile
        )
        .setup()
        .to(device)
    )
    return encoder, next(encoder.parameters()).device


def make_control_encode_fn(
    *,
    dtype: torch.dtype = torch.bfloat16,
    checkpoint_path: str | None = None,
    device: str | torch.device = "cuda",
) -> Callable[[Tensor], Tensor]:
    """Build ``encode(clip[T, 3, H, W]) -> latent[C, T_lat, H_lat, W_lat]``.

    The clip is RGB in ``[-1, 1]``, which is what
    ``flashdreams.infra.runner_io.load_video_tensor`` returns and what
    :func:`~cosmos3_nano_sim_auto.impl.conditioning.controls.synthetic_control_clip` produces. The latent
    is mean/std-normalized -- the encoder config carries the Wan 2.2 statistics --
    so it lands in the same space the rollout emits and the decoder consumes, and
    needs no scaling on the way in.

    Every call encodes from an empty temporal state, so a clip has to arrive
    whole. See :func:`make_control_encode_streams` for the streaming form.
    """
    encoder, resident = _load_control_encoder(
        dtype=dtype, checkpoint_path=checkpoint_path, device=device
    )

    def encode_fn(clip: Tensor) -> Tensor:
        if clip.ndim != 4 or clip.shape[1] != 3:
            raise ValueError(f"expected a [T, 3, H, W] clip, got {tuple(clip.shape)}.")
        video = clip.to(device=resident, dtype=dtype)[None]  # [1, T, 3, H, W]
        with torch.no_grad():
            latent = encoder(video)  # [1, T_lat, C, H_lat, W_lat]
        return latent[0].transpose(0, 1).contiguous()  # [C, T_lat, H_lat, W_lat]

    return encode_fn


def make_control_encode_streams(
    count: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
    checkpoint_path: str | None = None,
    device: str | torch.device = "cuda",
    compile: bool = False,
) -> list[Callable[[Tensor], Tensor]]:
    """One ``encode`` per camera that carries its temporal state between calls.

    Same encoder as :func:`make_control_encode_fn` and only the cache differs,
    so ``count`` streams cost one set of weights. Each holds a ``WanVAECache``,
    the history the causal encoder looks back into, which is what lets a clip
    arrive a piece at a time and still encode as though it arrived whole.

    A stream is stateful and so order-dependent. Feed it consecutive spans of
    one camera's footage, starting at pixel frame 0, and give each camera its
    own stream: sharing one would run the front camera's road into the rear
    camera's. The encoder wants ``1 + 4n`` pixel frames on the opening call and
    ``4n`` after that, which
    :func:`~cosmos3_nano_sim_auto.impl.conditioning.control_provider.pixel_span_for_latents` provides.

    Raises:
        ValueError: ``count`` is not positive.
    """
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}.")
    encoder, resident = _load_control_encoder(
        dtype=dtype, checkpoint_path=checkpoint_path, device=device, compile=compile
    )

    def make_stream() -> Callable[[Tensor], Tensor]:
        cache = encoder.initialize_autoregressive_cache()

        def encode_fn(clip: Tensor) -> Tensor:
            if clip.ndim != 4 or clip.shape[1] != 3:
                raise ValueError(
                    f"expected a [T, 3, H, W] clip, got {tuple(clip.shape)}."
                )
            video = clip.to(device=resident, dtype=dtype)[None]  # [1, T, 3, H, W]
            with torch.no_grad():
                # Advanced in place, so this call's frames become the next
                # call's history.
                latent = encoder(video, cache=cache)  # [1, T_lat, C, H_lat, W_lat]
            return latent[0].transpose(0, 1).contiguous()

        return encode_fn

    return [make_stream() for _ in range(count)]


def control_latents(
    clips: Sequence[Tensor],
    encode_fn: Callable[[Tensor], Tensor],
    *,
    geometry: ClipGeometry,
    parallel: ParallelContext | None = None,
) -> Tensor:
    """Encode one control clip per camera into ``[V, C, T_lat, H_lat, W_lat]``.

    Args:
        clips: one ``[T, 3, H, W]`` clip per camera, in :attr:`MultiviewSample.cameras`
            order. With ``parallel``, only cameras ``[rank::world_size]``.
        encode_fn: from :func:`make_control_encode_fn`, or a stand-in.
        geometry: what the result is checked against.
        parallel: Divide the cameras across this mesh and share their latents.

    Raises:
        ValueError: the clip count or an encoded shape disagrees with
            ``geometry``, or the clip has no end and so no whole track to
            encode; those controls are drawn a chunk at a time instead.
    """
    if geometry.frames_per_view is None:
        raise ValueError(
            "a clip with no end has no whole control track to encode. Draw it a "
            "chunk at a time with a ControlProvider."
        )
    return _encoded_views(
        clips,
        encode_fn,
        num_views=geometry.num_views,
        frames=geometry.frames_per_view,
        parallel=parallel,
    )


def condition_latents(
    clips: Sequence[Tensor],
    encode_fn: Callable[[Tensor], Tensor],
    *,
    geometry: ClipGeometry,
    parallel: ParallelContext | None = None,
) -> Tensor:
    """Encode the supplied frames into ``[V, C, condition_frames, H_lat, W_lat]``.

    The same encoder as the control track, pointed at the real footage instead.
    This is the difference between image2video and text2video: the model is
    handed the first latent frame or two of what it is about to generate and
    continues from there rather than inventing an opening.

    Encoding only the opening is safe, which is not obvious. These latents have
    to match the first frames of encoding the whole clip, or the model would
    continue from something off its own trajectory. Wan 2.2's encoder never
    looks ahead, so an opening encodes to an opening; one that did look ahead
    would need the whole clip and a slice of the result.

    Args:
        clips: one ``[T, 3, H, W]`` clip per camera, already trimmed to
            :func:`condition_pixel_frames` frames. With ``parallel``, only
            cameras ``[rank::world_size]``.
        encode_fn: from :func:`make_control_encode_fn`; there is one VAE.
        geometry: what the result is checked against.
        parallel: Divide the cameras across this mesh and share their latents.

    Raises:
        ValueError: the geometry supplies no frames, or a shape disagrees.
    """
    if not geometry.condition_frames:
        raise ValueError(
            "the geometry has no conditioning frames, so there is nothing to encode."
        )
    return _encoded_views(
        clips,
        encode_fn,
        num_views=geometry.num_views,
        frames=geometry.condition_frames,
        parallel=parallel,
    )


def condition_pixel_frames(geometry: ClipGeometry) -> int:
    """Pixel frames to read per camera to fill the supplied latent frames.

    One latent frame is one pixel frame; two is five, the first standing alone
    and every later one covering four. Reading ``condition_frames`` pixels
    instead would quietly supply a quarter of the intended span.
    """
    return pixel_frames_for_latents(geometry.condition_frames)


def _encoded_views(
    clips: Sequence[Tensor],
    encode_fn: Callable[[Tensor], Tensor],
    *,
    num_views: int,
    frames: int,
    parallel: ParallelContext | None = None,
) -> Tensor:
    """Encode one clip per camera and stack, checking the count and each shape.

    One camera at a time rather than batched, because the encoder carries a
    temporal cache and views are independent tracks that must not share it.
    With a mesh, cameras go round-robin to the ranks, every owner encodes
    before any broadcast so they work concurrently, and the broadcasts restore
    camera order on every rank.
    """
    world = 1 if parallel is None else parallel.world_size
    owned = owned_views(parallel, num_views)
    if len(clips) != len(owned):
        raise ValueError(f"expected {len(owned)} clips, got {len(clips)}.")

    encoded: list[Tensor | None] = [None] * num_views
    for view, clip in zip(owned, clips, strict=True):
        encoded[view] = encode_fn(clip)
    for view, latent in enumerate(encoded):
        if parallel is not None and world > 1:
            device = _object_device(parallel)
            if latent is not None:
                latent = latent.to(device).contiguous()
            # Only the owner knows the shape, so it goes first.
            description = [
                None if latent is None else (tuple(latent.shape), latent.dtype)
            ]
            dist.broadcast_object_list(description, src=view % world, device=device)
            assert description[0] is not None
            shape, dtype = description[0]
            if latent is None:
                latent = torch.empty(shape, dtype=dtype, device=device)
            dist.broadcast(latent, src=view % world)
            latent = encoded[view] = latent.to(parallel.device)
        assert latent is not None
        if latent.ndim != 4:
            raise ValueError(
                f"view {view} encoded to {tuple(latent.shape)}, not [C, T, H, W]."
            )
        if latent.shape[1] != frames:
            raise ValueError(
                f"view {view} encoded to {latent.shape[1]} latent frames, "
                f"but {frames} were expected."
            )
    return torch.stack(cast(list[Tensor], encoded))


def _object_device(parallel: ParallelContext) -> torch.device:
    """Where the default group's collectives stage tensors: the rank's card under NCCL.

    Passed explicitly because the model thread need not inherit the launching
    thread's CUDA device.
    """
    return parallel.device if dist.get_backend() == "nccl" else torch.device("cpu")


def control_tokens(latents: Tensor, *, latent_patch_size: int) -> Tensor:
    """``[V, C, T, H, W]`` latents -> ``[V*T*patch_h*patch_w, patch_latent_dim]``.

    Serves the control track and the supplied frames alike, both being
    patchified the same way.

    View-outer, frame-inner, spatial-innermost: the order
    :mod:`cosmos3_nano_sim_auto.impl.packing` stamps metadata in and the order the positions
    are generated in. ``patchify_latent`` gives frames then spatial within a
    view, and concatenating views in rig order gives the rest.
    """
    if latents.ndim != 5:
        raise ValueError(
            f"expected [V, C, T, H, W] latents, got {tuple(latents.shape)}."
        )
    return torch.cat(
        [
            patchify_latent(
                pad_latent_right(
                    pad_latent_bottom(view, -view.shape[-2] % latent_patch_size),
                    -view.shape[-1] % latent_patch_size,
                ),
                latent_patch_size,
            )
            for view in latents
        ],
        dim=0,
    )

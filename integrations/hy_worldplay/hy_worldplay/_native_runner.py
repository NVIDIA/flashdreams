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

"""Native-pipeline driver for the HY-WorldPlay WAN-5B I2V runner.

Drives :data:`flashdreams.recipes.wan.PIPELINE_WAN22_TI2V_5B` directly
instead of upstream's :class:`wan.generate.WanRunner`. Shipped
incrementally with the rest of phase 2b: the base I2V case (2b.1) and
distilled scheduler (2b.2) are wired here unconditionally, the action
conditioner (2b.3) activates when
:attr:`HyWorldPlayWanI2VRunnerConfig.use_action_conditioning` is set,
the camera-trajectory PRoPE conditioner (2b.4) activates when
:attr:`HyWorldPlayWanI2VRunnerConfig.use_camera_conditioning` is set,
and reconstituted-context memory lands in 2b.5. The phase-1 vendor
wrapper in :class:`hy_worldplay.runner.HyWorldPlayWanI2VRunner` stays
as the default; this module's runner is selected by setting
``use_native_pipeline=True`` on
:class:`hy_worldplay.runner.HyWorldPlayWanI2VRunnerConfig`.

Module split rationale: the phase-1 vendor wrapper is ~320 LoC and
must stay byte-stable while phase 2b incubates. Putting the native-mode
driver next to it would double the size of ``runner.py`` and intermix
two unrelated control flows. Keeping it here makes the routing in
``runner.py``'s ``__post_init__`` the only place that bridges the two.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from flashdreams.infra.runner import Runner
from flashdreams.recipes.wan.pipeline import WanInferencePipeline

if TYPE_CHECKING:
    from hy_worldplay.runner import HyWorldPlayWanI2VRunnerConfig

__all__ = [
    "HyWorldPlayWanI2VNativeRunner",
    "preprocess_first_frame",
]


def preprocess_first_frame(
    image_path: Path,
    pixel_height: int,
    pixel_width: int,
) -> Tensor:
    """Load and resize the first-frame image to ``WanI2VCtrlEncoder``'s input shape.

    Returns a ``[1, 1, 3, H, W]`` float32 tensor in ``[-1, 1]``: leading
    ``1`` is the pipeline's ``batch_shape``, the next ``1`` is the
    single-time-step dimension required by
    :meth:`WanInferencePipeline.initialize_cache`.

    The aspect-ratio policy is **fit + centre-crop**, mirroring
    upstream's ``hyvideo/utils/image.py`` so the native pipeline sees
    the same conditioning frame as the vendor wrapper for matching
    pixel sizes.
    """
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    src_w, src_h = img.size
    target_h, target_w = pixel_height, pixel_width

    # Scale-to-fill (the longer side hits the target; the shorter side
    # overflows and is centre-cropped). Mirrors upstream's resize policy.
    scale = max(target_h / src_h, target_w / src_w)
    new_h = int(round(src_h * scale))
    new_w = int(round(src_w * scale))
    img = img.resize((new_w, new_h), Image.LANCZOS)

    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    img = img.crop((left, top, left + target_w, top + target_h))

    arr = torch.from_numpy(_pil_to_numpy(img)).float()  # [H, W, 3] in [0, 255]
    arr = arr.permute(2, 0, 1) / 127.5 - 1.0  # [3, H, W] in [-1, 1]
    return arr.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, H, W]


def _pil_to_numpy(img: object) -> object:
    """Indirection to keep the numpy import out of the module surface
    until the first preprocessing call (the CPU smoke tests import the
    module without pillow / numpy installed in some sub-venvs)."""
    import numpy as np

    return np.asarray(img)


class HyWorldPlayWanI2VNativeRunner(Runner["HyWorldPlayWanI2VRunnerConfig", WanInferencePipeline]):
    """Drive ``PIPELINE_WAN22_TI2V_5B`` end-to-end for the I2V base case.

    Phase 2b.1 deliverable. The runner inherits the standard
    :class:`Runner` machinery (torchrun bootstrap, distributed init,
    per-rank seed offset, ``pipeline.setup()`` + ``.to(device).eval()``)
    and supplies a single :meth:`run` method that resolves the prompt
    and first frame, calls ``pipeline.initialize_cache``, drives the AR
    loop with ``generate`` + ``finalize``, and writes an mp4 on rank 0.

    What's intentionally *not* here yet (lands incrementally per the
    phase-2b design spec):

    - **Action conditioning** (2b.3, landed). The action-aware encoder /
      transformer / network swap is wired through ``__post_init__``
      when ``use_action_conditioning=True``; this runner parses the
      pose string into discrete labels and binds them on the encoder
      before the rollout. With zero-init action weights the conditioner
      is still a strict identity, so output continues to match the base
      recipe until HY-WorldPlay's distilled checkpoint is loaded on top.
    - **Camera-trajectory conditioning** (2b.4, landed). The PRoPE
      dual-branch block swap is wired through ``__post_init__`` when
      ``use_camera_conditioning=True``; this runner parses the pose
      string into per-latent W2C extrinsics + intrinsics and binds them
      on the encoder before the rollout. With zero-init ``o_prope``
      weights the PRoPE branch contributes exactly zero residual, so
      output continues to match the base recipe until HY-WorldPlay's
      distilled checkpoint is loaded on top.
    - **Reconstituted-context memory** (2b.5). No KV prefill from past
      chunks; each chunk denoises independently from the previous
      chunk's last-frame KV cache only.
    """

    def run(self) -> None:
        """Roll one autoregressive sequence and persist the mp4 on rank 0."""
        from loguru import logger

        cfg = self.config
        if cfg.image_path is None:
            raise ValueError(
                "HY-WorldPlay WAN-5B is I2V only -- pass "
                "``--image-path <path-to-jpg>`` to provide the first frame."
            )
        if not cfg.image_path.exists():
            raise FileNotFoundError(f"image_path {cfg.image_path} does not exist")

        device = next(self.pipeline.parameters()).device
        image = preprocess_first_frame(
            cfg.image_path, cfg.pixel_height, cfg.pixel_width
        ).to(device)
        prompt = _resolve_prompt(cfg.prompt)

        cache = self.pipeline.initialize_cache(
            text=[prompt],
            image=image,
            height=None,  # derived from image
            width=None,
        )

        if cfg.use_action_conditioning:
            self._bind_action_labels()
        if cfg.use_camera_conditioning:
            self._bind_camera_data()

        chunks: list[Tensor] = []
        start_time = time.time()
        for ar_idx in range(cfg.num_chunk):
            chunk = self.pipeline.generate(ar_idx, cache)
            chunks.append(chunk)
            if ar_idx < cfg.num_chunk - 1:
                self.pipeline.finalize(ar_idx, cache)
        elapsed = time.time() - start_time

        if not self.is_rank_zero:
            return

        video = torch.cat(chunks, dim=-4)  # cat along T axis: [..., T, C, H, W]
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = cfg.output_dir / f"{cfg.runner_name}.mp4"
        _write_mp4(video, out_path, fps=cfg.fps)
        logger.info(
            f"[{cfg.runner_name}] (native) wrote video "
            f"({tuple(video.shape)}) -> {out_path.resolve()} in {elapsed:.2f}s"
        )

    def _bind_action_labels(self) -> None:
        """Parse the pose string and bind per-rollout action labels on the encoder.

        Pulled out of :meth:`run` so the action-conditioning branch can
        be exercised in isolation by tests without spinning up the full
        rollout.
        """
        from hy_worldplay._action import HyWorldPlayWanCtrlEncoder
        from hy_worldplay._pose import parse_pose_action_labels

        encoder, n_latents = self._resolve_encoder_and_n_latents(
            flag_name="use_action_conditioning"
        )
        assert isinstance(encoder, HyWorldPlayWanCtrlEncoder)
        labels = parse_pose_action_labels(self.config.pose, n_latents)
        encoder.set_action_labels(labels)

    def _bind_camera_data(self) -> None:
        """Parse the pose string and bind per-rollout viewmats + intrinsics on the encoder.

        Mirrors :meth:`_bind_action_labels`: the same :func:`parse_pose_data`
        call returns both the per-latent W2C / K and the action labels, but
        we only consume the camera tensors here so callers can flip the
        two flags independently.
        """
        from hy_worldplay._action import HyWorldPlayWanCtrlEncoder
        from hy_worldplay._pose import parse_pose_data

        encoder, n_latents = self._resolve_encoder_and_n_latents(
            flag_name="use_camera_conditioning"
        )
        assert isinstance(encoder, HyWorldPlayWanCtrlEncoder)
        viewmats, Ks, _ = parse_pose_data(self.config.pose, n_latents)
        # PRoPE math + cudnn attention run in the pipeline dtype (fp16 /
        # bf16); cast here so the per-frame transforms inside
        # ``prope_qkv`` don't kick the network into fp64 unintentionally.
        target_dtype = next(self.pipeline.parameters()).dtype
        encoder.set_camera_data(
            viewmats.to(dtype=target_dtype), Ks.to(dtype=target_dtype)
        )

    def _resolve_encoder_and_n_latents(
        self, *, flag_name: str
    ) -> tuple[object, int]:
        """Return ``(encoder, n_latents)`` after asserting the swap ran.

        Centralises the ``isinstance`` + ``len_t``-lookup boilerplate
        shared by :meth:`_bind_action_labels` and :meth:`_bind_camera_data`.
        ``flag_name`` is included in the assertion text so misconfigured
        runs point at the right config knob.
        """
        from flashdreams.recipes.wan.transformer.wan21 import Wan21TransformerConfig

        from hy_worldplay._action import HyWorldPlayWanCtrlEncoder

        cfg = self.config
        encoder = self.pipeline.encoder
        assert isinstance(encoder, HyWorldPlayWanCtrlEncoder), (
            f"{flag_name}=True requires the pipeline's encoder to be "
            f"HyWorldPlayWanCtrlEncoder; got {type(encoder).__name__}. "
            "Did __post_init__ run? (Constructing the config via setup() drives it.)"
        )
        transformer_cfg = self.pipeline.diffusion_model.transformer.config
        assert isinstance(transformer_cfg, Wan21TransformerConfig), (
            f"{flag_name}=True expected a Wan21TransformerConfig (or subclass) "
            f"on the diffusion model; got {type(transformer_cfg).__name__}."
        )
        n_latents = cfg.num_chunk * transformer_cfg.len_t
        return encoder, n_latents


def _resolve_prompt(value: str | Path) -> str:
    """Read an inline prompt or the first non-empty line of a prompt file."""
    if isinstance(value, Path):
        lines = [ln.strip() for ln in value.read_text().splitlines() if ln.strip()]
        assert lines, f"prompt file {value} has no non-empty lines"
        return lines[0]
    assert value, "--prompt must be a non-empty string or a path to a .txt file"
    return value


def _write_mp4(video: Tensor, out_path: Path, *, fps: int) -> None:
    """Persist a decoded video tensor as mp4.

    Expects ``video`` shape ``[*batch, T, C, H, W]`` in ``[-1, 1]``.
    Drops the leading batch axis (size 1), converts to ``[T, H, W, C]``
    uint8 in ``[0, 255]``, and hands the frame list to
    ``diffusers.utils.export_to_video``.
    """
    import numpy as np
    from diffusers.utils import export_to_video

    if video.dim() > 4:
        # Squeeze leading batch axes one at a time (asserting size 1) so the
        # error message is precise if a future batch>1 config sneaks through.
        while video.dim() > 4:
            assert video.shape[0] == 1, (
                f"_write_mp4 expects batch_size=1; got leading shape {video.shape[0]}."
            )
            video = video.squeeze(0)
    # video is now [T, C, H, W] in [-1, 1].
    arr = ((video.clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)
    arr_thwc = arr.permute(0, 2, 3, 1).cpu().numpy()  # [T, H, W, C]
    frames: list[np.ndarray] = list(arr_thwc)
    export_to_video(frames, str(out_path), fps=fps)

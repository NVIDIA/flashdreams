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
and reconstituted-context memory-frame **selection** (2b.5a) activates
when :attr:`HyWorldPlayWanI2VRunnerConfig.use_memory_selection` is set
on top of camera conditioning -- this binds the FOV-overlap selection
policy on the encoder so each AR step's
:class:`hy_worldplay._action.HyWorldPlayCtrl` carries the historical
``memory_frame_indices`` the future KV-prefill executor (2b.5b) will
consume. The phase-1 vendor wrapper in
:class:`hy_worldplay.runner.HyWorldPlayWanI2VRunner` stays as the
default; this module's runner is selected by setting
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
    - **Reconstituted-context memory -- selection only** (2b.5a, landed).
      When ``use_memory_selection=True`` (requires
      ``use_camera_conditioning=True``), this runner builds the
      Monte-Carlo FOV sphere on the pipeline device and arms the
      encoder so each AR step that has enough history attaches a
      sorted ``memory_frame_indices`` list to its
      :class:`HyWorldPlayCtrl`. The matching prefill executor +
      cache infrastructure landed in 2b.5b-part2 (see below).
    - **Distilled-checkpoint weight remap** (2b.5b-part1, landed).
      When ``--ckpt-path`` is supplied alongside any conditioner
      flag, the runner config's ``__post_init__`` reroutes the
      transformer's ``checkpoint_path`` to the upstream distilled
      ``.pt`` and swaps in
      :func:`hy_worldplay._checkpoint.hy_worldplay_distilled_state_dict_transform`
      for the load. The action MLP and per-block PRoPE output
      projection move from zero-init to the distilled weights, so
      the conditioner residuals stop being strict identities; the
      noise prediction now reflects the trained conditioner
      contributions on top of the base 5B trunk.
    - **Reconstituted-context memory -- KV-prefill executor**
      (2b.5b-part2, landed). The new
      :class:`HyWorldPlayWan21TransformerCache` carries a per-rollout
      ``clean_latent_history`` buffer (appended via the
      ``finalize_kv_cache`` override that supersedes the parent's
      rolling-window stamp). Each
      :class:`HyWorldPlayPRoPEBlockCache` gains a
      :class:`HyWorldPlayMemoryKVCache` slot that stores prefilled
      K / V at upstream's RoPE-collapsed positions ``[0, K)`` for
      both the standard and PRoPE branches. At AR step 0 of every
      chunk past the first,
      :meth:`HyWorldPlayWan21Transformer.predict_flow` dispatches
      to the new ``prefill_memory_kv_cache`` driver, which slices
      the history at the encoder-supplied ``memory_frame_indices``,
      builds RoPE freqs at the collapsed positions, and runs
      :meth:`HyWorldPlayWanDiTNetwork.prefill_memory_kv_cache`
      (a patchify + AdaLN re-pass over the memory frames that
      writes each block's K / V into its memory cache and skips
      cross-attn / FFN / head). Subsequent denoising steps in the
      chunk read the prefilled K / V via the dual-branch attention's
      new ``cat([memory_K, current_K], dim=seq)`` prepend, with a
      strict no-op short-circuit on the empty-cache path so chunk 0
      stays bit-identical to the 2b.4 baseline.
    - **Reconstituted-context memory -- per-rollout metadata threading**
      (2b.5b-part2-followup, landed). The encoder
      (:meth:`HyWorldPlayWanCtrlEncoder.forward`) now attaches the
      full-trajectory ``viewmats`` / ``Ks`` / ``action`` tensors to
      every per-AR-step :class:`HyWorldPlayCtrl` via
      ``rollout_viewmats`` / ``rollout_Ks`` / ``rollout_action``,
      alongside the existing per-step slices. The prefill driver
      replaces the parity-incorrect ``_slice_per_frame`` stub with
      :meth:`HyWorldPlayWan21Transformer._index_rollout_buffer`,
      which uses ``tensor.index_select(axis, memory_frame_indices)``
      on the per-rollout buffer to hand the executor camera + action
      data for the *historical* frames it's prefilling rather than
      the current chunk's slice. The remaining 2b.5b-part2-followup
      items (GPU smoke + parity diff + sub-venv cleanup + default
      flag flip) require real-checkpoint GPU validation that surfaces
      structural bugs the CPU tests can't catch (fused RoPE kernel
      dtype, CP wiring, dtype promotion through the prefill).
    """

    def run(self) -> None:
        """Roll one autoregressive sequence and persist the mp4 on rank 0."""
        import os

        from loguru import logger

        cfg = self.config
        # Phase 2b.6.2 diagnostic toggle. ``HY_DEBUG_DISABLE_CUDA_GRAPH=1``
        # disables the per-network CUDAGraphWrapper so the env-var-gated
        # tensor dumps in :mod:`_debug_dump` (which do file I/O + host-
        # synchronous ``.item()`` calls) don't crash CUDA stream capture
        # with ``cudaErrorStreamCaptureInvalidated``. Production runs leave
        # the env var unset so the CUDA graph fast path stays intact.
        if os.environ.get("HY_DEBUG_DISABLE_CUDA_GRAPH", "") == "1":
            # ``self.pipeline.diffusion_model`` is the
            # :class:`DiffusionModel` wrapper (scheduler + transformer);
            # the CUDA-graph state lives on the inner ``.transformer``
            # (Wan21Transformer), not on the wrapper. Reach through one
            # level so the disable actually takes effect. The previous
            # build set the attributes on the wrapper, which silently
            # no-op'd because ``DiffusionModel`` has no
            # ``_use_cuda_graph`` field -- the capture proceeded for
            # AR steps >= ``_cuda_graph_capture_ar_idx`` (== 1 for
            # ``len_t == window_size_t == 4``) and the dump harness'
            # ``is_current_stream_capturing()`` guard suppressed every
            # per-block dump from inside the captured region (chunk-1
            # steps 2 / 3 silently lost their ``attn.x_in`` records).
            diffusion_model = getattr(self.pipeline, "diffusion_model", None)
            wan_transformer = getattr(diffusion_model, "transformer", None)
            if wan_transformer is not None and hasattr(wan_transformer, "network"):
                wan_transformer._use_cuda_graph = False
                wan_transformer._network_call = wan_transformer.network
                wan_transformer._network_call_uncond = wan_transformer.network
                logger.info(
                    "HY_DEBUG_DISABLE_CUDA_GRAPH=1: bypassing the per-network "
                    "CUDAGraphWrapper for diagnostic dumps."
                )

        # Phase 2b.6.2 parity toggle. ``HY_VENDOR_NOISE_MODE=1`` pre-draws
        # one big ``randn([1, 48, num_chunk*len_t, H_lat, W_lat])`` tensor
        # with the rollout seed (matching vendor's ``prepare_latents`` call
        # in ``wan/inference/pipeline_wan_w_mem_relative_rope.py`` which
        # samples all chunks' noise in a single ``randn`` and slices per
        # chunk thereafter) and patchifies a slice per AR step so native's
        # ``DiffusionModel.generate`` consumes bit-identical noise to
        # vendor's ``latents[:, :, ar*len_t:(ar+1)*len_t]`` slice. Without
        # this flag native draws ``randn(latent_shape)`` once per chunk via
        # a private ``torch.Generator`` seeded at ``DiffusionModelConfig.seed``
        # (default 42), which is a completely independent RNG stream from
        # vendor's global ``torch.manual_seed(seed)`` stream -- the chunk-1+
        # noise diverges bit-for-bit (e.g. vendor's chunk-1 first 8 values
        # ``[0.875, 0.965, -0.132, -1.602, ...]`` vs native's
        # ``[0.139, -0.108, -0.719, 0.758, ...]``) and the resulting clean
        # latents drift apart, which dominates the parity diff once the
        # CFG mismatch is fixed.

        if cfg.image_path is None:
            raise ValueError(
                "HY-WorldPlay WAN-5B is I2V only -- pass "
                "``--image-path <path-to-jpg>`` to provide the first frame."
            )
        if not cfg.image_path.exists():
            raise FileNotFoundError(f"image_path {cfg.image_path} does not exist")

        first_param = next(self.pipeline.parameters())
        device = first_param.device
        # The VAE encoder runs in the pipeline's parameter dtype (bf16 /
        # fp16 in production, fp32 in the CPU smoke); the float32 tensor
        # produced by ``preprocess_first_frame`` would fail the
        # ``F.conv3d`` dtype check in the residual VAE's first
        # ``CausalConv3d``. Cast here so the cast-once cost stays in
        # the runner rather than the per-AR-step encode path.
        image = preprocess_first_frame(
            cfg.image_path, cfg.pixel_height, cfg.pixel_width
        ).to(device=device, dtype=first_param.dtype)
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
        if cfg.use_memory_selection:
            # ``use_memory_selection`` requires camera conditioning (the
            # runner config's ``__post_init__`` enforces this), so the
            # encoder is guaranteed to have its viewmats bound by this
            # point.
            self._bind_memory_config(device=device)

        vendor_noise_ctx = self._maybe_vendor_aligned_noise_ctx(
            device=device, dtype=first_param.dtype
        )

        chunks: list[Tensor] = []
        start_time = time.time()
        with vendor_noise_ctx:
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
        # ``parse_pose_data`` emits ``[n_latents, 4, 4]`` /
        # ``[n_latents, 3, 3]`` (no batch axis) but
        # :func:`flashdreams.core.attention.prope.prope_qkv` requires
        # ``[batch=1, cameras, 4, 4]``. The ``[..., start:end, :, :]``
        # slice in the encoder's per-AR-step ``forward`` preserves
        # leading dims, so an ``unsqueeze(0)`` here lifts the per-step
        # slice (and the per-rollout buffer threaded into the prefill
        # via ``rollout_viewmats``) to the rank PRoPE expects.
        target_dtype = next(self.pipeline.parameters()).dtype
        encoder.set_camera_data(
            viewmats.to(dtype=target_dtype).unsqueeze(0),
            Ks.to(dtype=target_dtype).unsqueeze(0),
        )

    def _bind_memory_config(self, *, device: torch.device) -> None:
        """Arm reconstituted-context memory selection on the encoder.

        Builds the Monte-Carlo point cloud once (size + radius mirror
        upstream's ``generate_points_in_sphere(50_000, 8.0)`` call in
        ``WanInferencePipeline.__init__``) and hands it + the rest of
        the selection knobs to
        :meth:`HyWorldPlayWanCtrlEncoder.set_memory_config`. The
        encoder then computes the per-AR-step
        ``memory_frame_indices`` on demand inside :meth:`forward`.
        """
        from hy_worldplay._action import HyWorldPlayWanCtrlEncoder
        from hy_worldplay._memory import generate_points_in_sphere

        cfg = self.config
        encoder, _ = self._resolve_encoder_and_n_latents(
            flag_name="use_memory_selection"
        )
        assert isinstance(encoder, HyWorldPlayWanCtrlEncoder)
        points_local = generate_points_in_sphere(
            cfg.memory_points_count,
            cfg.memory_points_radius,
            device=device,
        )
        encoder.set_memory_config(
            points_local=points_local,
            context_window_length=cfg.context_window_length,
            memory_frames=cfg.memory_frames,
            temporal_context_size=cfg.temporal_context_size,
            pred_latent_size=cfg.memory_pred_latent_size,
            fov_h_deg=cfg.memory_fov_h_deg,
            fov_v_deg=cfg.memory_fov_v_deg,
            device=device,
        )

    def _maybe_vendor_aligned_noise_ctx(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> object:
        """Build a context manager that overrides the diffusion noise per chunk.

        When ``HY_VENDOR_NOISE_MODE=1`` is set the runner pre-draws the
        full multi-chunk noise tensor using the same shape and seed as
        vendor's ``prepare_latents`` call (a single ``randn([1, 48, T,
        H_lat, W_lat])`` over all chunks) and patchifies a per-AR-step
        slice. While the chunk loop runs the helper monkey-patches
        ``torch.randn`` to return the pre-computed slice whenever the
        request matches the diffusion model's ``latent_shape``; all
        other randn calls fall through to the original implementation.

        When the env var is unset returns ``nullcontext()`` and the
        pipeline draws noise from its private ``torch.Generator`` as
        usual.
        """
        import contextlib
        import math
        import os
        from unittest.mock import patch as _mock_patch

        from einops import rearrange
        from loguru import logger

        if os.environ.get("HY_VENDOR_NOISE_MODE", "") != "1":
            return contextlib.nullcontext()

        cfg = self.config
        diffusion_model = self.pipeline.diffusion_model
        transformer = diffusion_model.transformer
        transformer_cfg = transformer.config

        len_t = transformer_cfg.len_t
        kt, kh, kw = transformer_cfg.network.patch_size
        # 16x spatial compression for the WAN-5B residual VAE.
        h_lat = cfg.pixel_height // 16
        w_lat = cfg.pixel_width // 16
        # Vendor's prepare_latents draws the full ``num_latent_frames`` of
        # noise in one randn call (see
        # ``HY-WorldPlay/wan/inference/pipeline_wan_w_mem_relative_rope.py``
        # line 313 and the ``num_latent_frames`` derivation a few lines
        # above), independent of how many chunks the rollout actually
        # consumes. To match the RNG stream bit-for-bit we must replicate
        # that full tensor here -- a smaller ``randn(big_t = num_chunk *
        # len_t, ...)`` would put each subsequent channel slice at a
        # different flat-memory offset and the per-chunk noise would no
        # longer line up with vendor's. ``num_latent_frames = (num_frames
        # - 1) // 4 + 1`` for the 4x temporal compression in the WAN-5B
        # residual VAE.
        vendor_full_t = (cfg.num_frames - 1) // 4 + 1
        unpatched_shape = (
            1,
            transformer_cfg.network.in_dim,
            vendor_full_t,
            h_lat,
            w_lat,
        )
        target_shape = tuple(transformer.latent_shape)
        target_numel = math.prod(target_shape)
        # Sanity: the patched per-chunk noise must reshape into the
        # patchified latent shape.
        per_chunk_numel = (
            transformer_cfg.network.in_dim * len_t * h_lat * w_lat
        )
        assert per_chunk_numel == target_numel, (
            f"vendor-aligned noise per-chunk numel ({per_chunk_numel}) "
            f"!= native latent_shape numel ({target_numel}); shapes are "
            f"unpatched={unpatched_shape}, target={target_shape}."
        )

        seed = cfg.seed
        if cfg.offset_seed_by_global_rank and self.global_rank != 0:
            seed = seed + self.global_rank

        # Draw the full noise tensor in fp32 to mirror vendor's
        # ``randn_tensor(..., dtype=torch.float32)`` call site, then cast
        # to the diffusion model's dtype. ``torch.manual_seed`` seeds the
        # global RNG which matches vendor's ``generate.py``'s
        # ``torch.manual_seed(seed)`` call at the top of ``predict``.
        torch.manual_seed(seed)
        big_noise_fp32 = torch.randn(
            unpatched_shape,
            dtype=torch.float32,
            device=device,
        )

        # Patchify per chunk to match the format ``DiffusionModel.generate``
        # would otherwise draw directly via ``randn(latent_shape)``. Vendor's
        # transformer applies the same ``... (t kt) c (h kh) (w kw) ->
        # ... (t h w) (c kt kh kw)`` rearrange inside its patch embedding
        # (see ``patchify_and_maybe_split_cp`` in
        # ``flashdreams/recipes/wan/transformer/impl/network.py``); replicating
        # it on the unpatched slice keeps the per-position bit values aligned
        # between native and vendor.
        chunk_noise_queue: list[Tensor] = []
        for ar_idx in range(cfg.num_chunk):
            chunk_slice = big_noise_fp32[
                :, :, ar_idx * len_t : (ar_idx + 1) * len_t, :, :
            ]
            # Permute [B, C, T, H, W] -> [B, T, C, H, W] so the patchify
            # pattern's ``(t kt) c`` axes line up with the input order.
            chunk_slice = chunk_slice.permute(0, 2, 1, 3, 4).contiguous()
            patched = rearrange(
                chunk_slice,
                "b (t kt) c (h kh) (w kw) -> b (t h w) (c kt kh kw)",
                kt=kt,
                kh=kh,
                kw=kw,
            )
            # Drop the batch axis when the transformer's batch_shape is
            # empty; native's ``latent_shape = (L, D)`` in that case and
            # the reshape below would otherwise complain.
            if not transformer_cfg.batch_shape:
                patched = patched.squeeze(0)
            chunk_noise_queue.append(patched.to(dtype=dtype))

        orig_randn = torch.randn

        def patched_randn(*args: object, **kwargs: object) -> Tensor:
            shape_arg: tuple[int, ...] | None = None
            if args:
                first = args[0]
                if isinstance(first, (tuple, list, torch.Size)):
                    shape_arg = tuple(int(x) for x in first)
                elif isinstance(first, int):
                    shape_arg = tuple(int(x) for x in args)
            else:
                size = kwargs.get("size", None)
                if isinstance(size, (tuple, list, torch.Size)):
                    shape_arg = tuple(int(x) for x in size)

            if (
                shape_arg is not None
                and shape_arg == target_shape
                and chunk_noise_queue
            ):
                noise = chunk_noise_queue.pop(0)
                kwarg_device = kwargs.get("device", noise.device)
                kwarg_dtype = kwargs.get("dtype", noise.dtype)
                return noise.to(device=kwarg_device, dtype=kwarg_dtype)
            return orig_randn(*args, **kwargs)

        logger.info(
            f"HY_VENDOR_NOISE_MODE=1: pre-drew "
            f"randn({list(unpatched_shape)}) at seed={seed} and queued "
            f"{cfg.num_chunk} patchified per-chunk slices matching "
            f"latent_shape={target_shape}."
        )
        return _mock_patch.object(torch, "randn", patched_randn)

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
    float32 in ``[0, 1]``, and hands the frame list to
    ``diffusers.utils.export_to_video``.

    Note: ``export_to_video`` interprets numpy ``ndarray`` frames as
    floats in ``[0, 1]`` and internally multiplies by 255 before
    ``.astype(np.uint8)`` (see diffusers' implementation). Passing
    uint8 ``[0, 255]`` arrays here would overflow that multiply and
    produce visibly shifted RGB means (~40 units per channel for typical
    pixel values), which is what caused the long-running 2b.6
    parity-divergence symptom. Keep frames in ``[0, 1]`` float to match
    diffusers' contract.
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
    # video is now [T, C, H, W] in [-1, 1]. Map to [0, 1] float32 to
    # match diffusers' ``export_to_video`` contract for ndarray frames.
    arr = ((video.clamp(-1.0, 1.0) + 1.0) * 0.5).to(torch.float32)
    arr_thwc = arr.permute(0, 2, 3, 1).cpu().numpy()  # [T, H, W, C]
    frames: list[np.ndarray] = list(arr_thwc)
    export_to_video(frames, str(out_path), fps=fps)

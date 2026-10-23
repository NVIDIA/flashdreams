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

"""Joint RGB/LiDAR generation on the camera-anchored source clock."""

from __future__ import annotations

import math
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import replace

import torch
from cosmos_interactive._compile_modes import torch_compile_mode_arg
from cosmos_interactive._sampler import fixed_step_denoise
from cosmos_interactive.lidar import LidarTokenizer
from torch import Tensor

from cosmos3_nano_sim_auto.impl.conditioning.control_provider import EncodedControls
from cosmos3_nano_sim_auto.impl.packing import (
    ClipGeometry,
    _stamp_view_major,
    _vision,
    ar_chunk_range,
    concat,
    text_stream,
)
from cosmos3_nano_sim_auto.impl.rollout import (
    COMPILE_DENOISE_MODES,
    SDE_CHUNK_STRIDE,
    SDE_STEP_STRIDE,
    ChunkTrace,
    _to_latent,
)
from cosmos3_nano_sim_auto.impl.transformer.embedding import (
    _timestep_embedding,
    text_embeddings,
)
from cosmos3_nano_sim_auto.impl.transformer.fold import build_fold_plan
from cosmos3_nano_sim_auto.impl.transformer.forward import (
    run_chunk_layers,
    run_text_layers,
)
from cosmos3_nano_sim_auto.impl.transformer.mask import (
    LIDAR_VIEW_ID,
    ROLE_CLEAN_TARGET,
    ROLE_CONTROL,
    ROLE_CURRENT_TARGET,
    ROLE_TARGET_CONDITION,
    StreamFields,
)
from cosmos3_nano_sim_auto.impl.transformer.rope import (
    caption_mrope_ids,
    chunk_mrope_ids,
    vision_temporal_offset,
)
from cosmos3_nano_sim_auto.inputs import control_tokens as rgb_tokens
from flashdreams.core.distributed.context_parallel import build_shard, gather_tokens
from flashdreams.core.distributed.parallel import ParallelContext


@contextmanager
def stage_timer(device, label=None):
    """Attribute each stage after GPU completion, including host submission time."""
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.synchronize(device)
        if label is not None:
            torch.cuda.nvtx.range_push(label)
    started = time.perf_counter()
    elapsed = [0.0]
    try:
        yield lambda: elapsed[0]
    finally:
        if cuda:
            try:
                torch.cuda.synchronize(device)
            finally:
                if label is not None:
                    torch.cuda.nvtx.range_pop()
        elapsed[0] = time.perf_counter() - started


@contextmanager
def stage_events(device, label=None):
    """Time a stage with CUDA events, without synchronizing around it.

    The returned callable waits for the stage's end event only when read, so a
    step's stages no longer stall the host between them; reading after the
    caller's own end-of-chunk synchronize costs nothing. CPU falls back to
    :func:`stage_timer`.
    """
    if device.type != "cuda":
        with stage_timer(device, label) as elapsed:
            yield elapsed
        return
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    stream = torch.cuda.current_stream(device)
    if label is not None:
        torch.cuda.nvtx.range_push(label)
    start.record(stream)

    def elapsed() -> float:
        end.synchronize()
        return start.elapsed_time(end) / 1000

    try:
        yield elapsed
    finally:
        end.record(stream)
        if label is not None:
            torch.cuda.nvtx.range_pop()


def sweep_range(
    start: int, end: int, *, geometry: ClipGeometry, lidar_fps: float
) -> tuple[int, int]:
    """Select 10 Hz sweeps in the camera chunk's singleton-zero partition."""
    if not 0 <= start < end:
        raise ValueError("Expected a nonempty camera frame range")
    scale = geometry.seconds_per_frame * lidar_fps
    first = 0 if start == 0 else math.floor((start - 1) * scale + 1e-5) + 1
    return first, math.floor((end - 1) * scale + 1e-5) + 1


class JointChunkRollout:
    """Denoise both sensors together, retaining a sink plus completed joint chunks.

    The existing MoT layer stack, tensor/context mesh, sampler and maskless folds
    are shared with RGB inference. LiDAR controls implement the same ``tokens``
    provider contract, indexed in sweeps rather than camera latent frames.
    """

    def __init__(
        self,
        network,
        *,
        geometry: ClipGeometry,
        control_tokens,
        lidar_controls,
        lidar_tokenizer: LidarTokenizer,
        text_ids: Tensor,
        view_text_tokens=None,
        condition_tokens=None,
        lidar_condition=None,
        fps=30.0,
        seed=0,
        t_list=None,
        history_frames=3,
        parallel=None,
        attention_impl=None,
        accelerated_rms_norm=False,
        decomposed_temporal_window_seconds=0.4,
        fused_attention_merge=False,
        compile_denoise=None,
    ):
        if not hasattr(network, "lidar2llm") or not hasattr(network, "llm2lidar"):
            raise ValueError(
                "Checkpoint has no LiDAR adapters; re-export the joint student with --force --no-prefer-ema"
            )
        if network.lidar2llm.in_features != lidar_tokenizer.geometry.patch_dim:
            raise ValueError("Checkpoint LiDAR patch width disagrees with tokenizer")
        self._deduplicate_cross_view = hasattr(network, "rig_view_embed")
        if self._deduplicate_cross_view and geometry.rig_view_ids is None:
            raise ValueError("This checkpoint requires canonical camera rig_view_ids")
        geometry.latent_hw(network.config.spec.vfm.latent_patch_size)
        if geometry.condition_frames not in (0, 1):
            raise ValueError(
                "Joint generation supports zero or one initial camera latent"
            )
        if geometry.condition_frames and condition_tokens is None:
            raise ValueError("Initial RGB tokens are required")
        if history_frames is None or history_frames < 1:
            raise ValueError("Joint generation requires a bounded positive history")
        if (
            decomposed_temporal_window_seconds is None
            or decomposed_temporal_window_seconds <= 0
        ):
            raise ValueError(
                "Joint generation requires an explicit positive temporal window"
            )
        if network.config.spec.vfm.latent_patch_size != 2:
            raise ValueError("Joint inference requires 2x2 RGB patches")
        if compile_denoise is not None and compile_denoise not in COMPILE_DENOISE_MODES:
            raise ValueError(
                f"compile_denoise must be one of {COMPILE_DENOISE_MODES} or None"
            )
        # The same refusal as the RGB rollout: the tensor axis all-reduces in
        # place inside the compiled region.
        if (
            compile_denoise is not None
            and parallel is not None
            and parallel.tensor_parallel
        ):
            raise ValueError(
                "compile_denoise and tensor parallelism are not compatible; set "
                "COSMOS3_NANO_SIM_AUTO_TENSOR_PARALLEL=1 or run without compilation"
            )
        if (
            not math.isfinite(fps)
            or fps <= 0
            or not math.isclose(geometry.seconds_per_frame * fps, 4.0)
        ):
            raise ValueError(
                "Camera latent periods must match the 4x compressed source clock"
            )
        if geometry.condition_frames == 0 and condition_tokens is not None:
            raise ValueError(
                "RGB conditioning tokens supplied without a conditioned frame"
            )
        if condition_tokens is not None and condition_tokens.shape != (
            geometry.tokens_per_frame,
            network.config.spec.vfm.patch_latent_dim,
        ):
            raise ValueError("Initial RGB tokens disagree with the camera geometry")
        if lidar_condition is not None and lidar_condition.shape != (
            lidar_tokenizer.geometry.spatial_tokens,
            lidar_tokenizer.geometry.patch_dim,
        ):
            raise ValueError("Initial LiDAR tokens must describe exactly one sweep")
        self.network, self.geometry = network, geometry
        self.tokenizer, self.lidar_controls = lidar_tokenizer, lidar_controls
        self.device = network.embed_tokens.weight.device
        self.ctx = parallel or ParallelContext.single(self.device)
        self.controls = (
            EncodedControls(control_tokens, geometry)
            if isinstance(control_tokens, Tensor)
            else control_tokens
        )
        self.fps, self.seed = fps, seed
        self.schedule = list(t_list or network.config.spec.vfm.fixed_step_t_list)
        self.impl, self.accelerated = attention_impl, accelerated_rms_norm
        self.window, self.fused = (
            decomposed_temporal_window_seconds,
            fused_attention_merge,
        )
        self._camera_ids = (
            torch.tensor(geometry.rig_view_ids, device=self.device)
            if self._deduplicate_cross_view
            else None
        )

        # The noisy denoise steps and the K/V-collecting control and clean passes
        # compile when asked, but only run compiled in the steady state (see
        # `step`): every warmup shape would be a graph of its own, and enough of
        # them exhaust Dynamo's recompile limit, after which it silently runs
        # eager for the rest of the drive.
        def compiled(fn):
            if compile_denoise is None:
                return fn
            return torch.compile(fn, mode=torch_compile_mode_arg(compile_denoise))

        self._noisy, self._collector = compiled(self._forward), compiled(self._collect)
        self._steady = False
        self.condition = condition_tokens
        self.lidar_condition = lidar_condition
        self._frame, self.chunks_done, self._stopped = 0, 0, False
        self.warmup_chunks = math.ceil(history_frames / geometry.frames_per_chunk) + 1
        self.history = deque(maxlen=self.warmup_chunks - 1)
        self.sink = None
        self.last_rgb = None
        self.last_lidar = None
        self.last_lidar_range = (0, 0)
        self._metrics = {}
        self._last_start = 0
        self.text_fields = text_stream(
            len(text_ids), self.device, view_text_tokens=view_text_tokens
        )
        if view_text_tokens is None:
            positions = torch.arange(len(text_ids)).float().expand(3, -1)
            extent = len(text_ids)
        else:
            positions, extent = caption_mrope_ids(view_text_tokens)
        self.offset = vision_temporal_offset(extent)
        with torch.inference_mode():
            _, self.text_kv = run_text_layers(
                network.layers,
                network.config.spec.llm,
                text_embeddings(network, text_ids),
                positions,
                view_text_tokens=view_text_tokens,
                native_gqa_attention=True,
            )
        self.empty_kv = [(k[:, :, :0], v[:, :, :0]) for k, v in self.text_kv]
        self.empty_fields = replace(
            self.text_fields,
            **{key: value[:0] for key, value in vars(self.text_fields).items()},
        )
        # Recorded LiDAR controls are known in full, so a provider that can is
        # asked to encode them ahead rather than inside every chunk.
        preencode = getattr(lidar_controls, "preencode", None)
        if preencode is not None:
            preencode(self._sweep_ranges())

    def _sweep_ranges(self):
        """The LiDAR sweep range of every chunk, in order; endless for an open clip."""
        frame = 0
        while (bounds := ar_chunk_range(self.geometry, frame)) is not None:
            yield sweep_range(
                *bounds, geometry=self.geometry, lidar_fps=self.tokenizer.geometry.fps
            )
            frame = bounds[1]

    @property
    def last_metrics(self):
        """The last step's stage seconds (waiting for its GPU work) and counts."""
        return {
            key: value() if callable(value) else value
            for key, value in self._metrics.items()
        }

    @property
    def chunk_count(self):
        n = self.geometry.frames_per_view
        return (
            None
            if n is None
            else 1 + math.ceil((n - 1) / self.geometry.frames_per_chunk)
        )

    @property
    def next_range(self):
        return None if self._stopped else ar_chunk_range(self.geometry, self._frame)

    @property
    def is_finished(self):
        return self.next_range is None

    def stop(self):
        self._stopped = True

    def _fields(self, start, end, *, lidar, role):
        if not lidar:
            frame, view = _stamp_view_major(
                torch.arange(start, end, device=self.device),
                self.geometry.num_views,
                self.geometry.spatial_tokens,
            )
            return _vision(
                frame,
                view,
                role=role,
                is_noisy=role == ROLE_CURRENT_TARGET,
                geometry=self.geometry,
            )
        g = self.tokenizer.geometry
        frame = torch.arange(start, end, device=self.device).repeat_interleave(
            g.spatial_tokens
        )
        timestamp = frame.float() / g.fps
        step = torch.ceil(
            timestamp
            / (self.geometry.frames_per_chunk * self.geometry.seconds_per_frame)
            - 1e-5
        ).long()
        return StreamFields(
            torch.zeros_like(frame),
            frame,
            torch.full_like(frame, LIDAR_VIEW_ID),
            torch.full_like(frame, role == ROLE_CURRENT_TARGET, dtype=torch.bool),
            torch.full_like(frame, role == ROLE_CONTROL, dtype=torch.bool),
            timestamp,
            torch.full_like(frame, role),
            step,
        )

    def _positions(self, start, end, *, lidar):
        if not lidar:
            return chunk_mrope_ids(
                self.geometry,
                chunk_start=start,
                chunk_frames=end - start,
                fps=self.fps,
                temporal_offset=self.offset,
            ).to(self.device)
        g = self.tokenizer.geometry
        h, w = g.grid
        t = (
            torch.arange(start, end, device=self.device).float() / g.fps * 30 / 4
            + self.offset
        )
        return torch.stack(
            (
                t.repeat_interleave(h * w),
                torch.arange(h, device=self.device)
                .repeat_interleave(w)
                .repeat(end - start)
                .float(),
                torch.arange(w, device=self.device).repeat((end - start) * h).float(),
            )
        )

    def _memory(self, extra=()):
        entries = (
            ([] if self.sink is None else [self.sink])
            + list(self.history)
            + list(extra)
        )
        if not entries:
            return self.empty_fields, self.empty_kv
        fields = concat(*(item[0] for item in entries))
        # ponytail: concatenate a bounded history per pass; a joint preallocated
        # arena can remove these copies after the multimodal path is profiled.
        kv = [
            (
                torch.cat([item[1][i][0] for item in entries], 2),
                torch.cat([item[1][i][1] for item in entries], 2),
            )
            for i in range(len(self.text_kv))
        ]
        return fields, kv

    def _plan(self, rgb_fields, lidar_fields, memory):
        """Fields, fold plan, CP shard and rig rows for one pass.

        Built once per pass kind and reused by every denoise step.
        """
        fields = concat(rgb_fields, lidar_fields)
        keys = concat(self.text_fields, fields, memory[0])
        query = fields
        shard = None
        if self.ctx.context_parallel:
            shard = build_shard(len(fields), self.ctx)
            query = fields.narrow(*shard.local)
        plan = build_fold_plan(
            query,
            keys,
            scope="decomposed",
            decomposed_temporal_window_seconds=self.window,
            joint=True,
            deduplicate_cross_view=self._deduplicate_cross_view,
        )
        view_ids = None
        if self._camera_ids is not None:
            n_rgb = len(rgb_fields)
            view_ids = torch.cat(
                (
                    self._camera_ids[fields.view_id[:n_rgb]],
                    torch.full_like(fields.view_id[n_rgb:], 11),
                )
            )
        return fields, replace(plan, fused_merge=self.fused), shard, view_ids

    def _forward(
        self, rgb, lidar, positions, view_ids, is_noisy, memory_kv, mask, timestep
    ):
        """The network over one noisy pass: tensors and a fold plan in, flows out.

        Everything that varies between passes is an argument, so one compiled
        graph serves every chunk of the same shape.
        """
        n_rgb, n_lidar = len(rgb), len(lidar)
        hidden = self._embed(rgb, lidar, view_ids)
        hidden = (
            hidden
            + _timestep_embedding(self.network, timestep, len(hidden))
            * is_noisy[:, None]
        )
        output, _ = run_chunk_layers(
            self.network.layers,
            self.network.config.spec.llm,
            hidden[None],
            positions,
            text_kv=self.text_kv,
            memory_kv=memory_kv,
            mask=mask,
            final_norm=self.network.norm_moe_gen,
            attention_implementation=self.impl,
            native_gqa_attention=True,
            accelerated_rms_norm=self.accelerated,
            collect_kv=False,
            ctx=self.ctx,
        )
        if self.ctx.context_parallel:
            output = gather_tokens(
                output, build_shard(len(hidden), self.ctx), self.ctx, dim=1
            )
        return self.network.llm2vae(output[0, :n_rgb]), self.network.llm2lidar(
            output[0, n_rgb : n_rgb + n_lidar]
        )

    def _collect(self, rgb, lidar, positions, view_ids, memory_kv, mask):
        """The network over a control or clean pass, returning its keys/values."""
        _, kv = run_chunk_layers(
            self.network.layers,
            self.network.config.spec.llm,
            self._embed(rgb, lidar, view_ids)[None],
            positions,
            text_kv=self.text_kv,
            memory_kv=memory_kv,
            mask=mask,
            final_norm=self.network.norm_moe_gen,
            attention_implementation=self.impl,
            native_gqa_attention=True,
            accelerated_rms_norm=self.accelerated,
            collect_kv=True,
            ctx=self.ctx,
        )
        return kv

    def _embed(self, rgb, lidar, view_ids):
        hidden = torch.cat(
            (
                self.network.vae2llm(rgb.to(self.network.config.dtype)),
                self.network.lidar2llm(lidar.to(self.network.config.dtype)),
            )
        )
        if view_ids is not None:
            hidden = hidden + self.network.rig_view_embed(view_ids).to(hidden.dtype)
        return hidden

    def _pass(
        self,
        rgb,
        lidar,
        planned,
        rgb_pos,
        lidar_pos,
        memory,
        *,
        timestep=None,
        collect=False,
    ):
        fields, plan, _shard, view_ids = planned
        positions = torch.cat((rgb_pos, lidar_pos), 1)
        _, mem_kv = memory
        if not collect:
            assert timestep is not None, "only denoise passes skip K/V collection"
            return (self._noisy if self._steady else self._forward)(
                rgb,
                lidar,
                positions,
                view_ids,
                fields.is_noisy,
                mem_kv,
                plan,
                torch.as_tensor(timestep, device=self.device, dtype=torch.float32),
            )
        assert timestep is None, "control and clean passes carry no timestep"
        collector = self._collector if self._steady else self._collect
        return (fields, collector(rgb, lidar, positions, view_ids, mem_kv, plan))

    @torch.inference_mode()
    def step(self):
        bounds = self.next_range
        if bounds is None:
            raise RuntimeError("Joint rollout is complete")
        start, end = bounds
        # Compiled only for a full chunk against a full history; the opening
        # chunk, the chunks filling the history and a short final one run eager.
        self._steady = (
            end - start == self.geometry.frames_per_chunk
            and len(self.history) == self.history.maxlen
        )
        lo, hi = sweep_range(
            start, end, geometry=self.geometry, lidar_fps=self.tokenizer.geometry.fps
        )
        if hi <= lo:
            raise ValueError(
                "Camera chunk contains no LiDAR sweeps; use three RGB latents per chunk"
            )
        g, lg = self.geometry, self.tokenizer.geometry
        rp, lp = (
            self._positions(start, end, lidar=False),
            self._positions(lo, hi, lidar=True),
        )

        def rf(role):
            return self._fields(start, end, lidar=False, role=role)

        def lf(role):
            return self._fields(lo, hi, lidar=True, role=role)

        # Start the RGB map worker (if any) on this chunk, then the LiDAR ranks'
        # render/encode, both before the camera collectives so they overlap.
        for provider, bounds in (
            (self.controls, (start, end)),
            (self.lidar_controls, (lo, hi)),
        ):
            prefetch = getattr(provider, "prefetch", None)
            if prefetch is not None:
                prefetch(*bounds)
        with stage_events(self.device, "rgb/control_encode") as elapsed:
            controls_rgb = self.controls.tokens(start, end).to(self.device)
        metrics = {"rgb_control_encode_s": elapsed}
        # As in ChunkRollout: a provider that can draw ahead starts the next
        # chunk's controls now and has the whole denoise to finish them.
        ahead = ar_chunk_range(g, end)
        prefetch = getattr(self.controls, "prefetch", None)
        if ahead is not None and prefetch is not None:
            prefetch(*ahead)
        with stage_events(self.device, "lidar/encode") as elapsed:
            controls_lidar = self.lidar_controls.tokens(lo, hi).to(self.device)
        metrics["lidar_encode_s"] = elapsed
        past = self._memory()
        with stage_events(self.device, "joint/control") as elapsed:
            control = self._pass(
                controls_rgb,
                controls_lidar,
                self._plan(rf(ROLE_CONTROL), lf(ROLE_CONTROL), past),
                rp,
                lp,
                past,
                collect=True,
            )
        metrics["joint_control_s"] = elapsed
        del past  # Release the assembled history before allocating current memory.
        memory = self._memory((control,))
        memory_tokens = len(memory[0])
        rgb_conditioned = start == 0 and g.condition_frames == 1
        lidar_conditioned = lo == 0 and self.lidar_condition is not None
        rgb_shape = (
            g.num_views,
            self.network.config.spec.vfm.vae_latent_channels,
            end - start,
            *g.latent_hw(2),
        )
        lidar_shape = (1, lg.channels, hi - lo, lg.height, lg.width)
        nr = 0 if rgb_conditioned else math.prod(rgb_shape)
        nl = 0 if lidar_conditioned else math.prod(lidar_shape)

        def patches(x):
            if rgb_conditioned:
                assert self.condition is not None
                rgb = self.condition.to(self.device)
            else:
                rgb = rgb_tokens(
                    x[:nr]
                    .reshape(rgb_shape[1], rgb_shape[0], *rgb_shape[2:])
                    .transpose(0, 1),
                    latent_patch_size=2,
                )
            if lidar_conditioned:
                assert self.lidar_condition is not None
                lidar = self.lidar_condition.to(self.device)
            else:
                lidar = self.tokenizer.patchify(x[nr:].reshape(lidar_shape))
            return rgb, lidar

        noisy_rf = rf(ROLE_TARGET_CONDITION if rgb_conditioned else ROLE_CURRENT_TARGET)
        noisy_lf = lf(
            ROLE_TARGET_CONDITION if lidar_conditioned else ROLE_CURRENT_TARGET
        )
        # One plan for every denoise step, as the RGB rollout builds its noisy mask.
        noisy = self._plan(noisy_rf, noisy_lf, memory)

        def flow(x, timestep):
            rgb, lidar = patches(x)
            vr, vl = self._pass(rgb, lidar, noisy, rp, lp, memory, timestep=timestep)
            outputs = []
            if nr:
                outputs.append(
                    _to_latent(
                        vr.reshape(g.num_views, end - start, g.spatial_tokens, -1),
                        g,
                        channels=rgb_shape[1],
                        patch=2,
                    )
                    .transpose(0, 1)
                    .flatten()
                )
            if nl:
                outputs.append(self.tokenizer.unpatchify(vl).flatten())
            return torch.cat(outputs)

        def noise(seed):
            return torch.randn(
                nr + nl,
                generator=torch.Generator(device=self.device).manual_seed(seed),
                device=self.device,
            )

        with stage_events(self.device, "joint/denoise") as elapsed:
            if nr + nl:
                denoised = fixed_step_denoise(
                    flow,
                    noise(self.seed + self.chunks_done),
                    t_list=self.schedule,
                    sde_noise_fn=lambda i: noise(
                        self.seed
                        + self.chunks_done * SDE_CHUNK_STRIDE
                        + i * SDE_STEP_STRIDE
                    ),
                    fp32_math=True,
                )
            else:
                denoised = torch.empty(0, device=self.device)
            rgb, lidar = patches(denoised)
        metrics["joint_denoise_s"] = elapsed
        self.last_lidar = self.tokenizer.unpatchify(lidar)
        # A decoder with a worker starts on these sweeps now and has the
        # clean-cache pass to finish them, rather than following it on rank 0.
        decode_ahead = getattr(self.lidar_controls, "decode_ahead", None)
        if decode_ahead is not None and self.ctx.is_main:
            decode_ahead(self.last_lidar)
        with stage_events(self.device, "joint/clean_cache") as elapsed:
            clean = self._pass(
                rgb,
                lidar,
                self._plan(
                    rf(ROLE_TARGET_CONDITION if rgb_conditioned else ROLE_CLEAN_TARGET),
                    lf(
                        ROLE_TARGET_CONDITION
                        if lidar_conditioned
                        else ROLE_CLEAN_TARGET
                    ),
                    memory,
                ),
                rp,
                lp,
                memory,
                collect=True,
            )
            memory = noisy = None  # Release attention input also captured by flow().
            combined = (
                concat(control[0], clean[0]),
                [
                    (torch.cat((ck, tk), 2), torch.cat((cv, tv), 2))
                    for (ck, cv), (tk, tv) in zip(control[1], clean[1], strict=True)
                ],
            )
            if start == 0:
                self.sink = combined
            else:
                self.history.append(combined)
        metrics["joint_clean_cache_s"] = elapsed
        self.last_rgb = rgb.reshape(g.num_views, end - start, g.spatial_tokens, -1)
        self.last_lidar_range = (lo, hi)
        metrics.update(
            lidar_sweeps=hi - lo, lidar_tokens=len(lidar), rgb_tokens=len(rgb)
        )
        self._metrics = metrics
        self._last_start, self._frame = start, end
        self.chunks_done += 1
        return ChunkTrace(
            start, end, len(self.schedule) if nr + nl else 0, True, memory_tokens
        ), self.last_rgb

    def latent_for(self, start, end):
        if start != self._last_start or end != self._frame or self.last_rgb is None:
            raise ValueError("Decode each joint chunk before requesting the next")
        return _to_latent(
            self.last_rgb,
            self.geometry,
            channels=self.network.config.spec.vfm.vae_latent_channels,
            patch=2,
        )

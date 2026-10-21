# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Our rollout behind Interactive Drive's render backend.

The app drives and we draw. It reads one command a chunk, integrates its own
vehicle model across the whole chunk and hands over the rig's poses; our rollout
pulls a frame span whenever it needs a chunk's map drawn. The two meet at
:class:`~cosmos3_nano_sim_auto.apps.interactive_drive.poses.SuppliedPoses`, and they
line up because the app sizes its chunk from what this backend reports.

The run has no end, which is what the app wants and what the rollout already
supports: ``frames_per_view=None`` with the cache bounded by ``history_frames``
and control topped up a chunk at a time. Nothing here is a special case for
being live.

The app's other seam, :class:`~interactive_drive.video_model.chunk_pipeline.ChunkPipeline`,
stays unused, and under a mesh it has to. It generates on a worker thread so a
chunk can be in flight while the last one presents, which is right for a model
on one card and wrong for one spread over several: the collectives inside a
step need every rank to arrive at them in the same order, and the runtime gets
that by admitting steps in lockstep. A thread that decides for itself when to
generate takes that guarantee away. A render backend is called from the step,
so lockstep is the default rather than something to arrange.

On the RGB-only path, the chunk sizes do not line the
two up. The rollout draws its whole control window while it is being built,
which is several chunks of map before the first chunk is generated and before
the app has sampled any trajectory at all. So the opening is drawn along the
path the scene was already on, the app's first commands land on ground whose map
is drawn, and steering reaches the picture a window later. That is the demo's
behaviour, and undoing it means teaching the rollout to leave control slots
unfilled rather than anything here. Joint RGB/LiDAR instead consumes only the
current control chunk: an opening frame, then twelve camera frames with four
completed LiDAR sweeps. Both renderers receive the same supplied vehicle path.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.distributed as dist
from interactive_drive.backends.base import RenderBackend
from interactive_drive.config import ChunkConfig, RasterConfig
from interactive_drive.types import (
    FrameChunk,
    PresentedFrame,
    SceneBundle,
    TrajectoryChunk,
)
from loguru import logger

from cosmos3_nano_sim_auto.apps.interactive_drive.coverage import MapCoverage
from cosmos3_nano_sim_auto.apps.interactive_drive.poses import SuppliedPoses
from cosmos3_nano_sim_auto.config import (
    DEFAULT_ATTENTION_SCOPE,
    DEFAULT_CHECKPOINT,
    DEFAULT_DENOISE_STEPS,
    DEFAULT_HISTORY_FRAMES,
    DEFAULT_TEMPORAL_WINDOW_SECONDS,
    denoise_schedule,
)
from cosmos3_nano_sim_auto.impl.conditioning.control_provider import (
    RenderedControls,
    pixel_span_for_latents,
)
from cosmos3_nano_sim_auto.impl.conditioning.ludus_controls import open_clip
from cosmos3_nano_sim_auto.impl.decode import (
    decoded_views,
    make_multiview_chunk_decoder,
)
from cosmos3_nano_sim_auto.impl.distributed.parallel import (
    gather_views,
    owned_views,
)
from cosmos3_nano_sim_auto.impl.packing import DEFAULT_CHUNK_FRAMES, ClipGeometry
from cosmos3_nano_sim_auto.impl.transformer.mask import AttentionScope
from cosmos3_nano_sim_auto.inputs import (
    WAN_TEMPORAL_COMPRESSION,
    MultiviewSample,
    camera_view_ids,
    condition_latents,
    condition_pixel_frames,
    control_tokens,
    make_control_encode_fn,
    make_control_encode_streams,
    patch_grid,
    pixel_frames_for_latents,
    prompt_token_ids,
)
from flashdreams.core.distributed.parallel import ParallelContext
from flashdreams.core.visualization.lidar import LidarView, LidarViewport
from flashdreams.infra.profiler import EventProfiler
from flashdreams.infra.video_output import lazy_rgb_frames_from_video_tensor

__all__ = ["MultiViewRenderBackend", "MultiViewSettings"]


def _note(line: str) -> None:
    """Report what the demo reports: the scene read, and each span's cost.

    ``open_clip`` and ``LudusControls`` both take this and both default it to
    silence, which for a host application means losing the only account of the
    one stage of a chunk that is ours.
    """
    logger.info(line.rstrip())


@dataclass(frozen=True)
class MultiViewSettings:
    """What the integration chooses, as against what the app chooses.

    The app owns the rig, the scene and the frame rate; these are the knobs that
    belong to the checkpoint.

    Every default here comes from :mod:`cosmos3_nano_sim_auto.config` rather than being
    a number restated, because they describe the weights and not how hard the
    model is being asked to work: run the chunk, the schedule, the cache or the
    scope at anything else and the picture quietly degrades. Taking them by
    import means a checkpoint change moves every caller at once. The demo's
    profiles add compilation, a fused RMSNorm and a K/V arena on top; those were
    gated on the older weights and are untested on these, so they stay off here
    too.
    """

    checkpoint: str | None = None
    device: str = "cuda"
    seed: int = 0
    frames_per_chunk: int = DEFAULT_CHUNK_FRAMES
    history_frames: int = DEFAULT_HISTORY_FRAMES
    condition_frames: int = 1
    denoise_steps: int | None = DEFAULT_DENOISE_STEPS
    attention_scope: AttentionScope = DEFAULT_ATTENTION_SCOPE
    maskless_attention: bool = True
    fused_attention_merge: bool = True
    temporal_window_seconds: float = DEFAULT_TEMPORAL_WINDOW_SECONDS

    lidar_checkpoint: Path | None = None
    """Enable live joint generation with this trained LiDAR codec."""
    lidar_tokenizer_version: Literal["v1p1", "v1p2"] = "v1p2"
    """Codec architecture matching ``lidar_checkpoint``."""
    lidar_contiguous_linears: bool = False
    """Use contiguous FP32 codec projections; benchmark for the target GPU."""
    lidar_view: LidarView = LidarView.POINT_CLOUD
    """Visualization of the generated and HD-map LiDAR sensor cells."""
    optimized_lidar_renderer: bool = False
    """Opt in to the prepared-scene LiDAR renderer optimizations."""
    culled_lidar_renderer: bool = True
    """Cull moving scans to points within range and settle each point's
    rolling-shutter solve. Matches the fully converged solve exactly on GB300
    (the baseline's mean stopping rule does not)."""
    lidar_codec_tf32: bool = False
    """Opt in to running the FP32 LiDAR codec's matmuls in TF32: about 2x on GB300 with 0.005%
    validity flips, 2.4 cm p99 range error and unchanged reconstruction error
    in the reference validation."""

    compile_denoise: str | None = None
    """A ``torch.compile`` mode for the noisy pass, as the batch command's flag.
    RGB or joint RGB+LiDAR, but not with a tensor axis."""
    parallel_decode: bool = False
    """Each rank decodes the cameras it owns and rank 0 gathers them."""
    decoder_compile: bool = False
    """Run the TAEHV decoder under ``torch.compile``."""
    compile_control_encode: bool = False
    """Run the map's control encoder under ``torch.compile``."""
    prefetch_controls: bool = False
    """Draw and encode the map in a worker process. RGB-only, the next chunk's
    is drawn while this one denoises, along the host's path one chunk late; with
    LiDAR, only this chunk's, once the host has supplied its poses."""
    prefetch_controls_gpu_offset: int = 0
    """The worker's GPU: this rank's index plus this, modulo the node's count."""

    parallel: ParallelContext | None = None
    """The mesh this rank runs on, or ``None`` for a drive on one card.

    Parallelism belongs to the model binding rather than to the app, which is
    how OmniDreams arranges it and the only arrangement that works here: the
    axes are the checkpoint's own, a context axis over tokens and a tensor axis
    over attention heads, and the app has no business knowing either exists.
    What the app needs from a mesh is one fact, which rank owns the window, and
    it gets that through the session.

    A mesh of several ranks names this rank's card and overrides :attr:`device`,
    because every rank must sit on its own: two ranks on one GPU is a run that
    works and halves the memory it has. A mesh of one names a card too, that
    being whatever the launcher's absence left current, and there :attr:`device`
    wins, so asking for one card out of eight still works. This is the rule the
    batch command follows.
    """


class MultiViewRenderBackend(RenderBackend):
    """Interactive Drive's backend, generating with the multiview checkpoint."""

    def __init__(
        self,
        chunk: ChunkConfig,
        raster: RasterConfig,
        *,
        settings: MultiViewSettings | None = None,
    ) -> None:
        super().__init__(chunk, raster)
        self._settings = settings or MultiViewSettings()
        if self._settings.lidar_checkpoint is not None and (
            chunk.fps != 30
            or self._settings.frames_per_chunk != 3
            or self._settings.condition_frames != 1
            or self._settings.attention_scope != "decomposed"
        ):
            raise ValueError(
                "Live LiDAR requires 30 FPS, three RGB latents per chunk, "
                "one opening frame and decomposed attention"
            )
        mesh = self._settings.parallel
        # Refused here, before 28 GiB of weights load, rather than by the rollout.
        if (
            self._settings.compile_denoise is not None
            and mesh is not None
            and mesh.tensor_parallel
        ):
            raise ValueError(
                "--compile-denoise runs without a tensor axis: it all-reduces "
                "inside the compiled region. Set COSMOS3_NANO_SIM_AUTO_TENSOR_PARALLEL=1, "
                "or drop --compile-denoise."
            )
        self._device = torch.device(
            str(mesh.device)
            if mesh is not None and mesh.world_size > 1
            else self._settings.device
        )
        self._network: Any = None
        self._rollout: Any = None
        self._scene: SceneBundle | None = None
        self._drawn: _DrawnRig | None = None
        self._poses: SuppliedPoses | None = None
        self._coverage: MapCoverage | None = None
        self._decode: Any = None
        self._lidar: Any = None
        self._worker: Any = None
        self._frames_emitted = 0
        self._decoded_to = 0
        self._first_chunk_frames = 0
        self._chunks = 0
        self._window_ms = 0.0
        self._timings: dict[str, float] = {}

    @property
    def initial_chunk_frames(self) -> int:
        """Pixel frames the first chunk presents: the opening frame and a chunk.

        Not the control window, which is several chunks longer and is drawn
        while the rollout is built, before the app has a trajectory to sample.
        What the app must supply is what this hands back, and that is the chunk
        the rollout generates first.
        """
        if not self._first_chunk_frames:
            raise RuntimeError("load_scene() must run before the first chunk.")
        return self._first_chunk_frames

    @property
    def chunk_frames(self) -> int:
        return _span_frames(self._settings.frames_per_chunk)

    @property
    def can_prewarm(self) -> bool:
        """The checkpoint loads without a scene; only the map needs one."""
        return True

    @property
    def optimizes_on_first_chunk(self) -> bool:
        """The prefill and the first chunk pay for autotuning and compilation."""
        return True

    def warmup_model(self) -> None:
        """Load the checkpoint, once per process and cut to this rank's share."""
        if self._settings.lidar_checkpoint is not None:
            from cosmos3_nano_sim_auto.impl.conditioning.world_scenario.prepared_lidar import (
                require_implementation,
            )

            require_implementation()
        if self._network is not None:
            return
        from cosmos_interactive.checkpoint_load import resolve_checkpoint_source

        from cosmos3_nano_sim_auto.checkpoint import load_exported_checkpoint

        source = resolve_checkpoint_source(
            self._settings.checkpoint, default=DEFAULT_CHECKPOINT
        )
        mesh = self._settings.parallel
        if mesh is not None and mesh.tensor_parallel:
            from cosmos3_nano_sim_auto.impl.distributed.tensor_parallel import (
                shard_network,
            )

            # Host first, then cut, then move this rank's slice. Loading to the
            # card and sharding after would put all 28.2 GiB on every GPU, which
            # defeats the axis on exactly the card that needs it.
            network = load_exported_checkpoint(source, device="cpu").eval()
            shard_network(network, mesh)
            self._network = network.to(self._device)
            return
        self._network = load_exported_checkpoint(
            source, device=str(self._device)
        ).eval()

    def load_scene(self, scene: SceneBundle) -> None:
        """Bind one drive: its map, its cameras, its prompt and its first frames."""
        prepared = scene.metadata.get("prepared_scene")
        if (
            self._settings.lidar_checkpoint is not None
            and prepared is not None
            and prepared.lidar is None
        ):
            raise ValueError(
                "LiDAR requires a v2 prepared scene; use the LiDAR scene ZIP"
            )
        self.warmup_model()
        self._scene = scene
        started = time.perf_counter()
        self._start_rollout()
        logger.info(
            "[cosmos3-nano-sim-auto] load_scene "
            f"views={len(scene.cameras)} "
            f"window_ms={self._window_ms:.1f} "
            f"total_ms={(time.perf_counter() - started) * 1000.0:.1f}"
        )

    def reset(self) -> None:
        """Start the drive again, keeping the checkpoint loaded."""
        if self._scene is not None:
            self._start_rollout()

    def reset_scene_conditioning(self) -> None:
        """Let go of the rollout rather than start it over.

        The base class resets, which for this backend means building a rollout,
        and its only caller binds a new scene straight afterwards and builds
        another. Binding is nearly the whole cost of a restart, so doing it
        twice made restarting cost about what opening the drive did. Everything
        dropped here :meth:`_start_rollout` assigns again.
        """
        self._rollout = None
        self._decode = None
        self._drawn = None
        self._lidar = None
        self._poses = None
        self._coverage = None
        self._close_worker()

    def _close_worker(self) -> None:
        if self._worker is not None:
            self._worker.close()
        self._worker = None

    def render_first_chunk(self, trajectory: TrajectoryChunk) -> FrameChunk:
        return self._render(trajectory, opening=True)

    def render_next_chunk(self, trajectory: TrajectoryChunk) -> FrameChunk:
        return self._render(trajectory, opening=False)

    def finalize(self) -> dict[str, float] | None:
        return dict(self._timings)

    def close(self) -> None:
        self._rollout = None
        self._decode = None
        self._drawn = None
        self._lidar = None
        self._network = None
        self._coverage = None
        self._poses = None
        self._close_worker()

    def _render(self, trajectory: TrajectoryChunk, *, opening: bool) -> FrameChunk:
        """One chunk: take the path, generate it, decode it, hand back pictures.

        Timed on CUDA events rather than the host clock, because the host only
        knows how long it took to *ask*: the work is queued, and whichever block
        happens to wait for it is the one that appears to have paid. Events are
        recorded on the stream and read once at the end, so the stages are the
        GPU's own and the measurement does not serialise what it measures.
        """
        rollout, poses, drawn = self._rollout, self._poses, self._drawn
        if rollout is None or poses is None or drawn is None:
            raise RuntimeError("load_scene() must run before a chunk is rendered.")
        poses.supply(np.asarray(trajectory.rig_poses_world, dtype=np.float32))
        if self._lidar is not None:
            self._lidar.supply(np.asarray(trajectory.rig_poses_world, dtype=np.float32))

        profiler = EventProfiler(device=self._device)
        trace, _tokens = rollout.step()
        profiler.record("generate")
        latent = rollout.latent_for(self._decoded_to, trace.end)
        self._decoded_to = trace.end
        mesh = self._settings.parallel
        # The LiDAR owner decodes while the cameras decode; rank 0 waits below.
        lidar_ranges = (
            None
            if self._lidar is None
            else self._lidar.start_decode(rollout.last_lidar)
        )
        if self._decode is not None:
            pixels = decoded_views(
                latent,
                self._decode,
                parallel=mesh if self._settings.parallel_decode else None,
            )
        if not _presents(mesh):
            # The app presents on rank 0 alone, and reads nothing a worker returns.
            # A worker's part was generating, its own cameras' decode and its
            # share of the drawn map, so the LiDAR decode and render stop here too.
            self._frames_emitted += len(trajectory.timestamps_us)
            return FrameChunk(
                frames=(),
                boundary_state_after_chunk=trajectory.boundary_state_after_chunk,
                source_name="cosmos3-nano-sim-auto",
            )
        profiler.record("decode")
        lidar_pixels = control_pixels = None
        if self._lidar is not None:
            assert lidar_ranges is not None
            ranges = lidar_ranges()
            profiler.record("lidar_decode")
            lidar_pixels = (
                self._lidar.frames(
                    ranges,
                    first_sweep=rollout.last_lidar_range[0],
                    pixel_start=self._frames_emitted,
                    pixel_count=int(pixels.shape[1]),
                    fps=float(self._chunk.fps),
                    viewport=LidarViewport(
                        width=int(pixels.shape[-1]), height=int(pixels.shape[-2])
                    ),
                    view=self._settings.lidar_view,
                )
                .to(pixels)
                .mul(2)
                .sub(1)
            )
            control_pixels = (
                self._lidar.control_frames(
                    pixel_start=self._frames_emitted,
                    pixel_count=int(pixels.shape[1]),
                    fps=float(self._chunk.fps),
                    viewport=LidarViewport(
                        width=int(pixels.shape[-1]), height=int(pixels.shape[-2])
                    ),
                    view=self._settings.lidar_view,
                )
                .to(pixels)
                .mul(2)
                .sub(1)
            )
            profiler.record("lidar_render")
        stages = profiler.sync_and_summarize()

        frames = int(pixels.shape[1])
        # After that sync, which is what makes the drawing's own events readable.
        # Keep the historical aggregate: this includes control encoding and,
        # for joint generation, LiDAR rendering. Log the joint stages separately.
        draw_ms = drawn.take_drawn_ms()
        self._timings = {
            "chunk_start": float(trace.start),
            "chunk_end": float(trace.end),
            "memory_tokens": float(trace.memory_tokens),
            "draw_ms": draw_ms,
            "model_ms": stages["generate"] - draw_ms,
            "decode_ms": stages["decode"],
            "total_ms": sum(stages.values()),
        }
        if self._lidar is not None:
            self._timings.update(rollout.last_metrics)
            self._timings.update(self._lidar.last_metrics)
            self._timings.update(
                lidar_enabled=1.0,
                lidar_decode_s=stages["lidar_decode"] / 1000,
                lidar_render_s=stages["lidar_render"] / 1000,
            )
        self._report(opening=opening, frames=frames, trace=trace)
        presented = _presented(
            model=pixels,
            drawn=drawn.take(frames),
            timestamps_us=trajectory.timestamps_us,
        )
        if lidar_pixels is not None:
            assert control_pixels is not None
            sensor = lazy_rgb_frames_from_video_tensor(lidar_pixels, layout="tchw")
            control_sensor = lazy_rgb_frames_from_video_tensor(
                control_pixels, layout="tchw"
            )
            presented = tuple(
                replace(
                    frame,
                    view_rgb_host_uint8=(
                        *frame.view_rgb_host_uint8,
                        control_sensor[index],
                    ),
                    view_model_rgb_host_uint8=(
                        *frame.view_model_rgb_host_uint8,
                        sensor[index],
                    ),
                )
                for index, frame in enumerate(presented)
            )
        self._frames_emitted += frames
        return FrameChunk(
            frames=presented,
            boundary_state_after_chunk=trajectory.boundary_state_after_chunk,
            source_name="cosmos3-nano-sim-auto",
            warning=self._coverage_warning(frames),
        )

    def _coverage_warning(self, frames: int) -> str | None:
        """Report coverage at the presented control pose, not the prefetched one."""
        if self._coverage is None:
            return None
        assert self._poses is not None
        previous = self._coverage.warning
        warning = self._coverage.update(self._poses.take_position(frames))
        if warning != previous:
            logger.info(warning or "Returned to mapped area; coverage warning cleared.")
        return warning

    def _report(self, *, opening: bool, frames: int, trace: Any) -> None:
        """One line a chunk, as the app's own backend reports itself.

        Every chunk at first and then one in ten, because a steady run says the
        same thing over and over, and anything slow enough to drop frames is
        worth seeing whenever it happens.
        """
        timings = self._timings
        total_ms = timings["total_ms"]
        if opening:
            self._chunks = 0
        else:
            self._chunks += 1
            if not (self._chunks <= 3 or self._chunks % 10 == 0 or total_ms > 500.0):
                return
        logger.info(
            f"[cosmos3-nano-sim-auto] {'first_chunk' if opening else 'next_chunk'} "
            + ("" if opening else f"index={self._chunks} ")
            + f"frames={frames} "
            f"latents=[{trace.start}, {trace.end}) "
            f"draw_ms={timings['draw_ms']:.1f} "
            f"model_ms={timings['model_ms']:.1f} "
            f"decode_ms={timings['decode_ms']:.1f} "
            f"total_ms={total_ms:.1f} "
            f"fps={frames / (total_ms / 1000.0):.2f}"
            + (
                f" lidar_renderer={_renderer_label(self._settings)} "
                f"lidar_contiguous_linears={self._settings.lidar_contiguous_linears} "
                + " ".join(
                    f"{label}_ms={timings[key] * 1000:.1f}"
                    for label, key in (
                        ("lidar_render", "lidar_control_render_s"),
                        ("lidar_encode", "lidar_codec_encode_s"),
                        ("lidar_tokens", "lidar_encode_s"),
                        ("joint_control", "joint_control_s"),
                        ("joint_denoise", "joint_denoise_s"),
                        ("joint_cache", "joint_clean_cache_s"),
                        ("lidar_decode", "lidar_decode_s"),
                        ("lidar_owner_decode", "lidar_codec_decode_s"),
                        ("lidar_present", "lidar_render_s"),
                    )
                    if key in timings
                )
                if timings.get("lidar_enabled")
                else ""
            )
        )

    def _start_rollout(self) -> None:
        """Build the map, the controls and the rollout for the bound scene."""
        from cosmos3_nano_sim_auto.impl.rollout import ChunkRollout

        scene = self._scene
        assert scene is not None
        # The previous rollout owns its codec, map and history. Release it before
        # allocating the next scene's assets, including on a direct reset.
        self.reset_scene_conditioning()
        settings = self._settings
        self._frames_emitted = 0
        spec = self._network.config.spec
        patch = spec.vfm.latent_patch_size
        geometry = _geometry(
            scene, settings, latent_patch_size=patch, fps=float(self._chunk.fps)
        )

        mesh = settings.parallel
        presents = _presents(mesh)
        prepared = scene.metadata.get("prepared_scene")
        if presents and prepared is not None:
            self._coverage = MapCoverage(prepared.scene)
        cameras = [camera.clipgt_name for camera in scene.cameras]
        lidar = settings.lidar_checkpoint is not None
        # The rig frame is the anchor camera's, which the scene loader arranges,
        # so the car's pose is already the anchor's and there is nothing to bolt.
        self._poses = SuppliedPoses(
            np.eye(4, dtype=np.float32),
            opening_rig_to_world=np.asarray(
                scene.initial_rig_to_world, dtype=np.float32
            ),
            speed_mps=float(scene.initial_speed_mps),
            seconds_per_frame=1.0 / float(self._chunk.fps),
            track_positions=self._coverage is not None,
            # Drawing the next chunk while this one denoises needs its poses
            # now, and the host only hands them over a chunk at a time. So the
            # map runs a chunk behind the wheel; the LiDAR sweeps read the
            # host's own frames and must see the same path, so they cannot.
            delay_frames=self.chunk_frames
            if settings.prefetch_controls and not lidar
            else 0,
        )
        drawn_device = self._device if presents else None
        if settings.prefetch_controls:
            self._drawn = _DrawnRig(
                None, cameras=len(cameras), device=drawn_device, parallel=mesh
            )
        else:
            self._drawn = _DrawnRig(
                open_clip(
                    scene.scene_path,
                    cameras,
                    device=self._device,
                    against=Path(str(scene.metadata["sample"])),
                    start_frame=int(scene.metadata["start_frame"]),
                    drive=self._poses,
                    note=_note,
                    # A chunk's own line already carries what the drawing cost, and
                    # the rollout draws once a chunk for the whole run.
                    span_note=lambda _: None,
                    prepared=scene.metadata.get("prepared_scene"),
                    requested_fps=float(self._chunk.fps),
                ),
                device=drawn_device,
                parallel=mesh,
            )

        encode = make_control_encode_fn(
            dtype=self._network.config.dtype, device=self._device
        )
        conditions = control_tokens(
            condition_latents(
                _condition_clips(scene, geometry, device=self._device),
                encode,
                geometry=geometry,
            ),
            latent_patch_size=patch,
        )
        text_ids, view_text_tokens = _prompt(
            scene,
            fps=float(self._chunk.fps),
            lidar=lidar,
        )
        if settings.prefetch_controls:
            from cosmos3_nano_sim_auto.impl.conditioning.control_worker import (
                ControlWorker,
                WorkerSpec,
                worker_gpu,
            )

            owned = owned_views(mesh, geometry.num_views)
            # A rank with no camera draws nothing, so it uploads no scene.
            self._worker = (
                None
                if not owned
                else ControlWorker.spawn(
                    WorkerSpec(
                        clip=Path(scene.scene_path),
                        cameras=tuple(cameras),
                        owned=tuple(owned),
                        device=worker_gpu(
                            self._device, settings.prefetch_controls_gpu_offset
                        ),
                        against=Path(str(scene.metadata["sample"])),
                        start_frame=int(scene.metadata["start_frame"]),
                        wanted=None,
                        requested_fps=float(self._chunk.fps),
                        dtype=self._network.config.dtype,
                        compile_encode=settings.compile_control_encode,
                        driven=True,
                    ),
                    device=self._device,
                )
            )
            controls = RenderedControls(
                _drawn_by_worker,
                [lambda clip: clip] * geometry.num_views,
                geometry=geometry,
                latent_patch_size=patch,
                parallel=mesh,
                worker=_DrivenWorker(
                    self._worker, poses=self._poses, drawn=self._drawn, owned=owned
                ),
            )
        else:
            controls = RenderedControls(
                self._drawn.render,
                make_control_encode_streams(
                    geometry.num_views,
                    dtype=self._network.config.dtype,
                    device=self._device,
                    compile=settings.compile_control_encode,
                ),
                geometry=geometry,
                latent_patch_size=patch,
                parallel=mesh,
            )
        common: dict[str, Any] = dict(
            geometry=geometry,
            control_tokens=controls,
            condition_tokens=conditions,
            text_ids=text_ids,
            view_text_tokens=view_text_tokens,
            fps=float(self._chunk.fps),
            seed=settings.seed,
            history_frames=settings.history_frames,
            t_list=denoise_schedule(spec.vfm.fixed_step_t_list, settings.denoise_steps),
            fused_attention_merge=settings.fused_attention_merge,
            decomposed_temporal_window_seconds=settings.temporal_window_seconds,
            parallel=settings.parallel,
        )
        if settings.lidar_checkpoint is not None:
            from cosmos3_nano_sim_auto.apps.interactive_drive.lidar import (
                make_live_lidar,
            )
            from cosmos3_nano_sim_auto.impl.joint import JointChunkRollout

            if not hasattr(self._network, "lidar2llm") or not hasattr(
                self._network, "llm2lidar"
            ):
                raise ValueError("Live LiDAR requires a joint RGB+LiDAR checkpoint")
            self._lidar = make_live_lidar(
                scene,
                checkpoint=settings.lidar_checkpoint,
                device=self._device,
                version=settings.lidar_tokenizer_version,
                contiguous_linears=settings.lidar_contiguous_linears,
                fps=float(self._chunk.fps),
                patch_dim=self._network.lidar2llm.in_features,
                optimized=settings.optimized_lidar_renderer,
                cull=settings.culled_lidar_renderer,
                tf32=settings.lidar_codec_tf32,
                parallel=mesh,
            )
            self._rollout = JointChunkRollout(
                self._network,
                **common,
                lidar_controls=self._lidar,
                lidar_tokenizer=self._lidar.assets.tokenizer,
                lidar_condition=self._lidar.condition_tokens(),
                compile_denoise=settings.compile_denoise,
            )
        else:
            self._rollout = ChunkRollout(
                self._network,
                **common,
                # One window bounds both: the app presents every chunk as it lands,
                # so nothing looks further back than the cache does.
                token_frames=settings.history_frames,
                attention_scope=settings.attention_scope,
                maskless_attention=settings.maskless_attention,
                compile_denoise=settings.compile_denoise,
            )
        self._decoded_to = 0
        self._first_chunk_frames = (
            1 if self._lidar is not None else _opening_span(geometry)
        )
        # As the batch command decides it: every rank its own cameras, or rank 0
        # all of them while the others, which never present, decode nothing.
        decoding = mesh if settings.parallel_decode else None
        decode_views = len(owned_views(decoding, geometry.num_views))
        self._decode = (
            make_multiview_chunk_decoder(
                num_views=decode_views,
                dtype=self._network.config.dtype,
                device=self._device,
                compile=settings.decoder_compile,
            )
            if decode_views and (decoding is not None or presents)
            else None
        )
        # Building the rollout prefills it, which draws the whole control window
        # before any chunk is generated. Read that here or the first chunk is
        # charged for it, and its model stage understated by the same amount.
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        self._window_ms = self._drawn.take_drawn_ms()


class _DrawnRig:
    """The map renderer, queueing what it draws for the app to present.

    The rollout asks for control so it can condition on it, and the app wants
    the same pictures to show under its HD-map view. Drawing them twice would
    double the one stage of a chunk that is ours, so each span is kept on the
    way past.

    A queue rather than a single span, because control runs ahead of what is
    shown: the whole window is drawn before the first chunk is generated, so the
    frames a chunk presents were drawn some chunks ago. Spans arrive in frame
    order and contiguously, which is what lets this be a queue and nothing more.

    On a mesh the two readers want different things, which is the whole reason
    this sits between them. The encoder wants the cameras this rank owns and
    nothing else, since that is the point of sharding the draw. Presenting wants
    every camera of every frame, and only rank 0 presents. So the span handed
    back is the rank's own and the span queued, on rank 0 alone, is the rig's.

    With ``--prefetch-controls`` the drawing is a worker process's and there is
    no renderer here: :class:`_DrivenWorker` hands over what it drew through
    :meth:`keep`, and the queue and the gather are the same.
    """

    def __init__(
        self,
        ludus: Any,
        *,
        cameras: int | None = None,
        device: torch.device | None = None,
        parallel: ParallelContext | None = None,
    ) -> None:
        self._ludus = ludus
        self._cameras = len(ludus.cameras) if cameras is None else cameras
        self._queued: list[tuple[Any, ...]] = []
        self._device = device if device is not None and device.type == "cuda" else None
        self._parallel = parallel
        self._spans: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    def render(
        self, first: int, last: int, views: Sequence[int] | None = None
    ) -> torch.Tensor:
        since = self.mark()
        frames = self._ludus.render(first, last, views)
        self.keep(frames, views, span=last - first, since=since)
        return frames

    def keep(
        self,
        frames: torch.Tensor,
        views: Sequence[int] | None,
        *,
        span: int,
        since: torch.cuda.Event | None,
    ) -> None:
        """Queue a drawn span for presenting, timing it from ``since``."""
        rig = self._whole_rig(frames, views, span=span)
        if rig is not None:
            self._queued.extend(_per_frame(rig, self._cameras))
        end = self.mark()
        if since is not None and end is not None:
            self._spans.append((since, end))

    def _whole_rig(
        self, drawn: torch.Tensor, owned: Sequence[int] | None, *, span: int
    ) -> torch.Tensor | None:
        """Every camera's pictures on rank 0, from ranks that drew only their own.

        ``None`` on the other ranks, which present nothing. Inside the timed
        region deliberately, so what the sharing costs lands in ``draw_ms``
        beside what it saves rather than hiding in the model's share. It is a
        small bill: a camera's span is a few megabytes at the size the model is
        fed, where the encode it buys back is a VAE pass per camera.
        """
        mesh = self._parallel
        if owned is None or mesh is None or mesh.world_size == 1:
            return drawn
        # Collectives stage on the rank's own card under NCCL, and on the host
        # under gloo. The same rule as the control encode's broadcasts.
        staging = mesh.device if dist.get_backend() == "nccl" else torch.device("cpu")
        held = [
            drawn[index * span : (index + 1) * span].to(staging)
            for index in range(len(owned))
        ]
        rig = gather_views(held, num_views=self._cameras, parallel=mesh)
        return None if rig is None else torch.cat(rig)

    def mark(self) -> torch.cuda.Event | None:
        """An event on the stream, or nothing where there is no GPU to time."""
        if self._device is None:
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record(torch.cuda.current_stream(self._device))
        return event

    def take_drawn_ms(self) -> float:
        """Milliseconds the GPU spent drawing since the last chunk asked.

        Drawing happens inside the rollout's step, as it tops up the control it
        is about to read, so a chunk's own cost can only be told apart here. A
        chunk usually draws one span and sometimes none at all, the window the
        prefill drew covering the opening chunks, so this sums whatever arrived.
        With a worker it is what the step waited for the worker plus the gather,
        which is the drawing's share of the critical path.

        Only call this after a synchronize: unfinished events raise rather than
        return a wrong number, and the chunk's own profiler is what waits.
        """
        spans, self._spans = self._spans, []
        return sum(start.elapsed_time(end) for start, end in spans)

    def take(self, frames: int) -> list[tuple[Any, ...]]:
        """The next ``frames`` drawn, each carrying every camera of the rig.

        Raises:
            RuntimeError: fewer were drawn than the chunk presents, which means
                the rollout generated frames it never read a map for.
        """
        if len(self._queued) < frames:
            raise RuntimeError(
                f"{len(self._queued)} frame(s) of map drawn against {frames} to "
                "present; the rollout generated frames it read no control for."
            )
        taken = self._queued[:frames]
        del self._queued[:frames]
        return taken


class _DrivenWorker:
    """A :class:`ControlWorker` drawing along the host's path, for ``--prefetch-controls``.

    What :class:`RenderedControls` asks of a worker, with the drive's two
    differences. The poses go with each request, taken from
    :class:`SuppliedPoses` here, because the host hands them to this process; and
    the drawn frames come back with the latents and go into the same queue,
    across the same gather, that an in-process draw feeds.

    A request is sent only once the host has supplied every pose it covers.
    Asked earlier it would draw an extrapolation, and the host's poses for that
    ground would then be dropped as already drawn, so the car would never steer
    again. A request held back is sent when the range is taken instead, which is
    then simply the in-process order of things with the drawing elsewhere.

    A rank owning no camera has no worker: it only moves the car along, which
    every rank must do in step, and sends nothing to the gather.
    """

    def __init__(
        self,
        worker: Any | None,
        *,
        poses: SuppliedPoses,
        drawn: _DrawnRig,
        owned: Sequence[int],
    ) -> None:
        self._worker = worker
        self._poses = poses
        self._drawn = drawn
        self._owned = list(owned)
        self._pending: tuple[int, int] | None = None

    def request(self, start: int, end: int) -> None:
        if self._pending == (start, end):
            return
        if self._pending is not None:
            raise ValueError(
                f"[{self._pending[0]}, {self._pending[1]}) is already in flight; "
                f"[{start}, {end}) cannot be until it is taken."
            )
        first, last = pixel_span_for_latents(start, end)
        if self._worker is None or self._poses.supplied < last:
            return
        self._worker.request(start, end, poses=self._poses.poses(first, last))
        self._pending = (start, end)

    def take(self, start: int, end: int) -> tuple[list[torch.Tensor], int]:
        if self._worker is None:
            first, last = pixel_span_for_latents(start, end)
            self._poses.poses(first, last)
            self._drawn.keep(torch.empty(0), [], span=last - first, since=None)
            return [], last - first
        if self._pending is None:
            first, last = pixel_span_for_latents(start, end)
            self._worker.request(start, end, poses=self._poses.poses(first, last))
            self._pending = (start, end)
        since = self._drawn.mark()
        latents, pixel_frames = self._worker.take(start, end)
        self._pending = None
        self._drawn.keep(
            self._worker.last_drawn, self._owned, span=pixel_frames, since=since
        )
        return latents, pixel_frames


def _drawn_by_worker(*_args: Any, **_kwargs: Any) -> torch.Tensor:
    raise RuntimeError("The control worker draws this map, not the model process")


def _renderer_label(settings: MultiViewSettings) -> str:
    if settings.culled_lidar_renderer:
        return "culled"
    return "optimized" if settings.optimized_lidar_renderer else "baseline"


def _presents(mesh: ParallelContext | None) -> bool:
    """Whether this rank is the one the app presents on."""
    return mesh is None or mesh.is_main


def _per_frame(drawn: torch.Tensor, views: int) -> list[tuple[Any, ...]]:
    """A span of control, split into one entry a frame carrying every camera.

    ``[V * T, H, W, C]`` as ludus draws it, with a view's frames together, which
    is the opposite of how a frame is presented.
    """
    frames = drawn.shape[0] // views
    host = drawn[..., :3].to("cpu").numpy()
    return [
        tuple(host[view * frames + index] for view in range(views))
        for index in range(frames)
    ]


def _presented(
    *,
    model: torch.Tensor,
    drawn: Sequence[Sequence[Any]],
    timestamps_us: Any,
) -> tuple[PresentedFrame, ...]:
    """One frame per moment, carrying every camera of the rig.

    The generated pictures and the map they were drawn over cover the same pixel
    frames, so a disagreement means the control span and the decode have come
    apart, which is worth stopping for rather than presenting half a chunk.
    """
    views, frames = int(model.shape[0]), int(model.shape[1])
    if len(timestamps_us) != frames:
        raise ValueError(
            f"{frames} generated frame(s) against {len(timestamps_us)} timestamp(s); "
            "the app's chunk size and this backend's disagree."
        )
    if len(drawn) != frames:
        raise ValueError(
            f"{len(drawn)} drawn frame(s) against {frames} generated; the control "
            "render and the rollout have come apart."
        )
    if any(len(one) != views for one in drawn):
        raise ValueError(
            f"a drawn frame does not carry the {views} camera(s) generated; the "
            "control render and the rollout disagree about the rig."
        )
    model_views = [
        lazy_rgb_frames_from_video_tensor(
            model.unsqueeze(0), layout="bvtchw", view_index=view
        )
        for view in range(views)
    ]
    return tuple(
        PresentedFrame(
            timestamp_us=int(timestamps_us[index]),
            view_rgb_host_uint8=tuple(drawn[index]),
            view_model_rgb_host_uint8=tuple(view[index] for view in model_views),
            depth_host_f32=None,
        )
        for index in range(frames)
    )


def _span_frames(latent_frames: int) -> int:
    """Pixel frames a chunk of ``latent_frames`` covers, past the opening one.

    Every latent frame after the first covers four pixel frames, which is what
    ``pixel_span_for_latents`` tiles the timeline by.
    """
    return WAN_TEMPORAL_COMPRESSION * latent_frames


def _opening_span(geometry: ClipGeometry) -> int:
    """Pixel frames the first generated chunk covers, conditioning included.

    The conditioning frames are decoded along with it, being latent frames of
    the same run, so the app presents them and this counts them.
    """
    return pixel_frames_for_latents(
        geometry.condition_frames + geometry.frames_per_chunk
    )


def _geometry(
    scene: SceneBundle,
    settings: MultiViewSettings,
    *,
    latent_patch_size: int,
    fps: float,
) -> ClipGeometry:
    """The rig's token shape, for a run with no end.

    ``fps`` is the app's, the same rate the poses and the rollout are given.
    The timestamps this spaces are what the decomposed scope compares against
    its window, so a rate stated here and not there would attend over the
    wrong stretch of time rather than merely mislabel one.
    """
    frame = scene.initial_rgbs[0]
    patch_h, patch_w = patch_grid(
        int(frame.shape[0]), int(frame.shape[1]), latent_patch_size=latent_patch_size
    )
    return ClipGeometry(
        num_views=len(scene.cameras),
        frames_per_view=None,
        patch_h=patch_h,
        patch_w=patch_w,
        frames_per_chunk=settings.frames_per_chunk,
        condition_frames=settings.condition_frames,
        seconds_per_frame=WAN_TEMPORAL_COMPRESSION / fps,
        rig_view_ids=camera_view_ids([camera.clipgt_name for camera in scene.cameras]),
    )


def _condition_clips(
    scene: SceneBundle, geometry: ClipGeometry, *, device: str | torch.device
) -> list[torch.Tensor]:
    """The opening frames in the encoder's range, one clip per camera.

    The app holds one frame a camera, and one latent conditioning frame is one
    pixel frame, so a geometry wanting more than one repeats what there is
    rather than inventing motion the scene does not state.
    """
    wanted = condition_pixel_frames(geometry)
    clips = []
    for frame in scene.initial_rgbs:
        pixels = torch.from_numpy(np.ascontiguousarray(frame)).to(device)
        clip = pixels.permute(2, 0, 1).to(torch.float32).div(127.5).sub(1.0)
        clips.append(clip.unsqueeze(0).repeat(wanted, 1, 1, 1).contiguous())
    return clips


def _prompt(
    scene: SceneBundle, *, fps: float, lidar: bool = False
) -> tuple[torch.Tensor, tuple[int, ...] | None]:
    """The words this drive conditions on, as token ids.

    Built from the sample through the helper the batch command uses, so the two
    commands running the same drive say the same thing to the model. A fetched
    recorded-scene record carries a caption per camera and no single prompt, which is the
    layout the delivered checkpoint was trained on; reading the bundle's lone
    sentence instead left every camera with nothing but the prefix saying which
    way it points.

    A sentence the host supplied does replace them, whether that is
    ``--prompt`` or one typed into a restart. It describes the whole rig rather
    than a camera, so it takes the single-prompt branch and is pointed at each
    camera in turn. That is also what the vendored sample hits, having captions
    for nobody.
    """
    sample = MultiviewSample.load(
        Path(str(scene.metadata["sample"])) / "sample.json"
    ).for_rig([camera.clipgt_name for camera in scene.cameras])
    if scene.prompt:
        sample = replace(sample, prompt=scene.prompt, view_captions=())
    # Only the caption payload states a duration, and a sample carrying
    # captions carries the span they were written against. A drive has no end
    # of its own to state, so that span is the pairing training would have had.
    first, last = sample.frame_range or (0, 0)
    prompt = prompt_token_ids(sample, pixel_frames=last - first, fps=fps, lidar=lidar)
    return prompt.text_ids, prompt.view_text_tokens

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The v2 application: eleven cameras through the standard model loop.

One ``IModelLoop.step`` is one autoregressive chunk. The eleven views are tiled
into a single mosaic frame here, because the default screen loop would otherwise
draw them over each other; ``--camera`` emits one view at its native size
instead. Steps do not all cover the same number of frames, so read
``StepResult.frame_count`` rather than assuming a constant.
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, get_args

if TYPE_CHECKING:
    from cosmos3_nano_sim_auto.impl.lidar import LidarAssets, LidarStream

import torch
from cosmos_interactive._network import CosmosMoTNetwork
from cosmos_interactive.checkpoint_load import resolve_checkpoint_source
from loguru import logger
from torch import Tensor

from cosmos3_nano_sim_auto.checkpoint import (
    load_exported_checkpoint,
    resolve_lidar_checkpoint,
)
from cosmos3_nano_sim_auto.config import (
    DEFAULT_ATTENTION_SCOPE,
    DEFAULT_CHECKPOINT,
    DEFAULT_DENOISE_STEPS,
    DEFAULT_HISTORY_FRAMES,
    DEFAULT_SCENE,
    DEFAULT_TEMPORAL_WINDOW_SECONDS,
    denoise_schedule,
)
from cosmos3_nano_sim_auto.impl.conditioning.control_provider import (
    ControlProvider,
    RenderedControls,
)
from cosmos3_nano_sim_auto.impl.conditioning.controls import (
    available_pixel_frames,
    generated_control_clips,
    loaded_control_clips,
    require_local_files,
)
from cosmos3_nano_sim_auto.impl.decode import (
    decoded_views,
    make_multiview_chunk_decoder,
)
from cosmos3_nano_sim_auto.impl.distributed.parallel import (
    init_parallel,
    owned_views,
)
from cosmos3_nano_sim_auto.impl.distributed.tensor_parallel import shard_network
from cosmos3_nano_sim_auto.impl.joint import JointChunkRollout
from cosmos3_nano_sim_auto.impl.packing import DEFAULT_CHUNK_FRAMES
from cosmos3_nano_sim_auto.impl.rollout import COMPILE_DENOISE_MODES, ChunkRollout
from cosmos3_nano_sim_auto.impl.transformer.attention import (
    attention_implementation,
    rtx_pro_6000_blackwell_attention_implementation,
)
from cosmos3_nano_sim_auto.impl.transformer.mask import (
    MASKLESS_SCOPES,
    AttentionScope,
)
from cosmos3_nano_sim_auto.inputs import (
    MultiviewSample,
    condition_latents,
    condition_pixel_frames,
    control_latents,
    control_tokens,
    make_control_encode_fn,
    make_control_encode_streams,
    prompt_token_ids,
)
from cosmos3_nano_sim_auto.scenes import SCENE_NAMES, prepare_scene
from flashdreams.accelerated.multi_head_attention.optimized import (
    OptimizedImplConfig,
    SDPABackend,
)
from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.loop import IModelLoop, IUILoop
from flashdreams.api_v2.session import ISession
from flashdreams.core.distributed.parallel import ParallelContext
from flashdreams.core.visualization.lidar import LidarView
from flashdreams.runtime_v2.session_desc import PresentationMode, SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

DEFAULT_FRAMES = 33
"""Pixel frames per view. A ``1 + 4n`` grid point, and the first one that states
a whole second rather than the ``"0s"`` the caption augmentor truncates to."""

DEFAULT_VIEWS = 11
"""Cameras in the full rig."""

DEFAULT_MASKLESS_ATTENTION = None
"""On wherever the scope can be folded, which is every scope but ``all_views``.

Left as ``None`` rather than ``True`` so that the two ways of arriving at an
unfoldable scope can be told apart. Asking for ``all_views`` *and* the fold is
a contradiction and is refused; asking for ``all_views`` alone is a diagnostic
someone wants to run, and it should not have to switch off a default to get it.

The checkpoint selects legacy overlapping folds or the final Sim-Auto
deduplicated folds. Each runs dense attention over its contiguous key spans.
"""

RTX_PRO_6000_BLACKWELL_WARM_PROFILE = "rtx-pro-6000-blackwell"
"""Quality-qualified compiled warm-service profile for the 11-view rig."""

RTX_PRO_6000_BLACKWELL_PERF_PROFILE = f"{RTX_PRO_6000_BLACKWELL_WARM_PROFILE}-perf"
"""Faster fused-RMSNorm variant that passed five of six quality thresholds."""

_RTX_PRO_6000_BLACKWELL_PROFILES = frozenset(
    (RTX_PRO_6000_BLACKWELL_WARM_PROFILE, RTX_PRO_6000_BLACKWELL_PERF_PROFILE)
)


@dataclass(frozen=True, slots=True)
class Cosmos3NanoSimAutoConfig:
    """Resolved settings for one multiview application."""

    checkpoint: str | None
    sample_dir: Path | None
    views: int
    frames: int | None
    """Pixel frames per view, or ``None`` for a run with no end."""
    fps: float
    seed: int
    optimization_profile: str
    camera: str | None
    device: str
    attention_impl: OptimizedImplConfig
    accelerated_rms_norm: bool
    attention_kv_arena: bool
    parallel: ParallelContext
    history_frames: int | None
    compile_denoise: str | None
    denoise_steps: int | None
    chunk_frames: int | None
    outlive_caption: bool
    loop_controls: bool
    attention_scope: AttentionScope
    maskless_attention: bool
    decomposed_temporal_window_seconds: float | None

    draw_map: Path | None
    """A prepared scene bundle to render controls from along its camera path."""

    fused_attention_merge: bool = False
    """Fuse qualified eager maskless vector merges, keeping native row weights."""

    ui: bool = False
    """Present named camera and sensor channels in the shared grid viewer."""

    lidar: LidarAssets | None = None
    """Prepared optional LiDAR weights, calibrated controls and timestamps."""

    lidar_view: LidarView = LidarView.POINT_CLOUD
    """Initial point-cloud, range, or intensity presentation."""

    parallel_encode: bool = True
    """Read and encode each camera on one rank, then share the latents."""

    parallel_decode: bool = False
    """Decode each camera on the rank that owns it, then send the pixels to rank 0."""

    compile_control_encode: bool = False
    """Run the --draw-map control encoder under torch.compile."""

    decoder_compile: bool = False
    """Run the TAEHV decoder under torch.compile."""

    prefetch_controls: bool = False
    """Draw and encode the next chunk's --draw-map controls in a worker process while this one denoises."""

    prefetch_controls_gpu_offset: int = 0
    """Which GPU the worker draws on: this rank's plus this, modulo the node's count. 0 shares the rank's GPU."""

    lidar_decode_worker: bool = False
    """Decode the LiDAR sweeps in a worker process on the --prefetch-controls-gpu-offset GPU, off the model thread."""

    @property
    def flex_attention(self) -> bool:
        """Whether this run uses block-sparse FlexAttention."""
        return self.attention_impl.sdpa_backend is SDPABackend.FLEX


@dataclass(slots=True)
class Cosmos3NanoSimAutoModelState:
    """What the model loop owns: the rollout, the decoder, and where it is."""

    config: Cosmos3NanoSimAutoConfig
    session_desc: SessionDesc
    sample: MultiviewSample
    network: CosmosMoTNetwork
    rollout: ChunkRollout | JointChunkRollout | None = None
    decode: Callable[[Tensor], Tensor] | None = None
    lidar_stream: LidarStream | None = None
    lidar_view: LidarView = LidarView.POINT_CLOUD
    ui_loop: IUILoop | None = None
    decoded_to: int = 0
    """Latent frames handed to the decoder so far, conditioning included."""

    frames_emitted: int = 0
    metrics: dict[str, float | int] = field(default_factory=dict)

    steady_started_at: float | None = None
    """When the cache stopped filling, or ``None`` while it still is."""

    steady_frames: int = 0
    """Frames emitted since :attr:`steady_started_at`."""


class Cosmos3NanoSimAutoModelLoop(IModelLoop[Cosmos3NanoSimAutoModelState]):
    """Generate model shards; the runtime coordinates inputs and step admission."""

    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        """Generate the next chunk, and on rank 0 decode and return it.

        Returns:
            One channel holding the chunk's frames, or no channel at all on a
            rank nobody is watching -- which the v2 runtime reads as a step that
            presented nothing.
        """
        state = self.state
        built = state.rollout is None
        if built:
            with _timed(state.config.device) as elapsed:
                state.rollout, state.decode = _build(state)
            state.metrics["prefill_s"] = elapsed()
            if state.config.parallel.is_main:
                _log_prefill(state)
            else:
                _log_worker(state)

        # This command replays a recorded drive, so there is nobody to read: a
        # run steered as it goes is the `interactive-drive-cosmos3-nano-sim-auto`
        # application, which has the wheel, the vehicle model and the HUD.
        del events

        # Timed as one block as well as in parts, so the parts have something to
        # be checked against. A stream cannot afford a per-chunk cost that no
        # timer covers, and the 197-frame clip had one: its last four chunks each
        # spent ~3s outside `generate` and `decode` both.
        rollout = state.rollout
        assert rollout is not None
        with _timed(state.config.device) as elapsed:
            frames, metrics = self._chunk(state)
        metrics["step_s"] = elapsed()

        if frames is None:
            # A worker rank. Nothing was decoded and nothing is published, so
            # there is no result to hang these metrics on either; rank 0's are
            # the run's.
            return []

        frame_count = int(frames[0].shape[0])
        metrics.update(
            _throughput(state, frame_count=frame_count, step_s=float(metrics["step_s"]))
        )
        metrics["peak_reserved_gib"] = _peak_reserved_gib(state.config.device)
        metrics["peak_allocated_gib"] = _peak_allocated_gib(state.config.device)
        if built:
            # Prefill is a one-off, so it rides on the step that paid for it
            # rather than being lost to a sink that only sees steps.
            metrics.update(state.metrics)
        # One rank reports. Sixteen copies of the same line is sixteen times
        # the noise and none of the information.
        if state.config.parallel.is_main:
            _log_step(
                state, step_index=step_index, frame_count=frame_count, metrics=metrics
            )
        return [
            StepResult(
                step_index=step_index,
                output=channel,
                frame_count=frame_count,
                output_layout=state.session_desc.output_layout,
                metrics=metrics if index == 0 else None,
            )
            for index, channel in enumerate(frames)
        ]

    def _chunk(
        self, state: Cosmos3NanoSimAutoModelState
    ) -> tuple[tuple[Tensor, ...] | None, dict[str, float | int]]:
        """Generate a chunk and decode it on the presenting rank.

        **A rank that is not rank 0 stops after generating.** The chunk is
        identical on every rank -- the rollout gathers it before returning -- so
        the decode a worker would run is a copy of one rank 0 is already running,
        and the mp4 it would write is a copy of the file rank 0 is already
        writing. It publishes no channel instead, which the v2 runtime reads as a
        step that presented nothing. The runtime creates no client window on
        workers. That is why every rank of a launch can be
        given the same ``--output-path``.

        Returns:
            The chunk's frames and the split the step took, or ``None`` frames on
            a rank that generated without presenting.
        """
        rollout = state.rollout
        assert rollout is not None

        with _timed(state.config.device, "joint/generate") as elapsed:
            trace, _tokens = rollout.step()

        metrics: dict[str, float | int] = {
            "chunk_start": trace.start,
            "chunk_end": trace.end,
            "denoising_passes": trace.denoising_passes,
            # MetricsOutputSink intentionally rejects booleans. Persist this as
            # a numeric marker so benchmark probes can exclude the cheaper
            # terminal chunk, which skips the cache-populating replay.
            "clean_pass": int(trace.clean_pass),
            "memory_tokens": trace.memory_tokens,
            "generate_s": elapsed(),
        }
        if isinstance(rollout, JointChunkRollout):
            metrics.update(rollout.last_metrics)
            metrics["lidar_enabled"] = 1
        parallel = state.config.parallel
        decoding = parallel if state.config.parallel_decode else None
        if decoding is None and not parallel.is_main:
            return None, metrics

        # The conditioning frames are real footage that belongs in the output and
        # no chunk generated them, so the first decode starts at 0 rather than at
        # this chunk's start.
        with _timed(state.config.device, "rgb/decode") as elapsed:
            latent = rollout.latent_for(state.decoded_to, trace.end)
            state.decoded_to = trace.end
            pixels = decoded_views(latent, state.decode, parallel=decoding)
        metrics["decode_s"] = elapsed()
        if not parallel.is_main:
            # A worker that decoded its cameras and sent them on. Nothing to
            # publish here either; see the docstring.
            return None, metrics

        with _timed(state.config.device) as elapsed:
            frames = (
                pixels.unbind(0)
                if state.config.ui
                else (_framed(pixels, camera=state.config.camera, sample=state.sample),)
            )
        metrics["emit_s"] = elapsed()
        if state.lidar_stream is not None:
            assert isinstance(rollout, JointChunkRollout)
            assert rollout.last_lidar is not None
            from flashdreams.core.visualization.lidar import LidarViewport

            with _timed(state.config.device, "lidar/decode") as elapsed:
                ranges = state.lidar_stream.ranges(rollout.last_lidar)
            metrics["lidar_decode_s"] = elapsed()
            with _timed(state.config.device, "lidar/render") as elapsed:
                viewport = LidarViewport(
                    width=state.sample.view_size[0], height=frames[0].shape[-2]
                )
                lidar_frames = state.lidar_stream.frames(
                    ranges,
                    first_sweep=rollout.last_lidar_range[0],
                    pixel_start=state.frames_emitted,
                    pixel_count=int(pixels.shape[1]),
                    fps=state.config.fps,
                    viewport=viewport,
                    view=state.lidar_view,
                )
                lidar_frames = lidar_frames.to(frames[0]).mul(2).sub(1)
                frames = (
                    (*frames, lidar_frames)
                    if state.config.ui
                    else (torch.cat((frames[0], lidar_frames), dim=-1),)
                )
            metrics["lidar_render_s"] = elapsed()

        state.frames_emitted += int(frames[0].shape[0])
        metrics["frames_emitted"] = state.frames_emitted
        return frames, metrics

    def is_finished(self) -> bool:
        """Whether the clip has run out, which is the only ending a replay has.

        The clip's span is an ending it cannot help: there are no recorded
        actors or light states past the twenty seconds it holds.
        """
        state = self.state
        return state.rollout is not None and state.rollout.is_finished

    def reset(self) -> None:
        """Drop the rollout and the decoder, keeping the loaded network."""
        state = self.state
        state.rollout = None
        # The decoder carries a temporal cache between calls, so a new clip needs
        # a new one rather than the tail of the last.
        state.decode = None
        if state.lidar_stream is not None:
            state.lidar_stream.close()
        state.lidar_stream = None
        state.decoded_to = 0
        state.frames_emitted = 0
        # A new clip refills the cache, so the old clip's steady window says
        # nothing about it.
        state.steady_started_at = None
        state.steady_frames = 0

    def close(self) -> None:
        """Release the rollout and decoder without a collective."""
        self.reset()


class Cosmos3NanoSimAutoSession(ISession):
    """One multiview clip, generated a chunk at a time.

    The session decides the output size rather than taking whatever the runtime
    asked for, since a frame is the model's size and a mosaic is eleven of those
    in a grid. :attr:`session_desc` reports the real dimensions back, and the
    output sink opens with those.
    """

    def __init__(
        self,
        config: Cosmos3NanoSimAutoConfig,
        session_desc: SessionDesc,
        sample: MultiviewSample,
    ) -> None:
        """
        Args:
            config: resolved settings from the owning application.
            session_desc: what the runtime asked for. The layout is honoured; the
                dimensions and step rate are replaced with the model's.
            sample: the clip to generate, already trimmed to ``config.views``.

        Raises:
            ValueError: the requested layout is not ``tchw``.
        """
        if session_desc.output_layout is not VideoTensorLayout.tchw:
            raise ValueError(
                "cosmos3-nano-sim-auto produces tchw output, got "
                f"{session_desc.output_layout.value}."
            )
        self._config = config
        self._sample = sample
        width, height = sample.view_size
        if config.ui:
            width, height = session_desc.video_width, session_desc.video_height
        elif config.camera is None:
            columns, rows = _mosaic_grid(config.views)
            width, height = width * columns, height * rows
        if config.lidar is not None and not config.ui:
            width += sample.view_size[0]
        self._session_desc = replace(
            session_desc,
            video_width=width,
            video_height=height,
            frames_per_second_for_step=round(config.fps),
        )
        self._network: CosmosMoTNetwork | None = None

    def init(self) -> None:
        """Load the checkpoint and register the model loop; the default UI blits it."""
        source = resolve_checkpoint_source(
            self._config.checkpoint, default=DEFAULT_CHECKPOINT
        )
        ctx = self._config.parallel
        if ctx.tensor_parallel:
            # Host first, then cut, then move this rank's slice. Loading to the
            # card and sharding after would put all 28.2 GiB on every GPU, which
            # defeats the axis on exactly the card that needs it.
            network = load_exported_checkpoint(source, device="cpu").eval()
            shard_network(network, ctx)
            self._network = network.to(ctx.device)
        else:
            self._network = load_exported_checkpoint(
                source, device=self._config.device
            ).eval()
        if (lidar := self._config.lidar) is not None and hasattr(
            self._network, "lidar2llm"
        ):
            self._config = replace(
                self._config,
                lidar=replace(
                    lidar,
                    tokenizer=lidar.tokenizer.with_patch_dim(
                        self._network.lidar2llm.in_features
                    ),
                ),
            )
        state = Cosmos3NanoSimAutoModelState(
            config=self._config,
            session_desc=self._session_desc,
            sample=self._sample,
            network=self._network,
            lidar_view=self._config.lidar_view,
        )
        model_loop = self.register_model_loop(Cosmos3NanoSimAutoModelLoop, state=state)
        if self._config.ui and ctx.is_main:
            from interactive_drive.multiview import AuxiliaryView

            from cosmos3_nano_sim_auto.impl.ui import (
                CosmosMultiviewUILoop,
                CosmosMultiviewUIState,
                camera_views,
            )

            cameras = camera_views(self._sample)
            state.ui_loop = self.register_ui_loop(
                CosmosMultiviewUILoop,
                state=CosmosMultiviewUIState(
                    model_loop=model_loop,
                    title="Cosmos 3 MV Auto",
                    prompt=self._sample.prompt,
                    scene_options=(),
                    cameras=cameras,
                    auxiliary_views=(
                        AuxiliaryView("lidar", "LiDAR", "Generated LiDAR"),
                    )
                    if self._config.lidar is not None
                    else (),
                    lidar_view=self._config.lidar_view,
                    visible_count=len(cameras),
                    focus_channel=self._sample.cameras.index(self._config.camera)
                    if self._config.camera
                    else -1,
                    status="Replaying recorded conditioning. Driving inputs are disabled.",
                ),
                width=self._session_desc.video_width,
                height=self._session_desc.video_height,
                cuda_device=self._config.device,
            )

    @property
    def parallel_context(self) -> ParallelContext:
        """Return the model mesh for runtime-owned step coordination."""
        return self._config.parallel

    @property
    def session_desc(self) -> SessionDesc:
        return self._session_desc

    def close(self) -> None:
        """Drop the network reference so the weights can be freed."""
        self._network = None


class Cosmos3NanoSimAutoApplication(IApplication):
    """Application generating a multiview Nano Sim Auto clip."""

    def __init__(self) -> None:
        self._config: Cosmos3NanoSimAutoConfig | None = None
        self._sample: MultiviewSample | None = None

    def session_desc(self) -> SessionDesc:
        """Ask for a session that presents on demand rather than every UI tick.

        By default the UI redraws at 60 Hz with the newest frame it has, which
        suits a display that must not stall. Against a file it writes the same
        frame once per tick for as long as a chunk takes: the first eleven-view
        MP4 was 107 seconds of mostly freeze for 1.2 seconds of video. On demand
        writes each generated frame exactly once.

        Only the pacing is asked for here. The frame size and rate follow the
        sample and the model, so they are the session's to resolve.
        """
        return SessionDesc(presentation_mode=PresentationMode.ON_DEMAND)

    def init(self, commandline_args: Sequence[str]) -> None:
        """Parse the arguments and load the sample, without touching the GPU.

        The sample is read here rather than in the session so that a bad path or
        a caption that does not parse fails before the runtime has opened a
        window and loaded 30 GB of weights.

        Raises:
            ValueError: an argument is out of range, or the sample has fewer
                cameras than ``--views`` asks for.
            FileNotFoundError: ``--sample`` does not hold a ``sample.json``.
        """
        args = _parse(commandline_args)
        if args.optimization_profile == RTX_PRO_6000_BLACKWELL_PERF_PROFILE:
            logger.warning(
                "selected the opt-in performance profile: it passed five of six "
                "quality thresholds but its seed-1 worst frame measured 26.350 "
                "dB PSNR against the 28 dB floor"
            )
        # Joins whatever torchrun set up, or returns the inert single-rank
        # context when nothing did -- so an un-launched run takes exactly the
        # path it took before this existed. Done here rather than in the session
        # because the device it settles on is what the config is built around.
        ctx = init_parallel()
        if ctx.world_size > 1 and args.device != "cuda":
            raise ValueError(
                f"--device {args.device!r} with {ctx.world_size} ranks: the device is "
                "the rank's, not the flag's. Drop --device, or run on one rank."
            )
        if ctx.world_size == 1 and args.device != "cuda":
            ctx = ParallelContext.single(args.device)
        sample = _sample_for(args)
        if args.views > sample.num_views:
            raise ValueError(
                f"--views {args.views} but the sample has {sample.num_views} camera(s)."
            )
        if args.loop_controls and (
            args.sample is None or args.draw_map is not None or not args.frames
        ):
            raise ValueError(
                "--loop-controls replays a sample's recorded control videos, so it "
                "needs --sample, a --frames to fill, and no --draw-map."
            )
        if args.draw_map is not None and args.sample is None:
            raise ValueError(
                "--draw-map needs --sample too. The drawing replaces the sample's "
                "control videos and nothing else: the prompt, the opening frames "
                "and the framing the drawing is timed against all still come from "
                "the matching prepared scene bundle."
            )
        if args.fused_attention_merge and not args.maskless_attention:
            raise ValueError("--fused-attention-merge needs --maskless-attention.")
        if args.camera is not None and args.camera not in sample.cameras[: args.views]:
            raise ValueError(
                f"--camera {args.camera!r} is not among the first {args.views} "
                f"cameras: {', '.join(sample.cameras[: args.views])}."
            )
        if args.frames == 0:
            args.frames = _frames_the_controls_allow(sample, args)
        self._config = Cosmos3NanoSimAutoConfig(
            checkpoint=args.checkpoint,
            sample_dir=Path(args.sample) if args.sample else None,
            views=args.views,
            frames=args.frames,
            fps=args.fps,
            seed=args.seed,
            optimization_profile=args.optimization_profile,
            camera=args.camera,
            # The context's device under a launcher, the flag's without one.
            # Every rank of a mesh must sit on its own card, and only the
            # context knows which that is.
            device=str(ctx.device) if ctx.world_size > 1 else args.device,
            attention_impl=_attention_impl(args),
            accelerated_rms_norm=args.accelerated_rms_norm,
            attention_kv_arena=args.attention_kv_arena,
            parallel=ctx,
            history_frames=args.history_frames,
            compile_denoise=args.compile_denoise,
            denoise_steps=args.denoise_steps,
            chunk_frames=args.chunk_frames,
            attention_scope=args.attention_scope,
            maskless_attention=args.maskless_attention,
            fused_attention_merge=args.fused_attention_merge,
            decomposed_temporal_window_seconds=args.decomposed_temporal_window_seconds
            or None,
            outlive_caption=args.outlive_caption or args.loop_controls,
            loop_controls=args.loop_controls,
            draw_map=args.draw_map,
            parallel_encode=args.parallel_encode,
            parallel_decode=args.parallel_decode,
            compile_control_encode=args.compile_control_encode,
            decoder_compile=args.decoder_compile,
            prefetch_controls=args.prefetch_controls,
            prefetch_controls_gpu_offset=args.prefetch_controls_gpu_offset,
            lidar_decode_worker=args.lidar_decode_worker,
            lidar_view=LidarView(args.lidar_view),
            ui=args.ui,
        )
        if args.lidar_sample is not None:
            from cosmos_interactive.lidar import LidarTokenizer

            from cosmos3_nano_sim_auto.impl.lidar import LidarAssets, LidarSample

            assert args.frames is not None
            lidar_sample = LidarSample.load(
                args.lidar_sample,
                start_seconds=sample.offset_for(
                    args.frames, outlive_caption=args.outlive_caption
                )
                / args.fps,
                duration_seconds=(args.frames - 1) / args.fps,
            )
            source = resolve_checkpoint_source(
                args.checkpoint, default=DEFAULT_CHECKPOINT
            )
            checkpoint = str(
                resolve_lidar_checkpoint(
                    source,
                    checkpoint=args.lidar_checkpoint,
                    version=args.lidar_tokenizer_version,
                )
            )
            tokenizer = LidarTokenizer.from_checkpoint(
                checkpoint,
                device=self._config.device,
                version=args.lidar_tokenizer_version,
            )
            self._config = replace(
                self._config,
                checkpoint=source,
                lidar=LidarAssets(
                    tokenizer,
                    lidar_sample,
                    checkpoint=checkpoint,
                    version=args.lidar_tokenizer_version,
                ),
            )
        self._sample = sample.trimmed_to(args.views)

    def create_session(self, session_desc: SessionDesc) -> ISession:
        """Create one uninitialized multiview session.

        Raises:
            RuntimeError: :meth:`init` has not run yet.
        """
        if self._config is None or self._sample is None:
            raise RuntimeError(
                "Cosmos3NanoSimAutoApplication.init() must run before create_session()."
            )
        return Cosmos3NanoSimAutoSession(self._config, session_desc, self._sample)


def create_app() -> IApplication:
    """Return a new multiview Nano Sim Auto application."""
    return Cosmos3NanoSimAutoApplication()


def mosaic(views: Tensor, *, columns: int, rows: int) -> Tensor:
    """``[V, T, 3, H, W]`` per camera -> ``[T, 3, H*rows, W*columns]``.

    Cameras fill the grid in rig order, left to right and then top to bottom.
    Unused cells are black rather than an error: eleven views do not divide into
    a rectangle, and the twelfth cell being empty is the honest picture of that.
    """
    if views.ndim != 5:
        raise ValueError(f"expected [V, T, 3, H, W] frames, got {tuple(views.shape)}.")
    count, frames, channels, height, width = views.shape
    if count > columns * rows:
        raise ValueError(f"{count} views do not fit a {columns}x{rows} grid.")
    grid = views.new_full((rows * columns, frames, channels, height, width), -1.0)
    grid[:count] = views
    return (
        grid.reshape(rows, columns, frames, channels, height, width)
        # -> [T, C, rows, H, columns, W], so the two spatial merges are adjacent.
        .permute(2, 3, 0, 4, 1, 5)
        .reshape(frames, channels, rows * height, columns * width)
        .contiguous()
    )


def torn(frame: Tensor, *, views: int, columns: int, rows: int) -> Tensor:
    """``[T, 3, H*rows, W*columns]`` -> ``[V, T, 3, H, W]``, :func:`mosaic` undone.

    What a run wrote is a mosaic, so anything that wants a camera back out of
    one has to know how the cells were filled. Keeping that here means the two
    halves of the convention cannot drift apart: cells past ``views`` are the
    black ones the tiling left over and are dropped rather than returned.
    """
    if frame.ndim != 4:
        raise ValueError(f"expected [T, 3, H, W] frames, got {tuple(frame.shape)}.")
    frames, channels, tall, wide = frame.shape
    if tall % rows or wide % columns:
        raise ValueError(f"{wide}x{tall} does not divide into a {columns}x{rows} grid.")
    if views > columns * rows:
        raise ValueError(f"{views} views do not fit a {columns}x{rows} grid.")
    height, width = tall // rows, wide // columns
    return (
        frame.reshape(frames, channels, rows, height, columns, width)
        # -> [rows, columns, T, C, H, W], so the cells merge into one view axis.
        .permute(2, 4, 0, 1, 3, 5)
        .reshape(rows * columns, frames, channels, height, width)[:views]
        .contiguous()
    )


def _mosaic_grid(views: int) -> tuple[int, int]:
    """Columns and rows for ``views`` tiles, as close to square as they go."""
    columns = math.ceil(math.sqrt(views))
    return columns, math.ceil(views / columns)


def _framed(pixels: Tensor, *, camera: str | None, sample: MultiviewSample) -> Tensor:
    """One step's decoded cameras as a single ``[T, 3, H, W]`` channel."""
    if camera is None:
        columns, rows = _mosaic_grid(int(pixels.shape[0]))
        return mosaic(pixels, columns=columns, rows=rows)
    return pixels[sample.cameras.index(camera)].contiguous()


def _build(
    state: Cosmos3NanoSimAutoModelState,
) -> tuple[ChunkRollout | JointChunkRollout, Callable[[Tensor], Tensor] | None]:
    """Encode and prefill on every rank; build the decoder only on rank 0.

    Deferred to the first step rather than done in ``init``: this is where the
    control encode and the whole-clip prefill happen, so their cost is included
    in the first step's metrics.

    Timed by stage into ``state.metrics``. Prefill is a third of a short run and
    was one opaque number, which left no way to tell the video read from the VAE
    encode from the attention pass that is the only part a stream repeats.
    """
    config = state.config
    sample = state.sample
    network = state.network
    spec = network.config.spec
    stage = _stage_timer(state)
    encoding = config.parallel if config.parallel_encode else None
    local_views = owned_views(encoding, sample.num_views)
    geometry = sample.geometry(
        pixel_frames=config.frames,
        latent_patch_size=spec.vfm.latent_patch_size,
        frames_per_chunk=config.chunk_frames or spec.vfm.ar_chunk_latent_frames,
        fps=config.fps,
    )
    logger.info(
        "cosmos3-nano-sim-auto attention backend={} scope={} mask_block_size={} "
        "kernel_options={} attention_merge_fused={}",
        "maskless"
        if config.maskless_attention
        else config.attention_impl.sdpa_backend.value,
        config.attention_scope,
        config.attention_impl.flex_attention.mask_block_size,
        config.attention_impl.flex_attention.kernel_options,
        config.fused_attention_merge,
    )
    with stage("vae_load_s"):
        encode = make_control_encode_fn(
            dtype=network.config.dtype, device=config.device
        )
    view_width, view_height = sample.view_size
    # Both tracks are cut at the caption's own start frame, so the pixels are the
    # ones the eleven captions were chosen to describe.
    start_frame = sample.offset_for(
        config.frames, outlive_caption=config.outlive_caption
    )
    _warn_if_outliving_caption(sample, config)
    if config.sample_dir is not None:
        # Every rank checks every camera's files before reading only its own:
        # a missing one then fails the whole mesh at once, where a check on the
        # owner alone leaves the other ranks waiting on its broadcast.
        require_local_files(
            (() if config.draw_map is not None else sample.control_paths)
            + (sample.image_paths if geometry.condition_frames else ())
        )
    controls: Tensor | ControlProvider
    if config.draw_map is not None:
        # Nothing is encoded here. The provider draws and encodes a chunk when the
        # rollout asks for it, so this stage is the scene upload and the read of
        # the clip's geometry -- a fixed cost, where the file path's read grows
        # with the length of the run.
        if config.prefetch_controls:
            from cosmos3_nano_sim_auto.impl.conditioning.control_worker import (
                ControlWorker,
                WorkerSpec,
                worker_gpu,
            )

            # The scene upload and the encoder load both happen in the worker,
            # so one stage covers the two; the main process holds neither.
            with stage("control_read_s", "starting the control worker"):
                device = torch.device(config.device)
                worker = ControlWorker.spawn(
                    WorkerSpec(
                        clip=Path(config.draw_map),
                        cameras=tuple(sample.cameras[: config.views]),
                        owned=tuple(local_views),
                        device=worker_gpu(device, config.prefetch_controls_gpu_offset),
                        against=None
                        if config.sample_dir is None
                        else Path(config.sample_dir),
                        start_frame=start_frame,
                        wanted=config.frames,
                        dtype=network.config.dtype,
                        compile_encode=config.compile_control_encode,
                        requested_fps=config.fps,
                    ),
                    device=device,
                )
            state.metrics["control_encode_s"] = 0.0
            controls = RenderedControls(
                None,
                [lambda clip: clip] * geometry.num_views,
                geometry=geometry,
                latent_patch_size=spec.vfm.latent_patch_size,
                parallel=encoding,
                worker=worker,
            )
            logger.info(
                "cosmos3-nano-sim-auto control prefetch on: chunk n+1 is drawn and "
                "encoded in a worker process while chunk n denoises"
            )
        else:
            with stage("control_read_s", "reading the clip and uploading its scene"):
                from cosmos3_nano_sim_auto.impl.conditioning import ludus_controls

                drawn = ludus_controls.open_clip(
                    config.draw_map,
                    sample.cameras[: config.views],
                    device=config.device,
                    against=config.sample_dir,
                    start_frame=start_frame,
                    wanted=config.frames,
                    note=logger.info,
                    requested_fps=config.fps,
                )
            # Nothing is encoded in this one either; it is the streamed encoder's own
            # weights, and the encoding itself happens in whichever chunk asks.
            with stage("control_encode_s", "loading the per-camera control encoder"):
                controls = RenderedControls(
                    drawn.render,
                    make_control_encode_streams(
                        geometry.num_views,
                        dtype=network.config.dtype,
                        device=config.device,
                        compile=config.compile_control_encode,
                    ),
                    geometry=geometry,
                    latent_patch_size=spec.vfm.latent_patch_size,
                    parallel=encoding,
                )
    else:
        assert config.frames is not None
        with stage("control_read_s"):
            clips = (
                loaded_control_clips(
                    [sample.control_paths[v] for v in local_views],
                    pixel_frames=config.frames,
                    pixel_height=view_height,
                    pixel_width=view_width,
                    start_frame=start_frame,
                    loop=config.loop_controls,
                    # The whole rig's shortest, so every camera loops on the
                    # same frame even where a rank reads only its own.
                    loop_frames=available_pixel_frames(
                        sample.control_paths, start_frame=start_frame
                    )
                    if config.loop_controls
                    else None,
                )
                if config.sample_dir is not None
                else [
                    generated_control_clips(sample, pixel_frames=config.frames)[v]
                    for v in local_views
                ]
            )
        with stage("control_encode_s"):
            controls = control_tokens(
                control_latents(clips, encode, geometry=geometry, parallel=encoding),
                latent_patch_size=spec.vfm.latent_patch_size,
            )

    conditions = None
    if geometry.condition_frames:
        with stage("condition_s"):
            opening = loaded_control_clips(
                [sample.image_paths[v] for v in local_views],
                pixel_frames=condition_pixel_frames(geometry),
                pixel_height=view_height,
                pixel_width=view_width,
                start_frame=start_frame,
            )
            conditions = control_tokens(
                condition_latents(
                    opening, encode, geometry=geometry, parallel=encoding
                ),
                latent_patch_size=spec.vfm.latent_patch_size,
            )

    with stage("tokenize_s"):
        prompt = prompt_token_ids(
            sample,
            pixel_frames=_prompt_frames(sample, config),
            lidar=config.lidar is not None,
            fps=config.fps,
            parallel=config.parallel,
        )
    if config.lidar is not None:
        state.lidar_stream = config.lidar.stream()
        if config.lidar_decode_worker and config.parallel.is_main:
            with stage("lidar_worker_s", "starting the LiDAR decode worker"):
                device = torch.device(config.device)
                mine = (
                    torch.cuda.current_device()
                    if device.index is None
                    else device.index
                )
                state.lidar_stream.decode_in_worker(
                    worker_device=(mine + config.prefetch_controls_gpu_offset)
                    % torch.cuda.device_count(),
                    device=device,
                )
        with stage("lidar_condition_s", "encoding the opening LiDAR sweep"):
            lidar_condition = state.lidar_stream.condition_tokens()
        with stage("prefill_pass_s"):
            rollout = JointChunkRollout(
                network,
                geometry=geometry,
                control_tokens=controls,
                lidar_controls=state.lidar_stream,
                lidar_tokenizer=config.lidar.tokenizer,
                condition_tokens=conditions,
                lidar_condition=lidar_condition,
                text_ids=prompt.text_ids,
                view_text_tokens=prompt.view_text_tokens,
                fps=config.fps,
                seed=config.seed,
                attention_impl=config.attention_impl,
                accelerated_rms_norm=config.accelerated_rms_norm,
                history_frames=config.history_frames,
                t_list=denoise_schedule(
                    spec.vfm.fixed_step_t_list, config.denoise_steps
                ),
                decomposed_temporal_window_seconds=config.decomposed_temporal_window_seconds,
                fused_attention_merge=config.fused_attention_merge,
                compile_denoise=config.compile_denoise,
                parallel=config.parallel,
            )
    else:
        with stage("prefill_pass_s"):
            rollout = ChunkRollout(
                network,
                geometry=geometry,
                control_tokens=controls,
                condition_tokens=conditions,
                text_ids=prompt.text_ids,
                view_text_tokens=prompt.view_text_tokens,
                fps=config.fps,
                seed=config.seed,
                attention_impl=config.attention_impl,
                accelerated_rms_norm=config.accelerated_rms_norm,
                attention_kv_arena=config.attention_kv_arena,
                history_frames=config.history_frames,
                # The same window, because the app decodes each chunk as it arrives
                # and so never looks further back than the cache does. One knob
                # bounds both, and `--history-frames 0` leaves both unbounded.
                token_frames=config.history_frames,
                compile_denoise=config.compile_denoise,
                t_list=denoise_schedule(
                    spec.vfm.fixed_step_t_list, config.denoise_steps
                ),
                attention_scope=config.attention_scope,
                maskless_attention=config.maskless_attention,
                fused_attention_merge=config.fused_attention_merge,
                decomposed_temporal_window_seconds=config.decomposed_temporal_window_seconds,
                parallel=config.parallel,
            )
    decode = None
    decoding = config.parallel if config.parallel_decode else None
    decode_views = len(owned_views(decoding, geometry.num_views))
    if decode_views and (decoding is not None or config.parallel.is_main):
        with stage("decoder_load_s"):
            decode = make_multiview_chunk_decoder(
                num_views=decode_views,
                dtype=network.config.dtype,
                device=config.device,
                compile=config.decoder_compile,
            )
    return rollout, decode


_STAGE_SAYS = {
    "vae_load_s": "loading the control encoder",
    "control_read_s": "reading the controls",
    "control_encode_s": "encoding the controls",
    "condition_s": "reading the opening frames",
    "tokenize_s": "tokenizing the prompt",
    "prefill_pass_s": "prefilling text and opening control",
    "decoder_load_s": "loading the decoder",
}
"""What each stage is doing, said before it starts rather than after.

Startup is minutes and every stage in it was silent until its own split line,
which prints once they have all finished -- so a run that stopped in one of them
looked the same as a run that was merely slow, and the way to tell them apart
was a stack dump. Said on the way in, the last line names where a wait is.
"""


def _stage_timer(
    state: Cosmos3NanoSimAutoModelState,
) -> Callable[..., AbstractContextManager[None]]:
    """A context manager per named stage, each recording into ``state.metrics``.

    Announces the stage as it opens, so a long one says what it is before it
    spends the time. What it took is left to the split line: two lines a stage
    doubles the startup's output to repeat what one line already says.
    """

    @contextmanager
    def stage(name: str, says: str | None = None) -> Iterator[None]:
        logger.info("cosmos3-nano-sim-auto {}...", says or _STAGE_SAYS.get(name, name))
        with _timed(state.config.device) as elapsed:
            yield
        state.metrics[name] = elapsed()

    return stage


def _cuda_available(device: str) -> bool:
    return device.startswith("cuda") and torch.cuda.is_available()


@contextmanager
def _timed(device: str, label: str | None = None) -> Iterator[Callable[[], float]]:
    """Time a block of GPU work, waiting for it rather than for its launch.

    CUDA work is queued and not waited for, so timing a block on the host
    measures how long it took to ask, and the work itself lands on whichever
    later block happens to wait. Without the waits here, the split between
    prefill, generation and decode cannot be trusted even when the total can.

    Yields:
        A callable returning the elapsed seconds, valid after the block exits.
    """
    if _cuda_available(device):
        torch.cuda.synchronize()
    started = time.perf_counter()
    elapsed = float("nan")
    if label is not None and _cuda_available(device):
        torch.cuda.nvtx.range_push(label)
    try:
        yield lambda: elapsed
    finally:
        if _cuda_available(device):
            torch.cuda.synchronize()
            if label is not None:
                torch.cuda.nvtx.range_pop()
        elapsed = time.perf_counter() - started


def _prompt_frames(sample: MultiviewSample, config: Cosmos3NanoSimAutoConfig) -> int:
    """The duration the prompt states, which a run with no end has not got.

    Training's caption payload carried the clip's duration, so the payload has
    to state one whatever the run does. A run with no end states the span its
    captions describe: those are the frames the words were written against, so
    it is the pairing of text and duration that training would have had.
    """
    if config.frames is not None:
        return config.frames
    if sample.frame_range is None:
        return DEFAULT_FRAMES
    start, end = sample.frame_range
    return end - start


def _warn_if_outliving_caption(
    sample: MultiviewSample, config: Cosmos3NanoSimAutoConfig
) -> None:
    """Say so when the run is longer than its prompt describes.

    ``offset_for`` refuses this by default, and the opt-out exists to measure
    what the refusal was protecting against. A run that quietly went
    off-distribution would defeat the measurement as much as the mistake.

    A run with no end outlives every caption there is, since a caption covers a
    span and this run has none. There is no flag to ask for that and nothing to
    refuse, so it is simply said.
    """
    if sample.frame_range is None:
        return
    start, end = sample.frame_range
    described = end - start
    if config.frames is None:
        logger.warning(
            "cosmos3-nano-sim-auto is generating with no end on captions that describe "
            "{} frames ({:.1f}s of prompt); the prompt goes stale past frame {}.",
            described,
            described / config.fps,
            described,
        )
        return
    if not config.outlive_caption or config.frames <= described:
        return
    logger.warning(
        "cosmos3-nano-sim-auto is generating {} frames on captions that describe {} "
        "({:.1f}s of prompt for {:.1f}s of video); the prompt goes stale past "
        "frame {}.",
        config.frames,
        described,
        described / config.fps,
        config.frames / config.fps,
        described,
    )


def _log_prefill(state: Cosmos3NanoSimAutoModelState) -> None:
    """Say what the run is and how many chunks it will take, before the wait.

    The runtime itself logs nothing per step, and eleven views spend minutes
    inside one, so without this the demo looks hung between loading the weights
    and writing the file. The stage split follows on its own line because only
    one of those stages is work a stream would repeat.
    """
    rollout = state.rollout
    assert rollout is not None
    geometry = rollout.geometry
    metrics = state.metrics
    logger.info(
        "cosmos3-nano-sim-auto prefilled {} view(s) x {} pixel frames "
        "({} latent, {} chunk(s), {} conditioned) "
        "in {:.1f}s, peak {}",
        geometry.num_views,
        "no end" if state.config.frames is None else state.config.frames,
        "no end" if geometry.frames_per_view is None else geometry.frames_per_view,
        rollout.chunk_count if rollout.chunk_count is not None else "no end",
        geometry.condition_frames,
        metrics["prefill_s"],
        _peak_reserved(state.config.device),
    )
    stages = [
        ("vae", "vae_load_s"),
        ("read", "control_read_s"),
        ("encode", "control_encode_s"),
        ("condition", "condition_s"),
        ("lidar condition", "lidar_condition_s"),
        ("tokenize", "tokenize_s"),
        ("pass", "prefill_pass_s"),
        ("decoder", "decoder_load_s"),
    ]
    named = [(label, metrics[key]) for label, key in stages if key in metrics]
    accounted = sum(float(value) for _label, value in named)
    logger.info(
        "cosmos3-nano-sim-auto prefill split: {}, other={:.1f}s",
        ", ".join(f"{label}={float(value):.1f}s" for label, value in named),
        float(metrics["prefill_s"]) - accounted,
    )


def _log_worker(state: Cosmos3NanoSimAutoModelState) -> None:
    """Say once, on a rank nobody is watching, that its silence is on purpose.

    Written because the alternative is a rank whose log ends after the checkpoint
    load and never mentions a chunk again, which reads exactly like a hang. One
    line, at the prefill, rather than one per step: sixteen copies of every chunk
    line was the noise `_log_step` is already main-only to avoid.
    """
    ctx = state.config.parallel
    logger.info(
        "cosmos3-nano-sim-auto rank {} of {} on the {} mesh generates without presenting: "
        "the chunk is identical on every rank, so the decode and the file are "
        "rank 0's alone. Prefilled in {:.1f}s, peak {}",
        ctx.rank,
        ctx.world_size,
        ctx.label,
        state.metrics["prefill_s"],
        _peak_reserved(state.config.device),
    )


def _log_step(
    state: Cosmos3NanoSimAutoModelState,
    *,
    step_index: int,
    frame_count: int,
    metrics: Mapping[str, float | int],
) -> None:
    """Log one chunk's progress and wall-time split.

    ``other`` is the step's own total less generate and decode, so a cost that
    belongs to neither shows up here rather than only in the gap between two log
    timestamps. Tiling the views is one such cost, and has always been too small
    to print a column for; ``--stats-path`` still records it separately.
    """
    rollout = state.rollout
    assert rollout is not None
    parts = ("generate_s", "decode_s")
    actual = metrics.get("steady_state_fps")
    logger.info(
        "cosmos3-nano-sim-auto step={} chunk {}/{} latent={}-{} frames={} "
        "fps={:.2f} actual={} "
        "generate={:.1f}s decode={:.1f}s other={:.1f}s peak={}",
        step_index,
        rollout.chunks_done,
        rollout.chunk_count if rollout.chunk_count is not None else "no end",
        metrics["chunk_start"],
        metrics["chunk_end"],
        frame_count,
        metrics["chunk_fps"],
        # "filling" and not a number, because the cache is still growing and any
        # average over these chunks would read faster than the run really is.
        f"{actual:.2f}" if actual is not None else "filling",
        metrics["generate_s"],
        metrics["decode_s"],
        float(metrics["step_s"]) - sum(float(metrics[part]) for part in parts),
        _peak_reserved(state.config.device),
    )


def _throughput(
    state: Cosmos3NanoSimAutoModelState, *, frame_count: int, step_s: float
) -> dict[str, float | int]:
    """Generated frames per second, by two measures.

    ``chunk_fps`` divides a chunk's frames by its own step, so it reports what
    the model manages. ``steady_state_fps`` divides by wall clock since the
    cache filled, so it also includes the time between steps, and is the rate a
    viewer sees.

    The steady-state rate waits for the cache to fill, because the chunks before
    that are cheaper and averaging them in flatters a long run.

    The ``rig`` variants multiply by the number of cameras, for comparing
    against a one-camera demo. Only ``--stats-path`` reports them, being a
    constant times a rate already logged.
    """
    rollout = state.rollout
    assert rollout is not None
    views = state.config.views
    per_second = frame_count / max(step_s, 1e-9)
    metrics: dict[str, float | int] = {
        "chunk_fps": per_second,
        "chunk_rig_fps": views * per_second,
    }

    if state.steady_started_at is None and rollout.chunks_done > rollout.warmup_chunks:
        # Opened at this step's start rather than its end, so the first steady
        # chunk counts inside the window it opens.
        state.steady_started_at = time.perf_counter() - step_s
    if state.steady_started_at is None:
        return metrics

    state.steady_frames += frame_count
    elapsed = max(time.perf_counter() - state.steady_started_at, 1e-9)
    metrics["steady_state_fps"] = state.steady_frames / elapsed
    metrics["steady_state_rig_fps"] = views * state.steady_frames / elapsed
    return metrics


def _peak_reserved_gib(device: str) -> float:
    """The high-water mark of reserved GPU memory in GiB, or NaN off CUDA."""
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return float("nan")
    return torch.cuda.max_memory_reserved(device) / 2**30


def _peak_allocated_gib(device: str) -> float:
    """The high-water mark of allocated (live) GPU memory in GiB, or NaN off CUDA.

    Reported beside ``peak_reserved_gib`` rather than instead of it: reserved is
    what has to fit and is never given back, but it is the allocator's
    high-water mark, not live bytes. The gap between the two is the portion of
    ``peak_reserved_gib`` that is fragmentation/headroom rather than tensors
    actually held at the peak moment.
    """
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return float("nan")
    return torch.cuda.max_memory_allocated(device) / 2**30


def _peak_reserved(device: str) -> str:
    """The high-water mark of reserved GPU memory, as a printable string.

    Reserved rather than allocated, because reserved is what has to fit: PyTorch
    does not hand pages back, so a brief spike holds the memory for the rest of
    the run. It is also what ``nvidia-smi`` shows. Never reset, so a late line
    still reports the prefill spike that sized the run.
    """
    peak = _peak_reserved_gib(device)
    return "n/a" if math.isnan(peak) else f"{peak:.1f} GiB"


def _sample_for(args: argparse.Namespace) -> MultiviewSample:
    """Load an explicit local sample or the prepared demonstration scene."""
    if args.draw_map is not None:
        args.draw_map = prepare_scene(args.draw_map)
    if args.sample is None:
        args.draw_map = args.draw_map or prepare_scene(DEFAULT_SCENE)
        args.sample = str(args.draw_map)
    record = Path(args.sample) / "sample.json"
    if not record.is_file():
        raise FileNotFoundError(
            f"{record} not found. Extract a prepared scene bundle and pass --scene PATH or --sample PATH."
        )
    if args.draw_map is not None and not (args.draw_map / "scene.json").is_file():
        raise ValueError("--scene/--draw-map requires a prepared scene.json bundle")
    sample = MultiviewSample.load(record)
    if args.draw_map is not None and getattr(args, "views_defaulted", False):
        args.views = min(args.views, sample.num_views)
    return sample


def _frames_the_controls_allow(
    sample: MultiviewSample, args: argparse.Namespace
) -> int | None:
    """How long a run is when ``--frames 0`` leaves it to the control source.

    A source that ends gives the run its length. Two things can end: the
    control tracks stop, and the captions describe a span. The shorter wins,
    except that ``--outlive-caption`` says the caption is not a bound, which is
    the same thing it says about a declared length.

    The map under ``--draw-map`` is not read here, since opening a clip is GPU
    work. A recorded one is checked against this length when it does open, the
    same as it is against a declared one.

    Returns:
        The frames the source holds.

    Raises:
        ValueError: the controls are generated, which are not measured: they
            are drawn as one track to whatever length is asked for, so there is
            nothing here to ask.
    """
    if args.sample is None:
        raise ValueError(
            "--frames 0 takes the length from the controls, and generated "
            "controls have none: they are drawn as one track to fit whatever is "
            "asked for. Pass --sample or give --frames a number."
        )
    start = 0 if sample.frame_range is None else sample.frame_range[0]
    if args.draw_map is not None:
        from .conditioning.world_scenario import prepared_scene

        scene = prepared_scene.load(args.draw_map)
        frames = len(scene.frame_mapping(0, None))
        frames = 1 + 4 * ((frames - 1) // 4)
    else:
        frames = available_pixel_frames(
            sample.control_paths[: args.views], start_frame=start
        )
    bound = "the controls"
    if sample.frame_range is not None and not args.outlive_caption:
        described = sample.frame_range[1] - sample.frame_range[0]
        if described < frames:
            frames = 1 + 4 * ((described - 1) // 4)
            bound = "the captions"
    logger.info(
        "cosmos3-nano-sim-auto --frames 0: {} end this run at {} frames ({:.1f}s at {:g}fps)",
        bound,
        frames,
        frames / args.fps,
        args.fps,
    )
    return frames


def _attention_impl(args: argparse.Namespace) -> OptimizedImplConfig:
    """Build the shared accelerated-attention policy from CLI controls."""
    if args.optimization_profile in _RTX_PRO_6000_BLACKWELL_PROFILES:
        return rtx_pro_6000_blackwell_attention_implementation(args.views)
    return attention_implementation(flex=args.flex_attention)


def _resolve_optimization_profile(args: argparse.Namespace) -> None:
    """Resolve a named profile as one measured, indivisible configuration."""
    perf = args.optimization_profile == RTX_PRO_6000_BLACKWELL_PERF_PROFILE
    if args.accelerated_rms_norm is None:
        # On by default; the measured profiles pin their own setting.
        args.accelerated_rms_norm = (
            perf or args.optimization_profile not in _RTX_PRO_6000_BLACKWELL_PROFILES
        )
    if args.optimization_profile in _RTX_PRO_6000_BLACKWELL_PROFILES:
        args.compile_denoise = "default"
        if perf:
            # Measured on the older checkpoint, whose chunk was the spec's four.
            args.denoise_steps, args.history_frames = 4, 12
            args.chunk_frames = 4
    elif args.compile_denoise == "off":
        args.compile_denoise = None


def _parse(commandline_args: Sequence[str]) -> argparse.Namespace:
    """Parse and range-check the application's arguments."""
    parser = argparse.ArgumentParser(
        prog="cosmos3-nano-sim-auto",
        description="Generate a multiview Nano Sim Auto clip from HD map controls.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--sample",
        default=None,
        help=(
            "Prepared local input directory holding sample.json, the "
            "control clips and the RGB footage. Omitted, the demo runs on "
            "generated controls and the vendored prompt."
        ),
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--views", type=int, default=None)
    parser.add_argument(
        "--frames",
        type=int,
        default=DEFAULT_FRAMES,
        help=(
            "Pixel frames per view to generate, on the 1 + 4n grid the encoder "
            "wants. 0 leaves the length to the controls: the run goes as far as "
            "the control tracks reach, or as far as the captions describe, "
            "whichever ends first, which --outlive-caption lifts. "
            f"Default: {DEFAULT_FRAMES}."
        ),
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--parallel-encode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Under torchrun, read and encode each camera's controls and opening "
            "frames on one rank, then share the latents in camera order. Drawn "
            "controls (--draw-map) are encoded a chunk at a time and unaffected. "
            "On 4 GPUs at 4 views and 197 frames this left the output "
            "byte-identical and steady fps unchanged, and cut prefill from "
            "about 83 s to about 51 s. --no-parallel-encode has every rank "
            "encode every camera."
        ),
    )
    parser.add_argument(
        "--parallel-decode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Under torchrun, decode each camera on the rank that owns it (the "
            "same round-robin as --parallel-encode) and send the pixels to rank "
            "0, instead of rank 0 decoding every camera while the other ranks "
            "wait at the next chunk's first collective. Opt-in until measured."
        ),
    )
    parser.add_argument(
        "--compile-control-encode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run the --draw-map control encoder (Wan 2.2 VAE) under "
            "torch.compile. It encodes 12 frames per camera per chunk eagerly "
            "today, about 0.6 s of every chunk. Opt-in until measured."
        ),
    )
    parser.add_argument(
        "--decoder-compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run the TAEHV decoder under torch.compile. Opt-in until measured.",
    )
    parser.add_argument(
        "--prefetch-controls",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Draw and encode the next chunk's --draw-map controls in a worker "
            "process while the current chunk denoises. The live control path is "
            "a constant ~0.2 s per chunk of CPU work; a thread would only share "
            "the model thread's GIL, a process hides it."
        ),
    )
    parser.add_argument(
        "--prefetch-controls-gpu-offset",
        type=int,
        default=0,
        help=(
            "GPU for the --prefetch-controls worker: this rank's index plus this, "
            "modulo the node's GPU count. 0 shares the rank's GPU, where the "
            "render and encode (~0.2 s of GPU time per chunk) still add to the "
            "model step; with two ranks on a four-GPU node, 2 gives each worker "
            "a GPU of its own and the model step is all that is left."
        ),
    )
    parser.add_argument(
        "--optimization-profile",
        choices=(
            "auto",
            RTX_PRO_6000_BLACKWELL_WARM_PROFILE,
            RTX_PRO_6000_BLACKWELL_PERF_PROFILE,
        ),
        default="auto",
        help=(
            "Apply a measured 11-view RTX PRO 6000 Blackwell warm-service "
            "policy. The base profile passed all quality gates on seeds 0 and "
            "1. The explicit -perf variant adds fused RMSNorm: +1.78%% paired "
            "incremental throughput and +12.40%% observed cross-run versus the "
            "generic baseline. It passed only five of six quality thresholds: "
            "seed-1 worst-frame PSNR was 26.350 dB against the 28 dB floor. "
            "Other view counts must use auto. Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--attention-kv-arena",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Stage only current K/V into a persistent contiguous attention arena "
            "after bounded cache shapes settle. Enabled by default only for the "
            "quality-qualified four-view, non-compiled configuration."
        ),
    )
    parser.add_argument(
        "--flex-attention",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Gate attention with a block-sparse mask through compiled FlexAttention "
            "rather than a dense mask through cuDNN. Same visibility either way, "
            "and much cheaper, so --no-flex-attention is for a machine that "
            "cannot compile the kernel. Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--attention-scope",
        choices=get_args(AttentionScope),
        default=DEFAULT_ATTENTION_SCOPE,
        help=(
            "Which keys a view's tokens may attend, which is a property of the "
            "checkpoint rather than a preference: a checkpoint reads whatever "
            "it was trained to read, and giving it another scope degrades the "
            "picture without failing. decomposed restricts a token to its own "
            "view plus the same instant in the others, and is what the cosmos3 "
            "joint deliveries train with (multiview_attention_scope=decomposed). "
            "all_views is the older stage 3 checkpoint's, and is what to pass "
            "when running those weights. same_view restricts attention to each "
            "camera independently. Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--maskless-attention",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_MASKLESS_ATTENTION,
        help=(
            "Run dense attention per scope partition and merge by log-sum-exp. "
            "Final Sim-Auto checkpoints deduplicate same-view keys between "
            "partitions; older checkpoints retain their trained overlap. "
            "On by default for every scope except all_views."
        ),
    )
    parser.add_argument(
        "--fused-attention-merge",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Fuse the FP32 vector merge and final cast in eager maskless "
            "attention, retaining native row weights and rounding. Requires "
            "--maskless-attention; unsupported inputs and compiled denoising "
            "keep the native merge. On whenever maskless attention is on."
        ),
    )
    parser.add_argument(
        "--accelerated-rms-norm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Run RMSNorm as FlashDreams Accelerated's fused kernel instead of the "
            "checkpoint's module. On by default; the rtx-pro-6000-blackwell "
            "profiles pin their own setting unless this flag is given."
        ),
    )
    parser.add_argument(
        "--decomposed-temporal-window-seconds",
        type=float,
        default=DEFAULT_TEMPORAL_WINDOW_SECONDS,
        help=(
            "Under the decomposed scope, let a query read every camera's frames "
            "up to this many seconds before its own rather than only its own "
            "instant. The joint self-forcing checkpoints were trained with 0.4 "
            "seconds, the default. 0 keeps "
            "only the same-instant cross-view pass."
        ),
    )
    parser.add_argument(
        "--denoise-steps",
        type=int,
        default=DEFAULT_DENOISE_STEPS,
        help=(
            "Trim the checkpoint's distilled sigma schedule to this many steps. "
            "0 keeps all four. The final Sim-Auto guide specifies four steps; "
            "shorter schedules are diagnostic overrides. "
            "A non-final chunk costs steps + one clean replay; the final "
            "chunk skips that replay. Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=DEFAULT_CHUNK_FRAMES,
        help=(
            "Latent frames generated per autoregressive step, per camera. What "
            "the checkpoint was trained with, not a tuning knob: the rig's is "
            "three where the single-view chunk=4 weights the shared spec "
            "describes want four. 0 takes the spec's. Unrelated to the 1 + 4n "
            "shape of --frames, which is the VAE's temporal compression. "
            "Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--compile-denoise",
        choices=("off", *COMPILE_DENOISE_MODES),
        default="off",
        help=(
            "Compile the noisy pass with torch.compile. One compilation serves "
            "every chunk once the bounded cache settles, so this pays for its "
            "warmup over a long stream and roughly breaks even over one clip. "
            "Needs a bounded cache, so --history-frames 0 is refused with it. "
            "Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--history-frames",
        type=int,
        default=DEFAULT_HISTORY_FRAMES,
        help=(
            "How many generated frames the key/value cache keeps, with control "
            "kept for one chunk more. 0 keeps the whole clip, which costs memory "
            "and time that grow with its length. Rounded up to whole chunks. "
            "Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--draw-map",
        "--scene",
        type=Path,
        default=None,
        help=f"Prepared scene directory or built-in name: {', '.join(SCENE_NAMES)}. "
        f"Defaults to {DEFAULT_SCENE} when no sample is supplied.",
    )
    parser.add_argument(
        "--loop-controls",
        action="store_true",
        help=(
            "Replay the sample's control videos from the start whenever they run "
            "out, so --frames can exceed the clip, as the reference's long "
            "rollouts do. Implies --outlive-caption."
        ),
    )
    parser.add_argument(
        "--outlive-caption",
        action="store_true",
        help=(
            "Generate more frames than the sample's captions describe. A caption "
            "chunk covers 200 source frames and training always matched the clip "
            "to that span, so the prompt goes stale past it. For measuring how "
            "badly, which a stream has to know."
        ),
    )
    parser.add_argument(
        "--camera",
        default=None,
        help="Emit this camera alone, at native size, instead of the mosaic.",
    )
    parser.add_argument(
        "--ui",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show the camera-grid UI and sensor views (default with --lidar-sample). --no-ui emits raw video; --camera selects the initial focus.",
    )
    parser.add_argument("--device", default="cuda")
    commandline_args = list(commandline_args)
    parser.add_argument(
        "--lidar-sample",
        type=Path,
        help="Aligned metric LiDAR controls and calibration NPZ",
    )
    parser.add_argument(
        "--lidar-checkpoint",
        help="Local codec override; defaults to the model's matching HF V1.2 companion",
    )
    parser.add_argument(
        "--lidar-tokenizer-version",
        choices=("v1p1", "v1p2"),
        default="v1p2",
        help="Tokenizer architecture: v1p2 for the joint student, v1p1 for older symmetric checkpoints",
    )
    parser.add_argument(
        "--lidar-decode-worker",
        action="store_true",
        help=(
            "Decode the LiDAR sweeps in a worker process on the --prefetch-controls-gpu-offset "
            "GPU instead of on rank 0's model thread. Same decoder, same output."
        ),
    )
    parser.add_argument(
        "--lidar-view",
        choices=[view.value for view in LidarView],
        default=LidarView.POINT_CLOUD.value,
        help="Initial LiDAR visualization (default: %(default)s)",
    )
    args = parser.parse_args(commandline_args)
    args.views_defaulted = args.views is None
    if args.views_defaulted:
        args.views = DEFAULT_VIEWS
    args.ui = (args.lidar_sample is not None) if args.ui is None else args.ui

    if args.optimization_profile != "auto":
        owned_flags = (
            "--flex-attention",
            "--no-flex-attention",
            "--compile-denoise",
        )
        scope = "kernel and compile policy"
        if args.optimization_profile == RTX_PRO_6000_BLACKWELL_PERF_PROFILE:
            owned_flags += (
                "--denoise-steps",
                "--history-frames",
                "--chunk-frames",
            )
            scope = "complete measured configuration"
        conflicts = [
            flag
            for flag in owned_flags
            if any(
                argument == flag or argument.startswith(f"{flag}=")
                for argument in commandline_args
            )
        ]
        if conflicts:
            raise ValueError(
                f"--optimization-profile {args.optimization_profile} owns its "
                f"{scope}; remove: {', '.join(conflicts)}"
            )

    if args.views < 1:
        raise ValueError(f"--views must be >= 1, got {args.views}.")
    if args.frames < 0:
        raise ValueError(
            f"--frames must be >= 0, got {args.frames}. Use 0 to take the length "
            "from the controls."
        )
    if not math.isfinite(args.fps) or args.fps <= 0:
        raise ValueError(f"--fps must be finite and > 0, got {args.fps}.")
    if args.denoise_steps < 0:
        raise ValueError(
            f"--denoise-steps must be >= 0, got {args.denoise_steps}. Use 0 for "
            "the checkpoint's own schedule."
        )
    # 0 reads as "the whole schedule", which only the checkpoint can name.
    args.denoise_steps = args.denoise_steps or None
    if args.chunk_frames < 0:
        raise ValueError(
            f"--chunk-frames must be >= 0, got {args.chunk_frames}. Use 0 for "
            "the spec's own chunk."
        )
    args.chunk_frames = args.chunk_frames or None
    if args.history_frames < 0:
        raise ValueError(
            f"--history-frames must be >= 0, got {args.history_frames}. Use 0 to "
            "keep the whole clip."
        )
    # 0 reads as "no bound" downstream, where None is what says so.
    args.history_frames = args.history_frames or None
    # Named profiles resolve after range checks, before dependent defaults.
    _resolve_optimization_profile(args)
    if args.attention_kv_arena is None:
        args.attention_kv_arena = (
            args.views == 4
            and args.compile_denoise is None
            and args.history_frames is not None
        )
    foldable = args.attention_scope in MASKLESS_SCOPES
    if args.maskless_attention and not foldable:
        raise ValueError(
            f"--maskless-attention cannot express --attention-scope "
            f"{args.attention_scope}; it covers {', '.join(MASKLESS_SCOPES)}. "
            "all_views is one unmasked pass over the whole key stream rather "
            "than a partition of it, so there is nothing to fold"
        )
    # Only an explicit --maskless-attention contradicts an unfoldable scope.
    # Reaching one by leaving the default alone is not a mistake to report.
    args.maskless_attention = (
        foldable if args.maskless_attention is None else args.maskless_attention
    )
    if args.fused_attention_merge is None:
        args.fused_attention_merge = args.maskless_attention
    if args.attention_kv_arena and args.views != 4:
        raise ValueError("--attention-kv-arena is quality-qualified only for 4 views")
    if args.attention_kv_arena and args.history_frames is None:
        raise ValueError("--attention-kv-arena needs --history-frames to settle")
    if args.attention_kv_arena and args.compile_denoise is not None:
        raise ValueError(
            "--attention-kv-arena with --compile-denoise failed multi-seed "
            "quality qualification; disable one of them"
        )
    if args.compile_denoise is not None and args.history_frames is None:
        raise ValueError(
            "--compile-denoise needs --history-frames to bound the cache. Left "
            "unbounded the key stream grows every chunk, so each one is a new "
            "shape and recompiles."
        )
    if args.lidar_checkpoint is not None and args.lidar_sample is None:
        raise ValueError("--lidar-checkpoint requires --lidar-sample")
    if args.lidar_decode_worker and args.lidar_sample is None:
        raise ValueError("--lidar-decode-worker needs --lidar-sample")
    if args.lidar_sample is not None:
        if args.loop_controls:
            raise ValueError(
                "Recorded LiDAR controls cannot follow --loop-controls; use a live LiDAR provider"
            )
        if not args.sample:
            raise ValueError(
                "--lidar-sample requires matching real RGB --sample inputs"
            )
        if not args.maskless_attention or args.attention_scope != "decomposed":
            raise ValueError("LiDAR requires maskless decomposed attention")
        if args.attention_kv_arena or args.history_frames is None:
            raise ValueError(
                "LiDAR requires a bounded history, without the RGB-only K/V arena"
            )
        if args.chunk_frames != 3 or args.fps != 30:
            raise ValueError(
                "LiDAR currently requires 30 FPS, three RGB latents per chunk, and zero or one initial RGB latent"
            )
        if not args.decomposed_temporal_window_seconds:
            raise ValueError("LiDAR requires an explicit temporal attention window")
    return args


__all__ = [
    "Cosmos3NanoSimAutoApplication",
    "Cosmos3NanoSimAutoModelLoop",
    "Cosmos3NanoSimAutoSession",
    "create_app",
    "mosaic",
    "torn",
]

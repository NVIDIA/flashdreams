"""Alpadreams conditioning wrapper used by the gRPC server.

This module combines:
- Pipeline setup and autoregressive generation (`AlpadreamsPipeline`)
- HD map / bbox rendering (`LudusRenderer`)
- Session state handling for start/continue generation calls
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from alpadreams.conditioning.renderer import LudusRenderer
from alpadreams.conditioning.world_scenario.data_types import SceneData
from alpadreams.conditioning.world_scenario.ftheta import FThetaCamera
from alpadreams.grpc.profiling_server import get_profiler, get_profiling_context
from loguru import logger
from torch import Tensor, nn

from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler.fm import FlowMatchSchedulerConfig
from flashdreams.infra.encoder.text.cosmos_reason1 import CosmosReason1TextEncoderConfig
from flashdreams.recipes.alpadreams.config import AVAILABLE_ALPADREAMS_CHECKPOINT_PATHS
from flashdreams.recipes.alpadreams.encoder.pixel_shuffle import (
    PixelShuffleVAEEncoderConfig,
)
from flashdreams.recipes.alpadreams.pipeline import (
    AlpadreamsPipeline,
    AlpadreamsPipelineCache,
    AlpadreamsPipelineConfig,
)
from flashdreams.recipes.alpadreams.transformer import CosmosTransformerConfig
from flashdreams.recipes.alpadreams.transformer.impl.network import (
    CosmosDiTNetworkConfig,
)
from flashdreams.recipes.taehv import (
    AVAILABLE_TAEHV_CHECKPOINT_PATHS,
    TeahvVAEDecoderConfig,
)
from flashdreams.recipes.wan.autoencoder.vae import (
    AVAILABLE_WAN_VAE_CHECKPOINT_PATHS,
    WanVAEDecoderConfig,
    WanVAEEncoderConfig,
)


@dataclass
class TextPrompt:
    """A text prompt for video generation (single video)."""

    positive: str
    negative: str | None = None

    # Precomputed embeddings (unused for now but kept for API parity).
    positive_embeddings: Tensor | None = None
    negative_embeddings: Tensor | None = None


# Autonomous driving prompt
AV_POSITIVE_PROMPT = (
    "Driving scene from a front-facing car camera. Urban environment with roads, vehicles, pedestrians, "
    "traffic signs, and buildings. Clear visibility, realistic lighting, photorealistic quality. "
    "High resolution dashcam footage of city driving."
)


@dataclass
class AlpadreamsConditioningState:
    """State for generation, including renderer and pipeline state."""

    renderer: LudusRenderer
    pipeline_cache: AlpadreamsPipelineCache | None = None


@dataclass
class GenerationOutput:
    """Output from start_generation or continue_generation."""

    state: AlpadreamsConditioningState  # Updated state for next call
    condition_frames: Tensor  # Rendered HDMap frames, shape [B, V, T, 3, H, W], uint8
    rgb_frames: Tensor | None = (
        None  # Generated video frames [B, V, T, 3, H, W] (None if skip_video_generation)
    )
    finalization_state: dict | None = None  # Finalization state from the video model


class AlpadreamsConditioningWrapper(nn.Module):
    """
    Alpadreams-specific conditioning wrapper that owns rendering and generation.

    This class provides the gRPC-facing start/continue API directly, including
    camera-pose-driven condition rendering and autoregressive generation calls.
    """

    def __init__(
        self,
        *,
        n_cameras: int,
        resolution_wh: tuple[int, int],
        local_attn_size: int,
        sink_size: int,
        denoising_step_list: list[int],
        num_frames_per_block: int,
        compile_net: bool,
        seed_for_every_rollout: int | None,
        encode_with_pixel_shuffle: bool,
        no_tae: bool,
        upsampler: str = "none",
        use_cuda_graphs: bool = True,
        kv_cache_on_side_stream: bool = False,
        s3_credential_path: str = "credentials/s3_checkpoint.secret",
        device: torch.device = torch.device("cuda:0"),
    ) -> None:
        super().__init__()

        if num_frames_per_block % 4 != 0:
            raise ValueError(
                "num_frames_per_block must be divisible by 4 for flashdreams pipeline backend"
            )
        if upsampler != "none":
            raise ValueError(
                "Upsampler support is not wired in flashdreams pipeline backend yet."
            )
        if not use_cuda_graphs:
            logger.warning(
                "use_cuda_graphs flag is ignored by flashdreams pipeline backend."
            )
        if kv_cache_on_side_stream:
            logger.warning(
                "kv_cache_on_side_stream flag is ignored by flashdreams pipeline backend."
            )
        if s3_credential_path != "credentials/s3_checkpoint.secret":
            logger.warning(
                "s3_credential_path is controlled by flashdreams checkpoint loader defaults."
            )

        self._device = device
        self._n_cameras = n_cameras
        self.video_resolution_wh = resolution_wh
        self._rollout_seed = seed_for_every_rollout
        self.fps = 30

        self.frame_chunk_size = num_frames_per_block
        len_t = num_frames_per_block // 4
        self.initial_frame_chunk_size = 1 + (len_t - 1) * 4

        pipeline_config = self._build_pipeline_config(
            n_cameras=n_cameras,
            resolution_wh=resolution_wh,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            denoising_step_list=denoising_step_list,
            len_t=len_t,
            compile_net=compile_net,
            encode_with_pixel_shuffle=encode_with_pixel_shuffle,
            no_tae=no_tae,
            seed=seed_for_every_rollout if seed_for_every_rollout is not None else 42,
        )
        pipeline = pipeline_config.setup().to(device=device)
        assert isinstance(pipeline, AlpadreamsPipeline)  # for type checking
        self.pipeline: AlpadreamsPipeline = pipeline

    @property
    def V_group(self) -> torch.distributed.ProcessGroup | None:
        # Pipeline backend handles CP internally, so server-side split/gather
        # should remain disabled.
        return None

    @property
    def n_cameras(self) -> int:
        return self._n_cameras

    @property
    def input_device(self) -> torch.device:
        return self._device

    @property
    def output_device(self) -> torch.device:
        return self._device

    @staticmethod
    def _build_pipeline_config(
        *,
        n_cameras: int,
        resolution_wh: tuple[int, int],
        local_attn_size: int,
        sink_size: int,
        denoising_step_list: list[int],
        len_t: int,
        compile_net: bool,
        encode_with_pixel_shuffle: bool,
        no_tae: bool,
        seed: int,
    ) -> AlpadreamsPipelineConfig:
        if n_cameras not in (1, 4):
            raise ValueError(
                f"Only n_cameras in {{1, 4}} is supported by current checkpoints, got {n_cameras}"
            )

        if n_cameras == 1:
            if encode_with_pixel_shuffle:
                if len_t != 4:
                    raise ValueError(
                        "Single-view pixel-shuffle checkpoints currently support len_t=4 only."
                    )
                checkpoint_path = AVAILABLE_ALPADREAMS_CHECKPOINT_PATHS[
                    "1view-pshuffle-chunk4"
                ]
                hdmap_encoder_config = PixelShuffleVAEEncoderConfig()
            else:
                if len_t not in (2, 3):
                    raise ValueError(
                        "Single-view VAE-encoding checkpoints currently support len_t in {2, 3}."
                    )
                checkpoint_path = AVAILABLE_ALPADREAMS_CHECKPOINT_PATHS[
                    f"1view-vae-chunk{len_t}"
                ]
                tokenizer_key = "vae" if no_tae else "lightvae"
                hdmap_encoder_config = WanVAEEncoderConfig(
                    checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS[tokenizer_key],
                )
        else:
            if len_t != 4:
                raise ValueError(
                    "Multi-view checkpoints currently support len_t=4 only."
                )
            if encode_with_pixel_shuffle:
                checkpoint_path = AVAILABLE_ALPADREAMS_CHECKPOINT_PATHS[
                    "4view-pshuffle-chunk4"
                ]
                hdmap_encoder_config = PixelShuffleVAEEncoderConfig()
            else:
                checkpoint_path = AVAILABLE_ALPADREAMS_CHECKPOINT_PATHS[
                    "4view-vae-chunk4"
                ]
                hdmap_encoder_config = WanVAEEncoderConfig(
                    checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"],
                )

        if no_tae:
            decoder_config = WanVAEDecoderConfig(
                checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"],
            )
        else:
            decoder_config = TeahvVAEDecoderConfig(
                checkpoint_path=AVAILABLE_TAEHV_CHECKPOINT_PATHS["lighttae"],
            )

        _, height = resolution_wh
        extrapolation = 2.0 if height <= 480 else 3.0
        # Per-rollout latent (height, width) is supplied later via
        # :meth:`AlpadreamsPipeline.initialize_cache` (derived from the
        # encoded first-frame's shape).
        transformer_config = CosmosTransformerConfig(
            network=CosmosDiTNetworkConfig(
                # HDMap conditioning channel count: 192 for pixel-shuffle,
                # 16 for the Wan-VAE branch.
                additional_concat_ch=192 if encode_with_pixel_shuffle else 16,
                enable_cross_view_attn=n_cameras > 1,
            ),
            batch_shape=(1,),
            num_views=n_cameras,
            h_extrapolation_ratio=extrapolation,
            w_extrapolation_ratio=extrapolation,
            window_size_t=local_attn_size,
            sink_size_t=sink_size,
            len_t=len_t,
            checkpoint_path=checkpoint_path,
            compile_network=compile_net,
        )

        scheduler_config = FlowMatchSchedulerConfig(
            num_inference_steps=len(denoising_step_list),
            denoising_timesteps=denoising_step_list,
            warp_denoising_step=True,
            shift=5.0,
            sigma_min=0.0,
            extra_one_step=True,
        )

        # `image_encoder` (first-frame) is pinned to the full Wan VAE to
        # match the training distribution regardless of which encoder is
        # used for the per-AR-step HDMap.
        image_encoder_config = WanVAEEncoderConfig(
            checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"],
        )

        return AlpadreamsPipelineConfig(
            text_encoder=CosmosReason1TextEncoderConfig(),
            image_encoder=image_encoder_config,
            encoder=hdmap_encoder_config,
            decoder=decoder_config,
            diffusion_model=DiffusionModelConfig(
                seed=seed,
                context_noise=128,
                transformer=transformer_config,
                scheduler=scheduler_config,
            ),
        )

    def set_rollout_seed(self, seed: int | None) -> None:
        self._rollout_seed = seed

    def finalize_block_generation(
        self,
        pipeline_cache: AlpadreamsPipelineCache,
        finalization_state: dict | None,
    ) -> None:
        if finalization_state is None:
            return
        block_idx = int(finalization_state["autoregressive_index"])
        self.pipeline.finalize(
            autoregressive_index=block_idx,
            cache=pipeline_cache,
        )

    def _seed_pipeline_for_next_rollout(self) -> None:
        # `AlpadreamsPipeline` delegates RNG to DiffusionModel, which lazily
        # creates a torch.Generator seeded from DiffusionModelConfig.seed.
        rng = self.pipeline.diffusion_model.rng
        assert rng is not None, (
            "DiffusionModelConfig.seed must not be None for streaming rollouts."
        )
        if self._rollout_seed is None:
            _ = rng.seed()
        else:
            rng.manual_seed(int(self._rollout_seed))

    def _validate_camera_inputs(
        self,
        *,
        camera_names: list[str],
        camera_poses_per_view: dict[str, torch.Tensor],
        frame_timestamps_us: list[int],
        expected_length: int,
    ) -> None:
        if len(frame_timestamps_us) != expected_length:
            raise ValueError(
                f"frame_timestamps_us length ({len(frame_timestamps_us)}) must be {expected_length}"
            )
        for cam_name in camera_names:
            if cam_name not in camera_poses_per_view:
                raise ValueError(
                    f"Missing camera pose sequence for camera '{cam_name}'"
                )
            camera_poses = camera_poses_per_view[cam_name]
            if not isinstance(camera_poses, torch.Tensor):
                raise TypeError(
                    f"camera_poses for '{cam_name}' must be torch.Tensor, got {type(camera_poses)}"
                )
            if camera_poses.shape != (expected_length, 4, 4):
                raise ValueError(
                    f"camera_poses for '{cam_name}' must be [{expected_length}, 4, 4], got {tuple(camera_poses.shape)}"
                )

    def create_renderer(
        self,
        scene_data: SceneData,
        camera_names: list[str],
    ) -> LudusRenderer:
        """Create a renderer from scene data for one or more cameras.

        Args:
            scene_data: Static world / HD map data.
            camera_names: Camera names to include in the renderer.

        Returns:
            A renderer capable of rendering all listed cameras.
        """
        res_W, res_H = self.video_resolution_wh
        camera_models: dict[str, FThetaCamera] = {}

        for camera_name in camera_names:
            # Get or create camera model
            if scene_data.camera_models.get(camera_name) is None:
                # Create a default 120 FOV camera model.
                # from_numpy format: [cx, cy, width, height, *poly(6), is_bw_poly, linear_c, linear_d, linear_e]
                cx = res_W / 2
                cy = res_H / 2
                # For a 120 FOV equidistant camera: poly maps angle→pixel_dist
                # pixel_dist = f * angle, where f = width / (2 * FOV_half_rad)
                f = res_W / (2 * np.radians(60))  # 60 deg half-FOV
                intrinsics = np.array(
                    [cx, cy, res_W, res_H, f, 0, 0, 0, 0, 0, 0.0, 1.0, 0.0, 0.0],
                    dtype=np.float64,
                )
                camera_model = FThetaCamera.from_numpy(intrinsics)
                scene_data.camera_models[camera_name] = camera_model

            camera_model_raw = scene_data.camera_models[camera_name]
            assert isinstance(camera_model_raw, FThetaCamera), (
                f"Currently only supporting FTheta cameras, got {type(camera_model_raw)=}"
            )
            camera_model = camera_model_raw

            # Resize camera model if needed
            if camera_model.height != res_H or camera_model.width != res_W:
                scale_h = res_H / camera_model.height
                scale_w = res_W / camera_model.width
                camera_model = FThetaCamera.from_numpy(camera_model.intrinsics.copy())
                camera_model.rescale(ratio_h=scale_h, ratio_w=scale_w)

            camera_models[camera_name] = camera_model

        return LudusRenderer(
            camera_models=camera_models,
            scene_data=scene_data,
            hdmap_color_version="v3",
            bbox_color_version="v3",
            windowless=True,
            device=self._device,
        )

    def _render_condition_frames(
        self,
        renderer: LudusRenderer,
        camera_names: list[str],
        camera_poses_per_view: dict[str, torch.Tensor],
        frame_timestamps_us: list[int],
        object_info_per_frame: list[dict] | None = None,
    ) -> Tensor:
        """Render conditioning frames for all cameras.

        Args:
            renderer: The LudusRenderer to use.
            camera_names: Ordered list of camera names (defines V ordering).
            camera_poses_per_view: ``{camera_name: [T, 4, 4]}`` poses.
            frame_timestamps_us: Timestamps in microseconds for each frame.
            object_info_per_frame: Optional per-frame object info dicts.

        Returns:
            ``[B, V, T, 3, H, W]`` uint8 tensor on ``self.input_device`` (B=1).
        """
        # We use the same object info for all cameras
        obj_infos: list[dict | None] = []
        for i in range(len(frame_timestamps_us)):
            obj_info = None
            if object_info_per_frame is not None and i < len(object_info_per_frame):
                frame_obj_info = object_info_per_frame[i]
                if frame_obj_info:
                    obj_info = frame_obj_info
            obj_infos.append(obj_info)

        # NOTE: We do not support object info per frame for LudusRenderer yet
        for o in obj_infos:
            assert o is None, f"Object info not supported yet for LudusRenderer: {o}"

        # Render all frames and cameras in a single pass
        all_view_frames = renderer.render_all_frames_and_cameras(
            camera_names, camera_poses_per_view, frame_timestamps_us
        )
        assert all_view_frames.ndim == 5 and all_view_frames.dtype == torch.uint8

        # [1, V, T, 3, H, W]
        return all_view_frames.unsqueeze(0)

    def _normalize_start_inputs(
        self, initial_rgb_frames: Tensor, initial_condition_frames: Tensor
    ) -> tuple[Tensor, Tensor]:
        if self._n_cameras == 1:
            if initial_rgb_frames.ndim == 4:
                initial_rgb_frames = initial_rgb_frames.unsqueeze(1)
            if initial_condition_frames.ndim == 5:
                initial_condition_frames = initial_condition_frames.unsqueeze(1)
        if initial_rgb_frames.ndim != 5:
            raise ValueError(
                f"initial_rgb_frames must be [B,V,3,H,W], got shape {tuple(initial_rgb_frames.shape)}"
            )
        if initial_condition_frames.ndim != 6:
            raise ValueError(
                "initial_condition_frames must be [B,V,T,3,H,W], "
                f"got shape {tuple(initial_condition_frames.shape)}"
            )
        if initial_rgb_frames.shape[1] != self._n_cameras:
            raise ValueError(
                f"Expected V={self._n_cameras}, got V={initial_rgb_frames.shape[1]}"
            )
        if initial_condition_frames.shape[1] != self._n_cameras:
            raise ValueError(
                f"Expected V={self._n_cameras}, got V={initial_condition_frames.shape[1]}"
            )
        return initial_rgb_frames, initial_condition_frames

    def _normalize_condition_input(self, condition_frames: Tensor) -> Tensor:
        if self._n_cameras == 1 and condition_frames.ndim == 5:
            condition_frames = condition_frames.unsqueeze(1)
        if condition_frames.ndim != 6:
            raise ValueError(
                f"condition_frames must be [B,V,T,3,H,W], got shape {tuple(condition_frames.shape)}"
            )
        if condition_frames.shape[1] != self._n_cameras:
            raise ValueError(
                f"Expected V={self._n_cameras}, got V={condition_frames.shape[1]}"
            )
        return condition_frames

    def _build_text_batch(self, text_prompts: list[TextPrompt]) -> list[list[str]]:
        return [
            [prompt.positive for _ in range(self._n_cameras)] for prompt in text_prompts
        ]

    def _to_model_range(self, x: Tensor) -> Tensor:
        if x.dtype == torch.uint8:
            x = x.to(self._device, dtype=torch.bfloat16)
            return x / 127.5 - 1.0
        return x.to(self._device, dtype=torch.bfloat16)

    def _to_uint8(self, x: Tensor) -> Tensor:
        if x.dtype == torch.uint8:
            return x
        x = x.clamp(-1.0, 1.0)
        return ((x + 1.0) * 127.5).round().to(torch.uint8)

    def start_generation(
        self,
        text_prompts: list[TextPrompt],
        initial_rgb_frames: Tensor,
        renderer: LudusRenderer,
        camera_names: list[str],
        camera_poses_per_view: dict[str, torch.Tensor],
        frame_timestamps_us: list[int],
        skip_video_generation: bool = False,
    ) -> GenerationOutput:
        """Render initial condition frames and start video generation.

        Args:
            text_prompts: Text prompts for generation (length B, typically 1).
            initial_rgb_frames: Initial RGB images, shape ``[B, V, 3, H, W]``, uint8.
            renderer: Pre-created renderer for HDMap rendering.
            camera_names: Ordered list of camera names (defines V ordering).
            camera_poses_per_view: ``{camera_name: [T, 4, 4]}`` poses.
            frame_timestamps_us: Timestamps in microseconds for each frame.
            skip_video_generation: If True, only render HDMap without running the video model.

        Returns:
            GenerationOutput with ``condition_frames`` ``[B, V, T, 3, H, W]``
            and ``rgb_frames`` ``[B, V, T, 3, H, W]`` (or None).
        """

        assert len(text_prompts) == 1, (
            "Only one text prompt (batch size == 1) is supported for now"
        )

        self._validate_camera_inputs(
            camera_names=camera_names,
            camera_poses_per_view=camera_poses_per_view,
            frame_timestamps_us=frame_timestamps_us,
            expected_length=self.initial_frame_chunk_size,
        )

        # condition_frames: [B, V, T, 3, H, W]
        condition_frames = self._render_condition_frames(
            renderer, camera_names, camera_poses_per_view, frame_timestamps_us
        )

        if skip_video_generation:
            state = AlpadreamsConditioningState(
                renderer=renderer,
            )
            return GenerationOutput(
                state=state, condition_frames=condition_frames, rgb_frames=None
            )

        initial_rgb_frames, condition_frames = self._normalize_start_inputs(
            initial_rgb_frames, condition_frames
        )
        self._seed_pipeline_for_next_rollout()
        text = self._build_text_batch(text_prompts)

        first_frame = self._to_model_range(initial_rgb_frames).unsqueeze(2)
        condition = self._to_model_range(condition_frames)

        pipeline_cache = self.pipeline.initialize_cache(
            text=text, image=first_frame, view_names=camera_names
        )
        rgb_frames = self.pipeline.generate(
            autoregressive_index=0,
            hdmap=condition,
            cache=pipeline_cache,
        )
        rgb_frames = self._to_uint8(rgb_frames).contiguous()

        state = AlpadreamsConditioningState(
            renderer=renderer,
            pipeline_cache=pipeline_cache,
        )
        return GenerationOutput(
            state=state,
            condition_frames=condition_frames,
            rgb_frames=rgb_frames,
            finalization_state={"autoregressive_index": 0},
        )

    def continue_generation(
        self,
        state: AlpadreamsConditioningState,
        camera_names: list[str],
        camera_poses_per_view: dict[str, torch.Tensor],
        frame_timestamps_us: list[int],
        object_info_per_frame: list[dict] | None = None,
        skip_video_generation: bool = False,
        text_prompts: list[TextPrompt] | None = None,
    ) -> GenerationOutput:
        """Render condition frames and continue video generation.

        Args:
            state: State from previous generation (contains renderer and pipeline cache).
            camera_names: Ordered list of camera names (defines V ordering).
            camera_poses_per_view: ``{camera_name: [T, 4, 4]}`` poses.
            frame_timestamps_us: Timestamps in microseconds for each frame.
            object_info_per_frame: Optional per-frame object info for dynamic actors.
            skip_video_generation: If True, only render HDMap without running the video model.
            text_prompts: Optional new text prompts.

        Returns:
            GenerationOutput with ``condition_frames`` ``[B, V, T, 3, H, W]``
            and ``rgb_frames`` ``[B, V, T, 3, H, W]`` (or None).
        """
        self._validate_camera_inputs(
            camera_names=camera_names,
            camera_poses_per_view=camera_poses_per_view,
            frame_timestamps_us=frame_timestamps_us,
            expected_length=self.frame_chunk_size,
        )
        renderer = state.renderer

        profiler = get_profiler()
        session_id, chunk_idx = get_profiling_context()

        # condition_frames: [B, V, T, 3, H, W]
        with profiler.measure(
            "render_condition_frames", session_id=session_id, chunk_idx=chunk_idx
        ):
            condition_frames = self._render_condition_frames(
                renderer,
                camera_names,
                camera_poses_per_view,
                frame_timestamps_us,
                object_info_per_frame,
            )

        if skip_video_generation:
            return GenerationOutput(
                state=state, condition_frames=condition_frames, rgb_frames=None
            )

        if state.pipeline_cache is None:
            raise ValueError(
                "Cannot continue video generation: pipeline_cache is None "
                "(session was started with skip_video_generation=True)"
            )

        model_cond = self._normalize_condition_input(condition_frames)
        condition = self._to_model_range(model_cond)
        prev_block_idx = state.pipeline_cache.autoregressive_index
        block_idx = 0 if prev_block_idx is None else prev_block_idx + 1

        with profiler.measure(
            "pipeline.continue_generation",
            session_id=session_id,
            chunk_idx=chunk_idx,
        ):
            del text_prompts  # Pipeline currently keeps prompts from initialize_cache.
            rgb_frames = self.pipeline.generate(
                autoregressive_index=block_idx,
                hdmap=condition,
                cache=state.pipeline_cache,
            )
            rgb_frames = self._to_uint8(rgb_frames).contiguous()

        new_state = AlpadreamsConditioningState(
            renderer=renderer,
            pipeline_cache=state.pipeline_cache,
        )
        return GenerationOutput(
            state=new_state,
            condition_frames=condition_frames,
            rgb_frames=rgb_frames,
            finalization_state={"autoregressive_index": block_idx},
        )

    def cleanup(self, state: AlpadreamsConditioningState) -> None:
        """Clean up renderer resources."""
        if state.renderer is not None:
            state.renderer.cleanup()

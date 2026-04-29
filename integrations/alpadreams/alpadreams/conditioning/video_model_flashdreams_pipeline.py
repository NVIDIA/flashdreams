"""FlashDreams-native VideoModelAPI adapter for gRPC serving."""

from __future__ import annotations

import torch
from loguru import logger
from torch import Tensor

from alpadreams.conditioning.video_model_api import (
    BaseLatentCache,
    TextPrompt,
    VideoModelAPI,
)
from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler.fm import FlowMatchSchedulerConfig
from flashdreams.infra.encoder.text.cosmos_qwen import (
    CosmosReason1TextEncoderConfig,
)
from flashdreams.recipes.alpadreams.config import (
    AVAILABLE_ALPADREAMS_CHECKPOINT_PATHS,
)
from flashdreams.recipes.alpadreams.encoder.pixel_shuffle import (
    PixelShuffleVAEEncoderConfig,
)
from flashdreams.recipes.alpadreams.pipeline import (
    AlpadreamsPipeline,
    AlpadreamsPipelineCache,
    AlpadreamsPipelineConfig,
)
from flashdreams.recipes.alpadreams.transformer import (
    CosmosTransformerConfig,
)
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


class FlashDreamsPipelineLatentCache(BaseLatentCache):
    """Latent cache wrapper around `AlpadreamsPipelineCache`."""

    def __init__(
        self,
        *,
        batch_size: int,
        history_length_in_frames: int,
        pipeline_cache: AlpadreamsPipelineCache,
        next_autoregressive_index: int,
        device: torch.device,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.history_length_in_frames = history_length_in_frames
        self.pipeline_cache = pipeline_cache
        self.next_autoregressive_index = next_autoregressive_index
        self.register_buffer("_dummy", torch.zeros(1, device=device))


class FlashDreamsPipelineVideoModelAPI(VideoModelAPI[FlashDreamsPipelineLatentCache]):
    """VideoModelAPI backed by `AlpadreamsPipeline`."""

    latent_cache_type = FlashDreamsPipelineLatentCache

    def __init__(
        self,
        *,
        n_cameras: int,
        resolution_wh: tuple[int, int],
        local_attn_size: int,
        sink_size: int,
        cp_size: int = 1,
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
    ):
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
            cp_size=cp_size,
            denoising_step_list=denoising_step_list,
            len_t=len_t,
            compile_net=compile_net,
            encode_with_pixel_shuffle=encode_with_pixel_shuffle,
            no_tae=no_tae,
            seed=seed_for_every_rollout if seed_for_every_rollout is not None else 42,
        )
        self.pipeline: AlpadreamsPipeline = pipeline_config.setup().to(device=device)

    @property
    def n_cameras(self) -> int:
        return self._n_cameras

    @property
    def V_group(self):  # noqa: ANN201
        # Pipeline backend already handles CP internally.
        # Server-side view split/gather must remain disabled.
        return None

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
        cp_size: int,
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
                checkpoint_path = AVAILABLE_ALPADREAMS_CHECKPOINT_PATHS["single_view"][
                    "pixel_shuffle"
                ]
                hdmap_encoder_config = PixelShuffleVAEEncoderConfig()
            else:
                if len_t not in (2, 3):
                    raise ValueError(
                        "Single-view VAE-encoding checkpoints currently support len_t in {2, 3}."
                    )
                checkpoint_path = AVAILABLE_ALPADREAMS_CHECKPOINT_PATHS["single_view"][
                    "vae_encoding"
                ][f"chunk{len_t}"]
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
                checkpoint_path = AVAILABLE_ALPADREAMS_CHECKPOINT_PATHS["4views"][
                    "pixel_shuffle"
                ]
                hdmap_encoder_config = PixelShuffleVAEEncoderConfig()
            else:
                checkpoint_path = AVAILABLE_ALPADREAMS_CHECKPOINT_PATHS["4views"][
                    "vae_encoding"
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
        transformer_config = CosmosTransformerConfig(
            network=CosmosDiTNetworkConfig(),
            batch_shape=(1,),
            height=height // 8,
            width=resolution_wh[0] // 8,
            enable_hdmap_condition=True,
            encode_with_pixel_shuffle=encode_with_pixel_shuffle,
            num_views=n_cameras,
            cp_size=cp_size,
            h_extrapolation_ratio=extrapolation,
            w_extrapolation_ratio=extrapolation,
            window_size_t=local_attn_size,
            sink_size_t=sink_size,
            len_t=len_t,
            checkpoint_path=checkpoint_path,
            compile_network=compile_net,
            # Server pins legacy non-overlapping rollout: the production
            # checkpoints were trained without ``kv_drop_t > 0``, and the
            # gRPC server has not been adapted to the kv_drop_t experiment
            # yet (e.g. encoder caveats, frame-count split, decoder slice).
            kv_drop_t=0,
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
        latent_cache: FlashDreamsPipelineLatentCache,
        finalization_state: dict | None,
    ) -> None:
        if finalization_state is None:
            return
        block_idx = int(finalization_state["autoregressive_index"])
        self.pipeline.finalize(
            autoregressive_index=block_idx,
            cache=latent_cache.pipeline_cache,
        )

    def _seed_pipeline_for_next_rollout(self) -> None:
        # `AlpadreamsPipeline` delegates RNG to the underlying DiffusionModel,
        # which lazily materializes a torch.Generator seeded from
        # DiffusionModelConfig.seed (set in `_build_pipeline_config`, never None).
        rng = self.pipeline.diffusion_model.rng
        assert rng is not None, (
            "DiffusionModelConfig.seed must not be None for streaming rollouts."
        )
        if self._rollout_seed is None:
            _ = rng.seed()
        else:
            rng.manual_seed(int(self._rollout_seed))

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
        initial_condition_frames: Tensor,
        view_names: list[str],
    ) -> tuple[FlashDreamsPipelineLatentCache, Tensor, dict]:
        batch_size = len(text_prompts)
        initial_rgb_frames, initial_condition_frames = self._normalize_start_inputs(
            initial_rgb_frames, initial_condition_frames
        )

        self._seed_pipeline_for_next_rollout()
        text = self._build_text_batch(text_prompts)

        first_frame = self._to_model_range(initial_rgb_frames).unsqueeze(2)
        condition = self._to_model_range(initial_condition_frames)

        pipeline_cache = self.pipeline.initialize_cache(
            text=text, image=first_frame, view_names=view_names
        )
        rgb_frames = self.pipeline.generate(
            autoregressive_index=0,
            hdmap=condition,
            cache=pipeline_cache,
        )
        rgb_frames = self._to_uint8(rgb_frames).contiguous()

        latent_cache = FlashDreamsPipelineLatentCache(
            batch_size=batch_size,
            history_length_in_frames=rgb_frames.shape[2],
            pipeline_cache=pipeline_cache,
            next_autoregressive_index=1,
            device=self._device,
        )
        if self._n_cameras == 1:
            rgb_frames = rgb_frames[:, 0]

        return latent_cache, rgb_frames, {"autoregressive_index": 0}

    def continue_generation(
        self,
        latent_cache: FlashDreamsPipelineLatentCache,
        condition_frames: Tensor,
        text_prompts: list[TextPrompt] | None = None,
    ) -> tuple[FlashDreamsPipelineLatentCache, Tensor, dict]:
        del text_prompts  # Pipeline currently keeps prompts from initialize_cache.

        condition_frames = self._normalize_condition_input(condition_frames)
        condition = self._to_model_range(condition_frames)

        block_idx = latent_cache.next_autoregressive_index
        rgb_frames = self.pipeline.generate(
            autoregressive_index=block_idx,
            hdmap=condition,
            cache=latent_cache.pipeline_cache,
        )
        rgb_frames = self._to_uint8(rgb_frames).contiguous()

        latent_cache.history_length_in_frames += rgb_frames.shape[2]
        latent_cache.next_autoregressive_index += 1

        if self._n_cameras == 1:
            rgb_frames = rgb_frames[:, 0]

        return latent_cache, rgb_frames, {"autoregressive_index": block_idx}

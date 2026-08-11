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

"""LingBot-specific encoder, DiT, and decoder service stages."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from torch.distributed import ProcessGroup

from flashdreams.infra.decoder import StreamingVideoDecoder
from flashdreams.infra.diffusion.scheduler import FlowMatchScheduler
from flashdreams.infra.pipeline import DecoderStage, DiffusionStage, DiffusionStageCache
from flashdreams.infra.transfer import TensorBundle
from flashdreams.recipes.wan.autoencoder.i2v import I2VCtrl
from flashdreams.recipes.wan.autoencoder.vae import WanVAECache
from flashdreams.recipes.wan.transformer.constants import NEGATIVE_PROMPT
from lingbot.encoder.camctrl import (
    CamCtrlInput,
    I2VCamCtrlEmbeddings,
    I2VCamCtrlEncoder,
    I2VCamCtrlEncoderCache,
    I2VCamCtrlInput,
)
from lingbot.pipeline import LingbotWorldInferencePipelineConfig
from lingbot.transformer import (
    LingbotWorldTransformer,
    LingbotWorldTransformerCache,
    LingbotWorldTransformerConfig,
)


@dataclass(kw_only=True)
class LingbotConditioning:
    """One-shot encoder output transferred into the DiT worker."""

    height: int
    """Pre-patchify latent height."""

    width: int
    """Pre-patchify latent width."""

    text_embeddings: Tensor
    """Positive UMT5 prompt embeddings."""

    negative_text_embeddings: Tensor | None = None
    """Negative UMT5 embeddings when classifier-free guidance is enabled."""

    image_embeddings: Tensor | None = None
    """Optional CLIP first-frame embeddings."""


@dataclass(kw_only=True)
class LingbotEncoderStageCache:
    """Per-session state retained by the LingBot encoder worker."""

    encoder_cache: I2VCamCtrlEncoderCache
    """Streaming VAE and camera-control cache."""

    image: Tensor
    """First-frame pixels used to construct every I2V input chunk."""


def conditioning_to_bundle(conditioning: LingbotConditioning) -> TensorBundle:
    """Flatten one-shot conditioning into a transferable tensor bundle."""
    bundle = {"text_embeddings": conditioning.text_embeddings.contiguous()}
    if conditioning.negative_text_embeddings is not None:
        bundle["negative_text_embeddings"] = (
            conditioning.negative_text_embeddings.contiguous()
        )
    if conditioning.image_embeddings is not None:
        bundle["image_embeddings"] = conditioning.image_embeddings.contiguous()
    return bundle


def conditioning_from_bundle(
    bundle: TensorBundle,
    *,
    height: int,
    width: int,
) -> LingbotConditioning:
    """Reconstruct one-shot conditioning after a transfer."""
    return LingbotConditioning(
        height=height,
        width=width,
        text_embeddings=bundle["text_embeddings"],
        negative_text_embeddings=bundle.get("negative_text_embeddings"),
        image_embeddings=bundle.get("image_embeddings"),
    )


def encoder_output_to_bundle(output: I2VCamCtrlEmbeddings) -> TensorBundle:
    """Flatten per-step LingBot encoder output for transport."""
    assert not output._is_patchified, (
        "Encoder output must cross the service boundary before DiT patchification."
    )
    return {
        "i2v.latent": output.i2v.latent.contiguous(),
        "i2v.mask": output.i2v.mask.contiguous(),
        "plucker": output.plucker.contiguous(),
    }


def encoder_output_from_bundle(
    bundle: TensorBundle,
    *,
    patchified: bool = False,
) -> I2VCamCtrlEmbeddings:
    """Reconstruct per-step LingBot encoder output after transport."""
    return I2VCamCtrlEmbeddings(
        i2v=I2VCtrl(
            latent=bundle["i2v.latent"],
            mask=bundle["i2v.mask"],
            _is_patchified=patchified,
        ),
        plucker=bundle["plucker"],
        _is_patchified=patchified,
    )


def encoder_output_to_cp_bundles(
    output: I2VCamCtrlEmbeddings,
    *,
    cp_size: int,
    patch_size: tuple[int, int, int],
) -> tuple[TensorBundle, ...]:
    """Patchify once on the encoder and return one direct-transfer shard per rank.

    This removes the DiT leader's input broadcast/fan-out. Each destination
    rank receives only its token shard and reconstructs the payload with
    ``patchified=True``.
    """
    if cp_size < 1:
        raise ValueError(f"cp_size must be positive, got {cp_size}.")
    if output._is_patchified or output.i2v._is_patchified:
        raise ValueError("Expected raw encoder output before patchification.")

    def patchify(tensor: Tensor) -> Tensor:
        kt, kh, kw = patch_size
        return rearrange(
            tensor,
            "... (t kt) c (h kh) (w kw) -> ... (t h w) (c kt kh kw)",
            kt=kt,
            kh=kh,
            kw=kw,
        ).contiguous()

    patched = {
        "i2v.latent": patchify(output.i2v.latent),
        "i2v.mask": patchify(output.i2v.mask),
        "plucker": patchify(output.plucker),
    }
    token_counts = {name: tensor.shape[-2] for name, tensor in patched.items()}
    if len(set(token_counts.values())) != 1:
        raise ValueError(f"Patchified token counts do not match: {token_counts}.")
    token_count = next(iter(token_counts.values()))
    if token_count % cp_size:
        raise ValueError(
            f"Patchified token count {token_count} is not divisible by CP{cp_size}."
        )

    per_field = {
        name: tensor.chunk(cp_size, dim=-2) for name, tensor in patched.items()
    }
    return tuple(
        {name: shards[rank].contiguous() for name, shards in per_field.items()}
        for rank in range(cp_size)
    )


class LingbotEncoderStage(nn.Module):
    """Own LingBot's text, image, streaming VAE, and camera encoders."""

    def __init__(self, config: LingbotWorldInferencePipelineConfig) -> None:
        super().__init__()
        assert config.encoder is not None, "LingBot requires an I2V control encoder."
        encoder = config.encoder.setup()
        assert isinstance(encoder, I2VCamCtrlEncoder)
        self.encoder = encoder
        self.text_encoder = (
            config.text_encoder.setup() if config.text_encoder is not None else None
        )
        self.image_encoder = (
            config.image_encoder.setup() if config.image_encoder is not None else None
        )
        transformer_config = config.diffusion_model.transformer
        assert isinstance(transformer_config, LingbotWorldTransformerConfig)
        self.transformer_config = transformer_config

    @torch.no_grad()
    def initialize_cache(
        self,
        *,
        text: list[str],
        image: Tensor,
    ) -> tuple[LingbotEncoderStageCache, LingbotConditioning]:
        """Encode session-level context and initialize streaming encoder state."""
        assert self.text_encoder is not None, "LingBot text encoder is not configured."
        assert image.shape[-4] == 1, (
            f"image must contain exactly one frame, got shape {tuple(image.shape)}."
        )
        spatial_ratio = self.encoder.spatial_compression_ratio
        pixel_height, pixel_width = image.shape[-2:]
        assert pixel_height % spatial_ratio == 0
        assert pixel_width % spatial_ratio == 0

        text_embeddings = self.text_encoder(text)
        negative_text_embeddings = None
        if self.transformer_config.guidance_scale > 1.0:
            negative_text_embeddings = self.text_encoder([NEGATIVE_PROMPT] * len(text))
        image_embeddings = None
        if self.image_encoder is not None:
            image_embeddings = self.image_encoder(image.squeeze(-4))

        cache = LingbotEncoderStageCache(
            encoder_cache=self.encoder.initialize_autoregressive_cache(),
            image=image,
        )
        conditioning = LingbotConditioning(
            height=pixel_height // spatial_ratio,
            width=pixel_width // spatial_ratio,
            text_embeddings=text_embeddings,
            negative_text_embeddings=negative_text_embeddings,
            image_embeddings=image_embeddings,
        )
        return cache, conditioning

    def get_num_input_frames(self, autoregressive_index: int) -> int:
        """Return the pixel frames consumed by one encoder step."""
        return self.encoder.get_input_temporal_size(
            autoregressive_index,
            self.transformer_config.len_t,
        )

    def _preprocess_i2v_input(
        self,
        autoregressive_index: int,
        image: Tensor,
    ) -> Tensor:
        """Build the first-frame-plus-padding chunk expected by the Wan VAE."""
        expected_frames = self.get_num_input_frames(autoregressive_index)
        if autoregressive_index == 0:
            return F.pad(image, (0, 0, 0, 0, 0, 0, 0, expected_frames - 1))
        return torch.zeros(
            *image.shape[:-4],
            expected_frames,
            3,
            image.shape[-2],
            image.shape[-1],
            device=image.device,
            dtype=image.dtype,
        )

    @torch.no_grad()
    def encode(
        self,
        *,
        autoregressive_index: int,
        cache: LingbotEncoderStageCache,
        input: CamCtrlInput,
    ) -> I2VCamCtrlEmbeddings:
        """Encode one camera-control and I2V chunk."""
        i2v_input = self._preprocess_i2v_input(
            autoregressive_index,
            cache.image,
        )
        return self.encoder(
            input=I2VCamCtrlInput(i2v=i2v_input, camctrl=input),
            autoregressive_index=autoregressive_index,
            cache=cache.encoder_cache,
        )


class LingbotDiTStage(DiffusionStage[LingbotWorldTransformerCache]):
    """Own LingBot's scheduler, DiT weights, and evolving KV cache."""

    def __init__(self, config: LingbotWorldInferencePipelineConfig) -> None:
        super().__init__(config.diffusion_model)

    def set_context_parallel_group(self, cp_group: ProcessGroup | None) -> None:
        """Bind a DiT-only context-parallel group before cache construction."""
        transformer = self.diffusion_model.transformer
        assert isinstance(transformer, LingbotWorldTransformer)
        transformer.set_context_parallel_group(cp_group)

    def configure_pipeline_parallel(
        self,
        *,
        stage_index: int,
        stage_count: int,
        group: ProcessGroup,
        ranks: tuple[int, ...],
    ) -> None:
        """Partition the DiT and bind its ordered NCCL rank group.

        Args:
            stage_index: Zero-based position inside the pipeline group.
            stage_count: Number of pipeline stages.
            group: NCCL process group containing the pipeline ranks.
            ranks: Global ranks ordered from input to output stage.
        """
        transformer = self.diffusion_model.transformer
        assert isinstance(transformer, LingbotWorldTransformer)
        transformer.configure_pipeline_parallel(
            stage_index=stage_index,
            stage_count=stage_count,
            group=group,
            ranks=ranks,
        )

    def initialize_cache(
        self,
        conditioning: LingbotConditioning,
    ) -> DiffusionStageCache[LingbotWorldTransformerCache]:
        """Build the resident DiT cache from encoder-stage conditioning."""
        return super().initialize_cache(
            height=conditioning.height,
            width=conditioning.width,
            text_embeddings=conditioning.text_embeddings,
            negative_text_embeddings=conditioning.negative_text_embeddings,
            image_embeddings=conditioning.image_embeddings,
        )

    @torch.no_grad()
    def generate_double_buffered(
        self,
        *,
        autoregressive_index: int,
        caches: tuple[
            DiffusionStageCache[LingbotWorldTransformerCache],
            DiffusionStageCache[LingbotWorldTransformerCache],
        ],
        inputs: tuple[I2VCamCtrlEmbeddings, I2VCamCtrlEmbeddings],
    ) -> tuple[Tensor, Tensor]:
        """Denoise two sessions with an overlapped two-stage DiT schedule.

        Args:
            autoregressive_index: Shared autoregressive chunk index.
            caches: Two session-affine DiT caches.
            inputs: Two unpatchified encoder outputs.

        Returns:
            Two clean unpatchified latents in session order.

        Raises:
            RuntimeError: The configured scheduler or diffusion mode is unsupported.
            AssertionError: Either session is generated out of order.
        """
        model = self.diffusion_model
        transformer = model.transformer
        if not isinstance(transformer, LingbotWorldTransformer):
            raise RuntimeError("Double buffering requires LingbotWorldTransformer.")
        scheduler = model.scheduler
        if not isinstance(scheduler, FlowMatchScheduler):
            raise RuntimeError("Double buffering requires FlowMatchScheduler.")
        if model.config.noise_in_unpatchified_shape:
            raise RuntimeError("Double buffering requires patchified scheduler noise.")

        for cache in caches:
            previous = cache.autoregressive_index
            expected = previous + 1 if previous is not None else 0
            assert autoregressive_index == expected, (
                f"AR step out of order: previous step was {previous}, expected "
                f"{expected}, got {autoregressive_index}."
            )

        patchified_inputs_list: list[I2VCamCtrlEmbeddings] = []
        for input in inputs:
            patchified = transformer.patchify_and_maybe_split_cp(input)
            assert isinstance(patchified, I2VCamCtrlEmbeddings)
            patchified_inputs_list.append(patchified)
        patchified_inputs = (patchified_inputs_list[0], patchified_inputs_list[1])

        transformer_caches = (
            caches[0].transformer_cache,
            caches[1].transformer_cache,
        )
        for cache in transformer_caches:
            cache.start(autoregressive_index)

        input_dtype = model.dtype
        noisy_latents = (
            torch.randn(
                model.latent_shape,
                device=model.device,
                dtype=input_dtype,
                generator=model.rng,
            ),
            torch.randn(
                model.latent_shape,
                device=model.device,
                dtype=input_dtype,
                generator=model.rng,
            ),
        )
        clean_latents: tuple[Tensor, Tensor] | None = None
        for step_index in range(scheduler.denoising_step_list.shape[0]):
            sigma = scheduler.denoising_sigmas[step_index]
            timestep = scheduler.denoising_step_list[step_index].to(dtype=input_dtype)
            if step_index > 0:
                assert clean_latents is not None
                noises = (
                    torch.empty_like(noisy_latents[0]).normal_(generator=model.rng),
                    torch.empty_like(noisy_latents[1]).normal_(generator=model.rng),
                )
                noisy_latents = (
                    ((1.0 - sigma) * clean_latents[0] + sigma * noises[0]).to(
                        input_dtype
                    ),
                    ((1.0 - sigma) * clean_latents[1] + sigma * noises[1]).to(
                        input_dtype
                    ),
                )
            flows = transformer.predict_flow_double_buffered(
                noisy_latents=noisy_latents,
                timesteps=(timestep, timestep),
                caches=transformer_caches,
                inputs=patchified_inputs,
            )
            clean_latents = (
                noisy_latents[0] - sigma * flows[0],
                noisy_latents[1] - sigma * flows[1],
            )
        assert clean_latents is not None

        patchified_clean = (
            transformer.postprocess_clean_latent(
                clean_latent=clean_latents[0].to(input_dtype),
                cache=transformer_caches[0],
                input=patchified_inputs[0].i2v,
            ),
            transformer.postprocess_clean_latent(
                clean_latent=clean_latents[1].to(input_dtype),
                cache=transformer_caches[1],
                input=patchified_inputs[1].i2v,
            ),
        )
        for index, cache in enumerate(caches):
            cache.autoregressive_index = autoregressive_index
            cache.final_state = model.FinalState(
                clean_latent=patchified_clean[index],
                autoregressive_index=autoregressive_index,
                cache=transformer_caches[index],
                input=patchified_inputs[index],
            )

        return (
            transformer.unpatchify_and_maybe_gather_cp(patchified_clean[0]),
            transformer.unpatchify_and_maybe_gather_cp(patchified_clean[1]),
        )

    @torch.no_grad()
    def finalize_double_buffered(
        self,
        *,
        autoregressive_index: int,
        caches: tuple[
            DiffusionStageCache[LingbotWorldTransformerCache],
            DiffusionStageCache[LingbotWorldTransformerCache],
        ],
    ) -> None:
        """Advance two session caches with an overlapped pipeline forward.

        Args:
            autoregressive_index: Shared autoregressive chunk index.
            caches: Two session-affine DiT caches after generation.

        Raises:
            RuntimeError: Context-noise finalization is enabled.
            AssertionError: A cache does not contain the matching final state.
        """
        model = self.diffusion_model
        transformer = model.transformer
        assert isinstance(transformer, LingbotWorldTransformer)
        if model.config.context_noise != 0:
            raise RuntimeError("Double buffering requires context_noise == 0.")

        final_states = (caches[0].final_state, caches[1].final_state)
        for cache, final_state in zip(caches, final_states):
            assert cache.autoregressive_index == autoregressive_index
            assert final_state is not None

        state_0 = final_states[0]
        state_1 = final_states[1]
        assert state_0 is not None and state_1 is not None
        timestep = torch.tensor(0, device=model.device, dtype=model.dtype)
        inputs = (state_0.input, state_1.input)
        assert isinstance(inputs[0], I2VCamCtrlEmbeddings)
        assert isinstance(inputs[1], I2VCamCtrlEmbeddings)
        transformer.predict_flow_double_buffered(
            noisy_latents=(state_0.clean_latent, state_1.clean_latent),
            timesteps=(timestep, timestep),
            caches=(state_0.cache, state_1.cache),
            inputs=(inputs[0], inputs[1]),
        )
        state_0.cache.finalize(autoregressive_index)
        state_1.cache.finalize(autoregressive_index)
        caches[0].final_state = None
        caches[1].final_state = None


class LingbotDecoderStage(DecoderStage[WanVAECache]):
    """Own LingBot's streaming pixel decoder."""

    decoder: StreamingVideoDecoder[WanVAECache]

    def __init__(self, config: LingbotWorldInferencePipelineConfig) -> None:
        assert config.decoder is not None, "LingBot requires a video decoder."
        super().__init__(config.decoder)
        assert isinstance(self.decoder, StreamingVideoDecoder)

    def get_num_output_frames(self, autoregressive_index: int, len_t: int) -> int:
        """Return decoded pixel frames produced by one latent chunk."""
        return self.decoder.get_output_temporal_size(autoregressive_index, len_t)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-shot native Qwen Image Edit inference."""

from __future__ import annotations

import gc
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn

from flashdreams.core.checkpoint.load import load_checkpoint
from flashdreams.infra.diffusion.scheduler.fm_euler import (
    FlowMatchEulerDiscreteScheduler,
    FlowMatchEulerDiscreteSchedulerConfig,
)

from .config import QWEN_IMAGE_EDIT_2511, QwenImageEditConfig
from .transformer import QwenImageTransformer, true_cfg
from .vae import QwenImageVAE

_PROMPT_TEMPLATE = """<|im_start|>system
Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>
<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
"""


def calculate_dimensions(target_area: int, ratio: float) -> tuple[int, int]:
    """Return aspect-preserving dimensions rounded to Qwen's 32-pixel grid."""
    width = round(math.sqrt(target_area * ratio) / 32) * 32
    height = round(width / ratio / 32) * 32
    return width, height


def pack_latents(latents: Tensor) -> Tensor:
    """Pack each 2-by-2 latent patch into one transformer token."""
    batch, channels, height, width = latents.shape
    return (
        latents.view(batch, channels, height // 2, 2, width // 2, 2)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(batch, height // 2 * (width // 2), channels * 4)
    )


def unpack_latents(latents: Tensor, height: int, width: int) -> Tensor:
    """Unpack transformer tokens into 8x-compressed VAE latents."""
    batch, _, channels = latents.shape
    latent_height, latent_width = height // 8, width // 8
    return (
        latents.view(
            batch,
            latent_height // 2,
            latent_width // 2,
            channels // 4,
            2,
            2,
        )
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(batch, channels // 4, latent_height, latent_width)
    )


def _checkpoint_url(config: QwenImageEditConfig, path: str) -> str:
    return f"https://huggingface.co/{config.repo_id}/resolve/{config.revision}/{path}"


class QwenImageEditor:
    """Edit one image with native Qwen transformer and VAE implementations.

    Components are moved onto the accelerator one at a time so the 7B vision
    encoder, diffusion transformer, and VAE do not consume device memory
    concurrently.
    """

    def __init__(
        self,
        config: QwenImageEditConfig = QWEN_IMAGE_EDIT_2511,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.dtype = dtype
        self.processor = None
        self.text_encoder: nn.Module | None = None
        self.transformer: QwenImageTransformer | None = None
        self.vae: QwenImageVAE | None = None

    @staticmethod
    def _resize(image: Image.Image, size: tuple[int, int]) -> Image.Image:
        return image.convert("RGB").resize(size, Image.Resampling.LANCZOS)

    @staticmethod
    def _pixels(image: Image.Image, device: torch.device, dtype: torch.dtype) -> Tensor:
        array = np.asarray(image, dtype=np.float32).copy()
        return (
            torch.from_numpy(array)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=device, dtype=dtype)
            .div_(127.5)
            .sub_(1.0)
        )

    def _release_device(self, module: nn.Module) -> None:
        module.to("cpu")
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _load_text_encoder(self) -> None:
        if self.text_encoder is not None:
            return
        from transformers import (
            Qwen2_5_VLForConditionalGeneration,
            Qwen2Tokenizer,
            Qwen2VLImageProcessor,
            Qwen2VLProcessor,
        )
        from transformers.models.qwen2_vl.video_processing_qwen2_vl import (
            Qwen2VLVideoProcessor,
        )

        image_processor = Qwen2VLImageProcessor.from_pretrained(
            self.config.repo_id,
            subfolder="processor",
            revision=self.config.revision,
        )
        tokenizer = Qwen2Tokenizer.from_pretrained(
            self.config.repo_id,
            subfolder="tokenizer",
            revision=self.config.revision,
        )
        video_processor = Qwen2VLVideoProcessor.from_pretrained(
            self.config.repo_id,
            subfolder="processor",
            revision=self.config.revision,
        )
        self.processor = Qwen2VLProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
        )
        self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config.repo_id,
            subfolder="text_encoder",
            revision=self.config.revision,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        ).eval()

    def _load_transformer(self) -> None:
        if self.transformer is not None:
            return
        self.transformer = QwenImageTransformer().to(dtype=self.dtype).eval()
        load_checkpoint(
            _checkpoint_url(
                self.config,
                "transformer/diffusion_pytorch_model.safetensors.index.json",
            ),
            model=self.transformer,
        )

    def _load_vae(self) -> None:
        if self.vae is not None:
            return
        self.vae = QwenImageVAE().to(dtype=self.dtype).eval()
        load_checkpoint(
            _checkpoint_url(
                self.config,
                "vae/diffusion_pytorch_model.safetensors",
            ),
            model=self.vae,
        )

    @staticmethod
    def _masked_embeddings(
        hidden_states: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        selected = hidden_states[mask.bool()][64:]
        return selected.unsqueeze(0), torch.ones(
            (1, selected.shape[0]), device=selected.device, dtype=torch.long
        )

    @torch.inference_mode()
    def _encode_prompt(
        self, prompt: str, condition_image: Image.Image
    ) -> tuple[Tensor, Tensor]:
        assert self.processor is not None and self.text_encoder is not None
        text = _PROMPT_TEMPLATE.format(
            "Picture 1: <|vision_start|><|image_pad|><|vision_end|>" + prompt
        )
        inputs = self.processor(
            text=[text], images=[condition_image], padding=True, return_tensors="pt"
        ).to(self.device)
        outputs = self.text_encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            output_hidden_states=True,
        )
        return self._masked_embeddings(
            outputs.hidden_states[-1], inputs["attention_mask"]
        )

    @torch.inference_mode()
    def generate(
        self,
        image: Image.Image | str | Path,
        prompt: str,
        *,
        output_size: tuple[int, int],
        seed: int = 0,
        num_inference_steps: int | None = None,
        negative_prompt: str | None = None,
        true_cfg_scale: float | None = None,
    ) -> Image.Image:
        """Generate one edited RGB image.

        Args:
            image: Input image or filesystem path.
            prompt: Image-edit instruction.
            output_size: Output ``(width, height)``; both values must be
                divisible by 16.
            seed: Deterministic initial-noise seed.
            num_inference_steps: Optional sampling-step override.
            negative_prompt: Optional classifier-free negative instruction.
            true_cfg_scale: Optional classifier-free guidance override.

        Returns:
            Generated RGB image.
        """
        if isinstance(image, (str, Path)):
            with Image.open(image) as opened:
                image = opened.convert("RGB")
        else:
            image = image.convert("RGB")
        width, height = output_size
        if width <= 0 or height <= 0 or width % 16 or height % 16:
            raise ValueError("output dimensions must be positive multiples of 16")
        steps = num_inference_steps or self.config.num_inference_steps
        if steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        negative_prompt = (
            self.config.negative_prompt if negative_prompt is None else negative_prompt
        )
        true_cfg_scale = (
            self.config.true_cfg_scale if true_cfg_scale is None else true_cfg_scale
        )
        if true_cfg_scale <= 1.0:
            raise ValueError("true_cfg_scale must be greater than 1")

        ratio = image.width / image.height
        condition_size = calculate_dimensions(384 * 384, ratio)
        vae_size = calculate_dimensions(1024 * 1024, ratio)
        condition_image = self._resize(image, condition_size)
        vae_image = self._resize(image, vae_size)

        self._load_text_encoder()
        assert self.text_encoder is not None
        self.text_encoder.to(self.device)
        prompt_embeds, prompt_mask = self._encode_prompt(prompt, condition_image)
        negative_embeds, negative_mask = self._encode_prompt(
            negative_prompt, condition_image
        )
        prompt_embeds, prompt_mask = prompt_embeds.cpu(), prompt_mask.cpu()
        negative_embeds, negative_mask = negative_embeds.cpu(), negative_mask.cpu()
        self._release_device(self.text_encoder)

        self._load_vae()
        assert self.vae is not None
        self.vae.to(self.device)
        reference = self.vae.encode(self._pixels(vae_image, self.device, self.dtype))[
            :, :, 0
        ]
        packed_reference = pack_latents(reference).cpu()
        reference_shape = (1, reference.shape[-2] // 2, reference.shape[-1] // 2)
        self._release_device(self.vae)

        self._load_transformer()
        assert self.transformer is not None
        transformer = self.transformer
        transformer.to(self.device)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noise = torch.randn(
            (1, 16, height // 8, width // 8),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        output = pack_latents(noise)
        reference_tokens = packed_reference.to(self.device)
        output_shape = (1, height // 16, width // 16)
        image_shapes = [output_shape, reference_shape]
        prompt_embeds, prompt_mask = (
            prompt_embeds.to(self.device),
            prompt_mask.to(self.device),
        )
        negative_embeds = negative_embeds.to(self.device)
        negative_mask = negative_mask.to(self.device)
        scheduler = FlowMatchEulerDiscreteScheduler(
            FlowMatchEulerDiscreteSchedulerConfig(
                num_inference_steps=steps,
                use_dynamic_shifting=True,
                base_image_seq_len=256,
                max_image_seq_len=8192,
                base_shift=0.5,
                max_shift=0.9,
                shift_terminal=0.02,
                time_shift_type="exponential",
            )
        ).to(self.device)

        def predict_flow(noisy_latent: Tensor, timestep: Tensor) -> Tensor:
            model_input = torch.cat([noisy_latent, reference_tokens], dim=1)
            time = (timestep / 1000).expand(noisy_latent.shape[0])
            conditional = transformer(
                model_input, prompt_embeds, time, image_shapes, prompt_mask
            )[:, : noisy_latent.shape[1]]
            unconditional = transformer(
                model_input, negative_embeds, time, image_shapes, negative_mask
            )[:, : noisy_latent.shape[1]]
            return true_cfg(conditional, unconditional, true_cfg_scale)

        output = scheduler.sample_for_sequence_length(
            output, predict_flow, output.shape[1]
        )
        decoded_latents = unpack_latents(output, height, width).cpu()
        self._release_device(transformer)

        self.vae.to(self.device)
        pixels = self.vae.decode(decoded_latents.to(self.device))
        pixels = (
            pixels.add(1)
            .mul(127.5)
            .round()
            .clamp(0, 255)
            .to(torch.uint8)[0]
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        self._release_device(self.vae)
        return Image.fromarray(pixels, mode="RGB")


__all__ = [
    "QwenImageEditor",
    "calculate_dimensions",
    "pack_latents",
    "unpack_latents",
]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Architecture dimensions for the Cosmos3 Mixture-of-Transformer network.

Qwen3-VL defines the text and vision dimensions. Video latent dimensions,
patching and temporal settings describe the supported checkpoint layout.
This module depends only on the Python standard library.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LLMSpec:
    """Qwen3-VL-8B text-tower dims. Each layer is doubled into a MoT (und + gen)."""

    hidden_size: int = 4096
    num_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8  # grouped-query attention
    head_dim: int = 128
    intermediate_size: int = 12288  # SwiGLU gate/up/down
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5_000_000.0
    mrope_section: tuple[int, int, int] = (24, 20, 20)
    mrope_interleaved: bool = True
    qk_norm: bool = True  # per-head RMSNorm on Q and K
    attention_bias: bool = False

    def __post_init__(self) -> None:
        assert self.num_attention_heads * self.head_dim == self.hidden_size, (
            f"heads*head_dim ({self.num_attention_heads}*{self.head_dim}) "
            f"!= hidden_size ({self.hidden_size})"
        )
        assert self.num_attention_heads % self.num_key_value_heads == 0, (
            "num_attention_heads must be divisible by num_key_value_heads (GQA)"
        )
        assert sum(self.mrope_section) * 2 == self.head_dim, (
            f"2*sum(mrope_section) ({2 * sum(self.mrope_section)}) must equal "
            f"head_dim ({self.head_dim})"
        )


@dataclass(frozen=True)
class VisionSpec:
    """Qwen3-VL ViT tower dims (used for image-to-video conditioning only)."""

    depth: int = 27
    hidden_size: int = 1152
    num_heads: int = 16
    intermediate_size: int = 4304
    in_channels: int = 3
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    num_position_embeddings: int = 2304
    out_hidden_size: int = 4096  # merges to the LLM hidden size
    deepstack_visual_indexes: tuple[int, ...] = (8, 16, 24)


@dataclass(frozen=True)
class VFMSpec:
    """Video-foundation-model wrapper dims (tokenizer, adapters, AR layout)."""

    # Wan 2.2 VAE (4x16x16): temporal /4, spatial /16.
    vae_latent_channels: int = 48  # exported config.json ``state_ch``
    temporal_compression_factor: int = 4
    spatial_compression_factor: int = 16
    latent_patch_size: int = 2  # DiT ``patch_spatial``
    latent_downsample_factor: int = 16
    # Streaming / AR layout for the chunk=4 checkpoint.
    resolution: int = 480
    ar_chunk_latent_frames: int = 4  # "chunk4"
    max_frames: int = 1601
    # Few-step distilled schedule (rectified flow); 4 SDE steps.
    num_inference_steps: int = 4
    fixed_step_t_list: tuple[float, ...] = (1.0, 0.9375, 0.8333333333333334, 0.625)

    # Diffusion timestep embedder: sinusoidal freq dim -> MLP into hidden.
    # Verified from the checkpoint: time_embedder.mlp.0 is Linear(256 -> 4096),
    # .2 is Linear(4096 -> 4096).
    time_embed_freq_dim: int = 256
    # Raw diffusion timesteps are multiplied by this before the sinusoidal embed
    # (Cosmos3 ``timestep_scale``).
    timestep_scale: float = 0.001

    # 3D mRoPE conventions (exported config.json ``diffusion_expert_config``).
    # The margin is added to the temporal offset at the text->vision boundary so
    # the two modalities occupy disjoint temporal bands: text spans 0..N_text-1,
    # while vision starts at N_text + margin + frame.
    mrope_temporal_modality_margin: int = 15000
    mrope_reset_spatial_ids: bool = True
    enable_fps_modulation: bool = True
    base_fps: float = 24.0

    @property
    def patch_latent_dim(self) -> int:
        """Post-patchify per-token channel width feeding ``vae2llm``."""
        return self.latent_patch_size**2 * self.vae_latent_channels


@dataclass(frozen=True)
class Cosmos3InteractiveSpec:
    """Full model spec: MoT-on-Qwen3-VL VFM, vision-only no-action variant."""

    llm: LLMSpec = field(default_factory=LLMSpec)
    vision: VisionSpec = field(default_factory=VisionSpec)
    vfm: VFMSpec = field(default_factory=VFMSpec)

    # MoT / causal-video wiring.
    joint_attn_implementation: str = "three_way"
    video_temporal_causal: bool = True
    vision_gen: bool = True
    action_gen: bool = False
    sound_gen: bool = False

    # Total exported params / on-disk size (understanding + gen expert + ViT).
    # NOTE: the checkpoint metadata's ``total_parameters`` (15_156_222_144) counts
    # the LLM MoT + adapters only and *excludes* the 576_388_336-param ViT tower;
    # the true full count is ``total_bytes // 2`` (bf16) = 15_732_610_480.
    total_parameters: int = 15_732_610_480  # full, incl. ViT
    llm_parameters: int = 15_156_222_144  # LLM MoT + adapters, excl. ViT
    total_bytes: int = 31_465_220_960

    def expected_layer_key_leaves(self) -> tuple[str, ...]:
        """The 11 per-pathway tensor leaves each layer carries (und and gen)."""
        return (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        )

    def expected_top_level_keys(self) -> tuple[str, ...]:
        """Non-layer, non-ViT target keys.

        Note both MoT pathways carry a *top-level* final RMSNorm: ``norm``
        (understanding) and ``norm_moe_gen`` (generation). The three VFM
        adapters (``vae2llm`` / ``llm2vae`` / ``time_embedder``) bridge the
        video latent and diffusion timestep into the shared LLM space.
        """
        return (
            "embed_tokens.weight",
            "norm.weight",
            "norm_moe_gen.weight",
            "lm_head.weight",
            "vae2llm.weight",
            "vae2llm.bias",
            "llm2vae.weight",
            "llm2vae.bias",
            "time_embedder.mlp.0.weight",
            "time_embedder.mlp.0.bias",
            "time_embedder.mlp.2.weight",
            "time_embedder.mlp.2.bias",
        )


SPEC = Cosmos3InteractiveSpec()
"""Module-level singleton — import this."""


def _summary() -> str:
    s = SPEC
    return "\n".join(
        [
            "Cosmos3 Interactive 8B (no-action) — architecture spec",
            f"  params            : {s.total_parameters:,} ({s.total_bytes / 1e9:.1f} GB)",
            f"  LLM               : hidden={s.llm.hidden_size} layers={s.llm.num_layers} "
            f"heads={s.llm.num_attention_heads} kv={s.llm.num_key_value_heads} "
            f"head_dim={s.llm.head_dim} mlp={s.llm.intermediate_size}",
            f"  MoT               : per-layer und + gen ({len(s.expected_layer_key_leaves())} leaves each), "
            f"attn={s.joint_attn_implementation} causal={s.video_temporal_causal}",
            f"  ViT               : depth={s.vision.depth} hidden={s.vision.hidden_size} "
            f"heads={s.vision.num_heads} -> out {s.vision.out_hidden_size}",
            f"  VFM tokenizer     : Wan2.2 VAE ch={s.vfm.vae_latent_channels} "
            f"(t/{s.vfm.temporal_compression_factor}, s/{s.vfm.spatial_compression_factor}), "
            f"patch_latent_dim={s.vfm.patch_latent_dim}",
            f"  AR / streaming    : res={s.vfm.resolution}p chunk={s.vfm.ar_chunk_latent_frames} "
            f"steps={s.vfm.num_inference_steps} max_frames={s.vfm.max_frames}",
        ]
    )


if __name__ == "__main__":
    print(_summary())

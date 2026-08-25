# SPDX-FileCopyrightText: Copyright 2025 The MiniMax authors and The HuggingFace Team. All rights reserved.
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

"""Native MiniMax H3 waveform autoencoder and stereo output adapter.

Modified from the Apache-2.0 H3 audio VAE in Hugging Face Diffusers commit
``175fe6b2419a01db9c2ceabd01ec37d2c0305fc2``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from flashdreams.core.checkpoint.load import load_checkpoint
from flashdreams.infra.config import InstantiateConfig
from flashdreams.runtime_v2.audio_output import AudioOutput
from torch import Tensor, nn
from torch.nn.utils import weight_norm

H3_AUDIO_VAE_CHECKPOINT = (
    "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/"
    "42ed227ee7df40d41602854ae760620d6eb651fe/audio_vae/"
    "diffusion_pytorch_model.safetensors"
)
"""Immutable released MiniMax H3 audio VAE checkpoint."""

_AUDIO_LATENTS_MEAN = (
    -0.020211687488382354,
    0.3876466479950502,
    -0.04398279799186767,
    -0.28591514936373,
    0.08179686214561671,
    -0.35782641352446604,
    0.040623809960919084,
    -0.01552534501956604,
    -0.223362481667332,
    0.1821006842509091,
    0.2941778783780663,
    -0.07901167601970885,
    -0.056815072777201,
    -0.3699028221860095,
    -0.31616315591624855,
    0.5905951377425391,
    -0.052139568068853864,
    0.013673160263486295,
    -0.03691647864630577,
    0.09732660653298163,
    -0.3394662328788498,
    -0.30685677538541667,
    -0.24504598907458763,
    -0.034698524462007344,
    0.02868032184767538,
    -0.21217779266454084,
    -0.1678263169941987,
    0.3221287889040614,
    -0.1223055851554907,
    0.4356604928128464,
    -0.0502599202236253,
    0.3979258376211797,
)
"""Released per-channel mean for H3 audio diffusion latents."""

_AUDIO_LATENTS_STD = (
    1.6895524230479284,
    2.76263727217653,
    1.7945344281264435,
    1.6801681847309828,
    1.6390226546605453,
    2.7788298348882177,
    1.7659090095747236,
    1.6199757612137327,
    2.6336525640336896,
    1.8539356672817833,
    2.5056497896915633,
    1.811019237886178,
    1.9579657790720237,
    1.6685498243529284,
    1.4922469314453364,
    3.298670198067373,
    1.9491804496832168,
    1.8720003270431442,
    1.8334080103291832,
    1.6488070416529093,
    1.6176957696319716,
    1.9131449234774398,
    1.5695245398428617,
    1.6943659940415912,
    1.8318420762504692,
    1.5540637421583379,
    1.9344930328968526,
    1.599198216109855,
    1.718045989838149,
    1.6307219190837705,
    1.8661226051202384,
    1.5613768203168363,
)
"""Released per-channel standard deviation for H3 audio diffusion latents."""


class MiniMaxH3AudioDiagonalGaussianDistribution:
    """Posterior parameterized by separate mean and log-standard-deviation heads."""

    def __init__(self, mean: Tensor, logs: Tensor):
        self.mean = mean
        self.logs = logs
        self.std = torch.exp(logs)

    def mode(self) -> Tensor:
        return self.mean

    def sample(self, generator: torch.Generator | None = None) -> Tensor:
        noise_device = self.mean.device if generator is None else generator.device
        noise = torch.randn(
            self.mean.shape,
            generator=generator,
            device=noise_device,
            dtype=self.mean.dtype,
        ).to(self.mean.device)
        return self.mean + self.std * noise


@dataclass
class MiniMaxH3AudioEncoderOutput:
    """Audio posterior returned by the native waveform encoder."""

    latent_dist: MiniMaxH3AudioDiagonalGaussianDistribution
    """Posterior over the encoded audio latents."""


def kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> Tensor:
    """Build the persistent alias-free Kaiser-sinc filter."""
    half_size = kernel_size // 2

    attenuation = 2.285 * (half_size - 1) * math.pi * (4 * half_width) + 7.95
    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)

    if kernel_size % 2 == 0:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size

    filter_ = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    # Normalize to sum 1 so a constant input does not leak through the resampler.
    filter_ /= filter_.sum()
    return filter_.view(1, 1, kernel_size)


class MiniMaxH3AudioSnake1d(nn.Module):
    """Apply the per-channel Snake activation used by the DAC encoder."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, hidden_states: Tensor) -> Tensor:
        return hidden_states + (self.alpha + 1e-9).reciprocal() * torch.sin(
            self.alpha * hidden_states
        ).pow(2)


class MiniMaxH3AudioSnakeBeta(nn.Module):
    """Apply the log-parameterized SnakeBeta activation used by BigVGAN."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, hidden_states: Tensor) -> Tensor:
        alpha = torch.exp(self.alpha.unsqueeze(0).unsqueeze(-1))
        beta = torch.exp(self.beta.unsqueeze(0).unsqueeze(-1))
        return hidden_states + (beta + 1e-9).reciprocal() * torch.sin(
            alpha * hidden_states
        ).pow(2)


class MiniMaxH3AudioLowPassFilter1d(nn.Module):
    """Apply a depthwise Kaiser-sinc low-pass filter with stride."""

    def __init__(self, cutoff: float, half_width: float, stride: int, kernel_size: int):
        super().__init__()
        even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.register_buffer(
            "filter", kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        num_channels = hidden_states.shape[1]
        hidden_states = F.pad(
            hidden_states, (self.pad_left, self.pad_right), mode="replicate"
        )
        return F.conv1d(
            hidden_states,
            self.filter.expand(num_channels, -1, -1),
            stride=self.stride,
            groups=num_channels,
        )


class MiniMaxH3AudioUpSample1d(nn.Module):
    """Apply alias-free transposed-convolution upsampling."""

    def __init__(self, ratio: int, kernel_size: int):
        super().__init__()
        self.ratio = ratio
        self.stride = ratio
        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (kernel_size - self.stride + 1) // 2
        self.register_buffer(
            "filter",
            kaiser_sinc_filter1d(
                cutoff=0.5 / ratio,
                half_width=0.6 / ratio,
                kernel_size=kernel_size,
            ),
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        num_channels = hidden_states.shape[1]
        hidden_states = F.pad(hidden_states, (self.pad, self.pad), mode="replicate")
        hidden_states = self.ratio * F.conv_transpose1d(
            hidden_states,
            self.filter.expand(num_channels, -1, -1),
            stride=self.stride,
            groups=num_channels,
        )
        return hidden_states[..., self.pad_left : -self.pad_right]


class MiniMaxH3AudioDownSample1d(nn.Module):
    """Apply alias-free Kaiser-sinc downsampling."""

    def __init__(self, ratio: int, kernel_size: int):
        super().__init__()
        self.lowpass = MiniMaxH3AudioLowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=kernel_size,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.lowpass(hidden_states)


class MiniMaxH3AudioActivation1d(nn.Module):
    """Wrap an activation with alias-free upsampling and downsampling."""

    def __init__(self, activation: nn.Module, ratio: int = 2, kernel_size: int = 12):
        super().__init__()
        self.act = activation
        self.upsample = MiniMaxH3AudioUpSample1d(ratio, kernel_size)
        self.downsample = MiniMaxH3AudioDownSample1d(ratio, kernel_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.upsample(hidden_states)
        hidden_states = self.act(hidden_states)
        return self.downsample(hidden_states)


class MiniMaxH3AudioResidualUnit(nn.Module):
    """Apply one dilated DAC residual unit."""

    def __init__(self, dim: int, dilation: int):
        super().__init__()
        self.block = nn.Sequential(
            MiniMaxH3AudioSnake1d(dim),
            weight_norm(
                nn.Conv1d(
                    dim,
                    dim,
                    kernel_size=7,
                    dilation=dilation,
                    padding=((7 - 1) * dilation) // 2,
                )
            ),
            MiniMaxH3AudioSnake1d(dim),
            weight_norm(nn.Conv1d(dim, dim, kernel_size=1)),
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = self.block(hidden_states)
        pad = (hidden_states.shape[-1] - residual.shape[-1]) // 2
        if pad > 0:
            hidden_states = hidden_states[..., pad:-pad]
        return hidden_states + residual


class MiniMaxH3AudioEncoderBlock(nn.Module):
    """Apply three DAC residual units and one strided channel expansion."""

    def __init__(self, dim: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            MiniMaxH3AudioResidualUnit(dim // 2, dilation=1),
            MiniMaxH3AudioResidualUnit(dim // 2, dilation=3),
            MiniMaxH3AudioResidualUnit(dim // 2, dilation=9),
            MiniMaxH3AudioSnake1d(dim // 2),
            weight_norm(
                nn.Conv1d(
                    dim // 2,
                    dim,
                    kernel_size=2 * stride,
                    stride=stride,
                    padding=math.ceil(stride / 2),
                )
            ),
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.block(hidden_states)


class MiniMaxH3AudioEncoder(nn.Module):
    """DAC encoder from mono waveforms to a downsampled latent trunk."""

    def __init__(self, d_model: int, strides: tuple[int, ...], d_latent: int):
        super().__init__()
        block: list[nn.Module] = [
            weight_norm(nn.Conv1d(1, d_model, kernel_size=7, padding=3))
        ]
        for stride in strides:
            d_model *= 2
            block.append(MiniMaxH3AudioEncoderBlock(d_model, stride=stride))
        block += [
            MiniMaxH3AudioSnake1d(d_model),
            weight_norm(nn.Conv1d(d_model, d_latent, kernel_size=3, padding=1)),
        ]
        self.block = nn.Sequential(*block)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.block(hidden_states)


class MiniMaxH3AudioGeGluMlp(nn.Module):
    """Apply the pre-normalized GeGLU projection MLP."""

    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.act = nn.GELU(approximate="tanh")
        self.w0 = nn.Linear(in_features, hidden_features)
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(hidden_features, in_features)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.norm(hidden_states)
        hidden_states = self.act(self.w0(hidden_states)) * self.w1(hidden_states)
        return self.w2(hidden_states)


class MiniMaxH3AudioCausalAttention(nn.Module):
    """Narrow features through H3's mean-pooled causal self-attention."""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int):
        super().__init__()
        if in_dim % num_heads:
            raise ValueError("Audio attention input width must divide num_heads.")
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        self.qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(in_dim))
        self.v_bias = nn.Parameter(torch.zeros(in_dim))
        self.register_buffer("zero_k_bias", torch.zeros(in_dim))
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, hidden_states: Tensor) -> Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        qkv = F.linear(
            hidden_states,
            self.qkv.weight,
            torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)),
        )
        query, key, value = qkv.reshape(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            self.head_dim,
        ).permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(
            query, key, value, is_causal=True
        ).permute(0, 2, 1, 3)
        attended = attended.mean(dim=2)
        attended = F.adaptive_avg_pool1d(attended, self.out_dim)
        return self.proj(attended)


class MiniMaxH3AudioAttnProjection(nn.Module):
    """Project the encoder trunk into diffusion channels with causal attention."""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int, mlp_ratio: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = MiniMaxH3AudioCausalAttention(in_dim, out_dim, num_heads)
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.mlp = MiniMaxH3AudioGeGluMlp(
            in_features=out_dim, hidden_features=out_dim * mlp_ratio
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.proj(self.norm3(hidden_states)) + self.attn(
            self.norm1(hidden_states)
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


class MiniMaxH3AudioAMPBlock(nn.Module):
    """Apply one anti-aliased BigVGAN multi-periodicity block."""

    def __init__(self, channels: int, kernel_size: int, dilation: tuple[int, ...]):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        dilation=d,
                        padding=(kernel_size * d - d) // 2,
                    )
                )
                for d in dilation
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        dilation=1,
                        padding=(kernel_size - 1) // 2,
                    )
                )
                for _ in dilation
            ]
        )
        self.activations = nn.ModuleList(
            [
                MiniMaxH3AudioActivation1d(
                    activation=MiniMaxH3AudioSnakeBeta(channels)
                )
                for _ in range(2 * len(dilation))
            ]
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for conv1, conv2, act1, act2 in zip(self.convs1, self.convs2, acts1, acts2):
            residual = conv1(act1(hidden_states))
            residual = conv2(act2(residual))
            hidden_states = residual + hidden_states
        return hidden_states


class MiniMaxH3AudioBigVGANDecoder(nn.Module):
    """Decode mono latent batches into normalized waveforms."""

    def __init__(
        self,
        in_channels: int,
        upsample_initial_channel: int,
        upsample_rates: tuple[int, ...],
        upsample_kernel_sizes: tuple[int, ...],
        resblock_kernel_sizes: tuple[int, ...],
        resblock_dilation_sizes: tuple[tuple[int, ...], ...],
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.conv_pre = weight_norm(
            nn.Conv1d(in_channels, upsample_initial_channel, 7, 1, padding=3)
        )

        # Keep the original one-element ``ModuleList`` nesting so checkpoint keys
        # retain the released ``ups.<index>.0`` spelling.
        self.ups = nn.ModuleList()
        for i, (rate, kernel) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.ModuleList(
                    [
                        weight_norm(
                            nn.ConvTranspose1d(
                                upsample_initial_channel // (2**i),
                                upsample_initial_channel // (2 ** (i + 1)),
                                kernel,
                                rate,
                                padding=(kernel - rate) // 2,
                            )
                        )
                    ]
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(self.num_upsamples):
            channels = upsample_initial_channel // (2 ** (i + 1))
            for kernel, dilation in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(
                    MiniMaxH3AudioAMPBlock(channels, kernel, tuple(dilation))
                )

        self.activation_post = MiniMaxH3AudioActivation1d(
            activation=MiniMaxH3AudioSnakeBeta(channels)
        )
        self.conv_post = weight_norm(
            nn.Conv1d(channels, 1, 7, 1, padding=3, bias=False)
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.conv_pre(hidden_states)

        for i in range(self.num_upsamples):
            hidden_states = self.ups[i][0](hidden_states)
            residual: Tensor | None = None
            for j in range(self.num_kernels):
                block = self.resblocks[i * self.num_kernels + j](hidden_states)
                residual = block if residual is None else residual + block
            if residual is None:
                raise RuntimeError("BigVGAN requires at least one residual kernel.")
            hidden_states = residual / self.num_kernels

        hidden_states = self.activation_post(hidden_states)
        hidden_states = self.conv_post(hidden_states)
        return torch.clamp(hidden_states, min=-1.0, max=1.0)


@dataclass(kw_only=True)
class MiniMaxH3AudioVAEConfig(InstantiateConfig):
    """Configure the native FP32 MiniMax H3 waveform autoencoder."""

    _target: type[MiniMaxH3AudioVAE] = field(
        default_factory=lambda: MiniMaxH3AudioVAE
    )
    checkpoint_path: str | None = H3_AUDIO_VAE_CHECKPOINT
    """Checkpoint URL or local path; ``None`` keeps initialized weights."""

    checkpoint_min_free_gb: float | None = None
    """Optional free-space floor used before checkpoint downloads."""

    device: str = "cpu"
    """Device on which parameters and persistent filters are constructed."""

    encoder_dim: int = 64
    """Initial DAC encoder width."""

    encoder_rates: tuple[int, ...] = (2, 4, 4, 5, 5)
    """Encoder strides whose product is the waveform hop length."""

    latent_dim: int = 2048
    """Width of the DAC trunk and BigVGAN input projection."""

    latent_channels: int = 32
    """Width of the normalized diffusion audio latent."""

    num_attention_heads: int = 8
    """Head count in the causal latent projection."""

    decoder_dim: int = 1024
    """Initial BigVGAN decoder width."""

    decoder_rates: tuple[int, ...] = (5, 5, 2, 2, 2, 2, 2)
    """BigVGAN upsampling rates whose product equals the hop length."""

    decoder_kernel_sizes: tuple[int, ...] = (9, 9, 4, 4, 4, 4, 4)
    """Transposed-convolution kernel size for each decoder rate."""

    resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11)
    """Parallel AMP residual kernel sizes at every decoder stage."""

    resblock_dilation_sizes: tuple[tuple[int, ...], ...] = (
        (1, 3, 5),
        (1, 3, 5),
        (1, 3, 5),
    )
    """Per-kernel AMP dilation schedules."""

    sampling_rate: int = 32_000
    """Generated waveform rate in samples per second."""

    latents_mean: tuple[float, ...] = _AUDIO_LATENTS_MEAN
    """Released per-channel latent means."""

    latents_std: tuple[float, ...] = _AUDIO_LATENTS_STD
    """Released per-channel latent standard deviations."""


class MiniMaxH3AudioVAE(nn.Module):
    """Encode and decode H3's mono-batch, 32 kHz waveform latents in FP32."""

    config: MiniMaxH3AudioVAEConfig

    def __init__(self, config: MiniMaxH3AudioVAEConfig) -> None:
        super().__init__()
        self.config = config
        self._validate_config()
        encoder_rates = tuple(int(rate) for rate in config.encoder_rates)
        decoder_rates = tuple(int(rate) for rate in config.decoder_rates)
        self.hop_length = math.prod(encoder_rates)

        with torch.device(config.device):
            self.encoder = MiniMaxH3AudioEncoder(
                d_model=config.encoder_dim,
                strides=encoder_rates,
                d_latent=config.latent_dim,
            )
            self.pre_block = MiniMaxH3AudioAttnProjection(
                config.latent_dim,
                config.latent_channels,
                num_heads=config.num_attention_heads,
            )
            self.mean_proj = nn.Conv1d(
                config.latent_channels, config.latent_channels, 1
            )
            self.logs_proj = nn.Conv1d(
                config.latent_channels, config.latent_channels, 1
            )
            self.dec_in_proj = nn.Conv1d(
                config.latent_channels, config.latent_dim, 1
            )
            self.decoder = MiniMaxH3AudioBigVGANDecoder(
                in_channels=config.latent_dim,
                upsample_initial_channel=config.decoder_dim,
                upsample_rates=decoder_rates,
                upsample_kernel_sizes=tuple(
                    int(kernel) for kernel in config.decoder_kernel_sizes
                ),
                resblock_kernel_sizes=tuple(
                    int(kernel) for kernel in config.resblock_kernel_sizes
                ),
                resblock_dilation_sizes=tuple(
                    tuple(int(dilation) for dilation in dilations)
                    for dilations in config.resblock_dilation_sizes
                ),
            )
        self.float()
        if config.checkpoint_path is not None:
            load_checkpoint(
                config.checkpoint_path,
                model=self,
                checkpoint_min_free_gb=config.checkpoint_min_free_gb,
            )
        self.eval()

    def _validate_config(self) -> None:
        config = self.config
        positive_integers = {
            "encoder_dim": config.encoder_dim,
            "latent_dim": config.latent_dim,
            "latent_channels": config.latent_channels,
            "num_attention_heads": config.num_attention_heads,
            "decoder_dim": config.decoder_dim,
            "sampling_rate": config.sampling_rate,
        }
        for name, value in positive_integers.items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if not config.encoder_rates or not config.decoder_rates:
            raise ValueError("Audio VAE rate sequences cannot be empty.")
        rates = (*config.encoder_rates, *config.decoder_rates)
        if any(type(rate) is not int or rate <= 0 for rate in rates):
            raise ValueError("Audio VAE rates must be positive integers.")
        hop_length = math.prod(config.encoder_rates)
        if math.prod(config.decoder_rates) != hop_length:
            raise ValueError("Decoder rates must upsample by the encoder hop length.")
        if len(config.decoder_rates) != len(config.decoder_kernel_sizes):
            raise ValueError("Each decoder rate requires one kernel size.")
        if not config.resblock_kernel_sizes or len(
            config.resblock_kernel_sizes
        ) != len(config.resblock_dilation_sizes):
            raise ValueError("Each residual kernel requires one dilation schedule.")
        kernels = (*config.decoder_kernel_sizes, *config.resblock_kernel_sizes)
        dilations = tuple(
            dilation
            for schedule in config.resblock_dilation_sizes
            for dilation in schedule
        )
        if any(type(kernel) is not int or kernel <= 0 for kernel in kernels):
            raise ValueError("Audio VAE kernel sizes must be positive integers.")
        if not dilations or any(
            type(dilation) is not int or dilation <= 0 for dilation in dilations
        ):
            raise ValueError("Audio VAE dilations must be positive integers.")
        decoder_divisor = 2 ** len(config.decoder_rates)
        if config.decoder_dim % decoder_divisor:
            raise ValueError("decoder_dim must divide evenly across decoder stages.")
        if config.latent_dim % config.latent_channels:
            raise ValueError("latent_dim must be divisible by latent_channels.")
        if config.latent_dim % config.num_attention_heads:
            raise ValueError("latent_dim must be divisible by num_attention_heads.")
        if config.sampling_rate != 32_000:
            raise ValueError("MiniMax H3 audio output requires a 32000 Hz rate.")
        if len(config.latents_mean) != config.latent_channels or len(
            config.latents_std
        ) != config.latent_channels:
            raise ValueError("Audio latent statistics must match latent_channels.")
        if not all(math.isfinite(value) for value in config.latents_mean):
            raise ValueError("Audio latent means must be finite.")
        if not all(
            math.isfinite(value) and value > 0 for value in config.latents_std
        ):
            raise ValueError("Audio latent standard deviations must be positive.")

    @property
    def device(self) -> torch.device:
        """Return the device of the first autoencoder parameter."""
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Return the autoencoder parameter dtype."""
        return next(self.parameters()).dtype

    def _require_fp32(self) -> None:
        if self.dtype != torch.float32:
            raise RuntimeError("MiniMax H3 audio VAE weights must remain float32.")

    @torch.no_grad()
    def encode(self, sample: Tensor) -> MiniMaxH3AudioEncoderOutput:
        """Encode mono-batch waveforms into the native posterior.

        Args:
            sample: Waveforms shaped ``[batch, 1, samples]``.

        Returns:
            Posterior whose mode is shaped ``[batch, channels, steps]``.

        Raises:
            ValueError: ``sample`` has invalid shape or values.
        """
        if (
            sample.ndim != 3
            or sample.shape[0] <= 0
            or sample.shape[1] != 1
            or sample.shape[2] <= 0
        ):
            raise ValueError("sample must have shape [batch, 1, positive_samples].")
        if not sample.is_floating_point() or not bool(torch.isfinite(sample).all()):
            raise ValueError("sample must contain finite floating-point values.")
        self._require_fp32()
        right_pad = (-sample.shape[-1]) % self.hop_length
        if right_pad:
            sample = F.pad(sample, (0, right_pad))
        hidden_states = self.encoder(sample.to(self.device, torch.float32))
        hidden_states = self.pre_block(hidden_states.transpose(1, 2)).transpose(1, 2)
        posterior = MiniMaxH3AudioDiagonalGaussianDistribution(
            self.mean_proj(hidden_states),
            self.logs_proj(hidden_states),
        )
        return MiniMaxH3AudioEncoderOutput(latent_dist=posterior)

    @torch.no_grad()
    def decode(self, latents: Tensor) -> Tensor:
        """Decode denormalized mono-batch latents into waveforms.

        Args:
            latents: Latents shaped ``[batch, channels, steps]``.

        Returns:
            Normalized waveforms shaped ``[batch, 1, steps * hop_length]``.

        Raises:
            ValueError: ``latents`` has invalid shape or values.
        """
        if (
            latents.ndim != 3
            or latents.shape[0] <= 0
            or latents.shape[1] != self.config.latent_channels
            or latents.shape[2] <= 0
        ):
            raise ValueError(
                "latents must have shape [batch, latent_channels, positive_steps]."
            )
        if not latents.is_floating_point() or not bool(
            torch.isfinite(latents).all()
        ):
            raise ValueError("latents must contain finite floating-point values.")
        self._require_fp32()
        decoded = self.decoder(
            self.dec_in_proj(latents.to(self.device, torch.float32))
        )
        return decoded.float()

    def denormalize(self, latents: Tensor) -> Tensor:
        """Apply the released per-channel audio latent statistics.

        Args:
            latents: Normalized latents shaped ``[batch, channels, steps]``.

        Returns:
            Denormalized latents with the same shape.
        """
        if (
            latents.ndim != 3
            or latents.shape[0] <= 0
            or latents.shape[1] != self.config.latent_channels
            or latents.shape[2] <= 0
        ):
            raise ValueError("Audio latents must have shape [batch, channels, steps].")
        if not latents.is_floating_point() or not bool(
            torch.isfinite(latents).all()
        ):
            raise ValueError("Audio latents must contain finite floating-point values.")
        mean = latents.new_tensor(self.config.latents_mean).view(1, -1, 1)
        std = latents.new_tensor(self.config.latents_std).view(1, -1, 1)
        return latents * std + mean

    @torch.no_grad()
    def encode_condition(self, samples: Tensor) -> Tensor:
        """Encode stereo reference audio into channel-major conditioning rows.

        Args:
            samples: Stereo waveform shaped ``[2, samples]``.

        Returns:
            Normalized CPU ``float32`` rows shaped ``[2 * steps, channels]``.
        """
        if samples.ndim != 2 or samples.shape[0] != 2:
            raise ValueError("Reference audio must have shape [2, samples].")
        posterior = self.encode(samples[:, None])
        latents = posterior.latent_dist.mode().float().cpu().transpose(1, 2)
        mean = latents.new_tensor(self.config.latents_mean).view(1, 1, -1)
        std = latents.new_tensor(self.config.latents_std).view(1, 1, -1)
        return ((latents - mean) / std).reshape(
            -1, self.config.latent_channels
        ).contiguous()

    @torch.no_grad()
    def decode_output(self, normalized_latents: Tensor) -> AudioOutput:
        """Decode H3 stereo latents into the V2 audio output contract.

        Args:
            normalized_latents: Latents shaped ``[2, channels, steps]``.

        Returns:
            Stereo 32 kHz samples with absolute offset zero.
        """
        if normalized_latents.ndim != 3 or normalized_latents.shape[0] != 2:
            raise ValueError(
                "Generated audio latents must have shape [2, channels, steps]."
            )
        mono_batch = self.decode(self.denormalize(normalized_latents))
        return AudioOutput(
            samples=mono_batch[:, 0].contiguous(),
            sample_rate=self.config.sampling_rate,
            sample_offset=0,
        )

    def forward(self, sample: Tensor, *, sample_posterior: bool = False) -> Tensor:
        """Encode and decode a mono-batch waveform.

        Args:
            sample: Waveforms shaped ``[batch, 1, samples]``.
            sample_posterior: Sample the posterior instead of taking its mode.

        Returns:
            Round-tripped normalized waveforms.
        """
        posterior = self.encode(sample).latent_dist
        latents = posterior.sample() if sample_posterior else posterior.mode()
        return self.decode(latents)


__all__ = [
    "H3_AUDIO_VAE_CHECKPOINT",
    "MiniMaxH3AudioDiagonalGaussianDistribution",
    "MiniMaxH3AudioEncoderOutput",
    "MiniMaxH3AudioVAE",
    "MiniMaxH3AudioVAEConfig",
]

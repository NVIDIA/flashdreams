# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User-facing config for the deterministic NULL model."""

from dataclasses import dataclass
from typing import ClassVar

from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler import FlowMatchSchedulerConfig
from flashdreams.infra.pipeline import StreamInferencePipelineConfig
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

from .decoder import NullDecoderConfig
from .encoder import NullInputEncoderConfig
from .transformer import NullTransformerConfig


@dataclass(kw_only=True)
class NullModelConfig(StreamInferencePipelineConfig):
    """Pipeline config that is externally visible."""

    output_layout: ClassVar[VideoTensorLayout] = VideoTensorLayout.bcthw
    """Layout of the emitted tensor.
    """


## Pipeline definition for our 'Null Model'

NULL_MODEL_CONFIG = NullModelConfig(
    name="null-model",
    encoder=NullInputEncoderConfig(),
    diffusion_model=DiffusionModelConfig(
        transformer=NullTransformerConfig(),
        scheduler=FlowMatchSchedulerConfig(
            num_inference_steps=1,
            denoising_timesteps=[1000],
        ),
    ),
    decoder=NullDecoderConfig(),
)

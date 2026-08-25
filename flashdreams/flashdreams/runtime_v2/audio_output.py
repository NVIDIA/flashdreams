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

"""Typed normalized PCM output for the v2 runtime."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, kw_only=True, slots=True)
class AudioOutput:
    """Normalized PCM samples generated alongside video."""

    samples: Tensor
    """Channel-major PCM samples with shape ``[channels, samples]`` in ``[-1, 1]``."""

    sample_rate: int
    """Samples per second."""

    sample_offset: int = 0
    """Zero-based sample position of the first sample in the session timeline."""

    def __post_init__(self) -> None:
        _validate_samples(self.samples)
        if (
            isinstance(self.sample_rate, bool)
            or not isinstance(self.sample_rate, int)
            or self.sample_rate <= 0
        ):
            raise ValueError("AudioOutput.sample_rate must be a positive integer.")
        if isinstance(self.sample_offset, bool) or not isinstance(
            self.sample_offset, int
        ):
            raise ValueError(
                "AudioOutput.sample_offset must be a non-negative integer."
            )
        if self.sample_offset < 0:
            raise ValueError(
                "AudioOutput.sample_offset must be a non-negative integer."
            )


def _validate_samples(samples: Tensor) -> None:
    if not isinstance(samples, Tensor):
        raise TypeError("AudioOutput.samples must be a torch.Tensor.")
    if samples.ndim != 2:
        raise ValueError(
            "AudioOutput.samples must have shape [channels, samples], got "
            f"{tuple(samples.shape)}."
        )
    if samples.shape[0] not in (1, 2):
        raise ValueError(
            "AudioOutput.samples must contain mono or stereo PCM, got "
            f"{samples.shape[0]} channels."
        )
    if samples.shape[1] <= 0:
        raise ValueError("AudioOutput.samples must contain at least one sample.")
    if not samples.is_floating_point():
        raise ValueError("AudioOutput.samples must use a floating-point dtype.")
    if not bool(torch.isfinite(samples).all()):
        raise ValueError("AudioOutput.samples must contain only finite values.")
    if bool((samples < -1.0).any()) or bool((samples > 1.0).any()):
        raise ValueError("AudioOutput.samples must stay within [-1, 1].")


def normalized_pcm(audio: AudioOutput) -> Tensor:
    """Return validated contiguous ``float32`` PCM on the CPU.

    Args:
        audio: Audio payload to validate and convert.

    Returns:
        Channel-major samples with shape ``[channels, samples]``.

    Raises:
        ValueError: Samples are non-floating, non-finite, or outside ``[-1, 1]``.
    """
    _validate_samples(audio.samples)
    return audio.samples.detach().to(device="cpu", dtype=torch.float32).contiguous()


__all__ = ["AudioOutput", "normalized_pcm"]

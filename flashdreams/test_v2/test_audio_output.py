# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the model-neutral v2 audio result contract."""

import pytest
import torch

from flashdreams.runtime_v2.audio_output import AudioOutput, normalized_pcm
from flashdreams.runtime_v2.session_desc import SessionDesc

pytestmark = pytest.mark.ci_cpu

_AUDIO_RATE = 8_000


@pytest.mark.parametrize(
    ("samples", "message"),
    [
        (torch.zeros(8), "shape"),
        (torch.zeros(3, 8), "mono or stereo"),
        (torch.zeros(2, 0), "at least one"),
    ],
)
def test_audio_output_rejects_malformed_shapes(
    samples: torch.Tensor, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        AudioOutput(samples=samples, sample_rate=_AUDIO_RATE)


def test_audio_output_requires_a_tensor() -> None:
    with pytest.raises(TypeError, match="torch.Tensor"):
        AudioOutput(samples=[[0.0]], sample_rate=_AUDIO_RATE)  # type: ignore[arg-type]


@pytest.mark.parametrize("sample_rate", [0, -1, True, 8_000.0])
def test_audio_output_rejects_invalid_sample_rates(sample_rate: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        AudioOutput(
            samples=torch.zeros(2, 8),
            sample_rate=sample_rate,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("sample_offset", [-1, True, 1.5])
def test_audio_output_rejects_invalid_offsets(sample_offset: object) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        AudioOutput(
            samples=torch.zeros(2, 8),
            sample_rate=_AUDIO_RATE,
            sample_offset=sample_offset,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "samples",
    [
        torch.tensor([[float("nan")]]),
        torch.tensor([[float("inf")]]),
        torch.tensor([[1.01]]),
        torch.ones(1, 1, dtype=torch.int16),
    ],
)
def test_audio_output_rejects_invalid_samples(samples: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        AudioOutput(samples=samples, sample_rate=_AUDIO_RATE)


def test_normalized_pcm_revalidates_mutated_samples() -> None:
    audio = AudioOutput(samples=torch.zeros(1, 2), sample_rate=_AUDIO_RATE)
    audio.samples[0, 0] = torch.nan

    with pytest.raises(ValueError, match="finite"):
        normalized_pcm(audio)


def test_normalized_pcm_returns_contiguous_cpu_float32() -> None:
    samples = torch.zeros(8, 2, dtype=torch.float16).transpose(0, 1)

    normalized = normalized_pcm(AudioOutput(samples=samples, sample_rate=_AUDIO_RATE))

    assert normalized.shape == (2, 8)
    assert normalized.device.type == "cpu"
    assert normalized.dtype is torch.float32
    assert normalized.is_contiguous()


def test_session_desc_accepts_paired_audio_contract() -> None:
    desc = SessionDesc(audio_sample_rate=_AUDIO_RATE, audio_channels=2)

    assert desc.audio_sample_rate == _AUDIO_RATE
    assert desc.audio_channels == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"audio_sample_rate": _AUDIO_RATE},
        {"audio_channels": 2},
        {"audio_sample_rate": 0, "audio_channels": 2},
        {"audio_sample_rate": 8_000.0, "audio_channels": 2},
        {"audio_sample_rate": _AUDIO_RATE, "audio_channels": 3},
        {"audio_sample_rate": _AUDIO_RATE, "audio_channels": True},
    ],
)
def test_session_desc_rejects_invalid_audio_contract(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        SessionDesc(**kwargs)  # type: ignore[arg-type]

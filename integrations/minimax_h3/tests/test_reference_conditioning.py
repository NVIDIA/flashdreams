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

"""CPU contracts for native MiniMax H3 REF2VA reference normalization."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from minimax_h3.reference_conditioning import (
    MiniMaxH3AudioReference,
    MiniMaxH3ImageReference,
    MiniMaxH3VideoReference,
    encode_references,
    normalize_references,
)

pytestmark = pytest.mark.ci_cpu


def _normalization_options() -> dict[str, int]:
    """Return tiny geometry options that preserve the released arithmetic."""
    return {
        "canvas_short_edge": 32,
        "canvas_max_pixels": 32 * 64,
        "reference_image_short_edge": 64,
        "multiple": 32,
    }


def test_normalize_references_preserves_order_and_maximum_alignment() -> None:
    """Keep semantic order and accept the 362-frame aligned 15-second request."""
    image = MiniMaxH3ImageReference(
        np.full((4, 8, 3), 0.5, dtype=np.float32)
    )
    audio = MiniMaxH3AudioReference(torch.arange(8, dtype=torch.float32)[None])
    video = MiniMaxH3VideoReference(
        frames=np.zeros((2, 4, 8, 3), dtype=np.uint8), fps=24.0
    )
    normalized = normalize_references(
        [image, audio, video], num_frames=362, **_normalization_options()
    )

    assert [reference.kind for reference in normalized] == [
        "image",
        "audio",
        "video",
    ]
    assert isinstance(normalized[0].image, Image.Image)
    assert normalized[0].image.size == (128, 64)
    torch.testing.assert_close(
        normalized[1].audio,
        torch.arange(8, dtype=torch.float32).repeat(2, 1),
    )
    assert normalized[1].sample_rate == 32000
    assert normalized[2].frames.shape == (2, 32, 64, 3)


def test_video_normalization_matches_whole_frame_resampling() -> None:
    """Duplicate 12 fps frames into the exact released 24 fps slot mapping."""
    frames = np.stack(
        [np.full((32, 32, 3), value, dtype=np.uint8) for value in (1, 2, 3)]
    )
    normalized = normalize_references(
        [MiniMaxH3VideoReference(frames=frames, fps=12.0)],
        num_frames=124,
        **_normalization_options(),
    )[0]

    assert normalized.fps == 24
    assert normalized.frames[:, 0, 0, 0].tolist() == [1, 1, 2, 2, 3, 3]
    assert normalized.frames.flags.c_contiguous


def test_audio_normalization_truncates_and_upmixes_without_resampler() -> None:
    """Use CPU float32 stereo at 32 kHz and require adapters to resample."""
    mono = torch.arange(200_000, dtype=torch.float32)[None]
    normalized = normalize_references(
        [
            MiniMaxH3ImageReference(np.zeros((32, 32, 3), dtype=np.uint8)),
            MiniMaxH3AudioReference(mono, sample_rate=32000),
        ],
        num_frames=124,
        **_normalization_options(),
    )[1]

    assert normalized.audio.dtype == torch.float32
    assert normalized.audio.device.type == "cpu"
    assert normalized.audio.shape == (2, 165_333)
    torch.testing.assert_close(normalized.audio[0], normalized.audio[1])
    with pytest.raises(ValueError, match="decoded at 32000 Hz"):
        normalize_references(
            [
                MiniMaxH3ImageReference(
                    np.zeros((32, 32, 3), dtype=np.uint8)
                ),
                MiniMaxH3AudioReference(torch.zeros(1, 4), sample_rate=16000),
            ],
            num_frames=124,
            **_normalization_options(),
        )


def test_reference_validation_rejects_invalid_limits_shapes_and_rates() -> None:
    """Fail before model allocation on unsupported REF2VA request contracts."""
    with pytest.raises(ValueError, match="paired with an image or video"):
        normalize_references(
            [MiniMaxH3AudioReference(torch.zeros(1, 4))],
            num_frames=124,
            **_normalization_options(),
        )
    with pytest.raises(ValueError, match="at most 3 video"):
        normalize_references(
            [
                MiniMaxH3VideoReference(
                    np.zeros((1, 32, 32, 3), dtype=np.uint8)
                )
                for _ in range(4)
            ],
            num_frames=124,
            **_normalization_options(),
        )
    with pytest.raises(ValueError, match="positive frame rate"):
        normalize_references(
            [
                MiniMaxH3VideoReference(
                    np.zeros((1, 32, 32, 3), dtype=np.uint8), fps=-1.0
                )
            ],
            num_frames=124,
            **_normalization_options(),
        )
    with pytest.raises(ValueError, match="aligned to the H3 frame grid"):
        normalize_references(
            [
                MiniMaxH3ImageReference(
                    np.zeros((32, 32, 3), dtype=np.uint8)
                )
            ],
            num_frames=125,
            **_normalization_options(),
        )


class _VideoVAE:
    """Record normalized video pixels and return shape-derived H3 latents."""

    def __init__(self) -> None:
        self.inputs: list[torch.Tensor] = []

    def encode_condition_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        """Return one latent frame per supplied pixel frame."""
        self.inputs.append(pixels.clone())
        return torch.full(
            (1, 24, pixels.shape[2], 2, 2),
            float(len(self.inputs)),
            dtype=torch.float32,
        )


class _AudioVAE:
    """Record stereo waveforms and return channel-major latent rows."""

    def __init__(self) -> None:
        self.inputs: list[torch.Tensor] = []

    def encode_condition(self, samples: torch.Tensor) -> torch.Tensor:
        """Return two stereo steps with a call-specific value."""
        self.inputs.append(samples.clone())
        return torch.full(
            (4, 32), float(len(self.inputs)), dtype=torch.float32
        )


def test_encode_references_filters_modalities_in_semantic_order() -> None:
    """Encode visual and audio-bearing subsequences without reordering either."""
    video_vae = _VideoVAE()
    audio_vae = _AudioVAE()
    references = [
        MiniMaxH3ImageReference(Image.new("RGB", (32, 32), color=(255, 0, 0))),
        MiniMaxH3VideoReference(
            frames=np.full((25, 32, 32, 3), 64, dtype=np.uint8),
            fps=24.0,
            audio=torch.full((2, 8), 0.25),
            sample_rate=32000,
        ),
        MiniMaxH3AudioReference(torch.full((2, 6), 0.5), sample_rate=32000),
    ]

    encoded = encode_references(
        video_vae, audio_vae, references, device="cpu"
    )

    assert [tuple(value.shape) for value in encoded.video] == [
        (1, 24, 1, 2, 2),
        (1, 24, 22, 2, 2),
    ]
    assert [tuple(value.shape) for value in encoded.audio] == [(4, 32), (4, 32)]
    assert bool((encoded.video[0] == 1).all())
    assert bool((encoded.video[1] == 2).all())
    assert bool((encoded.audio[0] == 1).all())
    assert bool((encoded.audio[1] == 2).all())
    assert video_vae.inputs[0].shape == (1, 3, 1, 32, 32)
    assert video_vae.inputs[1].shape == (1, 3, 22, 32, 32)
    assert float(video_vae.inputs[0][0, 0, 0, 0, 0]) == 1.0
    assert float(video_vae.inputs[1][0, 0, 0, 0, 0]) == pytest.approx(64 / 255)
    torch.testing.assert_close(audio_vae.inputs[0], references[1].audio)
    torch.testing.assert_close(audio_vae.inputs[1], references[2].audio)


def test_encode_references_requires_normalized_media() -> None:
    """Reject raw array images before invoking either native VAE."""
    with pytest.raises(ValueError, match="image references must be normalized"):
        encode_references(
            _VideoVAE(),
            _AudioVAE(),
            [
                MiniMaxH3ImageReference(
                    np.zeros((32, 32, 3), dtype=np.uint8)
                )
            ],
            device="cpu",
        )

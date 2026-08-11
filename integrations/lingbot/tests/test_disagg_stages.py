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

"""CPU tests for LingBot disaggregation payload boundaries."""

from __future__ import annotations

import pytest
import torch
from einops import rearrange
from lingbot.disagg.stages import (
    LingbotConditioning,
    conditioning_from_bundle,
    conditioning_to_bundle,
    encoder_output_from_bundle,
    encoder_output_to_bundle,
    encoder_output_to_cp_bundles,
)
from lingbot.encoder.camctrl import I2VCamCtrlEmbeddings

from flashdreams.recipes.wan.autoencoder.i2v import I2VCtrl

pytestmark = pytest.mark.ci_cpu


def test_conditioning_bundle_preserves_optional_fields_and_spatial_metadata() -> None:
    conditioning = LingbotConditioning(
        height=58,
        width=104,
        text_embeddings=torch.randn(1, 4, 8),
        negative_text_embeddings=None,
        image_embeddings=torch.randn(1, 2, 3),
    )
    bundle = conditioning_to_bundle(conditioning)
    restored = conditioning_from_bundle(bundle, height=58, width=104)

    assert tuple(bundle) == ("text_embeddings", "image_embeddings")
    assert restored.height == 58
    assert restored.width == 104
    assert restored.negative_text_embeddings is None
    torch.testing.assert_close(
        restored.text_embeddings,
        conditioning.text_embeddings,
    )


def test_encoder_output_round_trips_without_crossing_patchify_boundary() -> None:
    output = I2VCamCtrlEmbeddings(
        i2v=I2VCtrl(
            latent=torch.randn(3, 16, 2, 2),
            mask=torch.ones(3, 16, 2, 2),
        ),
        plucker=torch.randn(3, 384, 2, 2),
    )
    restored = encoder_output_from_bundle(encoder_output_to_bundle(output))

    assert not restored._is_patchified
    assert not restored.i2v._is_patchified
    torch.testing.assert_close(restored.i2v.latent, output.i2v.latent)
    torch.testing.assert_close(restored.i2v.mask, output.i2v.mask)
    torch.testing.assert_close(restored.plucker, output.plucker)


def test_encoder_output_direct_cp_shards_match_global_patchify() -> None:
    output = I2VCamCtrlEmbeddings(
        i2v=I2VCtrl(
            latent=torch.arange(4 * 2 * 4 * 6).reshape(4, 2, 4, 6),
            mask=torch.ones(4, 2, 4, 6),
        ),
        plucker=torch.arange(4 * 3 * 4 * 6).reshape(4, 3, 4, 6),
    )

    shards = encoder_output_to_cp_bundles(
        output,
        cp_size=3,
        patch_size=(2, 2, 2),
    )
    restored = [
        encoder_output_from_bundle(bundle, patchified=True) for bundle in shards
    ]

    assert len(restored) == 3
    assert all(item._is_patchified and item.i2v._is_patchified for item in restored)
    assert all(item.i2v.latent.shape == (4, 16) for item in restored)
    expected = rearrange(
        output.i2v.latent,
        "... (t kt) c (h kh) (w kw) -> ... (t h w) (c kt kh kw)",
        kt=2,
        kh=2,
        kw=2,
    )
    torch.testing.assert_close(
        torch.cat([item.i2v.latent for item in restored], dim=-2),
        expected,
    )


def test_encoder_output_direct_cp_shards_reject_uneven_tokens() -> None:
    output = I2VCamCtrlEmbeddings(
        i2v=I2VCtrl(
            latent=torch.randn(2, 2, 2, 2),
            mask=torch.ones(2, 2, 2, 2),
        ),
        plucker=torch.randn(2, 3, 2, 2),
    )

    with pytest.raises(ValueError, match="not divisible by CP3"):
        encoder_output_to_cp_bundles(
            output,
            cp_size=3,
            patch_size=(1, 1, 1),
        )

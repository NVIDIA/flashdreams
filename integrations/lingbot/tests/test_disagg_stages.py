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
from lingbot.disagg.stages import (
    LingbotConditioning,
    conditioning_from_bundle,
    conditioning_to_bundle,
    encoder_output_from_bundle,
    encoder_output_to_bundle,
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

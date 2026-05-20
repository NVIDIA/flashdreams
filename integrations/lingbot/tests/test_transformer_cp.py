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

"""Context-parallel patchify smoke test for the Lingbot World transformer."""

import pytest
import torch
from lingbot.encoder.camctrl import I2VCamCtrlEmbeddings
from lingbot.transformer import (
    LingbotWorldTransformer,
    LingbotWorldTransformerConfig,
)
from lingbot.transformer.impl.network import (
    LingbotWorldDiTNetworkConfig,
)

from flashdreams.recipes.wan.autoencoder.i2v import I2VCtrl

pytestmark = pytest.mark.ci_cpu


def test_lingbot_patchify_marks_i2v_and_plucker_as_patchified() -> None:
    transformer = LingbotWorldTransformer(
        LingbotWorldTransformerConfig(
            network=LingbotWorldDiTNetworkConfig(
                dim=64,
                ffn_dim=128,
                num_heads=4,
                num_layers=1,
                patch_embedding_type="linear",
                control_type="cam",
            ),
            batch_shape=(),
            len_t=2,
            window_size_t=2,
            sink_size_t=0,
            compile_network=False,
        )
    )

    camctrl_embeddings = I2VCamCtrlEmbeddings(
        i2v=I2VCtrl(
            latent=torch.randn(2, 16, 4, 4),
            mask=torch.randn(2, 16, 4, 4),
        ),
        plucker=torch.randn(2, 6 * 64, 4, 4),
    )

    patched = transformer.patchify_and_maybe_split_cp(camctrl_embeddings)
    assert isinstance(patched, I2VCamCtrlEmbeddings)
    assert patched._is_patchified
    assert patched.i2v._is_patchified
    assert patched.i2v.latent.shape == (8, 64)
    assert patched.i2v.mask.shape == (8, 64)
    assert patched.plucker.shape == (8, 1536)

    # Idempotent once marked patchified.
    assert transformer.patchify_and_maybe_split_cp(patched) is patched

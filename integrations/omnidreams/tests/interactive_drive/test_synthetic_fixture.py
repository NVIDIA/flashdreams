# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import torch
from omnidreams.interactive_drive.world_model.synthetic_fixture import (
    build_synthetic_world_model_assets,
)
from omnidreams.vae_native import OmnidreamsWanVAEEncoderConfig

from flashdreams.core.checkpoint import load as checkpoint_load
from flashdreams.recipes.taehv import TeahvVAEDecoderConfig


def _has_meta_state(module: torch.nn.Module) -> bool:
    return any(t.is_meta for t in module.parameters()) or any(
        t.is_meta for t in module.buffers()
    )


def test_synthetic_fixture_round_trips_offline(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    def fail_hf_download(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("synthetic fixture must not download from Hugging Face")

    monkeypatch.setattr(checkpoint_load, "hf_hub_download", fail_hf_download)

    encoder_cfg = OmnidreamsWanVAEEncoderConfig(
        dtype=torch.float32,
        use_compile=False,
        use_cuda_graph=False,
        native_vae_acceleration="disabled",
    )
    decoder_cfg = TeahvVAEDecoderConfig(
        dtype=torch.float32,
        use_compile=False,
        use_cuda_graph=False,
        state_dict_transform=None,
    )

    assets = build_synthetic_world_model_assets(
        tmp_path,
        encoder_cfg=encoder_cfg,
        decoder_cfg=decoder_cfg,
    )

    assert assets.encoder_checkpoint_path is not None
    assert assets.encoder_checkpoint_path.is_file()
    assert assets.decoder_checkpoint_path.is_file()

    encoder = OmnidreamsWanVAEEncoderConfig(
        checkpoint_path=str(assets.encoder_checkpoint_path),
        dtype=torch.float32,
        use_compile=False,
        use_cuda_graph=False,
        native_vae_acceleration="disabled",
    ).setup()
    decoder = TeahvVAEDecoderConfig(
        checkpoint_path=str(assets.decoder_checkpoint_path),
        dtype=torch.float32,
        use_compile=False,
        use_cuda_graph=False,
        state_dict_transform=None,
    ).setup()

    assert not _has_meta_state(encoder)
    assert not _has_meta_state(decoder)

    video = torch.zeros((1, 1, 5, 3, 32, 32), dtype=torch.float32)
    latent = encoder(video)
    assert latent.shape[:4] == (1, 1, 2, 16)

    decoded = decoder(torch.zeros((1, 1, 2, 16, 4, 4), dtype=torch.float32))
    assert decoded.shape[:4] == (1, 1, 5, 3)

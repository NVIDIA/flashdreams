# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned, component-scoped MiniMax H3 checkpoint loading."""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from flashdreams.core.checkpoint.load import load_checkpoint


@dataclass(frozen=True)
class MiniMaxH3Weights:
    """One repository revision shared by codecs, conditioner, and transformer."""

    model_id: str
    revision: str

    def asset(self, filename: str) -> Path:
        """Resolve one small metadata asset, without fetching model weights."""
        root = Path(self.model_id).expanduser()
        if root.is_dir():
            path = root / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            return path
        return Path(hf_hub_download(self.model_id, filename, revision=self.revision))

    def config(self, component: str) -> dict:
        """Read component geometry from the pinned checkpoint."""
        with self.asset(f"{component}/config.json").open() as handle:
            return json.load(handle)

    def checkpoint(self, component: str, *, sharded: bool = True) -> str:
        """Return the native loader's index or safetensors locator."""
        filename = "diffusion_pytorch_model.safetensors" + (
            ".index.json" if sharded else ""
        )
        root = Path(self.model_id).expanduser()
        if root.is_dir():
            return str(root / component / filename)
        return f"https://huggingface.co/{self.model_id}/blob/{self.revision}/{component}/{filename}"

    def video_encoder(self, device: torch.device):
        """Load only the video encoder and its posterior projection."""
        from minimax_h3.impl.video_vae import MiniMaxH3VideoEncoder, VideoVAEConfig

        config = VideoVAEConfig.from_dict(self.config("vae"))
        with torch.device(device):
            model = MiniMaxH3VideoEncoder(config)
        return load_checkpoint(
            self.checkpoint("vae"),
            model=model.float(),
            include_prefixes=("encoder.", "quant_conv."),
        ).eval()

    def video_decoder(self, device: torch.device):
        """Load only the video decoder and its input projection."""
        from minimax_h3.impl.video_vae import MiniMaxH3VideoDecoder, VideoVAEConfig

        config = VideoVAEConfig.from_dict(self.config("vae"))
        with torch.device(device):
            model = MiniMaxH3VideoDecoder(config)
        return load_checkpoint(
            self.checkpoint("vae"),
            model=model.float(),
            include_prefixes=("decoder.", "post_quant_conv."),
        ).eval()

    def audio_encoder(self, device: torch.device):
        """Load only the reference-waveform encoder and posterior mean."""
        from minimax_h3.impl.audio_encoder import (
            AudioEncoderConfig,
            MiniMaxH3AudioEncoder,
        )

        config = AudioEncoderConfig.from_dict(self.config("audio_vae"))
        with torch.device(device):
            model = MiniMaxH3AudioEncoder(config)
        return load_checkpoint(
            self.checkpoint("audio_vae", sharded=False),
            model=model.float(),
            include_prefixes=("encoder.", "pre_block.", "mean_proj."),
        ).eval()

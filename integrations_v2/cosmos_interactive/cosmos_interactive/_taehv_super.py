# SPDX-FileCopyrightText: Copyright (c) 2025 Ollin Boer Bohan
#
# NVIDIA modifications:
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND MIT

"""TAEHV ``*_super`` streaming decoder for Cosmos Interactive."""

from __future__ import annotations

import hashlib
import os
from urllib.parse import unquote, urlparse
from urllib.request import urlretrieve

import torch
import torch.nn as nn
from loguru import logger

from flashdreams.infra.cuda_graph import set_or_copy
from flashdreams.recipes.taehv.checkpoint import StateDictTransform
from flashdreams.recipes.taehv.impl import TAEHV, TGrow

SUPER_CHANNELS = (512, 256, 128, 64)


def _conv(n_in: int, n_out: int, **kwargs) -> nn.Conv2d:
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


def _default_cache_dir() -> str:
    return os.path.expanduser(
        os.getenv("FLASHDREAMS_CACHE_DIR", "~/.cache/flashdreams")
    )


def _is_http_url(path: str) -> bool:
    return path.startswith(("http://", "https://"))


def _url_checkpoint_cache_path(url: str, local_cache_dir: str | None = None) -> str:
    parsed = urlparse(url)
    filename = os.path.basename(unquote(parsed.path)) or "checkpoint"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return os.path.join(
        local_cache_dir or _default_cache_dir(),
        "cosmos_interactive",
        "url_checkpoints",
        f"{digest}_{filename}",
    )


def cache_url_checkpoint(url: str, local_cache_dir: str | None = None) -> str:
    """Download an HTTP checkpoint into the Cosmos cache."""
    cache_path = _url_checkpoint_cache_path(url, local_cache_dir)
    if os.path.exists(cache_path):
        logger.info(f"Loading checkpoint URL from local cache: {cache_path}")
        return cache_path

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp_path = f"{cache_path}.{os.getpid()}.tmp"
    logger.info(f"Downloading checkpoint from URL: {url}")
    try:
        urlretrieve(url, tmp_path)
        os.replace(tmp_path, cache_path)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise
    logger.info(f"Checkpoint downloaded to local cache: {cache_path}")
    return cache_path


def resolve_super_checkpoint(
    checkpoint_path: str | None,
    *,
    local_cache_dir: str | None = None,
) -> str | None:
    if checkpoint_path is None or not _is_http_url(checkpoint_path):
        return checkpoint_path
    return cache_url_checkpoint(checkpoint_path, local_cache_dir=local_cache_dir)


class SuperMemBlock(nn.Module):
    """ConvNeXt-style memory block used by upstream TAEHV ``*_super`` decoders."""

    def __init__(self, n_f: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_f * 2, n_f * 2, 7, padding=3, groups=n_f * 2, bias=False),
            nn.Conv2d(n_f * 2, n_f * 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_f * 4, n_f, 1, bias=False),
        )

    def forward(self, x: torch.Tensor, past: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat([x, past], 1)) + x

    def cache_step(
        self,
        x: torch.Tensor,
        state: dict[int, torch.Tensor],
        batch: int,
    ) -> torch.Tensor:
        key = id(self)
        bt, c, h, w = x.shape
        t = bt // batch
        x5 = x.view(batch, t, c, h, w)
        prev = state[key]
        past = torch.cat([prev, x5[:, :-1]], dim=1)
        out = self.forward(x, past.reshape(bt, c, h, w))
        set_or_copy(state, key, x5[:, -1:])
        return out


class SuperDecoder(nn.Module):
    """Streaming decoder body matching madebyollin/taehv's Wan 2.2 super layout."""

    def __init__(
        self,
        n_f: tuple[int, int, int, int],
        latent_channels: int,
        image_channels: int,
        patch_size: int,
        decoder_time_upscale: tuple[bool, bool],
        decoder_space_upscale: tuple[bool, bool, bool],
        act_func: nn.Module,
    ):
        super().__init__()
        # Layer indices match upstream's ``decoder.<idx>`` keys after the
        # public TAEHV loader rewrites them to ``decoder.blocks.<idx>``.
        self.blocks = nn.Sequential(
            nn.Conv2d(latent_channels, n_f[0], 1, bias=False),
            SuperMemBlock(n_f[0]),
            SuperMemBlock(n_f[0]),
            SuperMemBlock(n_f[0]),
            _conv(n_f[0], n_f[1] * (2 if decoder_space_upscale[0] else 1) ** 2),
            act_func,
            nn.PixelShuffle(2 if decoder_space_upscale[0] else 1),
            TGrow(n_f[1], 1),
            SuperMemBlock(n_f[1]),
            SuperMemBlock(n_f[1]),
            SuperMemBlock(n_f[1]),
            _conv(n_f[1], n_f[2] * (2 if decoder_space_upscale[1] else 1) ** 2),
            act_func,
            nn.PixelShuffle(2 if decoder_space_upscale[1] else 1),
            TGrow(n_f[2], 2 if decoder_time_upscale[0] else 1),
            SuperMemBlock(n_f[2]),
            SuperMemBlock(n_f[2]),
            SuperMemBlock(n_f[2]),
            _conv(n_f[2], n_f[3] * (2 if decoder_space_upscale[2] else 1) ** 2),
            act_func,
            nn.PixelShuffle(2 if decoder_space_upscale[2] else 1),
            TGrow(n_f[3], 2 if decoder_time_upscale[1] else 1),
            _conv(n_f[3], image_channels * patch_size**2),
        )

    def forward(
        self,
        z: torch.Tensor,
        state: dict[int, torch.Tensor],
        batch: int,
    ) -> torch.Tensor:
        b, t, c, h, w = z.shape
        x = z.reshape(b * t, c, h, w)
        for blk in self.blocks:
            if isinstance(blk, SuperMemBlock):
                x = blk.cache_step(x, state, batch)
            else:
                x = blk(x)
        bt, c_out, h_out, w_out = x.shape
        return x.reshape(b, bt // b, c_out, h_out, w_out)

    @torch.no_grad()
    def initialize_state(
        self,
        z_shape: tuple[int, int, int, int, int],
        dtype: torch.dtype,
        device: torch.device,
        state: dict[int, torch.Tensor],
    ) -> None:
        batch, t, c_z, h_z, w_z = z_shape
        x = torch.zeros(batch * t, c_z, h_z, w_z, dtype=dtype, device=device)
        for blk in self.blocks:
            if isinstance(blk, SuperMemBlock):
                bt_x, c_x, h_x, w_x = x.shape
                t_x = bt_x // batch
                state[id(blk)] = torch.zeros(
                    batch, 1, c_x, h_x, w_x, dtype=dtype, device=device
                )
                past = (
                    state[id(blk)]
                    .expand(-1, t_x, -1, -1, -1)
                    .reshape(bt_x, c_x, h_x, w_x)
                )
                x = blk.forward(x, past)
            else:
                x = blk(x)


class SuperTAEHV(TAEHV):
    """Streaming wrapper that loads upstream TAEHV ``*_super`` checkpoints."""

    decoder: SuperDecoder

    def __init__(
        self,
        checkpoint_path: str | None,
        *,
        decoder_time_upscale: tuple[bool, bool] = (True, True),
        decoder_space_upscale: tuple[bool, bool, bool] = (True, True, True),
        clamp_output: bool = True,
        use_cuda_graph: bool = True,
        use_compile: bool = False,
        warmup_iters: int = 2,
        state_dict_transform: StateDictTransform | None = None,
        local_cache_dir: str | None = None,
    ):
        super().__init__(
            checkpoint_path=None,
            decoder_time_upscale=decoder_time_upscale,
            decoder_space_upscale=decoder_space_upscale,
            channels=SUPER_CHANNELS,
            clamp_output=clamp_output,
            model_type="wan22",
            use_cuda_graph=use_cuda_graph,
            use_compile=use_compile,
            warmup_iters=warmup_iters,
        )
        self.arch_variant = "super"
        act_func = nn.ReLU(inplace=True)
        with torch.device("meta"):
            self.decoder = SuperDecoder(
                n_f=SUPER_CHANNELS,
                latent_channels=self.latent_channels,
                image_channels=self.image_channels,
                patch_size=self.patch_size,
                decoder_time_upscale=decoder_time_upscale,
                decoder_space_upscale=decoder_space_upscale,
                act_func=act_func,
            )

        local_checkpoint = resolve_super_checkpoint(
            checkpoint_path,
            local_cache_dir=local_cache_dir,
        )
        if local_checkpoint is not None:
            self.load_from_checkpoint(
                local_checkpoint,
                state_dict_transform=state_dict_transform,
            )

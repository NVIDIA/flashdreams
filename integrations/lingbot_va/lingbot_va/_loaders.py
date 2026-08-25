# SPDX-FileCopyrightText: Copyright 2024-2025 The Robbyant Team Authors
# SPDX-FileCopyrightText: Copyright (c) 2026 Hongyu Zhou
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

"""Checkpoint resolution and model loading for LingBot-VA."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn

from flashdreams.core.io.hf import maybe_download_hf_repo_on_rank0

_CHECKPOINT_COMPONENTS = ("transformer", "vae", "text_encoder", "tokenizer")
"""Subdirectories required from a LingBot-VA checkpoint snapshot."""


def validate_checkpoint_root(path: str | Path) -> Path:
    """Validate and return a local LingBot-VA checkpoint root.

    Args:
        path: Local snapshot directory.

    Returns:
        Expanded checkpoint directory.

    Raises:
        FileNotFoundError: The root or a required component is missing.
        NotADirectoryError: The root is not a directory.
    """
    root = Path(path).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"LingBot-VA checkpoint root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(
            f"LingBot-VA checkpoint root is not a directory: {root}"
        )
    missing = [
        component
        for component in _CHECKPOINT_COMPONENTS
        if not (root / component).is_dir()
    ]
    if missing:
        raise FileNotFoundError(
            f"LingBot-VA checkpoint root {root} is missing component "
            + ", ".join(repr(component) for component in missing)
            + "."
        )
    return root


def resolve_checkpoint_root(
    checkpoint_root: str | Path,
    *,
    revision: str | None = None,
) -> Path:
    """Resolve a local root or revision-pinned Hugging Face repo to a snapshot.

    Existing paths are always local. Nonexistent absolute, tilde-prefixed, and
    dot-prefixed paths fail locally; other strings are treated as Hugging Face
    repository IDs.

    Args:
        checkpoint_root: Local directory or Hugging Face repository ID.
        revision: Optional Hugging Face revision.

    Returns:
        Validated local snapshot directory.
    """
    value = str(checkpoint_root)
    expanded = Path(value).expanduser()
    if expanded.exists():
        return validate_checkpoint_root(expanded)
    if (
        isinstance(checkpoint_root, Path)
        or expanded.is_absolute()
        or value.startswith((".", "~"))
    ):
        raise FileNotFoundError(
            f"LingBot-VA checkpoint root does not exist: {expanded}"
        )

    allow_patterns = [
        component
        for name in _CHECKPOINT_COMPONENTS
        for component in (name, f"{name}/*", f"{name}/**")
    ]
    maybe_download_hf_repo_on_rank0(
        value,
        revision=revision,
        allow_patterns=allow_patterns,
    )
    from huggingface_hub import snapshot_download

    snapshot_kwargs: dict[str, Any] = {
        "repo_id": value,
        "allow_patterns": allow_patterns,
        "local_files_only": True,
    }
    if revision is not None:
        snapshot_kwargs["revision"] = revision
    local_root = snapshot_download(**snapshot_kwargs)
    return validate_checkpoint_root(local_root)


def load_vae(
    checkpoint_root: Path,
    torch_dtype: torch.dtype,
    torch_device: torch.device | str,
) -> nn.Module:
    """Load the Wan VAE from a resolved snapshot."""
    from diffusers import AutoencoderKLWan

    vae = cast(
        nn.Module,
        AutoencoderKLWan.from_pretrained(
            checkpoint_root,
            subfolder="vae",
            torch_dtype=torch_dtype,
            local_files_only=True,
        ),
    )
    return vae.to(torch_device)


def load_text_encoder(
    checkpoint_root: Path,
    torch_dtype: torch.dtype,
    torch_device: torch.device | str,
) -> nn.Module:
    """Load the UMT5 text encoder from a resolved snapshot."""
    from transformers import UMT5EncoderModel

    text_encoder = cast(
        nn.Module,
        UMT5EncoderModel.from_pretrained(
            checkpoint_root,
            subfolder="text_encoder",
            torch_dtype=torch_dtype,
            local_files_only=True,
        ),
    )
    return text_encoder.to(torch_device)


def load_tokenizer(checkpoint_root: Path) -> Any:
    """Load the T5 tokenizer from a resolved snapshot."""
    tokenizer_class = getattr(
        importlib.import_module("transformers"),
        "T5TokenizerFast",
    )

    return tokenizer_class.from_pretrained(
        checkpoint_root,
        subfolder="tokenizer",
        local_files_only=True,
    )


def patchify(x: Tensor, patch_size: int | None) -> Tensor:
    """Fold a spatial VAE patch into the channel dimension."""
    if patch_size is None or patch_size == 1:
        return x
    batch_size, channels, frames, height, width = x.shape
    x = x.view(
        batch_size,
        channels,
        frames,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
    )
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    return x.view(
        batch_size,
        channels * patch_size * patch_size,
        frames,
        height // patch_size,
        width // patch_size,
    )


class WanVAEStreamingWrapper:
    """Keep independent causal encoder state around a shared Wan VAE."""

    def __init__(self, vae_model: Any) -> None:
        """
        Args:
            vae_model: Wan VAE whose encoder and quantization projection are used.
        """
        self.vae = vae_model
        self.encoder = vae_model.encoder
        self.quant_conv = vae_model.quant_conv

        if hasattr(self.vae, "_cached_conv_counts"):
            self.enc_conv_num = self.vae._cached_conv_counts["encoder"]
        else:
            self.enc_conv_num = sum(
                module.__class__.__name__ == "WanCausalConv3d"
                for module in self.encoder.modules()
            )
        self.clear_cache()

    def clear_cache(self) -> None:
        """Discard causal encoder features from the previous observation."""
        self.feat_cache: list[Tensor | None] = [None] * self.enc_conv_num

    def encode_chunk(self, x_chunk: Tensor) -> Tensor:
        """Encode one observation chunk while advancing causal feature state."""
        patch_size = getattr(self.vae.config, "patch_size", None)
        if patch_size is not None:
            x_chunk = patchify(x_chunk, patch_size)
        feat_idx = [0]
        output = self.encoder(
            x_chunk,
            feat_cache=self.feat_cache,
            feat_idx=feat_idx,
        )
        return self.quant_conv(output)


__all__ = [
    "WanVAEStreamingWrapper",
    "load_text_encoder",
    "load_tokenizer",
    "load_vae",
    "resolve_checkpoint_root",
    "validate_checkpoint_root",
]

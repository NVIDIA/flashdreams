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

"""Compare native LingBot-VA video/action flows with pinned upstream code.

The harness runs the same deterministic first-chunk tensors through the
upstream ``WanTransformer3DModel`` and the FlashDreams-native transformer. It
loads the models sequentially so the comparison also works on GPUs that cannot
hold two copies of the five-billion-parameter network.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import importlib.machinery
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from einops import rearrange
from torch import Tensor

from lingbot_va.constants import (
    ROBOTWIN_ACTION_DIM,
    ROBOTWIN_ACTION_PER_FRAME,
    ROBOTWIN_ATTENTION_WINDOW,
    ROBOTWIN_FRAME_CHUNK_SIZE,
    ROBOTWIN_LATENT_CHANNELS,
    ROBOTWIN_LATENT_HEIGHT,
    ROBOTWIN_LATENT_TOKEN_PER_CHUNK,
    ROBOTWIN_LATENT_WIDTH,
    ROBOTWIN_ACTION_TOKEN_PER_CHUNK,
)
from lingbot_va.transformer import (
    LingbotVATransformer,
    LingbotVATransformerConfig,
)
from lingbot_va.utils import get_mesh_id


@dataclass(frozen=True, slots=True)
class _Fixture:
    """Deterministic CPU tensors shared by both implementations."""

    video: Tensor
    action: Tensor
    text: Tensor
    video_timesteps: Tensor
    action_timesteps: Tensor
    video_grid: Tensor
    action_grid: Tensor


@dataclass(frozen=True, slots=True)
class _Difference:
    """Absolute-error summary for one matched output."""

    maximum: float
    mean: float
    root_mean_square: float


def _parser() -> argparse.ArgumentParser:
    """Build the manual parity command parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        required=True,
        help="Resolved checkpoint snapshot containing transformer/.",
    )
    parser.add_argument(
        "--upstream-root",
        type=Path,
        required=True,
        help="Pinned robbyant/lingbot-va checkout root.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--compile-native",
        action="store_true",
        help="Compile the native video/action block loops before comparison.",
    )
    parser.add_argument(
        "--maximum-video-error",
        type=float,
        help="Fail when video flow maximum absolute error exceeds this value.",
    )
    parser.add_argument(
        "--mean-video-error",
        type=float,
        help="Fail when video flow mean absolute error exceeds this value.",
    )
    parser.add_argument(
        "--maximum-action-error",
        type=float,
        help="Fail when action flow maximum absolute error exceeds this value.",
    )
    parser.add_argument(
        "--mean-action-error",
        type=float,
        help="Fail when action flow mean absolute error exceeds this value.",
    )
    return parser


def _fixture(seed: int) -> _Fixture:
    """Create one realistic first-chunk input without loading model state."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    video = torch.randn(
        1,
        ROBOTWIN_LATENT_CHANNELS,
        ROBOTWIN_FRAME_CHUNK_SIZE,
        ROBOTWIN_LATENT_HEIGHT,
        ROBOTWIN_LATENT_WIDTH,
        generator=generator,
    ).to(torch.bfloat16)
    action = torch.randn(
        1,
        ROBOTWIN_ACTION_DIM,
        ROBOTWIN_FRAME_CHUNK_SIZE,
        ROBOTWIN_ACTION_PER_FRAME,
        1,
        generator=generator,
    ).to(torch.bfloat16)
    text = torch.randn(1, 512, 4096, generator=generator).to(torch.bfloat16)
    video_timesteps = torch.tensor([[0.0, 500.0]], dtype=torch.float32)
    action_timesteps = torch.tensor([[0.0, 500.0]], dtype=torch.float32)
    video_grid = get_mesh_id(
        ROBOTWIN_FRAME_CHUNK_SIZE,
        ROBOTWIN_LATENT_HEIGHT // 2,
        ROBOTWIN_LATENT_WIDTH // 2,
        0,
        1,
        0,
    )
    action_grid = get_mesh_id(
        ROBOTWIN_FRAME_CHUNK_SIZE,
        ROBOTWIN_ACTION_PER_FRAME,
        1,
        1,
        1,
        0,
        action=True,
    )
    return _Fixture(
        video=video,
        action=action,
        text=text,
        video_timesteps=video_timesteps,
        action_timesteps=action_timesteps,
        video_grid=video_grid,
        action_grid=action_grid,
    )


def _install_flash_attention_import_stub() -> None:
    """Let the torch-attention upstream model import without FlashAttention."""
    if "flash_attn" in sys.modules:
        return
    module = types.ModuleType("flash_attn")
    module.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)

    def unavailable(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError("The parity harness selects attn_mode='torch'.")

    setattr(module, "flash_attn_func", unavailable)
    sys.modules["flash_attn"] = module


def _upstream_outputs(
    fixture: _Fixture,
    checkpoint_root: Path,
    upstream_root: Path,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Run pinned upstream video then action flows with one shared cache."""
    _install_flash_attention_import_stub()
    sys.path.insert(0, str(upstream_root / "wan_va"))
    try:
        upstream_module = importlib.import_module("modules.model")
        wan_transformer_model = upstream_module.WanTransformer3DModel
    finally:
        sys.path.pop(0)

    model = wan_transformer_model.from_pretrained(
        checkpoint_root / "transformer",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        attn_mode="torch",
    ).to(device)
    model.eval()
    model.create_empty_cache(
        "parity",
        ROBOTWIN_ATTENTION_WINDOW,
        ROBOTWIN_LATENT_TOKEN_PER_CHUNK,
        ROBOTWIN_ACTION_TOKEN_PER_CHUNK,
        device=device,
        dtype=torch.bfloat16,
        batch_size=1,
    )
    with torch.no_grad():
        video_tokens = model(
            {
                "noisy_latents": fixture.video.to(device),
                "timesteps": fixture.video_timesteps.to(device),
                "grid_id": fixture.video_grid[:3].unsqueeze(0).to(device),
                "text_emb": fixture.text.to(device),
            },
            update_cache=1,
            cache_name="parity",
            action_mode=False,
        )
        action_tokens = model(
            {
                "noisy_latents": fixture.action.to(device),
                "timesteps": fixture.action_timesteps.to(device),
                "grid_id": fixture.action_grid[:3].unsqueeze(0).to(device),
                "text_emb": fixture.text.to(device),
            },
            update_cache=1,
            cache_name="parity",
            action_mode=True,
        )
    video = _upstream_video_tokens_to_tensor(video_tokens).float().cpu()
    action = action_tokens.float().cpu()
    model.to("cpu")
    del model, video_tokens, action_tokens
    gc.collect()
    torch.cuda.empty_cache()
    return video, action


def _upstream_video_tokens_to_tensor(tokens: Tensor) -> Tensor:
    """Invert the upstream patch-token ordering to ``[B, C, F, H, W]``."""
    tokens = tokens.reshape(
        tokens.shape[0],
        ROBOTWIN_FRAME_CHUNK_SIZE,
        ROBOTWIN_LATENT_HEIGHT // 2,
        ROBOTWIN_LATENT_WIDTH // 2,
        1,
        2,
        2,
        ROBOTWIN_LATENT_CHANNELS,
    )
    return (
        tokens.permute(0, 7, 1, 4, 2, 5, 3, 6).flatten(6, 7).flatten(4, 5).flatten(2, 3)
    )


def _native_outputs(
    fixture: _Fixture,
    checkpoint_root: Path,
    device: torch.device,
    *,
    compile_network: bool,
) -> tuple[Tensor, Tensor]:
    """Run native video then action flows with the matching cache lifecycle."""
    transformer = LingbotVATransformer(
        LingbotVATransformerConfig(
            checkpoint_root=str(checkpoint_root),
            dtype=torch.bfloat16,
            compile_network=compile_network,
            guidance_scale=1.0,
            action_guidance_scale=1.0,
            latent_height=ROBOTWIN_LATENT_HEIGHT,
            latent_width=ROBOTWIN_LATENT_WIDTH,
            frame_chunk_size=ROBOTWIN_FRAME_CHUNK_SIZE,
            action_per_frame=ROBOTWIN_ACTION_PER_FRAME,
            attn_window=ROBOTWIN_ATTENTION_WINDOW,
        )
    )
    transformer.load_model(device)
    cache = transformer.initialize_autoregressive_cache(
        text_embeddings=fixture.text.to(device),
        batch_size=1,
    )
    cache.start(0)
    video_input = rearrange(
        fixture.video.to(device),
        "b c (f kt) (h kh) (w kw) -> b (f h w) (c kt kh kw)",
        kt=1,
        kh=2,
        kw=2,
    )
    video_timesteps = torch.repeat_interleave(
        fixture.video_timesteps.to(device),
        (ROBOTWIN_LATENT_HEIGHT // 2) * (ROBOTWIN_LATENT_WIDTH // 2),
        dim=1,
    )
    action_input = rearrange(
        fixture.action.to(device),
        "b c f h w -> b (f h w) c",
    )
    action_timesteps = torch.repeat_interleave(
        fixture.action_timesteps.to(device),
        ROBOTWIN_ACTION_PER_FRAME,
        dim=1,
    )
    with torch.no_grad():
        video_tokens = transformer.predict_flow(
            video_input,
            video_timesteps,
            cache,
            input={"grid_id": fixture.video_grid.to(device)},
            persist=True,
        )
        action_tokens = transformer.predict_action_flow(
            action_input,
            action_timesteps,
            cache,
            input={"grid_id": fixture.action_grid.to(device)},
            persist=True,
        )
    cache.finalize(0)
    video = (
        rearrange(
            video_tokens,
            "b (f h w) (c kt kh kw) -> b c (f kt) (h kh) (w kw)",
            f=ROBOTWIN_FRAME_CHUNK_SIZE,
            h=ROBOTWIN_LATENT_HEIGHT // 2,
            w=ROBOTWIN_LATENT_WIDTH // 2,
            kt=1,
            kh=2,
            kw=2,
        )
        .float()
        .cpu()
    )
    action = action_tokens.float().cpu()
    transformer.network.to("cpu")
    del transformer, cache, video_tokens, action_tokens
    gc.collect()
    torch.cuda.empty_cache()
    return video, action


def _difference(reference: Tensor, actual: Tensor) -> _Difference:
    """Summarize absolute error after checking shape and finiteness."""
    if reference.shape != actual.shape:
        raise ValueError(
            f"Shape mismatch: upstream {reference.shape}, native {actual.shape}"
        )
    if not torch.isfinite(reference).all() or not torch.isfinite(actual).all():
        raise ValueError("Parity outputs must be finite.")
    error = (reference - actual).abs()
    return _Difference(
        maximum=float(error.max()),
        mean=float(error.mean()),
        root_mean_square=float(torch.sqrt(torch.mean(error.square()))),
    )


def _check_threshold(name: str, value: float, threshold: float | None) -> None:
    """Fail one explicitly requested parity bound."""
    if threshold is not None and value > threshold:
        raise SystemExit(f"{name} {value:.8g} exceeds threshold {threshold:.8g}")


def main() -> None:
    """Load both implementations, print differences, and enforce given bounds."""
    args = _parser().parse_args()
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    upstream_root = args.upstream_root.expanduser().resolve()
    device = torch.device(args.device)
    fixture = _fixture(args.seed)
    upstream_video, upstream_action = _upstream_outputs(
        fixture,
        checkpoint_root,
        upstream_root,
        device,
    )
    native_video, native_action = _native_outputs(
        fixture,
        checkpoint_root,
        device,
        compile_network=args.compile_native,
    )
    video_difference = _difference(upstream_video, native_video)
    action_difference = _difference(upstream_action, native_action)
    print(
        f"video_shape={tuple(native_video.shape)} video_difference={video_difference}"
    )
    print(
        f"action_shape={tuple(native_action.shape)} "
        f"action_difference={action_difference}"
    )
    _check_threshold(
        "maximum video error",
        video_difference.maximum,
        args.maximum_video_error,
    )
    _check_threshold("mean video error", video_difference.mean, args.mean_video_error)
    _check_threshold(
        "maximum action error",
        action_difference.maximum,
        args.maximum_action_error,
    )
    _check_threshold(
        "mean action error",
        action_difference.mean,
        args.mean_action_error,
    )


if __name__ == "__main__":
    main()

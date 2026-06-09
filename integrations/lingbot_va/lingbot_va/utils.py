"""Small upstream-compatible tensor helpers for LingBot-VA."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor


def get_mesh_id(
    f: int,
    h: int,
    w: int,
    t: int,
    f_w: int = 1,
    f_shift: int = 0,
    action: bool = False,
) -> Tensor:
    """Return upstream LingBot-VA 4-row RoPE/grid ids.

    Args:
        f: Number of frame positions.
        h: Number of height/action-per-frame positions.
        w: Number of width positions.
        t: Stream type id: upstream uses 0 for video and 1 for action.
        f_w: Frame-position multiplier.
        f_shift: Autoregressive frame offset.
        action: If true, use fractional action positions and sentinel ``h/w=-1``.

    Returns:
        Tensor of shape ``[4, f * h * w]``.
    """
    f_idx = torch.arange(f_shift, f + f_shift) * f_w
    h_idx = torch.arange(h)
    w_idx = torch.arange(w)
    ff, hh, ww = torch.meshgrid(f_idx, h_idx, w_idx, indexing="ij")
    if action:
        ff_offset = (torch.ones([h]).cumsum(0) / (h + 1)).view(1, -1, 1)
        ff = ff + ff_offset
        hh = torch.ones_like(hh) * -1
        ww = torch.ones_like(ww) * -1
    grid_id = torch.cat(
        [
            ff.unsqueeze(0),
            hh.unsqueeze(0),
            ww.unsqueeze(0),
        ],
        dim=0,
    ).flatten(1)
    return torch.cat([grid_id, torch.full_like(grid_id[:1], t)], dim=0)


def data_seq_to_patch(
    patch_size: tuple[int, int, int],
    data_seq: Tensor,
    latent_num_frames: int,
    latent_height: int,
    latent_width: int,
    batch_size: int = 1,
) -> Tensor:
    """Invert LingBot-VA's patch-sequence output into ``[B, C, F, H, W]``."""
    p_t, p_h, p_w = patch_size
    post_patch_num_frames = latent_num_frames // p_t
    post_patch_height = latent_height // p_h
    post_patch_width = latent_width // p_w

    data_patch = data_seq.reshape(
        batch_size,
        post_patch_num_frames,
        post_patch_height,
        post_patch_width,
        p_t,
        p_h,
        p_w,
        -1,
    )
    data_patch = data_patch.permute(0, 7, 1, 4, 2, 5, 3, 6)
    data_patch = data_patch.flatten(6, 7).flatten(4, 5).flatten(2, 3)
    return data_patch


def resolve_prompt(value: str | Path) -> str:
    """Resolve an inline prompt or a text file whose first non-empty line is used."""
    if isinstance(value, Path):
        lines = [line.strip() for line in value.read_text().splitlines() if line.strip()]
        assert lines, f"prompt file {value} has no non-empty lines"
        return lines[0]
    assert value, "prompt must be a non-empty string or a path"
    return value

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose synchronized original and post-processed Cam2V video frames."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from flashdreams.runtime_v2.video_tensor import VideoTensorLayout


def compose_postprocess_comparison(
    *,
    pending_generated_frames: torch.Tensor | None,
    generated_frames: torch.Tensor,
    postprocessed_frames: torch.Tensor,
    final_chunk: bool,
    output_layout: VideoTensorLayout,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Pair processed frames with original frames and place them side by side."""
    if output_layout is not VideoTensorLayout.tchw:
        raise ValueError("Cam2V postprocess comparison requires tchw output.")
    if pending_generated_frames is None:
        pending_generated_frames = generated_frames
    else:
        pending_generated_frames = torch.cat(
            (pending_generated_frames, generated_frames), dim=0
        )
    postprocessed_frame_count = int(postprocessed_frames.shape[0])
    if postprocessed_frame_count > pending_generated_frames.shape[0]:
        raise RuntimeError(
            "Post-processing comparison received more post-processed frames than "
            "generated frames "
            f"({postprocessed_frame_count} > {pending_generated_frames.shape[0]})."
        )
    original_frames = pending_generated_frames[:postprocessed_frame_count]
    pending_generated_frames = pending_generated_frames[postprocessed_frame_count:]
    if final_chunk and pending_generated_frames.shape[0] != 0:
        raise RuntimeError(
            "Post-processing comparison requires a frame-preserving processor; "
            "the final raw frames had no post-processed counterparts."
        )
    return (
        _side_by_side_video(original_frames, postprocessed_frames),
        None if pending_generated_frames.shape[0] == 0 else pending_generated_frames,
    )


def _side_by_side_video(
    original_frames: torch.Tensor,
    postprocessed_frames: torch.Tensor,
) -> torch.Tensor:
    """Resize original tchw frames and concatenate them left of processed frames."""
    if original_frames.ndim != 4 or postprocessed_frames.ndim != 4:
        raise ValueError("Cam2V postprocess comparison requires tchw video tensors.")
    if original_frames.shape[0] != postprocessed_frames.shape[0]:
        raise ValueError(
            "Cam2V postprocess comparison requires matching temporal extents."
        )
    if original_frames.shape[1] != postprocessed_frames.shape[1]:
        raise ValueError(
            "Cam2V postprocess comparison requires matching channel counts."
        )
    if original_frames.device != postprocessed_frames.device:
        raise ValueError(
            "Cam2V postprocess comparison requires original and processed frames "
            "on the same device."
        )
    if original_frames.shape[0] == 0:
        return postprocessed_frames.new_empty(
            (
                0,
                postprocessed_frames.shape[1],
                postprocessed_frames.shape[2],
                postprocessed_frames.shape[3] * 2,
            )
        )
    if original_frames.dtype != postprocessed_frames.dtype:
        original_frames = original_frames.to(dtype=postprocessed_frames.dtype)
    if original_frames.shape[-2:] != postprocessed_frames.shape[-2:]:
        original_frames = F.interpolate(
            original_frames,
            size=postprocessed_frames.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    return torch.cat((original_frames, postprocessed_frames), dim=-1)


__all__ = ["compose_postprocess_comparison"]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared demo output-target construction."""

from __future__ import annotations

from pathlib import Path

from flashdreams.runtime.output import NullOutputTarget, OutputTarget
from flashdreams.runtime.video_output import Mp4VideoOutputTarget, VideoWriter

from .spec import Mp4OutputSpec, NullOutputSpec, OutputSpec, WebRTCOutputSpec


def build_output_target(
    output: OutputSpec,
    *,
    mp4_writer: VideoWriter | None = None,
) -> OutputTarget:
    """Build a replay output target from a demo output spec."""
    if isinstance(output, NullOutputSpec):
        return NullOutputTarget(store_results=output.store_results)
    if isinstance(output, Mp4OutputSpec):
        output_path = Path(output.path)
        if mp4_writer is not None:
            return Mp4VideoOutputTarget(
                output_path=output_path,
                fps=output.fps,
                output_layout=output.output_layout,
                writer=mp4_writer,
                move_to_cpu=output.move_to_cpu,
            )
        return Mp4VideoOutputTarget(
            output_path=output_path,
            fps=output.fps,
            output_layout=output.output_layout,
            move_to_cpu=output.move_to_cpu,
        )
    if isinstance(output, WebRTCOutputSpec):
        raise ValueError("WebRTC output does not create a replay OutputTarget.")
    raise TypeError(f"Unsupported demo output spec: {type(output).__name__}.")


__all__ = ["build_output_target"]

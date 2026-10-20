# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal text-to-video application with lazy model initialization."""

import argparse
from collections.abc import Sequence
from typing import Any

from flashdreams.api_v2.application import IApplication
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

from .session import MinimalT2VSession


class MinimalT2VApplication(IApplication):
    """Tutorial app configured by a model adapter.

    The pipeline must accept text and latent dimensions in ``initialize_cache``
    and return ``tchw`` video from ``generate``, as Wan T2V pipelines do.
    """

    def __init__(
        self, pipeline_config: Any, desc: SessionDesc, total_blocks: int = 8
    ) -> None:
        """Keep model configuration without loading weights."""
        if desc.output_layout is not VideoTensorLayout.tchw:
            raise ValueError("Minimal T2V requires tchw output.")
        if total_blocks <= 0:
            raise ValueError("total_blocks must be positive.")
        self.pipeline_config = pipeline_config
        self.desc = desc
        self.total_blocks = total_blocks
        self.pipeline: Any = None
        self.prompt = ""

    def init(self, commandline_args: Sequence[str]) -> None:
        """Parse the initial prompt and load the shared pipeline."""
        parser = argparse.ArgumentParser(description="Generate video from text.")
        parser.add_argument("--default-prompt", default="")
        parser.add_argument("--device", default="cuda")
        args = parser.parse_args(commandline_args)
        self.prompt = args.default_prompt.strip()
        self.pipeline = self.pipeline_config.setup().to(args.device).eval()

    def session_desc(self) -> SessionDesc:
        """Return the model adapter's output description."""
        return self.desc

    def create_session(self, session_desc: SessionDesc) -> MinimalT2VSession:
        """Create a session at the model's supported size and layout."""
        if self.pipeline is None:
            raise RuntimeError("Call init before create_session.")
        if (
            session_desc.video_width != self.desc.video_width
            or session_desc.video_height != self.desc.video_height
            or session_desc.output_layout != self.desc.output_layout
            or session_desc.frames_per_second_for_step
            != self.desc.frames_per_second_for_step
        ):
            raise ValueError("The session must match the model's output description.")
        ratio = self.pipeline.decoder.spatial_compression_ratio
        if session_desc.video_width % ratio or session_desc.video_height % ratio:
            raise ValueError(
                "Video dimensions must be divisible by decoder compression."
            )
        return MinimalT2VSession(
            self.pipeline, session_desc, self.prompt, self.total_blocks
        )

    def close(self) -> None:
        """Release the shared pipeline reference."""
        self.pipeline = None

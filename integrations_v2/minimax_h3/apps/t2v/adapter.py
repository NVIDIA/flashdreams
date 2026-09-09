# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax H3 workflows over the shared FlashDreams v2 T2V application."""

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

from t2v import T2VApplication, T2VApplicationDefaults

from flashdreams.accelerated.multi_head_attention.optimized import QKVFusionOption
from flashdreams.api_v2.application import IApplication
from flashdreams.infra.config import derive_config
from flashdreams.runtime_v2.session_desc import SessionDesc
from minimax_h3.config import (
    PIPELINE_MINIMAX_H3_FL2VA,
    PIPELINE_MINIMAX_H3_REF2VA,
    PIPELINE_MINIMAX_H3_T2VA,
)
from minimax_h3.impl.constants import FPS, align_num_frames, validate_canvas
from minimax_h3.impl.pipeline import MiniMaxH3PipelineConfig
from minimax_h3.impl.references import parse_reference_specs


class MiniMaxH3Application(T2VApplication):
    """One-block joint audio/video inference with video-only v2 output."""

    def __init__(
        self, pipeline_config: MiniMaxH3PipelineConfig = PIPELINE_MINIMAX_H3_T2VA
    ) -> None:
        super().__init__(
            defaults=T2VApplicationDefaults(
                pipeline_config=pipeline_config,
                total_blocks=1,
                pixel_width=768,
                pixel_height=768,
                fps=FPS,
            )
        )
        self._request_inputs: dict[str, Any] = {}

    def _configure_argument_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--duration", type=float, default=5.0)
        parser.add_argument(
            "--steps",
            type=int,
            default=30,
            help="Scheduler grid points (30 means 29 joint predictions).",
        )
        parser.add_argument("--image-path", type=Path)
        parser.add_argument("--last-image-path", type=Path)
        parser.add_argument(
            "--reference",
            action="append",
            default=[],
            help="Ordered image:path, video:path, or audio:path input.",
        )
        parser.add_argument("--lora")
        parser.add_argument("--lora-weight-name")
        parser.add_argument("--lora-scale", type=float, default=1.0)
        parser.add_argument(
            "--attention", choices=("optimized", "torch"), default="optimized"
        )
        parser.add_argument(
            "--fuse-qkv",
            action="store_true",
            help="Opt in to extra fused-QKV weight storage; increases peak memory.",
        )
        parser.add_argument("--model-id", default=self.pipeline_config.model_id)
        parser.add_argument("--revision", default=self.pipeline_config.revision)

    def _apply_parsed_arguments(self, args: argparse.Namespace) -> None:
        align_num_frames(args.duration)
        if args.steps < 2:
            raise ValueError("--steps must be at least 2 scheduler points")
        if not 0 <= args.lora_scale <= 4:
            raise ValueError("--lora-scale must be between 0 and 4")
        references = parse_reference_specs(args.reference) if args.reference else ()
        workflow = self.pipeline_config.workflow
        if workflow == "t2va" and (
            args.image_path or args.last_image_path or references
        ):
            raise ValueError("t2va does not accept keyframes or references")
        if workflow == "fl2va" and (
            not (args.image_path or args.last_image_path) or references
        ):
            raise ValueError("fl2va requires first/last keyframes and no references")
        if workflow == "ref2va" and (
            not references or args.image_path or args.last_image_path
        ):
            raise ValueError("ref2va requires ordered references and no keyframes")
        for path in (args.image_path, args.last_image_path):
            if path is not None and not path.is_file():
                raise FileNotFoundError(path)
        self._request_inputs = dict(
            duration=args.duration,
            image_path=args.image_path,
            last_image_path=args.last_image_path,
            references=references,
            lora=args.lora,
            lora_weight_name=args.lora_weight_name,
            lora_scale=args.lora_scale,
        )
        config = self.pipeline_config
        optimized = replace(
            config.transformer.optimized_impl,
            qkv_fusion_option=QKVFusionOption.FULL
            if args.fuse_qkv
            else QKVFusionOption.NONE,
        )
        self._pipeline_config = replace(
            config,
            model_id=args.model_id,
            revision=args.revision,
            transformer=replace(
                config.transformer,
                attention_backend=args.attention,
                optimized_impl=optimized,
            ),
            scheduler=replace(config.scheduler, num_inference_steps=args.steps),
            audio_scheduler=replace(
                config.audio_scheduler, num_inference_steps=args.steps
            ),
        )

    def _cache_initialization_kwargs(self, session_desc: SessionDesc) -> dict[str, Any]:
        return dict(self._request_inputs)

    def _validate_total_blocks(self, total_blocks: int) -> None:
        if total_blocks != 1:
            raise ValueError("MiniMax H3 generates its complete clip in one block")

    def _validate_frame_size(self, session_desc: SessionDesc, pipeline: Any) -> None:
        validate_canvas(session_desc.video_width, session_desc.video_height)
        if session_desc.frames_per_second_for_step != FPS:
            raise ValueError("MiniMax H3 requires 24 fps")

    def _apply_compile_override(self, pipeline_config: Any, enabled: bool) -> Any:
        return derive_config(pipeline_config, compile_network=enabled)

    def _apply_seed_override(self, pipeline_config: Any, seed: int) -> Any:
        return derive_config(pipeline_config, seed=seed)


def create_app() -> IApplication:
    """Create the prompt-only H3 application without loading weights."""
    return MiniMaxH3Application()


def create_app_fl2va() -> IApplication:
    """Create the first/last-keyframe H3 application."""
    return MiniMaxH3Application(PIPELINE_MINIMAX_H3_FL2VA)


def create_app_ref2va() -> IApplication:
    """Create the ordered-reference H3 application."""
    return MiniMaxH3Application(PIPELINE_MINIMAX_H3_REF2VA)

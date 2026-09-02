# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax H3 text, keyframe, and ordered-reference runners."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

from loguru import logger
from tyro.conf import UseAppendAction

from flashdreams.infra.postprocess import VideoTensorLayout
from flashdreams.infra.runner import Runner, RunnerConfig
from flashdreams.infra.runner_io import runner_artifact_path, write_runner_stats
from flashdreams.runtime.video_output import Mp4VideoOutputTarget
from minimax_h3.pipeline import MiniMaxH3Pipeline, MiniMaxH3PipelineCache
from minimax_h3.references import parse_reference_specs


@dataclass(kw_only=True)
class MiniMaxH3RunnerConfig(RunnerConfig):
    """Options shared by all released MiniMax H3 workflows."""

    _target: type[MiniMaxH3Runner] = field(default_factory=lambda: MiniMaxH3Runner)

    prompt: str = "Animate this scene with coherent natural motion."
    """Text description of the desired video motion and appearance."""

    pixel_height: int = 768
    """Output video height in pixels."""

    pixel_width: int = 768
    """Output video width in pixels."""

    duration: float = 5.0
    """Requested duration before H3 frame-grid alignment."""

    steps: int = 30
    """Number of scheduler grid points."""

    seed: int = 42
    """CPU generator seed used by both H3 schedulers."""

    low_ram: bool = True
    """Split conditioning, denoising, and decoding into checkpointed stages."""

    restart: bool = False
    """Ignore matching stage checkpoints and regenerate the full rollout."""

    attention: Literal["auto", "flash", "default"] = "auto"
    """FlashDreams native SDPA backend selection."""

    lora: str | None = None
    """Local Musubi adapter path or Hugging Face repository ID."""

    lora_weight_name: str | None = None
    """Adapter filename override for Hugging Face repositories."""

    lora_scale: float = 1.0
    """LoRA adapter strength."""

    fps: int = 24
    """H3's fixed output frame rate."""

    postprocess_output_layout: VideoTensorLayout | None = "tchw"
    """Decoded H3 frame layout used by FlashDreams runtime output."""


@dataclass(kw_only=True)
class MiniMaxH3T2VARunnerConfig(MiniMaxH3RunnerConfig):
    """Runner config for prompt-only generation."""

    _target: type[MiniMaxH3T2VARunner] = field(
        default_factory=lambda: MiniMaxH3T2VARunner
    )


@dataclass(kw_only=True)
class MiniMaxH3FL2VARunnerConfig(MiniMaxH3RunnerConfig):
    """Runner config for first-frame, last-frame, or dual-keyframe generation."""

    _target: type[MiniMaxH3FL2VARunner] = field(
        default_factory=lambda: MiniMaxH3FL2VARunner
    )

    image_path: Path | None = None
    """Optional first-frame image path."""

    last_image_path: Path | None = None
    """Optional last-frame image path."""


@dataclass(kw_only=True)
class MiniMaxH3Ref2VARunnerConfig(MiniMaxH3RunnerConfig):
    """Runner config for ordered image, video, and audio references."""

    _target: type[MiniMaxH3Ref2VARunner] = field(
        default_factory=lambda: MiniMaxH3Ref2VARunner
    )

    reference: Annotated[list[str], UseAppendAction] = field(default_factory=list)
    """Ordered ``image:path``, ``video:path``, or ``audio:path`` references."""


class MiniMaxH3Runner(Runner[MiniMaxH3RunnerConfig, MiniMaxH3Pipeline]):
    """Drive one H3 workflow and persist its video-only artifact."""

    config: MiniMaxH3RunnerConfig
    pipeline: MiniMaxH3Pipeline

    def _initialize_cache(self, output_path: Path) -> MiniMaxH3PipelineCache:
        raise NotImplementedError

    def _initialize_common(
        self,
        output_path: Path,
        *,
        image_path: Path | None = None,
        last_image_path: Path | None = None,
        reference: list[str] | tuple[str, ...] = (),
    ) -> MiniMaxH3PipelineCache:
        config = self.config
        return self.pipeline.initialize_cache(
            prompt=config.prompt,
            image_path=image_path,
            last_image_path=last_image_path,
            references=parse_reference_specs(reference) if reference else (),
            output_path=output_path,
            width=config.pixel_width,
            height=config.pixel_height,
            duration=config.duration,
            steps=config.steps,
            seed=config.seed,
            low_ram=config.low_ram,
            restart=config.restart,
            attention=config.attention,
            lora=config.lora,
            lora_weight_name=config.lora_weight_name,
            lora_scale=config.lora_scale,
        )

    def run(self) -> None:
        """Generate the single H3 step and write a video-only MP4."""
        config = self.config
        video_path = runner_artifact_path(config.output_dir, config.runner_name, "mp4")
        cache = self._initialize_cache(video_path)
        output_stream = self.create_video_output_stream(fps=config.fps)
        output_target = Mp4VideoOutputTarget(
            output_path=video_path,
            fps=config.fps,
            output_layout=output_stream.output_layout,
            enabled=self.is_rank_zero,
        )
        output_target.open()
        frames = self.pipeline.generate(0, cache)
        metrics = self.pipeline.finalize(0, cache)
        output_target.write(
            output_stream.process(frames, autoregressive_index=0, metrics=metrics)
        )
        tail = output_stream.finish()
        if tail is not None:
            output_target.write(tail)
        artifacts = output_target.close()
        if not artifacts:
            return
        self.pipeline.mark_complete(cache)
        video_artifact = artifacts[0]
        logger.info(
            "[{}] wrote {} video to {}",
            config.runner_name,
            tuple(frames.shape),
            Path(video_artifact.uri).resolve(),
        )
        stats_history = video_artifact.metadata["stats_history"]
        if stats_history:
            stats_path = write_runner_stats(
                config.output_dir,
                config.runner_name,
                list(stats_history),
            )
            logger.info(
                "[{}] wrote stats to {}", config.runner_name, stats_path.resolve()
            )


class MiniMaxH3T2VARunner(MiniMaxH3Runner):
    """Prompt-only H3 runner."""

    config: MiniMaxH3T2VARunnerConfig

    def _initialize_cache(self, output_path: Path) -> MiniMaxH3PipelineCache:
        return self._initialize_common(output_path)


class MiniMaxH3FL2VARunner(MiniMaxH3Runner):
    """First-frame, last-frame, or dual-keyframe H3 runner."""

    config: MiniMaxH3FL2VARunnerConfig

    def _initialize_cache(self, output_path: Path) -> MiniMaxH3PipelineCache:
        return self._initialize_common(
            output_path,
            image_path=self.config.image_path,
            last_image_path=self.config.last_image_path,
        )


class MiniMaxH3Ref2VARunner(MiniMaxH3Runner):
    """Ordered-reference H3 runner."""

    config: MiniMaxH3Ref2VARunnerConfig

    def _initialize_cache(self, output_path: Path) -> MiniMaxH3PipelineCache:
        return self._initialize_common(output_path, reference=self.config.reference)


__all__ = [
    "MiniMaxH3FL2VARunner",
    "MiniMaxH3FL2VARunnerConfig",
    "MiniMaxH3Ref2VARunner",
    "MiniMaxH3Ref2VARunnerConfig",
    "MiniMaxH3Runner",
    "MiniMaxH3RunnerConfig",
    "MiniMaxH3T2VARunner",
    "MiniMaxH3T2VARunnerConfig",
]

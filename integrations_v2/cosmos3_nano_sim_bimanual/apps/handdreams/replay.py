# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controller-free replay of recorded GA actions through the HandDreams pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from handdreams.config import HanddreamsScenario
from handdreams.scenes import ScenePreview
from handdreams.ui import (
    HandDreamsPhase,
    HandDreamsUILoop,
    HandDreamsUIState,
    HandDreamsUIStatus,
    scene_strip_height,
)
from torch import Tensor

from cosmos3_nano_sim_bimanual import action_format
from cosmos3_nano_sim_bimanual.config import (
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_WIDTH,
)
from cosmos3_nano_sim_bimanual.encoder import HandActionInput
from cosmos3_nano_sim_bimanual.pipeline import HandStreamInferencePipeline
from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.loop import IModelLoop
from flashdreams.api_v2.session import ISession
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

from .adapter import (
    _add_model_arguments,
    _load_scene,
    _resolve_pipeline_config,
    _setup_inference_pipeline,
)
from .agibot_replay import prepare_dataset_replays
from .scenes import Scene, resolve_scene_path, scene_previews


def load_recorded_actions(path: str | Path, *, actions_per_step: int) -> Tensor:
    """Load normalized GA ``[N,59]`` actions and pad them without renormalizing.

    Rows must use unified_v1 global-asinh normalization, 16-frame backward
    wrist anchors, and metric grippers. Missing AgiBot fingertips are zeroed
    after normalization. The first row is the transition from the scene image
    to the next frame. Legacy 29D/64D recordings are not accepted.

    Raises:
        ValueError: The file is not a finite floating-point NPY array of complete
            action chunks, or contains unmasked fingertip channels.
    """
    if actions_per_step <= 0:
        raise ValueError("actions_per_step must be positive")
    values = np.load(Path(path).expanduser(), allow_pickle=False)
    if not isinstance(values, np.ndarray):
        values.close()
        raise ValueError("recorded actions must be a .npy array, not an archive")
    if (
        values.ndim != 2
        or values.shape[1] != action_format.RAW_ACTION_DIM
        or values.dtype.kind != "f"
    ):
        raise ValueError(
            "recorded actions must be normalized GA floating-point [N,59] values; "
            "legacy recordings and raw poses are not supported"
        )
    if not len(values) or len(values) % actions_per_step:
        raise ValueError(
            f"recorded actions must contain a positive multiple of {actions_per_step} "
            "rows; partial chunks are not truncated or padded"
        )
    actions = action_format.pad_action(values)
    if np.any(actions[:, 18:33]) or np.any(actions[:, 43:58]):
        raise ValueError("recorded AgiBot fingertip channels must be masked to zero")
    return torch.from_numpy(actions)


@dataclass
class ReplayState:
    """One replay's action cursor and pipeline-owned inference cache."""

    pipeline: HandStreamInferencePipeline
    scene: Scene
    actions: Tensor
    cache: Any
    domain_ids: Tensor | None = None
    """Per-row dataset domains; external recordings default to AgiBot domain 15."""
    row: int = 0
    """First action row not yet generated successfully."""
    selecting: bool = False
    """Keep the scene picker idle without creating an inference cache."""


class ReplayModelLoop(IModelLoop[ReplayState]):
    """Feed complete recorded chunks to the standard generation pipeline."""

    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        state = self.state
        if self.is_finished():
            return []
        if state.cache is None:
            raise RuntimeError("replay cache is closed")
        count = state.pipeline.actions_per_step
        index = state.row // count
        rows = slice(state.row, state.row + count)
        inputs = (
            state.actions[rows]
            if state.domain_ids is None
            else HandActionInput(state.actions[rows], state.domain_ids[rows])
        )
        video = state.pipeline.generate(index, state.cache, input=inputs)
        metrics = state.pipeline.finalize(index, state.cache) or {}
        state.row += count
        return [
            StepResult(
                step_index=step_index,
                output=video,
                frame_count=int(video.shape[0]),
                output_layout=VideoTensorLayout.tchw,
                metrics=metrics,
            )
        ]

    def is_finished(self) -> bool:
        return self.state.selecting or self.state.row >= len(self.state.actions)

    def reset(self) -> None:
        """Restart the recorded sequence with a fresh scene cache."""
        self.close()
        if not self.state.selecting:
            context = {"scene": self.state.scene}
            self.state.cache = self.state.pipeline.initialize_cache(
                transformer_context=context, decoder_context=context
            )
        self.state.row = 0

    def close(self) -> None:
        """Release this replay's cache once."""
        cache, self.state.cache = self.state.cache, None
        if cache is not None:
            self.state.pipeline.release_session(cache)


class ReplaySession(ISession):
    """Recorded rollout with an optional interactive scene picker."""

    def __init__(
        self,
        pipeline: HandStreamInferencePipeline,
        scene: Scene,
        actions: Tensor,
        session_desc: SessionDesc,
        *,
        domain_ids: Tensor | None = None,
        previews: tuple[ScenePreview, ...] = (),
        selecting: bool = False,
    ) -> None:
        self._state = ReplayState(
            pipeline,
            scene,
            actions,
            cache=None,
            domain_ids=domain_ids,
            selecting=selecting,
        )
        self._previews = previews
        self._session_desc = session_desc
        self._loop: IModelLoop[ReplayState] | None = None
        self.closed = False

    @property
    def session_desc(self) -> SessionDesc:
        return self._session_desc

    def init(self) -> None:
        """Initialize recorded playback without controller services."""
        if self.closed or self._loop is not None:
            raise RuntimeError(
                "replay session cannot be initialized twice or after close"
            )
        if self._previews:
            self.register_ui_loop(
                HandDreamsUILoop,
                state=HandDreamsUIState(
                    status=HandDreamsUIStatus(
                        phase=(
                            HandDreamsPhase.SELECTING
                            if self._state.selecting
                            else HandDreamsPhase.LIVE
                        ),
                        current_scene=(
                            ""
                            if self._state.selecting
                            else self._session_desc.metadata["sample"]
                        ),
                        scene_catalog=tuple(preview.name for preview in self._previews),
                        replay=True,
                    ),
                    scene_previews=self._previews,
                    stage_width=DEFAULT_VIDEO_WIDTH,
                    stage_height=DEFAULT_VIDEO_HEIGHT,
                ),
            )
        self._loop = self.register_model_loop(ReplayModelLoop, state=self._state)
        self._loop.reset()

    def close(self) -> None:
        self.closed = True
        if self._loop is not None:
            self._loop.close()
        if self._previews and self._loop is not None:
            self.ui_loop.close()


class HandDreamsReplayApplication(IApplication):
    """Recorded-action model test independent of Isaac Teleop and CloudXR."""

    def __init__(self) -> None:
        self._pipeline: HandStreamInferencePipeline | None = None
        self._recordings: dict[str, tuple[Scene, Tensor, Tensor | None]] = {}
        self._previews: tuple[ScenePreview, ...] = ()
        self._session: ReplaySession | None = None
        self._fps = 30

    def default_client_window_mode(self) -> str:
        return "webrtc"

    def session_desc(self) -> SessionDesc:
        return SessionDesc(
            output_layout=VideoTensorLayout.tchw,
            video_width=DEFAULT_VIDEO_WIDTH,
            video_height=DEFAULT_VIDEO_HEIGHT,
            frames_per_second_for_ui=self._fps,
            frames_per_second_for_step=self._fps,
        )

    def init(self, commandline_args: Sequence[str]) -> None:
        """Prepare recorded inputs and the same model used by live control."""
        parser = argparse.ArgumentParser(prog="handdreams-replay")
        parser.add_argument("--scene-path", type=Path)
        parser.add_argument(
            "--recorded-actions",
            type=Path,
            help="External .npy of GA-normalized [N,59] actions matching this scene.",
        )
        _add_model_arguments(parser)
        args = parser.parse_args(list(commandline_args))
        if (args.scene_path is None) != (args.recorded_actions is None):
            parser.error(
                "--scene-path and --recorded-actions must be supplied together"
            )
        if self._pipeline is not None:
            raise RuntimeError("replay application is already initialized")
        config = _resolve_pipeline_config(vars(args))
        count = args.chunk_size * action_format.TEMPORAL_COMPRESSION_FACTOR
        recordings: dict[str, tuple[Path, Tensor, Tensor | None]] = {}
        if args.scene_path is None:
            for name, (path, values, domains) in prepare_dataset_replays().items():
                usable = len(values) // count * count
                if not usable:
                    raise ValueError(
                        f"dataset replay {name!r} requires at least {count} action rows"
                    )
                if usable != len(values):
                    print(
                        f"[handdreams-replay] {name}: dropping "
                        f"{len(values) - usable} trailing actions to replay {usable} "
                        f"rows in complete {count}-row chunks",
                        flush=True,
                    )
                recordings[name] = (
                    path,
                    torch.from_numpy(values[:usable]),
                    torch.from_numpy(domains[:usable]),
                )
            if not recordings:
                raise ValueError("dataset replay requires at least one scenario")
            self._previews = scene_previews(
                [(name, str(path)) for name, (path, _, _) in recordings.items()]
            )
        else:
            scene_path = resolve_scene_path(args.scene_path)
            actions = load_recorded_actions(
                args.recorded_actions, actions_per_step=count
            )
            recordings[scene_path.name] = (scene_path, actions, None)
        try:
            self._pipeline = _setup_inference_pipeline(config)
            for name, (path, actions, domain_ids) in recordings.items():
                scene = _load_scene(HanddreamsScenario(scene_path=path), self._pipeline)
                fps = int(scene.fps)
                if fps <= 0 or fps != scene.fps:
                    raise ValueError(
                        "replay requires a positive whole-number scene FPS"
                    )
                if self._recordings and fps != self._fps:
                    raise ValueError("replay scenarios must share the same scene FPS")
                self._fps = fps
                self._recordings[name] = (
                    scene,
                    actions.to(self._pipeline.device),
                    None
                    if domain_ids is None
                    else domain_ids.to(self._pipeline.device),
                )
                self._pipeline.prepare_scene(scene)
        except BaseException:
            self.close()
            raise

    def create_session(self, session_desc: SessionDesc) -> ISession:
        """Create one replay paced at the recorded scene's conditioning cadence."""
        if self._pipeline is None or not self._recordings:
            raise RuntimeError("initialize replay before creating a session")
        if self._session is not None and not self._session.closed:
            raise RuntimeError(
                "close the active replay before creating another session"
            )
        if session_desc.output_layout is not VideoTensorLayout.tchw:
            raise ValueError("replay output requires tchw layout")
        sample = session_desc.metadata.get("sample", next(iter(self._recordings)))
        if not isinstance(sample, str) or sample not in self._recordings:
            raise ValueError(f"unknown replay scene {sample!r}")
        selecting = bool(self._previews) and (
            session_desc.metadata.get("select_scene") is True
            or "sample" not in session_desc.metadata
        )
        scene, actions, domain_ids = self._recordings[sample]
        session_desc = replace(
            session_desc,
            video_width=DEFAULT_VIDEO_WIDTH,
            video_height=DEFAULT_VIDEO_HEIGHT
            + (scene_strip_height(DEFAULT_VIDEO_WIDTH) if self._previews else 0),
            frames_per_second_for_ui=self._fps,
            frames_per_second_for_step=self._fps,
            metadata={**session_desc.metadata, "sample": sample},
        )
        self._session = ReplaySession(
            self._pipeline,
            scene,
            actions,
            session_desc,
            domain_ids=domain_ids,
            previews=self._previews,
            selecting=selecting,
        )
        return self._session

    def close(self) -> None:
        """Release the current replay and application-owned model."""
        pipeline, self._pipeline = self._pipeline, None
        session, self._session = self._session, None
        try:
            if session is not None:
                session.close()
        finally:
            self._recordings.clear()
            self._previews = ()
            if pipeline is not None:
                pipeline.close()


def create_app() -> IApplication:
    """Return the controller-free recorded-action application."""
    return HandDreamsReplayApplication()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-session HandDreams V2 inference, input, and UI state."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

import numpy as np
import torch
from flashdreams_controller_input.events import ControllerFrameEvent
from torch import Tensor
from torch.nn import functional as F

from flashdreams.api_v2.loop import IModelLoop, invoke_async
from flashdreams.api_v2.session import ISession
from flashdreams.infra.postprocess import VideoPostprocessStream
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents

from .alignment import (
    CALIBRATION_TRIGGER_THRESHOLD,
    AlignmentGuide,
    AlignmentPreview,
)
from .config import HandDreamsLaunchConfig
from .controller_preview import ControllerPreviewCalibration
from .controllers.quest_controller import controller_separation_basis
from .controllers.sampler import ControlWindowSampler
from .input import (
    OverlapWindowBuffer,
    controller_frame,
    controller_status,
)
from .scenes import ScenePreview
from .teleop import PreparedTeleopWindow, TeleopControllerPipeline
from .ui import (
    HandDreamsPhase,
    HandDreamsUILoop,
    HandDreamsUIState,
    HandDreamsUIStatus,
)

_MAX_RESAMPLE_GAP_NS = 1_000_000_000
"""Bound grid catch-up even when pose-age rejection is disabled by the user."""

_ALIGNMENT_HOLD_S = 1.0
"""Require a sustained tracked, aligned, open pose before accepting calibration."""

_CALIBRATION_HOLD_S = 0.5
"""Duration of a deliberate trigger hold before freezing the alignment guide."""


class HandDreamsRolloutBuilder(Protocol):
    """Model-owned scene encoding and rollout operations used by the demo."""

    scene: Any
    """Model-prepared scene containing the demo's initial poses and image."""

    @property
    def actions_per_step(self) -> int:
        """Return the action-row count consumed by one generation step."""
        ...

    def prepare_scene(self) -> None:
        """Warm the model for this scene before consuming live input."""
        ...

    def build_cache(self) -> Any:
        """Create isolated rollout state for this session."""
        ...

    def encode_actions(self, frames: list[dict[str, Any]]) -> Tensor:
        """Encode calibrated controller targets into model input."""
        ...

    def generate(
        self,
        step_index: int,
        cache: Any,
        actions: Tensor,
        on_accepted: Callable[[], None],
    ) -> Tensor:
        """Accept actions before decoding; failures before acceptance may retry."""
        ...

    def finalize(self, step_index: int, cache: Any) -> Mapping[str, Any]:
        """Complete one model step and return its measurements."""
        ...

    def release_cache(self, cache: Any) -> None:
        """Release a session's rollout state without closing the shared model."""
        ...


@dataclass(slots=True)
class HandDreamsModelState:
    """Mutable state owned only by the model loop thread."""

    builder: HandDreamsRolloutBuilder
    config: HandDreamsLaunchConfig
    session_desc: SessionDesc
    pipeline: TeleopControllerPipeline | None
    postprocess_stream: VideoPostprocessStream | None = None
    postprocess_enabled: bool = False
    stage_width: int = 640
    stage_height: int = 360
    scene_catalog: tuple[str, ...] = ()
    ui_loop: HandDreamsUILoop | None = None
    inference_cache: Any | None = None
    last_input_frame_id: int = -1
    """Latest accepted source frame identifier; reset only with the session."""
    last_input_capture_ns: int = -1
    """Latest accepted source timestamp; retained across tracking loss."""
    input_is_emulated: bool | None = None
    """Setup policy from accepted controller frames; unknown clients align normally."""
    poses: OverlapWindowBuffer = field(default_factory=OverlapWindowBuffer)
    control_sampler: ControlWindowSampler | None = None
    """Source observations retained for nearest-neighbour control-grid selection."""
    next_control_ns: int | None = None
    """Next source-clock grid point, independent of the UI's polling rate."""
    last_control_ns: int | None = None
    """Last selected original timestamp; repeated selections produce no action."""
    last_observed_ns: int | None = None
    """Newest live source time, used to reset sampling after a stale gap."""
    phase: HandDreamsPhase = HandDreamsPhase.WAITING
    command_sequence: int = 0
    steps_generated: int = 0
    frames_generated: int = 0
    rows_consumed: int = 0
    last_wrist_observed_at: float | None = None
    """Model-clock receipt time for detecting controller source silence."""
    calibration_since: float | None = None
    """Model-clock start of the uninterrupted trigger reference capture."""
    calibration_progress: float = 0.0
    """Completed fraction of the close-claw hold, advanced only by fresh input."""
    calibration_hand: str | None = None
    """Virtual controller whose trigger must remain held until capture completes."""
    alignment: AlignmentGuide | None = None
    alignment_preview: AlignmentPreview | None = None
    controller_calibration: ControllerPreviewCalibration | None = None
    """Frozen live pose reference published for independent UI input rendering."""
    alignment_since: float | None = None
    """Model-clock start of the current uninterrupted alignment hold."""
    closed: bool = False

    @property
    def scene(self) -> Any:
        return self.builder.scene

    @property
    def actions_per_step(self) -> int:
        return self.builder.actions_per_step

    @property
    def control_fps(self) -> float:
        """Effective source-time sampling and model-conditioning cadence."""
        return (
            self.scene.fps
            if self.config.control_fps is None
            else self.config.control_fps
        )

    @property
    def target_frames(self) -> int | None:
        duration = self.config.duration_seconds
        return None if duration is None else round(self.control_fps * duration)

    def set_postprocess_enabled(self, enabled: bool) -> None:
        """Select processed or baseline frames without resetting the warm stream."""
        if enabled and self.postprocess_stream is None:
            raise RuntimeError(
                "Cannot enable post-processing without --postprocess-preset."
            )
        self.postprocess_enabled = bool(enabled)


class HandDreamsModelLoop(IModelLoop[HandDreamsModelState]):
    """Generate chunks from calibrated controller targets."""

    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        state = self.state
        if state.closed or state.phase is HandDreamsPhase.SELECTING:
            return []
        started = time.perf_counter()
        warming = (
            state.phase is HandDreamsPhase.WAITING and state.inference_cache is None
        )

        # Reuse the application-prepared model cache with fresh decoder history.
        if state.inference_cache is None:
            warm_started = time.perf_counter()
            state.inference_cache = state.builder.build_cache()
            print(
                f"[handdreams] session cache ready in "
                f"{time.perf_counter() - warm_started:.1f}s",
                flush=True,
            )

        if warming:
            # This step's event snapshot predates potentially slow preparation.
            # Take the neutral pose from the next input batch after warmup.
            self._publish_status()
            return []

        action_rows = state.actions_per_step
        prepared: PreparedTeleopWindow | None = None
        controller_pipeline = state.pipeline
        cosmos_accepted = False
        try:
            now = time.perf_counter()
            self._expire_stale_input(now=now)
            self._consume_live_events(events, now=now)
            self._publish_status()
            if not state.poses.ready(action_rows):
                return []
            if controller_pipeline is None:
                raise RuntimeError("live HandDreams input requires a Teleop pipeline")
            window = state.poses.peek(action_rows)
            state.command_sequence += 1
            if state.steps_generated == 0:
                print(
                    f"[handdreams] preparing first {action_rows} actions "
                    f"from {len(window)} controller poses",
                    flush=True,
                )
            prepared = controller_pipeline.prepare_window(
                window,
                command_sequence=state.command_sequence,
            )
            normalized = state.builder.encode_actions(prepared.frames)

            model_started = time.perf_counter()
            model_finished = model_started
            first_generation = state.steps_generated == 0
            if first_generation:
                print(
                    "[handdreams] generating first video chunk",
                    flush=True,
                )

            def accept_cosmos_output() -> None:
                nonlocal cosmos_accepted, model_finished
                # The action transaction belongs to diffusion acceptance, not
                # presentation. Preserve the legacy behavior when decode fails.
                cosmos_accepted = True
                model_finished = time.perf_counter()
                if prepared is not None:
                    assert controller_pipeline is not None
                    controller_pipeline.commit(prepared)
                    state.poses.consume(action_rows)
                state.rows_consumed += action_rows
                state.steps_generated += 1
                if first_generation:
                    print(
                        "[handdreams] first diffusion complete; decoding video",
                        flush=True,
                    )

            ar_idx = state.steps_generated
            assert state.inference_cache is not None
            pixels = state.builder.generate(
                ar_idx,
                state.inference_cache,
                normalized,
                accept_cosmos_output,
            )
        except Exception:
            if prepared is not None and not cosmos_accepted:
                assert controller_pipeline is not None
                controller_pipeline.abort(prepared)
            raise

        if state.config.synchronize_metrics and pixels.is_cuda:
            torch.cuda.current_stream(pixels.device).synchronize()
        decoded = time.perf_counter()
        timings = state.builder.finalize(ar_idx, state.inference_cache)

        target = state.target_frames
        if target is not None:
            pixels = pixels[: max(0, target - state.frames_generated)]
        raw_frame_count = int(pixels.shape[0])
        state.frames_generated += raw_frame_count
        final_chunk = state.steps_generated >= state.config.total_steps or (
            target is not None and state.frames_generated >= target
        )
        output = pixels.detach()
        stream = state.postprocess_stream
        postprocess_submit_s: float | None = None
        if stream is not None:
            postprocess_started = time.perf_counter()
            processed = output
            if raw_frame_count:
                processed = stream.process(output, autoregressive_index=ar_idx)
            if final_chunk:
                tail = stream.finish()
                if tail is not None:
                    processed = (
                        tail if not processed.shape[0] else torch.cat((processed, tail))
                    )
            output = (
                processed
                if state.postprocess_enabled
                else _resize_raw_for_presentation(
                    output,
                    height=state.stage_height,
                    width=state.stage_width,
                )
            )
            postprocess_submit_s = time.perf_counter() - postprocess_started
        if state.config.synchronize_metrics and output.is_cuda:
            torch.cuda.current_stream(output.device).synchronize()
        frame_count = int(output.shape[0])
        if frame_count <= 0:
            self._publish_status()
            return []
        if first_generation:
            print(
                f"[handdreams] first video chunk ready: {frame_count} frames "
                f"in {time.perf_counter() - model_started:.1f}s",
                flush=True,
            )
        metrics: dict[str, float | int] = {
            "input_prepare_s": model_started - started,
            "model_wall_s": model_finished - model_started,
            "decode_s": decoded - model_finished,
            "total_s": decoded - started,
            "action_start": state.rows_consumed - action_rows,
            "action_rows": action_rows,
        }
        if postprocess_submit_s is not None:
            metrics["postprocess_submit_s"] = postprocess_submit_s
            metrics["postprocess_enabled"] = int(state.postprocess_enabled)
        metrics.update(_numeric_metrics(timings))
        self._publish_status()
        return [
            StepResult(
                step_index=step_index,
                output=output,
                frame_count=frame_count,
                output_layout=state.session_desc.output_layout,
                metrics=metrics,
            )
        ]

    def _consume_live_events(self, events: UserInputEvents, *, now: float) -> None:
        state = self.state
        if state.pipeline is None:
            raise RuntimeError("live HandDreams input requires a Teleop pipeline")
        pending: list[ControllerFrameEvent] = []
        disconnected = False
        for event in events.get_events():
            if not isinstance(event, ControllerFrameEvent):
                continue
            if event.action == "disconnected":
                pending.clear()
                disconnected = True
            else:
                pending.append(event)
        if disconnected:
            self._handle_tracking_loss()

        # Only the suffix after the final disconnect can advance source identity.
        accepted: list[ControllerFrameEvent] = []
        for event in sorted(pending, key=lambda event: cast(int, event.frame_id)):
            frame_id = cast(int, event.frame_id)
            capture_ns = cast(int, event.client_capture_ns)
            if (
                frame_id <= state.last_input_frame_id
                or capture_ns <= state.last_input_capture_ns
            ):
                continue
            state.last_input_frame_id = frame_id
            state.last_input_capture_ns = capture_ns
            accepted.append(event)
        if accepted:
            emulated = accepted[-1].is_emulated
            boundary = max(
                (
                    index + 1
                    for index, event in enumerate(accepted)
                    if event.is_emulated != emulated
                ),
                default=0,
            )
            if boundary or (
                state.input_is_emulated is not None
                and state.input_is_emulated != emulated
            ):
                # A source handoff cannot inherit another participant's rollout.
                frame_id, capture_ns = (
                    state.last_input_frame_id,
                    state.last_input_capture_ns,
                )
                self.reset()
                state.last_input_frame_id = frame_id
                state.last_input_capture_ns = capture_ns
            state.input_is_emulated = emulated
            accepted = accepted[boundary:]
        if state.phase is not HandDreamsPhase.LIVE:
            if not state.config.auto_start:
                self._check_alignment_history(accepted[:-1])
            # Startup queues can span model/decoder loading. Anchor at the
            # current pose instead of replaying movement from before calibration.
            accepted = accepted[-1:]
        for event in accepted:
            wrist = state.pipeline.transform(controller_frame(event))
            state.last_wrist_observed_at = now

            left_tracked, right_tracked, left_open, right_open = controller_status(
                wrist,
                threshold=state.config.open_claw_threshold,
            )
            if not (left_tracked and right_tracked):
                self._clear_live_poses()
                if state.phase is not HandDreamsPhase.LIVE:
                    self._reset_alignment()
                continue

            if state.phase is not HandDreamsPhase.LIVE:
                if not state.config.auto_start:
                    if not self._aligned(
                        wrist, now=now, opened=left_open and right_open
                    ):
                        continue
                if not (left_open and right_open):
                    continue
                basis = (
                    state.alignment.participant_basis
                    if state.alignment is not None
                    else None
                )
                self._calibrate(wrist, participant_basis=basis)
                self._clear_live_poses()
                self._append_live_pose(wrist)
                state.phase = HandDreamsPhase.LIVE
                state.calibration_since = None
                state.calibration_progress = 0.0
                state.calibration_hand = None
                state.alignment = None
                state.alignment_preview = None
                state.alignment_since = None
                print(
                    "[handdreams] controllers calibrated; starting demo",
                    flush=True,
                )
                continue

            self._append_live_pose(wrist)

    def _check_alignment_history(self, events: list[ControllerFrameEvent]) -> None:
        """Reject interrupted holds without anchoring to queued setup movement."""
        state = self.state
        assert state.pipeline is not None
        for event in events:
            left, right = event.left, event.right
            assert left is not None and right is not None
            if not (left.tracked and right.tracked):
                self._reset_alignment()
            elif state.alignment is None:
                if state.calibration_since is not None and not (
                    state.input_is_emulated and state.calibration_progress >= 1.0
                ):
                    wrist = state.pipeline.transform(controller_frame(event))
                    if not self._calibration_pose_ready(wrist):
                        state.calibration_since = None
                        state.calibration_progress = 0.0
                        state.calibration_hand = None
            elif (
                left.trigger > state.config.open_claw_threshold
                or right.trigger > state.config.open_claw_threshold
            ):
                state.alignment_since = None
            else:
                wrist = state.pipeline.transform(controller_frame(event))
                if not state.alignment.observe(wrist).aligned:
                    state.alignment_since = None

    def _calibration_pose_ready(self, wrist: dict[str, Any]) -> bool:
        state = self.state
        held = tuple(
            side
            for side in ("left", "right")
            if wrist[side]["trigger"] >= CALIBRATION_TRIGGER_THRESHOLD
        )
        if state.input_is_emulated:
            if state.calibration_hand is None:
                state.calibration_hand = next(iter(held), None)
            closed = (
                state.calibration_hand is not None and state.calibration_hand in held
            )
        else:
            closed = len(held) == 2
        if not closed:
            return False
        try:
            controller_separation_basis(wrist)
        except ValueError:
            # Overlapping controllers cannot define a horizontal basis.
            return False
        return True

    def _aligned(self, wrist: dict[str, Any], *, now: float, opened: bool) -> bool:
        state = self.state
        if state.alignment is None:
            state.phase = HandDreamsPhase.CALIBRATING
            if state.input_is_emulated and state.calibration_progress >= 1.0:
                try:
                    controller_separation_basis(wrist)
                except ValueError:
                    self._reset_alignment()
                    return False
                return opened
            if not self._calibration_pose_ready(wrist):
                state.calibration_since = None
                state.calibration_progress = 0.0
                state.calibration_hand = None
                return False
            if state.calibration_since is None:
                state.calibration_since = now
            state.calibration_progress = min(
                1.0, (now - state.calibration_since) / _CALIBRATION_HOLD_S
            )
            if state.calibration_progress < 1.0:
                return False
            if state.input_is_emulated:
                return False
            state.alignment = AlignmentGuide(
                wrist,
                state.scene.wrist_poses,
            )
            state.calibration_since = None
            state.phase = HandDreamsPhase.ALIGNING
            state.alignment_preview = state.alignment.observe(wrist)
            return False
        state.phase = HandDreamsPhase.ALIGNING
        state.alignment_preview = state.alignment.observe(wrist)
        if not opened or not state.alignment_preview.aligned:
            state.alignment_since = None
            return False
        if state.alignment_since is None:
            state.alignment_since = now
        return now - state.alignment_since >= _ALIGNMENT_HOLD_S

    def _reset_alignment(self) -> None:
        state = self.state
        state.calibration_since = None
        state.calibration_progress = 0.0
        state.calibration_hand = None
        state.alignment = None
        state.alignment_preview = None
        state.controller_calibration = None
        state.alignment_since = None
        state.phase = HandDreamsPhase.WAITING

    def _clear_live_poses(self) -> None:
        state = self.state
        state.poses.clear()
        if state.control_sampler is not None:
            state.control_sampler.clear()
        state.next_control_ns = None
        state.last_control_ns = None
        state.last_observed_ns = None

    def _append_live_pose(self, frame: dict[str, Any]) -> None:
        """Select original observations on the configured source-time grid."""
        state = self.state
        capture_ns = int(frame["client_capture_ns"])
        stale_ns = (
            _MAX_RESAMPLE_GAP_NS
            if state.config.stale_ms is None
            else int(state.config.stale_ms * 1_000_000)
        )
        if (
            state.last_observed_ns is not None
            and capture_ns - state.last_observed_ns > stale_ns
        ):
            self._clear_live_poses()
        if state.control_sampler is None:
            state.control_sampler = ControlWindowSampler(rate_hz=state.control_fps)
        sampler = state.control_sampler
        if not sampler.add(frame):
            return
        state.last_observed_ns = capture_ns
        if state.next_control_ns is None:
            state.poses.append(frame)
            state.last_control_ns = capture_ns
            state.next_control_ns = capture_ns + sampler.interval_ns
            return
        while state.next_control_ns <= capture_ns:
            pose = sampler.sample_at(state.next_control_ns)
            state.next_control_ns += sampler.interval_ns
            if pose is None:
                continue
            selected_ns = int(pose["client_capture_ns"])
            if state.last_control_ns is None or selected_ns > state.last_control_ns:
                state.poses.append(pose)
                state.last_control_ns = selected_ns

    def _handle_tracking_loss(self) -> None:
        state = self.state
        state.last_wrist_observed_at = None
        self._clear_live_poses()
        if state.phase is not HandDreamsPhase.LIVE:
            self._reset_alignment()

    def _expire_stale_input(self, *, now: float) -> None:
        state = self.state
        stale_ms = state.config.stale_ms
        if state.phase is not HandDreamsPhase.LIVE:
            stale_ms = min(500.0, stale_ms) if stale_ms is not None else 500.0
        if (
            state.last_wrist_observed_at is not None
            and stale_ms is not None
            and now - state.last_wrist_observed_at > stale_ms / 1000.0
        ):
            self._handle_tracking_loss()

    def _calibrate(
        self, neutral: dict[str, Any], *, participant_basis: Any | None = None
    ) -> None:
        state = self.state
        if state.pipeline is None:
            raise RuntimeError("live HandDreams input requires a Teleop pipeline")
        state.pipeline.calibrate(
            neutral,
            scene_gripper_open=state.scene.gripper_open,
            position_gain=state.config.position_gain,
            grip_threshold=state.config.grip_threshold,
            stale_ms=state.config.stale_ms,
            gripper_slew_per_action=state.config.gripper_slew_per_action,
            participant_basis=participant_basis,
        )
        state.controller_calibration = ControllerPreviewCalibration.capture(
            neutral,
            getattr(state.scene, "wrist_poses", None),
            participant_basis=participant_basis,
            position_gain=state.config.position_gain,
        )

    def _publish_status(self) -> None:
        state = self.state
        phase = state.phase
        scene = "" if phase is HandDreamsPhase.SELECTING else str(state.scene.name)
        catalog = state.scene_catalog
        alignment = state.alignment_preview
        controller_calibration = state.controller_calibration
        calibration_progress = state.calibration_progress
        is_emulated = bool(state.input_is_emulated)
        ui_loop = state.ui_loop
        if ui_loop is not None:
            try:
                invoke_async(
                    ui_loop,
                    lambda ui_state: ui_state.update_status(
                        phase=phase,
                        current_scene=scene,
                        scene_catalog=catalog,
                        alignment=alignment,
                        controller_calibration=controller_calibration,
                        calibration_progress=calibration_progress,
                        is_emulated=is_emulated,
                    ),
                )
            except RuntimeError:
                pass

    def is_finished(self) -> bool:
        state = self.state
        if state.steps_generated >= state.config.total_steps:
            return True
        target = state.target_frames
        if target is not None and state.frames_generated >= target:
            return True
        return False

    def reset(self) -> None:
        """Reset inference and calibration without closing controller input."""
        state = self.state
        selecting = state.phase is HandDreamsPhase.SELECTING
        if state.inference_cache is not None:
            state.builder.release_cache(state.inference_cache)
            state.inference_cache = None
        if state.postprocess_stream is not None:
            state.postprocess_stream.reset()
        state.last_input_frame_id = -1
        state.last_input_capture_ns = -1
        state.input_is_emulated = None
        self._clear_live_poses()
        if state.pipeline is not None:
            state.pipeline.reset()
        state.command_sequence = 0
        state.steps_generated = 0
        state.frames_generated = 0
        state.rows_consumed = 0
        state.last_wrist_observed_at = None
        self._reset_alignment()
        if selecting:
            state.phase = HandDreamsPhase.SELECTING
        self._publish_status()

    def close(self) -> None:
        state = self.state
        if state.closed:
            return
        state.closed = True
        # The runtime wires UI -> model loop, while status publishing wires
        # model state -> UI. Break that cycle before any fallible cleanup so a
        # replaced session releases its scene tensors deterministically rather
        # than waiting for cyclic GC.
        state.ui_loop = None
        postprocess_stream, state.postprocess_stream = state.postprocess_stream, None
        if state.inference_cache is not None:
            state.builder.release_cache(state.inference_cache)
            state.inference_cache = None
        state.last_input_frame_id = -1
        state.last_input_capture_ns = -1
        self._reset_alignment()
        self._clear_live_poses()
        # The application owns and retains the bridge across session handoff.
        if state.pipeline is not None:
            state.pipeline.reset()
        if postprocess_stream is not None:
            postprocess_stream.finish()


class HandDreamsSession(ISession):
    """One scene/cache sharing app-owned inference and controller pipelines."""

    def __init__(
        self,
        *,
        builder: HandDreamsRolloutBuilder,
        config: HandDreamsLaunchConfig,
        session_desc: SessionDesc,
        scene_catalog: tuple[str, ...],
        pipeline: TeleopControllerPipeline | None,
        postprocess_stream: VideoPostprocessStream | None = None,
        scene_previews: tuple[ScenePreview, ...] = (),
        stage_width: int | None = None,
        stage_height: int | None = None,
        require_scene_selection: bool = False,
    ) -> None:
        self._builder = builder
        self._config = config
        self._session_desc = session_desc
        self._scene_catalog = scene_catalog
        self._pipeline = pipeline
        self._postprocess_stream = postprocess_stream
        self._require_scene_selection = require_scene_selection
        source_height, source_width = np.asarray(builder.scene.init_rgb).shape[:2]
        self._stage_width = stage_width or int(source_width)
        self._stage_height = stage_height or int(source_height)
        self._scene_previews = tuple(
            ScenePreview(preview.name, preview.title, builder.scene.init_rgb)
            if preview.name == builder.scene.name and preview.thumbnail is None
            else preview
            for preview in scene_previews
        )

    @property
    def session_desc(self) -> SessionDesc:
        return self._session_desc

    def init(self) -> None:
        if (
            not self._config.auto_start
            and getattr(self._builder.scene, "wrist_poses", None) is None
        ):
            raise ValueError(
                "Visual alignment requires the scene's initial wrist poses; "
                "supply pose metadata or use --auto-start for a scene without it."
            )
        if self._pipeline is not None:
            self._pipeline.reset()
        phase = (
            HandDreamsPhase.SELECTING
            if self._require_scene_selection
            else HandDreamsPhase.WAITING
        )
        ui_loop = self.register_ui_loop(
            HandDreamsUILoop,
            state=HandDreamsUIState(
                status=HandDreamsUIStatus(
                    phase=phase,
                    current_scene=(
                        ""
                        if phase is HandDreamsPhase.SELECTING
                        else str(self._builder.scene.name)
                    ),
                    scene_catalog=self._scene_catalog,
                ),
                scene_previews=self._scene_previews,
                stage_width=self._stage_width,
                stage_height=self._stage_height,
                show_postprocess_toggle=self._postprocess_stream is not None,
                postprocess_enabled=self._postprocess_stream is not None,
                open_claw_threshold=self._config.open_claw_threshold,
            ),
        )
        assert isinstance(ui_loop, HandDreamsUILoop)
        self.register_model_loop(
            HandDreamsModelLoop,
            state=HandDreamsModelState(
                builder=self._builder,
                config=self._config,
                session_desc=self._session_desc,
                pipeline=self._pipeline,
                postprocess_stream=self._postprocess_stream,
                postprocess_enabled=self._postprocess_stream is not None,
                stage_width=self._stage_width,
                stage_height=self._stage_height,
                scene_catalog=self._scene_catalog,
                ui_loop=ui_loop,
                phase=phase,
            ),
        )


def _resize_raw_for_presentation(frames: Tensor, *, height: int, width: int) -> Tensor:
    """Bilinearly scale raw TCHW frames to the postprocessed output contract."""
    if frames.shape[-2:] == (height, width):
        return frames
    return F.interpolate(
        frames,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )


def _numeric_metrics(value: object) -> dict[str, float | int]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(name): metric
        for name, metric in value.items()
        if isinstance(metric, int | float) and not isinstance(metric, bool)
    }


__all__ = [
    "HandDreamsModelLoop",
    "HandDreamsModelState",
    "HandDreamsRolloutBuilder",
    "HandDreamsSession",
]

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from omnidreams.interactive_drive.types import PresentedFrame, SceneBundle
from PIL import Image

RECORDING_HOTKEY_COLLISIONS = frozenset({"r", "x", "1", "2"})


@dataclass(frozen=True)
class RecordingConfig:
    enabled: bool = False
    output_dir: Path | None = None
    hotkey: str = "f9"
    auto_start: bool = False
    max_buffer_frames: int = 600


@dataclass
class _RecordingSession:
    output_dir: Path
    start_time_utc: str
    hdmap_frames: list[np.ndarray]
    inferred_frames: list[np.ndarray]
    dropped_hdmap_frames: int = 0
    dropped_inferred_frames: int = 0
    frame_drop_warning_emitted: bool = False


def normalize_recording_hotkey(raw: object) -> str:
    """Normalize manifest/browser key names into one small vocabulary."""
    value = str(raw).strip()
    if not value:
        raise ValueError("recording_hotkey must be a non-empty key name")
    aliases = {
        " ": "space",
        "spacebar": "space",
        "arrowup": "up",
        "arrow_up": "up",
        "arrowdown": "down",
        "arrow_down": "down",
        "arrowleft": "left",
        "arrow_left": "left",
        "arrowright": "right",
        "arrow_right": "right",
    }
    lowered = value.lower()
    return aliases.get(lowered, lowered)


def recording_hotkey_matches(configured: str, key: object) -> bool:
    """Return whether ``key`` names the configured recording hotkey."""
    try:
        return normalize_recording_hotkey(key) == normalize_recording_hotkey(configured)
    except ValueError:
        return False


def recording_hotkey_collides_with_controls(hotkey: str) -> bool:
    return normalize_recording_hotkey(hotkey) in RECORDING_HOTKEY_COLLISIONS


def slangpy_key_name_candidates(hotkey: str) -> tuple[str, ...]:
    """Map a normalized hotkey to likely SlangPy ``KeyCode`` attribute names."""
    key = normalize_recording_hotkey(hotkey)
    if len(key) == 1 and key.isdigit():
        return (f"key{key}", f"digit{key}", f"num_{key}")
    return (key,)


class InteractiveDriveRecorder:
    """Collect a rollout recording and write its artifact bundle on stop."""

    def __init__(
        self,
        config: RecordingConfig,
        *,
        scene: SceneBundle,
        fps: int,
    ) -> None:
        if not config.enabled:
            raise ValueError("InteractiveDriveRecorder requires recording.enabled")
        if config.output_dir is None:
            raise ValueError("InteractiveDriveRecorder requires an output directory")
        if config.max_buffer_frames <= 0:
            raise ValueError("RecordingConfig.max_buffer_frames must be positive")
        self._config = config
        self._scene = scene
        self._fps = int(fps)
        self._active_session: _RecordingSession | None = None
        self._session_count = 0
        self._closed_paths: list[Path] = []

    @property
    def is_recording(self) -> bool:
        return self._active_session is not None

    @property
    def closed_paths(self) -> tuple[Path, ...]:
        return tuple(self._closed_paths)

    @property
    def auto_start(self) -> bool:
        return self._config.auto_start

    def toggle(self) -> None:
        if self.is_recording:
            self.stop(reason="hotkey")
            return
        self.start()

    def start(self) -> None:
        if self.is_recording:
            return
        self._session_count += 1
        start_time = datetime.now(timezone.utc)
        output_dir = self._next_output_dir(start_time)
        try:
            output_dir.mkdir(parents=True, exist_ok=False)
            (output_dir / "prompt.txt").write_text(
                self._scene.prompt,
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning(
                "[recording] failed to start dir={} error={!r}",
                output_dir,
                exc,
            )
            return
        self._active_session = _RecordingSession(
            output_dir=output_dir,
            start_time_utc=start_time.isoformat(),
            hdmap_frames=[],
            inferred_frames=[],
        )
        print(f"[recording] started dir={output_dir}", flush=True)

    def stop(self, *, reason: str) -> Path | None:
        session = self._active_session
        if session is None:
            return None
        self._active_session = None
        metadata = {
            "scene_id": self._scene.scene_id,
            "scene_path": str(self._scene.scene_path),
            "fps": self._fps,
            "start_time_utc": session.start_time_utc,
            "stop_time_utc": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "hdmap_frames": len(session.hdmap_frames),
            "inferred_frames": len(session.inferred_frames),
            "dropped_hdmap_frames": session.dropped_hdmap_frames,
            "dropped_inferred_frames": session.dropped_inferred_frames,
            "max_buffer_frames": self._config.max_buffer_frames,
        }
        try:
            (session.output_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if session.inferred_frames:
                Image.fromarray(session.inferred_frames[0]).save(
                    session.output_dir / "first_frame.png",
                )
            _write_video(
                session.hdmap_frames,
                session.output_dir / "hdmap.mp4",
                self._fps,
            )
            _write_video(
                session.inferred_frames,
                session.output_dir / "inferred.mp4",
                self._fps,
            )
        except Exception as exc:
            logger.warning(
                "[recording] failed to save dir={} reason={} error={!r}",
                session.output_dir,
                reason,
                exc,
            )
            return None
        self._closed_paths.append(session.output_dir)
        print(
            "[recording] saved "
            f"dir={session.output_dir} "
            f"hdmap_frames={len(session.hdmap_frames)} "
            f"inferred_frames={len(session.inferred_frames)}",
            flush=True,
        )
        return session.output_dir

    def record_frame(self, frame: PresentedFrame) -> None:
        session = self._active_session
        if session is None:
            return
        self._append_frame(
            session,
            buffer=session.hdmap_frames,
            frame=_as_rgb_host_uint8(frame.rgb_host_uint8),
            stream="hdmap",
        )
        if frame.model_rgb_host_uint8 is not None:
            inferred_frame = _as_rgb_host_uint8(frame.model_rgb_host_uint8)
            self._append_frame(
                session,
                buffer=session.inferred_frames,
                frame=inferred_frame,
                stream="inferred",
            )

    def close(self, *, reason: str) -> None:
        if self.is_recording:
            self.stop(reason=reason)

    def _next_output_dir(self, start_time: datetime) -> Path:
        root = self._config.output_dir
        if root is None:
            raise ValueError("Recording output_dir is required")
        timestamp = start_time.strftime("%Y%m%d-%H%M%S")
        scene_slug = _slugify(self._scene.scene_id or self._scene.scene_path.stem)
        base = root / f"{timestamp}-{scene_slug}-{self._session_count:03d}"
        candidate = base
        suffix = 0
        while candidate.exists():
            suffix += 1
            candidate = root / f"{base.name}-{suffix:02d}"
        return candidate

    def _append_frame(
        self,
        session: _RecordingSession,
        *,
        buffer: list[np.ndarray],
        frame: np.ndarray,
        stream: str,
    ) -> None:
        if len(buffer) >= self._config.max_buffer_frames:
            buffer.pop(0)
            if stream == "hdmap":
                session.dropped_hdmap_frames += 1
            elif stream == "inferred":
                session.dropped_inferred_frames += 1
            if not session.frame_drop_warning_emitted:
                logger.warning(
                    "[recording] frame buffer reached max_buffer_frames={}; "
                    "dropping oldest frames for dir={}",
                    self._config.max_buffer_frames,
                    session.output_dir,
                )
                session.frame_drop_warning_emitted = True
        buffer.append(frame)


def _as_rgb_host_uint8(value: Any) -> np.ndarray:
    if hasattr(value, "to_numpy"):
        array = value.to_numpy()
    else:
        array = np.asarray(value)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(f"Expected an HxWx3 RGB frame, got shape {array.shape!r}")
    rgb = array[:, :, :3]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb.copy())


def _write_video(frames: list[np.ndarray], path: Path, fps: int) -> None:
    if not frames:
        return
    try:
        import mediapy as media  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - import-time gate
        raise ImportError(
            "Writing interactive-drive recordings needs mediapy. "
            "Install the flashdreams-omnidreams package dependencies."
        ) from exc

    media.write_video(str(path), np.stack(frames, axis=0), fps=fps)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-._")
    return slug or "scene"

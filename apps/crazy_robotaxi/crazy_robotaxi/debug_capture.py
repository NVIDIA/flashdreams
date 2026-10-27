# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded recording of model observations independently of presentation."""

from __future__ import annotations

import json
import math
import threading
from collections import deque
from pathlib import Path
from queue import Full, Queue
from typing import Any

import numpy as np
import torch
from omnidreams_game_engine.math3d import quaternion_to_matrix_xyzw
from PIL import Image, ImageDraw, ImageFont

from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.video_encoder import Mp4Encoder, result_to_rgb24_tensor
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout


def render_physics_view(
    record: dict[str, Any],
    width: int,
    height: int,
    trail_xy: list[tuple[float, float]],
) -> np.ndarray:
    """Draw a world-aligned overhead view of measured bounds and curb response.

    Use the pre-policy physics pose rather than the ground-aligned camera pose.
    White outlines show the game's contact bounds; blue outlines show the native
    inset chassis. The curb-response flag includes the game's proximity rebound.
    """
    colliders = record["physics_colliders"]
    position = np.asarray(colliders["ego_position_m"], dtype=float)
    scale = min(width, height) / 32.0
    image = Image.new("RGB", (width, height), (18, 22, 30))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(12, min(width, height) / 32))

    def pixels(points: Any) -> list[tuple[float, float]]:
        xy = np.asarray(points)[..., :2] - position[:2]
        return list(
            zip(width / 2 + xy[..., 0] * scale, height / 2 - xy[..., 1] * scale)
        )

    for axis in range(2):
        extent = (width if axis == 0 else height) / scale / 2
        for coordinate in np.arange(
            math.floor((position[axis] - extent) / 2) * 2,
            position[axis] + extent,
            2,
        ):
            endpoints = np.repeat(position[None, :], 2, axis=0)
            endpoints[:, axis] = coordinate
            endpoints[:, 1 - axis] += [-1000, 1000]
            draw.line(pixels(endpoints), fill=(32, 38, 48))
    segments = np.asarray(colliders["barrier_segments_xy_m"])
    for segment, thickness in zip(
        segments, colliders["barrier_thicknesses_m"], strict=True
    ):
        draw.line(
            pixels(segment), fill=(0, 220, 235), width=max(2, round(thickness * scale))
        )
    if len(trail_xy) > 1:
        draw.line(pixels(trail_xy), fill=(130, 130, 145), width=2)

    def box(
        center: Any, quaternion: Any, dimensions: Any, color: tuple[int, int, int]
    ) -> None:
        rotation = quaternion_to_matrix_xyzw(np.asarray(quaternion).tolist())
        signs = np.asarray(
            [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        )
        corners = signs * np.asarray(dimensions) / 2 @ rotation.T + center
        projected = pixels(corners)
        for i in range(8):
            for bit in (1, 2, 4):
                if not i & bit:
                    draw.line([projected[i], projected[i | bit]], fill=color, width=2)

    for center, quaternion, dimensions in zip(
        colliders["actor_positions_m"],
        colliders["actor_orientations_xyzw"],
        colliders["actor_dimensions_lwh"],
        strict=True,
    ):
        box(center, quaternion, dimensions, (255, 185, 70))
    orientation = np.asarray(colliders["ego_orientation_xyzw"])
    rotation = quaternion_to_matrix_xyzw(orientation.tolist())
    box(position, orientation, colliders["ego_dimensions_lwh"], (245, 245, 245))
    if colliders["ego_chassis_dimensions_lwh"] is not None:
        center = position + rotation @ np.asarray(colliders["ego_chassis_offset_m"])
        box(
            center, orientation, colliders["ego_chassis_dimensions_lwh"], (70, 150, 255)
        )
    draw.line(
        pixels([position, position + rotation[:, 0] * 4]), fill=(110, 240, 110), width=3
    )
    physical = record["physics_vehicle"]
    contact = record["static_barrier_collision"]
    color = (255, 90, 70) if contact else (245, 245, 245)
    draw.rectangle((0, 0, width, min(height, 66)), fill=(18, 22, 30))
    draw.text(
        (8, 8),
        f"PHYSICS  frame {record['frame']}  {record['frame'] / 30:.2f}s"
        f"  {physical['speed_mps']:.1f} m/s  CURB RESPONSE: {'YES' if contact else 'no'}",
        fill=color,
        font=font,
    )
    draw.text(
        (8, 26),
        f"z {physical['z_m']:.3f} m  roll {math.degrees(physical['roll_rad']):.1f} deg"
        f"  pitch {math.degrees(physical['pitch_rad']):.1f} deg",
        fill=(245, 245, 245),
        font=font,
    )
    draw.text(
        (8, 44),
        "Cyan: curbs | White: contact bounds | Blue: chassis | Green: heading",
        fill=(245, 245, 245),
        font=font,
    )
    return np.asarray(image)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize debug value {type(value).__name__}")


class CaptureCoverage:
    """Track ordered saved IDs and compact inclusive ranges of missing IDs."""

    def __init__(self) -> None:
        self.produced = 0
        self.saved = 0
        self._next = 0
        self._missing: list[list[int]] = []

    def written(self, identifier: int) -> None:
        """Account for one saved item after its write succeeds."""
        if identifier < self._next:
            raise ValueError("Capture IDs must be strictly increasing")
        if identifier > self._next:
            self._missing.append([self._next, identifier - 1])
        self._next = identifier + 1
        self.saved += 1

    def summary(self) -> dict[str, Any]:
        """Return counts and gaps, including an unwritten tail."""
        missing = list(self._missing)
        if self._next < self.produced:
            missing.append([self._next, self.produced - 1])
        return {
            "produced": self.produced,
            "saved": self.saved,
            "missing_ranges": missing,
        }


class CaptureRecorder:
    """Write observations and optional images on a bounded background worker.

    Producers never wait for disk or encoding. Image slots bound retained tensor
    copies separately from the observation queue. Overflow makes the manifest
    incomplete; strict captures also fail at shutdown. CUDA copies are made on
    the producer stream, with ``StepResult`` carrying readiness to the writer.
    """

    def __init__(
        self,
        directory: Path,
        desc: SessionDesc,
        *,
        images: bool,
        snapshot_every: int,
        queue_size: int = 256,
        image_slots: int = 4,
    ) -> None:
        self.directory = directory
        self.desc = desc
        self.images = images
        self.snapshot_every = snapshot_every
        self.observations = CaptureCoverage()
        self.media = CaptureCoverage()
        self.presentations = CaptureCoverage()
        self.presentation_media = CaptureCoverage()
        self.contact_frames: list[int] = []
        self.errors: list[str] = []
        self._failed_kinds: set[str] = set()
        self.generated_frames = 0
        self.finalized = False
        self.termination_reason: str | None = None
        self._queue: Queue[Any] = Queue(maxsize=queue_size)
        self._image_slots = {
            "model": threading.BoundedSemaphore(image_slots),
            "presentation": threading.BoundedSemaphore(60),
        }
        self._encoders: dict[str, Mp4Encoder] = {}
        self._last_frames: dict[str, tuple[int, np.ndarray]] = {}
        self._trail: deque[tuple[float, float]] = deque(maxlen=90)
        self._epoch: int | None = None
        self._logs = {
            name: (directory / f"{name}.jsonl").open("x", encoding="utf-8")
            for name in ("telemetry", "presentation", "media")
        }
        self._thread = threading.Thread(target=self._run, name="robotaxi-capture")
        self._thread.start()

    def observe(self, records: tuple[dict[str, Any], ...]) -> None:
        """Enqueue simulation observations before generation begins."""
        for record in records:
            record["capture_frame"] = self.observations.produced
            self.observations.produced += 1
        if self.images:
            self.media.produced += len(records)
        self._submit("observations", records, ())

    def submit(
        self,
        records: tuple[dict[str, Any], ...],
        video: torch.Tensor,
        hdmap: torch.Tensor,
    ) -> None:
        """Enqueue generated images before presentation can discard them."""
        self.generated_frames += len(records)
        if self.images:
            self._submit("model", records, (video, hdmap))

    def present(self, record: dict[str, Any], output: torch.Tensor | None) -> None:
        """Record a UI tick and its source frame without waiting for encoding."""
        record["frame"] = self.presentations.produced
        self.presentations.produced += 1
        if self.images and output is not None:
            self.presentation_media.produced += 1
        self._submit("presentation", (record,), () if output is None else (output,))

    def _submit(self, kind: str, records: tuple[dict, ...], tensors: tuple) -> None:
        outputs = ()
        slots = self._image_slots.get(kind)
        if (
            self.images
            and tensors
            and slots is not None
            and slots.acquire(blocking=False)
        ):
            try:
                outputs = tuple(self._snapshot(t) for t in tensors)
            except BaseException:
                assert slots is not None
                slots.release()
                raise
        try:
            self._queue.put_nowait((kind, records, outputs))
        except Full:
            if outputs:
                assert slots is not None
                slots.release()

    def _snapshot(self, tensor: torch.Tensor) -> StepResult:
        # Conversion allocates owned RGB bytes on the producer stream. Retaining
        # float model chunks or UI targets would triple the queue's GPU footprint.
        source = StepResult(
            step_index=0,
            output=tensor.detach(),
            frame_count=len(tensor),
            output_layout=VideoTensorLayout.tchw,
        )
        rgb = result_to_rgb24_tensor(
            source, self.desc, (tensor.shape[-1], tensor.shape[-2])
        )
        return StepResult(
            step_index=0,
            output=rgb.permute(0, 3, 1, 2),
            frame_count=len(tensor),
            output_layout=VideoTensorLayout.tchw,
        )

    def _log(self, name: str, record: dict) -> None:
        log = self._logs[name]
        log.write(json.dumps(record, default=_json_value, allow_nan=False) + "\n")
        log.flush()

    def _pixels(self, result: StepResult) -> np.ndarray:
        output = result.read_output()
        return output.permute(0, 2, 3, 1).cpu().numpy()

    def _save(
        self, name: str, frame: np.ndarray, identifier: int, contact=False
    ) -> None:
        if name not in self._encoders:
            (self.directory / "frames" / name).mkdir(parents=True)
            self._encoders[name] = Mp4Encoder(
                self.directory / f"{name}.mp4",
                width=frame.shape[1],
                height=frame.shape[0],
                frames_per_second=(
                    self.desc.frames_per_second_for_ui
                    if name == "presentation"
                    else self.desc.frames_per_second_for_step
                ),
            )
        self._encoders[name].write(frame[None])
        self._last_frames[name] = identifier, frame.copy()
        if identifier % self.snapshot_every == 0 or contact:
            image = Image.fromarray(frame)
            image.save(self.directory / "frames" / name / f"{identifier:06d}.png")
            if name == "presentation":
                temporary = self.directory / "latest.tmp.png"
                image.save(temporary)
                temporary.replace(self.directory / "latest.png")

    def _write(self, kind: str, records: tuple[dict, ...], outputs: tuple) -> None:
        if kind == "presentation":
            record = records[0]
            self._log("presentation", record)
            self.presentations.written(record["frame"])
            if outputs:
                self._save("presentation", self._pixels(outputs[0])[0], record["frame"])
                self._log(
                    "media",
                    {
                        "view": "presentation",
                        "video_frame": self.presentation_media.saved,
                        "presentation_frame": record["frame"],
                    },
                )
                self.presentation_media.written(record["frame"])
            return
        if kind == "observations":
            for record in records:
                self._log("telemetry", record)
                self.observations.written(record["capture_frame"])
                physical = record["physics_vehicle"]
                if (
                    record["static_barrier_collision"]
                    or record["actor_collision"]
                    or physical["ragdoll_active"]
                ):
                    self.contact_frames.append(record["capture_frame"])
            return
        if not outputs:
            return
        generated, hdmap = (self._pixels(output) for output in outputs)
        for record, video_frame, map_frame in zip(
            records, generated, hdmap, strict=True
        ):
            identifier = record["capture_frame"]
            if record["rollout_epoch"] != self._epoch:
                self._trail.clear()
                self._epoch = record["rollout_epoch"]
            physical = record["physics_vehicle"]
            self._trail.append((physical["x_m"], physical["y_m"]))
            contact = (
                record["static_barrier_collision"]
                or record["actor_collision"]
                or physical["ragdoll_active"]
            )
            physics = render_physics_view(
                record, video_frame.shape[1], video_frame.shape[0], list(self._trail)
            )
            for name, frame in (
                ("generated", video_frame),
                ("hdmap", map_frame),
                ("physics", physics),
            ):
                self._save(name, frame, identifier, contact)
            self._log(
                "media",
                {
                    "view": "model",
                    "video_frame": self.media.saved,
                    "capture_frame": identifier,
                    "rollout_epoch": record["rollout_epoch"],
                    "frame": record["frame"],
                },
            )
            self.media.written(identifier)

    def _run(self) -> None:
        while (item := self._queue.get()) is not None:
            kind, records, outputs = item
            try:
                if kind not in self._failed_kinds:
                    self._write(kind, records, outputs)
            except Exception as error:
                self.errors.append(f"{type(error).__name__}: {error}")
                self._failed_kinds.add(kind)
            finally:
                if outputs:
                    self._image_slots[kind].release()
        for encoder in self._encoders.values():
            try:
                encoder.close()
            except Exception as error:
                self.errors.append(f"{type(error).__name__}: {error}")
        try:
            for name, (identifier, frame) in self._last_frames.items():
                path = self.directory / "frames" / name / f"{identifier:06d}.png"
                if not path.exists():
                    Image.fromarray(frame).save(path)
        except Exception as error:
            self.errors.append(f"{type(error).__name__}: {error}")
        finally:
            for log in self._logs.values():
                try:
                    log.close()
                except Exception as error:
                    self.errors.append(f"{type(error).__name__}: {error}")

    def close(self) -> None:
        """Drain accepted work after both runtime producers have stopped."""
        if not self.finalized:
            self._queue.put(None)
            self._thread.join()
            self.finalized = True

    def summary(self) -> dict[str, Any]:
        """Report completeness independently of successful file finalization."""
        coverage = {
            "observations": self.observations.summary(),
            "model_images": self.media.summary(),
            "presentations": self.presentations.summary(),
            "presentation_images": self.presentation_media.summary(),
        }
        return {
            "coverage": coverage,
            "capture_complete": self.finalized
            and not self.errors
            and all(not stream["missing_ranges"] for stream in coverage.values()),
            "capture_errors": self.errors,
            "capture_images": self.images,
            "contact_frames": self.contact_frames,
            "views": list(self._encoders),
            "frames_written": self.observations.saved,
            "frames_generated": self.generated_frames,
            "finalized": self.finalized and not self.errors,
        }

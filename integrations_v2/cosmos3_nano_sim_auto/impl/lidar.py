# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Timestamped LiDAR controls and point-cloud presentation for multiview Auto."""

from __future__ import annotations

import math
import traceback
from collections.abc import Iterator
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from cosmos_interactive.lidar import LidarDecoder, LidarEncoder, LidarTokenizer
from torch import Tensor

from flashdreams.core.visualization.lidar import LidarView, LidarViewport


@dataclass(frozen=True)
class LidarSample:
    """Recorded metric HD-map ranges on the camera source clock, plus calibration."""

    controls: Tensor
    timestamps: Tensor
    ray_directions: Tensor
    sensor_to_ego: Tensor
    condition: Tensor | None = None

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        start_seconds: float,
        duration_seconds: float,
        fps: float = 10.0,
    ):
        """Read an aligned, pickle-free NPZ without synthesizing missing sweeps."""
        if (
            not all(
                math.isfinite(value) for value in (start_seconds, duration_seconds, fps)
            )
            or min(start_seconds, duration_seconds) < 0
            or fps <= 0
        ):
            raise ValueError(
                "LiDAR source times must be finite and nonnegative, with positive fps"
            )
        with np.load(path, allow_pickle=False) as data:
            required = {"controls", "timestamps", "ray_directions", "sensor_to_ego"}
            if missing := required - set(data.files):
                raise ValueError(f"LiDAR sample is missing {sorted(missing)}")
            controls = torch.from_numpy(data["controls"].copy()).float()
            timestamps = torch.from_numpy(data["timestamps"].copy()).double()
            rays = torch.from_numpy(data["ray_directions"].copy()).float()
            pose = torch.from_numpy(data["sensor_to_ego"].copy()).float()
            condition = (
                torch.from_numpy(data["condition"].copy()).float()
                if "condition" in data
                else None
            )
        if controls.ndim != 4 or tuple(controls.shape[1:]) != (3, 128, 1800):
            raise ValueError(
                "LiDAR controls must be [sweeps,3,128,1800] metric range/intensity/validity"
            )
        if (
            timestamps.shape != (len(controls),)
            or not len(timestamps)
            or not torch.isfinite(timestamps).all()
        ):
            raise ValueError(
                "LiDAR timestamps must be one finite source time per sweep"
            )
        if not torch.allclose(
            timestamps[1:] - timestamps[:-1],
            torch.full_like(timestamps[1:], 1 / fps),
            atol=1e-4,
            rtol=0,
        ):
            raise ValueError(
                "LiDAR timestamps must increase at 10 Hz without missing sweeps"
            )
        offset = int(torch.argmin((timestamps - start_seconds).abs()))
        if abs(float(timestamps[offset]) - start_seconds) > 1e-4:
            raise ValueError(
                "LiDAR opening sweep is not aligned with the RGB source start"
            )
        count = int(np.floor(duration_seconds * fps + 1e-5)) + 1
        if offset + count > len(controls):
            raise ValueError("LiDAR controls end before the requested camera clip")
        if (
            rays.shape != (128, 1800, 3)
            or not torch.isfinite(rays).all()
            or not torch.allclose(
                rays.norm(dim=-1), torch.ones((128, 1800)), atol=1e-3, rtol=0
            )
        ):
            raise ValueError(
                "LiDAR rays must be calibrated unit directions [128,1800,3]"
            )
        if (
            pose.shape != (4, 4)
            or not torch.isfinite(pose).all()
            or not torch.allclose(pose[3], torch.tensor([0.0, 0.0, 0.0, 1.0]))
            or not torch.allclose(
                pose[:3, :3].T @ pose[:3, :3], torch.eye(3), atol=1e-4
            )
            or not torch.isclose(
                torch.linalg.det(pose[:3, :3]), torch.tensor(1.0), atol=1e-4
            )
        ):
            raise ValueError("sensor_to_ego must be a rigid 4x4 transform in metres")
        if condition is not None:
            if offset != 0:
                raise ValueError(
                    "Initial LiDAR condition requires the sample's original start time"
                )
            if condition.shape != (3, 128, 1800) or not torch.isfinite(condition).all():
                raise ValueError("LiDAR condition must be [3,128,1800]")
        controls = controls[offset : offset + count]
        if (
            not torch.isfinite(controls).all()
            or (controls[:, 0] < 0).any()
            or ((controls[:, 1:] < 0) | (controls[:, 1:] > 1)).any()
        ):
            raise ValueError(
                "LiDAR requires finite metric ranges and unit intensity/validity"
            )
        return cls(
            controls,
            timestamps[offset : offset + count] - start_seconds,
            rays,
            pose,
            condition,
        )


CHUNKS_AHEAD = 8
"""Recorded LiDAR chunks encoded ahead of the rollout; see :meth:`LidarStream.preencode`."""


@dataclass(frozen=True)
class LidarAssets:
    """Prepared tokenizer weights and sample, reusable across session resets."""

    tokenizer: LidarTokenizer
    sample: LidarSample
    checkpoint: str | None = None
    """Where the tokenizer came from, so a decode worker can load its own copy."""
    version: Literal["v1p1", "v1p2"] = "v1p2"

    def stream(self):
        return LidarStream(self)


class LidarDecodeWorker:
    """Decode each chunk's sweeps in a process of its own, one chunk in flight.

    The LiDAR decoder is a few hundred small kernels, and on rank 0 the host is
    the bottleneck: issuing them costs the model thread as much as a quarter of
    a chunk even when the GPU work is hidden on a side stream. A process has its
    own interpreter and, with ``--prefetch-controls-gpu-offset``, a GPU that is
    idle most of the chunk. Same decoder, same cache order, same numbers; the
    model thread pays a 2 MB send and an 11 MB receive.
    """

    def __init__(self, conn: Connection, *, device: torch.device) -> None:
        self._process: BaseProcess | None = None
        self._conn = conn
        self._device = device
        self._pending = False

    @classmethod
    def spawn(
        cls,
        checkpoint: str,
        version: Literal["v1p1", "v1p2"],
        worker_device: int,
        *,
        device: torch.device,
    ) -> LidarDecodeWorker:
        """Start the process on ``worker_device`` and wait until its tokenizer is loaded."""
        ctx = torch.multiprocessing.get_context("spawn")
        ours, theirs = ctx.Pipe()
        process = ctx.Process(
            target=_serve_decode,
            args=(theirs, checkpoint, version, worker_device),
            daemon=True,
        )
        process.start()
        # Only the child may hold its end, or a child that dies hard leaves the
        # pipe open and the recv below waits for ever.
        theirs.close()
        try:
            ready = ours.recv()
        except EOFError as died:
            raise RuntimeError(
                f"the LiDAR decode worker exited with code {process.exitcode} before it was ready"
            ) from died
        if isinstance(ready, BaseException):
            raise RuntimeError("the LiDAR decode worker failed to start") from ready
        worker = cls(ours, device=device)
        worker._process = process
        return worker

    def request(self, latent: Tensor) -> None:
        """Start decoding these sweeps now; :meth:`take` collects them."""
        if self._pending:
            raise ValueError("a LiDAR chunk is already in flight; take it first")
        self._conn.send(latent.to("cpu"))
        self._pending = True

    def take(self) -> Tensor:
        """The decoded sweeps of the chunk in flight, on this rank's device.

        Raises:
            RuntimeError: the worker failed; its traceback is the message.
        """
        if not self._pending:
            raise ValueError("no LiDAR chunk is in flight")
        try:
            reply = self._conn.recv()
        except EOFError as died:
            process = getattr(self, "_process", None)
            code = None if process is None else process.exitcode
            raise RuntimeError(
                f"the LiDAR decode worker exited (code {code})"
            ) from died
        self._pending = False
        if isinstance(reply, str):
            raise RuntimeError(f"the LiDAR decode worker failed:\n{reply}")
        return reply.to(self._device, non_blocking=True)

    def close(self) -> None:
        """Stop the process: the pipe closing asks, and a terminate makes sure."""
        self._conn.close()
        process = getattr(self, "_process", None)
        if process is not None:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)


def _serve_decode(
    conn: Connection,
    checkpoint: str,
    version: Literal["v1p1", "v1p2"],
    worker_device: int,
) -> None:
    """The worker's whole life: load the tokenizer, then decode what the pipe sends."""
    try:
        torch.cuda.set_device(worker_device)
        device = torch.device("cuda", worker_device)
        decoder = LidarDecoder(
            LidarTokenizer.from_checkpoint(
                checkpoint, device=str(device), version=version
            )
        )
    except BaseException:  # noqa: BLE001 - reported to the parent, which raises
        conn.send(RuntimeError(traceback.format_exc()))
        return
    conn.send("ready")
    while True:
        try:
            latent = conn.recv()
        except EOFError:
            return
        try:
            conn.send(decoder(latent.to(device)).to("cpu"))
        except BaseException:  # noqa: BLE001 - reported to the parent, which raises
            conn.send(traceback.format_exc())


class LidarStream:
    """Independent encode/decode state and last displayed sweep for one rollout."""

    def __init__(self, assets: LidarAssets):
        self.assets = assets
        self.encoder = LidarEncoder(assets.tokenizer)
        self.decoder = LidarDecoder(assets.tokenizer)
        self.next_sweep = 0
        self.previous_range: Tensor | None = None
        self.previous_time: float | None = None
        self._ready: dict[tuple[int, int], Tensor] = {}
        self._ahead: Iterator[tuple[int, int]] = iter(())
        self._worker: LidarDecodeWorker | None = None

    def decode_in_worker(self, *, worker_device: int, device: torch.device) -> None:
        """Hand every chunk's decode to a :class:`LidarDecodeWorker` on ``worker_device``."""
        if self.assets.checkpoint is None:
            raise ValueError(
                "A LiDAR decode worker needs the tokenizer checkpoint path"
            )
        self._worker = LidarDecodeWorker.spawn(
            self.assets.checkpoint, self.assets.version, worker_device, device=device
        )

    def close(self) -> None:
        if self._worker is not None:
            self._worker.close()
            self._worker = None

    def decode_ahead(self, latent: Tensor) -> None:
        """With a worker, start decoding this chunk's sweeps now; ``ranges`` collects them.

        The joint rollout calls this as soon as the sweeps are denoised, so the
        worker has the clean-cache pass and the RGB decode to finish in. Without
        a worker nothing happens and ``ranges`` decodes on this rank, as before.
        """
        if self._worker is not None:
            self._worker.request(latent)

    def ranges(self, latent: Tensor) -> Tensor:
        """This chunk's decoded sweeps: from the worker if there is one, else decoded now."""
        if self._worker is not None:
            return self._worker.take()
        return self.decoder(latent)

    def preencode(self, ranges) -> None:
        """Encode the recorded controls ahead of the rollout, in the chunks ``tokens`` will be asked for.

        The same encoder calls in the same order, so the tokens are the ones a
        chunk would have computed; it just no longer waits for them. Only
        ``CHUNKS_AHEAD`` chunks are kept, parked on the host, and each chunk
        taken encodes the next, so start-up time and memory do not grow with the
        clip. Stops at the end of the recording. ponytail: a live LiDAR provider
        cannot run ahead like this; it would prefetch one chunk the way the RGB
        controls do.
        """
        self._ahead = iter(ranges)
        self._encode_ahead()

    def _encode_ahead(self) -> None:
        available = len(self.assets.sample.controls)
        while len(self._ready) < CHUNKS_AHEAD:
            start, end = next(self._ahead, (0, available + 1))
            if end > available:
                self._ahead = iter(())
                return
            self._ready[(start, end)] = self.tokens(start, end).cpu()

    def tokens(self, start: int, end: int) -> Tensor:
        if (start, end) in self._ready:
            result = self._ready.pop((start, end))
            self._encode_ahead()
            return result
        if start != self.next_sweep or not start < end <= len(
            self.assets.sample.controls
        ):
            raise ValueError(
                "LiDAR controls must be consumed once, consecutively, within the recorded clip"
            )
        clip = self.assets.sample.controls[start:end].permute(1, 0, 2, 3)[None]
        result = self.assets.tokenizer.patchify(self.encoder(clip))
        self.next_sweep = end
        return result

    def condition_tokens(self):
        condition = self.assets.sample.condition
        if condition is None:
            return None
        # Observations and map controls must never share a temporal encoder cache.
        return self.assets.tokenizer.patchify(
            LidarEncoder(self.assets.tokenizer)(condition[None, :, None])
        )

    def sweep_times(self, first_sweep: int, count: int, device: torch.device) -> Tensor:
        """Return source times for decoded sweeps on the presentation device."""
        return self.assets.sample.timestamps[first_sweep : first_sweep + count].to(
            device
        )

    def frames(
        self,
        ranges: Tensor,
        *,
        first_sweep: int,
        pixel_start: int,
        pixel_count: int,
        fps: float,
        viewport: LidarViewport,
        view: LidarView = LidarView.POINT_CLOUD,
    ) -> Tensor:
        """Hold the most recent sweep at camera cadence, never a future sweep."""
        sample = self.assets.sample
        sweeps = ranges[0].permute(1, 0, 2, 3)
        times = self.sweep_times(first_sweep, len(sweeps), ranges.device)
        if self.previous_range is not None:
            sweeps = torch.cat((self.previous_range[None], sweeps))
            times = torch.cat((times.new_tensor([self.previous_time]), times))
        rendered = torch.stack(
            [
                viewport.render_sweep(
                    frame,
                    sample.ray_directions,
                    sensor_to_ego=sample.sensor_to_ego,
                    view=view,
                )
                for frame in sweeps
            ]
        )
        wanted = (
            torch.arange(
                pixel_start, pixel_start + pixel_count, device=rendered.device
            ).double()
            / fps
        )
        indexes = torch.searchsorted(times, wanted + 1e-4, right=True) - 1
        if (indexes < 0).any():
            raise ValueError(
                "No LiDAR sweep exists at or before the first camera frame"
            )
        self.previous_range, self.previous_time = sweeps[-1].clone(), float(times[-1])
        return rendered[indexes].contiguous()

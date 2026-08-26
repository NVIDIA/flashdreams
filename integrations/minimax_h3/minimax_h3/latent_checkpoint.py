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

"""Atomic, application-owned checkpoints for MiniMax H3 joint denoising."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor

from minimax_h3.model import MiniMaxH3DenoiseProgress

_FORMAT = "minimax-h3-joint-denoise-v1"
_JOB_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_IMMUTABLE_REVISION = re.compile(r"[0-9a-f]{40,64}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_WORKFLOWS = frozenset(("t2va", "fl2va", "ref2va"))


def _strict_positive_integer(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity:
        raise RuntimeError(f"Asset changed while it was being fingerprinted: {path}")
    return digest.hexdigest()


def _checkpoint_tensor(value: Tensor) -> Tensor:
    return value.detach().to("cpu").contiguous().clone()


def _tensor_sha256(value: Tensor) -> str:
    raw = value.contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _tensor_manifest(value: Tensor) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "sha256": _tensor_sha256(value),
    }


@dataclass(frozen=True, kw_only=True, slots=True)
class MiniMaxH3AssetIdentity:
    """Content-addressed local input or immutable Hub asset identity."""

    source: str
    resolved_revision: str | None = None
    sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("Asset source cannot be empty.")
        if self.resolved_revision is not None and not _IMMUTABLE_REVISION.fullmatch(
            self.resolved_revision
        ):
            raise ValueError(
                "resolved_revision must be an immutable 40- to 64-character "
                "lowercase hex ID."
            )
        if self.sha256 is not None and not _SHA256.fullmatch(self.sha256):
            raise ValueError("sha256 must be a lowercase 64-character hex digest.")
        if self.resolved_revision is None and self.sha256 is None:
            raise ValueError(
                "Asset identity requires an immutable revision or content SHA-256."
            )

    @classmethod
    def from_file(
        cls, path: Path, *, source: str | None = None
    ) -> MiniMaxH3AssetIdentity:
        """Hash a stable local file instead of trusting path metadata."""
        resolved = path.expanduser().resolve(strict=True)
        if not resolved.is_file():
            raise ValueError(f"Asset is not a regular file: {resolved}")
        return cls(
            source=str(resolved) if source is None else source,
            sha256=_sha256_file(resolved),
        )

    def manifest(self) -> dict[str, str | None]:
        """Return the canonical checkpoint representation."""
        return {
            "source": self.source,
            "resolved_revision": self.resolved_revision,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, kw_only=True, slots=True)
class MiniMaxH3CheckpointIdentity:
    """All request and model state that must match before H3 resume."""

    workflow: str
    prompt: str
    width: int
    height: int
    aligned_num_frames: int
    num_audio_latents: int
    seed: int
    num_inference_steps: int
    video_scheduler_shift: float
    audio_scheduler_shift: float
    model: MiniMaxH3AssetIdentity
    inputs: tuple[MiniMaxH3AssetIdentity, ...] = ()
    lora: MiniMaxH3AssetIdentity | None = None
    lora_scale: float = 1.0
    audio_sample_rate: int = 32_000
    audio_channels: int = 2

    def __post_init__(self) -> None:
        if self.workflow not in _WORKFLOWS:
            raise ValueError(f"Unsupported MiniMax H3 workflow: {self.workflow!r}")
        if not self.prompt.strip():
            raise ValueError("Checkpoint prompt cannot be empty.")
        for name, value in (
            ("width", self.width),
            ("height", self.height),
            ("aligned_num_frames", self.aligned_num_frames),
            ("num_audio_latents", self.num_audio_latents),
            ("num_inference_steps", self.num_inference_steps),
        ):
            _strict_positive_integer(name, value)
        if self.aligned_num_frames % 17 != 5:
            raise ValueError("aligned_num_frames must satisfy the H3 17n+5 grid.")
        if self.num_inference_steps < 2:
            raise ValueError("num_inference_steps must provide at least one update.")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer.")
        for name, value in (
            ("video_scheduler_shift", self.video_scheduler_shift),
            ("audio_scheduler_shift", self.audio_scheduler_shift),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a positive finite number.")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number.")
        if self.audio_sample_rate != 32_000 or type(self.audio_sample_rate) is not int:
            raise ValueError("MiniMax H3 checkpoints require a 32000 Hz audio rate.")
        if self.audio_channels != 2 or type(self.audio_channels) is not int:
            raise ValueError("MiniMax H3 checkpoints require exactly 2 audio channels.")
        if isinstance(self.lora_scale, bool) or not isinstance(
            self.lora_scale, (int, float)
        ):
            raise ValueError("lora_scale must be a finite number between 0 and 4.")
        if not math.isfinite(self.lora_scale) or not 0 <= self.lora_scale <= 4:
            raise ValueError("lora_scale must be a finite number between 0 and 4.")
        if self.lora is None and self.lora_scale != 1.0:
            raise ValueError("lora_scale must remain 1.0 when no LoRA is configured.")

    @property
    def update_count(self) -> int:
        """Return the number of Euler updates in the scheduler point grid."""
        return self.num_inference_steps - 1

    def manifest(self) -> dict[str, Any]:
        """Return canonical request, scheduler, and immutable asset identity."""
        return {
            "workflow": self.workflow,
            "prompt": self.prompt,
            "configuration": {
                "width": self.width,
                "height": self.height,
                "aligned_num_frames": self.aligned_num_frames,
                "num_audio_latents": self.num_audio_latents,
                "audio_sample_rate": self.audio_sample_rate,
                "audio_channels": self.audio_channels,
            },
            "seed": self.seed,
            "schedulers": {
                "video": {
                    "num_inference_steps": self.num_inference_steps,
                    "shift": float(self.video_scheduler_shift),
                },
                "audio": {
                    "num_inference_steps": self.num_inference_steps,
                    "shift": float(self.audio_scheduler_shift),
                },
            },
            "model": self.model.manifest(),
            "lora": (
                None
                if self.lora is None
                else {
                    **self.lora.manifest(),
                    "scale": float(self.lora_scale),
                }
            ),
            "inputs": [asset.manifest() for asset in self.inputs],
        }


@dataclass(frozen=True, kw_only=True, slots=True)
class MiniMaxH3LatentCheckpointStore:
    """Persist one atomic joint-latent record below ``work_dir/job_id``."""

    work_dir: Path
    job_id: str

    def __post_init__(self) -> None:
        if not _JOB_ID.fullmatch(self.job_id) or self.job_id in {".", ".."}:
            raise ValueError(
                "job_id must be 1-128 safe filename characters without traversal."
            )

    @property
    def path(self) -> Path:
        """Return the application-owned checkpoint path without creating it."""
        root = self.work_dir.expanduser().resolve()
        return root / self.job_id / "minimax_h3" / "joint_latents.safetensors"

    def save(
        self,
        identity: MiniMaxH3CheckpointIdentity,
        progress: MiniMaxH3DenoiseProgress,
    ) -> Path:
        """Atomically publish both synchronized streams and their manifest."""
        if progress.next_step > identity.update_count:
            raise ValueError("Checkpoint next_step exceeds the configured schedule.")
        tensors = {
            "video": _checkpoint_tensor(progress.video),
            "audio": _checkpoint_tensor(progress.audio),
        }
        manifest = {
            "format_version": 1,
            "request": identity.manifest(),
            "scheduler_state": {"next_step": progress.next_step},
            "tensors": {
                name: _tensor_manifest(value) for name, value in tensors.items()
            },
        }
        encoded_manifest = _canonical_json(manifest)
        metadata = {
            "format": _FORMAT,
            "manifest": encoded_manifest,
            "signature": hashlib.sha256(encoded_manifest.encode()).hexdigest(),
        }

        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            save_file(tensors, temporary, metadata=metadata)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def load(self, identity: MiniMaxH3CheckpointIdentity) -> MiniMaxH3DenoiseProgress:
        """Load a complete matching record or reject it without partial resume."""
        path = self.path
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            keys = set(handle.keys())
        if metadata.get("format") != _FORMAT:
            raise ValueError(f"Unsupported MiniMax H3 checkpoint format: {path}")
        encoded_manifest = metadata.get("manifest")
        if encoded_manifest is None:
            raise ValueError(f"MiniMax H3 checkpoint has no manifest: {path}")
        signature = hashlib.sha256(encoded_manifest.encode()).hexdigest()
        if metadata.get("signature") != signature:
            raise ValueError(f"MiniMax H3 checkpoint manifest is corrupted: {path}")
        try:
            manifest = json.loads(encoded_manifest)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"MiniMax H3 checkpoint manifest is invalid JSON: {path}"
            ) from exc
        if not isinstance(manifest, dict):
            raise ValueError(f"MiniMax H3 checkpoint manifest is invalid: {path}")
        if manifest.get("request") != identity.manifest():
            raise ValueError(
                f"MiniMax H3 checkpoint does not match this request: {path}"
            )
        if keys != {"video", "audio"}:
            raise ValueError(
                "MiniMax H3 checkpoint must contain exactly video and audio tensors."
            )

        tensors = load_file(path, device="cpu")
        expected_tensors = manifest.get("tensors")
        if not isinstance(expected_tensors, dict):
            raise ValueError(
                f"MiniMax H3 checkpoint tensor manifest is invalid: {path}"
            )
        for name, value in tensors.items():
            if expected_tensors.get(name) != _tensor_manifest(value):
                raise ValueError(
                    f"MiniMax H3 checkpoint {name} tensor does not match its manifest."
                )
        scheduler_state = manifest.get("scheduler_state")
        if not isinstance(scheduler_state, dict):
            raise ValueError(
                f"MiniMax H3 checkpoint scheduler state is invalid: {path}"
            )
        next_step = scheduler_state.get("next_step")
        progress = MiniMaxH3DenoiseProgress(
            video=tensors["video"],
            audio=tensors["audio"],
            next_step=next_step,
        )
        if progress.next_step > identity.update_count:
            raise ValueError("Checkpoint next_step exceeds the configured schedule.")
        return progress


__all__ = [
    "MiniMaxH3AssetIdentity",
    "MiniMaxH3CheckpointIdentity",
    "MiniMaxH3LatentCheckpointStore",
]

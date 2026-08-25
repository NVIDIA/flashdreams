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

"""CPU tests for atomic MiniMax H3 joint-latent checkpoints."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

import minimax_h3.latent_checkpoint as checkpoint_module
from minimax_h3.latent_checkpoint import (
    MiniMaxH3AssetIdentity,
    MiniMaxH3CheckpointIdentity,
    MiniMaxH3LatentCheckpointStore,
)
from minimax_h3.model import MiniMaxH3DenoiseProgress

pytestmark = pytest.mark.ci_cpu


def _identity(tmp_path: Path) -> MiniMaxH3CheckpointIdentity:
    input_path = tmp_path / "first-frame.png"
    input_path.write_bytes(b"content-addressed input")
    return MiniMaxH3CheckpointIdentity(
        workflow="fl2va",
        prompt="A small robot folds a map.",
        width=768,
        height=576,
        aligned_num_frames=124,
        num_audio_latents=200,
        seed=17,
        num_inference_steps=3,
        video_scheduler_shift=12.0,
        audio_scheduler_shift=3.0,
        model=MiniMaxH3AssetIdentity(
            source="MiniMaxAI/MiniMax-H3",
            resolved_revision="a" * 40,
        ),
        inputs=(
            MiniMaxH3AssetIdentity.from_file(
                input_path, source="first-frame:first-frame.png"
            ),
        ),
    )


def _progress() -> MiniMaxH3DenoiseProgress:
    return MiniMaxH3DenoiseProgress(
        video=torch.arange(8, dtype=torch.float32).reshape(2, 4),
        audio=torch.arange(18, dtype=torch.float32).reshape(6, 3),
        next_step=1,
    )


def _metadata(path: Path) -> dict[str, str]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return handle.metadata() or {}


def test_joint_checkpoint_round_trip_records_all_resume_identity(
    tmp_path: Path,
) -> None:
    """Store both streams, scheduler step, request, and immutable asset IDs."""
    store = MiniMaxH3LatentCheckpointStore(work_dir=tmp_path, job_id="job-17")
    identity = _identity(tmp_path)
    progress = _progress()
    expected_video = progress.video.clone()
    expected_audio = progress.audio.clone()

    path = store.save(identity, progress)
    progress.video.add_(100)
    progress.audio.add_(100)
    restored = store.load(identity)

    assert path == (
        tmp_path.resolve()
        / "job-17"
        / "minimax_h3"
        / "joint_latents.safetensors"
    )
    assert restored.next_step == 1
    torch.testing.assert_close(restored.video, expected_video)
    torch.testing.assert_close(restored.audio, expected_audio)
    manifest = json.loads(_metadata(path)["manifest"])
    assert manifest["scheduler_state"] == {"next_step": 1}
    assert manifest["request"]["seed"] == 17
    assert manifest["request"]["configuration"] == {
        "width": 768,
        "height": 576,
        "aligned_num_frames": 124,
        "num_audio_latents": 200,
        "audio_sample_rate": 32_000,
        "audio_channels": 2,
    }
    assert manifest["request"]["model"]["resolved_revision"] == "a" * 40
    assert manifest["request"]["inputs"][0]["sha256"] is not None


def test_joint_checkpoint_rejects_cross_request_and_cross_step_tensors(
    tmp_path: Path,
) -> None:
    """Bind both tensors to one request and one signed scheduler-step manifest."""
    store = MiniMaxH3LatentCheckpointStore(work_dir=tmp_path, job_id="job-17")
    identity = _identity(tmp_path)
    path = store.save(identity, _progress())

    with pytest.raises(ValueError, match="does not match this request"):
        store.load(replace(identity, prompt="A different request"))

    metadata = _metadata(path)
    tensors = load_file(path, device="cpu")
    tensors["audio"].add_(1)
    path.unlink()
    save_file(tensors, path, metadata=metadata)
    with pytest.raises(ValueError, match="audio tensor does not match"):
        store.load(identity)


def test_joint_checkpoint_rejects_partial_records(tmp_path: Path) -> None:
    """Never accept a video-only or audio-only resume record."""
    store = MiniMaxH3LatentCheckpointStore(work_dir=tmp_path, job_id="job-17")
    identity = _identity(tmp_path)
    path = store.save(identity, _progress())
    metadata = _metadata(path)
    video = load_file(path, device="cpu")["video"]
    path.unlink()
    save_file({"video": video}, path, metadata=metadata)

    with pytest.raises(ValueError, match="exactly video and audio"):
        store.load(identity)


def test_failed_checkpoint_write_preserves_previous_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Remove unique staging while leaving the prior checkpoint byte-exact."""
    store = MiniMaxH3LatentCheckpointStore(work_dir=tmp_path, job_id="job-17")
    target = store.path
    target.parent.mkdir(parents=True)
    target.write_bytes(b"previous complete checkpoint")

    def fail_save(
        tensors: dict[str, torch.Tensor],
        filename: Path,
        *,
        metadata: dict[str, str],
    ) -> None:
        del tensors, metadata
        filename.write_bytes(b"partial replacement")
        raise RuntimeError("injected checkpoint failure")

    monkeypatch.setattr(checkpoint_module, "save_file", fail_save)
    with pytest.raises(RuntimeError, match="injected checkpoint failure"):
        store.save(_identity(tmp_path), _progress())

    assert target.read_bytes() == b"previous complete checkpoint"
    assert not list(target.parent.glob(f".{target.name}.*.tmp"))


def test_failed_checkpoint_publication_preserves_previous_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep the prior record when the same-filesystem atomic replace fails."""
    store = MiniMaxH3LatentCheckpointStore(work_dir=tmp_path, job_id="job-17")
    target = store.path
    target.parent.mkdir(parents=True)
    target.write_bytes(b"previous complete checkpoint")

    def fail_replace(source: Path, destination: Path) -> None:
        assert source.parent == target.parent
        assert destination == target
        raise OSError("injected replace failure")

    monkeypatch.setattr(checkpoint_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        store.save(_identity(tmp_path), _progress())

    assert target.read_bytes() == b"previous complete checkpoint"
    assert not list(target.parent.glob(f".{target.name}.*.tmp"))


def test_checkpoint_assets_use_content_or_immutable_revision(tmp_path: Path) -> None:
    """Fingerprint local bytes and reject mutable model revision labels."""
    asset_path = tmp_path / "asset.bin"
    asset_path.write_bytes(b"first")
    first = MiniMaxH3AssetIdentity.from_file(asset_path)
    asset_path.write_bytes(b"other")
    second = MiniMaxH3AssetIdentity.from_file(asset_path)
    assert first.sha256 != second.sha256

    with pytest.raises(ValueError, match="immutable"):
        MiniMaxH3AssetIdentity(source="model", resolved_revision="main")
    with pytest.raises(ValueError, match="job_id"):
        MiniMaxH3LatentCheckpointStore(work_dir=tmp_path, job_id="../escape")

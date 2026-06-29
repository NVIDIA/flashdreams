# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manifest types and Hugging Face asset discovery helpers."""

from __future__ import annotations

import json
import posixpath
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

DEFAULT_DATASET_REPO = "nvidia/PhysicalAI-Autonomous-Vehicles-NuRec"
DEFAULT_DATASET_REVISION = "26.01"
DEFAULT_DATASET_SUBPATH = "sample_set/26.01_release"
DEFAULT_CAMERA = "camera_front_wide_120fov"
MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AssetRef:
    """Reference to one file in a remote dataset repository."""

    path: str
    size: int | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AssetRef":
        return cls(path=str(value["path"]), size=_optional_int(value.get("size")))


@dataclass(frozen=True)
class EvalCase:
    """One model-ready OmniDreams evaluation scene."""

    uuid: str
    camera: str
    dataset_repo: str
    dataset_revision: str
    dataset_subpath: str
    reference_video: AssetRef
    hdmap_video: AssetRef
    prompt: AssetRef

    @property
    def total_input_bytes(self) -> int:
        sizes = (
            self.reference_video.size,
            self.hdmap_video.size,
            self.prompt.size,
        )
        return sum(size for size in sizes if size is not None)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvalCase":
        return cls(
            uuid=str(value["uuid"]),
            camera=str(value["camera"]),
            dataset_repo=str(value["dataset_repo"]),
            dataset_revision=str(value["dataset_revision"]),
            dataset_subpath=str(value["dataset_subpath"]),
            reference_video=AssetRef.from_dict(value["reference_video"]),
            hdmap_video=AssetRef.from_dict(value["hdmap_video"]),
            prompt=AssetRef.from_dict(value["prompt"]),
        )


@dataclass(frozen=True)
class StagedCase:
    """Local paths for one downloaded/prepared evaluation case."""

    case: EvalCase
    reference_video_path: Path
    hdmap_video_path: Path
    prompt_path: Path
    first_frame_path: Path
    prompt_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case.to_dict(),
            "reference_video_path": str(self.reference_video_path),
            "hdmap_video_path": str(self.hdmap_video_path),
            "prompt_path": str(self.prompt_path),
            "first_frame_path": str(self.first_frame_path),
            "prompt_text": self.prompt_text,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StagedCase":
        return cls(
            case=EvalCase.from_dict(value["case"]),
            reference_video_path=Path(value["reference_video_path"]),
            hdmap_video_path=Path(value["hdmap_video_path"]),
            prompt_path=Path(value["prompt_path"]),
            first_frame_path=Path(value["first_frame_path"]),
            prompt_text=str(value["prompt_text"]),
        )


def build_cases_from_repo_files(
    files: Iterable[Any],
    *,
    dataset_repo: str = DEFAULT_DATASET_REPO,
    dataset_revision: str = DEFAULT_DATASET_REVISION,
    dataset_subpath: str = DEFAULT_DATASET_SUBPATH,
    camera: str = DEFAULT_CAMERA,
) -> list[EvalCase]:
    """Build cases from Hugging Face repo tree file entries.

    Supports the raw PAI-NuRec source layout under ``dataset_subpath``::

        <uuid>/<camera>_rgb.mp4
        <uuid>/<camera>_hdmap.mp4
        <uuid>/<camera>_prompt.txt
        <uuid>/<uuid>.prompt.txt

    and the fanned-out staged layout::

        data/video/<camera>/<uuid>.mp4
        data/hdmap/<camera>/<uuid>.mp4
        data/caption/<camera>/<uuid>.txt
    """

    base = _normalize_repo_path(dataset_subpath)
    rel_dirs = {
        "video": _join_repo_path(base, "data", "video", camera),
        "hdmap": _join_repo_path(base, "data", "hdmap", camera),
        "caption": _join_repo_path(base, "data", "caption", camera),
    }
    video: dict[str, AssetRef] = {}
    hdmap: dict[str, AssetRef] = {}
    caption: dict[str, AssetRef] = {}
    scene_caption: dict[str, AssetRef] = {}

    for info in files:
        path = _repo_file_path(info)
        if path is None:
            continue
        path = _normalize_repo_path(path)
        size = _repo_file_size(info)
        if _is_file_in_dir(path, rel_dirs["video"], suffix=".mp4"):
            video[Path(path).stem] = AssetRef(path=path, size=size)
        elif _is_file_in_dir(path, rel_dirs["hdmap"], suffix=".mp4"):
            hdmap[Path(path).stem] = AssetRef(path=path, size=size)
        elif _is_file_in_dir(path, rel_dirs["caption"], suffix=".txt"):
            caption[Path(path).stem] = AssetRef(path=path, size=size)
            continue

        source = _source_layout_match(path, base=base, camera=camera)
        if source is None:
            continue
        uuid, kind = source
        asset = AssetRef(path=path, size=size)
        if kind == "rgb":
            video[uuid] = asset
        elif kind == "hdmap":
            hdmap[uuid] = asset
        elif kind == "camera_prompt":
            caption[uuid] = asset
        elif kind == "scene_prompt":
            scene_caption[uuid] = asset

    uuids = sorted(set(video) & set(hdmap) & (set(caption) | set(scene_caption)))
    cases: list[EvalCase] = []
    for uuid in uuids:
        prompt = caption[uuid] if uuid in caption else scene_caption[uuid]
        cases.append(
            EvalCase(
                uuid=uuid,
                camera=camera,
                dataset_repo=dataset_repo,
                dataset_revision=dataset_revision,
                dataset_subpath=base,
                reference_video=video[uuid],
                hdmap_video=hdmap[uuid],
                prompt=prompt,
            )
        )
    return cases


def write_cases_jsonl(cases: Sequence[EvalCase], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "schema_version": MANIFEST_SCHEMA_VERSION,
                    "kind": "omnidreams_eval_manifest",
                },
                sort_keys=True,
            )
            + "\n"
        )
        for case in cases:
            f.write(json.dumps(case.to_dict(), sort_keys=True) + "\n")


def read_cases_jsonl(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            value = json.loads(line)
            if line_no == 1 and value.get("kind") == "omnidreams_eval_manifest":
                continue
            cases.append(EvalCase.from_dict(value))
    return cases


def write_staged_cases_jsonl(cases: Sequence[StagedCase], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "schema_version": MANIFEST_SCHEMA_VERSION,
                    "kind": "omnidreams_eval_staged_manifest",
                },
                sort_keys=True,
            )
            + "\n"
        )
        for case in cases:
            f.write(json.dumps(case.to_dict(), sort_keys=True) + "\n")


def read_staged_cases_jsonl(path: Path) -> list[StagedCase]:
    cases: list[StagedCase] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            value = json.loads(line)
            if line_no == 1 and value.get("kind") == "omnidreams_eval_staged_manifest":
                continue
            cases.append(StagedCase.from_dict(value))
    return cases


def _repo_file_path(info: Any) -> str | None:
    if isinstance(info, Mapping):
        value = info.get("path") or info.get("rfilename")
        return str(value) if value else None
    for attr in ("path", "rfilename"):
        value = getattr(info, attr, None)
        if value:
            return str(value)
    return None


def _repo_file_size(info: Any) -> int | None:
    value = (
        info.get("size") if isinstance(info, Mapping) else getattr(info, "size", None)
    )
    return _optional_int(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _normalize_repo_path(path: str) -> str:
    return posixpath.normpath(path.strip("/"))


def _join_repo_path(*parts: str) -> str:
    return _normalize_repo_path(posixpath.join(*(p for p in parts if p)))


def _is_file_in_dir(path: str, directory: str, *, suffix: str) -> bool:
    prefix = directory.rstrip("/") + "/"
    return path.startswith(prefix) and path.lower().endswith(suffix)


def _source_layout_match(
    path: str, *, base: str, camera: str
) -> tuple[str, str] | None:
    """Return ``(uuid, kind)`` for raw PAI-NuRec per-scene files."""

    prefix = base.rstrip("/") + "/" if base else ""
    if prefix and not path.startswith(prefix):
        return None
    rel = path[len(prefix) :] if prefix else path
    parts = rel.split("/")
    if len(parts) != 2:
        return None
    uuid, filename = parts
    legacy_prefix = f"{uuid}."
    tag = (
        filename[len(legacy_prefix) :]
        if filename.startswith(legacy_prefix)
        else filename
    )
    if tag == f"{camera}_rgb.mp4":
        return uuid, "rgb"
    if tag == f"{camera}_hdmap.mp4":
        return uuid, "hdmap"
    if tag == f"{camera}_prompt.txt":
        return uuid, "camera_prompt"
    if filename == f"{uuid}.prompt.txt":
        return uuid, "scene_prompt"
    return None

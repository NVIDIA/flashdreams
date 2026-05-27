#!/usr/bin/env python3
"""Prepare NuRec HF sample scenes for OmniDreams-style single-view sweeps.

The NVIDIA NuRec sample set contains large USDZ scene files plus camera videos,
HDMap condition videos, and prompts. This script lists the gated HF dataset,
selects only scene/camera entries that have all assets needed for generation,
and prepares a compact local layout:

    <output-root>/data/single_view/<scene-id>__<camera-name>/
        <camera-name>_hdmap.mp4
        first_frame.png
        prompt.txt
        source.json

That layout is accepted by ``scripts/generate_omnidreams_sweep_json.py``.

Example dry run:

    uv run python scripts/prepare_nurec_hf_dataset.py --dry-run

Example download front-wide samples only:

    uv run python scripts/prepare_nurec_hf_dataset.py \\
      --camera camera_front_wide_120fov \\
      --output-root ~/data/nurec-26.02-release
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
from huggingface_hub.hf_api import RepoFile


DEFAULT_REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles-NuRec"
DEFAULT_REVISION = "26.02_update"
DEFAULT_DATASET_PATH = "sample_set/26.02_release"
DEFAULT_OUTPUT_ROOT = Path("~/data/nurec-26.02-release")
DEFAULT_RGB_SUFFIX_PRIORITY = ("_rgb.mp4", ".mp4")
USDZ_SUFFIX = ".usdz"


@dataclass(frozen=True)
class Candidate:
    scene_id: str
    camera: str
    hdmap_path: str
    prompt_path: str
    rgb_path: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download NuRec scene assets needed for HDMap-conditioned video "
            "generation, excluding USDZ files."
        )
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Prepared dataset root. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--camera",
        action="append",
        default=[],
        help=(
            "Camera name to include. Repeat for multiple cameras. "
            "Default: include every qualifying camera."
        ),
    )
    parser.add_argument(
        "--limit-scenes",
        type=int,
        default=0,
        help="Limit the number of qualifying scene IDs prepared. 0 means no limit.",
    )
    parser.add_argument(
        "--limit-items",
        type=int,
        default=0,
        help=(
            "Limit the number of scene-camera items prepared after camera filtering. "
            "0 means no limit."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list/count qualifying scenes and files; do not download.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing prepared files.",
    )
    parser.add_argument(
        "--keep-rgb-video",
        action="store_true",
        help=(
            "Also copy the RGB source video into each prepared item directory. "
            "By default it is only used to extract first_frame.png."
        ),
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="ffmpeg executable used to extract first_frame.png.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help=(
            "Optional Hugging Face token. Defaults to the token from "
            "`huggingface-cli login` or HF_TOKEN."
        ),
    )
    parser.add_argument(
        "--manifest-name",
        default="manifest.json",
        help="Manifest filename written under --output-root.",
    )
    return parser.parse_args()


def _list_files(
    *,
    repo_id: str,
    repo_type: str,
    revision: str,
    dataset_path: str,
    token: str | None,
) -> list[str]:
    api = HfApi(token=token)
    try:
        entries = api.list_repo_tree(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            path_in_repo=dataset_path.strip("/"),
            recursive=True,
        )
        return sorted(entry.path for entry in entries if isinstance(entry, RepoFile))
    except GatedRepoError as exc:
        raise SystemExit(
            "The dataset is gated. Accept the dataset license on Hugging Face and "
            "authenticate with `huggingface-cli login` or pass --token."
        ) from exc
    except HfHubHTTPError as exc:
        raise SystemExit(f"Failed to list Hugging Face files: {exc}") from exc


def _group_scene_files(files: Iterable[str], dataset_path: str) -> dict[str, set[str]]:
    root = dataset_path.strip("/")
    prefix = root + "/"
    scene_files: dict[str, set[str]] = defaultdict(set)
    for path in files:
        if not path.startswith(prefix):
            continue
        rest = path[len(prefix) :]
        parts = rest.split("/", 1)
        if len(parts) != 2:
            # Top-level USDZ files live here; they are intentionally ignored.
            continue
        scene_id, filename = parts
        if filename.endswith(USDZ_SUFFIX):
            continue
        scene_files[scene_id].add(filename)
    return dict(scene_files)


def _camera_from_asset(filename: str) -> str | None:
    for suffix in ("_hdmap.mp4", "_prompt.txt", "_rgb.mp4"):
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
    if filename.endswith(".mp4") and not filename.endswith(("_hdmap.mp4", "_rgb.mp4")):
        return filename[:-4]
    return None


def _find_candidates(
    scene_files: dict[str, set[str]],
    *,
    dataset_path: str,
    cameras: set[str],
) -> list[Candidate]:
    candidates: list[Candidate] = []
    root = dataset_path.strip("/")
    for scene_id in sorted(scene_files):
        filenames = scene_files[scene_id]
        scene_cameras = sorted({
            camera for name in filenames if (camera := _camera_from_asset(name))
        })
        for camera in scene_cameras:
            if cameras and camera not in cameras:
                continue
            hdmap = f"{camera}_hdmap.mp4"
            prompt = f"{camera}_prompt.txt"
            rgb = next(
                (
                    f"{camera}{suffix}"
                    for suffix in DEFAULT_RGB_SUFFIX_PRIORITY
                    if f"{camera}{suffix}" in filenames
                ),
                None,
            )
            if hdmap in filenames and prompt in filenames and rgb is not None:
                scene_root = f"{root}/{scene_id}"
                candidates.append(
                    Candidate(
                        scene_id=scene_id,
                        camera=camera,
                        hdmap_path=f"{scene_root}/{hdmap}",
                        prompt_path=f"{scene_root}/{prompt}",
                        rgb_path=f"{scene_root}/{rgb}",
                    )
                )
    return candidates


def _limit_candidates(
    candidates: list[Candidate],
    *,
    limit_scenes: int,
    limit_items: int,
) -> list[Candidate]:
    limited = candidates
    if limit_scenes > 0:
        allowed_scenes = []
        seen = set()
        for item in limited:
            if item.scene_id in seen:
                continue
            seen.add(item.scene_id)
            allowed_scenes.append(item.scene_id)
            if len(allowed_scenes) >= limit_scenes:
                break
        allowed = set(allowed_scenes)
        limited = [item for item in limited if item.scene_id in allowed]
    if limit_items > 0:
        limited = limited[:limit_items]
    return limited


def _download_file(
    item_path: str,
    *,
    repo_id: str,
    repo_type: str,
    revision: str,
    token: str | None,
) -> Path:
    try:
        return Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                filename=item_path,
                token=token,
            )
        )
    except GatedRepoError as exc:
        raise SystemExit(
            "The dataset is gated. Accept the dataset license on Hugging Face and "
            "authenticate with `huggingface-cli login` or pass --token."
        ) from exc
    except HfHubHTTPError as exc:
        raise RuntimeError(f"Failed to download {item_path}: {exc}") from exc


def _copy_if_needed(src: Path, dst: Path, *, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return
    shutil.copy2(src, dst)


def _extract_first_frame(
    rgb_video: Path,
    dst: Path,
    *,
    ffmpeg_bin: str,
    overwrite: bool,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return
    with tempfile.TemporaryDirectory(prefix="nurec-first-frame-") as tmpdir:
        tmp = Path(tmpdir) / "first_frame.png"
        subprocess.run(
            [
                ffmpeg_bin,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(rgb_video),
                "-frames:v",
                "1",
                str(tmp),
            ],
            check=True,
        )
        shutil.move(str(tmp), dst)


def _prepared_item_dir(output_root: Path, item: Candidate) -> Path:
    sample_id = f"{item.scene_id}__{item.camera}"
    return output_root / "data" / "single_view" / sample_id


def _prepare_item(
    item: Candidate,
    *,
    repo_id: str,
    repo_type: str,
    revision: str,
    output_root: Path,
    token: str | None,
    ffmpeg_bin: str,
    overwrite: bool,
    keep_rgb_video: bool,
) -> dict[str, object]:
    item_dir = _prepared_item_dir(output_root, item)
    hdmap_dst = item_dir / f"{item.camera}_hdmap.mp4"
    prompt_dst = item_dir / "prompt.txt"
    first_frame_dst = item_dir / "first_frame.png"
    source_dst = item_dir / "source.json"

    hdmap_src = _download_file(
        item.hdmap_path,
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        token=token,
    )
    prompt_src = _download_file(
        item.prompt_path,
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        token=token,
    )
    rgb_src = _download_file(
        item.rgb_path,
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        token=token,
    )

    _copy_if_needed(hdmap_src, hdmap_dst, overwrite=overwrite)
    _copy_if_needed(prompt_src, prompt_dst, overwrite=overwrite)
    _extract_first_frame(
        rgb_src,
        first_frame_dst,
        ffmpeg_bin=ffmpeg_bin,
        overwrite=overwrite,
    )

    rgb_dst: Path | None = None
    if keep_rgb_video:
        rgb_dst = item_dir / Path(item.rgb_path).name
        _copy_if_needed(rgb_src, rgb_dst, overwrite=overwrite)

    source = {
        **asdict(item),
        "repo_id": repo_id,
        "revision": revision,
        "prepared_paths": {
            "sample_dir": str(item_dir),
            "hdmap_path": str(hdmap_dst),
            "prompt_path": str(prompt_dst),
            "first_frame_path": str(first_frame_dst),
            "rgb_path": str(rgb_dst) if rgb_dst else None,
        },
    }
    source_dst.write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    return source


def _print_summary(candidates: list[Candidate], *, total_scenes: int) -> None:
    scenes = sorted({item.scene_id for item in candidates})
    by_camera = Counter(item.camera for item in candidates)
    print(f"Total scene folders listed: {total_scenes}")
    print(f"Qualifying scene folders: {len(scenes)}")
    print(f"Qualifying scene-camera items: {len(candidates)}")
    print("Qualifying cameras:")
    for camera, count in sorted(by_camera.items()):
        print(f"  {camera}: {count}")
    if scenes:
        print("First qualifying scenes:")
        for scene_id in scenes[:10]:
            cams = ", ".join(
                item.camera for item in candidates if item.scene_id == scene_id
            )
            print(f"  {scene_id}: {cams}")


def main() -> int:
    args = _parse_args()
    repo_type = "dataset"
    files = _list_files(
        repo_id=args.repo_id,
        repo_type=repo_type,
        revision=args.revision,
        dataset_path=args.dataset_path,
        token=args.token,
    )
    scene_files = _group_scene_files(files, args.dataset_path)
    candidates = _find_candidates(
        scene_files,
        dataset_path=args.dataset_path,
        cameras=set(args.camera),
    )
    _print_summary(candidates, total_scenes=len(scene_files))

    selected = _limit_candidates(
        candidates,
        limit_scenes=args.limit_scenes,
        limit_items=args.limit_items,
    )
    if selected != candidates:
        print()
        print(
            f"Selected after limits/camera filtering: "
            f"{len({item.scene_id for item in selected})} scene(s), "
            f"{len(selected)} item(s)"
        )

    if args.dry_run:
        print()
        print("Dry run only; no files downloaded.")
        return 0

    if not selected:
        raise SystemExit("No qualifying scene-camera items selected.")

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    prepared = []
    for index, item in enumerate(selected, start=1):
        print(
            f"[{index}/{len(selected)}] {item.scene_id} {item.camera}",
            flush=True,
        )
        prepared.append(
            _prepare_item(
                item,
                repo_id=args.repo_id,
                repo_type=repo_type,
                revision=args.revision,
                output_root=output_root,
                token=args.token,
                ffmpeg_bin=args.ffmpeg_bin,
                overwrite=args.overwrite,
                keep_rgb_video=args.keep_rgb_video,
            )
        )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_id": args.repo_id,
        "revision": args.revision,
        "dataset_path": args.dataset_path,
        "output_root": str(output_root),
        "num_scene_folders_listed": len(scene_files),
        "num_qualifying_scene_folders": len({item.scene_id for item in candidates}),
        "num_qualifying_scene_camera_items": len(candidates),
        "num_prepared_scene_folders": len({item["scene_id"] for item in prepared}),
        "num_prepared_scene_camera_items": len(prepared),
        "items": prepared,
    }
    manifest_path = output_root / args.manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print()
    print(f"Wrote manifest: {manifest_path}")
    print(f"Prepared data root: {output_root / 'data' / 'single_view'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

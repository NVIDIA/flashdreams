# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DrivingGen checkout and layout adapter."""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from omnidreams.eval.manifest import StagedCase

DEFAULT_DRIVINGGEN_REPO = "https://github.com/youngzhou1999/DrivingGen.git"
DEFAULT_DRIVINGGEN_REVISION = "48ed35695855ef17d7a7cbd4adc0e8bd5fcc8223"
DEFAULT_I3D_TORCHSCRIPT_URL = (
    "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1"
)


@dataclass(frozen=True)
class DrivingGenCheckout:
    path: Path
    repo_url: str
    revision: str
    resolved_commit: str
    patches: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "repo_url": self.repo_url,
            "revision": self.revision,
            "resolved_commit": self.resolved_commit,
            "patches": list(self.patches),
        }


def ensure_drivinggen_checkout(
    *,
    cache_dir: Path,
    repo_url: str = DEFAULT_DRIVINGGEN_REPO,
    revision: str = DEFAULT_DRIVINGGEN_REVISION,
    fetch: bool = True,
    patch_checkout: bool = True,
) -> DrivingGenCheckout:
    """Clone/fetch DrivingGen and checkout ``revision``."""

    target = cache_dir / "DrivingGen"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not (target / ".git").exists():
        _run(["git", "clone", repo_url, str(target)], cwd=cache_dir)
    elif fetch:
        _run(["git", "fetch", "--tags", "origin"], cwd=target)
    _run(["git", "checkout", revision], cwd=target)
    commit = _run_capture(["git", "rev-parse", "HEAD"], cwd=target).strip()
    patches = patch_drivinggen_checkout(target) if patch_checkout else ()
    return DrivingGenCheckout(
        path=target,
        repo_url=repo_url,
        revision=revision,
        resolved_commit=commit,
        patches=patches,
    )


def patch_drivinggen_checkout(drivinggen_root: Path) -> tuple[str, ...]:
    """Patch cloned DrivingGen metric files for portable checkpoint paths."""

    applied: list[str] = []
    if _patch_detector_path(
        drivinggen_root
        / "third_parties/stylegan-v/src/metrics/frechet_video_distance.py",
        env_name="DRIVINGGEN_I3D_CKPT",
        default_path=DEFAULT_I3D_TORCHSCRIPT_URL,
        upstream_path="/shared_disk/users/yang.zhou/iclr_open_source/DrivingGen/ckpt/i3d_torchscript.pt",
    ):
        applied.append("fvd_checkpoint_env")
    if _patch_detector_path(
        drivinggen_root
        / "third_parties/stylegan-v/src/metrics/frechet_inception_distance.py",
        env_name="DRIVINGGEN_INCEPTION_CKPT",
        default_path="./ckpt/inception-2015-12-05.pkl",
        upstream_path=(
            "/shared_disk/users/yang.zhou/iclr_open_source/DrivingGen/ckpt/"
            "inception-2015-12-05.pkl"
        ),
    ):
        applied.append("fid_checkpoint_env")
    return tuple(applied)


def stage_drivinggen_video_inputs(
    staged_cases: Sequence[StagedCase],
    *,
    generated_root: Path,
    drivinggen_root: Path,
    split: str,
    model_name: str,
    exp_id: str,
    force: bool = False,
) -> None:
    """Create the file layout expected by DrivingGen video metrics."""

    scene_ids = [case.case.uuid for case in staged_cases]
    data_dir = drivinggen_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / f"{split}.json").write_text(
        json.dumps(scene_ids, indent=2) + "\n",
        encoding="utf-8",
    )

    for staged in staged_cases:
        uuid = staged.case.uuid
        generated_video = generated_root / uuid / "generated.mp4"
        if not generated_video.exists():
            raise FileNotFoundError(f"missing generated video: {generated_video}")

        generated_case_dir = (
            drivinggen_root
            / "cache"
            / "infer_results"
            / split
            / uuid
            / model_name
            / exp_id
        )
        reference_frame_dir = data_dir / "videos-fvd" / uuid
        generated_frame_dir = generated_case_dir / "images"
        generated_case_dir.mkdir(parents=True, exist_ok=True)
        _copy_or_link(generated_video, generated_case_dir / "video.mp4", force=force)
        generated_frame_count = decode_video_to_frames(
            generated_video,
            generated_frame_dir,
            force=force,
        )
        reference_frame_count = decode_video_to_frames(
            staged.reference_video_path,
            reference_frame_dir,
            force=force,
            max_frames=generated_frame_count,
        )
        _write_stage_metadata(
            generated_case_dir / "stage_metadata.json",
            uuid=uuid,
            generated_video=generated_video,
            reference_video=staged.reference_video_path,
            generated_frame_dir=generated_frame_dir,
            reference_frame_dir=reference_frame_dir,
            generated_frame_count=generated_frame_count,
            reference_frame_count=reference_frame_count,
        )


def video_metrics_command(
    *,
    drivinggen_root: Path,
    split: str,
    model_name: str,
    exp_id: str,
    metric: str = "fvd",
    python: str = "python",
) -> list[str]:
    """Return a DrivingGen video metric command with valid script arguments."""

    return [
        python,
        "drivinggen/z-sample_fvd.py",
        "--root_path",
        f"./cache/infer_results/{split}",
        "--outdir",
        f"./cache/eval_logs/{split}",
        "--gt_path",
        f"./data/{split}.json",
        "--model_name",
        model_name,
        "--exp_id",
        exp_id,
        "--metric",
        metric,
    ]


def run_video_metrics(
    *,
    drivinggen_root: Path,
    split: str,
    model_name: str,
    exp_id: str,
    metric: str = "fvd",
    python: str = "python",
    log_path: Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> list[str]:
    """Run DrivingGen video metrics, optionally capturing output to ``log_path``."""

    command = video_metrics_command(
        drivinggen_root=drivinggen_root,
        split=split,
        model_name=model_name,
        exp_id=exp_id,
        metric=metric,
        python=python,
    )
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    if log_path is None:
        subprocess.run(command, cwd=drivinggen_root, check=True, env=env)
        return command

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(command) + "\n\n")
        log_file.flush()
        subprocess.run(
            command,
            cwd=drivinggen_root,
            check=True,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    return command


def run_fvd_lite(
    *,
    drivinggen_root: Path,
    split: str,
    model_name: str,
    exp_id: str,
    log_path: Path | None = None,
    output_json: Path | None = None,
    force: bool = False,
) -> dict[str, object]:
    """Run DrivingGen's StyleGAN-V FVD path without importing all video metrics."""

    fake_root = stage_drivinggen_fvd_fake_frames(
        drivinggen_root=drivinggen_root,
        split=split,
        model_name=model_name,
        exp_id=exp_id,
        force=force,
    )
    reference_root = stage_drivinggen_fvd_reference_frames(
        drivinggen_root=drivinggen_root,
        split=split,
        force=force,
    )

    def _run() -> dict[str, object]:
        os.environ.setdefault("DRIVINGGEN_I3D_CKPT", DEFAULT_I3D_TORCHSCRIPT_URL)
        result = _calculate_styleganv_fvd(
            drivinggen_root=drivinggen_root,
            fake_root=fake_root,
            reference_root=reference_root,
        )
        metrics = _fvd_metrics_from_result(result)
        value = metrics.get("fvd2048_100f")
        payload = {
            "metric": "fvd2048_100f",
            "value": _jsonable(value),
            "results": _jsonable(metrics),
            "split": split,
            "model_name": model_name,
            "exp_id": exp_id,
            "fake_root": str(fake_root),
            "reference_root": str(reference_root),
            "i3d_checkpoint": os.environ.get("DRIVINGGEN_I3D_CKPT"),
        }
        if output_json is not None:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return payload

    if log_path is None:
        return _run()

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(
            "DrivingGen FVD-lite: StyleGAN-V fvd2048_100f\n"
            f"fake_root={fake_root}\n"
            f"reference_root={reference_root}\n\n"
        )
        log_file.flush()
        with redirect_stdout(log_file), redirect_stderr(log_file):
            return _run()


def run_fvd_reference_lite(
    *,
    drivinggen_root: Path,
    split_a: str,
    split_b: str,
    log_path: Path | None = None,
    output_json: Path | None = None,
    force: bool = False,
) -> dict[str, object]:
    """Run StyleGAN-V FVD between two DrivingGen reference splits."""

    reference_a_root = stage_drivinggen_fvd_reference_frames(
        drivinggen_root=drivinggen_root,
        split=split_a,
        force=force,
    )
    reference_b_root = stage_drivinggen_fvd_reference_frames(
        drivinggen_root=drivinggen_root,
        split=split_b,
        force=force,
    )

    def _run() -> dict[str, object]:
        os.environ.setdefault("DRIVINGGEN_I3D_CKPT", DEFAULT_I3D_TORCHSCRIPT_URL)
        result = _calculate_styleganv_fvd(
            drivinggen_root=drivinggen_root,
            reference_root=reference_a_root,
            fake_root=reference_b_root,
        )
        metrics = _fvd_metrics_from_result(result)
        value = metrics.get("fvd2048_100f")
        payload = {
            "metric": "fvd2048_100f",
            "value": _jsonable(value),
            "results": _jsonable(metrics),
            "split_a": split_a,
            "split_b": split_b,
            "reference_a_root": str(reference_a_root),
            "reference_b_root": str(reference_b_root),
            "i3d_checkpoint": os.environ.get("DRIVINGGEN_I3D_CKPT"),
        }
        if output_json is not None:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return payload

    if log_path is None:
        return _run()

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(
            "DrivingGen reference FVD-lite: StyleGAN-V fvd2048_100f\n"
            f"reference_a_root={reference_a_root}\n"
            f"reference_b_root={reference_b_root}\n\n"
        )
        log_file.flush()
        with redirect_stdout(log_file), redirect_stderr(log_file):
            return _run()


def stage_drivinggen_fvd_fake_frames(
    *,
    drivinggen_root: Path,
    split: str,
    model_name: str,
    exp_id: str,
    force: bool = False,
) -> Path:
    """Build DrivingGen's generated-frame FVD directory from infer results."""

    root_path = drivinggen_root / "cache" / "infer_results" / split
    if not root_path.exists():
        raise FileNotFoundError(f"missing DrivingGen infer results root: {root_path}")

    fake_root = (
        drivinggen_root / "cache" / "infer_results" / f"{split}+{model_name}_fvd"
    )
    fake_root.mkdir(parents=True, exist_ok=True)
    staged_count = 0
    for scene_dir in sorted(path for path in root_path.iterdir() if path.is_dir()):
        image_dir = scene_dir / model_name / exp_id / "images"
        if not image_dir.exists():
            continue
        frame_paths = sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        usable_frame_count = max(len(frame_paths) - 1, 0)
        if usable_frame_count < 100:
            raise RuntimeError(
                f"DrivingGen FVD requires at least 100 generated frames after "
                f"skipping the seed frame, found {usable_frame_count} usable "
                f"frames from {len(frame_paths)} total in {image_dir}"
            )
        output_dir = fake_root / f"{scene_dir.name}+{model_name}+{exp_id}"
        if output_dir.exists():
            if not force:
                staged_count += 1
                continue
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for index, frame in enumerate(frame_paths[1:], start=1):
            target = output_dir / f"{index:05d}{frame.suffix.lower()}"
            _copy_or_link(frame, target, force=force)
        staged_count += 1
    if staged_count == 0:
        raise RuntimeError(
            f"no generated frame directories found for split={split}, "
            f"model={model_name}, exp={exp_id}"
        )
    return fake_root


def stage_drivinggen_fvd_reference_frames(
    *,
    drivinggen_root: Path,
    split: str,
    force: bool = False,
) -> Path:
    """Build a split-specific reference-frame FVD directory.

    DrivingGen stores decoded reference frames in a global ``data/videos-fvd``
    directory. That directory accumulates as more batches are prepared, so FVD
    needs a split-specific view to avoid comparing a batch against references
    from previous batches.
    """

    split_path = drivinggen_root / "data" / f"{split}.json"
    if not split_path.exists():
        raise FileNotFoundError(f"missing DrivingGen split file: {split_path}")
    scene_ids = json.loads(split_path.read_text(encoding="utf-8"))
    if not isinstance(scene_ids, list) or not all(
        isinstance(item, str) for item in scene_ids
    ):
        raise ValueError(
            f"DrivingGen split file must contain a JSON string list: {split_path}"
        )

    source_root = drivinggen_root / "data" / "videos-fvd"
    if not source_root.exists():
        raise FileNotFoundError(
            f"missing DrivingGen reference frame root: {source_root}"
        )

    reference_root = (
        drivinggen_root / "cache" / "infer_results" / f"{split}+reference_fvd"
    )
    reference_root.mkdir(parents=True, exist_ok=True)
    staged_count = 0
    for uuid in scene_ids:
        source_dir = source_root / uuid
        if not source_dir.exists():
            raise FileNotFoundError(
                f"missing reference frames for {uuid}: {source_dir}"
            )
        frame_count = _count_frame_files(source_dir)
        if frame_count < 100:
            raise RuntimeError(
                f"DrivingGen FVD requires at least 100 reference frames, "
                f"found {frame_count} in {source_dir}"
            )
        _copy_or_link_directory(source_dir, reference_root / uuid, force=force)
        staged_count += 1
    if staged_count == 0:
        raise RuntimeError(f"no reference scenes found for split={split}")
    return reference_root


def decode_video_to_frames(
    video_path: Path,
    output_dir: Path,
    *,
    force: bool = False,
    max_frames: int | None = None,
) -> int:
    """Decode ``video_path`` to ``00000.png`` frames for DrivingGen."""

    if max_frames is not None and max_frames <= 0:
        raise ValueError(f"max_frames must be positive, got {max_frames}")

    existing_frames = _count_frame_files(output_dir)
    if existing_frames and not force:
        if max_frames is not None and existing_frames != max_frames:
            raise RuntimeError(
                f"existing frame directory {output_dir} has {existing_frames} frames, "
                f"expected {max_frames}; rerun with --force"
            )
        return existing_frames
    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import cv2  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "video frame extraction requires opencv-python-headless"
        ) from exc

    cap = cv2.VideoCapture(str(video_path))
    index = 0
    try:
        while True:
            if max_frames is not None and index >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            if not cv2.imwrite(str(output_dir / f"{index:05d}.png"), frame):
                raise RuntimeError(
                    f"failed to write decoded frame {index} from {video_path}"
                )
            index += 1
    finally:
        cap.release()
    if index == 0:
        raise RuntimeError(f"failed to decode any frames from {video_path}")
    if max_frames is not None and index < max_frames:
        raise RuntimeError(
            f"decoded only {index} frames from {video_path}, expected {max_frames}"
        )
    return index


def _write_stage_metadata(
    path: Path,
    *,
    uuid: str,
    generated_video: Path,
    reference_video: Path,
    generated_frame_dir: Path,
    reference_frame_dir: Path,
    generated_frame_count: int,
    reference_frame_count: int,
) -> None:
    path.write_text(
        json.dumps(
            {
                "uuid": uuid,
                "temporal_policy": "reference_first_n_frames_matching_generated",
                "generated_video": str(generated_video),
                "reference_video": str(reference_video),
                "generated_frame_dir": str(generated_frame_dir),
                "reference_frame_dir": str(reference_frame_dir),
                "generated_frame_count": generated_frame_count,
                "reference_frame_count": reference_frame_count,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _count_frame_files(output_dir: Path) -> int:
    if not output_dir.exists():
        return 0
    return sum(
        1
        for path in output_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )


def _copy_or_link(source: Path, target: Path, *, force: bool) -> None:
    if target.exists() or target.is_symlink():
        if not force:
            return
        target.unlink()
    try:
        target.symlink_to(
            os.path.relpath(source.resolve(), start=target.parent.resolve())
        )
    except OSError:
        shutil.copy2(source, target)


def _copy_or_link_directory(source: Path, target: Path, *, force: bool) -> None:
    if target.exists() or target.is_symlink():
        if not force:
            return
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    try:
        target.symlink_to(
            os.path.relpath(source.resolve(), start=target.parent.resolve()),
            target_is_directory=True,
        )
    except OSError:
        shutil.copytree(source, target)


def _calculate_styleganv_fvd(
    *,
    drivinggen_root: Path,
    fake_root: Path,
    reference_root: Path,
) -> dict[str, object]:
    stylegan_root = drivinggen_root / "third_parties" / "stylegan-v"
    if not stylegan_root.exists():
        raise FileNotFoundError(f"missing DrivingGen StyleGAN-V root: {stylegan_root}")

    _check_styleganv_fvd_imports()

    sys.path.insert(0, str(stylegan_root))
    old_cwd = Path.cwd()
    try:
        os.chdir(drivinggen_root)
        metric_module = importlib.import_module("src.scripts.calc_metrics_for_dataset")
        calc_metrics_ = getattr(metric_module, "calc_metrics_")

        results = calc_metrics_(
            metrics=["fvd2048_100f"],
            real_data_path=str(reference_root),
            fake_data_path=str(fake_root),
            mirror=False,
            resolution=256,
            gpus=1,
            verbose=True,
            use_cache=False,
            num_runs=1,
        )
    except ImportError as exc:
        raise ImportError(
            "DrivingGen FVD-lite requires torch, numpy, scipy, omegaconf, "
            "Pillow, and requests in the current Python environment. "
            f"Original import error: {exc!r}"
        ) from exc
    finally:
        os.chdir(old_cwd)
        try:
            sys.path.remove(str(stylegan_root))
        except ValueError:
            pass
    if not results:
        raise RuntimeError("StyleGAN-V FVD returned no results")
    first_result = results[0]
    if not isinstance(first_result, dict):
        raise RuntimeError(
            f"StyleGAN-V FVD returned unexpected result: {first_result!r}"
        )
    return {str(key): value for key, value in first_result.items()}


def _fvd_metrics_from_result(result: dict[str, object]) -> dict[str, object]:
    metrics = result.get("results")
    if isinstance(metrics, dict):
        return {str(key): value for key, value in metrics.items()}
    return dict(result)


def _check_styleganv_fvd_imports() -> None:
    required_modules = {
        "torch": "torch",
        "numpy": "numpy",
        "scipy.linalg": "scipy",
        "omegaconf": "omegaconf",
        "PIL.Image": "Pillow",
        "requests": "requests",
    }
    missing: list[str] = []
    for module_name, package_name in required_modules.items():
        try:
            importlib.import_module(module_name)
        except ImportError as exc:
            missing.append(f"{module_name} ({package_name}): {exc}")
    if missing:
        raise ImportError(
            "DrivingGen FVD-lite missing required Python modules: " + "; ".join(missing)
        )


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _patch_detector_path(
    path: Path,
    *,
    env_name: str,
    default_path: str,
    upstream_path: str,
) -> bool:
    if not path.exists():
        raise FileNotFoundError(f"DrivingGen metric file not found: {path}")

    text = path.read_text(encoding="utf-8-sig")
    replacement = f"detector_url = os.environ.get('{env_name}', '{default_path}')"
    if replacement in text:
        return False

    original = text
    text = _ensure_import_os(text)
    upstream_line = f"detector_url = '{upstream_path}'"
    if upstream_line not in text:
        raise RuntimeError(
            f"could not find upstream detector path in {path}; "
            "DrivingGen may have changed its metric implementation"
        )
    text = text.replace(upstream_line, replacement, 1)
    if text != original:
        path.write_text(text, encoding="utf-8")
    return text != original


def _ensure_import_os(text: str) -> str:
    if "\nimport os\n" in f"\n{text}\n":
        return text
    for marker in ("import copy\n", "import numpy as np\n"):
        if marker in text:
            return text.replace(marker, marker + "import os\n", 1)
    raise RuntimeError("could not find a suitable import block to add os import")


def _run(args: list[str], *, cwd: Path) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def _run_capture(args: list[str], *, cwd: Path) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True)

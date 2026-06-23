# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path

import omnidreams.eval.drivinggen as drivinggen
import omnidreams.eval.generation as generation
import omnidreams.eval.worldlens as worldlens
import pytest
from omnidreams.eval.batches import cases_for_batch, parse_byte_size, plan_batches
from omnidreams.eval.cli import DEFAULT_GENERATION_RECIPE, _build_parser
from omnidreams.eval.drivinggen import (
    DEFAULT_I3D_TORCHSCRIPT_URL,
    _check_styleganv_fvd_imports,
    decode_video_to_frames,
    patch_drivinggen_checkout,
    run_fvd_lite,
    run_fvd_reference_lite,
    run_video_metrics,
    stage_drivinggen_fvd_fake_frames,
    stage_drivinggen_fvd_reference_frames,
    video_metrics_command,
)
from omnidreams.eval.generation import generate_cases, generation_result_for_case
from omnidreams.eval.manifest import (
    AssetRef,
    EvalCase,
    StagedCase,
    build_cases_from_repo_files,
    read_cases_jsonl,
    read_staged_cases_jsonl,
    write_cases_jsonl,
    write_staged_cases_jsonl,
)
from omnidreams.eval.report import (
    build_run_summary,
    render_run_summary_markdown,
    write_run_summary_json,
    write_run_summary_markdown,
)
from omnidreams.eval.validation import validate_generated_run
from omnidreams.eval.worldlens import (
    DEFAULT_WORLDLENS_CONFIG_NAME,
    collect_worldlens_artifact_results,
    latest_worldlens_metric_results,
    run_worldlens_evaluation,
    stage_worldlens_video_inputs,
    worldlens_evaluate_command,
    write_worldlens_consistency_config,
)

pytestmark = pytest.mark.ci_cpu


def _repo_file(path: str, size: int) -> dict[str, object]:
    return {"path": path, "size": size}


def test_build_cases_from_repo_files_uses_asset_intersection() -> None:
    files = [
        _repo_file("sample_set/26.01_release/data/video/cam/a.mp4", 100),
        _repo_file("sample_set/26.01_release/data/hdmap/cam/a.mp4", 10),
        _repo_file("sample_set/26.01_release/data/caption/cam/a.txt", 1),
        _repo_file("sample_set/26.01_release/data/video/cam/missing_hdmap.mp4", 100),
        _repo_file("sample_set/26.01_release/data/hdmap/cam/missing_caption.mp4", 10),
        _repo_file("sample_set/26.01_release/data/caption/other/a.txt", 1),
    ]

    cases = build_cases_from_repo_files(
        files,
        dataset_repo="repo",
        dataset_revision="rev",
        dataset_subpath="sample_set/26.01_release",
        camera="cam",
    )

    assert [case.uuid for case in cases] == ["a"]
    assert cases[0].total_input_bytes == 111
    assert cases[0].reference_video.path.endswith("/video/cam/a.mp4")


def test_build_cases_from_raw_pai_nurec_layout() -> None:
    files = [
        _repo_file(
            "sample_set/26.01_release/scene-a/camera_front_wide_120fov_rgb.mp4", 100
        ),
        _repo_file(
            "sample_set/26.01_release/scene-a/camera_front_wide_120fov_hdmap.mp4", 10
        ),
        _repo_file(
            "sample_set/26.01_release/scene-a/camera_front_wide_120fov_prompt.txt", 1
        ),
        _repo_file(
            "sample_set/26.01_release/scene-a/camera_front_wide_120fov.mp4", 1_000
        ),
        _repo_file("sample_set/26.01_release/scene-a/scene-a.usdz", 1_000),
        _repo_file(
            "sample_set/26.01_release/scene-b/camera_front_wide_120fov_rgb.mp4", 100
        ),
        _repo_file(
            "sample_set/26.01_release/scene-b/camera_front_wide_120fov_hdmap.mp4", 10
        ),
    ]

    cases = build_cases_from_repo_files(
        files,
        dataset_repo="repo",
        dataset_revision="rev",
        dataset_subpath="sample_set/26.01_release",
        camera="camera_front_wide_120fov",
    )

    assert [case.uuid for case in cases] == ["scene-a"]
    assert cases[0].reference_video.path.endswith("camera_front_wide_120fov_rgb.mp4")
    assert cases[0].hdmap_video.path.endswith("camera_front_wide_120fov_hdmap.mp4")
    assert cases[0].prompt.path.endswith("camera_front_wide_120fov_prompt.txt")


def test_build_cases_from_raw_layout_supports_legacy_prefix_and_scene_prompt() -> None:
    files = [
        _repo_file(
            "sample_set/26.01_release/scene-a/scene-a.camera_front_wide_120fov_rgb.mp4",
            100,
        ),
        _repo_file(
            "sample_set/26.01_release/scene-a/scene-a.camera_front_wide_120fov_hdmap.mp4",
            10,
        ),
        _repo_file("sample_set/26.01_release/scene-a/scene-a.prompt.txt", 1),
    ]

    cases = build_cases_from_repo_files(
        files,
        dataset_repo="repo",
        dataset_revision="rev",
        dataset_subpath="sample_set/26.01_release",
        camera="camera_front_wide_120fov",
    )

    assert [case.uuid for case in cases] == ["scene-a"]
    assert cases[0].prompt.path.endswith("scene-a.prompt.txt")


def test_manifest_round_trip(tmp_path: Path) -> None:
    case = EvalCase(
        uuid="uuid-a",
        camera="cam",
        dataset_repo="repo",
        dataset_revision="rev",
        dataset_subpath="sub",
        reference_video=AssetRef("video.mp4", 10),
        hdmap_video=AssetRef("hdmap.mp4", 20),
        prompt=AssetRef("prompt.txt", 1),
    )
    path = tmp_path / "manifest.jsonl"

    write_cases_jsonl([case], path)

    assert read_cases_jsonl(path) == [case]


def test_staged_manifest_round_trip(tmp_path: Path) -> None:
    case = EvalCase(
        uuid="uuid-a",
        camera="cam",
        dataset_repo="repo",
        dataset_revision="rev",
        dataset_subpath="sub",
        reference_video=AssetRef("video.mp4", 10),
        hdmap_video=AssetRef("hdmap.mp4", 20),
        prompt=AssetRef("prompt.txt", 1),
    )
    staged = StagedCase(
        case=case,
        reference_video_path=tmp_path / "video.mp4",
        hdmap_video_path=tmp_path / "hdmap.mp4",
        prompt_path=tmp_path / "prompt.txt",
        first_frame_path=tmp_path / "first.png",
        prompt_text="drive forward",
    )
    path = tmp_path / "staged.jsonl"

    write_staged_cases_jsonl([staged], path)

    assert read_staged_cases_jsonl(path) == [staged]


def test_plan_batches_caps_by_count_and_bytes() -> None:
    cases = [
        _case("a", 70),
        _case("b", 70),
        _case("c", 70),
    ]

    batches = plan_batches(cases, batch_size=2, max_batch_bytes=100)

    assert [batch.case_uuids for batch in batches] == [("a",), ("b",), ("c",)]
    assert cases_for_batch(cases, batches[1]) == [cases[1]]


def test_parse_byte_size() -> None:
    assert parse_byte_size("1GB") == 1000**3
    assert parse_byte_size("1GiB") == 1024**3
    assert parse_byte_size("512") == 512
    with pytest.raises(ValueError, match="invalid byte size"):
        parse_byte_size("1ib")


def test_generation_command_uses_staged_inputs(tmp_path: Path) -> None:
    staged = StagedCase(
        case=_case("uuid-a", 10),
        reference_video_path=tmp_path / "ref.mp4",
        hdmap_video_path=tmp_path / "hdmap.mp4",
        prompt_path=tmp_path / "prompt.txt",
        first_frame_path=tmp_path / "first.png",
        prompt_text="a prompt",
    )

    result = generation_result_for_case(
        staged,
        run_root=tmp_path / "run",
        recipe="recipe",
        total_blocks=7,
        flashdreams_run="flashdreams-run",
    )

    command = list(result.command)
    assert command[:2] == ["flashdreams-run", "recipe"]
    assert command[command.index("--prompt") + 1] == "a prompt"
    assert command[command.index("--hdmap-video-paths") + 1] == str(
        tmp_path / "hdmap.mp4"
    )
    assert command[command.index("--total-blocks") + 1] == "7"
    assert (
        result.generated_video_path == tmp_path / "run/generated/uuid-a/generated.mp4"
    )
    assert result.log_path == tmp_path / "run/generated/uuid-a/flashdreams-run.log"


def test_generate_cli_defaults_to_stable_non_perf_recipe(tmp_path: Path) -> None:
    parser = _build_parser()

    args = parser.parse_args(
        [
            "generate",
            "--staged-manifest",
            str(tmp_path / "staged.jsonl"),
            "--run-root",
            str(tmp_path / "run"),
        ]
    )

    assert args.recipe == DEFAULT_GENERATION_RECIPE
    assert not args.recipe.endswith("-perf")
    assert args.stream_logs is False


def test_generate_resume_writes_missing_metadata_for_existing_video(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = StagedCase(
        case=_case("uuid-a", 10),
        reference_video_path=tmp_path / "ref.mp4",
        hdmap_video_path=tmp_path / "hdmap.mp4",
        prompt_path=tmp_path / "prompt.txt",
        first_frame_path=tmp_path / "first.png",
        prompt_text="a prompt",
    )
    generated_video = tmp_path / "run/generated/uuid-a/generated.mp4"
    generated_video.parent.mkdir(parents=True)
    generated_video.write_bytes(b"video")

    def fail_run(*args, **kwargs):
        raise AssertionError("generation should not rerun for existing generated.mp4")

    monkeypatch.setattr(generation.subprocess, "run", fail_run)

    results = generate_cases(
        [staged],
        run_root=tmp_path / "run",
        recipe="recipe",
        total_blocks=7,
        flashdreams_run="flashdreams-run",
    )

    metadata_path = generated_video.parent / "generation.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert results[0].generated_video_path == generated_video
    assert metadata["uuid"] == "uuid-a"
    assert metadata["generated_video_path"] == str(generated_video)


def test_streaming_generation_failure_is_contextual(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = generation_result_for_case(
        StagedCase(
            case=_case("uuid-a", 10),
            reference_video_path=tmp_path / "ref.mp4",
            hdmap_video_path=tmp_path / "hdmap.mp4",
            prompt_path=tmp_path / "prompt.txt",
            first_frame_path=tmp_path / "first.png",
            prompt_text="a prompt",
        ),
        run_root=tmp_path / "run",
        recipe="recipe",
        total_blocks=7,
        flashdreams_run="flashdreams-run",
    )

    def fake_run(command, *, check):
        assert command == list(result.command)
        assert check is False
        return generation.subprocess.CompletedProcess(command, 17)

    monkeypatch.setattr(generation.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="uuid-a.*exit code 17"):
        generation._run_generation_command(result, stream_logs=True)


def test_drivinggen_video_command_omits_invalid_track_arg(tmp_path: Path) -> None:
    command = video_metrics_command(
        drivinggen_root=tmp_path,
        split="split",
        model_name="model",
        exp_id="exp",
        metric="fvd",
        python="/env/bin/python",
    )

    assert command[0] == "/env/bin/python"
    assert "--track" not in command
    assert command[command.index("--root_path") + 1] == "./cache/infer_results/split"


def test_patch_drivinggen_checkout_makes_checkpoints_configurable(
    tmp_path: Path,
) -> None:
    fvd_path = (
        tmp_path / "third_parties/stylegan-v/src/metrics/frechet_video_distance.py"
    )
    fid_path = (
        tmp_path / "third_parties/stylegan-v/src/metrics/frechet_inception_distance.py"
    )
    fvd_path.parent.mkdir(parents=True)
    fvd_path.write_text(
        "\n".join(
            [
                "import copy",
                "import numpy as np",
                "",
                "def compute_fvd():",
                "    detector_url = '/shared_disk/users/yang.zhou/iclr_open_source/DrivingGen/ckpt/i3d_torchscript.pt'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fid_path.write_text(
        "\n".join(
            [
                "import numpy as np",
                "",
                "def compute_fid():",
                "    detector_url = '/shared_disk/users/yang.zhou/iclr_open_source/DrivingGen/ckpt/inception-2015-12-05.pkl'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert patch_drivinggen_checkout(tmp_path) == (
        "fvd_checkpoint_env",
        "fid_checkpoint_env",
    )
    assert patch_drivinggen_checkout(tmp_path) == ()

    fvd_text = fvd_path.read_text(encoding="utf-8")
    fid_text = fid_path.read_text(encoding="utf-8")
    assert "import os" in fvd_text
    assert (
        f"os.environ.get('DRIVINGGEN_I3D_CKPT', '{DEFAULT_I3D_TORCHSCRIPT_URL}')"
        in fvd_text
    )
    assert "import os" in fid_text
    assert (
        "os.environ.get('DRIVINGGEN_INCEPTION_CKPT', './ckpt/inception-2015-12-05.pkl')"
    ) in fid_text


def test_run_video_metrics_captures_output_to_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_run(command, *, cwd, check, env, stdout=None, stderr=None):
        calls.append(
            {
                "command": command,
                "cwd": cwd,
                "check": check,
                "env": env,
                "stderr": stderr,
            }
        )
        assert stdout is not None
        stdout.write("metric output\n")

    monkeypatch.setattr(drivinggen.subprocess, "run", fake_run)

    log_path = tmp_path / "metric.log"
    command = run_video_metrics(
        drivinggen_root=tmp_path,
        split="split",
        model_name="model",
        exp_id="exp",
        metric="fvd",
        log_path=log_path,
        extra_env={"DRIVINGGEN_I3D_CKPT": "/tmp/i3d.pt"},
    )

    assert calls[0]["cwd"] == tmp_path
    assert calls[0]["env"]["DRIVINGGEN_I3D_CKPT"] == "/tmp/i3d.pt"
    assert calls[0]["stderr"] == drivinggen.subprocess.STDOUT
    assert command[0] == "python"
    log_text = log_path.read_text(encoding="utf-8")
    assert log_text.startswith("$ python drivinggen/z-sample_fvd.py")
    assert "metric output" in log_text


def test_stage_drivinggen_fvd_fake_frames_skips_first_frame(tmp_path: Path) -> None:
    image_dir = tmp_path / "cache/infer_results/split/uuid-a/model/exp/images"
    image_dir.mkdir(parents=True)
    for index in range(101):
        (image_dir / f"{index:05d}.png").write_bytes(b"frame")

    fake_root = stage_drivinggen_fvd_fake_frames(
        drivinggen_root=tmp_path,
        split="split",
        model_name="model",
        exp_id="exp",
        force=True,
    )

    output_dir = fake_root / "uuid-a+model+exp"
    assert not (output_dir / "00000.png").exists()
    assert (output_dir / "00001.png").exists()
    assert (output_dir / "00100.png").exists()


def test_stage_drivinggen_fvd_fake_frames_requires_100_frames_after_skip(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "cache/infer_results/split/uuid-a/model/exp/images"
    image_dir.mkdir(parents=True)
    for index in range(100):
        (image_dir / f"{index:05d}.png").write_bytes(b"frame")

    with pytest.raises(RuntimeError, match="100 generated frames after skipping"):
        stage_drivinggen_fvd_fake_frames(
            drivinggen_root=tmp_path,
            split="split",
            model_name="model",
            exp_id="exp",
            force=True,
        )

    assert not (
        tmp_path / "cache/infer_results/split+model_fvd/uuid-a+model+exp"
    ).exists()


def test_stage_drivinggen_fvd_fake_frames_uses_portable_relative_symlinks(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "cache/infer_results/split/uuid-a/model/exp/images"
    image_dir.mkdir(parents=True)
    for index in range(101):
        (image_dir / f"{index:05d}.png").write_bytes(b"frame")

    fake_root = stage_drivinggen_fvd_fake_frames(
        drivinggen_root=tmp_path,
        split="split",
        model_name="model",
        exp_id="exp",
        force=True,
    )

    link = fake_root / "uuid-a+model+exp/00001.png"
    assert link.is_symlink()
    assert not Path(os.readlink(link)).is_absolute()
    assert link.exists()


def test_stage_drivinggen_fvd_reference_frames_uses_split_subset(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "split.json").write_text('["uuid-b"]\n', encoding="utf-8")
    for uuid in ("uuid-a", "uuid-b"):
        reference_dir = data_dir / "videos-fvd" / uuid
        reference_dir.mkdir(parents=True)
        for index in range(101):
            (reference_dir / f"{index:05d}.png").write_bytes(b"frame")

    reference_root = stage_drivinggen_fvd_reference_frames(
        drivinggen_root=tmp_path,
        split="split",
        force=True,
    )

    assert reference_root.name == "split+reference_fvd"
    assert not (reference_root / "uuid-a").exists()
    assert (reference_root / "uuid-b").exists()
    assert (reference_root / "uuid-b").is_symlink()
    assert not Path(os.readlink(reference_root / "uuid-b")).is_absolute()


def test_run_fvd_lite_writes_json_and_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DRIVINGGEN_I3D_CKPT", raising=False)
    image_dir = tmp_path / "cache/infer_results/split/uuid-a/model/exp/images"
    split_path = tmp_path / "data/split.json"
    reference_dir = tmp_path / "data/videos-fvd/uuid-a"
    image_dir.mkdir(parents=True)
    reference_dir.mkdir(parents=True)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text('["uuid-a"]\n', encoding="utf-8")
    for index in range(101):
        (image_dir / f"{index:05d}.png").write_bytes(b"frame")
        (reference_dir / f"{index:05d}.png").write_bytes(b"frame")

    def fake_calculate(*, drivinggen_root: Path, fake_root: Path, reference_root: Path):
        assert drivinggen_root == tmp_path
        assert fake_root.name == "split+model_fvd"
        assert reference_root.name == "split+reference_fvd"
        assert (reference_root / "uuid-a").exists()
        print("fake fvd run")
        return {"results": {"fvd2048_100f": 12.5}}

    monkeypatch.setattr(drivinggen, "_calculate_styleganv_fvd", fake_calculate)

    payload = run_fvd_lite(
        drivinggen_root=tmp_path,
        split="split",
        model_name="model",
        exp_id="exp",
        log_path=tmp_path / "eval.log",
        output_json=tmp_path / "eval.json",
        force=True,
    )

    assert payload["value"] == 12.5
    written_json = json.loads((tmp_path / "eval.json").read_text(encoding="utf-8"))
    assert written_json["value"] == 12.5
    assert written_json["i3d_checkpoint"] == DEFAULT_I3D_TORCHSCRIPT_URL
    assert "fake fvd run" in (tmp_path / "eval.log").read_text(encoding="utf-8")


def test_run_fvd_reference_lite_writes_json_and_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DRIVINGGEN_I3D_CKPT", raising=False)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "split-a.json").write_text('["uuid-a"]\n', encoding="utf-8")
    (data_dir / "split-b.json").write_text('["uuid-b"]\n', encoding="utf-8")
    for uuid in ("uuid-a", "uuid-b"):
        reference_dir = data_dir / "videos-fvd" / uuid
        reference_dir.mkdir(parents=True)
        for index in range(101):
            (reference_dir / f"{index:05d}.png").write_bytes(b"frame")

    def fake_calculate(*, drivinggen_root: Path, fake_root: Path, reference_root: Path):
        assert drivinggen_root == tmp_path
        assert reference_root.name == "split-a+reference_fvd"
        assert fake_root.name == "split-b+reference_fvd"
        print("fake reference fvd run")
        return {"results": {"fvd2048_100f": 7.25}}

    monkeypatch.setattr(drivinggen, "_calculate_styleganv_fvd", fake_calculate)

    payload = run_fvd_reference_lite(
        drivinggen_root=tmp_path,
        split_a="split-a",
        split_b="split-b",
        log_path=tmp_path / "reference.log",
        output_json=tmp_path / "reference.json",
        force=True,
    )

    assert payload["value"] == 7.25
    written_json = json.loads((tmp_path / "reference.json").read_text(encoding="utf-8"))
    assert written_json["split_a"] == "split-a"
    assert written_json["split_b"] == "split-b"
    assert written_json["i3d_checkpoint"] == DEFAULT_I3D_TORCHSCRIPT_URL
    assert "fake reference fvd run" in (tmp_path / "reference.log").read_text(
        encoding="utf-8"
    )


def test_write_worldlens_consistency_config(tmp_path: Path) -> None:
    config_path = write_worldlens_consistency_config(tmp_path / "WorldLens")

    text = config_path.read_text(encoding="utf-8")
    assert config_path.name == f"{DEFAULT_WORLDLENS_CONFIG_NAME}.yaml"
    assert "temporal_consistency" in text
    assert "subject_consistency" in text
    assert "repo_or_dir: worldbench/third_party/dino" in text


def test_stage_worldlens_video_inputs_uses_submission_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = StagedCase(
        case=_case("uuid-a", 10),
        reference_video_path=tmp_path / "reference.mp4",
        hdmap_video_path=tmp_path / "hdmap.mp4",
        prompt_path=tmp_path / "prompt.txt",
        first_frame_path=tmp_path / "first.png",
        prompt_text="a prompt",
    )
    generated_video = tmp_path / "run/generated/uuid-a/generated.mp4"
    generated_video.parent.mkdir(parents=True)
    generated_video.write_bytes(b"generated")
    staged.reference_video_path.write_bytes(b"reference")
    crop_calls = []

    def fake_frame_count(video_path: Path) -> int:
        assert video_path == generated_video
        return 157

    def fake_copy_first_frames(
        source: Path,
        target: Path,
        *,
        max_frames: int,
        force: bool = False,
    ) -> int:
        crop_calls.append((source, target, max_frames, force))
        target.write_bytes(b"cropped-reference")
        return max_frames

    monkeypatch.setattr(worldlens, "video_frame_count", fake_frame_count)
    monkeypatch.setattr(worldlens, "copy_video_first_frames", fake_copy_first_frames)

    manifest_path = stage_worldlens_video_inputs(
        [staged],
        generated_root=tmp_path / "run/generated",
        worldlens_root=tmp_path / "WorldLens",
        method_name="omnidreams",
        generation_index=0,
        force=True,
    )

    generated_target = (
        tmp_path
        / "WorldLens/generated_results/omnidreams/video_submission/"
        / "uuid-a_gen0/uuid-a_CAM_FRONT.mp4"
    )
    reference_target = (
        tmp_path
        / "WorldLens/generated_results/gt/video_submission/"
        / "uuid-a_gen0/uuid-a_CAM_FRONT.mp4"
    )
    assert generated_target.exists()
    assert reference_target.exists()
    assert generated_target.is_symlink()
    assert not reference_target.is_symlink()
    assert reference_target.read_bytes() == b"cropped-reference"
    assert not Path(os.readlink(generated_target)).is_absolute()
    assert crop_calls == [(staged.reference_video_path, reference_target, 157, True)]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["method_name"] == "omnidreams"
    assert manifest["cases"][0]["scene_dir"] == "uuid-a_gen0"
    assert manifest["cases"][0]["temporal_policy"] == (
        "reference_first_n_frames_matching_generated"
    )
    assert manifest["cases"][0]["generated_frame_count"] == 157
    assert manifest["cases"][0]["reference_frame_count"] == 157


def test_stage_worldlens_manifest_accumulates_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def staged_case(uuid: str) -> StagedCase:
        case = _case(uuid, 10)
        staged = StagedCase(
            case=case,
            reference_video_path=tmp_path / f"{uuid}-reference.mp4",
            hdmap_video_path=tmp_path / f"{uuid}-hdmap.mp4",
            prompt_path=tmp_path / f"{uuid}.txt",
            first_frame_path=tmp_path / f"{uuid}.png",
            prompt_text="a prompt",
        )
        generated_video = tmp_path / "run/generated" / uuid / "generated.mp4"
        generated_video.parent.mkdir(parents=True)
        generated_video.write_bytes(b"generated")
        staged.reference_video_path.write_bytes(b"reference")
        return staged

    monkeypatch.setattr(worldlens, "video_frame_count", lambda _path: 5)

    def fake_copy_first_frames(
        _source: Path,
        target: Path,
        *,
        max_frames: int,
        force: bool = False,
    ) -> int:
        target.write_bytes(b"cropped")
        return max_frames

    monkeypatch.setattr(worldlens, "copy_video_first_frames", fake_copy_first_frames)

    stage_worldlens_video_inputs(
        [staged_case("uuid-a")],
        generated_root=tmp_path / "run/generated",
        worldlens_root=tmp_path / "WorldLens",
        force=True,
    )
    manifest_path = stage_worldlens_video_inputs(
        [staged_case("uuid-b")],
        generated_root=tmp_path / "run/generated",
        worldlens_root=tmp_path / "WorldLens",
        force=True,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [case["uuid"] for case in manifest["cases"]] == ["uuid-a", "uuid-b"]


def test_worldlens_evaluate_command_uses_hydra_config_name(tmp_path: Path) -> None:
    command = worldlens_evaluate_command(
        worldlens_root=tmp_path,
        method_name="omnidreams",
        config_name="custom_config",
        generated_data_path="generated_results",
        python="/env/bin/python",
        hydra_overrides=["foo=bar"],
    )

    assert command[:4] == [
        "/env/bin/python",
        "tools/evaluate.py",
        "--config-name",
        "custom_config",
    ]
    assert "modality=videogen" in command
    assert "method_name=omnidreams" in command
    assert "generated_data_path=generated_results" in command
    assert command[-1] == "foo=bar"


def test_run_worldlens_evaluation_writes_summary_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exp_result = (
        tmp_path / "WorldLens/tools/exp/videogen/omnidreams/2026/metric_results.json"
    )
    exp_result.parent.mkdir(parents=True)
    exp_result.write_text('{"top": null}\n', encoding="utf-8")
    artifact_path = (
        tmp_path
        / "WorldLens/generated_results/omnidreams/temporal_consistency/repeat_0.json"
    )
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text('{"ts_per_frame": 0.9}\n', encoding="utf-8")
    calls = []

    def fake_run(command, *, cwd, check, env, stdout=None, stderr=None):
        calls.append(
            {
                "command": command,
                "cwd": cwd,
                "check": check,
                "env": env,
                "stderr": stderr,
            }
        )
        if stdout is not None:
            stdout.write("worldlens output\n")

    monkeypatch.setattr(worldlens.subprocess, "run", fake_run)

    payload = run_worldlens_evaluation(
        worldlens_root=tmp_path / "WorldLens",
        method_name="omnidreams",
        exp_root=tmp_path / "WorldLens/tools/exp",
        log_path=tmp_path / "worldlens.log",
        output_json=tmp_path / "worldlens.json",
    )

    assert calls[0]["cwd"] == tmp_path / "WorldLens"
    assert calls[0]["env"]["WORLDBENCH_EXP_ROOT"].endswith("WorldLens/tools/exp")
    assert calls[0]["stderr"] == worldlens.subprocess.STDOUT
    assert payload["metric_results"] == {"top": None}
    artifact_results = payload["artifact_results"]
    assert isinstance(artifact_results, dict)
    artifact_results_by_path = {
        str(key): value for key, value in artifact_results.items()
    }
    assert artifact_results_by_path["temporal_consistency/repeat_0.json"] == {
        "ts_per_frame": 0.9
    }
    written = json.loads((tmp_path / "worldlens.json").read_text(encoding="utf-8"))
    assert written["metric_results_path"] == str(exp_result)
    assert "worldlens output" in (tmp_path / "worldlens.log").read_text(
        encoding="utf-8"
    )


def test_latest_and_artifact_worldlens_results(tmp_path: Path) -> None:
    old = tmp_path / "exp/videogen/omnidreams/old/metric_results.json"
    new = tmp_path / "exp/videogen/omnidreams/new/metric_results.json"
    old.parent.mkdir(parents=True)
    new.parent.mkdir(parents=True)
    old.write_text("{}\n", encoding="utf-8")
    new.write_text("{}\n", encoding="utf-8")
    os.utime(old, (1, 1))
    os.utime(new, (2, 2))
    stage_manifest = (
        tmp_path / "WorldLens/generated_results/omnidreams/stage_manifest.json"
    )
    metric_json = (
        tmp_path / "WorldLens/generated_results/omnidreams/metric/repeat_0.json"
    )
    metric_json.parent.mkdir(parents=True)
    stage_manifest.write_text("{}\n", encoding="utf-8")
    metric_json.write_text('{"score": 1}\n', encoding="utf-8")

    assert (
        latest_worldlens_metric_results(
            exp_root=tmp_path / "exp",
            modality="videogen",
            method_name="omnidreams",
        )
        == new
    )
    assert collect_worldlens_artifact_results(
        worldlens_root=tmp_path / "WorldLens",
        method_name="omnidreams",
    ) == {"metric/repeat_0.json": {"score": 1}}


def test_build_run_summary_collects_validation_and_metrics(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _write_valid_generated_case(run_root, "uuid-a")
    drivinggen_json = (
        run_root
        / "evaluators/DrivingGen/cache/eval_logs/split/omnidreams-exp-fvd-lite.json"
    )
    drivinggen_json.parent.mkdir(parents=True)
    drivinggen_json.write_text(
        json.dumps(
            {
                "metric": "fvd2048_100f",
                "value": 12.5,
                "split": "split",
                "model_name": "omnidreams",
                "exp_id": "exp",
                "results": {"fvd2048_100f": 12.5},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    wl_root = run_root / "evaluators/WorldLens"
    wl_stage_manifest = wl_root / "generated_results/omnidreams/stage_manifest.json"
    wl_stage_manifest.parent.mkdir(parents=True)
    wl_stage_manifest.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "uuid": "uuid-a",
                        "generated_frame_count": 13,
                        "reference_frame_count": 13,
                        "temporal_policy": "reference_first_n_frames_matching_generated",
                    },
                    {"uuid": "uuid-incomplete"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    wl_summary_json = wl_root / "cache/eval_logs/wl-split/omnidreams-worldlens.json"
    wl_summary_json.parent.mkdir(parents=True)
    wl_summary_json.write_text(
        json.dumps(
            {
                "method_name": "omnidreams",
                "config_name": "config",
                "metric_results_path": "/remote/metric_results.json",
                "artifact_results": {
                    "temporal_consistency/repeat_0.json": {
                        "method_name": "omnidreams",
                        "repeat": 0,
                        "temporal_consistency_per_frame": 0.9,
                        "tji_per_frame": 0.4,
                        "ts_per_frame": 0.8,
                        "video_results": [{"ts": 0.7}, {"ts": 0.9}],
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = build_run_summary(run_root)

    assert summary["generated"]["generated_mp4_count"] == 1
    assert summary["validation"]["failure_count"] == 0
    assert summary["validation"]["runner_written_frames"] == {"13": 1}
    assert summary["drivinggen"]["fvd_lite"][0]["value"] == 12.5
    assert summary["worldlens"]["stage_manifest"]["case_count"] == 2
    assert summary["worldlens"]["stage_manifest"]["generated_frame_counts"] == [13]
    assert summary["worldlens"]["stage_manifest"]["reference_frame_counts"] == [13]
    metric = summary["worldlens"]["runs"][0]["artifact_metrics"][0]
    assert metric["artifact"] == "temporal_consistency/repeat_0.json"
    assert metric["ts_min"] == 0.7
    assert metric["ts_max"] == 0.9
    md = render_run_summary_markdown(summary)
    assert "OmniDreams Evaluation Summary" in md
    assert "DrivingGen" in md
    assert "WorldLens" in md
    assert "Deepening Options" not in md
    write_run_summary_json(summary, run_root / "evaluation-summary.json")
    write_run_summary_markdown(summary, run_root / "evaluation-summary.md")
    assert (run_root / "evaluation-summary.json").exists()
    assert (run_root / "evaluation-summary.md").exists()


def test_fvd_lite_import_check_reports_missing_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_import(module_name: str):
        if module_name == "torch":
            raise ImportError("No module named 'torch'")
        return object()

    monkeypatch.setattr(drivinggen.importlib, "import_module", fake_import)

    with pytest.raises(ImportError, match="torch .*No module named 'torch'"):
        _check_styleganv_fvd_imports()


def test_decode_video_to_frames_reuses_matching_existing_frames(tmp_path: Path) -> None:
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    (frame_dir / "00000.png").write_bytes(b"frame")
    (frame_dir / "00001.jpg").write_bytes(b"frame")
    (frame_dir / "notes.txt").write_text("ignored", encoding="utf-8")

    assert (
        decode_video_to_frames(tmp_path / "missing.mp4", frame_dir, max_frames=2) == 2
    )
    with pytest.raises(RuntimeError, match="rerun with --force"):
        decode_video_to_frames(tmp_path / "missing.mp4", frame_dir, max_frames=3)


def test_stage_drivinggen_crops_reference_to_generated_frame_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = StagedCase(
        case=_case("uuid-a", 10),
        reference_video_path=tmp_path / "reference.mp4",
        hdmap_video_path=tmp_path / "hdmap.mp4",
        prompt_path=tmp_path / "prompt.txt",
        first_frame_path=tmp_path / "first.png",
        prompt_text="a prompt",
    )
    generated_video = tmp_path / "run/generated/uuid-a/generated.mp4"
    generated_video.parent.mkdir(parents=True)
    generated_video.write_bytes(b"video")
    calls: list[tuple[Path, Path, int | None]] = []

    def fake_decode(
        video_path: Path,
        output_dir: Path,
        *,
        force: bool = False,
        max_frames: int | None = None,
    ) -> int:
        calls.append((video_path, output_dir, max_frames))
        return 157 if max_frames is None else max_frames

    monkeypatch.setattr(drivinggen, "decode_video_to_frames", fake_decode)
    monkeypatch.setattr(drivinggen, "_copy_or_link", lambda *args, **kwargs: None)

    drivinggen.stage_drivinggen_video_inputs(
        [staged],
        generated_root=tmp_path / "run/generated",
        drivinggen_root=tmp_path / "DrivingGen",
        split="split",
        model_name="model",
        exp_id="exp",
    )

    assert calls[0][0] == generated_video
    assert calls[0][2] is None
    assert calls[1][0] == staged.reference_video_path
    assert calls[1][2] == 157
    metadata = json.loads(
        (
            tmp_path
            / "DrivingGen/cache/infer_results/split/uuid-a/model/exp/stage_metadata.json"
        ).read_text(encoding="utf-8")
    )
    assert metadata["temporal_policy"] == "reference_first_n_frames_matching_generated"
    assert metadata["generated_frame_count"] == 157
    assert metadata["reference_frame_count"] == 157


def test_validate_generated_run_checks_runner_schedule(tmp_path: Path) -> None:
    case_dir = tmp_path / "run/generated/uuid-a"
    runner_dir = case_dir / "runner"
    runner_dir.mkdir(parents=True)
    (case_dir / "generated.mp4").write_bytes(b"video")
    (runner_dir / "recipe.mp4").write_bytes(b"stacked")
    (case_dir / "generation.json").write_text(
        json.dumps(
            {
                "command": ["flashdreams-run", "recipe", "--total-blocks", "2"],
                "uuid": "uuid-a",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (case_dir / "flashdreams-run.log").write_text(
        "\n".join(
            [
                "loaded hdmap_videos=(1, 1, 594, 3, 704, 1280), num_views=1",
                "AR step 0/2, num_frames=5, frames=[0, 5)",
                "AR step 1/2, num_frames=8, frames=[5, 13)",
                "wrote video (1, 1, 13, 3, 704, 1280) -> out.mp4",
            ]
        ),
        encoding="utf-8",
    )
    (runner_dir / "stats_recipe.json").write_text("[{}, {}]\n", encoding="utf-8")

    result = validate_generated_run(tmp_path / "run")[0]

    assert result.ok
    assert result.total_blocks == 2
    assert result.ar_steps == 2
    assert result.expected_frames_from_steps == 13
    assert result.runner_written_frames == 13
    assert result.hdmap_frames == 594
    assert result.stats_steps == 2


def test_validate_generated_run_requires_generated_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="generated output directory"):
        validate_generated_run(tmp_path / "run")


def test_cli_manifest_shape_is_jsonl(tmp_path: Path) -> None:
    case = _case("uuid-a", 10)
    path = tmp_path / "manifest.jsonl"

    write_cases_jsonl([case], path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["kind"] == "omnidreams_eval_manifest"
    assert json.loads(lines[1])["uuid"] == "uuid-a"


def _case(uuid: str, size: int) -> EvalCase:
    return EvalCase(
        uuid=uuid,
        camera="cam",
        dataset_repo="repo",
        dataset_revision="rev",
        dataset_subpath="sub",
        reference_video=AssetRef(f"{uuid}-video.mp4", size),
        hdmap_video=AssetRef(f"{uuid}-hdmap.mp4", 0),
        prompt=AssetRef(f"{uuid}.txt", 0),
    )


def _write_valid_generated_case(run_root: Path, uuid: str) -> None:
    case_dir = run_root / "generated" / uuid
    runner_dir = case_dir / "runner"
    runner_dir.mkdir(parents=True)
    (case_dir / "generated.mp4").write_bytes(b"video")
    (runner_dir / "recipe.mp4").write_bytes(b"stacked")
    (case_dir / "generation.json").write_text(
        json.dumps(
            {
                "command": ["flashdreams-run", "recipe", "--total-blocks", "2"],
                "uuid": uuid,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (case_dir / "flashdreams-run.log").write_text(
        "\n".join(
            [
                "loaded hdmap_videos=(1, 1, 594, 3, 704, 1280), num_views=1",
                "AR step 0/2, num_frames=5, frames=[0, 5)",
                "AR step 1/2, num_frames=8, frames=[5, 13)",
                "wrote video (1, 1, 13, 3, 704, 1280) -> out.mp4",
            ]
        ),
        encoding="utf-8",
    )
    (runner_dir / "stats_recipe.json").write_text("[{}, {}]\n", encoding="utf-8")

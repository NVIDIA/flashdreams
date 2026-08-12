# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from flashdreams.infra.runner import RunnerConfig
from flashdreams.scripts import cli
from flashdreams.serving.launch import ResolvedLaunch, resolve_launch
from flashdreams.serving.launch_manifest import load_launch_manifest

pytestmark = pytest.mark.ci_cpu


def _config(name: str = "demo-runner") -> RunnerConfig:
    return cast(
        RunnerConfig,
        SimpleNamespace(
            runner_name=name,
            launch_capability=None,
            device="cuda:0",
            pipeline=SimpleNamespace(diffusion_model=SimpleNamespace(seed=1)),
        ),
    )


def test_launch_manifest_loads_strict_sections_and_relative_paths(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "launch.yaml"
    manifest_path.write_text(
        """\
schema_version: 1
runner: demo-runner
mode: mp4
runner_overrides:
  device: cuda:3
scenario:
  image_path: assets/frame.png
output:
  path: results/demo.mp4
""",
        encoding="utf-8",
    )

    manifest = load_launch_manifest(manifest_path)

    assert manifest.runner == "demo-runner"
    assert manifest.mode == "mp4"
    assert manifest.scenario["image_path"] == tmp_path / "assets/frame.png"
    assert manifest.output["path"] == tmp_path / "results/demo.mp4"
    assert manifest.apply_runner_overrides(_config()).device == "cuda:3"


def test_launch_manifest_does_not_guess_configs_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "configs" / "launch_manifest" / "demo.yaml"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        'schema_version: 1\nrunner: demo-runner\nmode: "null"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    requested_path = tmp_path / "launch_manifest" / "demo.yaml"
    with pytest.raises(FileNotFoundError) as exc_info:
        load_launch_manifest("launch_manifest/demo.yaml")

    message = str(exc_info.value)
    assert str(requested_path) in message
    assert "resolved relative to the current working directory" in message


@pytest.mark.parametrize(
    "body, match",
    [
        ("schema_version: 2\nrunner: demo\nmode: run\n", "schema_version"),
        ("schema_version: 1\nrunner: demo\nmode: run\nextra: true\n", "extra"),
        ("schema_version: 1\nrunner: ''\nmode: run\n", "runner"),
        ("schema_version: 1\nrunner: demo\nmode: other\n", "unsupported mode"),
    ],
)
def test_launch_manifest_rejects_invalid_documents(
    tmp_path: Path,
    body: str,
    match: str,
) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(body, encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_launch_manifest(path)


def test_positional_mode_and_manifest_are_normalized_before_tyro(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "launch.yaml"
    manifest_path.write_text(
        """\
schema_version: 1
runner: demo-runner
mode: webrtc
runner_overrides:
  device: cuda:2
scenario:
  scene_dir: scenes
output:
  port: 9000
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "all_runners", lambda: {"demo-runner": _config()})

    args, runners, manifest, mode, legacy_manifest, overrides = cli._prepare_cli_args(
        ["demo-runner", "webrtc", "--manifest", str(manifest_path)]
    )

    assert args == ["demo-runner"]
    assert runners["demo-runner"].device == "cuda:2"
    assert manifest is not None
    assert manifest.scenario["scene_dir"] == tmp_path / "scenes"
    assert mode == "webrtc"
    assert legacy_manifest is None
    assert dict(overrides.scenario) == {}
    assert dict(overrides.output) == {}


def test_positional_mode_must_match_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "launch.yaml"
    path.write_text(
        'schema_version: 1\nrunner: demo-runner\nmode: "null"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "all_runners", lambda: {"demo-runner": _config()})

    with pytest.raises(ValueError, match="does not match selected mode"):
        cli._prepare_cli_args(["demo-runner", "webrtc", "--manifest", str(path)])


def test_run_mode_preserves_default_runner_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "all_runners", lambda: {"demo-runner": _config()})

    args, _, manifest, mode, legacy_manifest, _ = cli._prepare_cli_args(
        ["demo-runner", "run"]
    )

    assert args == ["demo-runner"]
    assert manifest is None
    assert mode == "run"
    assert legacy_manifest is None


def test_short_omnidreams_slug_and_mp4_mode_are_parsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli,
        "all_runners",
        lambda: {"omnidreams": _config("omnidreams")},
    )

    args, runners, manifest, mode, legacy_manifest, _ = cli._prepare_cli_args(
        ["omnidreams", "mp4"]
    )

    assert args == ["omnidreams"]
    assert runners["omnidreams"].runner_name == "omnidreams"
    assert manifest is None
    assert mode == "mp4"
    assert legacy_manifest is None


def test_central_options_are_allowed_after_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "all_runners", lambda: {"demo-runner": _config()})

    args, _, _, mode, _, _ = cli._prepare_cli_args(
        [
            "demo-runner",
            "webrtc",
            "--host",
            "127.0.0.1",
            "--no-instantiate",
        ]
    )

    assert args == [
        "--host",
        "127.0.0.1",
        "--no-instantiate",
        "demo-runner",
    ]
    assert mode == "webrtc"


def test_launch_cli_scenario_and_output_overrides_are_parsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli,
        "all_runners",
        lambda: {"omnidreams": _config("omnidreams")},
    )

    args, _, manifest, mode, legacy_manifest, overrides = cli._prepare_cli_args(
        [
            "omnidreams",
            "mp4",
            "--scenario.total-blocks",
            "12",
            "--scenario.example-data=true",
            "--scenario.hdmap-video-paths",
            "[hdmap.mp4]",
            "--output.path",
            "outputs/demo.mp4",
            "--output.fps=30",
        ]
    )

    assert args == ["omnidreams"]
    assert manifest is None
    assert mode == "mp4"
    assert legacy_manifest is None
    assert dict(overrides.scenario) == {
        "total_blocks": 12,
        "example_data": True,
        "hdmap_video_paths": ["hdmap.mp4"],
    }
    assert dict(overrides.output) == {
        "path": "outputs/demo.mp4",
        "fps": 30,
    }


def test_launch_cli_scenario_and_output_overrides_win_over_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "launch.yaml"
    manifest_path.write_text(
        """\
schema_version: 1
runner: demo-runner
mode: mp4
scenario:
  example_data: false
  total_blocks: 4
output:
  path: manifest.mp4
  fps: 12
""",
        encoding="utf-8",
    )
    captured: list[tuple[dict[str, object], dict[str, object]]] = []

    def fake_resolve_launch(
        config: RunnerConfig,
        *,
        mode: str,
        options: cli.LaunchOptions,
    ) -> ResolvedLaunch:
        del config, mode
        captured.append((dict(options.scenario), dict(options.output)))
        return ResolvedLaunch(mode="mp4", label="fake", launch=lambda: None)

    monkeypatch.setattr(cli, "resolve_launch", fake_resolve_launch)

    cli.main(
        _config(),
        no_instantiate=True,
        mode="mp4",
        launch_manifest=load_launch_manifest(manifest_path),
        scenario_overrides={"total_blocks": 9},
        output_overrides={"path": "cli.mp4"},
    )

    assert captured == [
        (
            {"example_data": False, "total_blocks": 9},
            {"path": "cli.mp4", "fps": 12},
        )
    ]


def test_legacy_local_window_manifest_is_routed_without_second_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "example_world_model.yaml"
    path.write_text("resolution_wh: [1280, 704]\n", encoding="utf-8")
    monkeypatch.setattr(cli, "all_runners", lambda: {"demo-runner": _config()})

    args, _, launch_manifest, mode, legacy_manifest, _ = cli._prepare_cli_args(
        ["demo-runner", "local-window", "--manifest", str(path)]
    )

    assert args == ["demo-runner"]
    assert launch_manifest is None
    assert mode == "local-window"
    assert legacy_manifest == path.resolve()


def test_mode_help_lists_only_mode_specific_central_overrides(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as webrtc_exit:
        cli.entrypoint(["lingbot-world-fast", "webrtc", "--help"])
    assert webrtc_exit.value.code == 0
    webrtc_help = capsys.readouterr().out
    assert "Available modes: run, mp4, null, webrtc" in webrtc_help
    assert "--host HOST" in webrtc_help

    with pytest.raises(SystemExit) as mp4_exit:
        cli.entrypoint(["lingbot-world-fast", "mp4", "--help"])
    assert mp4_exit.value.code == 0
    mp4_help = capsys.readouterr().out
    assert "Selected mode: mp4" in mp4_help
    assert "--host HOST" not in mp4_help


def test_entrypoint_builds_parser_for_selected_runner_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def real_config(name: str) -> RunnerConfig:
        return RunnerConfig(
            runner_name=name,
            description=f"{name} description",
            pipeline=cast(
                Any,
                SimpleNamespace(diffusion_model=SimpleNamespace(seed=1)),
            ),
        )

    runners = {
        "selected-runner": real_config("selected-runner"),
        "unrelated-runner": real_config("unrelated-runner"),
    }
    captured: list[tuple[str, ...]] = []

    def fake_union(selected: dict[str, RunnerConfig]) -> object:
        captured.append(tuple(selected))
        return RunnerConfig

    def fake_tyro_cli(args_cls: Any, **kwargs: object) -> object:
        del kwargs
        return args_cls(runner=runners["selected-runner"], no_instantiate=True)

    def fake_main(*args: object, **kwargs: object) -> None:
        del args, kwargs

    monkeypatch.setattr(cli, "all_runners", lambda: runners)
    monkeypatch.setattr(cli, "_annotated_base_runner_union", fake_union)
    monkeypatch.setattr(cli.tyro, "cli", fake_tyro_cli)
    monkeypatch.setattr(cli, "main", fake_main)

    cli.entrypoint(["selected-runner", "--no-instantiate"])

    assert captured == [("selected-runner",)]


def test_explicit_runner_cli_override_wins_over_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "launch.yaml"
    path.write_text(
        """\
schema_version: 1
runner: lingbot-world-fast
mode: webrtc
runner_overrides:
  device: cuda:2
""",
        encoding="utf-8",
    )
    captured: list[tuple[RunnerConfig, dict[str, object]]] = []

    def fake_main(config: RunnerConfig, no_instantiate: bool, **kwargs) -> None:
        assert no_instantiate is True
        captured.append((config, kwargs))

    monkeypatch.setattr(cli, "main", fake_main)

    cli.entrypoint(
        [
            "lingbot-world-fast",
            "webrtc",
            "--manifest",
            str(path),
            "--device",
            "cuda:3",
            "--no-instantiate",
        ]
    )

    assert captured[0][0].device == "cuda:3"
    assert captured[0][1]["mode"] == "webrtc"


@pytest.mark.parametrize(
    "filename",
    [
        "lingbot_mp4.yaml",
        "lingbot_webrtc.yaml",
        "omnidreams_local_window.yaml",
        "omnidreams_mp4.yaml",
        "omnidreams_null.yaml",
        "omnidreams_webrtc.yaml",
    ],
)
def test_documented_launch_manifests_resolve(filename: str) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest = load_launch_manifest(
        repo_root / "configs" / "launch_manifest" / filename
    )
    config = manifest.apply_runner_overrides(cli.all_runners()[manifest.runner])

    if manifest.mode != "run":
        resolved = resolve_launch(
            config,
            mode=cast(Any, manifest.mode),
            options=cli.LaunchOptions(
                launch_manifest=manifest.path,
                scenario=manifest.scenario,
                output=manifest.output,
            ),
        )
        assert resolved.mode == manifest.mode

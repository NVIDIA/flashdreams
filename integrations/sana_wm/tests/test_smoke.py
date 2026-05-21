# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-safe checks for the ``sana_wm`` plugin wiring."""

from __future__ import annotations

import ast
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import tomllib
import tyro
from flashdreams.infra.config import InstantiateConfig
from flashdreams.infra.runner import RunnerConfig
from PIL import Image
from sana_wm import config as config_mod
from sana_wm.config import RUNNER_CONFIGS
from sana_wm.pipeline import (
    DEFAULT_CAMERA_PATH,
    DEFAULT_CONFIG_PATH,
    DEFAULT_IMAGE_PATH,
    DEFAULT_INTRINSICS_PATH,
    DEFAULT_PROMPT_PATH,
    SanaWMGenerationParams,
    SanaWMNativePipelineConfig,
    _ensure_reference_model_registered,
)
from sana_wm.runner import SanaWMRunner, SanaWMRunnerConfig

pytestmark = pytest.mark.ci_cpu

ENTRY_POINT_GROUP = "flashdreams.runner_configs"


def test_runners_dict_is_non_empty() -> None:
    assert RUNNER_CONFIGS, "RUNNER_CONFIGS is empty"


def test_runner_name_mirrors_pipeline_recipe_name() -> None:
    drifted = {
        slug: (cfg.runner_name, cfg.pipeline.recipe_name)
        for slug, cfg in RUNNER_CONFIGS.items()
        if cfg.runner_name != cfg.pipeline.recipe_name
    }
    assert not drifted, f"runner_name != pipeline.recipe_name: {drifted}"


def test_runners_have_descriptions() -> None:
    empty = [
        slug for slug, cfg in RUNNER_CONFIGS.items() if not cfg.description.strip()
    ]
    assert not empty, f"runners missing description: {empty}"


def test_runner_uses_native_pipeline_config() -> None:
    cfg = RUNNER_CONFIGS["sana-wm-bidirectional"]
    assert isinstance(cfg.pipeline, SanaWMNativePipelineConfig)
    assert cfg.pipeline.config_path == DEFAULT_CONFIG_PATH
    assert cfg.pipeline.recipe_name == "sana-wm-bidirectional"


def test_packaged_defaults_exist() -> None:
    for path in (
        DEFAULT_CONFIG_PATH,
        DEFAULT_IMAGE_PATH,
        DEFAULT_PROMPT_PATH,
        DEFAULT_CAMERA_PATH,
        DEFAULT_INTRINSICS_PATH,
    ):
        assert path.exists(), f"missing packaged default asset: {path}"


def test_public_config_uses_shipped_runtime_path() -> None:
    text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")

    assert "model: SanaMSVideoCamCtrl_1600M_P1_D20" in text
    assert "vae_type: LTX2VAE_diffusers" in text
    assert "text_encoder_name: gemma-2-2b-it" in text


def test_default_cuda_runtime_dependencies_are_declared() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as fh:
        meta = tomllib.load(fh)
    dependencies = "\n".join(meta["project"]["dependencies"])

    assert "flash-linear-attention" in dependencies
    assert "triton" in dependencies


def test_no_namespace_packages_in_runtime_tree() -> None:
    package_root = Path(__file__).resolve().parents[1] / "sana_wm"
    python_dirs = {
        parent
        for py in package_root.rglob("*.py")
        for parent in (py.parent, *py.parent.parents)
        if parent == package_root or package_root in parent.parents
    }
    missing = sorted(
        path.relative_to(package_root)
        for path in python_dirs
        if not (path / "__init__.py").exists()
    )
    assert not missing, f"Python package dirs missing __init__.py: {missing}"


def test_vendored_reference_does_not_require_mmcv() -> None:
    reference_root = Path(__file__).resolve().parents[1] / "sana_wm" / "_reference"
    offenders = [
        path.relative_to(reference_root)
        for path in reference_root.rglob("*.py")
        if "mmcv" in path.read_text(encoding="utf-8", errors="ignore")
    ]

    assert not offenders, f"vendored source still references mmcv: {offenders}"


def test_vendored_reference_has_no_training_dependency_imports() -> None:
    reference_root = Path(__file__).resolve().parents[1] / "sana_wm" / "_reference"
    forbidden_imports = {"accelerate"}
    offenders: list[str] = []

    for path in reference_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = {alias.name.split(".", 1)[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported = {node.module.split(".", 1)[0]}
            else:
                continue
            overlap = imported & forbidden_imports
            if overlap:
                offenders.append(
                    f"{path.relative_to(reference_root)} imports {sorted(overlap)}"
                )

    assert not offenders


def test_vendored_reference_uses_private_namespace() -> None:
    reference_root = Path(__file__).resolve().parents[1] / "sana_wm" / "_reference"
    external_import = re.compile(
        r"^\s*(?:from|import)\s+(?:diffusion|sana|tools|inference_video_scripts)(?:[.\s]|$)"
    )
    offenders: list[str] = []
    for path in reference_root.rglob("*.py"):
        for line_no, line in enumerate(
            path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1
        ):
            if external_import.search(line):
                offenders.append(f"{path.relative_to(reference_root)}:{line_no}")

    assert not offenders, (
        f"vendored source imports external Sana namespaces: {offenders}"
    )


def test_unshipped_upstream_branches_are_lazy_imported() -> None:
    builder_path = (
        Path(__file__).resolve().parents[1]
        / "sana_wm"
        / "_reference"
        / "diffusion"
        / "model"
        / "builder.py"
    )
    tree = ast.parse(builder_path.read_text(encoding="utf-8"))
    top_level_modules = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    unshipped = {
        "sana_wm._reference.diffusion.data.datasets.video.sana_video_data",
        "sana_wm._reference.diffusion.model.dc_ae.efficientvit.ae_model_zoo",
        "sana_wm._reference.diffusion.model.qwen.qwen_vl",
        "sana_wm._reference.diffusion.model.wan2_2.vae",
    }

    assert top_level_modules.isdisjoint(unshipped)


def test_reference_model_preflight_noops_when_model_is_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRegistry:
        def get(self, name: str) -> object | None:
            return object() if name == "SanaMSVideoCamCtrl_1600M_P1_D20" else None

    class FakeBuilder:
        MODELS = FakeRegistry()

    imported: list[str] = []

    def fake_import_module(name: str) -> object:
        imported.append(name)
        assert name == "sana_wm._reference.diffusion.model.builder"
        return FakeBuilder

    monkeypatch.setattr("sana_wm.pipeline.importlib.import_module", fake_import_module)

    _ensure_reference_model_registered("SanaMSVideoCamCtrl_1600M_P1_D20")

    assert imported == ["sana_wm._reference.diffusion.model.builder"]


def test_reference_model_preflight_explains_missing_cuda_extras(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRegistry:
        def get(self, name: str) -> None:
            return None

    class FakeBuilder:
        MODELS = FakeRegistry()

    def fake_import_module(name: str) -> object:
        if name == "sana_wm._reference.diffusion.model.builder":
            return FakeBuilder
        if (
            name
            == "sana_wm._reference.diffusion.model.nets.sana_multi_scale_video_camctrl"
        ):
            raise ImportError("missing flash_attn")
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr("sana_wm.pipeline.importlib.import_module", fake_import_module)

    with pytest.raises(RuntimeError, match="required CUDA extras"):
        _ensure_reference_model_registered("SanaMSVideoCamCtrl_1600M_P1_D20")


def test_integration_has_no_external_checkout_or_launcher_hooks() -> None:
    root = Path(__file__).resolve().parents[1]
    forbidden_text = (
        "SANA_WM_REPO",
        "sana_repo",
        "inference_video_scripts",
        "subprocess",
        "git checkout",
    )
    offenders: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in {".md", ".py", ".toml", ".yaml"}:
            continue
        if path == Path(__file__):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for needle in forbidden_text:
            if needle in text:
                offenders.append(f"{path.relative_to(root)} contains {needle!r}")

    python_sys_path = [
        path.relative_to(root)
        for path in root.rglob("*.py")
        if path != Path(__file__)
        if "sys.path" in path.read_text(encoding="utf-8", errors="ignore")
    ]
    assert not offenders
    assert not python_sys_path, f"Python files mutate sys.path: {python_sys_path}"


def test_integration_does_not_ship_local_parity_scripts() -> None:
    root = Path(__file__).resolve().parents[1]
    assert not (root / "scripts").exists()


def test_vendored_reference_excludes_fastlinear_dev_helpers() -> None:
    reference_root = Path(__file__).resolve().parents[1] / "sana_wm" / "_reference"
    forbidden = {
        "develop_triton_ffn.py",
        "develop_triton_litemla.py",
        "compare_results.py",
        "export_onnx.py",
    }
    shipped = {path.name for path in reference_root.rglob("*") if path.is_file()}
    assert not (forbidden & shipped)


def test_vendored_reference_excludes_training_helpers() -> None:
    reference_root = Path(__file__).resolve().parents[1] / "sana_wm" / "_reference"
    forbidden = {
        "diffusion/utils/checkpoint.py",
        "diffusion/utils/data_sampler.py",
        "diffusion/utils/lr_scheduler.py",
    }
    shipped = {
        path.relative_to(reference_root).as_posix()
        for path in reference_root.rglob("*.py")
    }

    assert forbidden.isdisjoint(shipped)


def test_entry_points_match_module_literals() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as fh:
        meta = tomllib.load(fh)
    entries = meta["project"]["entry-points"][ENTRY_POINT_GROUP]
    declared_slugs = set(entries)
    module_slugs = set(RUNNER_CONFIGS)
    assert declared_slugs == module_slugs, (
        f"entry-point slugs ({sorted(declared_slugs)}) "
        f"!= module runners ({sorted(module_slugs)})"
    )

    for slug, target in entries.items():
        module_name, attr = target.split(":", 1)
        assert module_name == "sana_wm.config", (
            f"unexpected module in entry point {slug!r}: {module_name}"
        )
        cfg = cast(RunnerConfig, getattr(config_mod, attr))
        assert cfg.runner_name == slug, (
            f"entry point {slug!r} -> {attr} resolves to "
            f"runner_name={cfg.runner_name!r}"
        )


def test_readme_boolean_override_matches_flashdreams_cli_parser() -> None:
    """Nested booleans use explicit True/False values under FlagConversionOff."""
    parsed = tyro.cli(
        cast(Any, tyro.conf.FlagConversionOff[SanaWMRunnerConfig]),
        args=[
            "--runner-name",
            "sana-wm-bidirectional",
            "--pipeline.enable-refiner",
            "False",
        ],
    )

    assert parsed.pipeline.enable_refiner is False


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="entry-point discovery test relies on importlib.metadata shape",
)
def test_entry_points_discoverable_when_installed() -> None:
    from importlib.metadata import entry_points

    eps = entry_points(group=ENTRY_POINT_GROUP)
    discovered = {ep.name for ep in eps if ep.value.startswith("sana_wm.")}
    if not discovered:
        pytest.skip("plugin not installed; run `pip install -e .` first")
    assert discovered == set(RUNNER_CONFIGS), (
        f"discovered slugs ({sorted(discovered)}) != "
        f"plugin runners ({sorted(RUNNER_CONFIGS)})"
    )


def test_flashdreams_plugin_discovery_loads_runner_config() -> None:
    from flashdreams.plugins import discover_runners

    discovered = discover_runners()
    if "sana-wm-bidirectional" not in discovered:
        pytest.skip("plugin not installed; run `pip install -e .` first")

    cfg = discovered["sana-wm-bidirectional"]
    assert isinstance(cfg, SanaWMRunnerConfig)
    assert cfg.runner_name == "sana-wm-bidirectional"
    assert isinstance(cfg.pipeline, SanaWMNativePipelineConfig)
    assert cfg.pipeline.recipe_name == cfg.runner_name


def test_plugin_discovery_does_not_load_reference_runtime() -> None:
    from flashdreams.plugins import discover_runners

    discover_runners()

    assert "sana_wm._reference.inference_sana_wm" not in sys.modules
    assert (
        "sana_wm._reference.diffusion.model.nets.sana_multi_scale_video_camctrl"
        not in sys.modules
    )


@dataclass
class FakePipelineConfig(InstantiateConfig):
    _target: type = field(default_factory=lambda: FakePipeline)
    recipe_name: str = "sana-wm-bidirectional"
    enable_refiner: bool = False
    config_path: str = "fake-config.yaml"
    model_path: str = "fake-model.safetensors"
    refiner_root: str = "fake-refiner"
    refiner_gemma_root: str = "fake-gemma"
    refiner_seed: int = 42
    sink_size: int = 1
    offload_vae: bool = False
    offload_refiner: bool = False


class FakePipeline:
    def __init__(
        self,
        config: FakePipelineConfig,
        *,
        device: str,
        logger: object,
    ) -> None:
        self.config = config
        self.device = device
        self.logger = logger
        self.calls: dict[str, object] = {}

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        c2w: np.ndarray,
        intrinsics: np.ndarray,
        params: SanaWMGenerationParams,
    ) -> dict[str, object]:
        self.calls["generate"] = {
            "image_size": image.size,
            "prompt": prompt,
            "c2w_shape": c2w.shape,
            "intrinsics_shape": intrinsics.shape,
            "num_frames": params.num_frames,
            "step": params.step,
        }
        return {
            "video": np.zeros((2, 4, 4, 3), dtype=np.uint8),
            "c2w": c2w,
        }


def test_runner_drives_native_pipeline_with_local_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runner wiring should be testable without SANA-WM weights or a GPU."""
    asset_dir = tmp_path / "asset" / "sana_wm"
    asset_dir.mkdir(parents=True)
    image_path = asset_dir / "demo_0.png"
    prompt_path = asset_dir / "demo_0.txt"
    camera_path = asset_dir / "demo_0_pose.npy"
    intrinsics_path = asset_dir / "demo_0_intrinsics.npy"
    Image.new("RGB", (32, 24), color=(12, 34, 56)).save(image_path)
    prompt_path.write_text("drive across a desert\n", encoding="utf-8")
    np.save(camera_path, np.zeros((17, 4, 4), dtype=np.float32))
    np.save(intrinsics_path, np.zeros((17, 4), dtype=np.float32))

    calls: dict[str, object] = {}

    class FakeRefs:
        @staticmethod
        def _snap_num_frames(frames: int, stride: int, upper_bound: int) -> int:
            return frames

        @staticmethod
        def resize_and_center_crop(image: Image.Image):
            return image, image.size, image.size, (0, 0)

        @staticmethod
        def load_intrinsics(path: Path, frames: int) -> np.ndarray:
            return np.zeros((frames, 4), dtype=np.float32)

        @staticmethod
        def transform_intrinsics_for_crop(intrinsics: np.ndarray, *args: object):
            return intrinsics

        @staticmethod
        def get_root_logger():
            return object()

        @staticmethod
        def apply_overlay(video: np.ndarray, c2w: np.ndarray) -> np.ndarray:
            calls["overlay"] = c2w.shape
            return video

        @staticmethod
        def write_video(
            output_dir: Path,
            name: str,
            video_hwc: np.ndarray,
            fps: int,
            logger: object,
        ) -> Path:
            calls["write_video"] = {
                "shape": video_hwc.shape,
                "fps": fps,
            }
            path = output_dir / f"{name}_generated.mp4"
            path.write_bytes(b"fake mp4")
            return path

    monkeypatch.setattr("sana_wm.runner.get_reference_module", lambda: FakeRefs)

    output_dir = tmp_path / "out"
    cfg = SanaWMRunnerConfig(
        runner_name="sana-wm-bidirectional",
        description="test",
        pipeline=cast(Any, FakePipelineConfig()),
        image=image_path,
        prompt=prompt_path,
        camera=camera_path,
        intrinsics=intrinsics_path,
        num_frames=17,
        step=60,
        no_action_overlay=True,
        output_dir=output_dir,
        name="demo_0",
        device="cpu",
    )

    runner = SanaWMRunner(cfg)
    runner.run()

    assert (output_dir / "demo_0_generated.mp4").read_bytes() == b"fake mp4"
    metadata = json.loads((output_dir / "demo_0_metadata.json").read_text())
    assert metadata["runner_name"] == "sana-wm-bidirectional"
    assert metadata["num_frames"] == 17
    assert metadata["step"] == 60
    assert metadata["refiner"] is False
    fake_pipeline = cast(Any, runner.pipeline)
    assert fake_pipeline.calls["generate"] == {
        "image_size": (32, 24),
        "prompt": "drive across a desert",
        "c2w_shape": (17, 4, 4),
        "intrinsics_shape": (17, 4),
        "num_frames": 17,
        "step": 60,
    }
    assert calls["write_video"] == {"shape": (2, 4, 4, 3), "fps": 16}


def test_runner_ignores_partial_torchrun_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RANK", "3")
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    cfg = cast(SanaWMRunnerConfig, RUNNER_CONFIGS["sana-wm-bidirectional"])
    runner = SanaWMRunner(cfg)

    assert runner.local_rank == 0
    assert runner.global_rank == 0
    assert runner.world_size == 1
    assert runner.is_rank_zero


def test_default_cuda_device_falls_back_to_cpu_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sana_wm.runner.torch.cuda.is_available", lambda: False)

    from sana_wm.runner import _resolve_device

    assert _resolve_device("cuda", local_rank=0) == "cpu"


def test_nonzero_torchrun_rank_exits_before_loading_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")

    def fail_reference_import() -> object:
        raise AssertionError("non-zero rank should exit before loading SANA-WM")

    monkeypatch.setattr("sana_wm.runner.get_reference_module", fail_reference_import)

    cfg = cast(SanaWMRunnerConfig, RUNNER_CONFIGS["sana-wm-bidirectional"])
    runner = SanaWMRunner(cfg)
    runner.run()

    assert runner.local_rank == 1
    assert runner.global_rank == 1
    assert runner.world_size == 2
    assert not runner.is_rank_zero

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

"""Frozen model contracts and checkpoint construction without the V1 package."""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

import pytest
import tomllib
import torch
from cosmos3_nano_sim_bimanual.checkpoint_remap import state_dict_transform
from cosmos3_nano_sim_bimanual.transformer.network import CosmosMoTNetworkConfig

pytestmark = pytest.mark.ci_cpu

_PROJECT = Path(__file__).resolve().parents[1]


def _package_sources() -> tuple[Path, ...]:
    return tuple(
        source
        for source in _PROJECT.rglob("*.py")
        if source.relative_to(_PROJECT).parts[0] != "tests"
    )


def test_model_core_has_no_application_or_legacy_dependency() -> None:
    forbidden = {"internal", "cosmos_interactive_internal", "handdreams"}
    for source in _package_sources():
        allowed_app = source.relative_to(_PROJECT).parts[:2] == (
            "apps",
            "handdreams",
        )
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                modules = [node.module or ""]
            else:
                continue
            for module in modules:
                root = module.split(".")[0]
                assert root not in forbidden or (
                    root == "handdreams" and allowed_app
                ), f"{source}:{node.lineno}: out-of-bound import {module}"

    project = tomllib.loads((_PROJECT / "pyproject.toml").read_text())
    assert project["project"]["name"] == "flashdreams-cosmos3-nano-sim-bimanual"
    assert {"flashdreams", "flashdreams-handdreams"} <= set(
        project["project"]["dependencies"]
    )
    assert project["project"]["entry-points"]["flashdreams.applications_v2"][
        "handdreams"
    ] == ("cosmos3_nano_sim_bimanual.apps.handdreams.adapter:create_app")
    assert project["tool"]["setuptools"]["package-data"] == {
        "cosmos3_nano_sim_bimanual": ["normalizers/*.json"],
        "cosmos3_nano_sim_bimanual.apps.handdreams": ["agibot_scenes.json"],
    }


def test_runtime_dependency_closure_contains_only_public_workspace_packages() -> None:
    lock = tomllib.loads((_PROJECT.parents[1] / "uv.lock").read_text())
    packages = {}
    for package in lock["package"]:
        packages.setdefault(package["name"], []).append(package)
    pending = [("flashdreams-cosmos3-nano-sim-bimanual", frozenset())]
    visited = set()
    while pending:
        name, extras = key = pending.pop()
        if key in visited:
            continue
        visited.add(key)
        for package in packages[name]:
            source = package["source"]
            assert not source.get("editable", "").startswith("internal/"), name
            edges = list(package.get("dependencies", ()))
            for extra in extras:
                edges.extend(package.get("optional-dependencies", {}).get(extra, ()))
            pending.extend(
                (edge["name"], frozenset(edge.get("extra", ()))) for edge in edges
            )
    assert {
        "flashdreams",
        "flashdreams-handdreams",
        "flashdreams-controller-input",
        "flashdreams-swiftvr",
    } <= {name for name, _ in visited}


def test_built_wheel_contains_the_complete_model_integration(tmp_path: Path) -> None:
    pytest.importorskip("setuptools")
    pytest.importorskip("wheel")
    source = tmp_path / "cosmos3_nano_sim_bimanual"
    source.mkdir()
    for name in ("pyproject.toml", "README.md"):
        shutil.copy2(_PROJECT / name, source / name)
    for module in _package_sources():
        destination = source / module.relative_to(_PROJECT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(module, destination)
    shutil.copytree(_PROJECT / "normalizers", source / "normalizers")
    manifest = Path("apps/handdreams/agibot_scenes.json")
    shutil.copy2(_PROJECT / manifest, source / manifest)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from setuptools.build_meta import build_wheel; build_wheel('dist')",
        ],
        cwd=source,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    with zipfile.ZipFile(next((source / "dist").glob("*.whl"))) as archive:
        names = set(archive.namelist())
    package = "cosmos3_nano_sim_bimanual/"
    assert {
        name for name in names if name.startswith(package) and name.endswith(".py")
    } == {
        f"{package}{module.relative_to(_PROJECT).as_posix()}"
        for module in _package_sources()
    }
    assert (
        f"{package}normalizers/global_asinh_v1_backward_anchored_gripper_metric.json"
        in names
    )
    assert f"{package}{manifest.as_posix()}" in names
    assert not any(name.endswith((".jpg", ".mp4", ".h5")) for name in names)
    assert not any(
        name.startswith(("handdreams/", "cosmos_interactive_internal/"))
        for name in names
    )


@pytest.mark.parametrize(
    ("vision", "count", "parameters", "digest"),
    [
        (
            False,
            809,
            15173136576,
            "25a645200a9728f125fab11931391126a06322c26b7566eeb15ca2b7634218f3",
        ),
        (
            True,
            1160,
            15749524912,
            "01cf1d3540e54c38d4bed4a30495f7e1b7dbc6fa1f8669656fe9a4078db4662a",
        ),
    ],
)
def test_frozen_agibot_parameter_tree(vision, count, parameters, digest) -> None:
    with torch.device("meta"):
        net = CosmosMoTNetworkConfig(
            include_vision_tower=vision, action_gen=True
        ).setup()
    shapes = {key: list(value.shape) for key, value in net.state_dict().items()}
    payload = json.dumps(shapes, sort_keys=True, separators=(",", ":")).encode()
    assert len(shapes) == count
    assert sum(parameter.numel() for parameter in net.parameters()) == parameters
    assert hashlib.sha256(payload).hexdigest() == digest


def test_checkpoint_remap_rejects_colliding_or_unknown_keys() -> None:
    with pytest.raises(KeyError, match="multiple checkpoint keys"):
        state_dict_transform(
            {
                "model.net.language_model.model.layers.0.self_attn.q_proj_moe_gen.weight": 1,
                "model.net.language_model.model.layers.0.self_attn_moe_gen.q_proj.weight": 2,
            }
        )
    with pytest.raises(KeyError, match="no target mapping"):
        state_dict_transform({"unexpected.weight": 1})


@pytest.mark.parametrize("module", ["checkpoint", "fetch_checkpoint"])
def test_checkpoint_modules_import_without_transformer_preimport(module: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import cosmos3_nano_sim_bimanual.{module}",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("checkpoint_format", ["native", "diffusers"])
def test_real_checkpoint_and_transformer_start_without_app_or_v1(
    tmp_path, checkpoint_format
) -> None:
    script = textwrap.dedent("""
        import importlib.abc
        import json
        import os
        from pathlib import Path
        import sys
        from unittest.mock import patch

        checkpoint = Path(sys.argv[1])
        os.environ["FLASHDREAMS_CACHE_DIR"] = str(checkpoint / "cache")
        os.environ["FLASHDREAMS_MIN_CACHE_FREE_GB"] = "0"

        class ModelBoundary(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in {"cosmos_interactive_internal", "handdreams"}:
                    raise AssertionError("out-of-bound import attempted: " + fullname)

        sys.meta_path.insert(0, ModelBoundary())

        import torch
        from safetensors.torch import save_file
        import cosmos3_nano_sim_bimanual.transformer.network as network
        from cosmos3_nano_sim_bimanual.checkpoint import load_exported_checkpoint
        from cosmos3_nano_sim_bimanual.transformer import HandTransformer, HandTransformerConfig
        from cosmos3_nano_sim_bimanual.spec import Cosmos3InteractiveSpec, LLMSpec, VFMSpec

        network.SPEC = Cosmos3InteractiveSpec(
            llm=LLMSpec(hidden_size=64, num_layers=2, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=16, intermediate_size=128,
                        vocab_size=100, mrope_section=(4, 2, 2)),
            vfm=VFMSpec(vae_latent_channels=8, latent_patch_size=2),
        )
        original = network.CosmosMoTNetworkConfig(
            dtype=torch.bfloat16, include_vision_tower=False, action_gen=True
        ).setup()

        def exported_key(key):
            if key.startswith("layers."):
                _, index, pathway, leaf = key.split(".", 3)
                if pathway == "gen":
                    module, suffix = leaf.split(".", 1)
                    leaf = module + "_moe_gen." + suffix
                return "model.net.language_model.model.layers." + index + "." + leaf
            if key.startswith(("embed_tokens.", "norm.", "norm_moe_gen.")):
                return "model.net.language_model.model." + key
            if key.startswith("lm_head."):
                return "model.net.language_model." + key
            return "model.net." + key

        def diffusers_key(key):
            if key.startswith("layers."):
                _, index, pathway, leaf = key.split(".", 3)
                if leaf.startswith("self_attn."):
                    _, module, suffix = leaf.split(".", 2)
                    projection = {
                        "q_proj": ("to_q", "add_q_proj"),
                        "k_proj": ("to_k", "add_k_proj"),
                        "v_proj": ("to_v", "add_v_proj"),
                        "o_proj": ("to_out", "to_add_out"),
                        "q_norm": ("norm_q", "norm_added_q"),
                        "k_norm": ("norm_k", "norm_added_k"),
                    }[module][pathway == "gen"]
                    leaf = "self_attn." + projection + "." + suffix
                elif pathway == "gen":
                    module, suffix = leaf.split(".", 1)
                    leaf = module + "_moe_gen." + suffix
                return "layers." + index + "." + leaf
            root, separator, leaf = key.partition(".")
            root = {"vae2llm": "proj_in", "llm2vae": "proj_out",
                    "action2llm": "action_proj_in", "llm2action": "action_proj_out"}.get(root, root)
            if root == "time_embedder":
                leaf = leaf.replace("mlp.0.", "linear_1.").replace("mlp.2.", "linear_2.")
            return root + separator + leaf

        key_for_export = diffusers_key if sys.argv[2] == "diffusers" else exported_key
        raw = {key_for_export(key): value for key, value in original.state_dict().items()}
        save_file(raw, str(checkpoint / "model.safetensors"))
        (checkpoint / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {key: "model.safetensors" for key in raw}
        }))
        if sys.argv[2] == "diffusers":
            source = "https://huggingface.co/test/bimanual/resolve/revision/transformer/model.safetensors.index.json"
            def cached_hf_file(*, repo_id, filename, subfolder, revision):
                assert (repo_id, subfolder, revision) == ("test/bimanual", "transformer", "revision")
                return str(checkpoint / filename)
            with patch("flashdreams.core.checkpoint.load.hf_hub_download", cached_hf_file):
                restored = load_exported_checkpoint(source, device="cpu")
            with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}), patch(
                "flashdreams.core.checkpoint.load.hf_hub_download",
                side_effect=AssertionError("merged cache must avoid the Hub"),
            ):
                cached = load_exported_checkpoint(source, device="cpu")
            for key, expected in restored.state_dict().items():
                torch.testing.assert_close(cached.state_dict()[key], expected, rtol=0, atol=0)
        else:
            restored = load_exported_checkpoint(checkpoint, device="cpu")
        for key, expected in original.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key], expected, rtol=0, atol=0)

        transformer = HandTransformer(
            HandTransformerConfig(
                checkpoint=str(checkpoint), device="cpu", chunk_size=2,
                cache_mode="block", window_frames=4,
            ),
            network=restored,
        )
        assert transformer.net.config.action_gen
        for key, expected in original.state_dict().items():
            torch.testing.assert_close(transformer.net.state_dict()[key], expected, rtol=0, atol=0)
        assert not any(
            name.split(".")[0] in {"cosmos_interactive_internal", "handdreams"}
            for name in sys.modules
        )
        print("PASS: real checkpoint and transformer construction within integration")
    """)
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), checkpoint_format],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS: real checkpoint" in result.stdout

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public application wheels include usable entry points and codec notices."""

import hashlib
import shutil
import subprocess
import sys
import sysconfig
import venv
from importlib.util import find_spec
from pathlib import Path

import pytest

pytestmark = pytest.mark.ci_cpu


def test_public_entry_points_and_codecs_with_public_packages(tmp_path, monkeypatch):
    uv = shutil.which("uv")
    assert uv is not None, "uv is required to build the public package wheels"
    projects = Path(__file__).resolve().parents[2]
    wheels = tmp_path / "wheels"
    for name in ("cosmos_interactive", "cosmos3_nano_sim_auto"):
        source = tmp_path / "source" / name
        shutil.copytree(
            projects / name,
            source,
            ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", "*.egg-info", "build", "dist", ".venv"
            ),
        )
        result = subprocess.run(
            [
                uv,
                "build",
                "--wheel",
                "--no-config",
                "--no-sources",
                "--python",
                sys.executable,
                "--out-dir",
                str(wheels),
                str(source),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=tmp_path,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    environment = tmp_path / "venv"
    venv.EnvBuilder(with_pip=False).create(environment)
    python = environment / (
        "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    )
    result = subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--no-config",
            "--no-deps",
            "--python",
            str(python),
            *map(str, sorted(wheels.glob("*.whl"))),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    (tmp_path / "extra_scene_backend.py").write_text("AVAILABLE = True\n")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    script = """
import importlib.abc
import importlib.metadata
import importlib.util
import hashlib
import sys
from importlib.resources import files
from pathlib import Path

assert importlib.util.find_spec('extra_scene_backend') is None

class BlockOptionalKernel(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'natten', 'cosmos_interactive_internal'}:
            raise AssertionError('Unexpected runtime dependency: ' + fullname)

sys.meta_path.insert(0, BlockOptionalKernel())
from cosmos_interactive import checkpoint_load, _vae, _taehv_super
from cosmos_interactive._lidar_codec import network_config
from cosmos_interactive.lidar import LidarTokenizer
from cosmos3_nano_sim_auto.inputs import MultiviewSample
from cosmos3_nano_sim_auto.impl import lidar
from cosmos3_nano_sim_auto.impl.conditioning.world_scenario import prepared_lidar

installed = Path(sys.argv[1])
for name, module in tuple(sys.modules.items()):
    if name.split('.')[0] in {'cosmos_interactive', 'cosmos3_nano_sim_auto'}:
        assert Path(module.__file__).resolve().is_relative_to(installed), name
assert LidarTokenizer.__module__ == 'cosmos_interactive.lidar'
assert lidar.LidarTokenizer is LidarTokenizer
codec = files('cosmos_interactive._lidar_codec')
for name in ('LICENSE.vllm-omni', 'LICENSE.k-diffusion', 'NOTICE', 'README.md'):
    assert codec.joinpath(name).is_file(), name
readme = codec.joinpath('README.md').read_text()
assert '8ef7df57219688835f3752accb2c3b2f444b4058' in readme
assert 'Apache-2.0 AND MIT' in readme
for private_reference in ('cosmos-private', 'gitlab-master.nvidia.com', '/home/', 'internal/docs'):
    assert private_reference not in readme, private_reference
assert 'Version 2.0, January 2004' in codec.joinpath('LICENSE.vllm-omni').read_text()
assert 'Copyright (c) 2022 Katherine Crowson' in codec.joinpath('LICENSE.k-diffusion').read_text()
prepared_lidar.require_implementation()  # prepared-scene rendering is enabled
distribution = importlib.metadata.distribution('flashdreams-cosmos3-nano-sim-auto')
assert Path(distribution.locate_file('')).resolve().is_relative_to(installed)
entries = [e for e in distribution.entry_points if e.group == 'flashdreams.applications_v2']
assert {e.name for e in entries} == {
    'cosmos3-nano-sim-auto', 'interactive-drive-cosmos3-nano-sim-auto',
}
for entry in entries:
    assert entry.load()() is not None
"""
    # -I -S excludes ambient PYTHONPATH, the working directory and editable .pth files.
    roots = {"base": str(environment), "platbase": str(environment)}
    paths = [
        sysconfig.get_path("purelib", vars=roots),
        sysconfig.get_path("platlib", vars=roots),
        sysconfig.get_path("purelib"),
        sysconfig.get_path("platlib"),
    ]
    expected = {}
    for name in ("LICENSE.vllm-omni", "LICENSE.k-diffusion", "NOTICE", "README.md"):
        path = projects / "cosmos_interactive/cosmos_interactive/_lidar_codec" / name
        expected[f"codec/{name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("LICENSE", "NOTICE", "THIRD-PARTY-NOTICES"):
        path = projects / "cosmos_interactive" / name
        expected[f"distribution/{name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    script += (
        f"\nexpected = {expected!r}\n"
        + """
distribution = importlib.metadata.distribution('flashdreams-cosmos-interactive')
for key, digest in expected.items():
    kind, name = key.split('/')
    if kind == 'codec':
        data = codec.joinpath(name).read_bytes()
    else:
        matches = [p for p in distribution.files if str(p).endswith('.dist-info/licenses/' + name)]
        assert len(matches) == 1, key
        data = distribution.locate_file(matches[0]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == digest, key
"""
    )
    for name in (
        "flashdreams.core",
        "interactive_drive",
        "ludus_renderer",
    ):
        spec = find_spec(name)
        assert spec is not None and spec.origin is not None
        paths.append(str(Path(spec.origin).resolve().parents[len(name.split("."))]))
    script = f"import sys\nsys.path[:0] = {paths!r}\n" + script
    result = subprocess.run(
        [str(python), "-I", "-S", "-c", script, str(environment)],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr

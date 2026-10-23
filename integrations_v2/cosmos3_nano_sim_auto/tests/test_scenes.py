# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for hosted scene staging and local overrides."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile

import pytest
from cosmos3_nano_sim_auto import scenes

pytestmark = pytest.mark.ci_cpu


@pytest.fixture
def downloads(monkeypatch, tmp_path):
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        name = Path(kwargs["filename"]).stem
        archive = tmp_path / f"{name}.zip"
        if not archive.exists():
            write_bundle(archive, name)
        return archive

    monkeypatch.setattr(scenes, "hf_hub_download", download)
    monkeypatch.setattr(
        scenes, "default_flashdreams_cache_dir", lambda: tmp_path / "cache"
    )
    return calls


def write_bundle(archive, name):
    with ZipFile(archive, "w") as source:
        for member in (
            "scene.json",
            "sample.json",
            "initial_rgb/front.png",
            "README.md",
            "LIDAR-PROFILE-LICENSE.txt",
        ):
            source.writestr(f"{name}/{member}", name)


@pytest.mark.parametrize("name", scenes.SCENE_NAMES)
def test_download_keeps_each_bundles_matching_assets_and_reuses_extraction(
    downloads, monkeypatch, name
):
    result = scenes.prepare_scene(Path(name))
    assert downloads == [
        {
            "repo_id": "nvidia/omni-dreams-scenes",
            "repo_type": "dataset",
            "filename": f"scenes/synthetic-first-frames/{name}.zip",
        }
    ]
    for member in (
        "scene.json",
        "sample.json",
        "initial_rgb/front.png",
        "README.md",
        "LIDAR-PROFILE-LICENSE.txt",
    ):
        assert (result / member).read_text() == name
    monkeypatch.setattr(
        scenes.zipfile, "ZipFile", lambda *_a: pytest.fail("Extracted twice")
    )
    assert scenes.prepare_scene(Path(name)) == result


def test_local_and_unknown_paths_never_download(downloads, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    local = Path(scenes.SCENE_NAMES[0])
    local.mkdir()
    assert scenes.prepare_scene(local) == local
    missing = tmp_path / "missing-scene"
    assert scenes.prepare_scene(missing) == missing
    assert downloads == []


def test_concurrent_ranks_share_one_complete_extraction(downloads):
    scene = Path(scenes.SCENE_NAMES[0])
    # Prime only the mocked download, so threads race on extraction alone.
    scenes.hf_hub_download(repo_id=scenes.SCENE_REPO_ID, filename=f"{scene}.zip")
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(scenes.prepare_scene, [scene] * 4))
    assert len(set(paths)) == 1
    assert (paths[0] / "sample.json").is_file()


@pytest.mark.parametrize(
    "member",
    [
        "../escape",
        "clipgt-residential-iter1500/../../escape",
        "/absolute",
        "other-scene/scene.json",
        "clipgt-residential-iter1500/..\\escape",
    ],
)
def test_unsafe_archives_never_publish_a_bundle(downloads, tmp_path, member):
    name = scenes.SCENE_NAMES[0]
    with ZipFile(tmp_path / f"{name}.zip", "w") as source:
        source.writestr(member, "bad")
    with pytest.raises(ValueError, match="Unsafe scene archive member"):
        scenes.prepare_scene(Path(name))
    assert not (tmp_path / "escape").exists()
    assert not list((tmp_path / "cache").rglob("scene.json"))


def test_failed_extraction_can_be_retried(downloads, tmp_path):
    name = scenes.SCENE_NAMES[0]
    archive = tmp_path / f"{name}.zip"
    with ZipFile(archive, "w") as source:
        source.writestr(f"{name}/scene.json", "incomplete")
    with pytest.raises(ValueError, match="lacks scene.json or sample.json"):
        scenes.prepare_scene(Path(name))
    write_bundle(archive, name)
    assert (scenes.prepare_scene(Path(name)) / "sample.json").read_text() == name

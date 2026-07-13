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

"""CPU checks for LingBot WebRTC preset assets."""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest
from lingbot.webrtc import server

pytestmark = pytest.mark.ci_cpu


def _write_preset(directory: Path, *, wrapped_events: bool = False) -> None:
    """Write a minimal valid preset-assets directory."""
    directory.mkdir()
    (directory / server.PRESET_FIRST_FRAME_FILENAME).write_bytes(b"\x89PNG\r\n\x1a\n")
    (directory / server.PRESET_PROMPT_FILENAME).write_text(
        "A quiet portrait by a sunny window.\n",
        encoding="utf-8",
    )
    events: object = [
        {
            "event_id": "smile",
            "label": "Smile",
            "prompt": "A warm smile forms on her face.",
            "category": "portrait",
        }
    ]
    if wrapped_events:
        events = {"events": events}
    (directory / server.PRESET_TEXT_EVENTS_FILENAME).write_text(
        json.dumps(events),
        encoding="utf-8",
    )


def _server_args(preset_assets_dir: Path | None) -> Namespace:
    """Build WebRTC CLI arguments without parsing process arguments."""
    return Namespace(
        config_name="lingbot-world-v2-14b-causal-fast",
        no_compile=True,
        device="cuda:0",
        warmup_chunks=0,
        warmup_timeout_s=600.0,
        video_height=352,
        video_width=640,
        example_idx=0,
        preset_assets_dir=preset_assets_dir,
    )


@pytest.mark.parametrize("wrapped_events", [False, True])
def test_load_preset_assets_accepts_supported_event_layouts(
    tmp_path: Path,
    wrapped_events: bool,
) -> None:
    """Load direct and object-wrapped event catalogs."""
    preset_dir = tmp_path / "preset"
    _write_preset(preset_dir, wrapped_events=wrapped_events)

    resolved_dir, events = server.load_preset_assets(preset_dir)

    assert resolved_dir == preset_dir.resolve()
    assert [event.event_id for event in events] == ["smile"]
    assert events[0].prompt == "A warm smile forms on her face."


def test_load_preset_assets_reports_missing_files(tmp_path: Path) -> None:
    """List every required asset missing from an incomplete preset."""
    preset_dir = tmp_path / "preset"
    preset_dir.mkdir()

    with pytest.raises(
        ValueError, match="first_frame.png, prompt.txt, event_texts.json"
    ):
        server.load_preset_assets(preset_dir)


def test_build_runtime_config_uses_preset_assets(tmp_path: Path) -> None:
    """Override the default example frame and event catalog with a preset."""
    preset_dir = tmp_path / "preset"
    _write_preset(preset_dir)

    config = server.build_runtime_config(_server_args(preset_dir))

    assert config.example_data_dir == preset_dir.resolve()
    assert config.first_frame_filename == "first_frame.png"
    assert config.prompt_filename == "prompt.txt"
    assert config.default_image_url is None
    assert config.default_preset_id is None
    assert [event.event_id for event in config.text_events] == ["smile"]


def test_build_runtime_config_identifies_bundled_default_preset() -> None:
    """Mark a bundled launch preset as selected in initial-scene metadata."""
    preset_dir = (
        Path(server.__file__).resolve().parent / "presets" / "golden-hour-portrait"
    )

    config = server.build_runtime_config(_server_args(preset_dir))

    assert config.default_preset_id == "golden-hour-portrait"


def test_parse_args_accepts_preset_assets_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Expose the preset directory through both CLI spellings."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["lingbot.webrtc.server", "--preset-assets-dir", str(tmp_path)],
    )

    args = server.parse_args()

    assert args.preset_assets_dir == tmp_path


def test_bundled_preset_catalog_exposes_picker_metadata() -> None:
    """Load all bundled presets in their stable UI order."""
    presets = server.load_bundled_presets()

    assert [preset.preset_id for preset in presets] == list(server.BUNDLED_PRESET_IDS)
    assert all(preset.prompt for preset in presets)
    assert all(preset.first_frame.data.startswith(b"\x89PNG") for preset in presets)
    assert all(preset.text_events for preset in presets)
    assert presets[0].as_public_dict() == {
        "preset_id": "golden-hour-portrait",
        "label": "Golden Hour Portrait",
        "first_frame_url": "/api/presets/golden-hour-portrait/first_frame",
    }


@pytest.mark.parametrize(
    ("preset_name", "expected_event_ids"),
    [
        (
            "golden-hour-portrait",
            ["hair-tuck", "subtle-smile", "head-turn"],
        ),
        (
            "moonlit-portal",
            ["portal-awakens", "fireflies-gather", "storm-approaches"],
        ),
        (
            "cozy-reading-room",
            ["fire-brightens", "pages-turn", "rain-intensifies"],
        ),
        (
            "misty-dinosaur-valley",
            ["dinosaur-raises-head", "flock-crosses-sky", "mist-rolls-in"],
        ),
    ],
)
def test_bundled_preset_is_valid(
    preset_name: str,
    expected_event_ids: list[str],
) -> None:
    """Keep every bundled preset synchronized with the asset schema."""
    preset_dir = Path(server.__file__).resolve().parent / "presets" / preset_name

    resolved_dir, events = server.load_preset_assets(preset_dir)

    assert resolved_dir == preset_dir
    assert (preset_dir / "first_frame.png").read_bytes().startswith(b"\x89PNG")
    assert (preset_dir / "prompt.txt").read_text(encoding="utf-8").strip()
    assert [event.event_id for event in events] == expected_event_ids

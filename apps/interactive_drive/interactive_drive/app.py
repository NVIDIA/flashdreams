# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive-drive entry point built on the reusable HDMap2V loops."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from hdmap2v.interactive_drive.scene_loader import load_scene_bundle
from hdmap2v.app import (
    BackendFactory,
    Hdmap2VApplicationDefaults,
    Hdmap2VApplicationHooks,
    Hdmap2VApplication,
    SceneLoader,
)


class InteractiveDriveApplication(Hdmap2VApplication):
    """Longer-running driving application with the same mode-neutral v2 API."""

    def __init__(
        self,
        *,
        hooks: Hdmap2VApplicationHooks | None = None,
        backend_factory: BackendFactory | None = None,
        scene_loader: SceneLoader = load_scene_bundle,
        default_scene_resolver: Callable[[], Path] | None = None,
        ui_renderer_factory: Callable[[int, int], Any] | None = None,
    ) -> None:
        super().__init__(
            defaults=Hdmap2VApplicationDefaults(
                title="Interactive Drive",
                slug="interactive-drive",
                backend="world_model" if hooks is not None else "raster",
                total_blocks=600,
            ),
            hooks=hooks,
            backend_factory=backend_factory,
            scene_loader=scene_loader,
            default_scene_resolver=default_scene_resolver,
            ui_renderer_factory=ui_renderer_factory,
        )


__all__ = ["InteractiveDriveApplication"]

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Test-suite-only constants and HUD-output capture state.

Lives in the installed ``interactive_drive`` package (rather than under
``tests/conftest.py``) so root pytest's ``--import-mode=importlib`` can
resolve the imports unambiguously. ``from conftest import SAMPLE_SCENE``
collides across sibling test trees when more than one workspace member
defines a ``conftest.py`` -- the workspace's other ``tests/conftest.py``
(``integrations/omnidreams/ludus-renderer/tests/conftest.py``) would
shadow the sample's one and fail collection with an ``ImportError``.

The leading underscore signals these symbols aren't part of the public
``interactive_drive`` runtime API -- they exist purely so the sample's
test modules and ``conftest.py`` can share a small bit of session state.
"""

from __future__ import annotations

from pathlib import Path

from omnidreams.scenes import local_scene_archive_path

SAMPLE_SCENE: Path = local_scene_archive_path("0d404ff7-2b66-498c-b047-1ed8cded60d4")
"""Optional real USDZ scene, staged by ``omnidreams-prepare``
into the shared cache (``$FLASHDREAMS_CACHE_DIR/omnidreams-scenes/
clipgt-<uuid>.usdz``).

Tests that use this path must silently skip when the file is absent so
the suite stays green on machines / CI runners that haven't fetched the
large asset."""


# Mutable session-level holder for the presenter-device line
# ``test_app_smoke`` scrapes out of the demo's stdout. The sample's
# ``conftest.pytest_sessionfinish`` reads this at the end of the run so
# the operator can see which Vulkan adapter was used. Carried here (not
# on conftest, not on a ``pytest.StashKey``) because the smoke test
# spawns the demo as a subprocess and the simplest way to thread state
# from the stream-pumping reader thread back to the session hook is a
# plain module attribute.
captured_presenter_device: str | None = None

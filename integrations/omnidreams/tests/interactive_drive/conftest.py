# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared pytest fixtures and hooks for the interactive_drive test suite.

The session-level constants (``SAMPLE_SCENE``, ``captured_presenter_device``)
live in :mod:`omnidreams.interactive_drive._sample_assets`, not here, because root
pytest's ``--import-mode=importlib`` can't resolve ``from conftest import X``
unambiguously when other workspace members also ship a ``conftest.py``.
"""

from __future__ import annotations

import sys

import pytest
from omnidreams.interactive_drive import _sample_assets

_CI_TIER_MARKERS = {"ci_cpu", "ci_gpu", "manual"}


def pytest_configure(config: pytest.Config) -> None:
    """Register the subpackage-local pytest markers so unmarked-test
    warnings don't fire on every CI run.

    ``gpu`` and ``xvfb`` are interactive-drive specific (raster backend
    needs CUDA + libGL; the slangpy HUD smoke test spawns its own Xvfb
    via ``pyvirtualdisplay``); the workspace-wide ``ci_cpu`` / ``ci_gpu``
    / ``manual`` tier markers are registered at the workspace root.
    """
    config.addinivalue_line(
        "markers",
        "gpu: test requires an NVIDIA GPU / CUDA driver",
    )
    config.addinivalue_line(
        "markers",
        "xvfb: test uses pyvirtualdisplay/Xvfb (deselected by default outside CI)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Auto-assign the workspace-wide CI tier marker to every sample test.

    Root CI runs ``pytest -m ci_cpu`` and ``pytest -m ci_gpu``; the
    workspace's ``marker_enforcement`` plugin also rejects any test that
    doesn't carry exactly one of ``ci_cpu`` / ``ci_gpu`` / ``manual``.
    Rather than sprinkle ``pytestmark = pytest.mark.ci_cpu`` across 20+
    test modules, infer the right tier from the sample-local markers
    (xvfb takes precedence over gpu so a test that needs both a virtual
    display *and* a GPU still falls into the opt-in ``manual`` bucket
    -- the GPU CI runner image isn't guaranteed to have Xvfb):

    * ``xvfb`` (needs pyvirtualdisplay) -> ``manual``
    * ``gpu`` (raster backend, CUDA dispatch) -> ``ci_gpu``
    * everything else -> ``ci_cpu``

    Tests that already declare a CI tier marker explicitly are left
    alone. Running ``pytest`` from inside the sample dir keeps working
    because the auto-assigned tier markers are additive.
    """
    for item in items:
        existing = {marker.name for marker in item.iter_markers()}
        if existing & _CI_TIER_MARKERS:
            continue
        if "xvfb" in existing:
            item.add_marker(pytest.mark.manual)
        elif "gpu" in existing:
            item.add_marker(pytest.mark.ci_gpu)
        else:
            item.add_marker(pytest.mark.ci_cpu)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Print the Vulkan adapter used by smoke tests."""
    if _sample_assets.captured_presenter_device and sys.__stderr__:
        sys.__stderr__.write(f"\n{_sample_assets.captured_presenter_device}\n")
        sys.__stderr__.flush()

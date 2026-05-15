import sys
from pathlib import Path

import pytest

# ``test_cudaraster_port_invariants.py`` imports its sibling
# ``test_cudaraster_api`` as a plain module for shared harness code.
# Under pytest's default ``--import-mode=prepend`` that worked because
# pytest auto-prepended each collected test's directory to
# ``sys.path``; the workspace-root ``pyproject.toml`` now opts every
# pytest run into ``--import-mode=importlib``, which drops that side
# effect. Re-add the dir explicitly here so the sibling import keeps
# resolving without touching the test files themselves.
_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "gpu: test requires CUDA-capable GPU and torch with CUDA support",
    )


def _cuda_and_plugin_available() -> tuple[bool, str]:
    """Check whether CUDA is usable and the renderer plugin can be loaded."""
    try:
        import torch
    except ModuleNotFoundError:
        return False, "torch is not installed"

    if not torch.cuda.is_available():
        return False, "CUDA is not available"

    try:
        from ludus_renderer._ops._plugin import _get_plugin

        _get_plugin(gl=False)
    except Exception as exc:
        return False, f"ludus_renderer plugin failed to load: {exc}"

    return True, ""


# Evaluate once at collection time.
_GPU_OK, _GPU_SKIP_REASON = _cuda_and_plugin_available()


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if _GPU_OK:
        return

    skip_gpu = pytest.mark.skip(reason=_GPU_SKIP_REASON)
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)

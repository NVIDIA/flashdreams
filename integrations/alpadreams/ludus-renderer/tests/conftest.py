import pytest


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

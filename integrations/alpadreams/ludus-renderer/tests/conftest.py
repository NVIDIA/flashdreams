import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "gpu: test requires CUDA-capable GPU and torch with CUDA support",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    try:
        import torch
    except ModuleNotFoundError:
        torch = None  # type: ignore[ty:invalid-assignment]

    has_cuda = torch is not None and torch.cuda.is_available()
    if has_cuda:
        return

    skip_gpu = pytest.mark.skip(reason="CUDA is required for gpu-marked tests")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)

---
title: 'Testing'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

Pytest collection is configured in the root `pyproject.toml`. There is
no `testpaths` setting, so `pytest` from the repo root walks the tree:

- Files named `test_*.py`. Pytest's default also collects `*_test.py`;
  don't use that name here.
- Classes named `Test*` (pytest default). Some modules use classes,
  some only module-level functions.
- Functions and methods named `test_*` (pytest default), including
  `async def test_*`.

Same test filename in different folders is fine (several
`test_application.py` files exist). Root `pyproject.toml` already sets
`--import-mode=importlib` so pytest keeps them separate. Do not remove
that flag.

A `test_*.py` next to the code it validates is collected with no CI
change. The root `pyproject.toml` skips `parity_check`,
`parity_check_v2`, `baseline_fastvideo`, and `baseline_lightx2v` via
`norecursedirs`.

Every test function must be marked with exactly one **CI tier marker**.
A pytest plugin (`flashdreams._pytest_plugins.marker_enforcement`)
enforces this at collection time -- tests without a marker are
rejected, and tests with both `ci_cpu` and `ci_gpu` are rejected.

| Marker | When to use | CI runner |
|--------|-------------|-----------|
| `@pytest.mark.ci_cpu` | Pure CPU logic, no GPU or `libGL` needed | CPU runner |
| `@pytest.mark.ci_gpu` | Needs CUDA, `libGL`, or transitive `cv2` | GPU runner (RTX Pro 6000) |
| `@pytest.mark.manual` | Heavy (OOM risk), flaky, needs credentials or large downloads | Not run in CI |

Use a module-level `pytestmark` when every test in a file shares the
same marker:

```python
import pytest

pytestmark = pytest.mark.ci_cpu
```

Use per-function markers when tests in the same file have different
tiers:

```python
@pytest.mark.ci_cpu
def test_basic_math(): ...

@pytest.mark.ci_gpu
def test_cudagraph_path(): ...
```

Running tests locally:

```bash
uv run pytest -m ci_cpu          # CPU-safe tests only
uv run pytest -m ci_gpu          # GPU tests only (needs CUDA)
uv run pytest -m "not manual"    # everything that runs in CI
uv run pytest                    # all tests including manual
```

For local and Docker test-runner scripts, see
[tests/README.md](https://github.com/NVIDIA/flashdreams/blob/main/tests/README.md).

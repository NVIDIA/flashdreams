# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import lingbot.runtime as runtime

import pytest

pytestmark = pytest.mark.ci_cpu


def test_runtime_all_exports_are_defined() -> None:
    """Every name in ``lingbot.runtime.__all__`` must exist on the module.

    ``from lingbot.runtime import *`` raises ``AttributeError`` when
    ``__all__`` lists a name the module does not define, so the public API
    list must stay in sync with the module's definitions.
    """
    missing = [name for name in runtime.__all__ if not hasattr(runtime, name)]
    assert not missing, (
        "lingbot.runtime.__all__ references names that are not defined: "
        f"{', '.join(missing)}."
    )

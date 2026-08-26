# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for exception compatibility helpers."""

import pytest

from flashdreams.core.exceptions import add_exception_note

pytestmark = pytest.mark.ci_cpu


def test_add_exception_note_records_context_when_supported() -> None:
    error = RuntimeError("primary failure")

    add_exception_note(error, "cleanup also failed")

    expected = ["cleanup also failed"] if hasattr(BaseException, "add_note") else None
    assert getattr(error, "__notes__", None) == expected


def test_add_exception_note_is_safe_for_python_310_like_errors() -> None:
    class LegacyExceptionLike(Exception):
        def __getattribute__(self, name: str) -> object:
            if name == "add_note":
                raise AttributeError(name)
            return super().__getattribute__(name)

    error = LegacyExceptionLike("primary failure")

    add_exception_note(error, "cleanup also failed")
    assert getattr(error, "__notes__", None) is None

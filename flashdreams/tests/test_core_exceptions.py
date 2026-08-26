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

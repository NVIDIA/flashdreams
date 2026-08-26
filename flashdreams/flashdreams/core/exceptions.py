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

"""Exception compatibility helpers shared across FlashDreams."""


def add_exception_note(error: BaseException, note: str) -> None:
    """Add diagnostic context when the running Python supports exception notes.

    ``BaseException.add_note`` was added in Python 3.11, while FlashDreams
    supports Python 3.10. Cleanup must never replace the primary failure merely
    because that compatibility method is absent.
    """
    add_note = getattr(error, "add_note", None)
    if callable(add_note):
        add_note(note)


__all__ = ["add_exception_note"]

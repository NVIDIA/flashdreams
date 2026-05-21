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

"""``FLASHDREAMS_INTERNAL_STORAGE=1`` flips ``AVAILABLE_*_CHECKPOINT_PATHS`` and
the omnidreams ``--example-data`` source from the public HF URLs to their
``s3://flashdreams`` counterparts the team iterates on. Default unset = public.
"""

from __future__ import annotations

import os
from typing import Final

INTERNAL_STORAGE_ENV_VAR: Final[str] = "FLASHDREAMS_INTERNAL_STORAGE"


def use_internal_storage() -> bool:
    """True iff the env var is set to ``1`` or ``true`` (case-insensitive)."""
    return os.environ.get(INTERNAL_STORAGE_ENV_VAR, "").lower() in ("1", "true")

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-checkout wrappers for FlashDreams developer tools."""

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

_project_tools = Path(__file__).resolve().parents[1] / "flashdreams" / "tools"
if _project_tools.is_dir():
    __path__.append(str(_project_tools))

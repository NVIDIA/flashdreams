# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public surface for the runner registry and CLI aggregator."""

from flashdreams.configs.registry import register_runner, supported_runners
from flashdreams.configs.runner_configs import all_runners

__all__ = [
    "all_runners",
    "register_runner",
    "supported_runners",
]

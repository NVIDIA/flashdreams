# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Crazy Robotaxi through the canonical FlashDreams V2 host."""

import sys

from flashdreams.runtime_v2.cli import entrypoint

if __name__ == "__main__":
    entrypoint(["crazy-robotaxi", *sys.argv[1:]])

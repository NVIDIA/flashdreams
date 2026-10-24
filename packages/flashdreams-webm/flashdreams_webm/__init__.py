# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-process AV1 WebM recording and decoding without FFmpeg."""

from ._native import Reader, Writer

__all__ = ["Reader", "Writer"]

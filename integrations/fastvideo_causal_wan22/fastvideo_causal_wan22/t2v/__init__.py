# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastVideo CausalWan 2.2 T2V public demo app."""

from fastvideo_causal_wan22.t2v.app import MODEL, create_app, createApp

__all__ = ["MODEL", "createApp", "create_app"]

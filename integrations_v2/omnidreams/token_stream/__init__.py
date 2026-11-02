# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OmniDreams token-stream support: VAE-decoder export and identity handshake.

The generic wire transport lives in
:mod:`flashdreams.runtime_v2.serving.token_stream`; this package holds the
omnidreams-specific pieces that decide *which* decoder the browser must run and
produce the exported ONNX it decodes with.
"""

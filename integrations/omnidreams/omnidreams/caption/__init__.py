# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline training pipeline for the latent-native live-caption model.

The serving side lives in :mod:`omnidreams.webrtc.caption_artifacts` (locate an
exported ONNX + spec and describe it to the browser client). This package is the
*offline* counterpart that produces that artifact:

- :mod:`omnidreams.caption.shards` — WebDataset-compatible tar shard writer used
  by the capture harness (Phase 1) to persist ``(latent-window, RGB-window,
  metadata)`` records.

Later phases add ``capture`` (drive the world model with scripted controls and
emit shards), ``label_teacher`` (Cosmos-Reason1-7B distillation labels),
``dataset``/``train`` (the caption CNN), and ``export`` (ONNX + spec into the
caption-model cache dir). None of this ships in the runtime serving path.
"""

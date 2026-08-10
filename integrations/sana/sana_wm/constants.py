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

"""Constants for the public SANA-WM releases."""

SANA_WM_HF_REPO = "Efficient-Large-Model/SANA-WM_bidirectional"
"""Hugging Face repository containing SANA-WM bidirectional artefacts."""

SANA_WM_STREAMING_HF_REPO = "Efficient-Large-Model/SANA-WM_streaming"
"""Hugging Face repository containing SANA-WM streaming artefacts."""

_SANA_WM_HF_FILE_BASE = f"https://huggingface.co/{SANA_WM_HF_REPO}/resolve/main"
"""Hugging Face file-URL base consumed by ``load_checkpoint``."""

_SANA_WM_STREAMING_HF_FILE_BASE = (
    f"https://huggingface.co/{SANA_WM_STREAMING_HF_REPO}/resolve/main"
)
"""Streaming Hugging Face file-URL base consumed by ``load_checkpoint``."""

SANA_WM_MODEL_PATH = f"{_SANA_WM_HF_FILE_BASE}/dit/sana_wm_1600m_720p.safetensors"
"""Default Stage-1 SANA-WM DiT checkpoint."""

SANA_WM_STREAMING_MODEL_PATH = f"{_SANA_WM_STREAMING_HF_FILE_BASE}/sana_dit/model.pt"
"""Default streaming Stage-1 SANA-WM DiT checkpoint."""

SANA_WM_CONFIG_PATH = f"hf://{SANA_WM_HF_REPO}/config.yaml"
"""Default inference YAML."""

SANA_WM_STREAMING_CONFIG_PATH = "flashdreams://sana-wm-streaming-1600m-720p"
"""Built-in FlashDreams config identifier for the streaming inference YAML."""

SANA_WM_REFINER_ROOT = f"hf://{SANA_WM_HF_REPO}/refiner"
"""Default LTX-2 refiner root."""

SANA_WM_REFINER_GEMMA_ROOT = f"hf://{SANA_WM_HF_REPO}/refiner/text_encoder"
"""Default Gemma text-encoder root used by the refiner."""

SANA_WM_STREAMING_CAUSAL_VAE_ROOT = f"hf://{SANA_WM_STREAMING_HF_REPO}/ltx2_causal_vae"
"""Default causal LTX-2 VAE root used by streaming SANA-WM."""

SANA_WM_STREAMING_REFINER_ROOT = f"hf://{SANA_WM_STREAMING_HF_REPO}/refiner_diffusers"
"""Default chunk-causal LTX-2 refiner root used by streaming SANA-WM."""

SANA_WM_STREAMING_REFINER_GEMMA_ROOT = f"hf://{SANA_WM_STREAMING_HF_REPO}/gemma3_12b"
"""Default Gemma-3 text-encoder root used by the streaming refiner."""

DEFAULT_VIDEO_HEIGHT = 704
"""SANA-WM output height in pixels."""

DEFAULT_VIDEO_WIDTH = 1280
"""SANA-WM output width in pixels."""

SANA_WM_VAE_TEMPORAL_COMPRESSION = 8
"""Temporal compression ratio of the LTX2 VAE used by SANA-WM."""

SANA_WM_VAE_SPATIAL_COMPRESSION = 32
"""Spatial compression ratio of the LTX2 VAE used by SANA-WM."""

DEFAULT_FPS = 16
"""Frame rate used by the public SANA-WM examples."""

DEFAULT_ACTION = "w-100,dw-60,w-100,aw-60"
"""Default SANA-WM demo action string."""

DEFAULT_STREAMING_NUM_FRAMES = 241
"""Default streaming output frame count."""

SANA_WM_STREAMING_LATENT_CHUNK_SIZE = 3
"""Default number of latent frames generated per streaming AR block."""

DEFAULT_STREAMING_DENOISING_STEP_LIST = (1000, 960, 889, 727, 0)
"""Default distilled Stage-1 timestep schedule for streaming SANA-WM."""

SANA_WM_STREAMING_REFINER_KV_MAX_FRAMES = 11
"""Default refiner sliding-window size in latent frames."""

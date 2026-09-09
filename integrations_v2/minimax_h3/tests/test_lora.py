# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU coverage of checkpoint-native Musubi adapter conversion and merging."""

import pytest
import torch
from safetensors.torch import load_file, save_file

from minimax_h3.impl import lora

pytestmark = pytest.mark.ci_cpu


def test_musubi_conversion_preserves_native_targets(tmp_path):
    tensors = {}
    for block in range(50):
        for module in ("attn_qkv_proj", "attn_out_proj", "mlp_fc1", "mlp_fc2"):
            prefix = f"lora_unet_blocks_{block}_{module}"
            tensors[f"{prefix}.alpha"] = torch.tensor(2.0)
            tensors[f"{prefix}.lora_down.weight"] = torch.ones(2, 3)
            tensors[f"{prefix}.lora_up.weight"] = torch.ones(
                6 if module == "attn_qkv_proj" else 4, 2
            )
    source = tmp_path / "adapter.safetensors"
    save_file(tensors, source)
    converted = load_file(
        lora.convert_musubi_lora(source, tmp_path / "native.safetensors")
    )
    assert len(converted) == 600
    assert "transformer.transformer_blocks.0.attn.to_q.lora_A.weight" in converted
    assert "transformer.transformer_blocks.0.attn.to_v.lora_B.weight" in converted
    assert "transformer.transformer_blocks.49.ff.net.2.lora_B.weight" in converted


def test_native_lora_merge_uses_requested_scale(monkeypatch, tmp_path):
    model = torch.nn.Module()
    model.projection = torch.nn.Linear(3, 2, bias=False)
    original = model.projection.weight.detach().clone()
    path = tmp_path / "converted.safetensors"
    save_file(
        {
            "transformer.projection.lora_A.weight": torch.ones(1, 3),
            "transformer.projection.lora_B.weight": torch.ones(2, 1),
        },
        path,
    )
    monkeypatch.setattr(lora, "prepare_lora", lambda *args: path)
    lora.load_lora(model, "unused", scale=0.5)
    torch.testing.assert_close(model.projection.weight, original + 0.5)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU checks for compact compiled streaming caches."""

import pytest
import torch


@pytest.mark.ci_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_compiled_cache_tail_has_compact_storage_and_stable_pointer() -> None:
    from flashdreams.infra.compile import compile_module
    from flashdreams.recipes.wan.autoencoder.vae import Decoder3d

    decoder = Decoder3d(dim=8, z_dim=4).to(device="cuda", dtype=torch.bfloat16).eval()
    for module in decoder.modules():
        if isinstance(module, torch.nn.Conv3d):
            module.weight.data = module.weight.to(memory_format=torch.channels_last_3d)
        elif isinstance(module, torch.nn.Conv2d):
            module.weight.data = module.weight.to(memory_format=torch.channels_last)
    compiled = compile_module(decoder)
    state: dict[int, torch.Tensor] = {}
    pointers: dict[int, int] | None = None
    x = torch.randn(1, 4, 4, 4, 8, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        for step in range(4):
            output = compiled(x + step, state)
            for slot in state.values():
                assert (
                    slot.untyped_storage().nbytes()
                    == slot.numel() * slot.element_size()
                )
            assert torch.isfinite(output).all()
            current = {key: value.data_ptr() for key, value in state.items()}
            if pointers is not None:
                assert current == pointers
            pointers = current

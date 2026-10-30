# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-weight parity check for the TAEHV cache-as-IO ONNX export.

Exports the ``lighttae`` decoder to the streaming ONNX, then drives three latent
chunks through both the native :class:`TaehvVAEDecoder` and the exported graph via
``onnxruntime`` -- threading the ONNX ``cache_out`` back as the next ``cache_in``,
exactly as the browser client does -- and asserts they match. The exported graph
is always steady-state (a non-empty cache means no first-chunk trim), so chunk 0's
native output is the trimmed tail of the ONNX output; later chunks match frame for
frame.

Marked ``manual``: it needs the optional ``onnx``/``onnxruntime`` dependencies and
downloads the ``lighttae`` checkpoint, both out of scope for CI tiers. Run with::

    uv run --group test --inexact --with onnx --with onnxruntime --with onnxscript \\
        pytest flashdreams/tests/test_taehv_export.py -m manual -s
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.manual


def test_exported_onnx_matches_native_streaming_decode(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")

    from flashdreams.recipes.taehv import TaehvVAEDecoderConfig
    from flashdreams.recipes.taehv.export import export_streaming_decoder

    torch.manual_seed(0)
    device, dtype = "cpu", torch.float32
    # Small spatial dims: parity is a property of the graph and the temporal
    # cache, not the resolution, so this keeps the CPU decode cheap.
    t_l, c_l, h_l, w_l = 2, 16, 24, 40

    decoder = (
        TaehvVAEDecoderConfig(dtype=dtype, use_cuda_graph=False, use_compile=False)
        .setup()
        .to(device)
        .eval()
    )

    out = tmp_path / "taehv_stream.onnx"
    spec = export_streaming_decoder(
        decoder,
        out_path=out,
        latent_frames=t_l,
        latent_channels=c_l,
        latent_height=h_l,
        latent_width=w_l,
        device=device,
        dtype=dtype,
        version="test",
    )
    assert spec["latent_shape"] == [t_l, c_l, h_l, w_l]
    assert spec["cache"], "export produced no cache tensors"

    latents = [torch.randn(1, 1, t_l, c_l, h_l, w_l, dtype=dtype) for _ in range(3)]

    cache = decoder.initialize_autoregressive_cache()
    native = [
        decoder(input=z, autoregressive_index=i, cache=cache).float()
        for i, z in enumerate(latents)
    ]

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1  # avoid thread-affinity noise in sandboxes
    session = ort.InferenceSession(
        str(out), sess_options=options, providers=["CPUExecutionProvider"]
    )
    in_names, out_names = spec["input_names"], spec["output_names"]
    cache_vals = [np.zeros(c["shape"], dtype=np.float32) for c in spec["cache"]]
    onnx_out: list[torch.Tensor] = []
    for z in latents:
        feeds = {in_names[0]: z.numpy()}
        feeds.update(dict(zip(in_names[1:], cache_vals)))
        result = session.run(out_names, feeds)
        onnx_out.append(torch.from_numpy(np.asarray(result[0])))
        cache_vals = [np.asarray(r) for r in result[1:]]  # thread cache forward

    for i, (n, o) in enumerate(zip(native, onnx_out)):
        assert torch.isfinite(n).all() and torch.isfinite(o).all()
        # ``[B, V, T, C, H, W]``: the exported (steady-state) output has the
        # cold-start trim's frames still attached, so chunk 0 matches the tail.
        trim = o.shape[2] - n.shape[2]
        assert trim >= 0
        max_err = (n - o[:, :, trim:]).abs().max().item()
        assert max_err < 1e-3, f"AR{i} ONNX deviates from native: {max_err:.2e}"

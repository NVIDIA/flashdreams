# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise exported weights, control encoding, eviction and reset on a GPU.

Set ``COSMOS3_TRANSFER_RELEASE`` and ``COSMOS3_TRANSFER_VAE`` to local assets.
``COSMOS3_TRANSFER_ACCEL`` selects eager (default), graphs or compile. These
synthetic controls check the streaming contract, not reference-model quality.
Run with ``pytest -p no:manual_marker -m manual tests/test_stream_gpu.py``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
import torch

from cosmos3_nano_sim_transfer import TransferStream

pytestmark = pytest.mark.manual
RELEASE = os.environ.get("COSMOS3_TRANSFER_RELEASE")
VAE = os.environ.get("COSMOS3_TRANSFER_VAE")


@pytest.fixture(scope="module")
def loaded_stream() -> Iterator[TransferStream]:
    if not RELEASE or not VAE or not torch.cuda.is_available():
        pytest.skip("needs CUDA, COSMOS3_TRANSFER_RELEASE and COSMOS3_TRANSFER_VAE")
    stream = TransferStream.load(RELEASE, vae=VAE, history="3/1")
    try:
        yield stream
    finally:
        stream.close()


@pytest.mark.parametrize("hint", ["seg", "depth"])
def test_exported_stream_eviction_and_reset(
    loaded_stream: TransferStream, hint: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    network, encoder, decoder = loaded_stream._parts()
    stream = TransferStream(
        network,
        loaded_stream.model,
        encoder=encoder,
        decoder=decoder,
        tokenizer_path=loaded_stream.tokenizer_path,
        hint=hint,
        history="3/1",
        accel=os.environ.get("COSMOS3_TRANSFER_ACCEL", "eager"),
    )
    try:
        prompt = "A car travels along a city street in daylight."
        stream.warm_up(prompt, seed=42)
        stream.start(prompt, seed=42)
        assert stream.rollout is not None
        step = stream.rollout.step

        def finite_step(*args, **kwargs):
            latent = step(*args, **kwargs)
            assert torch.isfinite(latent).all(), "nonfinite denoised latent"
            return latent

        monkeypatch.setattr(stream.rollout, "step", finite_step)
        # Half-resolution inputs exercise nearest segmentation and bilinear depth.
        ramp = torch.arange(416, device="cuda").remainder(256).to(torch.uint8)
        frame = ramp[None, :, None].expand(240, 416, 3).contiguous()
        if hint == "seg":
            frame = (frame // 64) * 64
        cached = []
        first = None
        for index in range(8):
            control = frame[None].expand(stream.frames_for(index), -1, -1, -1)
            output = stream.step_tensor(control)
            assert output.shape == (stream.frames_for(index), 3, 480, 832)
            assert output.dtype == torch.uint8
            if index == 0:
                first = output.clone()
            cached.append(stream.cached_tokens)
        assert cached[2] > cached[0]
        assert len(set(cached[2:])) == 1, "history must stay bounded after eviction"
        assert stream.reset() == "reset"
        assert first is not None
        torch.testing.assert_close(
            stream.step_tensor(frame[None]), first, rtol=0, atol=0
        )
        stream.step_tensor(frame[None].expand(4, -1, -1, -1), last=True)
        with pytest.raises(RuntimeError, match="session ended"):
            stream.step_tensor(frame[None].expand(4, -1, -1, -1))
        if stream.accel != "eager":
            assert stream.graphs_captured > 0
    finally:
        stream.close()

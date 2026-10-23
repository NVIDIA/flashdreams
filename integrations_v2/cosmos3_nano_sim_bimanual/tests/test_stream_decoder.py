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

"""TAEHV2 Super topology, checkpoint loading, and streaming-cache behavior."""

from __future__ import annotations

import io
import urllib.request

import pytest
import torch
from cosmos3_nano_sim_bimanual.decoder import network
from cosmos3_nano_sim_bimanual.decoder.network import (
    TAEHV2SuperDecoder,
    _TemporalGrow,
)
from safetensors.torch import save as save_safetensors
from safetensors.torch import save_file

pytestmark = pytest.mark.ci_cpu


class _TinyTAEHV2SuperDecoder(TAEHV2SuperDecoder):
    """Small topology preserving the production decoder's temporal contracts."""

    latent_channels = 4
    stage_channels = (8, 8, 8, 8)
    memory_blocks_per_stage = 1


def _copy_decoder(decoder: TAEHV2SuperDecoder) -> _TinyTAEHV2SuperDecoder:
    copy = _TinyTAEHV2SuperDecoder()
    copy.load_state_dict(decoder.state_dict(), strict=True)
    return copy.eval()


def test_temporal_grow_expands_flattened_time() -> None:
    grow = _TemporalGrow(channels=3, stride=2)
    features = torch.randn(4, 3, 2, 2)
    assert grow(features).shape == (8, 3, 2, 2)


def test_streamed_chunks_match_one_full_decode_with_fresh_cache() -> None:
    torch.manual_seed(7)
    reference = _TinyTAEHV2SuperDecoder().eval()
    streamed = _copy_decoder(reference)
    latent = torch.randn(1, reference.latent_channels, 3, 1, 1)

    expected = reference(latent)
    cache = streamed.prepare_cache()
    first = streamed(latent[:, :, :1], cache=cache)
    remainder = streamed(latent[:, :, 1:], cache=cache)
    actual = torch.cat([first, remainder], dim=2)

    assert first.shape == (1, 3, 1, 16, 16)
    assert remainder.shape == (1, 3, 8, 16, 16)
    torch.testing.assert_close(actual, expected)
    assert actual.min() >= 0 and actual.max() <= 1

    fresh = streamed.prepare_cache()
    torch.testing.assert_close(streamed(latent[:, :, :1], cache=fresh), first)


@pytest.mark.parametrize("suffix", [".pth", ".safetensors"])
def test_checkpoint_loads_only_the_decoder_namespace_strictly(tmp_path, suffix) -> None:
    torch.manual_seed(11)
    source = _TinyTAEHV2SuperDecoder()
    payload = {
        "encoder.unused": torch.zeros(1),
        **{
            f"decoder.{key}": value.clone()
            for key, value in source.decoder.state_dict().items()
        },
    }
    checkpoint = tmp_path / f"taew2_2_super{suffix}"
    if suffix == ".safetensors":
        save_file(payload, checkpoint)
    else:
        torch.save(payload, checkpoint)

    loaded = _TinyTAEHV2SuperDecoder(checkpoint)

    for expected, actual in zip(
        source.decoder.state_dict().values(),
        loaded.decoder.state_dict().values(),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)


def test_default_decoder_download_is_cached_without_credentials(tmp_path, monkeypatch):
    import cosmos3_nano_sim_bimanual.checkpoint as checkpoint_module

    source = _TinyTAEHV2SuperDecoder()
    payload = {
        f"decoder.{key}": value for key, value in source.decoder.state_dict().items()
    }
    calls = []

    def fetch(request, *, timeout):
        calls.append(request.full_url)
        assert len(calls) == 1, "cached decoder must not need network access"
        return io.BytesIO(save_safetensors(payload))

    def reject_credentials(*args, **kwargs):
        pytest.fail("public decoder must not resolve S3 credentials")

    monkeypatch.setenv("FLASHDREAMS_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("FLASHDREAMS_MIN_CACHE_FREE_GB", "0")
    monkeypatch.setenv("COSMOS_CHECKPOINT_CREDENTIAL", "/missing/credentials")
    monkeypatch.setattr(urllib.request, "urlopen", fetch)
    monkeypatch.setattr(
        checkpoint_module, "resolve_credential_path", reject_credentials
    )
    monkeypatch.setattr(network, "TAEHV2SuperDecoder", _TinyTAEHV2SuperDecoder)
    for _ in range(2):
        loaded = network.make_taehv2super_decoder(device="cpu", dtype=torch.float32)
        for key, expected in source.decoder.state_dict().items():
            torch.testing.assert_close(loaded.decoder.state_dict()[key], expected)
        assert not loaded.training
        assert not any(parameter.requires_grad for parameter in loaded.parameters())
    assert calls == [network.TAEHV2SUPER_CHECKPOINT]
    assert list(tmp_path.rglob("*.safetensors"))
    assert not list(tmp_path.rglob("*.part"))


@pytest.mark.parametrize("invalid", ["malformed", "missing", "extra", "shape"])
def test_invalid_decoder_download_does_not_populate_cache(
    tmp_path, monkeypatch, invalid
):
    source = _TinyTAEHV2SuperDecoder()
    payload = {
        f"decoder.{key}": value for key, value in source.decoder.state_dict().items()
    }
    key = next(iter(payload))
    if invalid == "missing":
        payload.pop(key)
    elif invalid == "extra":
        payload["decoder.unexpected"] = torch.zeros(1)
    elif invalid == "shape":
        payload[key] = torch.zeros(1)
    response = (
        b"not safetensors" if invalid == "malformed" else save_safetensors(payload)
    )
    monkeypatch.setenv("FLASHDREAMS_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("FLASHDREAMS_MIN_CACHE_FREE_GB", "0")
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(response)
    )
    with pytest.raises(RuntimeError, match="Failed to download"):
        _TinyTAEHV2SuperDecoder(network.TAEHV2SUPER_CHECKPOINT)
    assert not list(tmp_path.rglob("*.safetensors"))
    assert not list(tmp_path.rglob("*.part"))


@pytest.mark.parametrize(
    "override",
    [
        "custom-decoder.safetensors",
        "https://huggingface.co/example/decoder/resolve/main/weights.safetensors",
        "s3://example/decoder.pth",
    ],
)
def test_decoder_overrides_use_existing_checkpoint_loader(monkeypatch, override):
    from flashdreams.core.checkpoint import load
    from flashdreams.core.io import download

    source = _TinyTAEHV2SuperDecoder()
    payload = {
        f"decoder.{key}": value for key, value in source.decoder.state_dict().items()
    }
    calls = []

    def load_checkpoint(path, **kwargs):
        calls.append((path, kwargs))
        return payload

    def reject_download(*args, **kwargs):
        pytest.fail("custom decoder must use the existing checkpoint loader")

    monkeypatch.setattr(load, "load_single_checkpoint", load_checkpoint)
    monkeypatch.setattr(download, "download_to_cache", reject_download)
    _TinyTAEHV2SuperDecoder(override, credential_path="explicit-credential")
    assert calls == [
        (override, {"credential_path": "explicit-credential", "map_location": "cpu"})
    ]


def test_cached_s3_decoder_override_does_not_require_credentials(tmp_path, monkeypatch):
    import cosmos3_nano_sim_bimanual.checkpoint as checkpoint_module

    from flashdreams.core.checkpoint import load

    source = _TinyTAEHV2SuperDecoder()
    payload = {
        f"decoder.{key}": value for key, value in source.decoder.state_dict().items()
    }
    cached = tmp_path / "cached-decoder.pth"
    torch.save(payload, cached)

    def reject_s3(*args, **kwargs):
        pytest.fail("cached S3 override must not access object storage")

    monkeypatch.setattr(load, "_s3_cache_path", lambda *args, **kwargs: str(cached))
    monkeypatch.setattr(load, "_load_checkpoint_from_s3", reject_s3)
    monkeypatch.setattr(
        checkpoint_module, "resolve_credential_path", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(network, "TAEHV2SuperDecoder", _TinyTAEHV2SuperDecoder)
    loaded = network.make_taehv2super_decoder(
        checkpoint_path="s3://example/decoder.pth", device="cpu", dtype=torch.float32
    )
    for key, expected in source.decoder.state_dict().items():
        torch.testing.assert_close(loaded.decoder.state_dict()[key], expected)


def test_decoder_rejects_wrong_latent_shape() -> None:
    decoder = _TinyTAEHV2SuperDecoder()
    with pytest.raises(ValueError, match="expected latent"):
        decoder(torch.randn(1, 3, 1, 1, 1))

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for checkpoint compatibility and the native codec's boundaries."""

import hashlib
import json
import subprocess
import sys
import sysconfig
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from cosmos_interactive._lidar_codec import load_checkpoint, network_config
from cosmos_interactive._lidar_codec.network import TransformerVAE
from cosmos_interactive._lidar_codec.range_image import (
    metric_lidar_to_network,
    network_lidar_to_metric_clip,
)
from cosmos_interactive.lidar import (
    LidarDecoder,
    LidarEncoder,
    LidarGeometry,
    LidarTokenizer,
)

pytestmark = pytest.mark.ci_cpu


@pytest.mark.parametrize(
    "version,expected",
    [
        ("v1p1", "92a8636ebe5fb836176d429a2e7f7cf5da837313cab4a7c6f4376f9587f8c951"),
        ("v1p2", "cea838c332b671bbff4e4bd019972df200c796d3fc37ea844b112ff82e67ad2b"),
    ],
)
def test_checkpoint_keys_and_shapes_match_pinned_reference(version, expected):
    # Reference fingerprints include all checkpoint parameters and persistent buffers.
    with torch.device("meta"):
        model = TransformerVAE(**network_config(version))
    shapes = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    assert (
        hashlib.sha256(json.dumps(shapes, sort_keys=True).encode()).hexdigest()
        == expected
    )


def test_metric_boundary_and_invalid_ray_conventions():
    raw = torch.tensor(
        [
            [0, 4.99, 5, 52.5, 100, 100.01, 20, float("nan")],
            [0, 0, -1, 0.25, 2, 0.5, 0.5, 0],
            [1, 1, 0.5, 1, 1, 1, 0.499, 1],
        ]
    ).reshape(1, 3, 1, 1, 8)
    actual, mask = metric_lidar_to_network(raw, min_range=5, max_range=100)
    expected = torch.tensor(
        [
            [-1, -1, -1, 0, 1, -1, -1, -1],
            [-1, -1, -1, -0.5, 1, -1, -1, -1],
            [0, 0, 1, 1, 1, 0, 0, 0],
        ]
    ).reshape_as(raw)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(mask, expected[:, 2:3])
    decoded = network_lidar_to_metric_clip(actual, mask, min_range=5, max_range=100)
    expected_metric = torch.tensor(
        [
            [0, 0, 5, 52.5, 100, 0, 0, 0],
            [0, 0, 0, 0.25, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0],
        ]
    ).reshape_as(raw)
    torch.testing.assert_close(decoded, expected_metric, rtol=0, atol=0)
    # Validity uses >= 0.5; output range/intensity clamp prediction overshoots.
    output = torch.tensor([-2, 2, 0]).reshape(1, 3, 1, 1, 1)
    decoded = network_lidar_to_metric_clip(
        output, torch.full((1, 1, 1, 1, 1), 0.5), min_range=5, max_range=100
    )
    torch.testing.assert_close(decoded.flatten(), torch.tensor([5.0, 1.0, 1.0]))


def tiny_model():
    # One joint-attention level runs on CPU without substituting GPU kernels.
    return TransformerVAE(
        resolution=(4, 8), depths=(2,), num_heads=(2,), base_channels=32, z_dim=8
    )


@pytest.mark.parametrize(
    "prefix", ["", "network.", "model.network.", "vae.", "model.vae."]
)
@pytest.mark.parametrize("format", ["pt", "safetensors"])
def test_local_checkpoint_roundtrip(tmp_path, prefix, format):
    model = tiny_model()
    state = {prefix + key: value for key, value in model.state_dict().items()}
    state.update(
        latent_mean=torch.arange(8).float(),
        latent_std=torch.arange(1, 9).float(),
        min_range=torch.tensor(5.0),
        max_range=torch.tensor(100.0),
    )
    path = tmp_path / f"checkpoint.{format}"
    if format == "pt":
        torch.save({"model": state, "iteration": 30_000}, path)
    else:
        from safetensors.torch import save_file

        save_file(state, path)
    restored = tiny_model()
    mean, std = load_checkpoint(restored, str(path))
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    torch.testing.assert_close(mean.flatten(), torch.arange(8).float())
    torch.testing.assert_close(std.flatten(), torch.arange(1, 9).float())


@pytest.mark.parametrize(
    "fault,match",
    [
        ("missing", "missing="),
        ("shape", "mismatched="),
        ("extra", "unexpected="),
        ("unknown_ema", "unexpected="),
        ("duplicate", "Duplicate"),
        ("no_mean", "missing latent_mean"),
        ("zero_std", "positive"),
        ("nan_std", "finite"),
        ("inf_mean", "finite"),
        ("wrong_channels", "8 values"),
        ("range", "range must match"),
        ("half_range", "both min_range"),
        ("uninitialized", "not initialized"),
    ],
)
def test_checkpoint_rejects_incompatible_weights_and_statistics(tmp_path, fault, match):
    model = tiny_model()
    state = {"network." + key: value for key, value in model.state_dict().items()}
    state.update(
        latent_mean=torch.zeros(8),
        latent_std=torch.ones(8),
        min_range=torch.tensor(5.0),
        max_range=torch.tensor(100.0),
    )
    key = "network.quant_conv.weight"
    if fault == "missing":
        del state[key]
    elif fault == "shape":
        state[key] = torch.zeros(1)
    elif fault == "extra":
        state["network.unexpected.weight"] = torch.zeros(1)
    elif fault == "unknown_ema":
        state["ema.network-unknown-weight"] = torch.zeros(1)
    elif fault == "duplicate":
        state["quant_conv.weight"] = state[key]
    elif fault == "no_mean":
        del state["latent_mean"]
    elif fault == "zero_std":
        state["latent_std"][0] = 0
    elif fault == "nan_std":
        state["latent_std"][0] = float("nan")
    elif fault == "inf_mean":
        state["latent_mean"][0] = float("inf")
    elif fault == "wrong_channels":
        state["latent_mean"] = torch.zeros(9)
    elif fault == "range":
        state["min_range"] = torch.tensor(0.0)
    elif fault == "half_range":
        del state["max_range"]
    elif fault == "uninitialized":
        state["_latent_stats_initialized"] = torch.tensor(False)
    path = tmp_path / "checkpoint.pt"
    torch.save({"model": state}, path)
    with pytest.raises((ValueError, RuntimeError), match=match):
        load_checkpoint(model, str(path))


@torch.inference_mode()
def test_streaming_is_causal_bounded_and_resettable():
    torch.manual_seed(3)
    model = tiny_model().eval()
    # Trained projections are not zero. Exercise attention/FFN paths rather
    # than accidentally testing the zero-initialized residual identities.
    for name, parameter in model.named_parameters():
        if name.endswith(("out_proj.weight", "ffn.linear.weight")):
            torch.nn.init.normal_(parameter, std=0.03)
    backend = SimpleNamespace(
        model=model,
        device="cpu",
        dtype=torch.float32,
        min_range=5.0,
        max_range=100.0,
        validity_threshold=0.5,
        latent_mean=torch.arange(8).reshape(1, 8, 1, 1, 1) * 0.1,
        latent_std=torch.full((1, 8, 1, 1, 1), 1.5),
    )
    tokenizer = LidarTokenizer(backend, LidarGeometry(channels=8, height=2, width=4))
    encoder, decoder = LidarEncoder(tokenizer), LidarDecoder(tokenizer)
    raw = torch.rand(1, 3, 13, 4, 8)
    raw[:, 0] = raw[:, 0] * 95 + 5
    raw[:, 2] = 1
    start = 0
    encoded = []
    for count in (1, 4, 4, 4):
        z = encoder(raw[:, :, start : start + count])
        decoded = decoder(z)
        assert decoded.shape == (1, 3, count, 4, 8)
        assert torch.isfinite(decoded).all()
        encoded.append(z)
        start += count
        for stream in (encoder, decoder):
            assert {k.shape[2] for k, _ in stream.cache.values()} == {min(9, start)}
    encoder.reset()
    decoder.reset()
    assert encoder.cache is None and decoder.cache is None
    torch.testing.assert_close(encoder(raw[:, :, :1]), encoded[0], rtol=0, atol=0)
    # Before eviction, split chunks equal a causal prefix evaluated at once.
    prefix = LidarEncoder(tokenizer)(raw[:, :, :9])
    torch.testing.assert_close(
        torch.cat(encoded[:3], dim=2), prefix, atol=2e-6, rtol=2e-5
    )
    altered = raw[:, :, :9].clone()
    altered[:, :, 1:] = 0
    torch.testing.assert_close(
        LidarEncoder(tokenizer)(altered)[:, :, :1], encoded[0], atol=2e-6, rtol=2e-5
    )


def test_native_import_has_no_optional_gpu_kernel_dependency():
    script = """
import importlib.abc
import sys
class BlockOptionalKernel(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'natten', 'imaginaire', 'projects', 'cosmos_auto', 'cosmos_interactive_internal'}:
            raise AssertionError('Unexpected dependency: ' + fullname)
sys.meta_path.insert(0, BlockOptionalKernel())
from cosmos_interactive._lidar_codec.network import TransformerVAE
import torch
with torch.device('meta'):
    model = TransformerVAE(resolution=(128,1808), depths=(3,3,3,3), num_heads=(4,4,8,8))
from cosmos_interactive.lidar import LidarTokenizer
assert model.temporal_compression == 1
"""
    # -I -S excludes other workspace packages, including their editable .pth files.
    paths = [sysconfig.get_path("purelib"), sysconfig.get_path("platlib")]
    for name in ("flashdreams.core", "cosmos_interactive"):
        spec = find_spec(name)
        assert spec is not None and spec.origin is not None
        paths.append(str(Path(spec.origin).resolve().parents[len(name.split("."))]))
    script = f"import sys\nsys.path[:0] = {paths!r}\n" + script
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_checkpoint_ignores_training_ema_without_substituting_it(tmp_path):
    model = tiny_model()
    state = {"network." + key: value for key, value in model.state_dict().items()}
    state.update(
        {
            "ema.network-" + name.replace(".", "-"): value.detach() + 100
            for name, value in model.named_parameters()
        }
    )
    state.update(latent_mean=torch.zeros(8), latent_std=torch.ones(8))
    path = tmp_path / "checkpoint.pt"
    torch.save({"model": state}, path)
    restored = tiny_model()
    load_checkpoint(restored, str(path))
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    del state["network.quant_conv.weight"]
    torch.save({"model": state}, path)
    with pytest.raises(RuntimeError, match="missing=.*quant_conv.weight"):
        load_checkpoint(restored, str(path))


@pytest.mark.parametrize("levels", [3, 4])
@torch.inference_mode()
def test_vllm_spatial_levels_preserve_causal_stream_and_cache_ownership(levels):
    torch.manual_seed(123)
    model = TransformerVAE(
        resolution=(16, 32),
        depths=(1,) * levels,
        num_heads=(2,) * levels,
        base_channels=16,
        z_dim=8,
        window_size=(1, 3),
        mlp_ratio=2.0,
    ).eval()
    for name, parameter in model.named_parameters():
        if name.endswith(
            ("out_proj.weight", "ffn.linear.weight", "residual_ffn.linear.weight")
        ):
            torch.nn.init.normal_(parameter, std=0.03)
    pixels = torch.randn(1, 3, 5, 16, 32)
    first, encoder_cache = model.encode_stream(pixels[:, :, :1])[1:]
    encoder_keys = set(encoder_cache)
    second, _ = model.encode_stream(pixels[:, :, 1:], encoder_cache)[1:]
    assert set(encoder_cache) == encoder_keys
    joined = torch.cat((first[0], second[0]), dim=2)
    torch.testing.assert_close(
        joined, model.encode_stream(pixels)[1][0], rtol=2e-5, atol=1e-5
    )
    output1, decoder_cache = model.decode_stream(joined[:, :, :1])
    decoder_keys = set(decoder_cache)
    output2, _ = model.decode_stream(joined[:, :, 1:], decoder_cache)
    assert set(decoder_cache) == decoder_keys
    # Chunk sizes select different CPU kernels; allow accumulated FP32 roundoff.
    torch.testing.assert_close(
        torch.cat((output1, output2), dim=2),
        model.decode_stream(joined)[0],
        rtol=2e-5,
        atol=2e-5,
    )

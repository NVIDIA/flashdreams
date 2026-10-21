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

"""CPU checks for joint sensor generation, causal timing and point presentation."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from cosmos3_nano_sim_auto.impl.joint import JointChunkRollout, sweep_range
from cosmos3_nano_sim_auto.impl.lidar import LidarStream
from cosmos3_nano_sim_auto.impl.packing import ClipGeometry
from cosmos3_nano_sim_auto.impl.transformer.mask import (
    ROLE_CONTROL,
    ROLE_CURRENT_TARGET,
    ROLE_TARGET_CONDITION,
    visibility,
)
from cosmos_interactive._network import CosmosMoTNetworkConfig
from cosmos_interactive._spec import Cosmos3InteractiveSpec, LLMSpec, VFMSpec
from cosmos_interactive.checkpoint_remap import remap_key
from cosmos_interactive.lidar import LidarGeometry, LidarTokenizer

from flashdreams.core.visualization.lidar import LidarViewport, range_image_points

TEST_SAMPLE = Path(__file__).with_name("sample.jsonl")


pytestmark = pytest.mark.ci_cpu


@pytest.fixture
def sample_dir(tmp_path: Path) -> str:
    (tmp_path / "sample.json").write_text(TEST_SAMPLE.read_text())
    return str(tmp_path)


def make_rollout(
    *,
    conditioned=False,
    lidar_conditioned=False,
    frames=10,
    device="cpu",
    dtype=torch.float32,
    latent_size=None,
    rig_embeddings=False,
    **options,
):
    torch.manual_seed(5)
    spec = Cosmos3InteractiveSpec(
        llm=LLMSpec(
            hidden_size=32,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            intermediate_size=16,
            vocab_size=100,
            mrope_section=(2, 1, 1),
        ),
        vfm=VFMSpec(vae_latent_channels=4, latent_patch_size=2, time_embed_freq_dim=8),
    )
    lg = LidarGeometry(channels=3, height=2, width=3)
    tokenizer = LidarTokenizer(None, lg)
    network = (
        CosmosMoTNetworkConfig(
            spec=spec,
            dtype=dtype,
            include_vision_tower=False,
            lidar_patch_dim=lg.patch_dim,
            num_view_embeddings=12 if rig_embeddings else 0,
        )
        .setup()
        .to(device)
        .eval()
    )
    geometry = ClipGeometry(
        2,
        frames,
        1,
        1,
        3,
        int(conditioned),
        4 / 30,
        latent_size=latent_size,
        rig_view_ids=(6, 4) if rig_embeddings else None,
    )
    controls = torch.randn(2 * frames, 16)
    lidar = torch.randn(1, 3, 20, 2, 3)
    provider = SimpleNamespace(tokens=lambda a, b: tokenizer.patchify(lidar[:, :, a:b]))
    condition = torch.randn(2, 16) if conditioned else None
    lidar_condition = tokenizer.patchify(lidar[:, :, :1]) if lidar_conditioned else None
    rollout = JointChunkRollout(
        network,
        geometry=geometry,
        control_tokens=controls,
        lidar_controls=options.pop("lidar_controls", provider),
        lidar_tokenizer=tokenizer,
        text_ids=torch.tensor([1, 2, 3, 4]),
        view_text_tokens=(2, 2),
        condition_tokens=condition,
        lidar_condition=lidar_condition,
        t_list=[1.0, 0.5],
        history_frames=3,
        **options,
    )
    return rollout


def test_joint_compiled_denoise_runs_steady_chunks_compiled(monkeypatch):
    """Dynamo traces the steady chunks' passes (eager backend on CPU), changing nothing.

    Warmup shapes stay eager, so the graph count does not grow with the history.
    """
    import torch._dynamo

    steady = []

    def steps(rollout):
        out = []
        while not rollout.is_finished:
            _, rgb = rollout.step()
            steady.append(rollout._steady)
            out.append((rgb.clone(), rollout.last_lidar.clone()))
        return out

    reference = make_rollout(conditioned=True, rig_embeddings=True)
    expected = steps(reference)
    modes, calls = [], []
    real = torch.compile

    def fake_compile(fn, *, mode=None):
        modes.append(mode)
        compiled = real(fn, backend="eager")
        return lambda *a: calls.append(fn.__name__) or compiled(*a)

    torch._dynamo.reset()
    monkeypatch.setattr(torch, "compile", fake_compile)
    rollout = make_rollout(
        conditioned=True, rig_embeddings=True, compile_denoise="max-autotune"
    )
    steady.clear()
    got = steps(rollout)
    assert modes == ["max-autotune"] * 2
    assert not steady[0] and any(steady), steady
    # Per steady chunk: control K/V, two denoise steps, clean K/V.
    assert calls == ["_collect", "_forward", "_forward", "_collect"] * sum(steady)
    for want, have in zip(expected, got, strict=True):
        for a, b in zip(want, have, strict=True):
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    # The retained cache, written from compiled K/V, matches eager too.
    for want, have in zip(reference._memory()[1], rollout._memory()[1], strict=True):
        for a, b in zip(want, have, strict=True):
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError, match="compile_denoise"):
        make_rollout(compile_denoise="fastest")
    from dataclasses import replace

    from flashdreams.core.distributed.parallel import ParallelContext

    with pytest.raises(ValueError, match="tensor parallelism"):
        make_rollout(
            compile_denoise="default",
            parallel=replace(ParallelContext.single("cpu"), tp_size=2),
        )


def test_lidar_patch_order_padding_and_roundtrip():
    tokenizer = LidarTokenizer(None, LidarGeometry(channels=2, height=2, width=3))
    latent = torch.arange(12).reshape(1, 2, 1, 2, 3).float()
    patches = tokenizer.patchify(latent)
    assert patches.tolist() == [[0, 6, 1, 7, 3, 9, 4, 10], [2, 8, 0, 0, 5, 11, 0, 0]]
    assert torch.equal(tokenizer.unpatchify(patches), latent)
    assert remap_key("model.net.lidar2llm.weight") == "lidar2llm.weight"
    assert remap_key("model.net.llm2lidar.weight") == "llm2lidar.weight"


@pytest.mark.parametrize(
    "conditioned,lidar_conditioned", [(False, False), (True, False), (True, True)]
)
def test_joint_rollout_generates_both_sensors_and_bounds_cache(
    conditioned, lidar_conditioned
):
    rollout = make_rollout(conditioned=conditioned, lidar_conditioned=lidar_conditioned)
    ranges = []
    while not rollout.is_finished:
        trace, tokens = rollout.step()
        ranges.append(rollout.last_lidar_range)
        assert torch.isfinite(tokens).all() and torch.isfinite(rollout.last_lidar).all()
        assert rollout.latent_for(trace.start, trace.end).shape[:3] == (
            2,
            4,
            trace.end - trace.start,
        )
        assert len(rollout.history) <= 1
        if trace.start == 0 and conditioned:
            assert torch.equal(tokens.reshape(2, 16), rollout.condition)
        if trace.start == 0 and lidar_conditioned:
            assert torch.equal(
                rollout.last_lidar,
                rollout.tokenizer.unpatchify(rollout.lidar_condition),
            )
    assert ranges == [(0, 1), (1, 5), (5, 9), (9, 13)]
    fields, _ = rollout._memory()
    assert set(fields.causal_step_id.tolist()) == {0, 3}
    assert not fields.is_noisy.any()
    assert rollout.last_metrics["lidar_tokens"] == 8
    with pytest.raises(RuntimeError, match="complete"):
        rollout.step()


def test_joint_step_plans_each_pass_once_and_prefetches_both_sensors_first(
    monkeypatch,
):
    from cosmos3_nano_sim_auto.impl import joint

    rollout = make_rollout(conditioned=True)
    order = []
    tokens = rollout.lidar_controls.tokens
    rollout.lidar_controls.prefetch = lambda a, b: order.append(("prefetch", a, b))
    rollout.lidar_controls.tokens = lambda a, b: order.append("tokens") or tokens(a, b)
    rgb_tokens = rollout.controls.tokens
    monkeypatch.setattr(
        rollout.controls,
        "prefetch",
        lambda a, b: order.append(("rgb_prefetch", a, b)),
        raising=False,
    )
    monkeypatch.setattr(
        rollout.controls, "tokens", lambda a, b: order.append("rgb") or rgb_tokens(a, b)
    )
    plans = []
    build = joint.build_fold_plan
    monkeypatch.setattr(
        joint, "build_fold_plan", lambda *a, **k: plans.append(1) or build(*a, **k)
    )
    rollout.step()
    rollout.step()
    # Control, one noisy plan shared by both denoise steps, and the clean pass.
    assert len(plans) == 6
    # Camera chunks are latent frames [0, 1) then [1, 4); sweeps [0, 1), [1, 5).
    # Once a chunk's camera controls are taken, the next chunk's are asked for,
    # so a worker draws them during this chunk's denoise (a no-op if in flight).
    assert order == [
        ("rgb_prefetch", 0, 1),
        ("prefetch", 0, 1),
        "rgb",
        ("rgb_prefetch", 1, 4),
        "tokens",
        ("rgb_prefetch", 1, 4),
        ("prefetch", 1, 5),
        "rgb",
        ("rgb_prefetch", 4, 7),
        "tokens",
    ]


def test_joint_caption_scope_clock_and_initial_rgb_visibility():
    rollout = make_rollout(conditioned=True)
    lidar = rollout._fields(0, 1, lidar=True, role=ROLE_CURRENT_TARGET)
    rgb = rollout._fields(0, 1, lidar=False, role=ROLE_TARGET_CONDITION)
    assert visibility(lidar, rollout.text_fields, joint=True).all()
    assert visibility(lidar, rgb, joint=True).all()
    assert not visibility(rgb, lidar, joint=True).any()
    assert torch.allclose(
        rollout._positions(1, 2, lidar=True)[0], torch.full((2,), rollout.offset + 0.75)
    )
    assert sweep_range(1, 4, geometry=rollout.geometry, lidar_fps=10) == (1, 5)
    assert not visibility(rgb, rollout.text_fields)[0, 2:].any()


def test_unprojection_and_viewport_ego_and_invalid_points():
    ranges = torch.tensor([[[2.0, 3.0]], [[0.1, 0.2]], [[1.0, 0.0]]])
    rays = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    pose = torch.eye(4)
    pose[2, 3] = 1.5
    assert torch.equal(
        range_image_points(ranges, rays, sensor_to_ego=pose),
        torch.tensor([[2.0, 0.0, 1.5]]),
    )
    viewport = LidarViewport(width=128, height=96, extent_m=12)
    image = viewport.render(
        torch.tensor([[float("nan"), 0, 0], [1000, 0, 0], [3, 0, 2]])
    )
    assert image.shape == (3, 96, 128) and torch.isfinite(image).all()
    assert image.min() >= 0 and image.max() <= 1
    assert ((image[0] > 0.9) & (image[2] > 0.7)).any()  # magenta ego / white nose
    assert not torch.equal(image, viewport.render(torch.empty(0, 3)))


def test_viewport_uses_past_sweep_at_camera_rate():
    stream = object.__new__(LidarStream)
    stream.assets = SimpleNamespace(
        sample=SimpleNamespace(
            timestamps=torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4], dtype=torch.float64),
            ray_directions=torch.tensor([[[1.0, 0.0, 0.0]]]),
            sensor_to_ego=torch.eye(4),
        )
    )
    stream.previous_range, stream.previous_time = None, None
    viewport = SimpleNamespace(
        render_sweep=lambda ranges, rays, **kwargs: ranges[0, 0, 0]
        .expand(3, 2, 2)
        .clone()
    )
    ranges = torch.ones(1, 3, 1, 1, 1)
    first = stream.frames(
        ranges, first_sweep=0, pixel_start=0, pixel_count=1, fps=30, viewport=viewport
    )
    assert (first == 1).all()
    ranges = torch.ones(1, 3, 4, 1, 1)
    ranges[:, 0, :, 0, 0] = torch.arange(2, 6)
    next_frames = stream.frames(
        ranges, first_sweep=1, pixel_start=1, pixel_count=12, fps=30, viewport=viewport
    )
    assert next_frames[:, 0, 0, 0].tolist() == [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5]


@pytest.mark.parametrize("rig_embeddings", [False, True])
def test_checkpoint_detects_lidar_width_and_independent_biases(
    monkeypatch, rig_embeddings
):
    from cosmos_interactive import _network, checkpoint_load

    network = make_rollout(rig_embeddings=rig_embeddings).network
    raw = {}
    for key, tensor in network.state_dict().items():
        if key.startswith("layers."):
            parts = key.split(".")
            leaf = ".".join(parts[3:])
            if parts[2] == "gen":
                if leaf.startswith("self_attn."):
                    parent, attr = leaf.rsplit(".", 1)
                    leaf = parent + "_moe_gen." + attr
                else:
                    parent, rest = leaf.split(".", 1)
                    leaf = parent + "_moe_gen." + rest
            source = "language_model.model.layers." + parts[1] + "." + leaf
        elif key.startswith(("embed_tokens.", "norm.", "norm_moe_gen.")):
            source = "language_model.model." + key
        elif key.startswith("lm_head."):
            source = "language_model." + key
        else:
            source = key
        raw["model.net." + source] = tensor.clone()
    monkeypatch.setattr(_network, "SPEC", network.config.spec)
    monkeypatch.setattr(checkpoint_load, "_load_raw_state_dict", lambda *a: raw)
    loaded = checkpoint_load.load_exported_checkpoint(
        "unused", device="cpu", dtype=torch.float32
    )
    assert loaded.lidar2llm.bias is None
    assert loaded.llm2lidar.bias is not None
    for name, value in network.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], value)
    del raw["model.net.llm2lidar.weight"]
    with pytest.raises(ValueError, match="both LiDAR"):
        checkpoint_load.load_exported_checkpoint("unused", device="cpu")


def test_lidar_options_require_complete_valid_configuration():
    from cosmos3_nano_sim_auto.impl.app import _parse

    with pytest.raises(ValueError, match="matching real RGB"):
        _parse(["--lidar-sample", "sample.npz"])
    with pytest.raises(ValueError, match="requires --lidar-sample"):
        _parse(["--lidar-checkpoint", "weights.pt"])
    assert (
        _parse(["--sample", "sample", "--lidar-sample", "sample.npz"]).lidar_checkpoint
        is None
    )
    args = [
        "--sample",
        "sample",
        "--lidar-sample",
        "sample.npz",
        "--lidar-checkpoint",
        "weights.pt",
    ]
    assert _parse(args).lidar_checkpoint == "weights.pt"
    assert _parse(args).lidar_view == "point-cloud"
    assert _parse(args).lidar_tokenizer_version == "v1p2"
    assert (
        _parse(args + ["--lidar-tokenizer-version", "v1p1"]).lidar_tokenizer_version
        == "v1p1"
    )
    assert _parse(args).ui
    assert _parse(args + ["--lidar-view", "range"]).lidar_view == "range"
    assert not _parse(args + ["--no-ui"]).ui
    with pytest.raises(SystemExit):
        _parse(args + ["--lidar-view", "unknown"])
    with pytest.raises(ValueError, match="Recorded LiDAR"):
        _parse(args + ["--loop-controls"])


def _joint_worker(rank, tp, init_file, output):
    from datetime import timedelta

    import torch.distributed as dist
    from cosmos3_nano_sim_auto.impl.distributed.tensor_parallel import shard_network

    from flashdreams.core.distributed.parallel import build_context

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method="file://" + init_file,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        rollout = make_rollout(conditioned=True, frames=4)
        ctx = build_context(tp, 2 // tp, torch.device("cpu"))
        # Rebuild the text cache on the sharded network just as the app does.
        network = shard_network(rollout.network, ctx)
        rollout = JointChunkRollout(
            network,
            geometry=rollout.geometry,
            control_tokens=rollout.controls,
            lidar_controls=rollout.lidar_controls,
            lidar_tokenizer=rollout.tokenizer,
            text_ids=torch.tensor([1, 2, 3, 4]),
            view_text_tokens=(2, 2),
            condition_tokens=rollout.condition,
            t_list=[1.0, 0.5],
            history_frames=3,
            parallel=ctx,
        )
        values = []
        while not rollout.is_finished:
            _, rgb = rollout.step()
            assert rollout.last_lidar is not None
            values.append((rgb.clone(), rollout.last_lidar.clone()))
        torch.save(values, output + str(rank) + ".pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("tp", [1, 2])
def test_joint_two_rank_equivalence(tmp_path, tp):
    import torch.multiprocessing as mp

    reference = make_rollout(conditioned=True, frames=4)
    expected = []
    while not reference.is_finished:
        _, rgb = reference.step()
        expected.append((rgb.clone(), reference.last_lidar.clone()))
    mp.spawn(
        _joint_worker,
        args=(tp, str(tmp_path / "init"), str(tmp_path / "result")),
        nprocs=2,
        join=True,
    )
    for rank in range(2):
        actual = torch.load(
            str(tmp_path / "result") + str(rank) + ".pt", weights_only=True
        )
        for wanted, got in zip(expected, actual, strict=True):
            for a, b in zip(wanted, got, strict=True):
                torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)


def test_stream_codecs_share_weights_but_keep_independent_caches(monkeypatch):
    from cosmos_interactive import lidar
    from cosmos_interactive.lidar import LidarDecoder, LidarEncoder

    monkeypatch.setattr(
        lidar, "metric_lidar_to_network", lambda x, **kwargs: (x, x[:, 2:3])
    )
    monkeypatch.setattr(
        lidar,
        "network_lidar_to_metric_clip",
        lambda x, v, **kwargs: torch.cat((x[:, :2], v), 1),
    )

    class Backend:
        device, dtype = torch.device("cpu"), torch.float32
        min_range, max_range, validity_threshold = 5.0, 100.0, 0.5
        latent_mean, latent_std = 2.0, 4.0

        def __init__(self):
            self.model = self

        def _limit_streaming_cache(self, cache, context_frames, *, new_frames):
            return min(cache or 0, context_frames - new_frames)

        tf32_seen: list[bool] = []

        def encode_stream(self, x, cache):
            self.tf32_seen.append(torch.backends.cuda.matmul.allow_tf32)
            return None, (x, None), cache + x.shape[2]

        def decode_stream(self, z, cache):
            self.tf32_seen.append(torch.backends.cuda.matmul.allow_tf32)
            return z, cache + z.shape[2]

    tokenizer = LidarTokenizer(Backend(), LidarGeometry(channels=3, height=2, width=3))
    first, second = LidarEncoder(tokenizer), LidarEncoder(tokenizer)
    decoder = LidarDecoder(tokenizer)
    raw = torch.ones(1, 3, 4, 2, 3)
    encoded = first(raw)
    assert torch.equal(encoded, (raw - 2) / 4)
    reconstructed = decoder(encoded)
    assert torch.equal(reconstructed[:, :2], raw[:, :2])
    assert torch.equal(reconstructed[:, 2], raw[:, 2].sigmoid())
    assert second.cache is None
    for _ in range(4):
        first(raw)
        decoder(encoded)
    assert first.cache == decoder.cache == 9
    # TF32 is scoped to the codec call and restored afterwards.
    before = torch.backends.cuda.matmul.allow_tf32
    Backend.tf32_seen.clear()
    LidarEncoder(tokenizer, tf32=True)(raw)
    LidarDecoder(tokenizer, tf32=True)(encoded)
    first(raw)
    assert Backend.tf32_seen == [True, True, before]
    assert torch.backends.cuda.matmul.allow_tf32 == before
    first.reset()
    decoder.reset()
    assert first.cache is None and decoder.cache is None
    with pytest.raises(ValueError, match="Expected"):
        first(torch.zeros(1, 3, 10, 2, 3))


def test_sample_rejects_clock_gaps_and_short_controls(tmp_path):
    import numpy as np
    from cosmos3_nano_sim_auto.impl.lidar import LidarSample

    path = tmp_path / "lidar.npz"
    values: dict[str, Any] = dict(
        controls=np.zeros((3, 3, 128, 1800), dtype=np.float32),
        timestamps=np.array([0.0, 0.1, 0.2]),
        ray_directions=np.broadcast_to(
            np.array([1.0, 0.0, 0.0], dtype=np.float32), (128, 1800, 3)
        ),
        sensor_to_ego=np.eye(4, dtype=np.float32),
    )
    np.savez(path, **values)
    sample = LidarSample.load(path, start_seconds=0, duration_seconds=0.2)
    assert len(sample.controls) == 3
    with pytest.raises(ValueError, match="end before"):
        LidarSample.load(path, start_seconds=0, duration_seconds=0.4)
    with pytest.raises(ValueError, match="aligned"):
        LidarSample.load(path, start_seconds=0.05, duration_seconds=0.1)
    values["timestamps"] = np.array([0.0, 0.1, 0.3])
    np.savez(path, **values)
    with pytest.raises(ValueError, match="missing sweeps"):
        LidarSample.load(path, start_seconds=0, duration_seconds=0.2)


@pytest.mark.parametrize("use_ui", [False, True])
def test_app_presents_lidar_beside_rgb_and_resets_stream(use_ui, sample_dir):
    import queue
    import threading
    from dataclasses import replace

    from cosmos3_nano_sim_auto.impl.app import (
        Cosmos3NanoSimAutoApplication,
        Cosmos3NanoSimAutoModelLoop,
        Cosmos3NanoSimAutoModelState,
        Cosmos3NanoSimAutoSession,
    )

    from flashdreams.runtime_v2.session_desc import SessionDesc

    app = Cosmos3NanoSimAutoApplication()
    app.init(
        ["--device", "cpu", "--views", "2", "--frames", "13", "--sample", sample_dir]
    )
    assert app._config is not None
    config = replace(app._config, lidar=SimpleNamespace(), ui=use_ui)
    sample = SimpleNamespace(view_size=(8, 8), cameras=("a", "b"))
    session = Cosmos3NanoSimAutoSession(config, SessionDesc(), cast(Any, sample))
    assert session.session_desc.video_width == (1280 if use_ui else 24)
    rollout = make_rollout(conditioned=True, frames=4)
    state = Cosmos3NanoSimAutoModelState(
        config=config,
        session_desc=session.session_desc,
        sample=cast(Any, sample),
        network=rollout.network,
        rollout=rollout,
        decode=lambda latent: torch.zeros(
            2, 1 if latent.shape[2] == 1 else 12, 3, 8, 8
        ),
        lidar_stream=cast(
            Any,
            SimpleNamespace(
                ranges=lambda x: x,
                close=lambda: None,
                frames=lambda x, **kw: torch.zeros(kw["pixel_count"], 3, 8, 8),
            ),
        ),
    )
    loop = Cosmos3NanoSimAutoModelLoop()
    loop.register_session_loop_objects(
        state=state,
        frequency=0,
        shutdown_event=threading.Event(),
        failure_queue=queue.Queue(),
    )
    from flashdreams.runtime_v2.user_input_events import UserInputEvents

    for step, expected in enumerate((1, 12)):
        results = loop.step(step, UserInputEvents([]))
        frames = tuple(result.read_output() for result in results)
        metrics = results[0].metrics
        assert all(result.frame_count == expected for result in results)
        assert all(result.metrics is None for result in results[1:])
        assert len(frames) == (3 if use_ui else 1)
        assert frames[0].shape == (expected, 3, 8, 8 if use_ui else 24)
        assert ((frames[-1] if use_ui else frames[0][:, :, :, -8:]) == -1).all()
        assert metrics is not None
        assert metrics["lidar_enabled"] == 1
        assert metrics["lidar_decode_s"] >= 0 and metrics["lidar_render_s"] >= 0
    loop.reset()
    assert state.lidar_stream is None and state.decoded_to == 0


@pytest.mark.parametrize("explicit", [False, True])
def test_app_prepares_lidar_assets_during_init(monkeypatch, explicit):
    from cosmos3_nano_sim_auto import checkpoint
    from cosmos3_nano_sim_auto.impl import app as application
    from cosmos3_nano_sim_auto.impl.lidar import LidarSample
    from cosmos3_nano_sim_auto.inputs import MultiviewSample

    sample = MultiviewSample.load(TEST_SAMPLE).trimmed_to(2)
    monkeypatch.delenv("COSMOS_INTERACTIVE_CHECKPOINT", raising=False)
    monkeypatch.setattr(application, "_sample_for", lambda args: sample)
    tokenizer, prepared_sample = object(), object()
    monkeypatch.setattr(LidarSample, "load", lambda *a, **kw: prepared_sample)
    cached = Path("/hf/snapshots/model/lidar_vae/diffusion_pytorch_model.safetensors")
    downloads = []

    def download(**kwargs):
        downloads.append(kwargs)
        return str(cached)

    monkeypatch.setattr(checkpoint, "hf_hub_download", download)

    def load_tokenizer(*args, **kwargs):
        assert args[0] == ("tokenizer.pt" if explicit else str(cached))
        assert kwargs["version"] == "v1p2"
        return tokenizer

    monkeypatch.setattr(LidarTokenizer, "from_checkpoint", load_tokenizer)
    app = application.Cosmos3NanoSimAutoApplication()
    app.init(
        [
            "--device",
            "cpu",
            "--views",
            "2",
            "--frames",
            "13",
            "--sample",
            "sample",
            "--lidar-sample",
            "lidar.npz",
            *(["--lidar-checkpoint", "tokenizer.pt"] if explicit else []),
        ]
    )
    assert app._config is not None
    assert app._config.lidar is not None
    assert app._config.lidar.tokenizer is tokenizer
    assert app._config.lidar.sample is prepared_sample
    assert app._config.lidar.checkpoint == ("tokenizer.pt" if explicit else str(cached))
    assert app._config.checkpoint == application.DEFAULT_CHECKPOINT
    assert len(downloads) == (0 if explicit else 2)


@pytest.mark.parametrize("view", ["range", "intensity"])
def test_range_views_mask_invalid_returns_and_keep_fixed_scales(view):
    from flashdreams.core.visualization.lidar import LidarView

    viewport = LidarViewport(width=4, height=4)
    ranges = torch.tensor(
        [
            [[25.0, 75.0, 50.0, float("nan")], [25.0, 75.0, 50.0, 100.0]],
            [[0.25, 0.75, 0.5, 1.0], [0.25, 0.75, 0.5, 1.0]],
            [[1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
        ]
    )
    image = viewport.render_sweep(ranges, torch.empty(0), view=LidarView(view))
    assert image.shape == (3, 4, 4) and torch.isfinite(image).all()
    assert (image[:, [0, 3]] == 0).all()  # panorama stays 2:1, with padding
    assert (image[:, 1, 2:] == 0).all()  # invalid flag and NaN
    expected = [0.25, 0.5, 0.75] if view == "range" else [0.25] * 3
    torch.testing.assert_close(image[:, 1, 0], torch.tensor(expected))
    changed = ranges.clone()
    changed[0, 1, -1] = 10  # no per-frame autoscale
    torch.testing.assert_close(
        viewport.render_sweep(changed, torch.empty(0), view=LidarView(view))[:, 1, 0],
        image[:, 1, 0],
    )


def test_switching_view_rerenders_the_held_previous_sweep():
    from flashdreams.core.visualization.lidar import LidarView

    stream = object.__new__(LidarStream)
    stream.assets = SimpleNamespace(
        sample=SimpleNamespace(
            timestamps=torch.tensor([0.0, 0.1], dtype=torch.float64),
            ray_directions=torch.tensor([[[1.0, 0.0, 0.0]]]),
            sensor_to_ego=torch.eye(4),
        )
    )
    stream.previous_range, stream.previous_time = None, None
    viewport = LidarViewport(width=2, height=2)
    ranges = torch.tensor([25.0, 0.75, 1.0]).reshape(1, 3, 1, 1, 1)
    first = stream.frames(
        ranges,
        first_sweep=0,
        pixel_start=0,
        pixel_count=1,
        fps=30,
        viewport=viewport,
        view=LidarView.RANGE,
    )
    switched = stream.frames(
        ranges,
        first_sweep=1,
        pixel_start=1,
        pixel_count=3,
        fps=30,
        viewport=viewport,
        view=LidarView.INTENSITY,
    )
    assert (switched == 0.75).all()
    assert not torch.equal(first[0], switched[0])


def test_lidar_ui_queues_view_change_without_resetting_rollout():
    import queue
    import threading
    from unittest.mock import Mock

    from cosmos3_nano_sim_auto.impl.app import Cosmos3NanoSimAutoModelLoop
    from cosmos3_nano_sim_auto.impl.ui import (
        CosmosMultiviewUILoop,
        CosmosMultiviewUIState,
    )
    from interactive_drive.multiview import AuxiliaryView

    from flashdreams.core.visualization.lidar import LidarView

    rollout = object()
    state = SimpleNamespace(lidar_view=LidarView.POINT_CLOUD, rollout=rollout)
    model = Cosmos3NanoSimAutoModelLoop()
    model.register_session_loop_objects(
        state=cast(Any, state),
        frequency=0,
        shutdown_event=threading.Event(),
        failure_queue=queue.Queue(),
    )
    ui = CosmosMultiviewUILoop(renderer=Mock())
    ui.register_session_loop_objects(
        state=CosmosMultiviewUIState(
            model_loop=model,
            title="Auto",
            prompt="",
            scene_options=(),
            auxiliary_views=(AuxiliaryView("lidar", "LiDAR"),),
        ),
        frequency=30,
        shutdown_event=threading.Event(),
        failure_queue=queue.Queue(),
    )
    imgui = Mock()
    imgui.combo.return_value = (True, 2)
    ui._draw_view_controls(imgui)
    assert ui.state.lidar_view is LidarView.INTENSITY
    assert state.lidar_view is LidarView.POINT_CLOUD
    model._run_message_batch()
    assert state.lidar_view is LidarView.INTENSITY and state.rollout is rollout
    ui.state.auxiliary_views = ()
    imgui.reset_mock()
    ui._draw_view_controls(imgui)
    imgui.combo.assert_not_called()


@pytest.mark.parametrize(
    "use_lidar,use_ui", [(False, True), (True, False), (True, True)]
)
def test_lidar_session_registers_selector_only_when_enabled(
    monkeypatch, use_lidar, use_ui, sample_dir
):
    from dataclasses import replace

    from cosmos3_nano_sim_auto.impl import app as application
    from cosmos3_nano_sim_auto.impl.lidar import LidarAssets
    from cosmos3_nano_sim_auto.impl.ui import CosmosMultiviewUILoop
    from cosmos3_nano_sim_auto.inputs import MultiviewSample

    from flashdreams.core.visualization.lidar import LidarView
    from flashdreams.runtime_v2.session_desc import SessionDesc

    app = application.Cosmos3NanoSimAutoApplication()
    app.init(
        [
            "--device",
            "cpu",
            "--views",
            "2",
            "--frames",
            "13",
            "--lidar-view",
            "range",
            "--sample",
            sample_dir,
        ]
    )
    rollout = make_rollout()
    assert app._config is not None
    config = replace(
        app._config,
        lidar=LidarAssets(rollout.tokenizer, cast(Any, object()))
        if use_lidar
        else None,
        ui=use_ui,
        device="cuda",
    )
    monkeypatch.setattr(
        application, "resolve_checkpoint_source", lambda _, **kwargs: "unused"
    )
    monkeypatch.setattr(
        application, "load_exported_checkpoint", lambda *a, **kw: rollout.network
    )
    session = application.Cosmos3NanoSimAutoSession(
        config,
        SessionDesc(),
        MultiviewSample.load(TEST_SAMPLE).trimmed_to(2),
    )
    session.init()
    assert session._registered_model_loop is not None
    assert session._registered_model_loop.state.lidar_view is LidarView.RANGE
    assert isinstance(session._registered_ui_loop, CosmosMultiviewUILoop) == use_ui
    if use_ui:
        assert session._registered_ui_loop is not None
        assert len(session._registered_ui_loop.state.auxiliary_views) == int(use_lidar)
        assert session.session_desc.video_width == SessionDesc().video_width


def test_joint_odd_rgb_extent_uses_only_unpadded_noise(monkeypatch):
    from cosmos3_nano_sim_auto.impl import joint as joint_module

    rollout = make_rollout(frames=4, latent_size=(1, 1))
    original_pass = rollout._pass
    original_denoise = joint_module.fixed_step_denoise
    sizes = []

    def denoise(flow, noise, **kwargs):
        sizes.append(noise.numel())
        return original_denoise(flow, noise, **kwargs)

    def forward(rgb, *args, **kwargs):
        if "timestep" in kwargs:
            patches = rgb.reshape(-1, 2, 2, 4)
            assert torch.count_nonzero(patches[:, 1]) == 0
            assert torch.count_nonzero(patches[:, :, 1]) == 0
        return original_pass(rgb, *args, **kwargs)

    monkeypatch.setattr(joint_module, "fixed_step_denoise", denoise)
    rollout._pass = forward
    while not rollout.is_finished:
        trace, _ = rollout.step()
        assert rollout.latent_for(trace.start, trace.end).shape[-2:] == (1, 1)
    assert sizes == [2 * 4 + 3 * 2 * 3, 2 * 4 * 3 + 3 * 4 * 2 * 3]


@pytest.mark.parametrize("role", [ROLE_CONTROL, ROLE_CURRENT_TARGET])
@torch.no_grad()
def test_joint_final_checkpoint_uses_physical_rig_rows_and_disjoint_attention(
    monkeypatch, role
):
    from cosmos3_nano_sim_auto.impl import joint

    rollout = make_rollout(rig_embeddings=True)
    network = rollout.network
    for module in (network.vae2llm, network.lidar2llm, network.time_embedder):
        for parameter in module.parameters():
            parameter.zero_()
    rgb_fields = rollout._fields(1, 2, lidar=False, role=role)
    lidar_fields = rollout._fields(1, 2, lidar=True, role=role)
    expected_ids = torch.tensor([6, 4, 11, 11])

    def check_layers(layers, spec, hidden, positions, **kwargs):
        torch.testing.assert_close(hidden[0], network.rig_view_embed(expected_ids))
        coverage = torch.zeros((4, 8), dtype=torch.int)
        for _, groups in kwargs["mask"].passes:
            for group in groups:
                for q in group.queries:
                    for k in group.keys:
                        coverage[q.start : q.stop, k.start : k.stop] += 1
        assert coverage.max() == 1
        assert (coverage.sum(1) > 0).all()
        return hidden, []

    monkeypatch.setattr(joint, "run_chunk_layers", check_layers)
    empty = (rollout.empty_fields, rollout.empty_kv)
    rollout._pass(
        torch.zeros(2, network.vae2llm.in_features),
        torch.zeros(2, network.lidar2llm.in_features),
        rollout._plan(rgb_fields, lidar_fields, empty),
        rollout._positions(1, 2, lidar=False),
        rollout._positions(1, 2, lidar=True),
        empty,
        collect=True,
    )


@pytest.mark.parametrize("patch_dim,patch,tokens", [(128, 1, 904), (512, 2, 228)])
def test_session_selects_lidar_patches_before_streaming(
    monkeypatch, patch_dim, patch, tokens, sample_dir
):
    from dataclasses import replace

    from cosmos3_nano_sim_auto.impl import app as application
    from cosmos3_nano_sim_auto.impl.lidar import LidarAssets

    from flashdreams.runtime_v2.session_desc import SessionDesc

    app = application.Cosmos3NanoSimAutoApplication()
    app.init(
        [
            "--device",
            "cpu",
            "--views",
            "2",
            "--frames",
            "13",
            "--no-ui",
            "--sample",
            sample_dir,
        ]
    )
    network = make_rollout(rig_embeddings=patch_dim == 128).network
    network.lidar2llm = torch.nn.Linear(patch_dim, 32)
    network.llm2lidar = torch.nn.Linear(32, patch_dim)
    monkeypatch.setattr(
        application, "load_exported_checkpoint", lambda *a, **kw: network
    )
    monkeypatch.setattr(
        application, "resolve_checkpoint_source", lambda source, **kwargs: "unused"
    )
    prepared = LidarTokenizer(object(), LidarGeometry(height=8, width=113))
    # A recorded sample with nothing to pre-encode: the stream is built, no controls are read.
    assert app._config is not None
    config = replace(
        app._config,
        lidar=LidarAssets(prepared, cast(Any, SimpleNamespace(controls=()))),
    )
    assert app._sample is not None
    session = application.Cosmos3NanoSimAutoSession(config, SessionDesc(), app._sample)
    states = []
    monkeypatch.setattr(
        session, "register_model_loop", lambda cls, *, state: states.append(state)
    )
    session.init()
    tokenizer = states[0].config.lidar.tokenizer
    assert tokenizer.geometry.patch == patch
    assert tokenizer.geometry.spatial_tokens == tokens
    assert tokenizer.backend is prepared.backend
    assert prepared.geometry.patch == 2
    # Reset creates fresh stream state from the same checkpoint-bound geometry.
    for _ in range(2):
        stream = states[0].config.lidar.stream()
        assert stream.encoder.tokenizer is tokenizer
        assert stream.decoder.tokenizer is tokenizer
    latent = torch.arange(128 * 8 * 113).reshape(1, 128, 1, 8, 113).float()
    packed = tokenizer.patchify(latent)
    assert packed.shape == (tokens, patch_dim)
    assert network.llm2lidar(network.lidar2llm(packed)).shape == packed.shape
    assert torch.equal(tokenizer.unpatchify(packed), latent)
    rollout = JointChunkRollout(
        network,
        geometry=ClipGeometry(2, 4, 1, 1, 3, 0, 4 / 30, rig_view_ids=(6, 4)),
        control_tokens=torch.zeros(8, 16),
        lidar_controls=stream,
        lidar_tokenizer=tokenizer,
        text_ids=torch.tensor([1, 2]),
    )
    assert rollout.tokenizer.geometry.patch_dim == patch_dim


def test_lidar_accepts_compiled_denoising():
    from cosmos3_nano_sim_auto.impl.app import _parse

    args = _parse(
        [
            "--sample",
            "s",
            "--lidar-sample",
            "l.npz",
            "--lidar-checkpoint",
            "w.pt",
            "--compile-denoise",
            "default",
        ]
    )
    assert args.compile_denoise == "default"


def test_joint_rollout_preencodes_the_recorded_lidar_controls():
    rollout = make_rollout()
    tokenizer, lidar = rollout.tokenizer, torch.randn(1, 3, 20, 2, 3)
    asked, served = [], []

    class Recorded:
        def preencode(self, ranges):
            for lo, hi in ranges:
                if hi > 13:
                    break
                asked.append((lo, hi))

        def tokens(self, lo, hi):
            served.append((lo, hi))
            return tokenizer.patchify(lidar[:, :, lo:hi])

    rollout = make_rollout(lidar_controls=Recorded())
    assert asked == [(0, 1), (1, 5), (5, 9), (9, 13)]
    while not rollout.is_finished:
        rollout.step()
    assert served == asked


def test_lidar_stream_serves_preencoded_tokens_once(monkeypatch):
    from cosmos3_nano_sim_auto.impl.lidar import LidarAssets, LidarStream

    calls = []
    tokenizer = SimpleNamespace(patchify=lambda z: z)
    sample = SimpleNamespace(
        controls=torch.arange(6.0).reshape(6, 1, 1, 1), condition=None
    )
    stream = LidarStream.__new__(LidarStream)
    stream.assets = LidarAssets(cast(Any, tokenizer), cast(Any, sample))
    stream.encoder = lambda clip: calls.append(clip.shape[2]) or clip
    stream.next_sweep, stream._ready = 0, {}
    monkeypatch.setattr("cosmos3_nano_sim_auto.impl.lidar.CHUNKS_AHEAD", 1)
    stream.preencode(iter([(0, 1), (1, 5), (5, 9)]))
    assert calls == [1] and stream.next_sweep == 1  # one chunk ahead, not the clip
    assert torch.equal(
        stream.tokens(0, 1),
        stream.assets.sample.controls[0:1].permute(1, 0, 2, 3)[None],
    )
    assert calls == [1, 4] and stream.next_sweep == 5  # taking one encodes the next
    assert torch.equal(
        stream.tokens(1, 5),
        stream.assets.sample.controls[1:5].permute(1, 0, 2, 3)[None],
    )
    assert calls == [1, 4]  # (5, 9) is past the recording: nothing more
    with pytest.raises(ValueError, match="consecutively"):
        stream.tokens(1, 5)


def test_lidar_stream_without_a_worker_decodes_on_demand_and_the_rollout_asks_before_the_clean_pass():
    from cosmos3_nano_sim_auto.impl.lidar import LidarStream

    stream = LidarStream.__new__(LidarStream)
    stream.decoder, stream._worker = (lambda z: z + 1), None
    latent = torch.zeros(1, 3, 4, 2, 3)
    stream.decode_ahead(latent)  # no worker: nothing to start
    assert torch.equal(stream.ranges(latent), latent + 1)

    rollout = make_rollout()
    handed = []
    rollout.lidar_controls = SimpleNamespace(
        tokens=rollout.lidar_controls.tokens, decode_ahead=lambda z: handed.append(z)
    )
    rollout.step()
    assert len(handed) == 1 and handed[0] is rollout.last_lidar


def test_lidar_decode_worker_round_trips_one_chunk_and_reports_failures():
    from cosmos3_nano_sim_auto.impl.lidar import LidarDecodeWorker, LidarStream

    class Pipe:
        def __init__(self):
            self.sent, self.replies = [], []

        def send(self, latent):
            self.sent.append(latent)

        def recv(self):
            return self.replies.pop(0)

    pipe = Pipe()
    worker = LidarDecodeWorker(cast(Any, pipe), device=torch.device("cpu"))
    stream = LidarStream.__new__(LidarStream)
    stream._worker = worker
    stream.decoder = lambda z: pytest.fail("the worker decodes, not this process")
    latent = torch.arange(6.0).reshape(1, 1, 6, 1, 1)
    pipe.replies.append(latent + 1)
    stream.decode_ahead(latent)
    assert torch.equal(pipe.sent[0], latent) and pipe.sent[0].device.type == "cpu"
    with pytest.raises(ValueError, match="in flight"):
        stream.decode_ahead(latent)
    assert torch.equal(stream.ranges(latent), latent + 1)
    with pytest.raises(ValueError, match="no LiDAR chunk"):
        worker.take()
    pipe.replies.append("Traceback: boom")
    stream.decode_ahead(latent)
    with pytest.raises(RuntimeError, match="boom"):
        stream.ranges(latent)


def test_lidar_decode_worker_flag_parses_and_assets_carry_the_checkpoint():
    from cosmos3_nano_sim_auto.impl.app import _parse
    from cosmos3_nano_sim_auto.impl.lidar import LidarAssets

    args = _parse(
        [
            "--sample",
            "s",
            "--lidar-sample",
            "l.npz",
            "--lidar-checkpoint",
            "w.pt",
            "--lidar-decode-worker",
        ]
    )
    assert args.lidar_decode_worker
    with pytest.raises(ValueError, match="needs --lidar-sample"):
        _parse(["--sample", "s", "--lidar-decode-worker"])
    assert LidarAssets(cast(Any, object()), cast(Any, object())).checkpoint is None
    stream = LidarAssets(
        cast(Any, object()), cast(Any, SimpleNamespace(controls=())), checkpoint=None
    ).stream()
    with pytest.raises(ValueError, match="checkpoint"):
        stream.decode_in_worker(worker_device=0, device=torch.device("cpu"))

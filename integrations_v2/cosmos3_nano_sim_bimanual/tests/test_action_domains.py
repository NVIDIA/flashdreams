# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-row action domains across hand encoding, denoising, and cache refresh."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import FrozenInstanceError
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
from cosmos3_nano_sim_bimanual.encoder import (
    HandActionEncoder,
    HandActionEncoderConfig,
    HandActionInput,
)
from cosmos3_nano_sim_bimanual.tests.test_kv_rollout import (
    _CHUNK,
    _TCF,
    _actions,
    _net,
    _session,
)
from cosmos3_nano_sim_bimanual.transformer import network as network_module
from cosmos3_nano_sim_bimanual.transformer.config import CacheRefreshMode
from torch import Tensor

from flashdreams.infra.cuda_graph import CUDAGraphWrapper

pytestmark = pytest.mark.ci_cpu


def test_encoder_preserves_domains_and_normalizes_action_storage() -> None:
    encoder = HandActionEncoder(HandActionEncoderConfig())
    actions = torch.ones(64, 8, dtype=torch.float64).T
    domains = torch.tensor([2, 15, 15, 2, 15, 2, 2, 15])
    encoded = encoder(HandActionInput(actions, domains))

    assert isinstance(encoded, HandActionInput)
    assert encoded.actions.dtype == torch.float32
    assert encoded.actions.is_contiguous()
    torch.testing.assert_close(encoded.actions, actions.float())
    torch.testing.assert_close(encoded.domain_ids, domains)
    assert encoded.domain_ids.dtype == torch.long
    with pytest.raises(FrozenInstanceError):
        encoded.domain_ids = domains.flip(0)


def test_legacy_tensor_defaults_every_action_row_to_agibot() -> None:
    actions = _actions(_CHUNK)
    encoded = HandActionEncoder(HandActionEncoderConfig())(actions)
    torch.testing.assert_close(encoded.actions, actions)
    torch.testing.assert_close(
        encoded.domain_ids, torch.full((len(actions),), 15, dtype=torch.long)
    )


@pytest.mark.parametrize(
    "domains",
    [
        torch.tensor(15),
        torch.full((8, 1), 15, dtype=torch.long),
        torch.full((7,), 15, dtype=torch.long),
        torch.full((8,), 15, dtype=torch.int32),
        torch.full((8,), 15.0),
        torch.full((8,), True),
        torch.tensor([2, 15, 15, 0, 15, 15, 15, 15]),
        torch.full((8,), 16, dtype=torch.long),
        torch.full((8,), -1, dtype=torch.long),
    ],
)
def test_encoder_rejects_invalid_domains(domains: Tensor) -> None:
    encoder = HandActionEncoder(HandActionEncoderConfig())
    with pytest.raises((TypeError, ValueError)):
        encoder(HandActionInput(_actions(_CHUNK), domains))


@pytest.mark.parametrize("shape", [(0, 64), (8, 63), (2, 4, 64)])
def test_encoder_rejects_invalid_action_shapes(shape: tuple[int, ...]) -> None:
    encoder = HandActionEncoder(HandActionEncoderConfig())
    with pytest.raises(ValueError, match="shape"):
        encoder(HandActionInput(torch.zeros(shape), torch.full((8,), 15)))


@torch.inference_mode()
def test_network_selects_each_rows_domain_embedding() -> None:
    net, _ = _net()
    actions = _actions(_CHUNK)
    domains = torch.tensor([2, 15, 2, 2, 15, 15, 2, 15])
    actual = net.encode_action(actions, domain_id=domains)
    expected = torch.cat(
        [
            net.encode_action(row[None], domain_id=int(domain))
            for row, domain in zip(actions, domains, strict=True)
        ]
    )
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, net.encode_action(actions, domain_id=15))


def _valid_length_cpu(
    q: Tensor,
    k_all: Tensor,
    v_all: Tensor,
    valid_cached: Tensor,
    current_tokens: int,
    *,
    causal_block_tokens: int | None = None,
) -> Tensor:
    """Evaluate the valid prefix with CPU SDPA instead of CUDA FlashAttention."""
    valid = int(valid_cached)
    k = torch.cat((k_all[..., :valid, :], k_all[..., -current_tokens:, :]), dim=-2)
    v = torch.cat((v_all[..., :valid, :], v_all[..., -current_tokens:, :]), dim=-2)
    mask = None
    if causal_block_tokens is not None:
        frame = torch.arange(current_tokens) // causal_block_tokens
        mask = torch.cat(
            (
                torch.ones(current_tokens, valid, dtype=torch.bool),
                frame[None, :] <= frame[:, None],
            ),
            dim=1,
        )
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)


@pytest.mark.parametrize("refresh", ["chunk", "framewise"])
@pytest.mark.parametrize("dispatch", ["dynamic", "fixed", "valid_length"])
@torch.inference_mode()
def test_mixed_domains_match_across_cache_dispatch_and_reset(
    monkeypatch: pytest.MonkeyPatch,
    refresh: CacheRefreshMode,
    dispatch: str,
) -> None:
    # Exercise the actual compile closure and dispatch without compiling CPU code.
    monkeypatch.setattr(torch, "compile", lambda fn, **_kwargs: fn)
    monkeypatch.setattr(
        network_module, "valid_length_cached_attention", _valid_length_cpu
    )
    shared = _net()
    eager = _session(
        shared=shared,
        cache_mode="block",
        window_frames=2 * _CHUNK,
        cache_refresh_mode=refresh,
    )
    compiled = _session(
        shared=shared,
        cache_mode="block",
        window_frames=2 * _CHUNK,
        cache_refresh_mode=refresh,
        compile_denoise=True,
        fixed_shape_cache_fill=dispatch == "fixed",
    )
    if dispatch == "valid_length":
        # The production constructor requires CUDA for its attention kernel.
        compiled.valid_length_cache_attention = True
        compiled._compile_start_step = 0
    actions = _actions(_CHUNK)
    domains = torch.tensor([2, 15, 15, 2, 15, 2, 2, 15])

    for steps in (4, 2):
        for index in range(steps):
            current = domains.roll(index)
            control = HandActionInput(actions, current)
            expected = eager.step(control)
            actual = compiled.step(control)
            torch.testing.assert_close(compiled._action_domain_ids, current)
            for expected_frame, actual_frame in zip(expected, actual, strict=True):
                torch.testing.assert_close(
                    actual_frame, expected_frame, rtol=2e-4, atol=2e-4
                )
        eager.reset_rollout()
        compiled.reset_rollout()
        assert eager._action_domain_ids is None
        assert compiled._action_domain_ids is None


@pytest.mark.parametrize("mode", ["cat", "chunk", "framewise", "valid_length"])
@torch.inference_mode()
def test_refresh_encodes_the_domains_of_each_action_slice(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.setattr(torch, "compile", lambda fn, **_kwargs: fn)
    monkeypatch.setattr(
        network_module, "valid_length_cached_attention", _valid_length_cpu
    )
    session = _session(
        cache_mode="cat" if mode == "cat" else "block",
        cache_refresh_mode="framewise" if mode == "framewise" else "chunk",
        window_frames=2 * _CHUNK,
        compile_denoise=mode == "valid_length",
    )
    if mode == "valid_length":
        session.valid_length_cache_attention = True
        session._compile_start_step = 0
    seen: list[Tensor] = []
    original = session.net.encode_action

    def record(actions: Tensor, *, domain_id: int | Tensor) -> Tensor:
        assert isinstance(domain_id, Tensor)
        assert len(domain_id) == len(actions)
        seen.append(domain_id.clone())
        return original(actions, domain_id=domain_id)

    monkeypatch.setattr(session.net, "encode_action", record)
    domains = torch.tensor([2, 15, 15, 2, 15, 2, 2, 15])
    session.step(HandActionInput(_actions(_CHUNK), domains))
    denoise_steps = session.model.scheduler.config.num_inference_steps
    for actual in seen[:denoise_steps]:
        torch.testing.assert_close(actual, domains)
    refresh_domains = seen[denoise_steps:]
    expected = list(domains.split(_TCF)) if mode == "framewise" else [domains]
    assert len(refresh_domains) == len(expected)
    for actual, reference in zip(refresh_domains, expected, strict=True):
        torch.testing.assert_close(actual, reference)


@torch.inference_mode()
def test_cache_legacy_actions_do_not_reuse_previous_domains() -> None:
    cache = _session().cache_object
    actions = _actions(_CHUNK)
    cache.prepare(actions, domain_ids=torch.full((len(actions),), 2))
    cache.prepare(actions)
    torch.testing.assert_close(
        cache._action_domain_ids, torch.full((len(actions),), 15)
    )


@torch.inference_mode()
def test_async_refresh_keeps_domain_storage_until_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _session(cache_mode="block", window_frames=2 * _CHUNK).cache_object
    cache._cache_stream = Mock()
    current, ready, done = Mock(), Mock(), Mock()
    done.query.return_value = False
    events = iter((ready, done))
    monkeypatch.setattr(torch.cuda, "Event", lambda: next(events))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: current)
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())
    recorded: list[Tensor] = []

    def record_stream(tensor: Tensor, stream: object) -> None:
        assert stream is cache._cache_stream
        recorded.append(tensor)

    monkeypatch.setattr(Tensor, "record_stream", record_stream)
    refresh = Mock()
    monkeypatch.setattr(cache, "_refresh_chunk", refresh)
    actions = _actions(_CHUNK)
    domains = torch.tensor([2, 15, 15, 2, 15, 2, 2, 15])
    clean = torch.zeros(_CHUNK * cache.hw, cache.spec.vfm.patch_latent_dim)

    cache._submit_refresh(1, clean, actions, _CHUNK, domains)

    assert len(cache._pending_cache) == 1
    event, retained = cache._pending_cache[0]
    assert event is done
    assert len(retained) == 3
    assert refresh.call_args.args[-1] is retained[-1]
    assert any(tensor is retained[-1] for tensor in recorded)
    assert retained[-1].data_ptr() == domains.data_ptr()
    torch.testing.assert_close(retained[-1], domains)
    current.record_event.assert_called_once_with(ready)
    cache._cache_stream.wait_event.assert_called_once_with(ready)
    done.record.assert_called_once_with(cache._cache_stream)

    cache.wait_for_pending_cache_update()
    assert len(cache._pending_cache) == 1
    done.query.return_value = True
    cache.wait_for_pending_cache_update()
    assert cache._pending_cache == []


@torch.inference_mode()
def test_dynamo_closure_stages_changed_domains_in_the_same_graph_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compile_fn = torch.compile
    monkeypatch.setattr(
        torch,
        "compile",
        lambda fn, **_kwargs: compile_fn(
            fn, backend="eager", fullgraph=True, dynamic=False
        ),
    )
    cache = _session(
        cache_mode="block", window_frames=2 * _CHUNK, compile_denoise=True
    ).cache_object
    wrapper = CUDAGraphWrapper(cache._denoise_step_fn)
    actions = _actions(_CHUNK)
    latent = torch.zeros(cache.channels, _CHUNK, cache.h_lat, cache.w_lat)
    sigma = torch.tensor([0.5])
    mixed = torch.tensor([2, 15, 15, 2, 15, 2, 2, 15])
    pointer = None
    outputs = []
    for domains in (torch.full_like(mixed, 15), torch.full_like(mixed, 2), mixed):
        cache.reset_rollout()
        cache.prepare(actions, domain_ids=domains)
        assert cache._compiled_args is not None
        actual = wrapper.drain(latent, sigma, *cache._compiled_args)
        expected = cache._denoise_step_fn(latent, sigma, *cache._compiled_args)
        torch.testing.assert_close(actual, expected)
        staged = wrapper._static_args[5]
        torch.testing.assert_close(staged, domains)
        assert staged.data_ptr() != domains.data_ptr()
        if pointer is not None:
            assert staged.data_ptr() == pointer
        pointer = staged.data_ptr()
        outputs.append(actual.clone())
    assert not torch.allclose(outputs[0], outputs[1])
    assert not torch.allclose(outputs[1], outputs[2])

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
from flashdreams.runtime.demo import registry

pytestmark = pytest.mark.ci_cpu


class _EntryPoint:
    def __init__(self, value: object) -> None:
        self.value = value

    def load(self) -> object:
        return self.value


class _EntryPoints(tuple):
    def select(self, *, group: str) -> _EntryPoints:
        assert group == registry.DEMO_ADAPTER_ENTRY_POINT_GROUP
        return self


def _adapter(model_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        model_id=model_id,
        inference_input_schema=object(),
        inference_output_schema=object(),
        supported_routes=lambda: (),
        prepare_session=lambda spec: spec,
        create_demo_runtime=lambda spec: spec,
        create_runtime=lambda config: config,
        list_sessions=lambda spec: (spec,),
    )


def test_discovery_materializes_registered_adapter_factories(monkeypatch) -> None:
    monkeypatch.setattr(
        registry.metadata,
        "entry_points",
        lambda: _EntryPoints((_EntryPoint(lambda: _adapter("model-a")),)),
    )

    adapters = registry.discover_demo_adapters()

    assert tuple(adapters) == ("model-a",)


def test_duplicate_model_identity_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        registry.metadata,
        "entry_points",
        lambda: _EntryPoints(
            (
                _EntryPoint(lambda: _adapter("duplicate")),
                _EntryPoint(lambda: _adapter("duplicate")),
            )
        ),
    )

    with pytest.raises(ValueError, match="Duplicate demo adapter"):
        registry.discover_demo_adapters()


def test_malformed_adapter_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        registry.metadata,
        "entry_points",
        lambda: _EntryPoints((_EntryPoint(lambda: SimpleNamespace(model_id="bad")),)),
    )

    with pytest.raises(TypeError, match="is missing"):
        registry.discover_demo_adapters()

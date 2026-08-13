# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Token codec contracts for encoding latent frames onto the wire."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from flashdreams.infra.config import InstantiateConfig

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True, slots=True)
class TokenCodecEncodeResult:
    """Result of encoding a single latent frame."""

    payload: bytes
    """Encoded frame payload written to the wire."""

    frame_params: bytes = b""
    """Optional per-frame codec parameters carried alongside the payload."""


ConfigT = TypeVar("ConfigT", bound="TokenCodecConfig")


class TokenCodec(ABC, Generic[ConfigT]):
    """Encodes latent frames into payload bytes for the token stream."""

    config: ConfigT

    def __init__(self, config: ConfigT) -> None:
        self.config = config

    @property
    @abstractmethod
    def codec_id(self) -> str:
        """Stable identifier advertised to the client in the session header."""

    @property
    def static_params(self) -> dict[str, Any]:
        """Codec parameters constant for the whole session.

        Sent once in the session header so the client can configure its
        decoder before any token frame arrives.
        """
        return {}

    @abstractmethod
    def encode_frame(self, latent: torch.Tensor) -> TokenCodecEncodeResult:
        """Encode a single latent frame into a payload and optional params."""


@dataclass(kw_only=True)
class TokenCodecConfig(InstantiateConfig):
    """Base config for a token codec implementation."""

    _target: type[TokenCodec] = field(default_factory=lambda: TokenCodec)

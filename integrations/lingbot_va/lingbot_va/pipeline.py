"""LingBot-VA inference pipeline config shell."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from flashdreams.infra.pipeline import StreamInferencePipeline, StreamInferencePipelineConfig


@dataclass(kw_only=True)
class LingbotVAInferencePipelineConfig(StreamInferencePipelineConfig):
    """Pipeline config for the LingBot-VA Robotwin I2AV integration.

    The full generation body is intentionally staged behind the transformer
    port. Keeping this as a normal ``StreamInferencePipelineConfig`` makes the
    runner discoverable and ``--no-instantiate``-safe while subsequent commits
    can swap in the native DiT implementation without changing the public slug.
    """

    _target: type["LingbotVAInferencePipeline"] = field(
        default_factory=lambda: LingbotVAInferencePipeline
    )

    checkpoint_root: str
    robotwin_height: int = 256
    robotwin_width: int = 320
    frame_chunk_size: int = 2
    action_dim: int = 30
    action_per_frame: int = 16
    attn_window: int = 64
    sink_size: int = 2
    latent_height: int = 24
    latent_width: int = 20
    latent_channels: int = 48
    latent_token_per_chunk: int = 240
    action_token_per_chunk: int = 32


class LingbotVAInferencePipeline(StreamInferencePipeline):
    """Current CPU-safe pipeline shell for LingBot-VA configs."""

    config: LingbotVAInferencePipelineConfig

    def __init__(self, config: LingbotVAInferencePipelineConfig) -> None:
        super().__init__(config)
        self.config = config

    def initialize_lingbot_va_cache(self, **context: Any) -> Any:
        """Placeholder for the future Robotwin-specific cache initialization API."""
        raise NotImplementedError(
            "LingBot-VA Robotwin cache initialization awaits the native DiT/VAE "
            "port. Use --no-instantiate or CPU tests for the scaffold stage."
        )

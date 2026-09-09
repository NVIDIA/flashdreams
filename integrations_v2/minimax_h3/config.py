# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native MiniMax H3 workflow configurations."""

from flashdreams.infra.config import derive_config
from minimax_h3.impl.pipeline import MiniMaxH3PipelineConfig

PIPELINE_MINIMAX_H3_T2VA = MiniMaxH3PipelineConfig(
    name="minimax-h3-t2va", workflow="t2va"
)
PIPELINE_MINIMAX_H3_FL2VA = derive_config(
    PIPELINE_MINIMAX_H3_T2VA, name="minimax-h3-fl2va", workflow="fl2va"
)
PIPELINE_MINIMAX_H3_REF2VA = derive_config(
    PIPELINE_MINIMAX_H3_T2VA, name="minimax-h3-ref2va", workflow="ref2va"
)

MINIMAX_H3_CONFIGS = {
    config.name: config
    for config in (
        PIPELINE_MINIMAX_H3_T2VA,
        PIPELINE_MINIMAX_H3_FL2VA,
        PIPELINE_MINIMAX_H3_REF2VA,
    )
}

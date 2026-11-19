/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/** Messages exchanged with the FlashDreams WebRTC data channel. */
export type Gear = 'P' | 'R' | 'N' | 'D';

export interface SimulationTelemetry {
  type: 'telemetry';
  speed_mps: number;
  speed_kmh: number;
  gear: Gear;
  steer: number;
  throttle: number;
  brake: number;
  model_latency_ms: number;
  blocks_generated: number;
  active_view: string;
  warning: string | null;
  bev_image: string | null;
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Registry mapping wire codec ids to their client-side decoder instances. The
// session header names one codec by id; the client looks it up here before
// decoding any token frame.

import { RawFloat16Decoder } from "./raw_f16.js"
import { SASDecoder } from "./sas.js"

/**
 * Codec id -> decoder instance. Keys must match the server-side
 * ``TokenCodec.codec_id`` values exactly.
 */
export const TOKEN_CODEC_REGISTRY = new Map([
  ["raw_f16", new RawFloat16Decoder()],
  // One entry per SAS preset the server can advertise, since SASTokenCodec.codec_id is
  // built from num_bits and n_stages. Separate instances rather than one shared object
  // so per-session configure() state cannot leak across codecs; the class itself is
  // preset-agnostic and reads every parameter from the session header.
  ["sas-int8-8s", new SASDecoder("sas-int8-8s")],
  ["sas-int4-8s", new SASDecoder("sas-int4-8s")],
  ["sas-int2-8s", new SASDecoder("sas-int2-8s")],
])

/**
 * Look up the decoder for a codec id.
 *
 * @param {string} codecId codec id from the session header
 * @returns {object} decoder instance with configure()/decode()
 * @throws {Error} when the codec id is not registered
 */
export function getDecoder(codecId) {
  const decoder = TOKEN_CODEC_REGISTRY.get(codecId)
  if (!decoder) {
    const known = Array.from(TOKEN_CODEC_REGISTRY.keys()).join(", ")
    throw new Error(
      `unknown token codec "${codecId}"; registered codecs: ${known || "none"}`
    )
  }
  return decoder
}

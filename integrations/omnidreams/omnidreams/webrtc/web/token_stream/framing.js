// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Client half of the shared token-stream wire contract.
//
// This file MUST match the server framing module byte-for-byte
// (serving/token_stream/framing.py). The binary layout is:
//
//   18-byte header + optional codec_params + payload, little-endian.
//   struct "<4sBBIBBHI":
//     offset 0  MAGIC             4 bytes  0xFD 0x54 0x4F 0x4B  (b"\xfdTOK")
//     offset 4  VERSION           u8       = 1
//     offset 5  FLAGS             u8       bit0 KEYFRAME, bit1 LAST_IN_CHUNK, bit7 CONTROL
//     offset 6  CHUNK_ID          u32
//     offset 10 FRAME_IDX         u8
//     offset 11 FRAME_TOTAL       u8
//     offset 12 CODEC_PARAMS_LEN  u16
//     offset 14 PAYLOAD_LEN       u32
//     offset 18 CODEC_PARAMS (CODEC_PARAMS_LEN bytes) then PAYLOAD (PAYLOAD_LEN bytes)
//
// Control frames carry the session header JSON as their payload.

export const MAGIC = new Uint8Array([0xfd, 0x54, 0x4f, 0x4b])
export const PROTOCOL_VERSION = 1
export const HEADER_SIZE = 18

export const FLAG_KEYFRAME = 0x01
export const FLAG_LAST_IN_CHUNK = 0x02
export const FLAG_CONTROL = 0x80

// Control frames use a sentinel chunk id.
export const CONTROL_CHUNK_ID = 0xffffffff

/**
 * Return true when the first four bytes of the buffer are the token-stream
 * magic. Accepts an ArrayBuffer or a DataView.
 */
export function hasMagic(bufferOrView) {
  const view = toDataView(bufferOrView)
  if (view.byteLength < MAGIC.length) {
    return false
  }
  for (let i = 0; i < MAGIC.length; i += 1) {
    if (view.getUint8(i) !== MAGIC[i]) {
      return false
    }
  }
  return true
}

/**
 * Parse the 18-byte binary header. Accepts an ArrayBuffer or a DataView and
 * reads every multi-byte field little-endian. Throws when the buffer is too
 * short or the magic/version do not match the contract.
 */
export function parseHeader(bufferOrView) {
  const view = toDataView(bufferOrView)
  if (view.byteLength < HEADER_SIZE) {
    throw new Error(
      `token frame too short: ${view.byteLength} < ${HEADER_SIZE} bytes`
    )
  }
  if (!hasMagic(view)) {
    throw new Error("token frame magic mismatch")
  }

  const version = view.getUint8(4)
  if (version !== PROTOCOL_VERSION) {
    throw new Error(
      `token frame version mismatch: ${version} != ${PROTOCOL_VERSION}`
    )
  }

  const flags = view.getUint8(5)
  const chunkId = view.getUint32(6, true)
  const frameIdx = view.getUint8(10)
  const frameTotal = view.getUint8(11)
  const codecParamsLen = view.getUint16(12, true)
  const payloadLen = view.getUint32(14, true)

  return {
    version,
    flags,
    chunkId,
    frameIdx,
    frameTotal,
    codecParamsLen,
    payloadLen,
    isControl: (flags & FLAG_CONTROL) !== 0,
    isKeyframe: (flags & FLAG_KEYFRAME) !== 0,
    isLastInChunk: (flags & FLAG_LAST_IN_CHUNK) !== 0,
  }
}

/**
 * Slice the codec-params and payload byte ranges out of a full binary frame,
 * given its parsed header. Returns ArrayBuffers so decoders can adopt them
 * directly. The base offset lets callers pass a view into a larger buffer.
 */
export function sliceFrameBody(bufferOrView, header, baseOffset = 0) {
  const view = toDataView(bufferOrView)
  const paramsStart = baseOffset + HEADER_SIZE
  const payloadStart = paramsStart + header.codecParamsLen
  const payloadEnd = payloadStart + header.payloadLen
  if (payloadEnd > view.byteOffset + view.byteLength) {
    throw new Error("token frame body extends past buffer end")
  }
  const source = view.buffer
  return {
    codecParams: source.slice(
      view.byteOffset + paramsStart,
      view.byteOffset + payloadStart
    ),
    payload: source.slice(
      view.byteOffset + payloadStart,
      view.byteOffset + payloadEnd
    ),
  }
}

/**
 * Build the JSON string the client sends back over the token-stream socket to
 * acknowledge a fully rendered chunk. Server side matches
 * {"type": "token_frame_ack", "chunk_id": <int>}.
 */
export function buildAckMessage(chunkId) {
  return JSON.stringify({ type: "token_frame_ack", chunk_id: chunkId })
}

function toDataView(bufferOrView) {
  if (bufferOrView instanceof DataView) {
    return bufferOrView
  }
  if (bufferOrView instanceof ArrayBuffer) {
    return new DataView(bufferOrView)
  }
  if (ArrayBuffer.isView(bufferOrView)) {
    return new DataView(
      bufferOrView.buffer,
      bufferOrView.byteOffset,
      bufferOrView.byteLength
    )
  }
  throw new TypeError("expected ArrayBuffer, DataView, or typed array")
}

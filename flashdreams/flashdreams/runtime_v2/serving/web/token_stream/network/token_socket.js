// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// WebSocket transport for the token stream. Opens a dedicated binary socket,
// parses each message header per the shared wire contract, and routes control
// frames (session header JSON) versus token frames to callbacks. Chunk acks
// are sent back over the same socket as JSON text.

import {
  buildAckMessage,
  parseHeader,
  sliceFrameBody,
} from "../framing.js"

/**
 * @typedef {Object} TokenStreamSocketCallbacks
 * @property {(header: object) => void} [onSessionHeader] parsed session header JSON
 * @property {(frame: object) => void} [onFrame] parsed token frame descriptor
 * @property {() => void} [onOpen]
 * @property {(event: CloseEvent) => void} [onClose]
 * @property {(error: Error) => void} [onError]
 */
export class TokenStreamSocket {
  /**
   * @param {string} url token-stream endpoint URL
   * @param {TokenStreamSocketCallbacks} [callbacks]
   */
  constructor(url, callbacks = {}) {
    this._url = url
    this._callbacks = callbacks
    this._socket = null
    this._decoder = new TextDecoder("utf-8")
  }

  /** Open the socket and begin routing messages. Safe to call once. */
  open() {
    if (this._socket) {
      return
    }
    const socket = new WebSocket(this._url)
    socket.binaryType = "arraybuffer"
    this._socket = socket

    socket.addEventListener("open", () => {
      this._callbacks.onOpen?.()
    })
    socket.addEventListener("close", (event) => {
      this._callbacks.onClose?.(event)
    })
    socket.addEventListener("error", () => {
      this._callbacks.onError?.(new Error("token-stream socket error"))
    })
    socket.addEventListener("message", (event) => {
      this._handleMessage(event.data)
    })
  }

  /** Close the socket if it is open. */
  close() {
    if (this._socket && this._socket.readyState !== WebSocket.CLOSED) {
      this._socket.close()
    }
    this._socket = null
  }

  get isOpen() {
    return Boolean(this._socket) && this._socket.readyState === WebSocket.OPEN
  }

  /**
   * Acknowledge a fully rendered chunk so the server can release a flow-control
   * slot. No-op when the socket is not open.
   */
  sendAck(chunkId) {
    if (!this.isOpen) {
      return
    }
    this._socket.send(buildAckMessage(chunkId))
  }

  _handleMessage(data) {
    // Only binary frames carry header + payload. Ignore stray text frames.
    if (!(data instanceof ArrayBuffer)) {
      return
    }
    let header
    try {
      header = parseHeader(data)
    } catch (error) {
      this._callbacks.onError?.(error)
      return
    }

    const { codecParams, payload } = sliceFrameBody(data, header)

    if (header.isControl) {
      this._routeControl(payload)
      return
    }

    this._callbacks.onFrame?.({
      chunkId: header.chunkId,
      frameIdx: header.frameIdx,
      frameTotal: header.frameTotal,
      isKeyframe: header.isKeyframe,
      isLastInChunk: header.isLastInChunk,
      codecParams,
      payload,
    })
  }

  _routeControl(payloadBuffer) {
    let parsed
    try {
      const text = this._decoder.decode(new Uint8Array(payloadBuffer))
      parsed = JSON.parse(text)
    } catch (error) {
      this._callbacks.onError?.(
        new Error(`invalid session header: ${error.message}`)
      )
      return
    }
    this._callbacks.onSessionHeader?.(parsed)
  }
}

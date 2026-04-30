const connectButton = document.getElementById("connectButton")
const statusText = document.getElementById("statusText")
const flowText = document.getElementById("flowText")
const eventLog = document.getElementById("eventLog")
const remoteVideo = document.getElementById("remoteVideo")

const allowedKeys = new Set(["w", "a", "s", "d", "q", "e", "i", "j", "k", "l"])
const activeKeys = new Set()

let peerConnection = null
let controlChannel = null
let inferenceInFlight = false
let connected = false

function logEvent(message) {
  const stamp = new Date().toLocaleTimeString()
  eventLog.textContent = `[${stamp}] ${message}\n${eventLog.textContent}`.slice(0, 5000)
}

function setStatus(message) {
  statusText.textContent = message
}

function setFlow(message) {
  flowText.textContent = message
}

function normalizeKey(rawKey) {
  return String(rawKey || "").toLowerCase()
}

function sendControlAction(action) {
  if (!connected || !controlChannel || controlChannel.readyState !== "open") {
    return false
  }
  inferenceInFlight = true
  controlChannel.send(
    JSON.stringify({
      type: "action",
      action,
    })
  )
  setFlow(`sent ${action.event}${action.key ? `:${action.key}` : ""}, waiting=${inferenceInFlight}`)
  return true
}

function enqueueAction(action) {
  const sent = sendControlAction(action)
  if (!sent) {
    setFlow(`not_sent ${action.event}${action.key ? `:${action.key}` : ""}`)
  }
}

function handleControlMessage(rawMessage) {
  let payload
  try {
    payload = JSON.parse(rawMessage)
  } catch (error) {
    logEvent(`invalid control payload: ${rawMessage}`)
    return
  }

  if (payload.type === "chunk_done") {
    inferenceInFlight = false
    logEvent(
      `chunk_done index=${payload.chunk_index}, frames=${payload.num_frames}, enqueued=${payload.enqueued_frames}`
    )
    if (activeKeys.size > 0) {
      enqueueAction({ event: "step" })
    }
    return
  }

  if (payload.type === "busy") {
    logEvent(`server busy: ${payload.message}`)
    return
  }

  if (payload.type === "error") {
    inferenceInFlight = false
    logEvent(`server error: ${payload.message}`)
    return
  }

  logEvent(`server message: ${rawMessage}`)
}

async function waitForIceGatheringComplete(pc) {
  if (pc.iceGatheringState === "complete") {
    return
  }
  await new Promise((resolve) => {
    const onStateChange = () => {
      if (pc.iceGatheringState === "complete") {
        pc.removeEventListener("icegatheringstatechange", onStateChange)
        resolve()
      }
    }
    pc.addEventListener("icegatheringstatechange", onStateChange)
  })
}

async function connectSession() {
  connectButton.disabled = true
  setStatus("connecting")
  setFlow("creating peer connection")

  try {
    peerConnection = new RTCPeerConnection()
    controlChannel = peerConnection.createDataChannel("controls")
    peerConnection.addTransceiver("video", { direction: "recvonly" })

    controlChannel.onopen = () => {
      logEvent("control data channel open")
      setFlow("ready for action")
    }
    controlChannel.onclose = () => {
      logEvent("control data channel closed")
      setFlow("channel closed")
    }
    controlChannel.onmessage = (event) => {
      handleControlMessage(event.data)
    }

    peerConnection.ontrack = (event) => {
      const [stream] = event.streams
      if (stream) {
        remoteVideo.srcObject = stream
      }
    }

    peerConnection.onconnectionstatechange = () => {
      setStatus(peerConnection.connectionState)
      logEvent(`connection_state=${peerConnection.connectionState}`)
      if (peerConnection.connectionState === "connected") {
        connected = true
        setFlow("connected; waiting for keyboard action")
      }
      if (["failed", "closed", "disconnected"].includes(peerConnection.connectionState)) {
        connected = false
      }
    }

    const offer = await peerConnection.createOffer()
    await peerConnection.setLocalDescription(offer)
    await waitForIceGatheringComplete(peerConnection)

    const response = await fetch("/api/webrtc/offer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(peerConnection.localDescription),
    })
    if (!response.ok) {
      const text = await response.text()
      throw new Error(`offer failed (${response.status}): ${text}`)
    }
    const answer = await response.json()
    await peerConnection.setRemoteDescription(answer)
    logEvent("offer/answer completed")
  } catch (error) {
    setStatus("error")
    setFlow("failed")
    logEvent(`connect failed: ${error.message}`)
    connectButton.disabled = false
  }
}

function handleKeyDown(event) {
  if (!connected) {
    return
  }

  const key = normalizeKey(event.key)
  if (!allowedKeys.has(key)) {
    return
  }
  event.preventDefault()

  if (event.repeat) {
    return
  }
  activeKeys.add(key)
  enqueueAction({ event: "keydown", key })
}

function handleKeyUp(event) {
  if (!connected) {
    return
  }

  const key = normalizeKey(event.key)
  if (!allowedKeys.has(key)) {
    return
  }
  event.preventDefault()

  activeKeys.delete(key)
  enqueueAction({ event: "keyup", key })
}

connectButton.addEventListener("click", () => {
  void connectSession()
})
window.addEventListener("keydown", handleKeyDown)
window.addEventListener("keyup", handleKeyUp)

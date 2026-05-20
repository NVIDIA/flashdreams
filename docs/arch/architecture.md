<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Architecture — FlashDreams

Compact, FlashDreams-scoped four-view architecture (static / dynamic / data / deployment) for the `flashdreams` repo. The broader omni-dreams + flashdreams architecture lives in [`../../../omni-dreams/docs/arch/architecture.md`](../../../omni-dreams/docs/arch/architecture.md); this document is the **TOE = flashdreams** subset that the [`sadd.md`](sadd.md), [`tava.md`](tava.md), and [`fsr_table.md`](fsr_table.md) reference.

Diagram-marker convention used throughout:

| Marker | Meaning |
|---|---|
| Unannotated node / edge | As-built on HEAD |
| Node / edge marked **(FSR_FD_NN)** | TAVA-recommended; contract+stub upstream at `../../../omni-dreams/internal/tests/security/<topic>.py` |
| OE-1..OE-7 callouts | Operational-environment assumption; not enforced by the TOE |

---

## Static view

### 1. System

```mermaid
flowchart LR
    classDef pkg fill:#eef,stroke:#447
    classDef ext fill:#ffe,stroke:#a82,stroke-dasharray: 3 3
    classDef adapter fill:#efe,stroke:#272

    subgraph FD[flashdreams repo]
        FD_CORE[flashdreams.core attention · distributed · checkpoint · io]:::pkg
        FD_INFRA[flashdreams.infra pipeline · diffusion · encoder · decoder · runner]:::pkg
        FD_REC[flashdreams.recipes cosmos · taehv · template · wan]:::pkg
        FD_PLUG[flashdreams.plugins registry · discovery]:::pkg
        FD_CLI[flashdreams.scripts.cli  console: flashdreams-run]:::pkg
        FD_INT_ALPA[integrations alpadreams · ludus-renderer]:::adapter
        FD_INT_LING[integrations lingbot]:::adapter
        FD_INT_OTHERS[integrations cosmos_predict2 · wan21 · self_forcing · causal_forcing · fastvideo_causal_wan22]:::adapter
    end

    HF[(HuggingFace nvidia/omni-dreams-* · nvidia/Cosmos-1.0-Guardrail)]:::ext
    S3[(S3 s3 flashdreams pdx.s8k.io)]:::ext
    CUDA_BASE[(nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04 upstream base image)]:::ext

    FD_CLI --> FD_PLUG
    FD_PLUG --> FD_REC
    FD_INT_ALPA -->|registers runner_name slug| FD_PLUG
    FD_INT_LING -->|registers runner_name slug| FD_PLUG
    FD_INT_OTHERS -->|registers runner_name slugs| FD_PLUG
    FD_REC --> FD_INFRA
    FD_REC --> FD_CORE
    FD_INFRA --> FD_CORE
    FD_INT_ALPA -.weights.-> HF
    FD_INT_LING -.weights.-> HF
    FD_REC -.weights.-> HF
    FD_REC -.optional internal storage.-> S3
    FD_CORE -.operator-built container · docker/Dockerfile.-> CUDA_BASE
```

**Key facts:**
- Single console script `flashdreams-run` declared at `flashdreams/pyproject.toml`; entrypoint at `flashdreams/flashdreams/scripts/cli.py`.
- Recipes registered via Python entry-points; integrations register additional slugs via the same registry (`flashdreams/flashdreams/plugins/registry.py`).
- Integration adapters live under `integrations/` as separately-installable workspace packages.
- FlashDreams publishes **no canonical container image**: operators build locally from `docker/Dockerfile` against `nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04` (commit `ab74b58`); CI runs against the upstream CUDA image directly with inline `apt-get` bootstrap.

### 2. flashdreams package / class structure

```mermaid
classDiagram
    direction LR

    class FlashdreamsRunCLI {
        +entrypoint()
        -registry: RunnerRegistry
        -dispatch(slug)
    }

    class RunnerRegistry {
        +register(slug, RunnerCls)
        +discover_plugins()
        +get(slug) Runner
    }

    class Runner {
        <<abstract>>
        +local_rank: int
        +world_size: int
        +is_rank_zero: bool
        +run()
        #init_distributed()
        #pin_device()
    }

    class Pipeline {
        <<abstract>>
        +diffusion_model
        +encoder
        +decoder
        +generate(prompt, conditioning) Frames
    }

    class DiTTransformer {
        <<abstract>>
        +len_t
        +chunk_size
        +context_parallel_size
        +forward(x, t, ctx) x_out
    }

    class VAEEncoderDecoder {
        <<abstract>>
        +encode(rgb) latents
        +decode(latents) rgb
    }

    class CheckpointLoader {
        +load(uri) state_dict
        -resolve_uri(uri)
        -verify_integrity()?
    }

    class DistributedManager {
        +init(world_size)
        +local_rank()
    }

    FlashdreamsRunCLI --> RunnerRegistry
    RunnerRegistry --> Runner
    Runner --> Pipeline
    Pipeline --> DiTTransformer
    Pipeline --> VAEEncoderDecoder
    Pipeline --> CheckpointLoader
    Runner --> DistributedManager
```

> **Path cites** — `flashdreams/flashdreams/{recipes,core,infra,plugins,scripts}/`. The `Runner` abstract class lives in `flashdreams/flashdreams/infra/runner.py`; `Pipeline` in `flashdreams/flashdreams/infra/pipeline/`; `CheckpointLoader` in `flashdreams/flashdreams/core/checkpoint/`.

### 3. Integration adapters

Each integration is a workspace package that *wraps* a flashdreams recipe with an over-the-network or pluggable serving surface.

```mermaid
flowchart LR
    classDef adapter fill:#efe,stroke:#272
    classDef recipe fill:#eef,stroke:#447
    classDef ext fill:#ffe,stroke:#a82

    subgraph ALPA["integrations/alpadreams"]
        ALPA_GRPC["alpadreams/grpc/server.py · session_recorder · recording_io · profiling_server"]:::adapter
        ALPA_COND["alpadreams/conditioning · renderer"]:::adapter
        ALPA_PROTO["grpc/protos/*.proto · compile_protos.sh"]:::adapter
        LUDUS["ludus-renderer (HD-map rasterizer)"]:::adapter
    end

    subgraph LING["integrations/lingbot"]
        LING_SERVER["lingbot/webrtc/server.py  GET /request_session  POST /api/webrtc/offer"]:::adapter
        LING_SESSION["lingbot/webrtc/session.py"]:::adapter
        LING_MEDIA["lingbot/webrtc/media.py"]:::adapter
        LING_CTRL["lingbot/webrtc/controls.py  DataChannel actions {keydown, keyup, step}"]:::adapter
    end

    REC_ALPA["AlpadreamsRunner (registered slug)"]:::recipe
    REC_LING["LingbotWorldRunner (registered slug)"]:::recipe

    BROWSER[("Browser viewer")]:::ext
    GRPC_CLIENT[("gRPC client")]:::ext

    BROWSER <-->|"WebRTC SDP · DataChannel · video track"| LING_SERVER
    LING_SERVER --> LING_SESSION --> REC_LING
    LING_CTRL --> REC_LING

    GRPC_CLIENT <-->|"gRPC unary + streaming"| ALPA_GRPC
    ALPA_GRPC --> REC_ALPA
    ALPA_COND --> REC_ALPA
    LUDUS -->|"HD-map raster"| ALPA_COND
    ALPA_PROTO -.generate stubs.-> ALPA_GRPC
    ALPA_PROTO -.generate stubs.-> GRPC_CLIENT
```

> **Trust boundary preview:** Both adapters expose a **local-server** surface (loopback by default once FSR_FD_10 lands; documented `--host 0.0.0.0` in lingbot README today). See [Data view](#data-view).

### 4. Naming index

| Short name | Long name | Where it lives |
|---|---|---|
| **FD-RUN** | `flashdreams-run` console entry | `flashdreams/pyproject.toml` → `flashdreams/flashdreams/scripts/cli.py` |
| **FD-REG** | Runner / plugin registry | `flashdreams/flashdreams/plugins/registry.py` |
| **FD-CKPT** | CheckpointLoader | `flashdreams/flashdreams/core/checkpoint/` |
| **FD-CORE** | `flashdreams.core.distributed` (torch.dist init) | `flashdreams/flashdreams/core/distributed/` |
| **FD-INFRA** | Pipeline / Runner base + diffusion / encoder / decoder | `flashdreams/flashdreams/infra/` |
| **FD-COSMOS** | Cosmos DiT recipe | `flashdreams/flashdreams/recipes/cosmos/` |
| **FD-WAN** | Wan recipe (t2v / i2v) | `flashdreams/flashdreams/recipes/wan/` |
| **FD-TAEHV** | TAEHV recipe | `flashdreams/flashdreams/recipes/taehv/` |
| **ALPA-GRPC** | Alpadreams gRPC server | `integrations/alpadreams/alpadreams/grpc/server.py` |
| **ALPA-PROF** | Alpadreams profiling server | `integrations/alpadreams/alpadreams/grpc/profiling_server.py` |
| **LING-RTC** | Lingbot WebRTC server | `integrations/lingbot/lingbot/webrtc/server.py` |
| **LUDUS** | Ludus HD-map rasterizer | `integrations/alpadreams/ludus-renderer/` |

This naming is used consistently in the dynamic, data, and deployment views.

---

## Dynamic view

### 1. Offline generation — `flashdreams-run <slug>` (CLI cold-path)

```mermaid
sequenceDiagram
    autonumber
    actor Operator
    participant CLI as FD-RUN (flashdreams-run)
    participant REG as FD-REG (RunnerRegistry)
    participant RUN as Runner (recipe-specific)
    participant CKPT as FD-CKPT
    participant HF as HF / S3 store
    participant PIPE as Pipeline (DiT + enc + dec)

    Operator->>CLI: flashdreams-run wan21-t2v-1.3b-480p --prompt "..."
    CLI->>REG: discover_plugins() + get("wan21-t2v-1.3b-480p")
    REG-->>CLI: RunnerCls
    CLI->>RUN: instantiate(config)
    RUN->>RUN: init_distributed() · pin_device()
    RUN->>CKPT: load(model_uri)
    CKPT->>HF: GET checkpoint (token-auth)
    HF-->>CKPT: state_dict (.pt / .safetensors)
    Note over CKPT: verify_integrity()?  ← FSR_FD_04 (gap today)
    CKPT-->>RUN: weights pinned to cuda
    RUN->>PIPE: build(transformer, encoder, decoder)
    RUN->>PIPE: generate(prompt, conditioning)
    PIPE-->>RUN: frames
    RUN-->>Operator: video file (rank-zero I/O)
```

### 2. lingbot WebRTC interactive session

```mermaid
sequenceDiagram
    autonumber
    actor Operator as Operator (browser)
    participant LING_RTC as LING-RTC (aiohttp + aiortc)
    participant LING_SESS as LingbotSession
    participant LR as LingbotWorldRunner
    participant Gate as Input content gate  (FSR_FD_06)

    Operator->>LING_RTC: GET /request_session
    LING_RTC-->>Operator: viewer page (HTML/CSS/JS)
    Operator->>LING_RTC: POST /api/webrtc/offer (SDP offer)
    LING_RTC->>LING_SESS: new session (only 1 active)
    opt initial RGB conditioning
        LING_SESS->>Gate: classify(initial_rgb)
        Gate-->>LING_SESS: ok | REJECT (FSR_FD_06)
        Note right of Gate: on REJECT: tear down session, no inference
    end
    LING_SESS-->>LING_RTC: SDP answer
    LING_RTC-->>Operator: SDP answer
    Note over Operator,LING_SESS: ICE handshake (STUN/TURN if configured)

    loop interactive
        Operator->>LING_SESS: DataChannel action ({keydown|keyup|step}, key ≤256 B)
        Note right of LING_SESS: schema validate (FSR_FD_13); drop on miss
        LING_SESS->>LR: enqueue(action)
        LR->>LR: AR inference chunk
        LR-->>LING_SESS: chunk frames
        LING_SESS-->>Operator: video track (RTP)
        LING_SESS-->>Operator: DataChannel "chunk_done"
    end
```

> Runtime + model preload happens before request handling (`integrations/lingbot/lingbot/webrtc/server.py`). Single active WebRTC session per server process.

### 3. alpadreams gRPC inference + optional session recording

```mermaid
sequenceDiagram
    autonumber
    participant Client as gRPC Client (replay_client / external)
    participant Server as ALPA-GRPC (grpc.aio.Server)
    participant Auth as Token interceptor  (FSR_FD_12)
    participant Gate as Input content gate  (FSR_FD_06)
    participant PIPE as AlpadreamsPipeline
    participant Rec as session_recorder
    participant Disk as recording_io (.pt / .json)

    Client->>Server: InitializeSession(request)
    Server->>Auth: validate Authorization: Bearer …
    Auth-->>Server: ok | REJECT (FSR_FD_12)
    Server->>Gate: classify(first frame / scene)
    Gate-->>Server: ok | REJECT (FSR_FD_06)
    Server-->>Client: session_id (high-entropy)

    loop streamed
        Client->>Server: Step(action, hdmap_chunk)
        Server->>PIPE: generate(chunk)
        PIPE-->>Server: frames
        opt --record enabled
            Server->>Rec: record(action, frames)
            Rec->>Disk: append (mode 0600, under HOME)
        end
        Server-->>Client: frames (stream)
    end

    Client->>Server: CloseSession(session_id)
    Server->>PIPE: clear_and_zero session state  (FSR_FD_18)
    Server->>Rec: flush()
    Server-->>Client: ok
```

### 4. Checkpoint resolution / fallback

```mermaid
stateDiagram-v2
    state "Resolve URI" as ResolveURI
    state "Check FLASHDREAMS_INTERNAL_STORAGE env" as CheckEnvFlag
    state "S3 path" as S3Path
    state "Reject (non internal-storage user)" as Reject
    state "HF route" as HFRoute
    state "Check OMNI_DREAMS_HF_ORG against allowlist (FSR_FD_08)" as CheckOrg
    state "nvidia-omni-dreams-lha (internal mirror)" as LHA
    state "nvidia (default)" as DefaultOrg
    state "Fetch" as Fetch
    state "Verify integrity (FSR_FD_04 today gap)" as VerifyIntegrity
    state "Load state_dict" as LoadStateDict

    [*] --> ResolveURI
    ResolveURI --> CheckEnvFlag : URI starts with s3
    CheckEnvFlag --> S3Path : env set
    CheckEnvFlag --> Reject : env unset
    ResolveURI --> HFRoute : URI starts with hf
    HFRoute --> CheckOrg : env or CLI flag set
    CheckOrg --> LHA : hf-org equals nvidia-omni-dreams-lha
    CheckOrg --> DefaultOrg : unset / default
    LHA --> Fetch
    DefaultOrg --> Fetch
    S3Path --> Fetch
    Fetch --> VerifyIntegrity
    VerifyIntegrity --> LoadStateDict : ok
    VerifyIntegrity --> Reject : mismatch
    LoadStateDict --> [*]
    Reject --> [*]
```

> **Threat hook for TAVA**: today the `Fetch → VerifyIntegrity` arrow is **dashed in code** — there is no in-process signature/SHA check beyond HTTPS-level transport auth and HF/S3 ACLs. Promoting this to a code-level check is **FSR_FD_04** (see [`fsr_table.md`](fsr_table.md)).

### 5. Distributed init (cross-cutting)

```mermaid
stateDiagram-v2
    state "torchrun spawn" as TorchrunSpawn
    state "Runner __init__" as RunnerInit
    state "Init process group  (backend nccl)" as InitProcessGroup
    state "Pin device cuda LOCAL_RANK" as PinDevice
    state "Derive CP size from WORLD group" as DeriveCP
    state "Run recipe" as RunRecipe
    state "Rank-zero I/O (mp4, stats.json, logs)" as RankZeroIO

    [*] --> TorchrunSpawn
    TorchrunSpawn --> RunnerInit
    RunnerInit --> InitProcessGroup
    InitProcessGroup --> PinDevice
    PinDevice --> DeriveCP
    DeriveCP --> RunRecipe
    RunRecipe --> RankZeroIO : is_rank_zero gate
    RankZeroIO --> [*]
```

---

## Data view

DFD legend: **Process** = rounded box; **Data store** = cylinder; **External entity** = square; **Dataflow** = labelled arrow; **Trust boundary** = dotted region; crossings are where STRIDE threats land.

### 1. Level-0 — FlashDreams as one TOE

```mermaid
flowchart LR
    classDef proc fill:#dfd,stroke:#272,rx:10,ry:10
    classDef ds fill:#eef,stroke:#447
    classDef ext fill:#ffe,stroke:#a82
    classDef filt fill:#fcd,stroke:#a44,stroke-dasharray:4 2

    Op[Operator]:::ext
    HF[(HuggingFace)]:::ds
    S3[(S3 s3 flashdreams)]:::ds
    CUDA[(nvidia/cuda upstream base image · OE-7)]:::ds

    Gate[Input content gate  Cosmos-Guardrail pre-Guard  FSR_FD_06]:::filt
    Infer[Inference / world model  FD-COSMOS · FD-WAN · FD-TAEHV · adapters]:::proc
    Anon[Output anonymizer  Cosmos-Guardrail post-Guard  FSR_FD_07]:::filt
    Serve[Serving  LING-RTC · ALPA-GRPC · CLI]:::proc

    Logs[(Local logs / metrics)]:::ds
    Ckpt[(Local ckpt cache)]:::ds

    subgraph TB1[Trust boundary — operator workstation]
        Gate
        Infer
        Anon
        Serve
        Logs
        Ckpt
    end

    Op -->|prompt · WebRTC SDP · gRPC requests · keystrokes| Serve
    Op -->|--synthetic-initial-rgb image · scene path| Gate
    Gate -->|cleared init frame · hdmap| Infer
    Gate -.reject.-> Op
    HF -->|weights · scenes| Ckpt
    S3 -->|weights| Ckpt
    CUDA -.operator-built image · docker/Dockerfile.-> Infer
    Ckpt --> Infer
    Infer --> Anon
    Anon --> Serve
    Serve -->|video frames · status| Op
    Serve --> Logs
    Infer --> Logs
```

> **Architecturally** the TOE is a self-contained workstation process tree; there is **no NVIDIA-side ingress / egress / storage** of operator-generated data (architectural invariant restated from MVSB-32946 ticket comment).

#### Filter staging — input vs. output asymmetry

| Stage | When it runs | Cost | Default |
|---|---|---|---|
| Input content gate (FSR_FD_06) — **Cosmos-Guardrail pre-Guard, image side** | One-shot at session init / scene load | One model invocation, ≤200 ms on a single GPU; reject → session never starts | **on** when Cosmos-Guardrail weights are present |
| Output anonymizer (FSR_FD_07) — **Cosmos-Guardrail post-Guard (RetinaFace + plate + blur)** | Per chunk post-VAE — every ~900 ms steady-state | Few-step network designed for this; ≤50 ms / chunk budget on a co-resident GPU slot | **off** by default; opt-in via `--anonymize` |

### 2. Level-1 — lingbot WebRTC server

```mermaid
flowchart LR
    classDef proc fill:#dfd,stroke:#272,rx:10,ry:10
    classDef ext fill:#ffe,stroke:#a82
    classDef ds fill:#eef,stroke:#447
    classDef filt fill:#fcd,stroke:#a44,stroke-dasharray:4 2

    Browser["Operator browser"]:::ext
    Static[("HTML CSS JS viewer assets on disk")]:::ds

    HTTP["HTTP routes: GET /request_session  POST /api/webrtc/offer"]:::proc
    AuthHTTP["Token check (FSR_FD_12; required if non-loopback bind)"]:::filt
    Session["Single active WebRTC session (aiortc PeerConnection)"]:::proc
    DataCh["DataChannel handler: keydown · keyup · step  (FSR_FD_13 schema)"]:::proc
    Gate["Input content gate  (FSR_FD_06)"]:::filt
    Lingbot["LingbotPipeline AR inference"]:::proc
    Anon["Output anonymizer  (FSR_FD_07)"]:::filt
    Track["Video track / RTP encoder"]:::proc

    subgraph TB_NET["Trust boundary — LAN / public internet if host=0.0.0.0"]
        Browser
    end
    subgraph TB_SERVER["Trust boundary — flashdreams server process"]
        HTTP
        AuthHTTP
        Session
        DataCh
        Gate
        Lingbot
        Anon
        Track
        Static
    end

    Browser -->|"TLS (FSR_FD_01) · SDP offer"| HTTP
    HTTP --> AuthHTTP --> Session
    HTTP --> Static
    Static -->|"viewer page"| Browser
    Session <-->|"ICE / DTLS / SRTP"| Browser
    Browser -->|"DataChannel action"| Session
    Session --> DataCh
    DataCh -->|"action ∈ {keydown,keyup,step}"| Lingbot
    Lingbot -->|"frames"| Anon
    Anon --> Track
    Track -->|"video RTP"| Browser
    Session -->|"chunk_done DataChannel"| Browser

    Browser -.optional init frame.-> Gate
    Gate --> Lingbot
    Gate -.reject.-> Session
```

> **Risk hot-spots:**
> - **Single active session**: simplifies STRIDE-D analysis but is a DoS pinhole (T-DOS-1).
> - **`--host 0.0.0.0`**: documented default in the README; binds publicly on any non-loopback interface (T-NET-1).
> - **DataChannel actions are JSON**: validate types and keys server-side (FSR_FD_13).

### 3. Level-1 — alpadreams gRPC server

```mermaid
flowchart LR
    classDef proc fill:#dfd,stroke:#272,rx:10,ry:10
    classDef ext fill:#ffe,stroke:#a82
    classDef ds fill:#eef,stroke:#447
    classDef filt fill:#fcd,stroke:#a44,stroke-dasharray:4 2

    Client[gRPC client]:::ext

    Server["grpc.aio.Server  protos: alpadreams/grpc/protos/*.proto"]:::proc
    AuthG["Token interceptor (FSR_FD_12)"]:::filt
    Cond["ConditioningWrapper · renderer · hdmap"]:::proc
    Gate["Input content gate (FSR_FD_06)"]:::filt
    Alpa["AlpadreamsPipeline"]:::proc
    Rec["session_recorder"]:::proc
    Recordings[("recording_io .pt / .json  (FSR_FD_21: 0600 + opt-in)")]:::ds
    Prof["profiling_server  (FSR_FD_11: loopback default)"]:::proc
    Anon["Output anonymizer (FSR_FD_07)"]:::filt
    Stream["server-streaming response"]:::proc

    LudusRenderer["Ludus HD-map renderer"]:::proc

    subgraph TB_CLIENT["Trust boundary — gRPC client side"]
        Client
    end
    subgraph TB_SERVER["Trust boundary — alpadreams server process"]
        Server
        AuthG
        Cond
        Gate
        LudusRenderer
        Alpa
        Rec
        Recordings
        Prof
        Anon
        Stream
    end

    Client -->|"InitializeSession · Step · Close (TLS: FSR_FD_01)"| Server
    Server --> AuthG --> Cond
    Cond --> LudusRenderer
    LudusRenderer --> Cond
    Cond --> Gate
    Gate --> Alpa
    Gate -.reject.-> Server
    Alpa --> Anon
    Anon --> Stream
    Stream -->|"frame stream"| Client
    Alpa --> Rec
    Rec --> Recordings
    Server --> Prof
```

### 4. Data classification (per dataflow)

| Dataflow | Direction | Classification | Notes |
|---|---|---|---|
| Operator prompt / keystroke | Op → Pipeline | **Operator-confidential** (synthetic / non-PII by default) | Don't log raw inputs above DEBUG (FSR_FD_02, FSR_FD_03) |
| Initial RGB frame | Op → Pipeline | **Possibly PII** (faces / plates if real photo) | FSR_FD_06 input gate |
| HD-map raster | Op → Pipeline | Public (synthetic) | |
| Generated video frames | Pipeline → Op | **Possibly PII** (model may emit identifiable faces / plates) | FSR_FD_07 output anonymizer; FSR_FD_19 per-session routing |
| Checkpoint weights | HF / S3 → Process | **NVIDIA proprietary** until license-class flips at GA | Integrity check FSR_FD_04 |
| Container image | upstream `nvidia/cuda` → operator build → Host | **Operator-owned** (FlashDreams ships no canonical image; OE-7) | Trust delegates upward to the `nvidia/cuda` base image; operator-side build is the new tamper surface |
| Logs / metrics | Process → Disk | Operator-confidential | Scrubbing on export only |
| Session recording (.pt) | Process → Disk | **Operator-confidential** | Off by default (FSR_FD_21) |
| S3 credentials file | Disk → Process | **Secret** | `credentials/s3_checkpoint.secret` — chmod 600, never committed |

---

## Deployment view

### 1. Single-node multi-GPU (lingbot WebRTC / alpadreams gRPC)

```mermaid
flowchart TB
    classDef cont fill:#eef,stroke:#447
    classDef gpu fill:#ffe,stroke:#a82

    Host["Workstation / DGX node"]
    Browser["Browser viewer / gRPC client"]

    subgraph Cont["Operator-built container (docker/Dockerfile · nvidia/cuda:13.2.1 base · OE-7)"]
        Launcher["torchrun --nproc_per_node=N"]:::cont
        R0["Rank 0 · cuda:0  (terminates serving listener)"]:::cont
        R1["Rank 1 · cuda:1"]:::cont
        R2["Rank 2 · cuda:2"]:::cont
        R3["Rank 3 · cuda:3"]:::cont
        Server["LING-RTC :8089  OR  ALPA-GRPC :PORT"]:::cont
    end

    subgraph GPUs["N × GPU"]
        G0["GPU 0"]:::gpu
        G1["GPU 1"]:::gpu
        G2["GPU 2"]:::gpu
        G3["GPU 3"]:::gpu
    end

    Host --> Cont
    Launcher --> R0
    Launcher --> R1
    Launcher --> R2
    Launcher --> R3
    R0 --> G0
    R1 --> G1
    R2 --> G2
    R3 --> G3
    R0 ---|"NCCL WORLD group"| R1
    R1 ---|"NCCL"| R2
    R2 ---|"NCCL"| R3
    R0 --> Server
    Server <-->|"WebRTC OR gRPC (TLS: FSR_FD_01)"| Browser
```

> Only **rank 0** terminates the WebRTC / gRPC connection (rank-zero I/O invariant). Other ranks do compute; they have no exposed listener.

### 2. Container layering

```mermaid
flowchart TB
    classDef img fill:#eef,stroke:#447
    classDef layer fill:#fff,stroke:#999

    Base["nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04 (upstream)"]:::img
    Build["docker/Dockerfile  apt: py3.12 + ffmpeg + libnccl-dev + uv + AWS CLI v2"]:::layer
    Local["operator-built image  (operator's registry / local daemon)"]:::img

    Base --> Build --> Local
```

> FlashDreams publishes **no canonical container image** (commit `ab74b58`). The Dockerfile lives at `docker/Dockerfile`; operators run `docker build` (or `docker/build_with_docker.sh <registry>/<tag>`) to produce an image under their own registry / access controls. Sigstore signing (formerly FSR_FD_05) is therefore an **operator guidance** item rather than a FlashDreams-side enforcement (see [`fsr_table.md`](fsr_table.md) FSR_FD_05).

### 3. Network exposure summary (for the TAVA)

| Component | Default bind (today) | Default port | Auth | Encryption | Recommended target |
|---|---|---|---|---|---|
| LING-RTC (HTTP signaling) | `0.0.0.0` per README | `8089` | none | none | **127.0.0.1** (FSR_FD_10); TLS (FSR_FD_01) + token (FSR_FD_12) when public |
| LING-RTC (WebRTC media) | dynamic UDP | ephemeral | DTLS-SRTP (aiortc) | yes | OK as-is |
| ALPA-GRPC | configurable | configurable | none in dev | none in dev | **127.0.0.1** (FSR_FD_10); mTLS + token interceptor (FSR_FD_12) |
| ALPA-PROF (profiling) | configurable | configurable | none in dev | none in dev | **127.0.0.1** (FSR_FD_11); separate opt-in from main server |
| torch.distributed | depends on launcher | depends | none | none | trust cluster network only; no public exposure |
| HF / S3 / upstream `nvidia/cuda` registry | outbound | 443 | token / sigv4 | TLS | OK |

> The `0.0.0.0` default binds are an audit hot-spot — see threat T-NET-1 in [`tava.md`](tava.md) § 2.3.

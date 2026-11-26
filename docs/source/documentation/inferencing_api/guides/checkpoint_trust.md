---
title: 'DCP checkpoint trust'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DCP checkpoint trust

PyTorch distributed checkpoints (DCP) contain executable pickle metadata.
FlashDreams requires an independently approved manifest pin before reading any
local or S3 DCP metadata. A trusted source name, TLS connection, S3 credentials,
or checksum downloaded alongside an unknown checkpoint does not establish trust.
Unknown sources and verification failures stop loading; there is no unsafe opt-out.

Single-file and sharded safetensors workflows remain available, including the
default Cosmos3 Nano Sim Bimanual Hugging Face export. No account or network
service is needed to approve and reuse a local custom DCP. This control applies
to DCP; it does not authenticate every other model format or asset.

## Approve a checkpoint once

Where the checkpoint was produced (the controlled training/export environment,
while its output is stable), run one command on the completed DCP component
directory, which contains `.metadata` (or rank-specific `__<rank>.metadata`)
and all referenced shards:

```bash
python -m flashdreams.core.io.dcp_trust pin /controlled/export/model
```

The command hashes every file, writes the manifest to a private `dcp-pins/`
folder beside the trust store, records the pin, and prints the manifest SHA-256.
Loaders on that machine then work with no further setup. To approve the location
you will upload to, add `--source s3://approved-bucket/run/iter_100/model`.

**Do not pin an unknown download merely because verification failed.** Approval
trusts the metadata as executable code running with the loader's authority.

To approve the same checkpoint on another machine, copy the manifest file by any
means, deliver the printed SHA-256 through an independently trusted release or
review channel, and register both:

```bash
python -m flashdreams.core.io.dcp_trust pin --manifest model-manifest.json \
  --sha256 <approved 64-hex SHA-256> s3://approved-bucket/run/iter_100/model
```

The command refuses a manifest that does not match the digest. Publishers of
supported DCP releases provide manifests and digests the same way; never derive
a pin from the storage copy you are trying to verify.

## Trust store

`pin` writes `~/.config/flashdreams/dcp-trust.json` (under `$XDG_CONFIG_HOME`
when set), which loaders use by default. To share an operator-managed store, set
`FLASHDREAMS_DCP_TRUST_STORE` to its path; `pin` and loaders both follow it.
The store maps each source ID to `{"manifest_path": "<absolute path>",
"sha256": "<64 hex digits>"}`; a manifest is exactly `{"version": 1, "files":
{...}}`, mapping each relative filename to `{"sha256": ..., "size": ...}`.

Treat the store like application configuration: replacing it grants approval
authority, and the owners of the store and the directories above it hold that
authority. On POSIX, loading resolves symlinks and rejects a group- or
world-writable store file, and a store below any directory that other accounts
can write unless that directory has the sticky bit; `pin` creates the folder
`0700` and files `0600`. A manifest may be copied or replaced without changing
that authority: its bytes still have to match the approved digest. Manifest
paths must stay inside the component directory (absolute paths, `..`, and
backslashes are rejected), JSON records are limited to 4 MiB, duplicate keys are
rejected, and extra files at the source are ignored and cannot be read by the
snapshot reader.

Local source IDs use absolute, expanded paths, without resolving symlinks. S3
IDs retain the full URI with trailing slashes removed. Add an explicit entry for
each approved location; no bucket, publisher, basename, or alias wildcard grants
trust. Pin an actual component directory, not its parent training run. The
Cosmos3 Nano Sim Bimanual fetcher selects a pinned root or pinned `root/model`
from this store without probing metadata. If a root is pinned, any failure stops
there.

## Supply a pin in Python

```python
from flashdreams.core.checkpoint.load import get_storage_reader, load_checkpoint
from flashdreams.core.io.dcp_trust import DCPTrust

trust = DCPTrust("/secure/model-pins/custom.json", approved_manifest_sha256)
load_checkpoint("/models/custom/model", model=model, dcp_trust=trust)

with get_storage_reader("/models/custom/model", dcp_trust=trust) as reader:
    metadata = reader.read_metadata()
```

`load_distributed_checkpoint` and direct `S3StorageReader` construction also
accept `dcp_trust`. Explicit pins take precedence over the trust store.
Retain a reader until all tensor reads finish, then close it or use a context
manager. A verified reader cannot be reset to another source.

## Migration and storage costs

Previous URL-only merged `.pt` DCP caches are ignored, and no new merged DCP
cache is written. The Cosmos3 Nano Sim Bimanual converter verifies the source on
every invocation and rebuilds converted bytes; an output index or self-reported
conversion ID does not authenticate those bytes. Existing custom output
mismatch/overwrite checks still apply. For repeated offline use, copy an
approved DCP locally and register that location with the same manifest pin, or
explicitly use a reviewed safetensors export through the normal
single-file/index workflow.

Every reader first streams all manifest-listed files to a new private temporary
directory and verifies their sizes and SHA-256 digests. Only that snapshot is
parsed and loaded. Source changes during copying fail verification; replacements
after verification cannot change the snapshot. The reader holds the snapshot
directory open and reads through it, so renaming or replacing the snapshot
path, even in a shared scratch directory, cannot change the parsed bytes. S3 bodies are streamed with a
pinned-size bound. Local manifests, stores, and checkpoint files must be regular
files: a FIFO, device, or directory in their place is rejected without blocking.
The snapshot exposes only pinned files and is deleted when the reader closes.

Allow scratch space for the complete checkpoint. Set the standard `TMPDIR`
environment variable before starting Python to choose any scratch directory
with enough space. Before copying, the reader checks that this directory has
room for every pinned file; running out of space is reported as a disk-space
error, not a trust failure. Each Cosmos conversion reads metadata and tensors
from one verified snapshot. Metadata-only probing still copies the complete
checkpoint. There is no persistent verified cache. No isolation sandbox or
untrusted-pickle converter is provided: approved metadata still executes in the
process, and the trust store, installed code, operating system, and loader user
account must remain trusted.

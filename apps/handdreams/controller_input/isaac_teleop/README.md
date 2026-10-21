# Isaac Teleop sidecar locks

These files describe the complete public-PyPI wheel dependency closure of
`isaacteleop[cloudxr,retargeters-lite]==1.4.142` for CPython 3.12 on Linux
`aarch64` and `x86_64`. The Isaac Teleop wheel requires glibc 2.35 or newer.
The operator installs these wheels in a separate environment. FlashDreams does
not distribute the wheels or their CloudXR native libraries.

The lock files were resolved on 2026-10-08 using uv 0.12.23, independently of
the workspace's dependency overrides. Each file pins all eight distributions
and permits only the wheel named in the comment above its requirement. Both
platform closures were downloaded using pip 24.0 with `--require-hashes`;
the downloaded SHA-256 values were checked against the corresponding public
PyPI release JSON. No CloudXR session was started for this verification.

To update a lock, resolve the new closure from a temporary input file containing
the exact Isaac Teleop requirement, replacing `<arch>` with `aarch64` or
`x86_64`:

```bash
uv pip compile /tmp/isaac-teleop.in --no-config \
  --default-index https://pypi.org/simple \
  --python-version 3.12 --python-platform <arch>-manylinux_2_35 \
  --only-binary :all: --generate-hashes --no-strip-extras \
  --emit-index-url --emit-build-options --output-file /tmp/isaac-teleop.txt
```

uv emits hashes for artifacts beyond the target platform. Download the resolved
closure with `pip download --require-hashes --only-binary=:all:` for CPython
3.12 and the target architecture, then retain only the downloaded wheel's hash
for each pinned distribution. Cross-platform downloads must enumerate supported
manylinux platform tags from `manylinux_2_35_<arch>` down through
`manylinux_2_17_<arch>` and `manylinux2014_<arch>`, and accept `cp312`, `abi3`,
and `none` ABIs. Record each selected wheel filename and verify its SHA-256
against `https://pypi.org/pypi/<distribution>/<version>/json`. Re-run the final
hash-locked download to verify closure completeness. Review dependency and
license changes, regenerate the release inventory, and run setup/doctor checks
on both supported hosts before promoting updated locks.

The lock identifies artifacts; it does not grant license acceptance. Setup and
CloudXR startup still require explicit operator acceptance of the applicable
[CloudXR EULA](https://github.com/NVIDIA/IsaacCapture/blob/main/deps/cloudxr/CLOUDXR_LICENSE).

## Sidecar SBOM

From the repository root, generate a draft SPDX 2.3 inventory without installing
packages or contacting a registry:

```bash
python3 apps/handdreams/controller_input/isaac_teleop/build_sbom.py \
  --output /tmp/isaac-teleop.spdx.json
```

Use `--repository /path/to/checkout` to select another checkout. The tool reads both
platform manifests at their actual internal or public paths, records manifest
and wheel SHA-256 values, and inventories every pinned distribution even when
Isaac Teleop is absent from the workspace `uv.lock`. Missing platform manifests
or unpinned/unhashed entries fail generation.

Public-path packages have scope `customer-installed-runtime`; internal-path
packages have scope `promotion-customer-installed-runtime-review`. Every wheel
is represented as a dependency of the logical sidecar environment, never as a
binary contained in the repository. Individual transitive edges are not inferred
from the flat requirements files. This focused inventory must be reconciled with
the release SBOM; it is not a complete FlashDreams SBOM or license approval.

The [Isaac Teleop 1.4.142 public wheel metadata](https://pypi.org/project/isaacteleop/1.4.142/)
declares Apache-2.0. The complete wheel retains `licenseConcluded=NOASSERTION`
because it contains CloudXR native components. Other packages remain
`NOASSERTION` pending their applicable license evidence and review.

Suggested OSRB description:

> FlashDreams does not copy Isaac Teleop or CloudXR binaries into its GitHub
> release. Users install the public PyPI distribution
> `isaacteleop[cloudxr,retargeters-lite]==1.4.142` into a dedicated virtual
> environment, and FlashDreams invokes it as an isolated subprocess. The operator
> must explicitly accept the NVIDIA CloudXR EULA before CloudXR startup. Because
> the wheel contains CloudXR native components, it remains represented in the
> SBOM with `licenseConcluded=NOASSERTION` pending confirmation of the applicable
> Isaac Teleop/CloudXR review.

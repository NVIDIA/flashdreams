# `docker/` -- flashdreams container image

This folder contains the Dockerfile and build tooling for a flashdreams-ready
container image. Build it locally or push to your own registry.

The image is based on `nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04` and adds
Python 3.12, build tools (gcc, g++, ninja), ffmpeg, libnccl-dev, uv, and the
AWS CLI v2 -- everything needed to compile and run flashdreams.

---

## Contents

| File | Purpose |
|---|---|
| `Dockerfile` | Image recipe. Based on `nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04`. |
| `build_with_docker.sh` | Build + push a multi-arch (`linux/arm64` + `linux/amd64`) image to a registry you specify. |
| `docker_farm_setup.sh` | One-time Buildx "farm" setup so arm64 builds run natively on `dgx-spark` instead of under QEMU emulation. |

---

## Building locally

For a quick single-arch local image (no registry push):

```bash
docker build -t flashdreams:local -f docker/Dockerfile .
```

Then use it with docker or Slurm:

```bash
docker run --rm --gpus all -it flashdreams:local bash
```

---

## Building + pushing (multi-arch)

Use `build_with_docker.sh` to produce a multi-arch manifest (linux/arm64 +
linux/amd64) and push it to your registry:

```bash
# Log in to your target registry first
docker login <your-registry>

# Build and push -- at least one fully-qualified tag is required
bash docker/build_with_docker.sh <your-registry>/flashdreams:<your-tag>

# Multiple tags are supported
bash docker/build_with_docker.sh reg1/flashdreams:v1.0 reg2/flashdreams:latest
```

---

## Multi-arch build farm (optional)

Multi-arch builds are much faster when each arch runs on a native node.
`docker_farm_setup.sh` wires a local `docker-container` driver plus an SSH
endpoint to `dgx-spark` into a single Buildx builder named `farm`:

```bash
# Prereqs:
#   - `ssh dgx-spark true` succeeds from your workstation
#   - `docker buildx version` works

bash docker/docker_farm_setup.sh
```

Verify:

```bash
docker buildx ls
docker buildx inspect farm
```

You should see two nodes with `linux/amd64` and `linux/arm64` respectively.

Skip this step if you are fine with QEMU emulation for the non-native arch;
`build_with_docker.sh` will still work, just slowly.

### About `dgx-spark`

`dgx-spark` is the short `~/.ssh/config` alias for a shared NVIDIA DGX
Spark workstation that the project uses as a native **arm64 (Grace)**
build node. `docker_farm_setup.sh` attaches it to the `farm` builder via
an SSH endpoint (`ssh://$USER@dgx-spark`), so any `--platform linux/arm64`
build is scheduled there instead of crawling through QEMU on an amd64
host.

Using it requires:

- Create an account on the machine.
- An SSH key installed on it for your `$USER`.
- A `Host dgx-spark` block in `~/.ssh/config` pointing at the right
  hostname / user / identity file so `ssh dgx-spark true` logs in
  non-interactively.
- Docker installed and runnable by your user on that host.

To request access and get the onboarding steps (SSH Host block, account
provisioning), contact **qiwu@nvidia.com**.

You don't need `dgx-spark` access to build images -- dropping it just
means `build_with_docker.sh` will emulate arm64 via QEMU on your amd64
workstation, which is correct but noticeably slower.

---

## Troubleshooting

**`ERROR: failed to solve: ... network ...` during build.**
Inside NVIDIA infra you usually need `--allow network.host --network host`
(already set in `build_with_docker.sh`) so apt/PyPI traffic goes through
the host's configured proxies.

**Buildx can't find an arm64 node.**
Re-run `docker buildx inspect farm --bootstrap`. If the SSH endpoint is
unhealthy, rebuild the farm:

```bash
docker buildx use default
docker buildx rm farm
bash docker/docker_farm_setup.sh
```

**`docker buildx build ... --load` complains about multi-platform.**
`--load` imports a single image into the local Docker daemon and is
incompatible with multi-arch output. Drop one of the `--platform` values
if you need a local-only build for testing.

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
| `Dockerfile` | Image integration. Based on `nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04`. |
| `build_with_docker.sh` | Build + push a multi-arch (`linux/arm64` + `linux/amd64`) image to a registry you specify. |

---

## Building locally

For a quick single-arch local image (no registry push):

```bash
docker build -t flashdreams:local -f docker/Dockerfile .
```

Then use it with docker:

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

## Multi-arch builds

`build_with_docker.sh` expects a Buildx builder that can build both
`linux/arm64` and `linux/amd64`. A default local builder can use QEMU
emulation for the non-native architecture, or you can configure your own
remote Buildx nodes outside this repository.

---

## Troubleshooting

**`ERROR: failed to solve: ... network ...` during build.**
`build_with_docker.sh` passes `--allow network.host --network host` so apt
and PyPI traffic can use the host's configured network path.

**Buildx can't find an arm64 node.**
Run `docker buildx ls` and confirm your selected builder supports
`linux/arm64`. If it does not, configure a multi-platform builder or use
QEMU emulation for the non-native architecture.

**`docker buildx build ... --load` complains about multi-platform.**
`--load` imports a single image into the local Docker daemon and is
incompatible with multi-arch output. Drop one of the `--platform` values
if you need a local-only build for testing.

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04

# Default workdir used by test scripts.
RUN mkdir -p /workspace

# System deps: Python 3.12, ffmpeg, build tools for transformer-engine-torch
# And some common CLI utilities
RUN apt-get update -qq \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
        python3 python3-dev python3-venv \
        ffmpeg \
        gcc g++ ninja-build \
        libnccl-dev \
        bash \
        bash-completion \
        less \
        nano \
        procps \
        iproute2 \
        iputils-ping \
        curl \
        git \
        openssh-client \
        ca-certificates \
        tmux \
        screen \
        unzip \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install AWS CLI v2
RUN ARCH="$(uname -m)" \
    && case "$ARCH" in \
         x86_64)  AWS_ARCH="x86_64" ;; \
         aarch64) AWS_ARCH="aarch64" ;; \
         *) echo "Unsupported arch: $ARCH" >&2; exit 1 ;; \
       esac \
    && curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-${AWS_ARCH}.zip" -o /tmp/awscliv2.zip \
    && unzip -q /tmp/awscliv2.zip -d /tmp \
    && /tmp/aws/install \
    && rm -rf /tmp/aws /tmp/awscliv2.zip

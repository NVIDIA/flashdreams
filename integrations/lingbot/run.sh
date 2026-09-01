#!/bin/bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

source .venv/bin/activate

python run_direct.py \
  --host=0.0.0.0 \
  --port=8089 \
  --device=cuda:0 \
  --example-idx=0 \
  --warmup-chunks=1


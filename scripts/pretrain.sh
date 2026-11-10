#!/usr/bin/env bash
# Pre-train one model on 1..N GPUs.
#
#   bash scripts/pretrain.sh kestrel-250m
#   GPUS=8 bash scripts/pretrain.sh condor-900m
#   GPUS=8 bash scripts/pretrain.sh condor-900m --resume checkpoints/condor-900m/pretrain/step-0040000.pt
set -euo pipefail

MODEL="${1:?usage: pretrain.sh <model> [extra args]}"; shift
GPUS="${GPUS:-1}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."

torchrun --standalone --nproc_per_node="$GPUS" pretrain/train.py --config "configs/${MODEL}.yaml" "$@"

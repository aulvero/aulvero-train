#!/usr/bin/env bash
# Fine-tune a pre-trained Base checkpoint into an Instruct model.
#
#   bash scripts/finetune.sh kestrel-250m
#   GPUS=4 bash scripts/finetune.sh heron-500m
set -euo pipefail

MODEL="${1:?usage: finetune.sh <model> [extra args]}"; shift
GPUS="${GPUS:-1}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."

torchrun --standalone --nproc_per_node="$GPUS" finetune/sft.py --config "configs/${MODEL}.yaml" "$@"

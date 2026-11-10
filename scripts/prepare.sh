#!/usr/bin/env bash
# Download a data mix, train the shared tokenizer (once), and pack token files.
#
#   bash scripts/prepare.sh              # main mix  -> data/packed/pretrain/
#   bash scripts/prepare.sh pilot        # pilot mix -> data/packed/pilot/
#   bash scripts/prepare.sh code         # code mix  -> data/packed/code/
set -euo pipefail

MIX="${1:-pretrain}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ ! -f "data/mixes/${MIX}.yaml" ]]; then
  echo "Unknown mix '${MIX}'. Available: $(ls data/mixes | sed 's/\.yaml//' | tr '\n' ' ')" >&2
  exit 1
fi

python scripts/download_data.py --config "data/mixes/${MIX}.yaml" --output-dir "data/raw/${MIX}"

# One tokenizer for every model. Trained on the first mix you prepare
# (normally "pretrain"); delete artifacts/tokenizer to retrain it.
if [[ ! -f artifacts/tokenizer/tokenizer.json ]]; then
  python scripts/train_tokenizer.py --input "data/raw/${MIX}/train.jsonl" --output artifacts/tokenizer
fi

python scripts/pack_data.py \
  --tokenizer artifacts/tokenizer/tokenizer.json \
  --train "data/raw/${MIX}/train.jsonl" \
  --val "data/raw/${MIX}/val.jsonl" \
  --output-dir "data/packed/${MIX}"

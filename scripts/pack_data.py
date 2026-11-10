#!/usr/bin/env python
"""Tokenise train/val JSONL into flat uint32 token files for pre-training."""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aulvero.data import iter_jsonl


def pack(pattern, output, tok):
    paths = glob.glob(pattern)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    bos = tok.token_to_id("<bos>")
    eos = tok.token_to_id("<eos>")
    total = 0
    with open(output, "wb") as f:
        for text in tqdm(iter_jsonl(paths), desc=output):
            ids = [bos] + tok.encode(text, add_special_tokens=False).ids + [eos]
            np.asarray(ids, dtype=np.uint32).tofile(f)
            total += len(ids)
    print(output, total, "tokens")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="artifacts/tokenizer/tokenizer.json")
    ap.add_argument("--train", default="data/raw/train.jsonl")
    ap.add_argument("--val", default="data/raw/val.jsonl")
    ap.add_argument("--output-dir", default="data/packed")
    args = ap.parse_args()
    tok = Tokenizer.from_file(args.tokenizer)
    pack(args.train, f"{args.output_dir}/train.bin", tok)
    pack(args.val, f"{args.output_dir}/val.bin", tok)


if __name__ == "__main__":
    main()

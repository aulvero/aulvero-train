#!/usr/bin/env python
"""Stream a data mix (data/mixes/*.yaml) from Hugging Face into train/val JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import yaml
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aulvero.data import acceptable, acceptable_code, clean_code, clean_text


def load_streaming_source(src: dict, seed: int, index: int):
    dataset_kwargs = dict(src.get("dataset_kwargs") or {})
    if "path" in dataset_kwargs or "name" in dataset_kwargs:
        raise ValueError(f"{src['name']}: put dataset path in name/config, not dataset_kwargs")
    kwargs = {"path": src["name"], "split": src.get("split", "train"), "streaming": True}
    if src.get("config"):
        kwargs["name"] = src["config"]
    kwargs.update(dataset_kwargs)
    ds = load_dataset(**kwargs)
    if src.get("shuffle_buffer"):
        ds = ds.shuffle(buffer_size=int(src["shuffle_buffer"]), seed=seed + index)
    return ds


def extract_text(row: dict, src: dict) -> str:
    field = src.get("text_field", "text")
    raw = row.get(field, "")
    text = clean_code(str(raw)) if src.get("kind") == "code" else clean_text(str(raw))
    prefix = src.get("prefix")
    suffix = src.get("suffix")
    if prefix and text:
        text = f"{prefix}\n{text}"
    if suffix and text:
        text = f"{text}\n{suffix}"
    return text


def source_accepts(text: str, src: dict) -> bool:
    if src.get("kind") == "code":
        return acceptable_code(
            text,
            min_chars=src.get("min_chars", 80),
            max_chars=src.get("max_chars", 300_000),
        )
    return acceptable(
        text,
        min_chars=src.get("min_chars", 200),
        max_chars=src.get("max_chars", 200_000),
    )


def build_record(text: str, row: dict, src: dict) -> str:
    metadata_fields = src.get("metadata_fields", [])
    metadata = {field: row.get(field) for field in metadata_fields if field in row}
    record = {
        "text": text,
        "source": src["name"],
        "source_kind": src.get("kind", "text"),
    }
    if src.get("config"):
        record["source_config"] = src["config"]
    if metadata:
        record["metadata"] = metadata
    return json.dumps(record, ensure_ascii=False) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="data/mixes/pretrain.yaml")
    ap.add_argument("--output-dir", default="data/raw")
    ap.add_argument("--val-fraction", type=float, default=0.002)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train = open(out / "train.jsonl", "w", encoding="utf-8")
    val = open(out / "val.jsonl", "w", encoding="utf-8")
    seen = set()
    rng = random.Random(args.seed)
    counts = {"train": 0, "val": 0, "rejected": 0}
    try:
        for source_index, src in enumerate(cfg["sources"]):
            ds = load_streaming_source(src, args.seed, source_index)
            limit = src.get("max_documents")
            for i, row in enumerate(ds):
                if limit and i >= limit:
                    break
                text = extract_text(row, src)
                digest = hashlib.blake2b(text.encode(), digest_size=16).digest()
                if not source_accepts(text, src) or digest in seen:
                    counts["rejected"] += 1
                    continue
                seen.add(digest)
                record = build_record(text, row, src)
                target = val if rng.random() < args.val_fraction else train
                key = "val" if target is val else "train"
                target.write(record)
                counts[key] += 1
                if (counts["train"] + counts["val"]) % 10000 == 0:
                    print(counts)
    finally:
        train.close()
        val.close()
    print("Done:", counts)


if __name__ == "__main__":
    main()

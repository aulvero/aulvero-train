from __future__ import annotations

import json
import re
import unicodedata

import numpy as np
import torch

SPACE_RE = re.compile(r"[ \t]+")
BLANK_RE = re.compile(r"\n{3,}")
CODE_BLANK_RE = re.compile(r"\n{6,}")
TRAILING_WS_RE = re.compile(r"[ \t]+$", re.MULTILINE)


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\x00", "")
    text = SPACE_RE.sub(" ", text)
    text = BLANK_RE.sub("\n\n", text)
    return text.strip()


def clean_code(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = TRAILING_WS_RE.sub("", text)
    text = CODE_BLANK_RE.sub("\n\n\n\n\n", text)
    return text.strip("\n")


def acceptable(text: str, min_chars=200, max_chars=200_000) -> bool:
    if not (min_chars <= len(text) <= max_chars):
        return False
    printable = sum(ch.isprintable() or ch in "\n\t" for ch in text) / max(1, len(text))
    alpha = sum(ch.isalpha() for ch in text) / max(1, len(text))
    return printable > 0.98 and alpha > 0.35


def acceptable_code(text: str, min_chars=80, max_chars=300_000) -> bool:
    if not (min_chars <= len(text) <= max_chars):
        return False
    printable = sum(ch.isprintable() or ch in "\n\t" for ch in text) / max(1, len(text))
    lines = text.splitlines() or [text]
    long_lines = sum(len(line) > 2000 for line in lines) / max(1, len(lines))
    return printable > 0.95 and long_lines < 0.02


def iter_jsonl(paths):
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("text"):
                    yield row["text"]


class TokenBinDataset:
    def __init__(self, path: str, seq_len: int, seed: int = 0):
        self.data = np.memmap(path, dtype=np.uint32, mode="r")
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)
        if len(self.data) <= seq_len + 1:
            raise ValueError(f"{path} is too small for sequence length {seq_len}")

    def batch(self, batch_size: int, device: torch.device):
        starts = self.rng.integers(0, len(self.data) - self.seq_len - 1, size=batch_size)
        x = np.stack([self.data[i : i + self.seq_len] for i in starts]).astype(np.int64)
        y = np.stack([self.data[i + 1 : i + 1 + self.seq_len] for i in starts]).astype(np.int64)
        return (
            torch.from_numpy(x).to(device, non_blocking=True),
            torch.from_numpy(y).to(device, non_blocking=True),
        )

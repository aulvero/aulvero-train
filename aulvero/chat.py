"""Chat format and supervised fine-tuning data.

Conversations use the ChatML layout (also used by Qwen and SmolLM)::

    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    Bonjour !<|im_end|>
    <|im_start|>assistant
    Bonjour ! Comment puis-je vous aider ?<|im_end|>

During fine-tuning the loss is computed on assistant turns only.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from pathlib import Path

import torch

from .model import IGNORE_INDEX

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
CHAT_SPECIAL_TOKENS = [IM_START, IM_END]

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

ROLES = {"system", "user", "assistant"}


def encode_conversation(tokenizer, messages: list[dict], max_len: int) -> tuple[list[int], list[int]]:
    """Return (input_ids, labels). Labels are IGNORE_INDEX except on assistant replies.

    ``tokenizer`` is a ``tokenizers.Tokenizer``.
    """
    ids: list[int] = []
    labels: list[int] = []

    def add(text: str, learn: bool) -> None:
        piece = tokenizer.encode(text, add_special_tokens=False).ids
        ids.extend(piece)
        labels.extend(piece if learn else [IGNORE_INDEX] * len(piece))

    for message in messages:
        role, content = message["role"], message["content"]
        if role not in ROLES:
            raise ValueError(f"Unknown role: {role}")
        add(f"{IM_START}{role}\n", learn=False)
        add(f"{content}{IM_END}", learn=role == "assistant")
        add("\n", learn=False)
    return ids[:max_len], labels[:max_len]


def load_conversations(
    source: str, split: str = "train", field: str = "messages", limit: int | None = None
) -> list[list[dict]]:
    """Load conversations from a local ``.jsonl`` file or a Hugging Face dataset name."""
    rows: Iterable[dict]
    if Path(source).suffix == ".jsonl":
        with open(source, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        from datasets import load_dataset

        rows = load_dataset(source, split=split)
    out = []
    for row in rows:
        messages = row[field]
        if any(m["role"] == "assistant" for m in messages):
            out.append([{"role": m["role"], "content": m["content"]} for m in messages])
        if limit and len(out) >= limit:
            break
    return out


class SFTDataset:
    """Tokenised conversations, served as right-padded, shifted batches."""

    def __init__(self, tokenizer, conversations: list[list[dict]], seq_len: int, pad_id: int, seed: int = 0):
        self.examples = []
        for messages in conversations:
            ids, labels = encode_conversation(tokenizer, messages, seq_len + 1)
            if any(label != IGNORE_INDEX for label in labels[1:]):
                self.examples.append((ids, labels))
        if not self.examples:
            raise ValueError("No usable conversations (each needs an assistant reply within seq_len)")
        self.pad_id = pad_id
        self.rng = random.Random(seed)
        self.order: list[int] = []

    def __len__(self) -> int:
        return len(self.examples)

    def batch(self, batch_size: int, device: torch.device):
        picked = []
        for _ in range(batch_size):
            if not self.order:
                self.order = list(range(len(self.examples)))
                self.rng.shuffle(self.order)
            picked.append(self.examples[self.order.pop()])
        width = max(len(ids) for ids, _ in picked) - 1
        x = torch.full((batch_size, width), self.pad_id, dtype=torch.long)
        y = torch.full((batch_size, width), IGNORE_INDEX, dtype=torch.long)
        for row, (ids, labels) in enumerate(picked):
            n = len(ids) - 1
            x[row, :n] = torch.tensor(ids[:-1])
            y[row, :n] = torch.tensor(labels[1:])
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

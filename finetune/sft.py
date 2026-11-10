#!/usr/bin/env python
"""Supervised fine-tuning: turn a Base checkpoint into an Instruct model.

    python finetune/sft.py --config configs/kestrel-250m.yaml

Reads the ``finetune`` section of the model file. By default it starts from
``checkpoints/<name>/pretrain/final.pt`` and writes ``checkpoints/<name>/sft/final.pt``.
Data is a local ``.jsonl`` file or a Hugging Face dataset whose rows have a
``messages`` list of ``{"role", "content"}`` turns. Loss is on assistant replies only.
"""

from __future__ import annotations

import argparse
import math
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from tokenizers import Tokenizer
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aulvero.chat import SFTDataset, load_conversations  # noqa: E402
from aulvero.config import ModelConfig, load_run_config  # noqa: E402
from aulvero.model import AulveroLM  # noqa: E402
from aulvero.schedule import lr_at  # noqa: E402
from aulvero.training import (  # noqa: E402
    MetricsLog,
    autocast,
    build_optimizer,
    pick_device,
    save_checkpoint,
    seed_everything,
    setup_distributed,
    teardown_distributed,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--init", help="override finetune.init_checkpoint")
    ap.add_argument("--max-steps", type=int, help="cap the number of optimizer steps (smoke tests)")
    args = ap.parse_args()

    run = load_run_config(args.config)
    c = run["finetune"]
    d = setup_distributed()
    seed_everything(c["seed"] + d.rank)
    device = pick_device(d)

    init = args.init or c["init_checkpoint"]
    ck = torch.load(init, map_location="cpu", weights_only=False)
    model_cfg = ModelConfig.from_dict(ck["model_config"])
    model = AulveroLM(model_cfg)
    model.load_state_dict(ck["model"])
    model.to(device)

    tok = Tokenizer.from_file(c["tokenizer"])
    pad_id = tok.token_to_id("<pad>")
    for special in ("<|im_start|>", "<|im_end|>"):
        if tok.token_to_id(special) is None:
            raise ValueError(f"Tokenizer has no {special} token; re-run scripts/train_tokenizer.py")

    conversations = load_conversations(
        c["dataset"], split=c.get("split", "train"), limit=c.get("max_conversations")
    )
    n_val = max(1, int(len(conversations) * c.get("val_fraction", 0.01)))
    train = SFTDataset(tok, conversations[n_val:], c["seq_len"], pad_id, seed=c["seed"] + d.rank)
    val = SFTDataset(tok, conversations[:n_val], c["seq_len"], pad_id, seed=c["seed"])

    accum = c["grad_accum_steps"]
    batch_convs = c["micro_batch_size"] * accum * d.world
    steps = math.ceil(len(train) * c["epochs"] / batch_convs)
    if args.max_steps:
        steps = min(steps, args.max_steps)
    sched = {**c, "max_steps": steps, "warmup_steps": max(1, int(steps * c.get("warmup_fraction", 0.03)))}

    optimizer = build_optimizer(model, c, device)
    if d.enabled:
        model = DDP(model, device_ids=[d.local_rank])
    out = Path(c["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    log = MetricsLog(out, d.main)
    if d.main:
        print(
            f"SFT {model_cfg.name}: {len(train):,} conversations, {steps:,} steps, "
            f"{batch_convs} conversations/step, from {init}"
        )

    for step in range(steps):
        model.train()
        lr = lr_at(step, **sched)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=device)
        for micro in range(accum):
            x, y = train.batch(c["micro_batch_size"], device)
            sync = model.no_sync() if d.enabled and micro < accum - 1 else nullcontext()
            with sync, autocast(device, c["precision"]):
                _, loss = model(x, y)
                loss = loss / accum
            loss.backward()
            loss_sum += loss.detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(), c["grad_clip"])
        optimizer.step()

        if step % c["log_interval"] == 0 or step == steps - 1:
            if d.enabled:
                dist.all_reduce(loss_sum)
                loss_sum /= d.world
            log.write(step=step, loss=loss_sum.item(), lr=lr)

        if (step and step % c["eval_interval"] == 0) or step == steps - 1:
            model.eval()
            with torch.no_grad():
                losses = []
                for _ in range(min(c["eval_batches"], len(val))):
                    x, y = val.batch(c["micro_batch_size"], device)
                    with autocast(device, c["precision"]):
                        losses.append(model(x, y)[1].detach())
                mean = torch.stack(losses).mean()
            if d.enabled:
                dist.all_reduce(mean)
                mean /= d.world
            log.write(step=step, val_loss=mean.item())

    if d.main:
        save_checkpoint(out / "final.pt", model, None, steps - 1, run)
        print(f"Saved {out / 'final.pt'}")
    teardown_distributed(d)


if __name__ == "__main__":
    main()

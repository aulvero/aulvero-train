#!/usr/bin/env python
"""Pre-train an Aulvero model from scratch.

    python pretrain/train.py --config configs/kestrel-250m.yaml
    torchrun --standalone --nproc_per_node=8 pretrain/train.py --config configs/condor-900m.yaml

Resume from a checkpoint with ``--resume checkpoints/kestrel-250m/pretrain/step-0010000.pt``.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aulvero.config import ModelConfig, load_run_config  # noqa: E402
from aulvero.data import TokenBinDataset  # noqa: E402
from aulvero.model import AulveroLM  # noqa: E402
from aulvero.schedule import lr_at  # noqa: E402
from aulvero.training import (  # noqa: E402
    MetricsLog,
    autocast,
    build_optimizer,
    grad_accum_steps,
    pick_device,
    prune_checkpoints,
    save_checkpoint,
    seed_everything,
    setup_distributed,
    teardown_distributed,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="model file, e.g. configs/kestrel-250m.yaml")
    ap.add_argument("--resume", help="checkpoint to resume from")
    ap.add_argument("--max-steps", type=int, help="override pretrain.max_steps (handy for smoke tests)")
    args = ap.parse_args()

    run = load_run_config(args.config)
    c = run["pretrain"]
    if args.max_steps:
        c["max_steps"] = args.max_steps
    d = setup_distributed()
    seed_everything(c["seed"] + d.rank)
    device = pick_device(d)

    model_cfg = ModelConfig.from_dict(run["model"])
    if c["seq_len"] > model_cfg.max_seq_len:
        raise ValueError(f"pretrain.seq_len={c['seq_len']} exceeds model.max_seq_len={model_cfg.max_seq_len}")
    model = AulveroLM(model_cfg).to(device)
    optimizer = build_optimizer(model, c, device)
    accum = grad_accum_steps(c, d.world)

    start = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start = ck["step"] + 1

    if c.get("compile") and device.type == "cuda":
        model = torch.compile(model)
    if d.enabled:
        model = DDP(model, device_ids=[d.local_rank])

    train = TokenBinDataset(c["train_bin"], c["seq_len"], c["seed"] + d.rank + start)
    val = TokenBinDataset(c["val_bin"], c["seq_len"], c["seed"] + 99)
    out = Path(c["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    log = MetricsLog(out, d.main)

    tokens_per_step = c["global_batch_tokens"]
    if d.main:
        n = sum(p.numel() for p in model.parameters())
        total_tokens = tokens_per_step * c["max_steps"]
        print(
            f"{model_cfg.name}: {n:,} parameters | {tokens_per_step:,} tokens/step | "
            f"grad_accum={accum} | {total_tokens / 1e9:.1f}B tokens planned "
            f"({total_tokens / n:.0f} tokens/parameter) | schedule={c.get('lr_schedule', 'wsd')}"
        )

    t0 = time.time()
    for step in range(start, c["max_steps"]):
        model.train()
        lr = lr_at(step, **c)
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
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c["grad_clip"])
        optimizer.step()

        if step % c["log_interval"] == 0:
            if d.enabled:
                dist.all_reduce(loss_sum)
                loss_sum /= d.world
            dt = time.time() - t0
            steps_done = c["log_interval"] if step > start else 1
            log.write(
                step=step,
                loss=loss_sum.item(),
                lr=lr,
                grad_norm=float(grad_norm),
                tokens=(step + 1) * tokens_per_step,
                tok_per_s=int(tokens_per_step * steps_done / max(dt, 1e-6)),
            )
            t0 = time.time()

        last = step == c["max_steps"] - 1
        if (step and step % c["eval_interval"] == 0) or last:
            model.eval()
            losses = []
            with torch.no_grad():
                for _ in range(c["eval_batches"]):
                    x, y = val.batch(c["micro_batch_size"], device)
                    with autocast(device, c["precision"]):
                        _, vl = model(x, y)
                    losses.append(vl.detach())
            mean = torch.stack(losses).mean()
            if d.enabled:
                dist.all_reduce(mean)
                mean /= d.world
            log.write(step=step, val_loss=mean.item(), val_ppl=math.exp(min(20.0, mean.item())))

        if d.main and step and step % c["save_interval"] == 0 and not last:
            save_checkpoint(out / f"step-{step:07d}.pt", model, optimizer, step, run)
            prune_checkpoints(out, c.get("keep_last"))

    if d.main:
        save_checkpoint(out / "final.pt", model, optimizer, c["max_steps"] - 1, run)
        print(f"Saved {out / 'final.pt'}")
    teardown_distributed(d)


if __name__ == "__main__":
    main()

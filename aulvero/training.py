"""Pieces shared by pre-training and fine-tuning."""

from __future__ import annotations

import json
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass
class Dist:
    enabled: bool
    local_rank: int
    rank: int
    world: int

    @property
    def main(self) -> bool:
        return self.rank == 0


def setup_distributed() -> Dist:
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group("nccl")
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        return Dist(True, local, dist.get_rank(), dist.get_world_size())
    return Dist(False, 0, 0, 1)


def teardown_distributed(d: Dist) -> None:
    if d.enabled:
        dist.destroy_process_group()


def pick_device(d: Dist) -> torch.device:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return torch.device(f"cuda:{d.local_rank}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def autocast(device: torch.device, precision: str):
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def build_optimizer(model: torch.nn.Module, c: dict, device: torch.device) -> torch.optim.Optimizer:
    """AdamW with no weight decay on norms (1-D parameters)."""
    decay, no_decay = [], []
    seen = set()
    for p in model.parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": c["weight_decay"]},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=c["learning_rate"],
        betas=(c["beta1"], c["beta2"]),
        eps=c.get("adam_eps", 1e-8),
        fused=device.type == "cuda",
    )


def grad_accum_steps(c: dict, world: int) -> int:
    """Derive gradient accumulation from the global batch size in tokens."""
    per_step = c["micro_batch_size"] * c["seq_len"] * world
    total = c["global_batch_tokens"]
    if total % per_step:
        raise ValueError(
            f"global_batch_tokens={total} is not divisible by micro_batch_size*seq_len*GPUs={per_step}"
        )
    return total // per_step


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    raw = model.module if isinstance(model, DDP) else model
    return raw._orig_mod if hasattr(raw, "_orig_mod") else raw


def save_checkpoint(path: Path, model, optimizer, step: int, run_config: dict) -> None:
    raw = unwrap(model)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model": raw.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "step": step,
            "run_config": run_config,
            "model_config": raw.cfg.to_dict(),
        },
        tmp,
    )
    os.replace(tmp, path)  # never leave a half-written checkpoint behind


def prune_checkpoints(out_dir: Path, keep_last: int | None) -> None:
    if not keep_last:
        return
    steps = sorted(out_dir.glob("step-*.pt"))
    for old in steps[:-keep_last]:
        old.unlink(missing_ok=True)


class MetricsLog:
    """Prints a line and appends JSON to ``metrics.jsonl`` in the run directory."""

    def __init__(self, out_dir: Path, enabled: bool):
        self.enabled = enabled
        self.path = out_dir / "metrics.jsonl"
        self.t0 = time.time()

    def write(self, **row) -> None:
        if not self.enabled:
            return
        row["time"] = round(time.time() - self.t0, 1)
        print(" ".join(f"{k}={_fmt(v)}" for k, v in row.items()), flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.3e}" if abs(v) < 1e-2 else f"{v:.4f}"
    return str(v)

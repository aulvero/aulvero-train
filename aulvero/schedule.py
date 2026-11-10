"""Learning-rate schedules.

``wsd`` (warmup - stable - decay) is the default. It is what most recent open model
reports use (MiniCPM, SmolLM 2/3, DeepSeek): the learning rate stays flat for most of
the run and drops only in the final ``decay_fraction`` of steps. The stable phase can
be resumed or extended without restarting, and a decay can be branched off any
stable checkpoint.

``cosine`` is kept for comparison runs.
"""

from __future__ import annotations

import math


def lr_at(
    step: int,
    *,
    max_steps: int,
    warmup_steps: int,
    learning_rate: float,
    min_learning_rate: float,
    lr_schedule: str = "wsd",
    decay_fraction: float = 0.2,
    **_: object,
) -> float:
    if step < warmup_steps:
        return learning_rate * (step + 1) / warmup_steps
    if lr_schedule == "cosine":
        ratio = min(1.0, (step - warmup_steps) / max(1, max_steps - warmup_steps))
        return min_learning_rate + 0.5 * (learning_rate - min_learning_rate) * (1 + math.cos(math.pi * ratio))
    if lr_schedule == "wsd":
        decay_steps = max(1, int(max_steps * decay_fraction))
        decay_start = max_steps - decay_steps
        if step < decay_start:
            return learning_rate
        ratio = min(1.0, (step - decay_start) / decay_steps)
        return learning_rate + (min_learning_rate - learning_rate) * ratio
    if lr_schedule == "constant":
        return learning_rate
    raise ValueError(f"Unknown lr_schedule: {lr_schedule}")

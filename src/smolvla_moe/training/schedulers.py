from __future__ import annotations

import math
from typing import Any

import torch


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    train_config: dict[str, Any],
    max_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR | None:
    scheduler_config = train_config.get("scheduler")
    if scheduler_config is None:
        return None
    scheduler_type = str(scheduler_config.get("type", "cosine_decay_with_warmup"))
    if scheduler_type != "cosine_decay_with_warmup":
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    peak_lr = float(scheduler_config.get("peak_lr", train_config.get("learning_rate", 1e-4)))
    decay_lr = float(scheduler_config.get("decay_lr", 2.5e-6))
    warmup_steps = int(scheduler_config.get("warmup_steps", 1000))
    decay_steps = int(scheduler_config.get("decay_steps", 30000))
    if max_steps < decay_steps:
        scale = max_steps / float(decay_steps)
        warmup_steps = int(warmup_steps * scale)
        decay_steps = max_steps

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return _linear_warmup(current_step, warmup_steps)
        return _cosine_decay(current_step, decay_steps, peak_lr, decay_lr)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _linear_warmup(current_step: int, warmup_steps: int) -> float:
    if current_step <= 0:
        return 1.0 / float(warmup_steps + 1)
    frac = 1.0 - current_step / float(warmup_steps)
    return (1.0 / float(warmup_steps + 1) - 1.0) * frac + 1.0


def _cosine_decay(current_step: int, decay_steps: int, peak_lr: float, decay_lr: float) -> float:
    step = min(current_step, decay_steps)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * step / float(decay_steps)))
    alpha = decay_lr / peak_lr
    return (1.0 - alpha) * cosine_decay + alpha

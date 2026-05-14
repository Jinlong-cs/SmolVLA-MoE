from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, config: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
        },
        path,
    )

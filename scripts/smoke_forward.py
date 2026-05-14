#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smolvla_moe.config import load_config
from smolvla_moe.data import build_train_data
from smolvla_moe.models.policy import SmolVLAMoEPolicy
from smolvla_moe.models.policy import count_parameters


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a forward and sampling smoke test.")
    parser.add_argument("--config", default=str(ROOT / "configs/train/libero_smoke.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    model = SmolVLAMoEPolicy(config)
    batch = next(iter(build_train_data(config)))
    metrics = model.compute_loss(batch)
    actions = model.predict_action(batch, generator=torch.Generator().manual_seed(0))
    print(f"params={count_parameters(model):,}")
    print(f"loss={float(metrics['loss'].detach()):.6f} flow_loss={float(metrics['flow_loss'].detach()):.6f}")
    print(f"sampled_actions_shape={tuple(actions.shape)}")
    if "expert_usage" in metrics:
        print(f"expert_usage={metrics['expert_usage'].detach().cpu().tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

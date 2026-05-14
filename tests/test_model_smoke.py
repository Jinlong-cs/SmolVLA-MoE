from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smolvla_moe.config import load_config
from smolvla_moe.data import build_train_data
from smolvla_moe.models.policy import SmolVLAMoEPolicy


def test_tiny_policy_forward_and_sample() -> None:
    config = load_config(ROOT / "configs/train/libero_smoke.yaml")
    model = SmolVLAMoEPolicy(config)
    batch = next(iter(build_train_data(config)))
    metrics = model.compute_loss(batch)
    assert metrics["loss"].ndim == 0
    assert metrics["flow_loss"].ndim == 0
    actions = model.predict_action(batch, generator=torch.Generator().manual_seed(0))
    assert tuple(actions.shape) == (2, 8, 7)

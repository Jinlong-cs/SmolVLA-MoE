#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smolvla_moe.config import load_config
from smolvla_moe.models.policy import SmolVLAMoEPolicy
from smolvla_moe.models.policy import count_parameters


def main() -> int:
    parser = argparse.ArgumentParser(description="Print SmolVLA-MoE parameter counts.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    model = SmolVLAMoEPolicy(config)
    print(f"total_params={count_parameters(model):,}")
    print(f"trainable_params={count_parameters(model, trainable_only=True):,}")
    print(f"backbone_params={count_parameters(model.backbone):,}")
    print(f"action_decoder_params={count_parameters(model.action_decoder):,}")
    if model.flash_draft_head is not None:
        print(f"flash_draft_params={count_parameters(model.flash_draft_head):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

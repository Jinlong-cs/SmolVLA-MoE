#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smolvla_moe.config import load_config
from smolvla_moe.training import train


def main() -> int:
    parser = argparse.ArgumentParser(description="Train SmolVLA-MoE.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, max_steps_override=args.max_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

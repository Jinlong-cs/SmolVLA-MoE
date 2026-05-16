#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME
from lerobot.utils.constants import POLICY_PREPROCESSOR_DEFAULT_NAME

from smolvla_moe.official.dense_policy import build_official_dense_smolvla_policy
from smolvla_moe.official.stats import load_official_smolvla_stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a trained official dense SmolVLA checkpoint to LeRobot format.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    payload = torch.load(checkpoint_path, map_location=args.device)
    config = payload["config"]
    config["device"] = args.device

    model = build_official_dense_smolvla_policy(config).to(args.device)
    model.load_state_dict(payload["model"], strict=True)
    model.policy.save_pretrained(output_dir)
    stats = load_official_smolvla_stats(config)
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg=model.policy.config, dataset_stats=stats)
    preprocessor.save_pretrained(output_dir, config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json")
    postprocessor.save_pretrained(output_dir, config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json")
    print(f"exported_step={payload['step']}")
    print(f"policy_path={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smolvla_moe.serving import LiberoSmolVLAMoEPolicy


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve a SmolVLA-MoE checkpoint using the OpenPI websocket protocol.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32", "none"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-binarize-gripper", action="store_true")
    parser.add_argument("--no-clip-actions", action="store_true")
    parser.add_argument("--use-flash", action="store_true")
    parser.add_argument("--latency-log", default=None)
    args = parser.parse_args()

    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    policy = LiberoSmolVLAMoEPolicy(
        checkpoint=args.checkpoint,
        config_path=args.config,
        device=args.device,
        amp_dtype=None if args.amp_dtype in {"none", "float32"} else args.amp_dtype,
        seed=args.seed,
        binarize_gripper=not args.no_binarize_gripper,
        clip_actions=not args.no_clip_actions,
        use_flash=args.use_flash,
        latency_log=args.latency_log,
    )
    logging.info("Serving SmolVLA-MoE policy on %s:%d with metadata=%s", args.host, args.port, policy.metadata)
    server = WebsocketPolicyServer(policy=policy, host=args.host, port=args.port, metadata=policy.metadata)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    raise SystemExit(main())

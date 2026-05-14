#!/usr/bin/env python3
from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="LIBERO eval wrapper placeholder.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--suite", default="all")
    parser.add_argument("--num-trials", type=int, default=50)
    args = parser.parse_args()
    raise SystemExit(
        "LIBERO closed-loop eval is not wired yet. "
        "Use this entrypoint to preserve CLI shape, then connect it to the simulator runner on the VastAI machine. "
        f"checkpoint={args.checkpoint} suite={args.suite} trials={args.num_trials}"
    )


if __name__ == "__main__":
    raise SystemExit(main())

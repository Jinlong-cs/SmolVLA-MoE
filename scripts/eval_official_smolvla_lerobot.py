#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


SUITE_SHARDS = (
    ("libero_spatial", [0, 1, 2, 3, 4], "spatial_0_4"),
    ("libero_spatial", [5, 6, 7, 8, 9], "spatial_5_9"),
    ("libero_object", [0, 1, 2, 3, 4], "object_0_4"),
    ("libero_object", [5, 6, 7, 8, 9], "object_5_9"),
    ("libero_goal", [0, 1, 2, 3, 4], "goal_0_4"),
    ("libero_goal", [5, 6, 7, 8, 9], "goal_5_9"),
    ("libero_10", [0, 1, 2, 3, 4], "libero10_0_4"),
    ("libero_10", [5, 6, 7, 8, 9], "libero10_5_9"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run and aggregate LeRobot-native LIBERO eval for SmolVLA.")
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lerobot-eval", default=os.environ.get("LEROBOT_EVAL", "lerobot-eval"))
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-parallel-tasks", type=int, default=1)
    parser.add_argument("--device-prefix", default="cuda")
    parser.add_argument("--use-amp", default="true", choices=["true", "false"])
    parser.add_argument("--use-async-envs", default="false", choices=["true", "false"])
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--launch", action="store_true", help="Run all shards before aggregating.")
    parser.add_argument("--no-video", action="store_true", help="Disable env video writing when supported.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.launch:
        _launch_shards(args, output_dir)
    summary = aggregate(output_dir)
    summary["policy_path"] = args.policy_path
    summary["protocol"] = {
        "runner": "lerobot-eval",
        "n_episodes_per_task": args.n_episodes,
        "batch_size": args.batch_size,
        "use_async_envs": args.use_async_envs == "true",
        "max_parallel_tasks": args.max_parallel_tasks,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def _launch_shards(args: argparse.Namespace, output_dir: Path) -> None:
    processes: list[tuple[str, subprocess.Popen[Any], Any]] = []
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    for index, (suite, task_ids, shard_name) in enumerate(SUITE_SHARDS):
        shard_dir = output_dir / shard_name
        shard_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.setdefault("MUJOCO_GL", "egl")
        env.setdefault("PYOPENGL_PLATFORM", "egl")
        if args.num_gpus > 0:
            env["CUDA_VISIBLE_DEVICES"] = str(index % args.num_gpus)
        cmd = [
            args.lerobot_eval,
            f"--policy.path={args.policy_path}",
            "--env.type=libero",
            f"--env.task={suite}",
            f"--env.task_ids={json.dumps(task_ids)}",
            f"--env.max_parallel_tasks={args.max_parallel_tasks}",
            f"--eval.batch_size={args.batch_size}",
            f"--eval.n_episodes={args.n_episodes}",
            f"--eval.use_async_envs={args.use_async_envs}",
            f"--policy.device={args.device_prefix}",
            f"--policy.use_amp={args.use_amp}",
            f"--output_dir={shard_dir}",
        ]
        if args.no_video:
            cmd.append("--env.video=false")
        log_file = (logs_dir / f"{shard_name}.log").open("w", encoding="utf-8")
        process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
        processes.append((shard_name, process, log_file))

    failed: list[tuple[str, int]] = []
    try:
        while processes:
            remaining = []
            for shard_name, process, log_file in processes:
                return_code = process.poll()
                if return_code is None:
                    remaining.append((shard_name, process, log_file))
                elif return_code != 0:
                    failed.append((shard_name, return_code))
            if failed:
                for _, process, _ in remaining:
                    process.terminate()
                break
            if not remaining:
                break
            processes = remaining
            time.sleep(5)
    finally:
        for _, process, log_file in processes:
            if process.poll() is None:
                process.terminate()
            log_file.close()
    if failed:
        details = ", ".join(f"{name} exited {code}" for name, code in failed)
        raise RuntimeError(f"LeRobot eval shard failed: {details}. See logs in {logs_dir}")


def aggregate(output_dir: Path) -> dict[str, Any]:
    eval_paths = sorted(output_dir.glob("*/eval_info.json"))
    if len(eval_paths) != len(SUITE_SHARDS):
        raise RuntimeError(f"Expected {len(SUITE_SHARDS)} eval_info.json files under {output_dir}, found {len(eval_paths)}")

    by_suite: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    per_task = []
    for path in eval_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in _walk_task_metrics(data):
            suite = str(item["task_group"])
            task_id = int(item["task_id"])
            successes = [bool(value) for value in item["metrics"]["successes"]]
            success_count = int(sum(successes))
            episode_count = len(successes)
            by_suite[suite][0] += success_count
            by_suite[suite][1] += episode_count
            per_task.append(
                {
                    "suite": suite,
                    "task_id": task_id,
                    "successes": success_count,
                    "episodes": episode_count,
                    "success_rate": success_count / episode_count if episode_count else 0.0,
                    "source": str(path),
                }
            )

    if len(per_task) != 40:
        raise RuntimeError(f"Expected 40 LIBERO task results, found {len(per_task)}")

    total_successes = sum(value[0] for value in by_suite.values())
    total_episodes = sum(value[1] for value in by_suite.values())
    return {
        "suites": {
            suite: {
                "successes": successes,
                "episodes": episodes,
                "success_rate": successes / episodes if episodes else 0.0,
            }
            for suite, (successes, episodes) in sorted(by_suite.items())
        },
        "overall": {
            "successes": total_successes,
            "episodes": total_episodes,
            "success_rate": total_successes / total_episodes if total_episodes else 0.0,
        },
        "per_task": sorted(per_task, key=lambda item: (item["suite"], item["task_id"])),
        "eval_info_files": [str(path) for path in eval_paths],
    }


def _walk_task_metrics(obj: Any) -> Any:
    if isinstance(obj, dict):
        metrics = obj.get("metrics")
        if "task_group" in obj and "task_id" in obj and isinstance(metrics, dict) and "successes" in metrics:
            yield obj
        for value in obj.values():
            yield from _walk_task_metrics(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _walk_task_metrics(value)


if __name__ == "__main__":
    raise SystemExit(main())

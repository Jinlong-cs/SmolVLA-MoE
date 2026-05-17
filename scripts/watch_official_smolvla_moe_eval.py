#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
TASKS_PER_SUITE = 10


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Wait for official SmolVLA-MoE training, select the best checkpoint by flow loss, and eval LIBERO."
    )
    parser.add_argument("--train-output-dir", required=True)
    parser.add_argument("--checkpoint-dir", action="append", required=True)
    parser.add_argument("--metric-path", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/train/official_smolvla_moe_libero_8gpu.yaml")
    parser.add_argument("--max-step", type=int, default=100000)
    parser.add_argument("--metric-key", default="train/flow_loss")
    parser.add_argument("--window-steps", type=int, default=1000)
    parser.add_argument("--wait-interval", type=int, default=300)
    parser.add_argument("--suites", nargs="+", choices=SUITES, default=list(SUITES))
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--base-port", type=int, default=8500)
    parser.add_argument("--openpi-root", default=os.environ.get("OPENPI_ROOT", "/workspace/openpi"))
    parser.add_argument("--server-python", default=sys.executable)
    parser.add_argument("--libero-python", default="/workspace/openpi/examples/libero/.venv/bin/python")
    parser.add_argument("--server-script", default="scripts/serve_official_smolvla_moe_policy.py")
    parser.add_argument("--eval-script", default="scripts/eval_libero.py")
    parser.add_argument("--train-session", default=None)
    parser.add_argument("--save-videos", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    wait_for_training(Path(args.train_output_dir), int(args.max_step), int(args.wait_interval), args.train_session)
    metric_records = load_metric_records([Path(path) for path in args.metric_path])
    candidates = score_checkpoints(
        [Path(path) for path in args.checkpoint_dir],
        metric_records,
        metric_key=str(args.metric_key),
        window_steps=int(args.window_steps),
    )
    best = min(candidates, key=lambda item: item["score"])
    selection = {"best": best, "candidates": candidates}
    selection_path = output_dir / "best_checkpoint.json"
    selection_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(json.dumps({"selected_best_checkpoint": best, "selection_path": str(selection_path)}, indent=2), flush=True)

    task_results = run_dynamic_eval(best["checkpoint"], args, root, output_dir)
    summary = summarize_eval(task_results, best, args)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def wait_for_training(train_output_dir: Path, max_step: int, interval: int, train_session: str | None) -> None:
    metrics_path = train_output_dir / "metrics.jsonl"
    checkpoints_dir = train_output_dir / "checkpoints"
    final_checkpoint = checkpoints_dir / "final.pt"
    step_checkpoint = checkpoints_dir / f"step_{max_step:06d}.pt"
    while True:
        latest = latest_step(metrics_path)
        if latest >= max_step and (stable_file(final_checkpoint) or stable_file(step_checkpoint)):
            return
        if train_session is not None and not tmux_session_exists(train_session) and latest < max_step:
            raise RuntimeError(f"Training session {train_session!r} ended before max_step={max_step}; latest={latest}")
        print(
            json.dumps(
                {
                    "event": "waiting_for_training",
                    "latest_step": latest,
                    "target_step": max_step,
                    "final_checkpoint": str(final_checkpoint),
                    "step_checkpoint": str(step_checkpoint),
                }
            ),
            flush=True,
        )
        time.sleep(interval)


def latest_step(metrics_path: Path) -> int:
    if not metrics_path.exists():
        return 0
    step = 0
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                step = max(step, int(record["step"]))
    return step


def stable_file(path: Path) -> bool:
    if not path.exists():
        return False
    size = path.stat().st_size
    if size <= 0:
        return False
    time.sleep(5)
    return path.exists() and path.stat().st_size == size


def tmux_session_exists(name: str) -> bool:
    return subprocess.run(["tmux", "has-session", "-t", name], check=False).returncode == 0


def load_metric_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Metric path does not exist: {path}")
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    records.sort(key=lambda record: int(record["step"]))
    if not records:
        raise RuntimeError("No metric records were loaded")
    return records


def score_checkpoints(
    checkpoint_dirs: list[Path],
    records: list[dict[str, Any]],
    metric_key: str,
    window_steps: int,
) -> list[dict[str, Any]]:
    candidates = []
    for checkpoint in checkpoint_files(checkpoint_dirs):
        step = checkpoint_step(checkpoint)
        window = [
            float(record[metric_key])
            for record in records
            if step - window_steps <= int(record["step"]) <= step and metric_key in record
        ]
        if not window:
            raise RuntimeError(f"No {metric_key} records found for checkpoint window ending at step {step}")
        candidates.append(
            {
                "checkpoint": str(checkpoint),
                "step": step,
                "metric_key": metric_key,
                "window_steps": window_steps,
                "score": sum(window) / len(window),
                "num_records": len(window),
            }
        )
    if not candidates:
        raise RuntimeError("No checkpoint candidates were found")
    candidates.sort(key=lambda item: (item["score"], item["step"]))
    return candidates


def checkpoint_files(checkpoint_dirs: list[Path]) -> list[Path]:
    files = []
    for checkpoint_dir in checkpoint_dirs:
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
        files.extend(sorted(checkpoint_dir.glob("step_*.pt")))
    return files


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"step_(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Checkpoint name does not encode a step: {path}")
    return int(match.group(1))


def run_dynamic_eval(checkpoint: str, args: argparse.Namespace, root: Path, output_dir: Path) -> list[dict[str, Any]]:
    gpus = [gpu.strip() for gpu in str(args.gpus).split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("At least one GPU id is required")
    tasks = [(suite, task_id) for suite in args.suites for task_id in range(TASKS_PER_SUITE)]
    pending = list(tasks)
    idle_gpus = list(gpus)
    running: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    eval_root = output_dir / "tasks"
    eval_root.mkdir(parents=True, exist_ok=True)

    while pending or running:
        while pending and idle_gpus:
            gpu = idle_gpus.pop(0)
            suite, task_id = pending.pop(0)
            task_output = eval_root / suite / f"task_{task_id:02d}"
            task_output.mkdir(parents=True, exist_ok=True)
            log_path = task_output / "eval.log"
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["PYTHONUNBUFFERED"] = "1"
            cmd = eval_command(checkpoint, args, root, suite, task_id, task_output, int(args.base_port) + int(gpu))
            log_file = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
            running.append(
                {
                    "process": process,
                    "log_file": log_file,
                    "gpu": gpu,
                    "suite": suite,
                    "task_id": task_id,
                    "output_dir": str(task_output),
                    "cmd": cmd,
                }
            )
            print(json.dumps({"event": "eval_started", "suite": suite, "task_id": task_id, "gpu": gpu}), flush=True)

        time.sleep(10)
        still_running = []
        for item in running:
            process = item["process"]
            return_code = process.poll()
            if return_code is None:
                still_running.append(item)
                continue
            item["log_file"].close()
            if return_code != 0:
                terminate_running([active for active in running if active is not item])
                raise RuntimeError(
                    f"Eval failed for {item['suite']} task {item['task_id']} on GPU {item['gpu']}; "
                    f"log={Path(item['output_dir']) / 'eval.log'}"
                )
            completed.append(read_task_summary(item))
            idle_gpus.append(str(item["gpu"]))
            print(
                json.dumps({"event": "eval_finished", "suite": item["suite"], "task_id": item["task_id"]}),
                flush=True,
            )
        running = still_running

    completed.sort(key=lambda item: (str(item["suite"]), int(item["task_id"])))
    return completed


def eval_command(
    checkpoint: str,
    args: argparse.Namespace,
    root: Path,
    suite: str,
    task_id: int,
    output_dir: Path,
    port: int,
) -> list[str]:
    cmd = [
        args.server_python,
        str(root / args.eval_script),
        f"--checkpoint={checkpoint}",
        f"--config={args.config}",
        f"--suite={suite}",
        f"--task-start={task_id}",
        f"--task-end={task_id + 1}",
        f"--num-trials={args.num_trials}",
        f"--openpi-root={args.openpi_root}",
        f"--server-python={args.server_python}",
        f"--libero-python={args.libero_python}",
        f"--output-dir={output_dir}",
        f"--port={port}",
        f"--server-script={args.server_script}",
    ]
    if not args.save_videos:
        cmd.append("--no-save-videos")
    return cmd


def terminate_running(items: list[dict[str, Any]]) -> None:
    for item in items:
        process = item["process"]
        if process.poll() is None:
            process.terminate()
    for item in items:
        process = item["process"]
        if process.poll() is None:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)
        item["log_file"].close()


def read_task_summary(item: dict[str, Any]) -> dict[str, Any]:
    summary_path = Path(item["output_dir"]) / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing eval summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    suite_summary = summary["suites"][item["suite"]]
    return {
        "suite": item["suite"],
        "task_id": int(item["task_id"]),
        "success_rate": float(suite_summary["success_rate"]),
        "total_episodes": int(suite_summary["total_episodes"]),
        "total_successes": int(suite_summary["total_successes"]),
        "output_dir": item["output_dir"],
    }


def summarize_eval(task_results: list[dict[str, Any]], best: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    suite_summary = {}
    for suite in args.suites:
        suite_results = [item for item in task_results if item["suite"] == suite]
        total_episodes = sum(int(item["total_episodes"]) for item in suite_results)
        total_successes = sum(int(item["total_successes"]) for item in suite_results)
        suite_summary[suite] = {
            "success_rate": total_successes / total_episodes if total_episodes else 0.0,
            "total_episodes": total_episodes,
            "total_successes": total_successes,
        }
    overall_episodes = sum(item["total_episodes"] for item in task_results)
    overall_successes = sum(item["total_successes"] for item in task_results)
    return {
        "best_checkpoint": best,
        "protocol": {
            "suites": list(args.suites),
            "tasks_per_suite": TASKS_PER_SUITE,
            "num_trials_per_task": int(args.num_trials),
            "runtime": "openpi-sync-server-client",
            "videos": "disabled",
        },
        "suites": suite_summary,
        "overall_success_rate": overall_successes / overall_episodes if overall_episodes else 0.0,
        "total_episodes": overall_episodes,
        "total_successes": overall_successes,
        "tasks": task_results,
    }


if __name__ == "__main__":
    raise SystemExit(main())

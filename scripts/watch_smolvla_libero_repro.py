#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


REFERENCE = {
    "libero_spatial": 0.90,
    "libero_object": 1.00,
    "libero_goal": 1.00,
    "libero_10": 0.60,
    "overall": 0.87,
}

UNTOUCHED_HF_NATIVE = {
    "libero_spatial": 0.78,
    "libero_object": 0.87,
    "libero_goal": 0.78,
    "libero_10": 0.47,
    "overall": 0.725,
}


EXPERIMENTS = [
    {
        "name": "hf_smolvla_libero_official_preset_finetune",
        "kind": "released_checkpoint_finetune",
        "config": "configs/train/hf_smolvla_libero_official_preset_finetune_8gpu.yaml",
        "output_dir": "outputs/libero/hf_smolvla_libero_official_preset_finetune",
        "wait_tmux": "hf_smolvla_libero_preset_finetune_8gpu_30k",
        "train_log": "train.log",
        "next_reason": "If released-checkpoint finetuning does not improve local native eval, retry the real base-init reproduction with the official optimizer/scheduler preset.",
    },
    {
        "name": "official_smolvla_dense_base_init_official_preset",
        "kind": "base_init_reproduction",
        "config": "configs/train/official_smolvla_dense_libero_official_preset_8gpu.yaml",
        "output_dir": "outputs/libero/official_smolvla_dense_base_init_official_preset",
        "train_log": "train.log",
        "next_reason": "If this fails, the remaining likely gap is protocol/data/official-runner mismatch rather than the generic optimizer settings.",
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch SmolVLA LIBERO training, run eval, and chain reproduction attempts.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--lerobot-eval", required=True)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--state-dir", default="outputs/libero/watchers/smolvla_official_repro")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    state_dir = repo_root / args.state_dir
    state_dir.mkdir(parents=True, exist_ok=True)
    state_jsonl = state_dir / "watch_state.jsonl"

    env = os.environ.copy()
    env.setdefault("HF_HOME", "/workspace/.hf_home")
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    env.setdefault("WANDB_MODE", "online")
    env["PYTHONPATH"] = str(repo_root / "src")

    for index, experiment in enumerate(EXPERIMENTS):
        _record(state_jsonl, "experiment_start", {"experiment": experiment})
        output_dir = repo_root / str(experiment["output_dir"])
        config_path = repo_root / str(experiment["config"])
        final_checkpoint = output_dir / "checkpoints" / "final.pt"

        if not final_checkpoint.exists():
            wait_tmux = experiment.get("wait_tmux")
            if wait_tmux:
                _wait_for_tmux(wait_tmux, args.poll_seconds, state_jsonl)
            else:
                _run_training(args.python, config_path, output_dir, int(args.num_gpus), env)

        _require_file(final_checkpoint)
        _check_train_log(output_dir / str(experiment["train_log"]))

        policy_dir = output_dir / "lerobot_policy_final"
        if not (policy_dir / "model.safetensors").exists():
            _run_logged(
                [
                    args.python,
                    "scripts/export_official_smolvla_checkpoint.py",
                    "--checkpoint",
                    str(final_checkpoint),
                    "--output-dir",
                    str(policy_dir),
                    "--device",
                    "cpu",
                ],
                output_dir / "export_lerobot_policy.log",
                repo_root,
                env,
            )

        eval_dir = output_dir / f"lerobot_native_eval_{args.n_episodes}trials_dynamic_8gpu_final"
        summary_path = eval_dir / "summary.json"
        if not summary_path.exists():
            _run_logged(
                [
                    args.python,
                    "scripts/eval_official_smolvla_lerobot.py",
                    "--policy-path",
                    str(policy_dir),
                    "--output-dir",
                    str(eval_dir),
                    "--lerobot-eval",
                    args.lerobot_eval,
                    "--n-episodes",
                    str(args.n_episodes),
                    "--batch-size",
                    "1",
                    "--max-parallel-tasks",
                    "1",
                    "--num-gpus",
                    str(args.num_gpus),
                    "--launch",
                    "--dynamic-tasks",
                ],
                output_dir / "eval_lerobot_native.log",
                repo_root,
                env,
            )

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        comparison = _compare(summary, args.tolerance)
        diagnosis = _diagnose(experiment, output_dir, summary, comparison)
        diagnosis_path = output_dir / "repro_diagnosis.json"
        diagnosis_path.write_text(json.dumps(diagnosis, indent=2), encoding="utf-8")
        _record(state_jsonl, "eval_complete", diagnosis)

        if comparison["aligned"]:
            _record(state_jsonl, "aligned", diagnosis)
            print(json.dumps(diagnosis, indent=2), flush=True)
            return 0

        if index + 1 >= len(EXPERIMENTS):
            _record(state_jsonl, "not_aligned_no_next_experiment", diagnosis)
            print(json.dumps(diagnosis, indent=2), flush=True)
            return 2

        _record(
            state_jsonl,
            "not_aligned_launch_next",
            {
                "failed_experiment": experiment["name"],
                "next_experiment": EXPERIMENTS[index + 1]["name"],
                "reason": experiment["next_reason"],
            },
        )

    return 2


def _wait_for_tmux(session: str, poll_seconds: int, state_jsonl: Path) -> None:
    while _tmux_exists(session):
        _record(state_jsonl, "waiting_for_training", {"tmux": session})
        time.sleep(poll_seconds)


def _tmux_exists(session: str) -> bool:
    result = subprocess.run(["tmux", "has-session", "-t", session], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def _run_training(python: str, config_path: Path, output_dir: Path, num_gpus: int, env: dict[str, str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _run_logged(
        [
            python,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={num_gpus}",
            "scripts/train_official_smolvla_dense.py",
            "--config",
            str(config_path),
        ],
        output_dir / "train.log",
        output_dir.parents[2],
        env,
    )


def _run_logged(command: list[str], log_path: Path, cwd: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n$ " + " ".join(command) + "\n")
        handle.flush()
        result = subprocess.run(command, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(command)}. See {log_path}")


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def _check_train_log(path: Path) -> None:
    _require_file(path)
    text = path.read_text(encoding="utf-8", errors="ignore").lower()
    markers = ("traceback", "runtimeerror", "out of memory", "nan", "killed")
    found = [marker for marker in markers if marker in text]
    if found:
        raise RuntimeError(f"Training log contains failure markers {found}: {path}")


def _compare(summary: dict[str, Any], tolerance: float) -> dict[str, Any]:
    rates = {
        suite: float(summary["suites"][suite]["success_rate"])
        for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    }
    rates["overall"] = float(summary["overall"]["success_rate"])
    gaps = {key: rates[key] - REFERENCE[key] for key in REFERENCE}
    aligned = all(rates[key] + tolerance >= REFERENCE[key] for key in REFERENCE)
    return {
        "reference": REFERENCE,
        "rates": rates,
        "gaps": gaps,
        "tolerance": tolerance,
        "aligned": aligned,
    }


def _diagnose(
    experiment: dict[str, Any],
    output_dir: Path,
    summary: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    train_stats = _load_train_stats(output_dir / "metrics.jsonl")
    reasons = []
    overall = comparison["rates"]["overall"]
    if experiment["kind"] == "released_checkpoint_finetune" and overall < UNTOUCHED_HF_NATIVE["overall"]:
        reasons.append(
            "This finetune underperforms the untouched HuggingFaceVLA/smolvla_libero native-eval baseline; continuing a released checkpoint is likely degrading policy behavior."
        )
    if comparison["rates"]["libero_10"] < REFERENCE["libero_10"]:
        reasons.append("LIBERO-10 remains the largest long-horizon gap to the official reference.")
    if experiment["kind"] == "base_init_reproduction" and not comparison["aligned"]:
        reasons.append(
            "Base-init official-preset training still failed to match the official reference, so the remaining gap likely involves protocol, dataset preprocessing, LeRobot version, simulator seeds/init states, or native official training code rather than only optimizer hyperparameters."
        )
    return {
        "experiment": experiment["name"],
        "kind": experiment["kind"],
        "output_dir": str(output_dir),
        "summary": summary,
        "comparison": comparison,
        "train_stats": train_stats,
        "diagnosis": reasons,
        "untouched_hf_native_baseline": UNTOUCHED_HF_NATIVE,
    }


def _load_train_stats(metrics_path: Path) -> dict[str, Any]:
    if not metrics_path.exists():
        return {}
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return {}
    last = rows[-1]
    last_100 = rows[-100:]
    return {
        "last_step": int(last.get("step", -1)),
        "last_loss": float(last.get("train/loss", 0.0)),
        "last_grad_norm": float(last.get("train/grad_norm", 0.0)),
        "last_lr": float(last.get("train/lr", 0.0)),
        "last_100_loss_mean": sum(float(row["train/loss"]) for row in last_100) / len(last_100),
    }


def _record(path: Path, event: str, payload: dict[str, Any]) -> None:
    row = {"time": time.time(), "event": event, **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(json.dumps(row, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

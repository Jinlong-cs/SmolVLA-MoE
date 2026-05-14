#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LIBERO closed-loop eval through the OpenPI LIBERO runner.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/train/libero_8gpu.yaml")
    parser.add_argument("--suite", default="all", choices=["all", *SUITES])
    parser.add_argument("--num-trials", type=int, default=50)
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument("--task-end", type=int, default=None)
    parser.add_argument("--openpi-root", default=os.environ.get("OPENPI_ROOT", "/workspace/openpi"))
    parser.add_argument("--server-python", default=sys.executable)
    parser.add_argument("--libero-python", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32", "none"])
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--resize-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-save-videos", action="store_true")
    parser.add_argument("--no-binarize-gripper", action="store_true")
    parser.add_argument("--no-clip-actions", action="store_true")
    parser.add_argument("--server-log", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir or _default_output_dir(args.checkpoint)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    server_log = Path(args.server_log or output_dir / "policy_server.log")

    server = _start_server(args, root, server_log)
    try:
        _wait_for_health(args.host, args.port)
        results = []
        suites = SUITES if args.suite == "all" else (args.suite,)
        for suite in suites:
            result_path = output_dir / f"{suite}_results.json"
            video_path = output_dir / "videos" / suite
            cmd = [
                _libero_python(args),
                str(Path(args.openpi_root) / "examples/libero/main_task_slice.py"),
                f"--host={args.host}",
                f"--port={args.port}",
                f"--resize-size={args.resize_size}",
                f"--replan-steps={args.replan_steps}",
                f"--task-suite-name={suite}",
                f"--task-start={args.task_start}",
                f"--num-trials-per-task={args.num_trials}",
                f"--video-out-path={video_path}",
                f"--result-path={result_path}",
                f"--seed={args.seed}",
            ]
            if args.task_end is not None:
                cmd.append(f"--task-end={args.task_end}")
            if args.no_save_videos:
                cmd.append("--no-save-videos")
            subprocess.run(cmd, check=True, env=_eval_env(args.openpi_root, root))
            results.append(json.loads(result_path.read_text(encoding="utf-8")))
        summary = _summarize(results)
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
    finally:
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()
    return 0


def _start_server(args: argparse.Namespace, root: Path, server_log: Path) -> subprocess.Popen:
    server_log.parent.mkdir(parents=True, exist_ok=True)
    log_file = server_log.open("w", encoding="utf-8")
    cmd = [
        args.server_python,
        str(root / "scripts/serve_libero_policy.py"),
        f"--checkpoint={args.checkpoint}",
        f"--config={args.config}",
        f"--host=0.0.0.0",
        f"--port={args.port}",
        f"--device={args.device}",
        f"--amp-dtype={args.amp_dtype}",
        f"--seed={args.seed}",
    ]
    if args.no_binarize_gripper:
        cmd.append("--no-binarize-gripper")
    if args.no_clip_actions:
        cmd.append("--no-clip-actions")
    return subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=_eval_env(args.openpi_root, root))


def _wait_for_health(host: str, port: int, timeout_s: float = 240.0) -> None:
    import urllib.request

    deadline = time.time() + timeout_s
    url = f"http://{host}:{port}/healthz"
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except Exception as exc:
            last_error = exc
            time.sleep(2)
    raise TimeoutError(f"Policy server did not become healthy at {url}: {last_error}")


def _eval_env(openpi_root: str, repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = [
        str(repo_root / "src"),
        str(Path(openpi_root) / "src"),
        str(Path(openpi_root) / "packages/openpi-client/src"),
        str(Path(openpi_root) / "third_party/libero"),
    ]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath)
    env.setdefault("HF_HOME", "/workspace/.hf_home")
    env.setdefault("LIBERO_CONFIG_PATH", str(Path(openpi_root) / "run_configs/libero_openpi"))
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    return env


def _libero_python(args: argparse.Namespace) -> str:
    if args.libero_python is not None:
        return str(args.libero_python)
    candidate = Path(args.openpi_root) / "examples/libero/.venv/bin/python"
    return str(candidate) if candidate.exists() else sys.executable


def _default_output_dir(checkpoint: str) -> str:
    ckpt = Path(checkpoint)
    return str(ckpt.parent.parent / "libero_eval" / ckpt.stem)


def _summarize(results: list[dict]) -> dict:
    total_episodes = sum(int(result["total_episodes"]) for result in results)
    total_successes = sum(int(result["total_successes"]) for result in results)
    return {
        "suites": {
            result["task_suite_name"]: {
                "success_rate": float(result["success_rate"]),
                "total_episodes": int(result["total_episodes"]),
                "total_successes": int(result["total_successes"]),
            }
            for result in results
        },
        "overall_success_rate": total_successes / total_episodes if total_episodes else 0.0,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
    }


if __name__ == "__main__":
    raise SystemExit(main())

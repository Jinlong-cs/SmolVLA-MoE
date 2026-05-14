#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
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
    parser.add_argument("--video-dir", default=None, help="Directory for rollout MP4 files. Defaults to OUTPUT/videos.")
    parser.add_argument("--video-index", default="videos.html", help="HTML video browser written inside OUTPUT.")
    parser.add_argument("--no-video-index", action="store_true", help="Do not write video_manifest.json or HTML index.")
    parser.add_argument("--no-save-videos", action="store_true")
    parser.add_argument("--no-binarize-gripper", action="store_true")
    parser.add_argument("--no-clip-actions", action="store_true")
    parser.add_argument("--server-log", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir or _default_output_dir(args.checkpoint)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    server_log = Path(args.server_log or output_dir / "policy_server.log")
    video_root = Path(args.video_dir).resolve() if args.video_dir is not None else output_dir / "videos"

    server = _start_server(args, root, server_log)
    try:
        _wait_for_health(args.host, args.port)
        results = []
        suites = SUITES if args.suite == "all" else (args.suite,)
        for suite in suites:
            result_path = output_dir / f"{suite}_results.json"
            video_path = video_root / suite
            cmd = [
                _libero_python(args),
                str(Path(args.openpi_root) / "examples/libero/main_task_slice.py"),
                f"--args.host={args.host}",
                f"--args.port={args.port}",
                f"--args.resize-size={args.resize_size}",
                f"--args.replan-steps={args.replan_steps}",
                f"--args.task-suite-name={suite}",
                f"--args.task-start={args.task_start}",
                f"--args.num-trials-per-task={args.num_trials}",
                f"--args.video-out-path={video_path}",
                f"--args.result-path={result_path}",
                f"--args.seed={args.seed}",
            ]
            if args.task_end is not None:
                cmd.append(f"--args.task-end={args.task_end}")
            if args.no_save_videos:
                cmd.append("--args.no-save-videos")
            subprocess.run(cmd, check=True, env=_eval_env(args.openpi_root, root))
            results.append(json.loads(result_path.read_text(encoding="utf-8")))
        summary = _summarize(results)
        if args.no_save_videos:
            summary["videos"] = {"enabled": False}
        else:
            manifest = _collect_video_manifest(video_root, output_dir)
            manifest_path = output_dir / "video_manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            summary["videos"] = {
                "enabled": True,
                "video_dir": str(video_root),
                "count": len(manifest["videos"]),
                "manifest_path": str(manifest_path),
            }
            if not args.no_video_index:
                index_path = output_dir / args.video_index
                _write_video_index(index_path, manifest, summary)
                summary["videos"]["index_path"] = str(index_path)
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


def _collect_video_manifest(video_root: Path, output_dir: Path) -> dict:
    videos = []
    for path in sorted(video_root.rglob("*.mp4")):
        rel_output = Path(os.path.relpath(path, output_dir)).as_posix()
        rel_video_root = Path(os.path.relpath(path, video_root)).as_posix()
        status = None
        if path.stem.endswith("_success"):
            status = "success"
        elif path.stem.endswith("_failure"):
            status = "failure"
        videos.append(
            {
                "path": str(path),
                "relative_path": rel_output,
                "suite": path.parent.name,
                "name": path.name,
                "status": status,
                "size_bytes": path.stat().st_size,
                "relative_to_video_dir": rel_video_root,
            }
        )
    return {
        "video_dir": str(video_root),
        "output_dir": str(output_dir),
        "videos": videos,
    }


def _write_video_index(index_path: Path, manifest: dict, summary: dict) -> None:
    grouped: dict[str, list[dict]] = {}
    for video in manifest["videos"]:
        grouped.setdefault(str(video["suite"]), []).append(video)

    lines = [
        "<!doctype html>",
        "<html>",
        "<head>",
        '  <meta charset="utf-8">',
        "  <title>SmolVLA-MoE LIBERO Eval Videos</title>",
        "  <style>",
        "    body { font-family: Arial, sans-serif; margin: 24px; color: #111827; }",
        "    h1 { font-size: 24px; margin-bottom: 4px; }",
        "    h2 { margin-top: 28px; border-bottom: 1px solid #e5e7eb; padding-bottom: 8px; }",
        "    .meta { color: #4b5563; margin-bottom: 18px; }",
        "    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px; }",
        "    .card { border: 1px solid #e5e7eb; border-radius: 6px; padding: 10px; background: #fff; }",
        "    video { width: 100%; background: #000; border-radius: 4px; }",
        "    .caption { font-size: 12px; overflow-wrap: anywhere; margin-top: 8px; color: #374151; }",
        "    .success { color: #047857; font-weight: 700; }",
        "    .failure { color: #b91c1c; font-weight: 700; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <h1>SmolVLA-MoE LIBERO Eval Videos</h1>",
        (
            '  <div class="meta">'
            f'Overall success: {summary["total_successes"]}/{summary["total_episodes"]} '
            f'({summary["overall_success_rate"]:.2%}) | Videos: {len(manifest["videos"])}'
            "</div>"
        ),
    ]
    if not grouped:
        lines.append("  <p>No videos were found.</p>")
    for suite, videos in sorted(grouped.items()):
        lines.append(f"  <h2>{html.escape(suite)}</h2>")
        lines.append('  <div class="grid">')
        for video in videos:
            src = html.escape(str(video["relative_path"]))
            name = html.escape(str(video["name"]))
            status = str(video.get("status") or "unknown")
            status_class = html.escape(status)
            lines.extend(
                [
                    '    <div class="card">',
                    f'      <video controls preload="metadata" src="{src}"></video>',
                    (
                        '      <div class="caption">'
                        f'<span class="{status_class}">{html.escape(status)}</span> | {name}'
                        "</div>"
                    ),
                    "    </div>",
                ]
            )
        lines.append("  </div>")
    lines.extend(["</body>", "</html>"])
    index_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

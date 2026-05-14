from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

import torch

from smolvla_moe.data import VLABatch


class JsonlLogger:
    def __init__(self, output_dir: Path) -> None:
        self.path = output_dir / "metrics.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, step: int, payload: dict[str, Any]) -> None:
        record = {"time": time.time(), "event": event, "step": int(step), **payload}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


class WandbLogger:
    def __init__(self, config: dict[str, Any], output_dir: Path, rank: int) -> None:
        self.run = None
        self.wandb = None
        self.enabled = False
        self.output_dir = output_dir
        self.config = config
        self.wandb_config = config.get("train", {}).get("wandb", {})
        if rank != 0 or not bool(self.wandb_config.get("enabled", False)):
            return

        try:
            import wandb
        except ImportError as exc:
            raise ImportError("wandb logging is enabled, but wandb is not installed.") from exc

        self.wandb = wandb
        self.enabled = True
        self.output_dir.mkdir(parents=True, exist_ok=True)
        init_kwargs = self._init_kwargs()
        self.run = wandb.init(**init_kwargs)
        self._write_run_id()
        self._define_metrics()
        if bool(self.wandb_config.get("log_code", False)):
            repo_root = Path(__file__).resolve().parents[3]
            self.run.log_code(str(repo_root))

    @property
    def url(self) -> str | None:
        if self.run is None:
            return None
        return getattr(self.run, "url", None)

    def log(self, payload: dict[str, Any], step: int) -> None:
        if self.run is None:
            return
        self.run.log({"train/step": int(step), **payload}, step=step)

    def log_first_batch(self, batch: VLABatch, step: int = 0) -> None:
        if self.run is None or not bool(self.wandb_config.get("log_first_batch", False)):
            return
        max_images = int(self.wandb_config.get("log_first_batch_max_images", 4))
        images = []
        for sample_idx in range(min(max_images, batch.images.shape[0])):
            sample = batch.images[sample_idx].detach().float().cpu().clamp(0, 1)
            tiled = torch.cat([camera for camera in sample], dim=2)
            tiled = (tiled.permute(1, 2, 0).numpy() * 255).astype("uint8")
            caption = None if batch.language is None else batch.language[sample_idx]
            images.append(self.wandb.Image(tiled, caption=caption))
        if images:
            self.run.log({"train/camera_views": images}, step=step)

    def log_checkpoint(self, checkpoint_path: Path, step: int, aliases: list[str] | None = None) -> None:
        if self.run is None or not bool(self.wandb_config.get("log_checkpoints", False)):
            return
        artifact = self.wandb.Artifact(name=f"{self.run.name}-checkpoint", type="model")
        artifact.add_file(str(checkpoint_path))
        self.run.log_artifact(artifact, aliases=aliases or [f"step-{step}"])

    def finish(self) -> None:
        if self.run is None:
            return
        self.run.finish()
        self.run = None

    def _init_kwargs(self) -> dict[str, Any]:
        group = self.wandb_config.get("group")
        group = None if group in (None, "", "null") else str(group)
        workspace = self.wandb_config.get("workspace", self.wandb_config.get("entity"))
        workspace = None if workspace in (None, "", "null") else str(workspace)
        mode = str(self.wandb_config.get("mode", "online"))
        run_id = self._resume_run_id()
        kwargs: dict[str, Any] = {
            "entity": workspace,
            "project": str(self.wandb_config.get("project", "smolvla-moe")),
            "name": str(self.wandb_config.get("name", "train")),
            "group": group,
            "mode": mode,
            "dir": str(self.wandb_config.get("dir") or self.output_dir),
            "tags": list(self.wandb_config.get("tags", [])),
            "notes": self.wandb_config.get("notes"),
            "job_type": str(self.wandb_config.get("job_type", "train")),
            "config": self.config,
        }
        if run_id is not None:
            kwargs["id"] = run_id
            kwargs["resume"] = str(self.wandb_config.get("resume", "allow"))
        return kwargs

    def _resume_run_id(self) -> str | None:
        resume = str(self.wandb_config.get("resume", "allow"))
        if resume in {"never", "false", "False"}:
            return None
        id_path = self.output_dir / str(self.wandb_config.get("id_file", "wandb_id.txt"))
        if id_path.exists():
            return id_path.read_text(encoding="utf-8").strip()
        return None

    def _write_run_id(self) -> None:
        if self.run is None:
            return
        id_path = self.output_dir / str(self.wandb_config.get("id_file", "wandb_id.txt"))
        id_path.write_text(str(self.run.id), encoding="utf-8")

    def _define_metrics(self) -> None:
        if self.run is None:
            return
        self.wandb.define_metric("train/step")
        self.wandb.define_metric("train/*", step_metric="train/step")
        self.wandb.define_metric("performance/*", step_metric="train/step")
        self.wandb.define_metric("resource/*", step_metric="train/step")
        self.wandb.define_metric("moe/*", step_metric="train/step")


def collect_resource_metrics() -> dict[str, float]:
    metrics: dict[str, float] = {}
    metrics.update(_collect_gpu_metrics())
    metrics.update(_collect_cpu_metrics())
    return metrics


def _collect_gpu_metrics() -> dict[str, float]:
    if shutil.which("nvidia-smi") is None:
        return {}
    query = "index,memory.used,memory.free,utilization.gpu,power.draw"
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {}
    metrics: dict[str, float] = {}
    memory_used = []
    utilization = []
    power = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        index = int(parts[0])
        used_gib = float(parts[1]) / 1024.0
        free_gib = float(parts[2]) / 1024.0
        util_pct = float(parts[3])
        power_w = float(parts[4])
        metrics[f"resource/gpu{index}_memory_used_gib"] = used_gib
        metrics[f"resource/gpu{index}_memory_free_gib"] = free_gib
        metrics[f"resource/gpu{index}_util_percent"] = util_pct
        metrics[f"resource/gpu{index}_power_w"] = power_w
        memory_used.append(used_gib)
        utilization.append(util_pct)
        power.append(power_w)
    if memory_used:
        metrics["resource/gpu_memory_used_gib_max"] = max(memory_used)
        metrics["resource/gpu_util_percent_mean"] = sum(utilization) / len(utilization)
        metrics["resource/gpu_power_w_sum"] = sum(power)
    return metrics


def _collect_cpu_metrics() -> dict[str, float]:
    try:
        import psutil
    except ImportError:
        return {}
    virtual = psutil.virtual_memory()
    return {
        "resource/cpu_percent": float(psutil.cpu_percent(interval=None)),
        "resource/ram_used_gib": float(virtual.used / 1024**3),
        "resource/ram_percent": float(virtual.percent),
    }

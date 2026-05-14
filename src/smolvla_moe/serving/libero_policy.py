from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

from smolvla_moe.config import load_config
from smolvla_moe.data.batch import VLABatch
from smolvla_moe.data.collate import VLACollator
from smolvla_moe.data.lerobot_dataset import _stats_tensors
from smolvla_moe.models.policy import SmolVLAMoEPolicy


class LiberoSmolVLAMoEPolicy:
    """OpenPI-compatible LIBERO policy wrapper for SmolVLA-MoE checkpoints."""

    def __init__(
        self,
        checkpoint: str | Path,
        config_path: str | Path | None = None,
        device: str = "cuda",
        amp_dtype: str | None = "bfloat16",
        seed: int | None = None,
        binarize_gripper: bool = True,
        clip_actions: bool = True,
        use_flash: bool = False,
        latency_log: str | Path | None = None,
    ) -> None:
        payload = torch.load(checkpoint, map_location="cpu")
        self.config = load_config(config_path) if config_path is not None else payload["config"]
        self.dataset_config = self.config["dataset"]
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.amp_dtype = _amp_dtype(amp_dtype)
        self.generator = torch.Generator(device=self.device)
        if seed is not None:
            self.generator.manual_seed(int(seed))
        self.binarize_gripper = bool(binarize_gripper)
        self.clip_actions = bool(clip_actions)
        self.use_flash = bool(use_flash)
        self.latency_log = Path(latency_log) if latency_log is not None else None
        self.infer_count = 0
        if self.latency_log is not None:
            self.latency_log.parent.mkdir(parents=True, exist_ok=True)
            self.latency_log.write_text("", encoding="utf-8")

        self.policy = SmolVLAMoEPolicy(self.config).to(self.device)
        missing, unexpected = self.policy.load_state_dict(payload["model"], strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Checkpoint state mismatch. missing={missing} unexpected={unexpected}")
        self.policy.eval()
        self.collator = VLACollator(self.config)

        stats = _load_dataset_stats(self.dataset_config)
        self.state_mean, self.state_std = _stats_tensors(stats, str(self.dataset_config["state_key"]))
        self.action_mean, self.action_std = _stats_tensors(stats, str(self.dataset_config["action_key"]))
        self.normalize_state = bool(self.dataset_config.get("normalize_state", True))
        self.normalize_actions = bool(self.dataset_config.get("normalize_actions", True))
        self.image_size = int(self.dataset_config.get("image_size", 256))
        self.action_dim = int(self.dataset_config.get("action_dim", 7))
        self.state_dim = int(self.dataset_config.get("state_dim", 8))

    @property
    def metadata(self) -> dict[str, Any]:
        model_config = self.config["model"]
        return {
            "policy": "smolvla-moe",
            "horizon": int(model_config["action_decoder"]["horizon"]),
            "action_dim": int(model_config["action_decoder"]["action_dim"]),
            "flow_inference_steps": int(model_config.get("flow", {}).get("inference_steps", 4)),
            "flash_enabled": bool(model_config.get("flash", {}).get("enabled", False)),
            "flash_runtime": self.use_flash,
        }

    @torch.no_grad()
    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        total_start = time.perf_counter()
        batch = self._obs_to_batch(obs).to(self.device)
        self._sync_device()
        preprocess_ms = (time.perf_counter() - total_start) * 1000.0
        policy_start = time.perf_counter()
        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
            actions = (
                self.policy.predict_action_flash(batch, generator=self.generator)
                if self.use_flash
                else self.policy.predict_action(batch, generator=self.generator)
            )
        self._sync_device()
        policy_ms = (time.perf_counter() - policy_start) * 1000.0
        postprocess_start = time.perf_counter()
        actions = actions[0].float().cpu()
        if self.normalize_actions:
            actions = _denormalize(actions, self.action_mean, self.action_std)
        actions_np = actions.numpy().astype(np.float32)
        output = {"actions": _postprocess_libero_actions(actions_np, self.binarize_gripper, self.clip_actions)}
        postprocess_ms = (time.perf_counter() - postprocess_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0
        self._log_latency(preprocess_ms, policy_ms, postprocess_ms, total_ms)
        return output

    def reset(self) -> None:
        return None

    def _obs_to_batch(self, obs: dict[str, Any]) -> VLABatch:
        image = _hwc_to_chw_float(obs["observation/image"], self.image_size)
        wrist_image = _hwc_to_chw_float(obs["observation/wrist_image"], self.image_size)
        state = torch.as_tensor(np.array(obs["observation/state"], dtype=np.float32, copy=True)).reshape(-1)[: self.state_dim]
        if self.normalize_state:
            state = _normalize(state, self.state_mean, self.state_std)
        sample = {
            "images": torch.stack([image, wrist_image], dim=0),
            "state": state,
            "actions": None,
            "action_mask": None,
            "language": str(obs.get("prompt", "")),
        }
        return self.collator([sample])

    def _sync_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _log_latency(self, preprocess_ms: float, policy_ms: float, postprocess_ms: float, total_ms: float) -> None:
        if self.latency_log is None:
            return
        self.infer_count += 1
        record = {
            "call": self.infer_count,
            "mode": "flash" if self.use_flash else "full",
            "preprocess_ms": float(preprocess_ms),
            "policy_ms": float(policy_ms),
            "postprocess_ms": float(postprocess_ms),
            "total_ms": float(total_ms),
        }
        with self.latency_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def _load_dataset_stats(dataset_config: dict[str, Any]) -> dict[str, Any]:
    repo_id = str(dataset_config["repo_id"])
    root = dataset_config.get("local_path")
    metadata = LeRobotDatasetMetadata(repo_id, root=root)
    return getattr(metadata, "stats", {})


def _hwc_to_chw_float(image: Any, image_size: int) -> torch.Tensor:
    array = np.array(image, copy=True)
    if array.ndim != 3:
        raise ValueError(f"Expected image rank 3, got {array.shape}")
    tensor = torch.as_tensor(array)
    if tensor.shape[0] in {1, 3}:
        chw = tensor.float()
    else:
        chw = tensor.permute(2, 0, 1).float()
    if chw.max() > 2:
        chw = chw / 255.0
    return F.interpolate(chw.unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False)[0]


def _normalize(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (value - mean.to(dtype=value.dtype)) / std.to(dtype=value.dtype)


def _denormalize(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return value * std.to(dtype=value.dtype) + mean.to(dtype=value.dtype)


def _postprocess_libero_actions(actions: np.ndarray, binarize_gripper: bool, clip_actions: bool) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32).copy()
    if clip_actions:
        actions[..., :6] = np.clip(actions[..., :6], -1.0, 1.0)
        actions[..., -1] = np.clip(actions[..., -1], -1.0, 1.0)
    if binarize_gripper:
        actions[..., -1] = np.where(actions[..., -1] >= 0.0, 1.0, -1.0)
    return actions


def _amp_dtype(name: str | None) -> torch.dtype | None:
    if name is None or str(name) in {"none", "no", "float32"}:
        return None
    if str(name) in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if str(name) in {"float16", "fp16"}:
        return torch.float16
    raise ValueError(f"Unsupported amp dtype: {name}")

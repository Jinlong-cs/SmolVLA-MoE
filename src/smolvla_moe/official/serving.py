from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from lerobot.utils.constants import ACTION
from lerobot.utils.constants import OBS_STATE

from smolvla_moe.config import load_config
from smolvla_moe.official.data import OfficialSmolVLACollator
from smolvla_moe.official.policy import build_official_smolvla_moe_policy
from smolvla_moe.official.stats import load_official_smolvla_stats
from smolvla_moe.official.stats import official_stats_tensors


class LiberoOfficialSmolVLAMoEPolicy:
    """OpenPI-compatible server wrapper for official SmolVLA + residual MoE checkpoints."""

    def __init__(
        self,
        checkpoint: str | Path,
        config_path: str | Path | None = None,
        device: str = "cuda",
        amp_dtype: str | None = "bfloat16",
        seed: int | None = None,
        binarize_gripper: bool = True,
        clip_actions: bool = True,
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

        self.policy = build_official_smolvla_moe_policy(self.config).to(self.device)
        missing, unexpected = self.policy.load_state_dict(payload["model"], strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Checkpoint state mismatch. missing={missing} unexpected={unexpected}")
        self.policy.eval()
        self.collator = OfficialSmolVLACollator(self.config)

        stats = official_stats_tensors(load_official_smolvla_stats(self.config))
        self.state_mean, self.state_std = stats[OBS_STATE]
        self.action_mean, self.action_std = stats[ACTION]
        self.normalize_state = bool(self.dataset_config.get("normalize_state", True))
        self.normalize_actions = bool(self.dataset_config.get("normalize_actions", True))
        self.image_size = int(self.dataset_config.get("image_size", 256))
        self.action_dim = int(self.dataset_config.get("action_dim", 7))
        self.state_dim = int(self.dataset_config.get("state_dim", 8))

    @property
    def metadata(self) -> dict[str, Any]:
        cfg = self.policy.policy.config
        return {
            "policy": "official-smolvla-moe",
            "horizon": int(cfg.chunk_size),
            "action_dim": self.action_dim,
            "flow_inference_steps": int(cfg.num_steps),
            "n_action_steps": int(cfg.n_action_steps),
        }

    @torch.no_grad()
    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        batch = _move_batch(self._obs_to_batch(obs), self.device)
        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
            actions = self.policy.predict_action_chunk(batch)
        actions = actions[0].float().cpu()
        if self.normalize_actions:
            actions = _denormalize(actions, self.action_mean, self.action_std)
        actions_np = actions.numpy().astype(np.float32)
        return {"actions": _postprocess_libero_actions(actions_np, self.binarize_gripper, self.clip_actions)}

    def reset(self) -> None:
        self.policy.policy.reset()

    def _obs_to_batch(self, obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        image = _hwc_to_chw_float(obs["observation/image"], self.image_size)
        wrist_image = _hwc_to_chw_float(obs["observation/wrist_image"], self.image_size)
        state = torch.as_tensor(np.array(obs["observation/state"], dtype=np.float32, copy=True)).reshape(-1)[
            : self.state_dim
        ]
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


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


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

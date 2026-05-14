from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class LeRobotVLADataset(Dataset):
    """Thin VLA adapter around Hugging Face LeRobotDataset.

    The adapter keeps benchmark-specific key mapping in config. It returns plain dict samples so the collator can
    choose either the smoke hash tokenizer or the production Hugging Face processor.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
            from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
        except ImportError as exc:
            raise ImportError("Install the `train` dependencies and LeRobot to use LeRobotVLADataset.") from exc

        repo_id = str(config["repo_id"])
        self.image_keys = list(config["image_keys"])
        self.state_key = str(config["state_key"])
        self.action_key = str(config["action_key"])
        self.language_key = str(config.get("language_key", "task"))
        self.horizon = int(config["horizon"])
        self.action_dim = int(config["action_dim"])
        self.state_dim = int(config.get("state_dim", 0))
        self.image_size = int(config.get("image_size", 224))

        root = config.get("local_path")
        metadata = LeRobotDatasetMetadata(repo_id, root=root)
        self.tasks = getattr(metadata, "tasks", {})
        self.dataset = LeRobotDataset(
            repo_id,
            root=root,
            delta_timestamps={self.action_key: [t / metadata.fps for t in range(self.horizon)]},
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.dataset[index]
        images = torch.stack([_to_chw_float(row[key], self.image_size) for key in self.image_keys], dim=0)
        state = _as_float_tensor(row[self.state_key]).reshape(-1)[: self.state_dim]
        actions = _as_float_tensor(row[self.action_key]).reshape(self.horizon, self.action_dim)
        language = self._language(row)
        return {"images": images, "state": state, "actions": actions, "language": language}

    def _language(self, row: dict[str, Any]) -> str:
        value = row.get(self.language_key, "")
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            value = int(value.item())
        if isinstance(value, int):
            return str(self.tasks.get(value, value))
        return str(value)


def _as_float_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.float()
    return torch.as_tensor(value, dtype=torch.float32)


def _to_chw_float(value: Any, image_size: int) -> torch.Tensor:
    if torch.is_tensor(value):
        image = value
    else:
        image = torch.as_tensor(value)
    if image.ndim != 3:
        raise ValueError(f"image must be rank 3, got {tuple(image.shape)}")
    if image.shape[0] not in {1, 3}:
        image = image.permute(2, 0, 1)
    image = image.float()
    if image.max() > 2:
        image = image / 255.0
    image = F.interpolate(image.unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False)[0]
    return image

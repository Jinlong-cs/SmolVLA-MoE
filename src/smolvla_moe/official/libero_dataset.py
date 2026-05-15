from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from io import BytesIO
import json
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from lerobot.utils.constants import ACTION
from lerobot.utils.constants import OBS_STATE

from smolvla_moe.official.stats import official_stats_tensors


class LiberoParquetVLADataset(Dataset):
    """Direct reader for the LIBERO LeRobot v2 parquet dataset used by the official checkpoint."""

    def __init__(self, config: dict[str, Any], stats: dict[str, Any]) -> None:
        self.repo_id = str(config["repo_id"])
        self.revision = config.get("revision")
        self.local_path = Path(str(config["local_path"])).expanduser() if config.get("local_path") else None
        self.image_keys = list(config["image_keys"])
        self.state_key = str(config["state_key"])
        self.action_key = str(config["action_key"])
        self.horizon = int(config["horizon"])
        self.action_dim = int(config["action_dim"])
        self.state_dim = int(config["state_dim"])
        self.image_size = int(config.get("image_size", 256))
        self.normalize_actions = bool(config.get("normalize_actions", True))
        self.normalize_state = bool(config.get("normalize_state", True))
        self.episode_cache_size = int(config.get("episode_cache_size", 8))

        self.info = self._load_json("meta/info.json")
        self.tasks = self._load_tasks("meta/tasks.jsonl")
        self.episodes = self._load_jsonl("meta/episodes.jsonl")
        self.offsets = _episode_offsets(self.episodes)
        self.total_frames = self.offsets[-1]
        tensors = official_stats_tensors(stats)
        self.action_mean, self.action_std = tensors[ACTION]
        self.state_mean, self.state_std = tensors[OBS_STATE]
        self._episode_cache: OrderedDict[int, dict[str, Any]] = OrderedDict()

    def __len__(self) -> int:
        return self.total_frames

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_index, frame_index = self._locate(index)
        episode = self._episode(episode_index)
        images = torch.stack([_image_to_chw(episode[key][frame_index], self.image_size) for key in self.image_keys])
        state = torch.as_tensor(episode[self.state_key][frame_index], dtype=torch.float32).reshape(-1)[: self.state_dim]
        actions, action_mask = self._action_chunk(episode, frame_index)
        if self.normalize_state:
            state = _normalize(state, self.state_mean, self.state_std)
        if self.normalize_actions:
            actions = _normalize(actions, self.action_mean, self.action_std)
        task_index = int(episode["task_index"][frame_index])
        return {
            "images": images,
            "state": state,
            "actions": actions,
            "action_mask": action_mask,
            "language": self.tasks[task_index],
        }

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0 or index >= self.total_frames:
            raise IndexError(index)
        episode_pos = bisect_right(self.offsets, index) - 1
        return int(self.episodes[episode_pos]["episode_index"]), int(index - self.offsets[episode_pos])

    def _episode(self, episode_index: int) -> dict[str, Any]:
        cached = self._episode_cache.get(episode_index)
        if cached is not None:
            self._episode_cache.move_to_end(episode_index)
            return cached
        path = self._episode_path(episode_index)
        columns = [*self.image_keys, self.state_key, self.action_key, "task_index"]
        table = pq.ParquetFile(path).read(columns=columns)
        episode = {column: table[column].to_pylist() for column in columns}
        self._episode_cache[episode_index] = episode
        while len(self._episode_cache) > self.episode_cache_size:
            self._episode_cache.popitem(last=False)
        return episode

    def _episode_path(self, episode_index: int) -> str:
        data_path = str(self.info["data_path"]).format(
            episode_chunk=episode_index // int(self.info["chunks_size"]),
            episode_index=episode_index,
        )
        if self.local_path is not None:
            return str(self.local_path / data_path)
        return hf_hub_download(self.repo_id, data_path, repo_type="dataset", revision=self.revision)

    def _action_chunk(self, episode: dict[str, Any], frame_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        length = len(episode[self.action_key])
        last_index = length - 1
        actions = []
        mask = []
        for step in range(self.horizon):
            src_index = frame_index + step
            valid = src_index < length
            actions.append(episode[self.action_key][src_index if valid else last_index])
            mask.append(valid)
        action_tensor = torch.as_tensor(np.asarray(actions, dtype=np.float32)).reshape(self.horizon, self.action_dim)
        return action_tensor, torch.as_tensor(mask, dtype=torch.bool)

    def _load_json(self, path: str) -> dict[str, Any]:
        with open(self._meta_path(path), encoding="utf-8") as handle:
            return json.load(handle)

    def _load_jsonl(self, path: str) -> list[dict[str, Any]]:
        with open(self._meta_path(path), encoding="utf-8") as handle:
            return [json.loads(line) for line in handle]

    def _load_tasks(self, path: str) -> dict[int, str]:
        rows = self._load_jsonl(path)
        return {int(row["task_index"]): str(row["task"]) for row in rows}

    def _meta_path(self, path: str) -> str:
        if self.local_path is not None:
            return str(self.local_path / path)
        return hf_hub_download(self.repo_id, path, repo_type="dataset", revision=self.revision)


def _episode_offsets(episodes: list[dict[str, Any]]) -> list[int]:
    offsets = [0]
    for episode in episodes:
        offsets.append(offsets[-1] + int(episode["length"]))
    return offsets


def _image_to_chw(value: Any, image_size: int) -> torch.Tensor:
    if isinstance(value, dict):
        value = Image.open(BytesIO(value["bytes"])).convert("RGB")
    array = np.array(value, dtype=np.uint8, copy=True) if isinstance(value, Image.Image) else np.array(value, copy=True)
    if array.ndim != 3:
        raise ValueError(f"image must be rank 3, got {array.shape}")
    tensor = torch.as_tensor(array)
    if tensor.shape[0] not in {1, 3}:
        tensor = tensor.permute(2, 0, 1)
    tensor = tensor.float()
    if tensor.max() > 2:
        tensor = tensor / 255.0
    return F.interpolate(tensor.unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False)[0]


def _normalize(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (value - mean.to(dtype=value.dtype)) / std.to(dtype=value.dtype)

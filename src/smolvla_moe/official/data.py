from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from transformers import AutoProcessor

from lerobot.utils.constants import ACTION
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS
from lerobot.utils.constants import OBS_STATE

from smolvla_moe.data.lerobot_dataset import LeRobotVLADataset


class OfficialSmolVLACollator:
    def __init__(self, config: dict[str, Any]) -> None:
        official_config = config["official_smolvla"]
        self.image_feature_keys = list(official_config["image_feature_keys"])
        self.vlm_model_name = str(official_config["vlm_model_name"])
        self.tokenizer_max_length = int(official_config.get("tokenizer_max_length", 48))
        self.pad_language_to = str(official_config.get("pad_language_to", "longest"))
        processor = AutoProcessor.from_pretrained(self.vlm_model_name)
        self.tokenizer = processor.tokenizer

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        images = torch.stack([sample["images"] for sample in samples], dim=0)
        if images.shape[1] != len(self.image_feature_keys):
            raise ValueError(
                f"Expected {len(self.image_feature_keys)} image views, got {images.shape[1]} from dataset"
            )

        batch: dict[str, torch.Tensor] = {}
        for view_idx, feature_key in enumerate(self.image_feature_keys):
            batch[feature_key] = images[:, view_idx]
        batch[OBS_STATE] = torch.stack([sample["state"] for sample in samples], dim=0)
        if samples[0].get("actions") is not None:
            batch[ACTION] = torch.stack([sample["actions"] for sample in samples], dim=0)
        if samples[0].get("action_mask") is not None:
            batch["action_is_pad"] = ~torch.stack([sample["action_mask"] for sample in samples], dim=0).bool()

        language = [_with_newline(str(sample.get("language", ""))) for sample in samples]
        tokenized = self.tokenizer(
            language,
            padding=self.pad_language_to,
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )
        batch[OBS_LANGUAGE_TOKENS] = tokenized["input_ids"]
        batch[OBS_LANGUAGE_ATTENTION_MASK] = tokenized["attention_mask"].bool()
        return batch


def build_official_train_data(config: dict[str, Any], rank: int = 0, world_size: int = 1) -> Any:
    dataset_config = config["dataset"]
    dataset_type = str(dataset_config.get("type", "lerobot"))
    if dataset_type != "lerobot":
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    dataset = LeRobotVLADataset(dataset_config)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("train", {}).get("batch_size", 1)),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(dataset_config.get("num_workers", 4)),
        pin_memory=True,
        collate_fn=OfficialSmolVLACollator(config),
        drop_last=True,
    )
    if sampler is not None:
        loader.sampler_for_epoch = sampler
    return loader


def _with_newline(value: str) -> str:
    return value if value.endswith("\n") else f"{value}\n"

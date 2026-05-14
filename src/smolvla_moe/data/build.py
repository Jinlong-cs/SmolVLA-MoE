from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler

from smolvla_moe.data.collate import VLACollator
from smolvla_moe.data.lerobot_dataset import LeRobotVLADataset


def build_train_data(config: dict[str, Any], rank: int = 0, world_size: int = 1) -> Any:
    dataset_config = config["dataset"]
    dataset_type = str(dataset_config.get("type", "lerobot"))
    if dataset_type == "lerobot":
        dataset = LeRobotVLADataset(dataset_config)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
        loader = DataLoader(
            dataset,
            batch_size=int(config.get("train", {}).get("batch_size", 1)),
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=int(dataset_config.get("num_workers", 4)),
            pin_memory=True,
            collate_fn=VLACollator(config),
            drop_last=True,
        )
        if sampler is not None:
            loader.sampler_for_epoch = sampler
        return loader
    raise ValueError(f"Unsupported dataset type: {dataset_type}")

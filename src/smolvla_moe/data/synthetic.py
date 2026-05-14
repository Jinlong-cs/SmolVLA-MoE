from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch

from smolvla_moe.data.batch import VLABatch


class SyntheticVLADataset:
    """Batch-yielding synthetic data source for smoke tests."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.batch_size = int(config.get("batch_size", 2))
        self.num_batches = int(config.get("num_batches", 4))
        self.num_cameras = int(config.get("num_cameras", 2))
        self.image_size = int(config.get("image_size", 64))
        self.text_len = int(config.get("text_len", 16))
        self.vocab_size = int(config.get("vocab_size", 32768))
        self.state_dim = int(config.get("state_dim", 8))
        self.action_dim = int(config.get("action_dim", 7))
        self.horizon = int(config.get("horizon", 8))

    def __iter__(self) -> Iterator[VLABatch]:
        for _ in range(self.num_batches):
            yield VLABatch(
                images=torch.randn(self.batch_size, self.num_cameras, 3, self.image_size, self.image_size),
                input_ids=torch.randint(0, self.vocab_size, (self.batch_size, self.text_len)),
                attention_mask=torch.ones(self.batch_size, self.text_len, dtype=torch.long),
                state=torch.randn(self.batch_size, self.state_dim),
                actions=torch.randn(self.batch_size, self.horizon, self.action_dim),
                language=["synthetic instruction"] * self.batch_size,
            )

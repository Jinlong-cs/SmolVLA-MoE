from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class VLABatch:
    images: torch.Tensor
    input_ids: torch.Tensor | None
    attention_mask: torch.Tensor | None
    state: torch.Tensor | None
    actions: torch.Tensor | None
    action_mask: torch.Tensor | None = None
    language: list[str] | None = None
    extras: dict[str, Any] | None = None

    def to(self, device: torch.device | str) -> "VLABatch":
        return VLABatch(
            images=self.images.to(device),
            input_ids=None if self.input_ids is None else self.input_ids.to(device),
            attention_mask=None if self.attention_mask is None else self.attention_mask.to(device),
            state=None if self.state is None else self.state.to(device),
            actions=None if self.actions is None else self.actions.to(device),
            action_mask=None if self.action_mask is None else self.action_mask.to(device),
            language=self.language,
            extras=_move_extras(self.extras, device),
        )


def _move_extras(extras: dict[str, Any] | None, device: torch.device | str) -> dict[str, Any] | None:
    if extras is None:
        return None
    moved: dict[str, Any] = {}
    for key, value in extras.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        elif isinstance(value, dict):
            moved[key] = _move_extras(value, device)
        elif hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved

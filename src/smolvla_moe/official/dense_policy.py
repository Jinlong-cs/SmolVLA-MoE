from __future__ import annotations

from typing import Any

import torch
from torch import nn

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


class OfficialDenseSmolVLAPolicy(nn.Module):
    def __init__(self, policy: SmolVLAPolicy) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        loss, output_dict = self.policy.forward(batch)
        metrics = {"loss": loss, "flow_loss": loss}
        if output_dict:
            for key, value in output_dict.items():
                if torch.is_tensor(value) and value.ndim == 0:
                    metrics[f"official_{key}"] = value
        return metrics

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.policy.select_action(batch)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.policy.predict_action_chunk(batch)


def build_official_dense_smolvla_policy(config: dict[str, Any]) -> OfficialDenseSmolVLAPolicy:
    official_config = config["official_smolvla"]
    checkpoint = str(official_config.get("checkpoint", "HuggingFaceVLA/smolvla_libero"))
    policy = SmolVLAPolicy.from_pretrained(checkpoint)

    policy_overrides = official_config.get("policy_overrides", {})
    for key, value in policy_overrides.items():
        if not hasattr(policy.config, key):
            raise ValueError(f"Official SmolVLA config has no field {key!r}")
        setattr(policy.config, key, value)
    policy.reset()

    freeze_backbone = bool(official_config.get("freeze_backbone", False))
    if freeze_backbone:
        _freeze_backbone(policy)
    return OfficialDenseSmolVLAPolicy(policy=policy)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    total = 0
    for param in model.parameters():
        if trainable_only and not param.requires_grad:
            continue
        total += param.numel()
    return total


def _freeze_backbone(policy: SmolVLAPolicy) -> None:
    model = policy.model
    for param in model.vlm_with_expert.vlm.parameters():
        param.requires_grad = False

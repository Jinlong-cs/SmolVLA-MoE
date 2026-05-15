from __future__ import annotations

from typing import Any

import torch
from torch import nn

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from smolvla_moe.official.moe import OfficialMoEConfig
from smolvla_moe.official.moe import apply_residual_moe_adapters
from smolvla_moe.official.moe import clear_moe_aux
from smolvla_moe.official.moe import collect_moe_aux
from smolvla_moe.official.moe import set_official_moe_trainability


class OfficialSmolVLAMoEPolicy(nn.Module):
    def __init__(self, policy: SmolVLAPolicy, patched_layers: list[str]) -> None:
        super().__init__()
        self.policy = policy
        self.patched_layers = patched_layers

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        clear_moe_aux(self.policy)
        base_loss, output_dict = self.policy.forward(batch)
        aux = collect_moe_aux(self.policy)
        loss = base_loss + aux["load_balance_loss"] + aux["router_z_loss"]
        metrics = {
            "loss": loss,
            "flow_loss": base_loss,
            "load_balance_loss": aux["load_balance_loss"],
            "router_z_loss": aux["router_z_loss"],
            "expert_usage": aux["expert_usage"],
        }
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


def build_official_smolvla_moe_policy(config: dict[str, Any]) -> OfficialSmolVLAMoEPolicy:
    official_config = config["official_smolvla"]
    checkpoint = str(official_config.get("checkpoint", "HuggingFaceVLA/smolvla_libero"))
    policy = SmolVLAPolicy.from_pretrained(checkpoint)

    policy_overrides = official_config.get("policy_overrides", {})
    for key, value in policy_overrides.items():
        if not hasattr(policy.config, key):
            raise ValueError(f"Official SmolVLA config has no field {key!r}")
        setattr(policy.config, key, value)
    policy.reset()

    moe_config = OfficialMoEConfig.from_dict(config["model"]["official_moe"])
    patched_layers = apply_residual_moe_adapters(policy, moe_config)
    set_official_moe_trainability(
        policy,
        freeze_official_dense=bool(official_config.get("freeze_official_dense", True)),
    )
    return OfficialSmolVLAMoEPolicy(policy=policy, patched_layers=patched_layers)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    total = 0
    for param in model.parameters():
        if trainable_only and not param.requires_grad:
            continue
        total += param.numel()
    return total

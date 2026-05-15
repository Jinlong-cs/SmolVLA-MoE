from __future__ import annotations

from typing import Any

import torch
from torch import nn

from lerobot.configs.types import FeatureType
from lerobot.configs.types import PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import ACTION
from lerobot.utils.constants import OBS_STATE


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
    policy_config = _build_policy_config(config) if bool(official_config.get("override_checkpoint_config", False)) else None
    policy = SmolVLAPolicy.from_pretrained(
        checkpoint,
        config=policy_config,
        strict=bool(official_config.get("checkpoint_strict", False)),
    )

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


def _build_policy_config(config: dict[str, Any]) -> SmolVLAConfig:
    official_config = config["official_smolvla"]
    dataset_config = config["dataset"]
    image_size = int(dataset_config.get("image_size", 256))
    image_features = {
        key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, image_size, image_size))
        for key in official_config["image_feature_keys"]
    }
    input_features = {
        **image_features,
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(int(dataset_config["state_dim"]),)),
    }
    output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(int(dataset_config["action_dim"]),)),
    }
    policy_overrides = dict(official_config.get("policy_overrides", {}))
    config_kwargs: dict[str, Any] = {
        "input_features": input_features,
        "output_features": output_features,
        "chunk_size": int(dataset_config["horizon"]),
        "n_action_steps": int(policy_overrides.get("n_action_steps", dataset_config["horizon"])),
        "num_steps": int(policy_overrides.get("num_steps", 10)),
        "vlm_model_name": str(official_config["vlm_model_name"]),
        "load_vlm_weights": bool(official_config.get("load_vlm_weights", True)),
        "device": str(config.get("device", "cpu")),
    }
    config_kwargs.update(dict(official_config.get("architecture_overrides", {})))
    return SmolVLAConfig(**config_kwargs)


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

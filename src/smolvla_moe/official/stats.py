from __future__ import annotations

from typing import Any

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import ACTION
from lerobot.utils.constants import OBS_STATE


def load_official_smolvla_stats(config: dict[str, Any]) -> dict[str, Any]:
    checkpoint = str(config["official_smolvla"].get("checkpoint", "HuggingFaceVLA/smolvla_libero"))
    policy_config = PreTrainedConfig.from_pretrained(checkpoint)
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg=policy_config, pretrained_path=checkpoint)
    for step in list(preprocessor.steps) + list(postprocessor.steps):
        stats = getattr(step, "stats", None)
        if stats is not None and ACTION in stats and OBS_STATE in stats:
            return stats
    raise RuntimeError(f"Official SmolVLA checkpoint {checkpoint!r} does not expose action/state stats")


def official_stats_tensors(stats: dict[str, Any]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    return {
        ACTION: _stats_tensors(stats, ACTION),
        OBS_STATE: _stats_tensors(stats, OBS_STATE),
    }


def _stats_tensors(stats: dict[str, Any], key: str) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.as_tensor(np.asarray(stats[key]["mean"]), dtype=torch.float32).reshape(-1)
    std = torch.as_tensor(np.asarray(stats[key]["std"]), dtype=torch.float32).reshape(-1).clamp_min(1e-6)
    return mean, std

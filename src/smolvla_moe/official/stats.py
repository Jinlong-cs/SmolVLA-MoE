from __future__ import annotations

from typing import Any

from huggingface_hub import hf_hub_download
import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import ACTION
from lerobot.utils.constants import OBS_STATE


def load_official_smolvla_stats(config: dict[str, Any]) -> dict[str, Any]:
    stats_source = str(config["official_smolvla"].get("stats_source", "checkpoint"))
    if stats_source == "dataset":
        return _load_dataset_stats(config["dataset"])
    if stats_source != "checkpoint":
        raise ValueError(f"Unsupported SmolVLA stats source: {stats_source}")

    checkpoint = str(
        config["official_smolvla"].get(
            "stats_checkpoint",
            config["official_smolvla"].get("checkpoint", "HuggingFaceVLA/smolvla_libero"),
        )
    )
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


def _load_dataset_stats(dataset_config: dict[str, Any]) -> dict[str, Any]:
    if dataset_config.get("local_path"):
        stats_path = f"{dataset_config['local_path']}/meta/stats.json"
    else:
        stats_path = hf_hub_download(
            str(dataset_config["repo_id"]),
            "meta/stats.json",
            repo_type="dataset",
            revision=dataset_config.get("revision"),
        )

    import json

    with open(stats_path, encoding="utf-8") as handle:
        raw_stats = json.load(handle)
    state_key = str(dataset_config["state_key"])
    action_key = str(dataset_config["action_key"])
    return {
        OBS_STATE: raw_stats[state_key],
        ACTION: raw_stats[action_key],
    }

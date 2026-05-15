from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class OfficialMoEConfig:
    num_experts: int = 4
    top_k: int = 1
    adapter_dim: int = 256
    init_scale: float = 0.0
    load_balance_weight: float = 0.01
    router_z_loss_weight: float = 0.0001
    layer_start: int = 0
    layer_stride: int = 1

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "OfficialMoEConfig":
        return cls(
            num_experts=int(config.get("num_experts", 4)),
            top_k=int(config.get("top_k", 1)),
            adapter_dim=int(config.get("adapter_dim", 256)),
            init_scale=float(config.get("init_scale", 0.0)),
            load_balance_weight=float(config.get("load_balance_weight", 0.01)),
            router_z_loss_weight=float(config.get("router_z_loss_weight", 0.0001)),
            layer_start=int(config.get("layer_start", 0)),
            layer_stride=int(config.get("layer_stride", 1)),
        )


class LowRankSwiGLUExpert(nn.Module):
    def __init__(self, hidden_dim: int, adapter_dim: int) -> None:
        super().__init__()
        self.w12 = nn.Linear(hidden_dim, adapter_dim * 2, bias=False)
        self.w3 = nn.Linear(adapter_dim, hidden_dim, bias=False)
        nn.init.normal_(self.w12.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.w3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(gate) * value)


class ResidualTopKMoEAdapter(nn.Module):
    """Preserve the official dense MLP and add a sparse trainable residual adapter.

    Routing is chunk-level: one router decision is made from the mean action-token
    hidden state for each sample, then that expert mixture is applied to all action
    tokens in that denoising chunk.
    """

    def __init__(self, base_mlp: nn.Module, hidden_dim: int, config: OfficialMoEConfig) -> None:
        super().__init__()
        if config.top_k < 1 or config.top_k > config.num_experts:
            raise ValueError(f"top_k must be in [1, {config.num_experts}], got {config.top_k}")
        self.base_mlp = base_mlp
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.load_balance_weight = config.load_balance_weight
        self.router_z_loss_weight = config.router_z_loss_weight
        self.router = nn.Linear(hidden_dim, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [LowRankSwiGLUExpert(hidden_dim, config.adapter_dim) for _ in range(config.num_experts)]
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(config.init_scale), dtype=torch.float32))
        self.last_aux: dict[str, torch.Tensor] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dense = self.base_mlp(x)
        routed, aux = self._route(x)
        self.last_aux = aux
        return dense + self.residual_scale.to(dtype=dense.dtype, device=dense.device) * routed.to(dtype=dense.dtype)

    def _route(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        router_context = x.float().mean(dim=1)
        logits = self.router(router_context)
        probs = logits.softmax(dim=-1)
        top_values, top_indices = torch.topk(probs, k=self.top_k, dim=-1)
        top_values = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        out = torch.zeros_like(x)
        for slot in range(self.top_k):
            for expert_idx, expert in enumerate(self.experts):
                mask = top_indices[:, slot] == expert_idx
                if mask.any():
                    out[mask] += expert(x[mask]) * top_values[mask, slot].to(dtype=x.dtype).view(-1, 1, 1)

        return out, self._aux_losses(logits, probs, top_indices)

    def _aux_losses(
        self, logits: torch.Tensor, probs: torch.Tensor, top_indices: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        selected = F.one_hot(top_indices, num_classes=self.num_experts).float().sum(dim=1)
        load = selected.mean(dim=0) / float(self.top_k)
        importance = probs.mean(dim=0)
        load_balance_loss = self.num_experts * torch.sum(load * importance)
        router_z_loss = torch.logsumexp(logits, dim=-1).square().mean()
        expert_usage = selected.mean(dim=0)
        return {
            "load_balance_loss": load_balance_loss * self.load_balance_weight,
            "router_z_loss": router_z_loss * self.router_z_loss_weight,
            "expert_usage": expert_usage.detach(),
        }


def apply_residual_moe_adapters(policy: nn.Module, config: OfficialMoEConfig) -> list[str]:
    layers = policy.model.vlm_with_expert.lm_expert.layers
    hidden_dim = int(policy.model.vlm_with_expert.expert_hidden_size)
    patched: list[str] = []
    for layer_idx, layer in enumerate(layers):
        if layer_idx < config.layer_start:
            continue
        if (layer_idx - config.layer_start) % config.layer_stride != 0:
            continue
        if isinstance(layer.mlp, ResidualTopKMoEAdapter):
            raise ValueError(f"Layer {layer_idx} is already patched with ResidualTopKMoEAdapter")
        layer.mlp = ResidualTopKMoEAdapter(layer.mlp, hidden_dim=hidden_dim, config=config)
        patched.append(f"model.vlm_with_expert.lm_expert.layers.{layer_idx}.mlp")
    if not patched:
        raise ValueError("No SmolVLA action-expert MLP layers were patched")
    return patched


def clear_moe_aux(policy: nn.Module) -> None:
    for module in policy.modules():
        if isinstance(module, ResidualTopKMoEAdapter):
            module.last_aux = None


def collect_moe_aux(policy: nn.Module) -> dict[str, torch.Tensor]:
    load_balance: list[torch.Tensor] = []
    router_z: list[torch.Tensor] = []
    expert_usage: list[torch.Tensor] = []
    for module in policy.modules():
        if not isinstance(module, ResidualTopKMoEAdapter) or module.last_aux is None:
            continue
        load_balance.append(module.last_aux["load_balance_loss"])
        router_z.append(module.last_aux["router_z_loss"])
        expert_usage.append(module.last_aux["expert_usage"])
    if not load_balance:
        raise RuntimeError("No MoE auxiliary losses were collected from the official SmolVLA forward pass")
    return {
        "load_balance_loss": torch.stack(load_balance).mean(),
        "router_z_loss": torch.stack(router_z).mean(),
        "expert_usage": torch.stack(expert_usage).mean(dim=0),
    }


def set_official_moe_trainability(policy: nn.Module, freeze_official_dense: bool) -> None:
    if freeze_official_dense:
        for param in policy.parameters():
            param.requires_grad = False
    for module in policy.modules():
        if isinstance(module, ResidualTopKMoEAdapter):
            for param in module.router.parameters():
                param.requires_grad = True
            for param in module.experts.parameters():
                param.requires_grad = True
            module.residual_scale.requires_grad = True

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class SwiGLUExpert(nn.Module):
    def __init__(self, hidden_dim: int, inner_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w12 = nn.Linear(hidden_dim, inner_dim * 2, bias=False)
        self.w3 = nn.Linear(inner_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.w12(x).chunk(2, dim=-1)
        return self.w3(self.dropout(F.silu(gate) * value))


class SparseMoE(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.hidden_dim = int(config["hidden_dim"])
        self.num_experts = int(config.get("num_routed_experts", 4))
        self.top_k = int(config.get("top_k", 1))
        self.load_balance_weight = float(config.get("load_balance_weight", 0.01))
        self.router_z_loss_weight = float(config.get("router_z_loss_weight", 0.0001))
        inner_dim = int(self.hidden_dim * float(config.get("ffn_mult", 4)))
        dropout = float(config.get("dropout", 0.0))

        if self.top_k < 1 or self.top_k > self.num_experts:
            raise ValueError(f"top_k must be in [1, {self.num_experts}], got {self.top_k}")

        self.router = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self.routed_experts = nn.ModuleList(
            [SwiGLUExpert(self.hidden_dim, inner_dim, dropout=dropout) for _ in range(self.num_experts)]
        )
        self.shared_expert = SwiGLUExpert(self.hidden_dim, inner_dim, dropout=dropout)

    def forward(self, x: torch.Tensor, router_context: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if x.ndim != 3:
            raise ValueError(f"SparseMoE expects [B, T, H], got {tuple(x.shape)}")

        shared = self.shared_expert(x)
        routed, aux = self._forward_chunk(x, router_context)
        return shared + routed, aux

    def _forward_chunk(
        self, x: torch.Tensor, router_context: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits = self.router(router_context)
        probs = logits.softmax(dim=-1)
        top_values, top_indices = torch.topk(probs, k=self.top_k, dim=-1)
        top_values = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        out = torch.zeros_like(x)
        for slot in range(self.top_k):
            for expert_idx, expert in enumerate(self.routed_experts):
                mask = top_indices[:, slot] == expert_idx
                if mask.any():
                    out[mask] += expert(x[mask]) * top_values[mask, slot].view(-1, 1, 1)

        return out, self._aux_losses(logits, probs, top_indices)

    def _aux_losses(
        self, logits: torch.Tensor, probs: torch.Tensor, top_indices: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        selected = F.one_hot(top_indices, num_classes=self.num_experts).float().sum(dim=1)
        load = selected.mean(dim=0) / float(self.top_k)
        importance = probs.mean(dim=0)
        load_balance_loss = self.num_experts * torch.sum(load * importance)
        router_z_loss = torch.logsumexp(logits, dim=-1).square().mean()
        expert_usage = load
        return {
            "load_balance_loss": load_balance_loss * self.load_balance_weight,
            "router_z_loss": router_z_loss * self.router_z_loss_weight,
            "expert_usage": expert_usage.detach(),
        }

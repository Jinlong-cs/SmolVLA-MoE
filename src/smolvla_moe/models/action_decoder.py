from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from smolvla_moe.models.moe import SparseMoE


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ActionDecoderBlock(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        hidden_dim = int(config["hidden_dim"])
        num_heads = int(config.get("num_heads", 8))
        dropout = float(config.get("dropout", 0.0))

        self.self_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden_dim)

        moe_config = dict(config.get("moe", {}))
        moe_config.update(
            {
                "hidden_dim": hidden_dim,
                "ffn_mult": config.get("ffn_mult", 4),
                "dropout": dropout,
            }
        )
        self.ffn = SparseMoE(moe_config)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_key_padding_mask: torch.Tensor | None,
        router_context: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = x + self.self_attn(self.self_norm(x), self.self_norm(x), self.self_norm(x), need_weights=False)[0]
        x = x + self.cross_attn(
            self.cross_norm(x),
            context,
            context,
            key_padding_mask=context_key_padding_mask,
            need_weights=False,
        )[0]

        ffn_out, aux = self.ffn(self.ffn_norm(x), router_context=router_context)
        x = x + ffn_out
        return x, aux


class FlowMatchingActionDecoder(nn.Module):
    def __init__(self, config: dict[str, Any], context_dim: int) -> None:
        super().__init__()
        self.hidden_dim = int(config["hidden_dim"])
        self.action_dim = int(config["action_dim"])
        self.state_dim = int(config.get("state_dim", 0))
        self.horizon = int(config["horizon"])

        self.action_in = nn.Linear(self.action_dim, self.hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
        )
        self.state_proj = nn.Linear(self.state_dim, self.hidden_dim) if self.state_dim > 0 else None
        self.context_proj = nn.Identity() if context_dim == self.hidden_dim else nn.Linear(context_dim, self.hidden_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.horizon, self.hidden_dim))
        self.blocks = nn.ModuleList([ActionDecoderBlock(config) for _ in range(int(config.get("num_layers", 8)))])
        self.final_norm = nn.LayerNorm(self.hidden_dim)
        self.velocity_head = nn.Linear(self.hidden_dim, self.action_dim)

        nn.init.normal_(self.pos_embedding, std=0.02)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if noisy_actions.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError(
                f"noisy_actions must have shape [B, {self.horizon}, {self.action_dim}], got {tuple(noisy_actions.shape)}"
            )

        context = self.context_proj(context)
        context_key_padding_mask = None if context_mask is None else context_mask == 0
        context_summary = _masked_mean(context, context_mask)
        time_cond = self.time_mlp(timestep_embedding(timesteps, self.hidden_dim))
        if self.state_proj is None:
            state_cond = torch.zeros_like(time_cond)
        else:
            if state is None:
                state = torch.zeros(noisy_actions.shape[0], self.state_dim, device=noisy_actions.device)
            state_cond = self.state_proj(state)
        router_context = context_summary + time_cond + state_cond

        x = self.action_in(noisy_actions)
        x = x + self.pos_embedding[:, : x.shape[1]]
        x = x + router_context.unsqueeze(1)

        aux_losses: dict[str, torch.Tensor] = {}
        expert_usages: list[torch.Tensor] = []
        for block in self.blocks:
            x, aux = block(x, context, context_key_padding_mask, router_context)
            for key, value in aux.items():
                if key == "expert_usage":
                    expert_usages.append(value)
                else:
                    aux_losses[key] = aux_losses.get(key, torch.zeros((), device=value.device, dtype=value.dtype)) + value
        if expert_usages:
            aux_losses["expert_usage"] = torch.stack(expert_usages).mean(dim=0)

        velocity = self.velocity_head(self.final_norm(x))
        return velocity, aux_losses


def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=1)
    weights = mask.to(dtype=x.dtype).unsqueeze(-1)
    return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

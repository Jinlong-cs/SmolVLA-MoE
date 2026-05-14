from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from smolvla_moe.models.moe import SwiGLUExpert


@dataclass
class FlashVerificationResult:
    actions: torch.Tensor
    accepted_prefix_len: torch.Tensor
    radius_dist: torch.Tensor


class FlashDraftBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ffn_mult: float, dropout: float) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = SwiGLUExpert(hidden_dim, int(hidden_dim * ffn_mult), dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        x_norm = self.self_norm(x)
        x = x + self.self_attn(x_norm, x_norm, x_norm, need_weights=False)[0]
        x = x + self.cross_attn(
            self.cross_norm(x),
            context,
            context,
            key_padding_mask=context_key_padding_mask,
            need_weights=False,
        )[0]
        return x + self.ffn(self.ffn_norm(x))


class FlashDraftActionHead(nn.Module):
    """Lightweight parallel action-chunk proposer for FLASH verification."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        context_dim: int,
        action_dim: int,
        state_dim: int,
        horizon: int,
    ) -> None:
        super().__init__()
        hidden_dim = int(config["hidden_dim"])
        num_heads = int(config.get("num_heads", 8))
        dropout = float(config.get("dropout", 0.0))

        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.horizon = int(horizon)
        self.context_proj = nn.Identity() if int(context_dim) == hidden_dim else nn.Linear(int(context_dim), hidden_dim)
        self.state_proj = nn.Linear(self.state_dim, hidden_dim) if self.state_dim > 0 else None
        self.action_queries = nn.Parameter(torch.zeros(1, self.horizon, hidden_dim))
        self.query_pos = nn.Parameter(torch.zeros(1, self.horizon, hidden_dim))
        self.blocks = nn.ModuleList(
            [
                FlashDraftBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ffn_mult=float(config.get("ffn_mult", 4)),
                    dropout=dropout,
                )
                for _ in range(int(config.get("num_layers", 2)))
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.action_head = nn.Linear(hidden_dim, self.action_dim)

        nn.init.normal_(self.action_queries, std=0.02)
        nn.init.normal_(self.query_pos, std=0.02)

    def forward(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None,
        state: torch.Tensor | None,
    ) -> torch.Tensor:
        context = self.context_proj(context)
        context_key_padding_mask = None if context_mask is None else context_mask == 0
        context_summary = masked_mean(context, context_mask)
        state_cond = self._state_cond(context.shape[0], context.device, context_summary.dtype, state)
        x = (
            self.action_queries.expand(context.shape[0], -1, -1)
            + self.query_pos
            + context_summary[:, None, :]
            + state_cond[:, None, :]
        )
        for block in self.blocks:
            x = block(x, context, context_key_padding_mask)
        return self.action_head(self.final_norm(x))

    def _state_cond(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        state: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.state_proj is None:
            return torch.zeros(batch_size, self.action_queries.shape[-1], device=device, dtype=dtype)
        if state is None:
            state = torch.zeros(batch_size, self.state_dim, device=device, dtype=dtype)
        return self.state_proj(state)


def draft_action_loss(
    pred_actions: torch.Tensor,
    target_actions: torch.Tensor,
    action_mask: torch.Tensor | None,
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    weights = prefix_step_weights(
        horizon=int(pred_actions.shape[1]),
        device=pred_actions.device,
        dtype=pred_actions.dtype,
        prefix_cap=int(config.get("prefix_cap", pred_actions.shape[1])),
        gamma=float(config.get("prefix_gamma", 0.9)),
        tail_weight=float(config.get("tail_weight", 0.1)),
    )
    loss_per_dim = F.smooth_l1_loss(
        pred_actions,
        target_actions,
        beta=float(config.get("huber_beta", 1.0)),
        reduction="none",
    )
    loss_per_step = loss_per_dim.mean(dim=-1)
    weighted_loss = reduce_weighted_steps(loss_per_step, weights, action_mask)

    action_l1 = (pred_actions - target_actions).abs().mean(dim=-1)
    prefix_len = int(min(pred_actions.shape[1], max(1, int(config.get("prefix_cap", pred_actions.shape[1])))))
    prefix_l1 = reduce_weighted_steps(
        action_l1[:, :prefix_len],
        torch.ones(prefix_len, device=pred_actions.device, dtype=pred_actions.dtype),
        None if action_mask is None else action_mask[:, :prefix_len],
    )
    return {"flash_draft_loss": weighted_loss, "flash_draft_prefix_l1": prefix_l1}


@torch.no_grad()
def verify_draft_actions(
    *,
    decoder: nn.Module,
    draft_actions: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor | None,
    state: torch.Tensor | None,
    config: dict[str, Any],
    generator: torch.Generator | None = None,
) -> FlashVerificationResult:
    timesteps = torch.as_tensor(config.get("timesteps", [0.75]), device=draft_actions.device, dtype=draft_actions.dtype)
    if timesteps.ndim != 1 or timesteps.numel() < 1:
        raise ValueError("flash verify timesteps must be a non-empty 1D list")

    batch_size, horizon, action_dim = draft_actions.shape
    verify_k = int(timesteps.numel())
    noise = torch.randn(
        draft_actions.shape,
        device=draft_actions.device,
        dtype=draft_actions.dtype,
        generator=generator,
    )
    t_view = timesteps.view(1, verify_k, 1, 1)
    x_t = ((1.0 - t_view) * noise[:, None, :, :] + t_view * draft_actions[:, None, :, :]).reshape(
        batch_size * verify_k,
        horizon,
        action_dim,
    )
    t_flat = timesteps[None, :].expand(batch_size, verify_k).reshape(batch_size * verify_k)
    context_bk = expand_batch(context, verify_k)
    context_mask_bk = None if context_mask is None else expand_batch(context_mask, verify_k)
    state_bk = None if state is None else expand_batch(state, verify_k)

    velocity, _ = decoder(x_t, t_flat, context_bk, context_mask_bk, state_bk)
    x_hat_flat = x_t + (1.0 - t_flat[:, None, None]) * velocity
    x_hat = x_hat_flat.reshape(batch_size, verify_k, horizon, action_dim)
    accepted_prefix_len, dist = radius_prefix_acceptance(
        draft_actions=draft_actions,
        reconstructed_actions=x_hat,
        radius=float(config.get("radius", 0.08)),
        dist_dims=int(config.get("dist_dims", 6)),
        eval_horizon=int(config.get("eval_horizon", horizon)),
    )
    tail_actions = x_hat.mean(dim=1)
    actions = stitch_verified_actions(draft_actions, tail_actions, accepted_prefix_len)
    return FlashVerificationResult(actions=actions, accepted_prefix_len=accepted_prefix_len, radius_dist=dist)


def prefix_step_weights(
    *,
    horizon: int,
    device: torch.device,
    dtype: torch.dtype,
    prefix_cap: int,
    gamma: float,
    tail_weight: float,
) -> torch.Tensor:
    prefix_len = int(min(horizon, max(1, prefix_cap)))
    weights = torch.full((horizon,), float(tail_weight), device=device, dtype=dtype)
    weights[:prefix_len] = torch.pow(
        torch.as_tensor(float(gamma), device=device, dtype=dtype),
        torch.arange(prefix_len, device=device, dtype=dtype),
    )
    return weights


def reduce_weighted_steps(
    values: torch.Tensor,
    weights: torch.Tensor,
    action_mask: torch.Tensor | None,
) -> torch.Tensor:
    step_weights = weights.to(device=values.device, dtype=values.dtype).view(1, -1)
    if action_mask is not None:
        step_weights = step_weights * action_mask.to(dtype=values.dtype)
    return (values * step_weights).sum() / step_weights.sum().clamp_min(1.0)


def radius_prefix_acceptance(
    *,
    draft_actions: torch.Tensor,
    reconstructed_actions: torch.Tensor,
    radius: float,
    dist_dims: int,
    eval_horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    eval_h = int(min(draft_actions.shape[1], max(1, eval_horizon)))
    eval_d = int(min(draft_actions.shape[2], max(1, dist_dims)))
    if draft_actions.shape[2] >= 7:
        eval_d = min(eval_d, 6)
    diff = reconstructed_actions[:, :, :eval_h, :eval_d] - draft_actions[:, None, :eval_h, :eval_d]
    dist = torch.linalg.vector_norm(diff.float(), ord=2, dim=-1) / float(eval_d) ** 0.5
    ok = dist <= float(radius)
    prefix_len_per_k = ok.to(dtype=torch.int64).cumprod(dim=-1).sum(dim=-1)
    return prefix_len_per_k.min(dim=1).values.to(dtype=torch.int64), dist


def stitch_verified_actions(
    draft_actions: torch.Tensor,
    tail_actions: torch.Tensor,
    accepted_prefix_len: torch.Tensor,
) -> torch.Tensor:
    idx = torch.arange(draft_actions.shape[1], device=draft_actions.device, dtype=torch.int64)[None, :]
    accept_mask = (idx < accepted_prefix_len[:, None])[:, :, None]
    return torch.where(accept_mask, draft_actions, tail_actions)


def expand_batch(x: torch.Tensor, repeat: int) -> torch.Tensor:
    return x[:, None, ...].expand(-1, int(repeat), *x.shape[1:]).reshape(x.shape[0] * int(repeat), *x.shape[1:])


def masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=1)
    weights = mask.to(dtype=x.dtype).unsqueeze(-1)
    return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

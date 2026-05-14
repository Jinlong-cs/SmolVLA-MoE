from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from smolvla_moe.models.action_decoder import FlowMatchingActionDecoder


class FlowMatchingObjective:
    def __init__(self, config: dict[str, Any]) -> None:
        self.train_time_epsilon = float(config.get("train_time_epsilon", 0.001))
        self.inference_steps = int(config.get("inference_steps", 4))

    def loss(
        self,
        decoder: FlowMatchingActionDecoder,
        actions: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None,
        state: torch.Tensor | None,
        action_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = actions.shape[0]
        t = torch.rand(batch_size, device=actions.device)
        eps = self.train_time_epsilon
        t = t * (1.0 - 2.0 * eps) + eps
        noise = torch.randn_like(actions)
        t_view = t.view(batch_size, 1, 1)
        noisy_actions = (1.0 - t_view) * noise + t_view * actions
        target_velocity = actions - noise

        pred_velocity, aux = decoder(noisy_actions, t, context, context_mask, state)
        if action_mask is None:
            flow_loss = F.mse_loss(pred_velocity, target_velocity)
            valid_action_fraction = torch.ones((), device=actions.device, dtype=actions.dtype)
        else:
            weights = action_mask.to(dtype=pred_velocity.dtype).unsqueeze(-1)
            squared_error = (pred_velocity - target_velocity).square() * weights
            flow_loss = squared_error.sum() / (weights.sum() * pred_velocity.shape[-1]).clamp_min(1.0)
            valid_action_fraction = weights.mean()
        total_loss = flow_loss
        for key, value in aux.items():
            if key.endswith("_loss"):
                total_loss = total_loss + value

        metrics = {"loss": total_loss, "flow_loss": flow_loss, "valid_action_fraction": valid_action_fraction}
        metrics.update(aux)
        return metrics

    @torch.no_grad()
    def sample(
        self,
        decoder: FlowMatchingActionDecoder,
        batch_size: int,
        context: torch.Tensor,
        context_mask: torch.Tensor | None,
        state: torch.Tensor | None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        actions = torch.randn(
            batch_size,
            decoder.horizon,
            decoder.action_dim,
            device=context.device,
            generator=generator,
        )
        steps = max(self.inference_steps, 1)
        dt = 1.0 / float(steps)
        for step in range(steps):
            t = torch.full((batch_size,), step / float(steps), device=context.device)
            velocity, _ = decoder(actions, t, context, context_mask, state)
            actions = actions + dt * velocity
        return actions

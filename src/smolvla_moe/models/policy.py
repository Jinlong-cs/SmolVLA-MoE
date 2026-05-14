from __future__ import annotations

from typing import Any

import torch
from torch import nn

from smolvla_moe.data.batch import VLABatch
from smolvla_moe.models.action_decoder import FlowMatchingActionDecoder
from smolvla_moe.models.backbone import build_backbone
from smolvla_moe.models.flow_matching import FlowMatchingObjective


class SmolVLAMoEPolicy(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model_config = config["model"]
        backbone_config = dict(model_config["backbone"])
        decoder_config = dict(model_config["action_decoder"])

        self.backbone = build_backbone(backbone_config)
        self.action_decoder = FlowMatchingActionDecoder(decoder_config, context_dim=self.backbone.hidden_dim)
        self.flow = FlowMatchingObjective(model_config.get("flow", {}))

    def encode_context(self, batch: VLABatch) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.backbone(
            batch.images,
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            extras=batch.extras,
        )

    def compute_loss(self, batch: VLABatch) -> dict[str, torch.Tensor]:
        if batch.actions is None:
            raise ValueError("batch.actions is required for training")
        context, context_mask = self.encode_context(batch)
        return self.flow.loss(self.action_decoder, batch.actions, context, context_mask, batch.state, batch.action_mask)

    def forward(self, batch: VLABatch) -> dict[str, torch.Tensor]:
        return self.compute_loss(batch)

    @torch.no_grad()
    def predict_action(self, batch: VLABatch, generator: torch.Generator | None = None) -> torch.Tensor:
        context, context_mask = self.encode_context(batch)
        return self.flow.sample(
            self.action_decoder,
            batch_size=batch.images.shape[0],
            context=context,
            context_mask=context_mask,
            state=batch.state,
            generator=generator,
        )


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    total = 0
    for param in model.parameters():
        if trainable_only and not param.requires_grad:
            continue
        total += param.numel()
    return total

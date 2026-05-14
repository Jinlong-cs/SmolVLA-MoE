from __future__ import annotations

from typing import Any

import torch
from torch import nn

from smolvla_moe.data.batch import VLABatch
from smolvla_moe.models.action_decoder import FlowMatchingActionDecoder
from smolvla_moe.models.backbone import build_backbone
from smolvla_moe.models.flash import FlashDraftActionHead
from smolvla_moe.models.flash import draft_action_loss
from smolvla_moe.models.flash import verify_draft_actions
from smolvla_moe.models.flow_matching import FlowMatchingObjective


class SmolVLAMoEPolicy(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model_config = config["model"]
        backbone_config = dict(model_config["backbone"])
        decoder_config = dict(model_config["action_decoder"])
        flash_config = dict(model_config.get("flash", {}))

        self.backbone = build_backbone(backbone_config)
        self.action_decoder = FlowMatchingActionDecoder(decoder_config, context_dim=self.backbone.hidden_dim)
        self.flow = FlowMatchingObjective(model_config.get("flow", {}))
        self.flash_config = flash_config
        self.flash_draft_head = (
            FlashDraftActionHead(
                dict(flash_config.get("draft", {})),
                context_dim=self.backbone.hidden_dim,
                action_dim=int(decoder_config["action_dim"]),
                state_dim=int(decoder_config.get("state_dim", 0)),
                horizon=int(decoder_config["horizon"]),
            )
            if bool(flash_config.get("enabled", False))
            else None
        )

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
        metrics = self.flow.loss(
            self.action_decoder,
            batch.actions,
            context,
            context_mask,
            batch.state,
            batch.action_mask,
        )
        if self.flash_draft_head is not None and bool(self.flash_config.get("train_draft", True)):
            draft_actions = self.flash_draft_head(context, context_mask, batch.state)
            draft_metrics = draft_action_loss(
                draft_actions,
                batch.actions,
                batch.action_mask,
                dict(self.flash_config.get("loss", {})),
            )
            metrics.update(draft_metrics)
            metrics["loss"] = (
                metrics["loss"] + float(self.flash_config.get("loss_weight", 0.2)) * draft_metrics["flash_draft_loss"]
            )
        return metrics

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

    @torch.no_grad()
    def predict_action_flash(self, batch: VLABatch, generator: torch.Generator | None = None) -> torch.Tensor:
        if self.flash_draft_head is None:
            raise RuntimeError("FLASH draft head is not enabled in this model config")
        context, context_mask = self.encode_context(batch)
        draft_actions = self.flash_draft_head(context, context_mask, batch.state)
        result = verify_draft_actions(
            decoder=self.action_decoder,
            draft_actions=draft_actions,
            context=context,
            context_mask=context_mask,
            state=batch.state,
            config=dict(self.flash_config.get("verify", {})),
            generator=generator,
        )
        if bool(self.flash_config.get("verify", {}).get("full_fallback", True)) and (
            result.accepted_prefix_len <= 0
        ).any():
            return self.flow.sample(
                self.action_decoder,
                batch_size=batch.images.shape[0],
                context=context,
                context_mask=context_mask,
                state=batch.state,
                generator=generator,
            )
        return result.actions


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    total = 0
    for param in model.parameters():
        if trainable_only and not param.requires_grad:
            continue
        total += param.numel()
    return total

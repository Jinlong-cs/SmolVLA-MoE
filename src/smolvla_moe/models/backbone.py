from __future__ import annotations

from typing import Any

import torch
from torch import nn


class TinyVisionLanguageBackbone(nn.Module):
    """Small VLM-like encoder for smoke tests and architecture debugging."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.hidden_dim = int(config["hidden_dim"])
        patch_size = int(config.get("patch_size", 16))
        vocab_size = int(config.get("text_vocab_size", 32768))
        max_text_len = int(config.get("max_text_len", 64))
        num_layers = int(config.get("num_layers", 2))
        num_heads = int(config.get("num_heads", 4))

        self.image_proj = nn.Conv2d(3, self.hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Embedding(vocab_size, self.hidden_dim)
        self.text_pos = nn.Parameter(torch.zeros(1, max_text_len, self.hidden_dim))
        self.camera_embedding = nn.Parameter(torch.zeros(1, 8, 1, self.hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=float(config.get("dropout", 0.0)),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(self.hidden_dim)

        nn.init.normal_(self.text_pos, std=0.02)
        nn.init.normal_(self.camera_embedding, std=0.02)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        extras: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del extras
        if images.ndim != 5:
            raise ValueError(f"images must have shape [B, num_cameras, 3, H, W], got {tuple(images.shape)}")

        batch_size, num_cameras = images.shape[:2]
        flat_images = images.flatten(0, 1)
        image_tokens = self.image_proj(flat_images).flatten(2).transpose(1, 2)
        image_tokens = image_tokens.reshape(batch_size, num_cameras, image_tokens.shape[1], self.hidden_dim)
        image_tokens = image_tokens + self.camera_embedding[:, :num_cameras]
        image_tokens = image_tokens.flatten(1, 2)

        if input_ids is None:
            input_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=images.device)
        text_tokens = self.text_embedding(input_ids)
        text_tokens = text_tokens + self.text_pos[:, : text_tokens.shape[1]]

        tokens = torch.cat([text_tokens, image_tokens], dim=1)
        if attention_mask is None:
            src_key_padding_mask = None
            full_mask = None
        else:
            image_mask = torch.ones(batch_size, image_tokens.shape[1], dtype=attention_mask.dtype, device=images.device)
            full_mask = torch.cat([attention_mask, image_mask], dim=1)
            src_key_padding_mask = full_mask == 0
        encoded = self.encoder(tokens, src_key_padding_mask=src_key_padding_mask)
        return self.norm(encoded), full_mask


class HFSmolVLM2Backbone(nn.Module):
    """Production hook for pretrained SmolVLM2-family context extraction.

    The real training dataloader should provide `batch.extras["hf_inputs"]`, produced by the matching Hugging Face
    processor. Keeping this wrapper explicit avoids silently treating a multimodal model like a plain text model.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError as exc:
            raise ImportError("Install SmolVLA-MoE with the `hf` extra to use hf_smolvlm2 backbones.") from exc

        model_name = str(config["model_name"])
        trust_remote_code = bool(config.get("trust_remote_code", True))
        self.model = AutoModelForImageTextToText.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.hidden_dim = int(config.get("context_dim", self.model.config.hidden_size))

        if bool(config.get("freeze", True)):
            self.model.requires_grad_(False)
            self.model.eval()

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        extras: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del images, input_ids, attention_mask
        hf_inputs = None if extras is None else extras.get("hf_inputs")
        if hf_inputs is None:
            raise ValueError(
                "hf_smolvlm2 backbone requires batch.extras['hf_inputs'] from the matching Hugging Face processor."
            )
        outputs = self.model(**hf_inputs, output_hidden_states=True, return_dict=True)
        hidden = outputs.hidden_states[-1]
        mask = hf_inputs.get("attention_mask")
        return hidden, mask


def build_backbone(config: dict[str, Any]) -> nn.Module:
    backbone_type = str(config.get("type", "tiny"))
    if backbone_type == "tiny":
        return TinyVisionLanguageBackbone(config)
    if backbone_type == "hf_smolvlm2":
        return HFSmolVLM2Backbone(config)
    raise ValueError(f"Unsupported backbone type: {backbone_type}")

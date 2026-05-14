from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import AutoModelForImageTextToText


class HFSmolVLM2Backbone(nn.Module):
    """Production hook for pretrained SmolVLM2-family context extraction.

    The real training dataloader should provide `batch.extras["hf_inputs"]`, produced by the matching Hugging Face
    processor. Keeping this wrapper explicit avoids silently treating a multimodal model like a plain text model.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.hidden_dim = int(config["hidden_dim"])
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(config["model_name"]),
            trust_remote_code=True,
            torch_dtype=_torch_dtype(config["torch_dtype"]),
        )
        self.model.gradient_checkpointing_enable()

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        extras: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del images, input_ids, attention_mask
        hf_inputs = extras["hf_inputs"]
        model = getattr(self.model, "model", self.model)
        forward_kwargs = dict(hf_inputs)
        forward_kwargs.update({"output_hidden_states": False, "return_dict": True, "use_cache": False})
        outputs = model(**forward_kwargs)
        return outputs.last_hidden_state, hf_inputs["attention_mask"]


def build_backbone(config: dict[str, Any]) -> nn.Module:
    return HFSmolVLM2Backbone(config)


def _torch_dtype(value: Any) -> torch.dtype | str | None:
    if value is None:
        return None
    name = str(value)
    if name in {"auto", "torch_dtype=auto"}:
        return "auto"
    if name in {"bfloat16", "bf16", "torch.bfloat16"}:
        return torch.bfloat16
    if name in {"float16", "fp16", "torch.float16"}:
        return torch.float16
    if name in {"float32", "fp32", "torch.float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {value}")

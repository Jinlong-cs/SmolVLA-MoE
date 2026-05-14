from __future__ import annotations

from typing import Any

import torch
from torch import nn


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
        self.freeze = bool(config.get("freeze", True))
        kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        torch_dtype = _torch_dtype(config.get("torch_dtype", "auto"))
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        attn_implementation = config.get("attn_implementation")
        if attn_implementation not in (None, "", "null"):
            kwargs["attn_implementation"] = str(attn_implementation)
        self.model = AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
        self.hidden_dim = _infer_hidden_dim(self.model.config)
        if bool(config.get("gradient_checkpointing", False)) and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        if self.freeze:
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
        model = getattr(self.model, "model", self.model)
        forward_kwargs = dict(hf_inputs)
        forward_kwargs.update({"output_hidden_states": False, "return_dict": True, "use_cache": False})
        with torch.set_grad_enabled(not self.freeze):
            outputs = model(**forward_kwargs)
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            hidden = outputs[0]
        mask = hf_inputs.get("attention_mask")
        return hidden, mask


def build_backbone(config: dict[str, Any]) -> nn.Module:
    backbone_type = str(config.get("type", "hf_smolvlm2"))
    if backbone_type == "hf_smolvlm2":
        return HFSmolVLM2Backbone(config)
    raise ValueError(f"Unsupported backbone type: {backbone_type}")


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


def _infer_hidden_dim(config: Any) -> int:
    for path in (("text_config", "hidden_size"), ("hidden_size",), ("vision_config", "hidden_size")):
        cursor = config
        for key in path:
            cursor = getattr(cursor, key, None)
            if cursor is None:
                break
        if cursor is not None:
            return int(cursor)
    raise ValueError("Could not infer hidden size from Hugging Face model config.")

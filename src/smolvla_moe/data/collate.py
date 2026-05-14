from __future__ import annotations

from typing import Any

import torch

from smolvla_moe.data.batch import VLABatch
from smolvla_moe.data.text import HashTokenizer


class VLACollator:
    def __init__(self, config: dict[str, Any]) -> None:
        model_config = config["model"]
        backbone_config = model_config["backbone"]
        self.backbone_type = str(backbone_config.get("type", "tiny"))
        self.hash_tokenizer = HashTokenizer(
            vocab_size=int(backbone_config.get("text_vocab_size", 32768)),
            max_length=int(backbone_config.get("max_text_len", 64)),
        )
        self.processor = None
        if self.backbone_type == "hf_smolvlm2":
            try:
                from transformers import AutoProcessor
            except ImportError as exc:
                raise ImportError("Install SmolVLA-MoE with the `hf` extra to collate hf_smolvlm2 inputs.") from exc
            self.processor = AutoProcessor.from_pretrained(
                str(backbone_config["model_name"]),
                trust_remote_code=bool(backbone_config.get("trust_remote_code", True)),
            )

    def __call__(self, samples: list[dict[str, Any]]) -> VLABatch:
        images = torch.stack([sample["images"] for sample in samples], dim=0)
        state = torch.stack([sample["state"] for sample in samples], dim=0) if samples[0].get("state") is not None else None
        actions = (
            torch.stack([sample["actions"] for sample in samples], dim=0) if samples[0].get("actions") is not None else None
        )
        language = [str(sample.get("language", "")) for sample in samples]

        if self.backbone_type == "hf_smolvlm2":
            hf_inputs = self._hf_inputs(images, language)
            input_ids = hf_inputs.get("input_ids")
            attention_mask = hf_inputs.get("attention_mask")
            extras = {"hf_inputs": hf_inputs}
        else:
            input_ids, attention_mask = self.hash_tokenizer(language)
            extras = None

        return VLABatch(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            state=state,
            actions=actions,
            language=language,
            extras=extras,
        )

    def _hf_inputs(self, images: torch.Tensor, language: list[str]) -> dict[str, torch.Tensor]:
        if self.processor is None:
            raise RuntimeError("HF processor is not initialized")
        # SmolVLM processors accept nested image lists. Preserve all configured cameras instead of using only one view.
        camera_images = [
            [camera.permute(1, 2, 0).cpu().numpy() for camera in sample_images]
            for sample_images in images
        ]
        return self.processor(images=camera_images, text=language, return_tensors="pt", padding=True)

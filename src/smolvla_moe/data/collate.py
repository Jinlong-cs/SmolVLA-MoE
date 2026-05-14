from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoProcessor

from smolvla_moe.data.batch import VLABatch


class VLACollator:
    def __init__(self, config: dict[str, Any]) -> None:
        backbone_config = config["model"]["backbone"]
        self.processor = AutoProcessor.from_pretrained(str(backbone_config["model_name"]), trust_remote_code=True)

    def __call__(self, samples: list[dict[str, Any]]) -> VLABatch:
        images = torch.stack([sample["images"] for sample in samples], dim=0)
        state = torch.stack([sample["state"] for sample in samples], dim=0) if samples[0].get("state") is not None else None
        actions = (
            torch.stack([sample["actions"] for sample in samples], dim=0) if samples[0].get("actions") is not None else None
        )
        action_mask = (
            torch.stack([sample["action_mask"] for sample in samples], dim=0)
            if samples[0].get("action_mask") is not None
            else None
        )
        language = [str(sample.get("language", "")) for sample in samples]

        hf_inputs = self._hf_inputs(images, language)
        input_ids = hf_inputs.get("input_ids")
        attention_mask = hf_inputs.get("attention_mask")
        extras = {"hf_inputs": hf_inputs}

        return VLABatch(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            state=state,
            actions=actions,
            action_mask=action_mask,
            language=language,
            extras=extras,
        )

    def _hf_inputs(self, images: torch.Tensor, language: list[str]) -> dict[str, torch.Tensor]:
        # SmolVLM processors require one <image> placeholder per camera view.
        image_tokens = "<image>" * int(images.shape[1])
        text = [f"{image_tokens}{instruction}" for instruction in language]
        camera_images = [
            [F.interpolate(camera.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False)[0]
             .permute(1, 2, 0)
             .cpu()
             .numpy()
             for camera in sample_images]
            for sample_images in images
        ]
        return self.processor(
            images=camera_images,
            text=text,
            return_tensors="pt",
            padding=True,
            size={"longest_edge": 256},
            do_rescale=False,
        )

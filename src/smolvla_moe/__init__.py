"""SmolVLA-MoE package."""

from smolvla_moe.config import load_config
from smolvla_moe.models.policy import SmolVLAMoEPolicy

__all__ = ["SmolVLAMoEPolicy", "load_config"]

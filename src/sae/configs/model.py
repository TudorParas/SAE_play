"""
Model configuration for SAE experiments.

Defines which language model to use and where to extract activations from.
"""

from dataclasses import dataclass
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass
class ModelConfig:
    """
    Configuration for the language model and activation extraction.

    Attributes:
        name: HuggingFace model identifier
        layer_idx: Which transformer layer to extract activations from
    """

    name: str = "EleutherAI/pythia-70m"
    layer_idx: int = 3

    def resolve(self) -> Tuple["PreTrainedModel", "PreTrainedTokenizer", str]:
        """
        Load and return the configured model.

        Returns:
            Tuple of (model, tokenizer, device_string)
        """
        from src.sae.experiments.common import load_model

        return load_model(self.name)

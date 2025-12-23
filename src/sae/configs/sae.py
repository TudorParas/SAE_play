"""
SAE architecture configurations.

Defines SimpleSAE and DeepSAE architectures with their sparsity mechanisms.
"""

from dataclasses import dataclass, field
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from src.sae.models.base import BaseSAE
    from src.sae.sparsity import Sparsity


def _create_sparsity(
    sparsity_type: str, sparsity_k: int | None, num_features: int
) -> "Sparsity":
    """Create a sparsity mechanism from config parameters."""
    from src.sae.sparsity import TopKSparsity, L1Sparsity, JumpReLUSparsity

    if sparsity_type == "topk":
        return TopKSparsity(k=sparsity_k)
    elif sparsity_type == "l1":
        return L1Sparsity()
    elif sparsity_type == "jumprelu":
        return JumpReLUSparsity(num_features=num_features)
    else:
        raise ValueError(f"Unknown sparsity type: {sparsity_type}")


@dataclass
class SimpleSAEConfig:
    """
    Configuration for a simple (single-layer) SAE.

    Attributes:
        hidden_dim_multiplier: Expansion factor for hidden dimension (e.g., 32 = 32x)
        sparsity_type: Type of sparsity mechanism ("topk", "l1", "jumprelu")
        sparsity_k: Number of active features (for TopK sparsity)
    """

    hidden_dim_multiplier: int
    sparsity_type: Literal["topk", "l1", "jumprelu"]
    sparsity_k: int | None = None

    def resolve(self, input_dim: int, device: str) -> "BaseSAE":
        """
        Create and return a SimpleSAE instance with the configured parameters.

        Args:
            input_dim: Input dimension size
            device: Device to place the model on ("cpu" or "cuda")

        Returns:
            Configured SimpleSAE instance
        """
        from src.sae.models.simple import SimpleSAE

        hidden_dim = input_dim * self.hidden_dim_multiplier
        sparsity = _create_sparsity(self.sparsity_type, self.sparsity_k, hidden_dim)

        sae = SimpleSAE(
            input_dim=input_dim, hidden_dim=hidden_dim, sparsity=sparsity
        ).to(device)

        return sae


@dataclass
class DeepSAEConfig:
    """
    Configuration for a deep (multi-layer) SAE.

    Attributes:
        encoder_hidden_dims: List of hidden dimensions for encoder layers
                            (as multipliers of input_dim)
        decoder_hidden_dims: List of hidden dimensions for decoder layers
                            (as multipliers of input_dim)
        sparsity_type: Type of sparsity mechanism ("topk", "l1", "jumprelu")
        sparsity_k: Number of active features (for TopK sparsity)
    """

    encoder_hidden_dims: list[int]
    decoder_hidden_dims: list[int]
    sparsity_type: Literal["topk", "l1", "jumprelu"]
    sparsity_k: int | None = None

    def resolve(self, input_dim: int, device: str) -> "BaseSAE":
        """
        Create and return a DeepSAE instance with the configured parameters.

        Args:
            input_dim: Input dimension size
            device: Device to place the model on ("cpu" or "cuda")

        Returns:
            Configured DeepSAE instance
        """
        from src.sae.models.deep import DeepSAE

        # Convert multipliers to actual dimensions
        encoder_hidden_dims = [input_dim * m for m in self.encoder_hidden_dims]
        decoder_hidden_dims = [input_dim * m for m in self.decoder_hidden_dims]

        sparsity = _create_sparsity(
            self.sparsity_type, self.sparsity_k, encoder_hidden_dims[-1]
        )

        sae = DeepSAE(
            input_dim=input_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
            sparsity=sparsity,
        ).to(device)

        return sae

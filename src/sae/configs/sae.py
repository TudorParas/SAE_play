"""
SAE architecture configurations.

Defines SimpleSAE and DeepSAE architectures with their sparsity mechanisms.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal, List, TYPE_CHECKING

if TYPE_CHECKING:
    from src.sae.models.base import BaseSAE
    from src.sae.sparsity import Sparsity


@dataclass
class SimpleSAEConfig:
    """
    Configuration for a simple (single-layer) SAE.

    Attributes:
        hidden_dim_multiplier: Expansion factor for hidden dimension (e.g., 32 = 32x)
        sparsity_type: Type of sparsity mechanism ("topk", "l1", "jumprelu")
        sparsity_k: Number of active features (for TopK sparsity)
        sparsity_coefficient: L1 penalty coefficient (for L1 sparsity)
    """

    hidden_dim_multiplier: int = 32
    sparsity_type: Literal["topk", "l1", "jumprelu"] = "topk"
    sparsity_k: Optional[int] = 128
    sparsity_coefficient: Optional[float] = None

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
        sparsity = self._create_sparsity(hidden_dim)

        sae = SimpleSAE(
            input_dim=input_dim, hidden_dim=hidden_dim, sparsity=sparsity
        ).to(device)

        return sae

    def _create_sparsity(self, hidden_dim: int) -> "Sparsity":
        """Helper to create the configured sparsity mechanism."""
        from src.sae.sparsity import TopKSparsity, L1Sparsity, JumpReLUSparsity

        if self.sparsity_type == "topk":
            return TopKSparsity(k=self.sparsity_k)
        elif self.sparsity_type == "l1":
            return L1Sparsity()
        elif self.sparsity_type == "jumprelu":
            return JumpReLUSparsity(num_features=hidden_dim)
        else:
            raise ValueError(f"Unknown sparsity type: {self.sparsity_type}")


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
        sparsity_coefficient: L1 penalty coefficient (for L1 sparsity)
    """

    encoder_hidden_dims: List[int] = field(default_factory=lambda: [4, 32])
    decoder_hidden_dims: List[int] = field(default_factory=lambda: [4])
    sparsity_type: Literal["topk", "l1", "jumprelu"] = "l1"
    sparsity_k: Optional[int] = None
    sparsity_coefficient: Optional[float] = None

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

        sparsity = self._create_sparsity(encoder_hidden_dims[-1])

        sae = DeepSAE(
            input_dim=input_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
            sparsity=sparsity,
        ).to(device)

        return sae

    def _create_sparsity(self, probe_dim: int) -> "Sparsity":
        """Helper to create the configured sparsity mechanism."""
        from src.sae.sparsity import TopKSparsity, L1Sparsity, JumpReLUSparsity

        if self.sparsity_type == "topk":
            return TopKSparsity(k=self.sparsity_k)
        elif self.sparsity_type == "l1":
            return L1Sparsity()
        elif self.sparsity_type == "jumprelu":
            return JumpReLUSparsity(num_features=probe_dim)
        else:
            raise ValueError(f"Unknown sparsity type: {self.sparsity_type}")

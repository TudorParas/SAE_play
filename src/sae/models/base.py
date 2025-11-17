"""
Base class for Sparse Autoencoders.

All SAE implementations should inherit from this base class to ensure
a consistent interface across different architectures.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any


class BaseSAE(nn.Module, ABC):
    """
    Abstract base class for Sparse Autoencoders.

    All SAE implementations must implement the forward method and provide
    architecture information through get_config().
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the SAE.

        Args:
            x: Input activations, shape (batch_size, input_dim)

        Returns:
            Tuple of (reconstructed, sparse_features):
                - reconstructed: Reconstructed activations, shape (batch_size, input_dim)
                - sparse_features: Sparse feature activations, shape (batch_size, hidden_dim)
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of this SAE for saving/loading.

        Returns:
            Dictionary containing all parameters needed to reconstruct this SAE
        """
        pass

    @property
    @abstractmethod
    def input_dim(self) -> int:
        """Input dimension of the SAE."""
        pass

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """Dimension of the sparse latent space (hidden dimension)."""
        pass

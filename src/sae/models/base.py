"""
Base class for Sparse Autoencoders.

Implements the template method pattern: forward() is defined here,
child classes only need to override encode() and decode().
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any
from ..sparsity import SparsityMechanism


class BaseSAE(nn.Module, ABC):
    """
    Abstract base class for Sparse Autoencoders.

    Uses template method pattern:
    - forward() is implemented here (same for all SAEs)
    - Child classes override encode() and decode()

    All SAE implementations should inherit from this and implement:
    - encode(x) → pre_activation features
    - decode(features) → reconstructed input
    - input_dim property
    - latent_dim property
    - get_config() method for saving/loading
    """

    def __init__(self, sparsity: SparsityMechanism):
        """
        Initialize base SAE.

        Args:
            sparsity: Sparsity mechanism (TopK, L1, etc.)
        """
        super().__init__()
        self.sparsity = sparsity

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to pre-activation features.

        Args:
            x: Input activations, shape (batch_size, input_dim)

        Returns:
            Pre-activation features, shape (batch_size, latent_dim)
        """
        pass

    @abstractmethod
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse features to reconstructed input.

        Args:
            features: Sparse features, shape (batch_size, latent_dim)

        Returns:
            Reconstructed input, shape (batch_size, input_dim)
        """
        pass

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass (template method - same for all SAEs).

        Flow:
        1. encode(x) → pre_activation features
        2. sparsity.apply(pre_activation) → sparse features
        3. decode(sparse_features) → reconstruction

        Args:
            x: Input activations, shape (batch_size, input_dim)

        Returns:
            Tuple of (reconstructed, sparse_features, pre_activation):
                - reconstructed: Reconstructed input, shape (batch_size, input_dim)
                - sparse_features: Sparse latent representation, shape (batch_size, latent_dim)
                - pre_activation: Pre-activation features (for computing sparsity penalty),
                                 shape (batch_size, latent_dim)
        """
        # Encode to pre-activation features
        pre_activation = self.encode(x)

        # Apply sparsity mechanism (ReLU + TopK, or ReLU + L1, etc.)
        sparse_features = self.sparsity.apply(pre_activation)

        # Decode back to input space
        reconstructed = self.decode(sparse_features)

        return reconstructed, sparse_features, pre_activation

    @property
    @abstractmethod
    def input_dim(self) -> int:
        """Dimension of input activations."""
        pass

    @property
    @abstractmethod
    def probe_dim(self) -> int:
        """Dimension of latent (sparse) representation."""
        pass

    @abstractmethod
    def anti_cheat(self) -> int:
        """
        The SAE model will attempt to cheat its learning by different methods. Avoid this by calling this method after
         each optimizer step.
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration dict for saving/loading.

        Should include all parameters needed to reconstruct the model,
        including the 'type' field identifying the SAE class.

        Returns:
            Dict containing model configuration
        """
        pass

"""
Simple single-layer Sparse Autoencoder.

This is a baseline SAE with one encoder layer and one decoder layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from .base import BaseSAE


class SimpleSAE(BaseSAE):
    """
    Single-layer sparse autoencoder.

    Architecture:
    - Encoder: Single linear layer that expands dimensions (e.g., 512 -> 2048)
    - Decoder: Single linear layer that contracts back (e.g., 2048 -> 512)
    - Sparsity: Enforced via L1 penalty or TopK activation

    Why overcomplete? Having more features than input dimensions (e.g., 4x expansion)
    gives space for different semantic concepts to occupy separate dimensions rather
    than being compressed together.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        top_k: int | None = 0
    ):
        """
        Initialize the simple SAE.

        Args:
            input_dim: Dimension of input activations (e.g., 512 for Pythia-70M layer)
            hidden_dim: Dimension of sparse representation (typically 2-8x input_dim)
            top_k: If > 0, use TopK activation (keep only top-k features per sample).
                   If 0, use standard ReLU (sparsity via L1 penalty during training).
        """
        super().__init__()

        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self.top_k = top_k

        # Learnable bias for centering inputs
        self.bias_pre = nn.Parameter(torch.zeros(input_dim))

        # Encoder: maps activations to sparse features
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=False)

        # Learnable bias for encoder output
        self.bias_enc = nn.Parameter(torch.zeros(hidden_dim))

        # Decoder: reconstructs original activations from sparse features
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)

        # Initialize weights with Xavier initialization for better convergence
        nn.init.xavier_uniform_(self.encoder.weight)
        # nn.init.zeros_(self.encoder.bias)
        nn.init.xavier_uniform_(self.decoder.weight)
        # nn.init.zeros_(self.decoder.bias)

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def latent_dim(self) -> int:
        return self._hidden_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the SAE.

        Args:
            x: Input activations, shape (batch_size, input_dim)

        Returns:
            Tuple of (reconstructed, sparse_features)
        """
        # Center input and encode
        pre_activation = self.encoder(x - self.bias_pre) + self.bias_enc

        # Apply activation function (TopK or ReLU)
        if self.top_k and self.top_k > 0:
            sparse_features = self._topk_activation(F.relu(pre_activation), self.top_k)
        else:
            sparse_features = F.relu(pre_activation)

        # Decode back to original space
        reconstructed = self.decoder(sparse_features) + self.bias_pre

        return reconstructed, sparse_features

    def _topk_activation(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """
        Keep only the top-k values per sample, zero out the rest.

        This is differentiable - gradients flow through the selected positions.

        Args:
            x: Input tensor, shape (batch_size, hidden_dim)
            k: Number of top values to keep per sample

        Returns:
            Tensor with same shape, but only top-k values non-zero per row
        """
        # Find top-k values and their indices for each sample
        topk_values, topk_indices = torch.topk(x, k=k, dim=-1)

        # Create output filled with zeros
        result = torch.zeros_like(x)

        # Scatter the top-k values back to their original positions
        result.scatter_(-1, topk_indices, topk_values)

        return result

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for saving/loading."""
        return {
            "type": "SimpleSAE",
            "input_dim": self._input_dim,
            "hidden_dim": self._hidden_dim,
            "top_k": self.top_k,
        }

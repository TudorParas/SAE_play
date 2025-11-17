"""
Deep (multi-layer) Sparse Autoencoder.

A sparse autoencoder with multiple encoder and decoder layers for learning
hierarchical feature representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any
from .base import BaseSAE


class DeepSAE(BaseSAE):
    """
    Multi-layer sparse autoencoder.

    Architecture:
    - Encoder: Multiple linear layers that progressively expand dimensions
      (e.g., 512 -> 1024 -> 2048)
    - Decoder: Multiple linear layers that progressively contract back
      (e.g., 2048 -> 1024 -> 512)
    - Sparsity: Enforced via L1 penalty or TopK activation on the latent representation

    Why deeper? Multiple layers can learn hierarchical feature representations,
    potentially capturing more complex patterns than a single-layer SAE. Each layer
    can learn different levels of abstraction.

    Example:
        >>> # Create a 3-layer encoder, 2-layer decoder
        >>> sae = DeepSAE(
        ...     input_dim=512,
        ...     encoder_hidden_dims=[1024, 2048, 4096],
        ...     decoder_hidden_dims=[2048, 1024]
        ... )
        >>> # This creates: 512 -> 1024 -> 2048 -> 4096 -> 2048 -> 1024 -> 512
    """

    def __init__(
        self,
        input_dim: int,
        encoder_hidden_dims: List[int],
        decoder_hidden_dims: List[int] | None = None,
        top_k: int | None = 0
    ):
        """
        Initialize the deep SAE.

        Args:
            input_dim: Dimension of input activations (e.g., 512 for Pythia-70M layer)
            encoder_hidden_dims: List of hidden dimensions for encoder layers.
                                The last dimension is the latent (sparse) dimension.
                                E.g., [1024, 2048] creates: input -> 1024 -> 2048
            decoder_hidden_dims: List of hidden dimensions for decoder layers.
                                If None or empty, uses a single linear layer from
                                latent space back to input space.
                                E.g., [1024] creates: latent -> 1024 -> input
            top_k: If > 0, use TopK activation (keep only top-k features per sample).
                   If 0 or None, use standard ReLU (sparsity via L1 penalty during training).
        """
        super().__init__()

        if not encoder_hidden_dims:
            raise ValueError("encoder_hidden_dims must contain at least one dimension")

        self._input_dim = input_dim
        self._latent_dim = encoder_hidden_dims[-1]  # Last encoder layer is latent space
        self.encoder_hidden_dims = encoder_hidden_dims
        self.decoder_hidden_dims = decoder_hidden_dims if decoder_hidden_dims else []
        self.top_k = top_k

        # Learnable biases for input/output centering
        self.bias_pre = nn.Parameter(torch.zeros(input_dim))
        self.bias_latent = nn.Parameter(torch.zeros(self._latent_dim))

        # Build encoder layers
        # E.g., if input_dim=512 and encoder_hidden_dims=[1024, 2048, 4096]:
        # Layer 0: 512 -> 1024
        # Layer 1: 1024 -> 2048
        # Layer 2: 2048 -> 4096
        encoder_layers = []
        layer_dims = [input_dim] + encoder_hidden_dims
        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            encoder_layers.append(nn.Linear(in_dim, out_dim, bias=False))
        self.encoder_layers = nn.ModuleList(encoder_layers)

        # Build decoder layers
        # E.g., if latent_dim=4096, decoder_hidden_dims=[2048, 1024], input_dim=512:
        # Layer 0: 4096 -> 2048
        # Layer 1: 2048 -> 1024
        # Layer 2: 1024 -> 512
        decoder_layers = []
        layer_dims = [self._latent_dim] + self.decoder_hidden_dims + [input_dim]
        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            decoder_layers.append(nn.Linear(in_dim, out_dim, bias=False))
        self.decoder_layers = nn.ModuleList(decoder_layers)

        # Initialize weights with Xavier initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize all layer weights with Xavier uniform initialization."""
        for layer in self.encoder_layers:
            nn.init.xavier_uniform_(layer.weight)
        for layer in self.decoder_layers:
            nn.init.xavier_uniform_(layer.weight)

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the deep SAE.

        Args:
            x: Input activations, shape (batch_size, input_dim)

        Returns:
            Tuple of (reconstructed, sparse_features):
                - reconstructed: Reconstructed activations, shape (batch_size, input_dim)
                - sparse_features: Sparse latent representation, shape (batch_size, latent_dim)
        """
        # Center input
        x = x - self.bias_pre

        # Encode through multiple layers
        for layer in self.encoder_layers:
            x = F.relu(layer(x))

        # Apply bias and sparsity activation to latent representation
        latent = x + self.bias_latent

        # Apply TopK or ReLU sparsity
        if self.top_k and self.top_k > 0:
            sparse_features = self._topk_activation(F.relu(latent), self.top_k)
        else:
            sparse_features = F.relu(latent)

        # Decode through multiple layers
        reconstructed = sparse_features
        for i, layer in enumerate(self.decoder_layers):
            reconstructed = layer(reconstructed)
            # Apply ReLU to all decoder layers except the last
            # (last layer should be able to output any value)
            if i < len(self.decoder_layers) - 1:
                reconstructed = F.relu(reconstructed)

        # Add back the input centering bias
        reconstructed = reconstructed + self.bias_pre

        return reconstructed, sparse_features

    def _topk_activation(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """
        Keep only the top-k values per sample, zero out the rest.

        This is differentiable - gradients flow through the selected positions.

        Args:
            x: Input tensor, shape (batch_size, latent_dim)
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
            "type": "DeepSAE",
            "input_dim": self._input_dim,
            "encoder_hidden_dims": self.encoder_hidden_dims,
            "decoder_hidden_dims": self.decoder_hidden_dims,
            "top_k": self.top_k,
        }
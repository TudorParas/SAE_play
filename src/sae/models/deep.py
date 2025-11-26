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
from ..sparsity import SparsityMechanism


class DeepSAE(BaseSAE):
    """
    Multi-layer sparse autoencoder.

    Architecture:
    - Encoder: Multiple linear layers that progressively expand dimensions
      (e.g., 512 -> 1024 -> 2048)
    - Decoder: Multiple linear layers that progressively contract back
      (e.g., 2048 -> 1024 -> 512)
    - Sparsity: Controlled by SparsityMechanism on the latent representation

    Why deeper? Multiple layers can learn hierarchical feature representations,
    potentially capturing more complex patterns than a single-layer SAE. Each layer
    can learn different levels of abstraction.

    The forward() method is inherited from BaseSAE (template method pattern).
    This class only defines encode() and decode().
    """

    def __init__(
        self,
        input_dim: int,
        encoder_hidden_dims: List[int],
        decoder_hidden_dims: List[int] | None = None,
        sparsity: SparsityMechanism | None = None
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
            sparsity: Sparsity mechanism. If None, uses TopKSparsity(k=64) as default.
        """
        # Import here to avoid circular dependency
        if sparsity is None:
            from ..sparsity import TopKSparsity
            sparsity = TopKSparsity(k=64)

        super().__init__(sparsity)

        if not encoder_hidden_dims:
            raise ValueError("encoder_hidden_dims must contain at least one dimension")

        self._input_dim = input_dim
        self._latent_dim = encoder_hidden_dims[-1]  # Last encoder layer is latent space
        self.encoder_hidden_dims = encoder_hidden_dims
        self.decoder_hidden_dims = decoder_hidden_dims if decoder_hidden_dims else []

        # Learnable biases for input/output centering
        self.bias_pre = nn.Parameter(torch.zeros(input_dim))
        self.bias_latent = nn.Parameter(torch.zeros(self._latent_dim))

        # Global scale parameter for spectral-norm deep decoders
        # Since spectral norm constrains the network gain to ~1.0, we need a learnable
        # scalar to match the magnitude of LLM activations (which can have norms of 100-500).
        # This decouples the "shape" (learned by constrained decoder) from "magnitude" (this scalar).
        # Initialized to 1.0; the model will learn to increase it as needed.
        self.global_scale = nn.Parameter(torch.tensor(1.0))

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

        # Build decoder layers with spectral normalization
        # E.g., if latent_dim=4096, decoder_hidden_dims=[2048, 1024], input_dim=512:
        # Layer 0: 4096 -> 2048
        # Layer 1: 2048 -> 1024
        # Layer 2: 1024 -> 512
        #
        # Spectral norm prevents decoder weight explosion by constraining
        # the largest singular value to ≤ 1. This is critical for training
        # stability, especially with JumpReLU and deep architectures.
        decoder_layers = []
        layer_dims = [self._latent_dim] + self.decoder_hidden_dims + [input_dim]
        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            layer = nn.Linear(in_dim, out_dim, bias=False)
            # ToDo: only apply spectral norm if deep; otherwise use anti_cheat.
            layer = nn.utils.spectral_norm(layer)  # Apply spectral normalization
            decoder_layers.append(layer)
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
    def probe_dim(self) -> int:
        return self._latent_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to pre-activation latent features.

        Flow:
        - Center input
        - Pass through encoder layers with ReLU between
        - Add latent bias
        - Return pre-activation features

        Args:
            x: Input activations, shape (batch_size, input_dim)

        Returns:
            Pre-activation latent features, shape (batch_size, latent_dim)
        """
        # Center input
        x = x - self.bias_pre

        # Encode through multiple layers with ReLU activations between
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)
            # Apply ReLU to all encoder layers except the last
            if i < len(self.encoder_layers) - 1:
                x = F.relu(x)

        # Add latent bias to get pre-activation features
        pre_activation = x + self.bias_latent

        return pre_activation

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse features to reconstructed input.

        Flow:
        - Pass through decoder layers
        - Apply ReLU between layers (except after last layer)
        - Add back centering bias

        Args:
            features: Sparse features, shape (batch_size, latent_dim)

        Returns:
            Reconstructed input, shape (batch_size, input_dim)
        """
        reconstructed = features

        # Decode through multiple layers
        # Spectral norm is applied automatically via hooks (set during initialization)
        for i, layer in enumerate(self.decoder_layers):
            reconstructed = layer(reconstructed)
            # Apply ReLU to all decoder layers except the last
            # (last layer should be able to output any value, including negative)
            if i < len(self.decoder_layers) - 1:
                reconstructed = F.relu(reconstructed)

        # Apply global scale and add back the input centering bias
        # The spectral-norm layers produce the "shape" of the reconstruction (gain ~1.0)
        # The global_scale allows the magnitude to match the actual activation norms
        reconstructed = (reconstructed * self.global_scale) + self.bias_pre

        return reconstructed

    @torch.no_grad()
    def anti_cheat(self):
        """
        Prevent decoder weight explosion (anti-cheat mechanism).

        For DeepSAE, this is handled automatically via spectral normalization
        applied to decoder layers during initialization. The spectral norm
        constraint (max singular value ≤ 1) is enforced via forward hooks.

        No manual intervention needed during training.
        """
        pass  # Spectral norm handles this automatically

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for saving/loading."""
        return {
            "type": "DeepSAE",
            "input_dim": self._input_dim,
            "encoder_hidden_dims": self.encoder_hidden_dims,
            "decoder_hidden_dims": self.decoder_hidden_dims,
            # Note: Sparsity mechanism should be saved separately by checkpoint system
        }

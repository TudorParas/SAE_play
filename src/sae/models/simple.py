"""
Simple single-layer Sparse Autoencoder.

This is a baseline SAE with one encoder layer and one decoder layer.
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from .base import BaseSAE
from ..sparsity import SparsityMechanism, TopKSparsity, L1Sparsity


class SimpleSAE(BaseSAE):
    """
    Single-layer sparse autoencoder.

    Architecture:
    - Encoder: Single linear layer that expands dimensions (e.g., 512 -> 2048)
    - Decoder: Single linear layer that contracts back (e.g., 2048 -> 512)
    - Sparsity: Controlled by SparsityMechanism (TopK, L1, etc.)

    Why overcomplete? Having more features than input dimensions (e.g., 4x expansion)
    gives space for different semantic concepts to occupy separate dimensions rather
    than being compressed together.

    The forward() method is inherited from BaseSAE (template method pattern).
    This class only defines encode() and decode().
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        sparsity: SparsityMechanism,
    ):
        """
        Initialize the simple SAE.

        Args:
            input_dim: Dimension of input activations (e.g., 512 for Pythia-70M layer)
            hidden_dim: Dimension of sparse representation (typically 2-8x input_dim)
            sparsity: Sparsity mechanism (TopKSparsity, L1Sparsity, etc.)
        """
        super().__init__(sparsity)

        self._input_dim = input_dim
        self._hidden_dim = hidden_dim

        # Learnable bias for centering inputs
        self.bias_pre = nn.Parameter(torch.zeros(input_dim))

        # Encoder: maps activations to sparse features
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=False)

        # Learnable bias for encoder output
        if isinstance(sparsity, L1Sparsity):
            self.bias_enc = nn.Parameter(torch.zeros(hidden_dim))
        else:
            # TopK sparsity does not need a separate bias.
            self.bias_enc = None

        # Decoder: reconstructs original activations from sparse features
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)

        # Initialize weights with Xavier initialization for better convergence
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        # Initialise encoder to transpose of decoder.
        self.encoder.weight.data = self.decoder.weight.data.t().clone()

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def probe_dim(self) -> int:
        return self._hidden_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to pre-activation features.

        Flow: x → center → linear → add bias → pre_activation

        Args:
            x: Input activations, shape (batch_size, input_dim)

        Returns:
            Pre-activation features, shape (batch_size, hidden_dim)
        """
        # Apply bias_pre, then encode
        encoded = self.encoder(x - self.bias_pre)
        if self.bias_enc is not None:
            encoded = encoded + self.bias_enc

        return  encoded

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse features to reconstructed input.

        Flow: sparse_features → linear → add bias → reconstruction

        Args:
            features: Sparse features, shape (batch_size, hidden_dim)

        Returns:
            Reconstructed input, shape (batch_size, input_dim)
        """
        # Decode and add back the centering bias
        return self.decoder(features) + self.bias_pre

    def decode_sparse(self, indices: torch.Tensor, values: torch.Tensor, apply_bias: bool = True) -> torch.Tensor:
        """
        Efficient sparse decode using only active features.

        Instead of computing dense_features @ decoder_weights (where dense_features
        is mostly zeros), this gathers only the k relevant decoder weight vectors
        and computes a weighted sum. This is much faster when k << hidden_dim.

        Speedup example: k=128, hidden_dim=16384 → 128x fewer FLOPs

        Args:
            indices: Indices of active features per sample, shape (batch_size, k)
            values: Values of active features per sample, shape (batch_size, k)
            apply_bias: whether to add bias_pre. Set to fasle during Auxiliary K computation.

        Returns:
            Reconstructed input, shape (batch_size, input_dim)
        """
        # STEP 1: Gather decoder weights for active features only
        # decoder.weight: (input_dim, hidden_dim)
        # decoder.weight.T: (hidden_dim, input_dim) - use as embedding lookup table
        # indices: (batch_size, k) - which features are active
        # Result: (batch_size, k, input_dim) - decoder weights for active features
        gathered_weights = torch.nn.functional.embedding(indices, self.decoder.weight.T)

        # STEP 2: Weighted sum of decoder weights
        # values: (batch_size, k) - activation strength for each active feature
        # gathered_weights: (batch_size, k, input_dim) - decoder vectors
        # einsum 'bk,bki->bi': for each sample, sum over k (weighted by values)
        # Result: (batch_size, input_dim)
        reconstruction = torch.einsum('bk,bki->bi', values, gathered_weights)
        # STEP 3: Add decoder bias (same as dense decode)
        if apply_bias:
            reconstruction = reconstruction + self.bias_pre
        return reconstruction

    @torch.no_grad()
    def anti_cheat(self):
        """
        Prevent decoder weight explosion via L2 column normalization.

        Each decoder column (feature's reconstruction direction) is normalized
        to unit L2 norm. This prevents the model from "cheating" by growing
        decoder weights instead of learning sparse features.

        TODO: Consider switching to spectral normalization (like DeepSAE) for
              consistency. Spectral norm is automatic via hooks and may be more
              principled, though L2 column norm is standard in SAE literature.
        """
        # Normalize decoder columns (dim=0) to unit L2 norm
        self.decoder.weight.data[:] = torch.nn.functional.normalize(self.decoder.weight.data, p=2, dim=0)

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for saving/loading."""
        return {
            "type": "SimpleSAE",
            "input_dim": self._input_dim,
            "hidden_dim": self._hidden_dim,
            # Note: Sparsity mechanism should be saved separately by checkpoint system
        }

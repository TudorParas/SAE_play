"""
Dead latent tracking for auxiliary loss (AuxK).

Tracks which latents have been "dead" (not activating) for too long,
enabling auxiliary reconstruction loss as described in:
https://arxiv.org/pdf/2406.04093 (Gao et al., 2024, Appendix A.2)
"""

import torch
from typing import Optional


class DeadLatentTracker:
    """
    Tracks activation history to identify dead latents.

    A latent is considered "dead" if it hasn't activated above threshold
    for a certain number of tokens.

    Args:
        num_latents: Total number of latent features
        dead_threshold_tokens: Number of tokens without activation before flagged as dead
        activation_threshold: Minimum pre-activation value to count as "active" (default: 0.0)
        device: Device to store tracking tensors
    """

    def __init__(
        self,
        num_latents: int,
        dead_threshold_tokens: int = 10_000_000,
        activation_threshold: float = 0.0,
        device: str = "cpu",
    ):
        self.num_latents = num_latents
        self.dead_threshold_tokens = dead_threshold_tokens
        self.activation_threshold = activation_threshold
        self._device = device

        # Track tokens since last activation for each latent
        # Initialized to 0 (no tokens processed yet)
        self.tokens_since_activation = torch.zeros(
            num_latents, device=device, dtype=torch.long
        )

        # Track total tokens processed
        self.total_tokens = 0

    def update(self, pre_activation: torch.Tensor):
        """
        Update activation history with a batch of pre-activation features.

        Args:
            pre_activation: Pre-activation features from SAE encoder
                           Shape: (batch_size, num_latents)
        """
        batch_size = pre_activation.shape[0]
        self.total_tokens += batch_size

        # Check which latents activated in this batch (any sample)
        # Shape: (num_latents,)
        activated = (pre_activation > self.activation_threshold).any(dim=0)

        # Reset counter for activated latents, increment for others
        self.tokens_since_activation[activated] = 0
        self.tokens_since_activation[~activated] += batch_size

    def get_dead_mask(self) -> torch.Tensor:
        """
        Get boolean mask of latents that are currently dead.

        Returns:
            Boolean tensor of shape (num_latents,) where True indicates dead
        """
        is_dead = self.tokens_since_activation >= self.dead_threshold_tokens
        return is_dead

    def get_dead_latents(self) -> torch.Tensor:
        """
        Get indices of latents that are currently dead.

        Returns:
            Tensor of dead latent indices
        """
        return torch.where(self.get_dead_mask())[0]

    def get_top_k_dead_latents(self, k: int) -> Optional[torch.Tensor]:
        """
        Get top-k "most dead" latents (longest time without activation).

        Args:
            k: Number of dead latents to return

        Returns:
            Tensor of top-k dead latent indices, or None if no dead latents
        """
        dead_latents = self.get_dead_latents()

        if len(dead_latents) == 0:
            return None

        # If we have fewer dead latents than k, return all of them
        if len(dead_latents) <= k:
            return dead_latents

        # Get the k most dead (longest time without activation)
        tokens_since = self.tokens_since_activation[dead_latents]
        _, top_k_indices = torch.topk(tokens_since, k=k)

        return dead_latents[top_k_indices]

    def get_stats(self) -> dict:
        """
        Get statistics about dead latents.

        Returns:
            Dict with keys: num_dead, frac_dead, total_tokens
        """
        dead_latents = self.get_dead_latents()
        return {
            "num_dead": len(dead_latents),
            "frac_dead": len(dead_latents) / self.num_latents,
            "total_tokens": self.total_tokens,
        }

"""
PyTorch Dataset classes for SAE training and evaluation.

Provides:
- ActivationDataset: Wraps pre-extracted activations with centering
- split_activations: Split raw tensors into train/test (before centering)

Philosophy: Use standard PyTorch patterns (Dataset, DataLoader)
for explicit, composable data handling. Center using TRAIN mean only.
"""

import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional


class ActivationDataset(Dataset):
    """
    Dataset wrapper for pre-extracted activations.

    Centers activations by subtracting mean. Use the `mean` parameter to
    provide an external mean (e.g., train mean for test set).

    Args:
        activations: Tensor of activations, shape (num_samples, activation_dim)
        mean: Mean to use for centering. If None, computes from activations.

    Attributes:
        activations: The centered activation tensor
        mean: The mean used for centering
        dim: Activation dimension
    """

    def __init__(
        self,
        activations: torch.Tensor,
        mean: Optional[torch.Tensor] = None,
    ):
        if activations.dim() != 2:
            raise ValueError(f"Expected 2D tensor (num_samples, dim), got shape {activations.shape}")

        # Use provided mean or compute from data
        if mean is not None:
            self.mean = mean
        else:
            self.mean = activations.mean(dim=0, keepdim=True)

        # Center activations
        self.activations = activations - self.mean
        self.dim = activations.shape[1]

    def __len__(self) -> int:
        return len(self.activations)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.activations[idx]

    def get_mean(self) -> torch.Tensor:
        """Get the mean used for centering."""
        return self.mean


def split_activations(
    activations: torch.Tensor,
    train_frac: float = 0.9,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split raw activations into train and test tensors.

    Use this BEFORE creating ActivationDataset to avoid data leakage.
    The mean should be computed on the train portion only.

    Args:
        activations: Raw activation tensor, shape (num_samples, dim)
        train_frac: Fraction of data for training (0.0 to 1.0)
        seed: Random seed for reproducible splits

    Returns:
        Tuple of (train_activations, test_activations)
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be between 0 and 1, got {train_frac}")

    num_samples = activations.shape[0]
    train_size = int(num_samples * train_frac)

    # Generate random permutation
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_samples, generator=generator)

    # Split indices
    train_indices = perm[:train_size]
    test_indices = perm[train_size:]

    return activations[train_indices], activations[test_indices]

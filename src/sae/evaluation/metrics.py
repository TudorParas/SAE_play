"""
Metrics for evaluating SAE quality.
"""

import torch


def compute_reconstruction_loss(
    original: torch.Tensor,
    reconstructed: torch.Tensor
) -> float:
    """
    Compute mean squared error between original and reconstructed activations.

    Args:
        original: Original activations
        reconstructed: Reconstructed activations

    Returns:
        Mean squared error
    """
    return torch.nn.functional.mse_loss(original, reconstructed).item()


def compute_sparsity(
    sparse_features: torch.Tensor,
    threshold: float = 0.01
) -> dict:
    """
    Compute sparsity metrics for sparse features.

    Args:
        sparse_features: Sparse feature activations, shape (batch_size, hidden_dim)
        threshold: Values below this are considered "zero"

    Returns:
        Dictionary with:
            - 'num_active': Average number of active features per sample
            - 'pct_active': Percentage of features that are active
            - 'l0_norm': Average L0 norm (number of non-zero features)
            - 'l1_norm': Average L1 norm
    """
    # Count active features (> threshold)
    is_active = sparse_features > threshold
    num_active = is_active.float().sum(dim=1).mean().item()
    pct_active = (num_active / sparse_features.shape[1]) * 100

    # L0 norm (count of non-zero)
    l0_norm = (sparse_features != 0).float().sum(dim=1).mean().item()

    # L1 norm
    l1_norm = torch.abs(sparse_features).sum(dim=1).mean().item()

    return {
        'num_active': num_active,
        'pct_active': pct_active,
        'l0_norm': l0_norm,
        'l1_norm': l1_norm,
    }

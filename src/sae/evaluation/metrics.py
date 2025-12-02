"""
Metrics for evaluating SAE quality.
"""

import torch
from typing import Dict, Any

from ..models.base import BaseSAE


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


def compute_dead_features(
    sae: BaseSAE,
    activations: torch.Tensor,
    threshold: float = 0.01,
    batch_size: int = 256,
) -> Dict[str, Any]:
    """
    Compute fraction of features that never activate on given activations.

    A "dead" feature is one that never activates above the threshold for any
    input in the evaluation set. High dead feature fractions indicate the SAE
    isn't using its full capacity.

    Args:
        sae: The trained SAE model
        activations: Evaluation activations, shape (num_samples, input_dim)
                    Should be centered (mean-subtracted) before passing.
        threshold: Activation threshold to consider a feature "active"
        batch_size: Batch size for processing (for memory efficiency)

    Returns:
        Dictionary with:
            - 'count': Number of dead features
            - 'fraction': Fraction of features that are dead
            - 'threshold': The threshold used
            - 'total': Total number of features in SAE
            - 'alive_count': Number of features that activated at least once
    """
    device = next(sae.parameters()).device
    num_features = sae.probe_dim
    num_samples = activations.shape[0]

    # Track which features have ever been active
    ever_active = torch.zeros(num_features, dtype=torch.bool, device=device)

    # Process in batches to avoid OOM
    sae.eval()
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch = activations[i:i + batch_size].to(device)

            # Get sparse features from SAE
            _, sparse_features, _ = sae(batch)

            # Mark features that are active in this batch
            batch_active = (sparse_features > threshold).any(dim=0)
            ever_active = ever_active | batch_active

    # Count dead features
    alive_count = ever_active.sum().item()
    dead_count = num_features - alive_count
    dead_fraction = dead_count / num_features

    return {
        'count': dead_count,
        'fraction': dead_fraction,
        'threshold': threshold,
        'total': num_features,
        'alive_count': alive_count,
    }


def get_spectral_stats(feature_acts):
    r"""
    Computes the Effective Latent Dimension (ELD) based on the spectral properties
    of the SAE feature activations.

    The ELD measures the effective number of independent dimensions being used
    by the features. It is calculated using the Participation Ratio of the
    covariance eigenvalues.

    Maths
    -----
    Given eigenvalues λ of the covariance matrix, ELD is defined as:

           (Σ λ_i)²
    ELD = ──────────
            Σ (λ_i²)

    Interpretation
    --------------
    * Low ELD (≈ 1.0): The SAE has collapsed into a single dominant feature.
    * High ELD: The SAE is utilizing many distinct, independent directions.

    Args:
        sae_model: The trained SAE model.
        activation_batch: Tensor of shape [batch, d_model].

    Returns:
        dict: Contains 'ELD' and 'Top_Component_Explained_Var'.
    """
    # 2. Center the data
    feature_acts = feature_acts - feature_acts.mean(dim=0)
    # 3. Compute Singular Values (S) using SVD
    # We use SVD on the data matrix because it's numerically more stable
    # than eig(Covariance). eigenvalues = S**2 / (N-1)
    _, S, _ = torch.linalg.svd(feature_acts, full_matrices=False)

    # Convert singular values to eigenvalues of the covariance matrix
    eigenvalues = (S ** 2) / (feature_acts.shape[0] - 1)

    # 4. Calculate Metrics

    # A. Effective Latent Dimension (The "Tudor Metric")
    # Higher is better (implies more distinct features are being used)
    sum_eigen = torch.sum(eigenvalues)
    sum_sq_eigen = torch.sum(eigenvalues ** 2)
    eld = (sum_eigen ** 2) / sum_sq_eigen

    # B. Explained Variance of Top Component
    # Lower is better (implies no single "super feature" dominates)
    top_component_dominance = eigenvalues[0] / sum_eigen

    return {
        "ELD": eld.item(),
        "Top_Component_Explained_Var": top_component_dominance.item()
    }

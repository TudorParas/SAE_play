"""
Analyze learned SAE features.

Tools for understanding what sparse features represent by seeing which inputs
activate them.
"""

import torch
from typing import List, Dict, Any
from transformers import PreTrainedModel, PreTrainedTokenizer
from ..models.base import BaseSAE
from ..activations import extract_activations


def analyze_features(
    sae: BaseSAE,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
    activation_mean: torch.Tensor,
    layer_idx: int,
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Analyze which sparse features activate for different texts.

    This is where interpretability happens - you can see which learned features
    "light up" for different inputs.

    Args:
        sae: Trained sparse autoencoder
        model: Language model to extract activations from
        tokenizer: Tokenizer for the model
        texts: List of texts to analyze
        activation_mean: Mean of training activations (for centering)
        layer_idx: Which layer to extract activations from
        top_k: How many top features to return per text

    Returns:
        List of dictionaries (one per text) containing:
            - 'text': The input text
            - 'top_features': List of (feature_idx, activation_value) tuples
            - 'num_active': Number of active features (primary metric)
            - 'total_features': Total number of features in the SAE
            - 'pct_active': Percentage of features active
    """
    device = next(sae.parameters()).device
    results = []

    for text in texts:
        # Extract activations for this text
        activations = extract_activations(
            model=model,
            tokenizer=tokenizer,
            texts=[text],
            layer_idx=layer_idx
        )

        # Center using training mean
        centered_activations = (activations - activation_mean).to(device)

        # Pass through SAE
        with torch.no_grad():
            reconstructed, sparse_features, pre_activation = sae(centered_activations)

        # Average features across all tokens in this text
        avg_features = sparse_features.mean(dim=0)

        # Get top-k most active features
        top_values, top_indices = torch.topk(avg_features, k=min(top_k, len(avg_features)))

        top_features = [
            (idx.item(), val.item())
            for idx, val in zip(top_indices, top_values)
        ]

        # Compute active features
        threshold = 0.01
        num_active = (avg_features > threshold).sum().item()
        total_features = len(avg_features)
        pct_active = (num_active / total_features) * 100

        results.append({
            'text': text,
            'top_features': top_features,
            'num_active': num_active,
            'total_features': total_features,
            'pct_active': pct_active,
        })

    return results


def print_feature_analysis(results: List[Dict[str, Any]]):
    """
    Pretty print feature analysis results.

    Args:
        results: Output from analyze_features()
    """
    print("\n" + "=" * 60)
    print("FEATURE ANALYSIS")
    print("=" * 60)

    for result in results:
        print(f"\nText: '{result['text']}'")
        print(f"  Active features: {result['num_active']}/{result['total_features']} ({result['pct_active']:.1f}%)")
        print(f"  Top {len(result['top_features'])} features:")
        for idx, val in result['top_features']:
            print(f"    Feature {idx}: {val:.3f}")

    print("=" * 60 + "\n")


def get_spectral_stats(sae_model, activation_batch):
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
    # 1. Get Feature Activations (f(x))
    # Don't decode; we want to check the "code" itself.
    with torch.no_grad():
        # For Simple SAE: encoder(x)
        # For Deep SAE: encoder(x)
        feature_acts = sae_model.encode(activation_batch)  # [Batch, d_sae]

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

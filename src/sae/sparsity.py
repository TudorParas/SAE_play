"""
Sparsity mechanisms for SAEs.

A sparsity mechanism controls how features become sparse. It's used in two places:
1. In the SAE model's forward pass: apply(pre_activation) → sparse_features
2. In the loss computation: compute_penalty(pre_activation) → penalty_loss

This abstraction allows easy comparison of different sparsity approaches
(TopK, L1, learned thresholds, etc.) while keeping the model architecture constant.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparsityMechanism(ABC):
    """
    Base class for sparsity mechanisms.

    A sparsity mechanism defines:
    1. How to convert pre-activation features to sparse features (apply)
    2. What loss penalty to add during training (compute_penalty)
    """

    @abstractmethod
    def apply(self, pre_activation: torch.Tensor) -> torch.Tensor:
        """
        Apply sparsity to pre-activation features.

        Args:
            pre_activation: Pre-activation features from encoder, shape (batch, features)

        Returns:
            Sparse features, shape (batch, features)
        """
        pass

    @abstractmethod
    def compute_penalty(self, pre_activation: torch.Tensor) -> torch.Tensor:
        """
        Compute sparsity penalty for loss function.

        Args:
            pre_activation: Pre-activation features from encoder, shape (batch, features)

        Returns:
            Scalar loss penalty
        """
        pass



class TopKSparsity(SparsityMechanism):
    """
    TopK sparsity: Keep only top-k features per sample.

    Applies ReLU, then keeps only the k largest activations per sample.
    No L1 penalty needed since sparsity is enforced explicitly.

    Advantages:
    - Fixed, predictable sparsity level
    - Easy to compare across runs
    - Training stability

    Disadvantages:
    - Not adaptive to input complexity
    - Forces same sparsity for simple and complex inputs
    - Can waste capacity or lose information
    """

    def __init__(self, k: int):
        """
        Args:
            k: Number of features to keep active per sample
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self.k = k

    def apply(self, pre_activation: torch.Tensor) -> torch.Tensor:
        """
        Apply ReLU then keep top-k features per sample.

        Args:
            pre_activation: Shape (batch_size, num_features)

        Returns:
            Sparse features with exactly k non-zero values per sample
        """
        # First apply ReLU to get positive activations
        activated = F.relu(pre_activation)

        # Then keep only top-k
        topk_values, topk_indices = torch.topk(activated, k=self.k, dim=-1)

        # Create output with zeros
        result = torch.zeros_like(activated)

        # Scatter top-k values back to their positions
        result.scatter_(-1, topk_indices, topk_values)

        return result

    def compute_penalty(self, pre_activation: torch.Tensor) -> torch.Tensor:
        """
        No penalty needed - sparsity is enforced by TopK.

        Returns:
            Zero tensor (no penalty)
        """
        return torch.tensor(0.0, device=pre_activation.device)


class L1Sparsity(SparsityMechanism):
    """
    L1 sparsity: Penalize L1 norm of activations.

    Applies ReLU, then adds L1 penalty to loss. Sparsity level is adaptive
    based on the trade-off between reconstruction and penalty.

    The penalty magnitude is controlled by the training pipeline's sparsity_schedule,
    not by this class. This compute_penalty() returns mean(|ReLU(features)|),
    which the pipeline multiplies by the current sparsity coefficient.

    Advantages:
    - Adaptive sparsity per input (simple inputs → fewer features)
    - More natural/principled
    - Better for variable complexity inputs
    - Important for circuit discovery (circuits have natural sizes)

    Disadvantages:
    - Harder to compare across runs (sparsity varies)
    - Requires tuning penalty coefficient via sparsity_schedule
    - Less stable training
    """

    def apply(self, pre_activation: torch.Tensor) -> torch.Tensor:
        """
        Apply ReLU activation.

        For L1 sparsity, we just use ReLU. The sparsity emerges from the
        L1 penalty during training, not from explicit topk selection.

        Args:
            pre_activation: Shape (batch_size, num_features)

        Returns:
            ReLU-activated features
        """
        return F.relu(pre_activation)

    def compute_penalty(self, pre_activation: torch.Tensor) -> torch.Tensor:
        """
        Compute L1 penalty on activations.

        We penalize the L1 norm after ReLU. This encourages features to be
        exactly zero rather than just small.

        Note: This returns the raw penalty value (mean L1 norm). The training
        pipeline multiplies this by the sparsity coefficient from sparsity_schedule.

        Args:
            pre_activation: Shape (batch_size, num_features)

        Returns:
            Scalar L1 penalty: mean(|ReLU(features)|)
        """
        activated = F.relu(pre_activation)
        return torch.abs(activated).mean()


class NoSparsity(SparsityMechanism):
    """
    No sparsity enforcement - just ReLU.

    Useful for debugging or as a baseline. The autoencoder will learn
    dense representations.
    """

    def apply(self, pre_activation: torch.Tensor) -> torch.Tensor:
        """Apply ReLU with no sparsity constraint."""
        return F.relu(pre_activation)

    def compute_penalty(self, pre_activation: torch.Tensor) -> torch.Tensor:
        """No penalty."""
        return torch.tensor(0.0, device=pre_activation.device)


class JumpReLUSparsity(nn.Module, SparsityMechanism):  # Must inherit nn.Module for parameters
    def __init__(self, num_features: int, bandwidth: float = 10.0, init_threshold: float = 0.001):
        super().__init__()
        self.num_features = num_features
        self.bandwidth = bandwidth

        # Learnable Threshold: One per feature
        # Init low (0.001) to prevent immediate feature death
        self.threshold = nn.Parameter(torch.full((num_features,), init_threshold))

    def apply(self, pre_activation: torch.Tensor) -> torch.Tensor:
        """
        Returns the sparse activations.
        Forward: Heaviside step function (Hard).
        Backward: Sigmoid function (Soft).
        """
        # 1. ReLU first (standard SAE practice)
        x = F.relu(pre_activation)

        # 2. Create the "Hard" mask (0 or 1) - Used for forward pass
        mask_hard = (x > self.threshold).float()

        # 3. Create the "Soft" mask (Sigmoid) - Used for gradients
        # bandwidth controls how steep the gradient is around the threshold
        mask_soft = torch.sigmoid((x - self.threshold) * self.bandwidth)

        # 4. Straight-Through Estimator (STE)
        # We detach 'mask_soft' so the forward pass value is exactly 'mask_hard'.
        # But gradients flow through 'mask_soft' to update the encoder and threshold.
        mask = mask_hard - mask_soft.detach() + mask_soft

        return x * mask

    def compute_penalty(self, pre_activation: torch.Tensor) -> torch.Tensor:
        """
        Returns the L0 Proxy loss to be added to the total loss.
        This drives the thresholds UP (to minimize active features).
        """
        x = F.relu(pre_activation)

        # We re-calculate the soft mask. 
        # Minimizing this minimizes the sum of sigmoids (approx active count)
        mask_soft = torch.sigmoid((x - self.threshold) * self.bandwidth)

        # Return mean expected L0 per token
        return mask_soft.sum(dim=-1).mean()
    
    @torch.no_grad()
    def clamp_thresholds(self, min_val=1e-4, max_val=10.0):
        """
        Call this after optimizer.step()!
        Prevents thresholds from going negative (dense) or infinite (dead).
        """
        self.threshold.data.clamp_(min=min_val, max=max_val)
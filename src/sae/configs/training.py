"""
Training configuration for SAE experiments.

Defines optimizer, learning rate schedules, and training hyperparameters.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .lr_schedule import LRScheduleConfig


@dataclass
class AuxKConfig:
    """
    Configuration for AuxK auxiliary loss to combat dead latents.

    Based on "Scaling and evaluating sparse autoencoders" (Gao et al., 2024).
    https://arxiv.org/pdf/2406.04093 (Appendix A.2)

    Attributes:
        coefficient: Loss weighting (α in paper, typically 1/32)
        k: Number of dead latents to use (k_aux in paper, typically 512)
        dead_threshold_tokens: Tokens without activation before flagged dead (typically 10M)
    """

    coefficient: float = 1 / 32
    k: int = 512
    dead_threshold_tokens: int = 10_000_000


@dataclass
class TrainingConfig:
    """
    Configuration for SAE training.

    Attributes:
        num_epochs: Number of training epochs
        lr: Base learning rate
        lr_schedule: Learning rate schedule config (or None for constant LR)
        sparsity_warmup_value: Initial sparsity penalty (warmup phase)
        sparsity_end_value: Final sparsity penalty
        sparsity_warmup_epochs: Number of epochs for sparsity warmup
        random_seed: Random seed for training
        use_compile: Use torch.compile for faster training (PyTorch 2.0+)
        use_amp: Use automatic mixed precision training (bfloat16)
    """

    num_epochs: int
    lr: float
    sparsity_warmup_value: float
    sparsity_end_value: float
    sparsity_warmup_epochs: int
    random_seed: int
    lr_schedule: "LRScheduleConfig | None" = None
    use_compile: bool = False
    use_amp: bool = False

    # AuxK auxiliary loss for combating dead latents (optional)
    auxk: AuxKConfig | None = None


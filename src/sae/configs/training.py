"""
Training configuration for SAE experiments.

Defines optimizer, learning rate schedules, and training hyperparameters.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .lr_schedule import LRScheduleConfig


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


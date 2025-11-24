"""
Training configuration for SAE experiments.

Defines optimizer, learning rate schedules, and training hyperparameters.
"""

from dataclasses import dataclass
from typing import Optional, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler


@dataclass
class TrainingConfig:
    """
    Configuration for SAE training.

    Attributes:
        num_epochs: Number of training epochs
        lr: Base learning rate
        lr_schedule: Learning rate schedule ("onecycle", "constant", or None)
        lr_max: Maximum LR for OneCycleLR (if using onecycle)
        lr_warmup_pct: Warmup percentage for OneCycleLR (if using onecycle)
        sparsity_warmup_value: Initial sparsity penalty (warmup phase)
        sparsity_end_value: Final sparsity penalty
        sparsity_warmup_epochs: Number of epochs for sparsity warmup
        random_seed: Random seed for training
    """

    num_epochs: int = 20
    lr: float = 1e-3
    lr_schedule: Optional[Literal["onecycle", "constant"]] = None
    lr_max: float = 1e-3
    lr_warmup_pct: float = 0.1
    sparsity_warmup_value: float = 1e-2
    sparsity_end_value: float = 2.0
    sparsity_warmup_epochs: int = 2
    random_seed: int = 53

    def resolve_lr_schedule(
        self, optimizer: "Optimizer", total_steps: int
    ) -> Optional["LRScheduler"]:
        """
        Create and return the configured learning rate scheduler.

        Args:
            optimizer: The optimizer to attach the scheduler to
            total_steps: Total number of training steps

        Returns:
            LR scheduler instance or None for constant learning rate
        """
        if self.lr_schedule == "onecycle":
            return self._resolve_onecycle(optimizer, total_steps)
        else:
            # No schedule (constant LR)
            return None

    def _resolve_onecycle(
        self, optimizer: "Optimizer", total_steps: int
    ) -> "LRScheduler":
        """Helper to create OneCycleLR scheduler."""
        from torch.optim.lr_scheduler import OneCycleLR

        return OneCycleLR(
            optimizer,
            max_lr=self.lr_max,
            total_steps=total_steps,
            pct_start=self.lr_warmup_pct,
            anneal_strategy="cos",
            div_factor=10,
            final_div_factor=10,
        )

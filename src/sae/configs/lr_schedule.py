"""
Learning rate schedule configurations.

Defines different LR scheduling strategies that can be serialized and resolved
at training time.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler, SequentialLR


@dataclass
class LRScheduleConfig(ABC):
    """
    Base class for learning rate schedule configurations.

    All LR schedule configs must implement resolve() which creates the actual
    PyTorch LR scheduler at training time.
    """

    @abstractmethod
    def resolve(
        self, optimizer: "Optimizer", num_epochs: int, steps_per_epoch: int
    ) -> "LRScheduler":
        """
        Create and return a PyTorch LR scheduler.

        Args:
            optimizer: The optimizer to attach the scheduler to
            num_epochs: Total number of training epochs
            steps_per_epoch: Number of optimization steps per epoch

        Returns:
            PyTorch LR scheduler instance
        """
        ...


@dataclass
class OneCycleLRConfig(LRScheduleConfig):
    """
    OneCycleLR schedule with warmup and cosine annealing.

    Attributes:
        max_lr: Peak learning rate
        warmup_pct: Fraction of training for warmup (0.0 to 1.0)
        div_factor: Initial LR = max_lr / div_factor
        final_div_factor: Final LR = max_lr / (div_factor * final_div_factor)
    """

    max_lr: float
    warmup_pct: float = 0.1
    div_factor: float = 10.0
    final_div_factor: float = 10.0

    def resolve(
        self, optimizer: "Optimizer", num_epochs: int, steps_per_epoch: int
    ) -> "LRScheduler":
        """Create OneCycleLR scheduler."""
        from torch.optim.lr_scheduler import OneCycleLR

        total_steps = num_epochs * steps_per_epoch

        return OneCycleLR(
            optimizer,
            max_lr=self.max_lr,
            total_steps=total_steps,
            pct_start=self.warmup_pct,
            anneal_strategy="cos",
            div_factor=self.div_factor,
            final_div_factor=self.final_div_factor,
        )


# ToDo: this is an example of how we could do composition
# @dataclass
# class WarmupWrapper(LRScheduleConfig):
#     warmup_steps: int
#     inner: LRScheduleConfig  # nest configs
#
#     def resolve(
#         self, optimizer: "Optimizer", num_epochs: int, steps_per_epoch: int
#     ) -> "LRScheduler":
#         inner_scheduler = self.inner.resolve(optimizer, num_epochs, steps_per_epoch)
#         return SequentialLR(optimizer, [warmup_sched, inner_scheduler], [self.warmup_steps])
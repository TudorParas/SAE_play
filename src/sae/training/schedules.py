"""
Schedules for training hyperparameters (EPOCH-BASED).

These are simple callable objects that return a hyperparameter value based on the
current epoch number (0, 1, 2, ...). They can be used for:
- Sparsity penalty (L1 coefficient) annealing
- Learning rate scheduling (epoch-based)
- Any other hyperparameter that needs to vary during training

Each schedule is a callable that takes an epoch number and returns the parameter value.

For step-based learning rate scheduling, use torch.optim.lr_scheduler instead.

ToDo: need to ensure that schedules designed this way offers the start / stop guarantees that are offered by the
   torch.optim.lr_scheduler.LearningRateScheduler. Need to Checkpointing & resuming without pain
"""

from abc import ABC, abstractmethod


class Schedule(ABC):
    """Base class for parameter schedules."""

    @abstractmethod
    def __call__(self, step: int) -> float:
        """
        Get the parameter value at a given epoch.

        Args:
            step: Current training epoch (0-indexed)

        Returns:
            Parameter value
        """
        pass


class ConstantSchedule(Schedule):
    """Constant value schedule (no change over time)."""

    def __init__(self, value: float):
        """
        Args:
            value: The constant value to return
        """
        self.value = value

    def __call__(self, step: int) -> float:
        return self.value


class LinearSchedule(Schedule):
    """Linear interpolation between two values."""

    def __init__(self, start_value: float, end_value: float, start_step: int, end_step: int):
        """
        Args:
            start_value: Value at start_step
            end_value: Value at end_step
            start_step: Step where interpolation begins
            end_step: Step where interpolation ends
        """
        self.start_value = start_value
        self.end_value = end_value
        self.start_step = start_step
        self.end_step = end_step

    def __call__(self, step: int) -> float:
        if step <= self.start_step:
            return self.start_value
        elif step >= self.end_step:
            return self.end_value
        else:
            # Linear interpolation
            progress = (step - self.start_step) / (self.end_step - self.start_step)
            return self.start_value + progress * (self.end_value - self.start_value)


class WarmupThenLinearSchedule(Schedule):
    """
    Constant value during warmup, then linear interpolation.

    Common use cases:
    - Learning rate: Warmup from 0 to max LR, then decay to lower LR
    - Sparsity penalty: Keep penalty low initially to learn reconstruction,
      then gradually increase to enforce sparsity

    Example (LR schedule):
        >>> lr_schedule = WarmupThenLinearSchedule(
        ...     warmup_value=0.0,      # Start from 0
        ...     end_value=1e-5,        # Decay to 1e-5
        ...     warmup_steps=1000,     # Warmup for 1000 steps
        ...     total_steps=100000     # Total training steps
        ... )
        >>> # Then multiply by base LR: lr = base_lr * lr_schedule(step)
    """

    def __init__(
        self,
        warmup_value: float,
        end_value: float,
        warmup_steps: int,
        total_steps: int
    ):
        """
        Args:
            warmup_value: Value during warmup period
            end_value: Final value at total_steps
            warmup_steps: Number of steps to keep constant
            total_steps: Total number of steps (where end_value is reached)
        """
        self.warmup_value = warmup_value
        self.end_value = end_value
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.warmup_value
        elif step >= self.total_steps:
            return self.end_value
        else:
            # Linear ramp from warmup_value to end_value
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.warmup_value + progress * (self.end_value - self.warmup_value)

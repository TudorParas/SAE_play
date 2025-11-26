"""
Baseline SAE experiment configurations.

These represent well-tested default configurations that serve as starting points
for new experiments. Use dataclasses.replace() to modify them.
"""

from .model import ModelConfig
from .data import DataConfig
from .sae import SimpleSAEConfig, DeepSAEConfig
from .training import TrainingConfig
from .experiment import SAEExperimentConfig
from .lr_schedule import OneCycleLRConfig


# Simple SAE with TopK sparsity
SIMPLE_SAE = SAEExperimentConfig(
    project_name="baseline",
    experiment_name="simple_sae_baseline",
    model=ModelConfig(
        name="EleutherAI/pythia-70m",
        layer_idx=4,
    ),
    data=DataConfig(
        num_samples=10000,
        train_frac=0.9,
        extraction_batch_size=32,
        training_batch_size=32,
        seed=42,
        num_workers=4,
    ),
    sae=SimpleSAEConfig(
        hidden_dim_multiplier=32,
        sparsity_type="topk",
        sparsity_k=128,
    ),
    training=TrainingConfig(
        num_epochs=10,
        lr=1e-3,
        sparsity_warmup_value=1e-2,
        sparsity_end_value=2.0,
        sparsity_warmup_epochs=2,
        random_seed=53,
        lr_schedule=OneCycleLRConfig(
            max_lr=1e-3,
            warmup_pct=0.1,
        ),
        use_compile=False,  # ToDo: Enable compile when moving off windows. Doesn't work on windows.
        use_amp=True,
    ),
)


# Deep SAE with L1 sparsity
DEEP_SAE = SAEExperimentConfig(
    project_name="baseline",
    experiment_name="deep_sae_baseline",
    model=ModelConfig(
        name="EleutherAI/pythia-70m",
        layer_idx=4,
    ),
    data=DataConfig(
        num_samples=10000,
        train_frac=0.9,
        extraction_batch_size=8,
        training_batch_size=32,
        seed=42,
        num_workers=2,
    ),
    sae=DeepSAEConfig(
        encoder_hidden_dims=[4, 32],  # Multipliers of input_dim
        decoder_hidden_dims=[4],
        sparsity_type="l1",
    ),
    training=TrainingConfig(
        num_epochs=10,
        lr=1e-3,
        lr_schedule=OneCycleLRConfig(
            max_lr=1e-3,
            warmup_pct=0.1,
        ),
        sparsity_warmup_value=1e-2,
        sparsity_end_value=2.0,
        sparsity_warmup_epochs=2,
        random_seed=53,
        use_compile=False,  # ToDo: Enable compile when moving off windows. Doesn't work on windows.
        use_amp=True,
    ),
)

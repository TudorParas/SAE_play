"""
Baseline SAE experiment configurations.

These represent well-tested default configurations that serve as starting points
for new experiments. Use dataclasses.replace() to modify them.
"""

import sys

from src.sae.configs.model import ModelConfig
from src.sae.configs.data import DataConfig, SourceConfig
from src.sae.configs.sae import SimpleSAEConfig, DeepSAEConfig
from src.sae.configs.training import TrainingConfig
from src.sae.configs.experiment import SAEExperimentConfig
from src.sae.configs.lr_schedule import OneCycleLRConfig
from src.sae.configs.evaluation import EvalConfig

# torch.compile doesn't work on Windows, auto-detect platform
_USE_COMPILE = sys.platform != "win32"


# Simple SAE with TopK sparsity
SIMPLE_SAE = SAEExperimentConfig(
    project_name="baseline",
    experiment_name="simple_sae_baseline",
    model=ModelConfig(
        name="EleutherAI/pythia-70m",
        layer_idx=4,
    ),
    data=DataConfig(
        sources=[
            SourceConfig(name="wikitext", train_frac=0.9, test_frac=0.1),
            SourceConfig(name="c4", train_frac=0.4, test_frac=0.3),
        ],
        num_samples=10000,
        extraction_batch_size=32,
        training_batch_size=32,
        seed=42,
        max_length=128,
        num_workers=0,
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
        use_compile=_USE_COMPILE,
        use_amp=True,
    ),
    evaluation=EvalConfig(
        dead_feature_threshold=0.01,
        max_spectral_samples=2000,
        feature_analysis_top_k=10,
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
        sources=[
            SourceConfig(name="wikitext", train_frac=0.9, test_frac=0.1),
            SourceConfig(name="c4", train_frac=0.4, test_frac=0.3),
        ],
        num_samples=10000,
        extraction_batch_size=8,
        training_batch_size=32,
        seed=42,
        max_length=128,
        num_workers=2,
    ),
    sae=DeepSAEConfig(
        encoder_hidden_dims=[4, 32],  # Multipliers of input_dim
        decoder_hidden_dims=[4],

        sparsity_type="topk",
        sparsity_k=128,
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
        use_compile=_USE_COMPILE,
        use_amp=True,
    ),
    evaluation=EvalConfig(
        dead_feature_threshold=0.01,
        max_spectral_samples=2000,
        feature_analysis_top_k=10,
    ),
)

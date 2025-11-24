"""
Experiment runner for config-based SAE experiments.

Provides run_sae_experiment() which orchestrates the full experiment pipeline
based on a configuration object. For custom experiments, you can still import
and use the library primitives directly.
"""

import os
import inspect
from datetime import datetime

import torch
from dataclasses import asdict
from typing import Callable

from src.sae.configs import SAEExperimentConfig, SimpleSAEConfig, DeepSAEConfig
from src.sae.data import load_pile_samples, create_dataloader
from src.sae.training.train_pipeline import TrainPipeline
from src.sae.training.schedules import WarmupThenLinearSchedule
from src.sae.checkpoints import save_checkpoint
from src.sae.util.logging import TeeLogger

from .common import prepare_activations, run_evaluation


def get_project_name(file_path: str) -> str:
    """
    Extract project name from experiment filename.

    Splits the filename on the first underscore to get the project identifier.

    Args:
        file_path: Path to the experiment file (typically __file__)

    Returns:
        Project name extracted from filename

    Example:
        >>> get_project_name(__file__)  # From "exp01a_simple_sae.py"
        'exp01a'
        >>> get_project_name("/path/to/exp02b_deep_sae.py")
        'exp02b'
    """
    filename = os.path.basename(file_path)
    # Remove .py extension if present
    if filename.endswith('.py'):
        filename = filename[:-3]
    # Split on first underscore
    return filename.split("_", 1)[0]


def get_experiment_name(func: Callable) -> str:
    """
    Extract experiment name from function.

    Simply returns the function's __name__ attribute, which is the experiment name.

    Args:
        func: The experiment function

    Returns:
        Function name as experiment identifier

    Example:
        >>> def simple_sae_fast(): pass
        >>> get_experiment_name(simple_sae_fast)
        'simple_sae_fast'
    """
    return func.__name__


def run_sae_experiment(config: SAEExperimentConfig) -> dict:
    """
    Run a complete SAE experiment from a configuration.

    This is the main entry point for config-based experiments. It handles:
    1. Loading data and model
    2. Extracting and preparing activations
    3. Creating and training the SAE
    4. Saving checkpoint
    5. Evaluating and generating report

    For custom experiments that don't fit this pipeline, you can import
    and use the library primitives (load_model, prepare_activations, etc.)
    directly instead.

    Args:
        config: Complete experiment configuration

    Returns:
        Dictionary containing:
            - 'training_results': Training history and metrics
            - 'eval_results': Evaluation metrics
            - 'analysis_results': Text analysis results
            - 'output_dir': Path to saved outputs
    """
    # Start logging
    logger = TeeLogger()
    logger.start()

    print("=" * 60)
    print(f"SAE EXPERIMENT: {config.experiment_name}")
    print("=" * 60)

    # ========================================================================
    # 1. Load data and model
    # ========================================================================
    print("\n[1/6] Loading data and model...")
    texts = load_pile_samples(num_samples=config.data.num_samples, shuffle=True)
    print(f"Loaded {len(texts)} texts")

    model, tokenizer, device = config.model.resolve()
    print(f"Using device: {device}")

    # ========================================================================
    # 2. Extract and prepare activations
    # ========================================================================
    print("\n[2/6] Extracting and preparing activations...")
    train_dataset, test_dataset, activation_mean = prepare_activations(
        model,
        tokenizer,
        texts,
        config.model.layer_idx,
        train_frac=config.data.train_frac,
        batch_size=config.data.extraction_batch_size,
        seed=config.data.seed,
    )
    print(f"Train set: {len(train_dataset):,} samples ({config.data.train_frac:.0%})")
    print(f"Test set: {len(test_dataset):,} samples ({1-config.data.train_frac:.0%})")

    # ========================================================================
    # 3. Create SAE
    # ========================================================================
    print("\n[3/6] Creating SAE...")
    input_dim = train_dataset.dim

    sae = config.sae.resolve(input_dim, device)

    # Print architecture info
    if isinstance(config.sae, SimpleSAEConfig):
        print(f"SimpleSAE: {input_dim} -> {sae.probe_dim} -> {input_dim}")
    elif isinstance(config.sae, DeepSAEConfig):
        encoder_dims = [input_dim * m for m in config.sae.encoder_hidden_dims]
        decoder_dims = [input_dim * m for m in config.sae.decoder_hidden_dims]
        print(f"DeepSAE:")
        print(f"  Encoder: {input_dim} -> {' -> '.join(map(str, encoder_dims))}")
        print(f"  Decoder: {encoder_dims[-1]} -> {' -> '.join(map(str, decoder_dims))} -> {input_dim}")

    print(f"Sparsity: {sae.sparsity.__class__.__name__}")

    # ========================================================================
    # 4. Train
    # ========================================================================
    print(f"\n[4/6] Training SAE...")
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.data.training_batch_size,
        shuffle=True,
        pin_memory=True,
    )
    print(f"Train DataLoader: {len(train_loader)} batches of size {config.data.training_batch_size}")

    optimizer = torch.optim.Adam(sae.parameters(), lr=config.training.lr)

    # Create learning rate schedule
    steps_per_epoch = len(train_loader)
    total_steps = config.training.num_epochs * steps_per_epoch
    lr_sched = config.training.resolve_lr_schedule(optimizer, total_steps)

    # Create sparsity schedule
    sparsity_schedule = WarmupThenLinearSchedule(
        warmup_value=config.training.sparsity_warmup_value,
        end_value=config.training.sparsity_end_value,
        warmup_steps=config.training.sparsity_warmup_epochs,
        total_steps=config.training.num_epochs,
    )

    pipeline = TrainPipeline(
        sae=sae,
        optimizer=optimizer,
        train_loader=train_loader,
        activation_mean=activation_mean,
        lr_schedule=lr_sched,
        sparsity_schedule=sparsity_schedule,
    )

    torch.manual_seed(config.training.random_seed)
    training_results = pipeline.train_sae(
        num_epochs=config.training.num_epochs,
        batch_size=config.data.training_batch_size,
        verbose=True,
    )

    # ###########################################################################
    # 5. Save Checkpoint
    # ###########################################################################
    print("\n[5/6] Saving checkpoint...")

    # Create experiment ID from project, experiment names, and timestamp
    experiment_id = f"{config.project_name}_{config.experiment_name}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get the directory where the experiment file is located
    # This allows outputs to be saved relative to the experiment
    caller_frame = inspect.stack()[1]
    caller_file = caller_frame.filename
    experiment_dir = os.path.dirname(os.path.abspath(caller_file))

    # Create output path: {experiment_dir}/reports/{project_name}/{experiment_name}/
    output_dir = os.path.join(experiment_dir, "reports", config.project_name, config.experiment_name, timestamp)

    # Save checkpoint with full experiment config
    save_checkpoint(
        path=output_dir,
        sae=sae,
        activation_mean=activation_mean,
        experiment_config=config,
        final_loss=training_results['final_metrics'].loss,
    )
    print(f"Checkpoint saved to {output_dir}")

    # ========================================================================
    # 6. Evaluate and Generate Report
    # ========================================================================
    print("\n[6/6] Evaluating SAE and generating report...")

    sample_texts = [
        "Dogs are man's best friend.",
        "Neural networks process information.",
        "London is a major city in England.",
    ]

    evaluator, eval_results, analysis_results = run_evaluation(
        sae, test_dataset, activation_mean, model, tokenizer, config.model.layer_idx, sample_texts
    )

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE!")
    print("=" * 60)
    print(f"Reconstruction Loss (test): {eval_results.reconstruction_loss:.6f}")
    print(f"Dead features: {eval_results.dead_features['count']}/{eval_results.dead_features['total']} ({eval_results.dead_features['fraction']:.2%})")
    print(f"Effective Latent Dimension (ELD): {eval_results.spectral_stats['ELD']:.2f}")
    print(f"Top Component Explained Variance: {eval_results.spectral_stats['Top_Component_Explained_Var']:.2%}")

    logger.stop()

    # Generate report
    report_metadata = {
        "device": device,
        "num_text_samples": len(texts),
        "num_tokens": len(train_dataset) + len(test_dataset),
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "config": asdict(config),
        "final_loss": training_results['final_metrics'].loss,
    }

    evaluator.generate_report(
        experiment_id=experiment_id,
        timestamp=timestamp,
        description=f"{config.experiment_name} on {config.model.name} layer {config.model.layer_idx}",
        training_results=training_results,
        metadata=report_metadata,
        log_text=logger.get_log(),
        checkpoint_path=output_dir,
        save_path=output_dir,
        eval_results=eval_results,
        analysis_results=analysis_results,
    )

    return {
        'training_results': training_results,
        'eval_results': eval_results,
        'analysis_results': analysis_results,
        'output_dir': output_dir,
    }

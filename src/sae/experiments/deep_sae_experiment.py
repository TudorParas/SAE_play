"""
Example experiment: Train a deep (multi-layer) SAE on Pythia-70M activations.

This demonstrates the library approach with a multi-layer architecture:
- Configuration is explicit, not hidden
- Helpers reduce boilerplate but you stay in control
- Easy to customize any step

Run from project root:
    python -m src.sae.experiments.deep_sae_experiment
"""
from datetime import datetime

import torch
from torch.optim import lr_scheduler

# Import library components
from src.sae.data import load_pile_samples, create_dataloader
from src.sae.models.deep import DeepSAE
from src.sae.sparsity import L1Sparsity
from src.sae.training.train_pipeline import TrainPipeline
from src.sae.training.schedules import WarmupThenLinearSchedule
from src.sae.checkpoints import save_checkpoint
from src.sae.util.logging import TeeLogger

# Import experiment helpers
from src.sae.experiments.common import load_model, prepare_activations, run_evaluation


def main():
    """Run the deep SAE experiment."""

    # Start capturing stdout for the report
    logger = TeeLogger()
    logger.start()

    print("=" * 60)
    print("DEEP SAE EXPERIMENT")
    print("=" * 60)

    # ========================================================================
    # Configuration (explicit, not hidden in a config object)
    # ========================================================================
    model_name = "EleutherAI/pythia-70m"
    layer_idx = 4
    num_samples = 10000
    num_epochs = 20
    batch_size = 32
    train_frac = 0.9

    # ========================================================================
    # 1. Load data and model
    # ========================================================================
    print("\n[1/6] Loading data and model...")
    texts = load_pile_samples(num_samples=num_samples, shuffle=True)
    print(f"Loaded {len(texts)} texts")

    model, tokenizer, device = load_model(model_name)
    print(f"Using device: {device}")

    # ========================================================================
    # 2. Extract and prepare activations
    # ========================================================================
    print("\n[2/6] Extracting and preparing activations...")
    train_dataset, test_dataset, activation_mean = prepare_activations(
        model, tokenizer, texts, layer_idx,
        train_frac=train_frac,
        batch_size=8,  # Batch size for extraction
    )
    print(f"Train set: {len(train_dataset):,} samples ({train_frac:.0%})")
    print(f"Test set: {len(test_dataset):,} samples ({1-train_frac:.0%})")

    # ========================================================================
    # 3. Create Deep SAE (this is what varies per experiment)
    # ========================================================================
    print("\n[3/6] Creating Deep SAE...")
    input_dim = train_dataset.dim

    # Define multi-layer architecture
    # Encoder progressively expands: 512 -> 2048 -> 16384
    encoder_hidden_dims = [input_dim * 4, input_dim * 32]

    # Decoder progressively contracts: 16384 -> 2048 -> 512
    decoder_hidden_dims = [input_dim * 4]

    # Choose sparsity mechanism
    # Option 1: TopK (fixed sparsity)
    # sparsity = TopKSparsity(k=64)

    # Option 2: L1 (adaptive sparsity - better for circuit discovery!)
    sparsity = L1Sparsity()

    # Option 3: JumpReLU
    # sparsity = JumpReLUSparsity(num_features=encoder_hidden_dims[-1])

    sae = DeepSAE(
        input_dim=input_dim,
        encoder_hidden_dims=encoder_hidden_dims,
        decoder_hidden_dims=decoder_hidden_dims,
        sparsity=sparsity
    ).to(device)

    print(f"Deep SAE architecture:")
    print(f"  Encoder: {input_dim} -> {' -> '.join(map(str, encoder_hidden_dims))}")
    print(f"  Latent: {encoder_hidden_dims[-1]} dimensions (sparse)")
    print(f"  Decoder: {encoder_hidden_dims[-1]} -> {' -> '.join(map(str, decoder_hidden_dims))} -> {input_dim}")
    print(f"  Sparsity: {sparsity.__class__.__name__}")

    # ========================================================================
    # 4. Train (with OneCycleLR scheduler)
    # ========================================================================
    print("\n[4/6] Training Deep SAE...")
    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )
    print(f"Train DataLoader: {len(train_loader)} batches of size {batch_size}")

    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)

    # Calculate total training steps for LR scheduler
    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch

    # Learning rate schedule: OneCycleLR with warmup
    lr_schedule = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        total_steps=total_steps,
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos',
        div_factor=10,  # Start at lr/10
        final_div_factor=10  # End at lr/100
    )

    sparsity_schedule = WarmupThenLinearSchedule(
        warmup_value=1e-2,
        end_value=2.0,
        warmup_steps=2,
        total_steps=num_epochs
    )

    pipeline = TrainPipeline(
        sae=sae,
        optimizer=optimizer,
        train_loader=train_loader,
        activation_mean=activation_mean,
        lr_schedule=lr_schedule,
        sparsity_schedule=sparsity_schedule,
    )

    results = pipeline.train_sae(num_epochs=num_epochs, batch_size=batch_size, verbose=True)

    print("\nDeep SAE trained successfully!")
    print(f"Architecture: {len(encoder_hidden_dims)}-layer encoder, {len(decoder_hidden_dims) + 1}-layer decoder")
    print(f"Latent dimension: {encoder_hidden_dims[-1]:,}")
    print(f"Final reconstruction loss: {results['final_metrics'].recon_loss:.4f}")
    print(f"Final sparsity: {results['final_metrics'].pct_active:.1f}% active features")

    # ###########################################################################
    # 5. Save Checkpoint
    # ###########################################################################
    print("\n[5/6] Saving checkpoint...")
    experiment_id = f"deep_sae_{datetime.now().strftime("%Y%m%d_%H%M%S")}"
    output_dir = f"outputs/{experiment_id}"

    save_checkpoint(
        path=output_dir,
        sae=sae,
        activation_mean=activation_mean,
        metadata={
            'model_name': model_name,
            'layer_idx': layer_idx,
            'num_epochs': num_epochs,
            'encoder_hidden_dims': encoder_hidden_dims,
            'decoder_hidden_dims': decoder_hidden_dims,
            'final_loss': results['final_metrics'].loss,
        }
    )
    print(f"Checkpoint saved to {output_dir}")

    # ========================================================================
    # 6. Evaluate and Generate Report
    # ========================================================================
    print("\n[6/6] Evaluating Deep SAE and generating report...")

    sample_texts = [
        "Dogs are man's best friend.",
        "Neural networks process information.",
        "London is a major city in England.",
    ]

    evaluator, eval_results, analysis_results = run_evaluation(
        sae, test_dataset, activation_mean, model, tokenizer, layer_idx, sample_texts
    )

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE!")
    print("=" * 60)
    print(f"Reconstruction Loss (test): {eval_results.reconstruction_loss:.6f}")
    print(f"Dead features: {eval_results.dead_features['count']}/{eval_results.dead_features['total']} ({eval_results.dead_features['fraction']:.2%})")
    print(f"Effective Latent Dimension (ELD): {eval_results.spectral_stats['ELD']:.2f}")
    print(f"Top Component Explained Variance: {eval_results.spectral_stats['Top_Component_Explained_Var']:.2%}")

    logger.stop()

    evaluator.generate_report(
        experiment_id=experiment_id,
        description=f"DeepSAE with {sparsity.__class__.__name__} on {model_name} layer {layer_idx}",
        training_results=results,
        metadata={
            "device": device,
            "model_name": model_name,
            "layer_idx": layer_idx,
            "num_text_samples": len(texts),
            "num_tokens": len(train_dataset) + len(test_dataset),
            "train_samples": len(train_dataset),
            "test_samples": len(test_dataset),
            "train_frac": train_frac,
            "encoder_hidden_dims": encoder_hidden_dims,
            "decoder_hidden_dims": decoder_hidden_dims,
            "sparsity_type": sparsity.__class__.__name__,
        },
        log_text=logger.get_log(),
        checkpoint_path=output_dir,
        save_path=output_dir,
        eval_results=eval_results,
        analysis_results=analysis_results,
    )


if __name__ == "__main__":
    main()
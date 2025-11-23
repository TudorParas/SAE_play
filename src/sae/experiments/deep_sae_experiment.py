"""
Example experiment: Train a deep (multi-layer) SAE on Pythia-70M activations.

This demonstrates the library approach with a multi-layer architecture:
- You create each component explicitly
- You pass objects around (no magic configuration)
- You control the flow

Run from project root:
    python -m src.sae.experiments.deep_sae_experiment
"""

import time
import torch
from torch.optim import lr_scheduler
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import library components
from src.sae.data.loader import load_pile_samples
from src.sae.data.datasets import ActivationDataset, split_activations, create_dataloader
from src.sae.activations import extract_activations
from src.sae.models.deep import DeepSAE
from src.sae.sparsity import TopKSparsity, L1Sparsity, JumpReLUSparsity
from src.sae.training.train_pipeline import TrainPipeline
from src.sae.training.schedules import WarmupThenLinearSchedule, ConstantSchedule
from src.sae.evaluation import Evaluator, EvalConfig, create_experiment_id
from src.sae.checkpoints import save_checkpoint
from src.sae.util.logging import TeeLogger


def main():
    """Run the deep SAE experiment."""

    # Start capturing stdout for the report
    logger = TeeLogger()
    logger.start()
    start_time = time.time()

    print("="*60)
    print("DEEP SAE EXPERIMENT")
    print("="*60)

    # ========================================================================
    # 1. Load data (you control this)
    # ========================================================================
    print("\n[1/7] Loading data...")
    # Loader automatically finds data file relative to module location
    texts = load_pile_samples(
        num_samples=2 ** 10,
        shuffle=True
    )
    print(f"Loaded {len(texts)} texts")

    # ========================================================================
    # 2. Load model (you control this)
    # ========================================================================
    print("\n[2/7] Loading model...")
    model_name = "EleutherAI/pythia-70m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # GPT models need this

    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = model.to(device)
    model.eval()

    # ========================================================================
    # 3. Extract activations and create train/test split
    # ========================================================================
    print("\n[3/7] Extracting activations...")
    layer_idx = 3  # You choose which layer

    activations = extract_activations(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        layer_idx=layer_idx,
        batch_size=8
    )
    # Move activations to CPU - DataLoader will handle GPU transfer
    # (pin_memory requires CPU tensors)
    activations = activations.cpu()

    print(f"Extracted activations: {activations.shape}")
    print(f"  - {activations.shape[0]:,} token activations")
    print(f"  - {activations.shape[1]} dimensions")

    # Split FIRST, then center (to avoid data leakage)
    print("\n[3.5/7] Creating train/test split...")
    train_frac = 0.9
    train_raw, test_raw = split_activations(activations, train_frac=train_frac, seed=42)

    # Compute mean on TRAIN only
    train_mean = train_raw.mean(dim=0, keepdim=True)

    # Create datasets (both centered with train mean)
    train_dataset = ActivationDataset(train_raw, mean=train_mean)
    test_dataset = ActivationDataset(test_raw, mean=train_mean)
    activation_mean = train_mean  # For use in evaluation

    print(f"Train set: {len(train_dataset):,} samples ({train_frac:.0%})")
    print(f"Test set: {len(test_dataset):,} samples ({1-train_frac:.0%})")

    # ========================================================================
    # 4. Create Deep SAE (you choose architecture)
    # ========================================================================
    print("\n[4/7] Creating Deep SAE...")
    input_dim = activations.shape[1]

    # Define multi-layer architecture
    # Encoder progressively expands: 512 -> 1024 -> 2048 -> 4096
    # Chosen so we have as many parameters as a single layer SAE with hidden dim 32 (1 * 4 + 4 * 7 = 32)
    encoder_hidden_dims = [input_dim * 4, input_dim * 32]

    # Decoder progressively contracts: 4096 -> 2048 -> 1024 -> 512
    # Note: Final layer back to input_dim is added automatically
    decoder_hidden_dims = [input_dim * 4]

    # Choose sparsity mechanism
    # Option 1: TopK (fixed sparsity)
    # sparsity = TopKSparsity(k=64)

    # Option 2: L1 (adaptive sparsity - better for circuit discovery!)
    sparsity = L1Sparsity()

    # Option 3: JumpRELU
    # sparsity = JumpReLUSparsity(num_features=encoder_hidden_dims[-1])

    sae = DeepSAE(
        input_dim=input_dim,
        encoder_hidden_dims=encoder_hidden_dims,
        decoder_hidden_dims=decoder_hidden_dims,
        sparsity=sparsity
    ).to(device)

    print(f"Deep SAE architecture:")
    print(f"  Encoder: {input_dim} → {' → '.join(map(str, encoder_hidden_dims))}")
    print(f"  Latent: {encoder_hidden_dims[-1]} dimensions (sparse)")
    print(f"  Decoder: {encoder_hidden_dims[-1]} → {' → '.join(map(str, decoder_hidden_dims))} → {input_dim}")
    print(f"  Sparsity: {sparsity.__class__.__name__}")

    # ========================================================================
    # 4.5 Create Optimizer. The LR doesn't matter, we'll adjust it later at each step.
    # ========================================================================
    print("\n[4.5/7] Creating optimizer...")
    # The learning rate doesn't matter, we'll adjust it later at each step.
    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)

    print(f"SAE optimizer: {optimizer}")

    # ========================================================================
    # 5. Create trainer with DataLoader (you configure)
    # ========================================================================
    print("\n[5/7] Creating trainer...")

    # Training configuration
    num_epochs = 20
    batch_size = 32

    # Create DataLoader for training (shuffle enabled)
    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )

    print(f"Train DataLoader: {len(train_loader)} batches of size {batch_size}")

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

    train_pipeline = TrainPipeline(
        sae=sae,
        optimizer=optimizer,
        train_loader=train_loader,  # Using DataLoader mode
        activation_mean=activation_mean,
        lr_schedule=lr_schedule,
        sparsity_schedule=sparsity_schedule,
    )

    # ========================================================================
    # 6. Train (library coordinates)
    # ========================================================================
    print("\n[6/7] Training Deep SAE...")

    num_epochs = 20
    results = train_pipeline.train_sae(
        num_epochs=num_epochs,
        batch_size=32,
        verbose=True
    )

    print("\nDeep SAE trained successfully!")
    print(f"Architecture: {len(encoder_hidden_dims)}-layer encoder, {len(decoder_hidden_dims) + 1}-layer decoder")
    print(f"Latent dimension: {encoder_hidden_dims[-1]:,}")
    print(f"Final reconstruction loss: {results['final_metrics'].recon_loss:.4f}")
    print(f"Final sparsity: {results['final_metrics'].pct_active:.1f}% active features")

    # ========================================================================
    # 7. Evaluate and Generate Report
    # ========================================================================
    print("\n[7/7] Evaluating Deep SAE and generating report...")

    # Create test DataLoader for evaluation
    test_loader = create_dataloader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        pin_memory=True,
    )

    # Create evaluator
    evaluator = Evaluator(
        sae=sae,
        activation_mean=activation_mean,
        config=EvalConfig(dead_feature_threshold=0.01),
    )

    # Run evaluation on test set
    eval_results = evaluator.evaluate(test_loader)

    # Print evaluation results (will be captured in log)
    print(f"\nReconstruction Loss (test): {eval_results.reconstruction_loss:.6f}")
    print(f"Dead features: {eval_results.dead_features['count']}/{eval_results.dead_features['total']} ({eval_results.dead_features['fraction']:.2%})")
    print(f"Effective Latent Dimension (ELD): {eval_results.spectral_stats['ELD']:.2f}")
    print(f"Top Component Explained Variance: {eval_results.spectral_stats['Top_Component_Explained_Var']:.2%}")

    # Analyze sample texts (separate from evaluation)
    sample_texts = [
        "Dogs are man's best friend.",
        "Neural networks process information.",
        "London is a major city in England.",
    ]
    analysis_results = evaluator.analyze_texts(model, tokenizer, sample_texts, layer_idx)

    print("\n" + "=" * 60)
    print("FEATURE ANALYSIS")
    print("=" * 60)
    for result in analysis_results.texts:
        print(f"\nText: '{result['text']}'")
        print(f"  Active features: {result['num_active']}/{result['total_features']} ({result['pct_active']:.1f}%)")
        print(f"  Top features:")
        for idx, val in result['top_features'][:5]:
            print(f"    Feature {idx}: {val:.3f}")
    print("=" * 60)

    # ========================================================================
    # Save checkpoint
    # ========================================================================
    checkpoint_path = f"checkpoints/deep_sae_layer{layer_idx}"
    print(f"\nSaving checkpoint to {checkpoint_path}...")
    save_checkpoint(
        path=checkpoint_path,
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

    # ========================================================================
    # Generate and save report
    # ========================================================================
    runtime_seconds = time.time() - start_time
    print(f"\nTotal runtime: {runtime_seconds:.1f}s ({runtime_seconds/60:.1f} min)")

    # Stop logging and get captured output
    logger.stop()

    # Generate report using Evaluator
    experiment_id = create_experiment_id("deep_sae")
    report_path = f"reports/{experiment_id}"

    evaluator.generate_report(
        experiment_id=experiment_id,
        description=f"DeepSAE with {sparsity.__class__.__name__} on {model_name} layer {layer_idx}",
        training_results=results,
        metadata={
            "runtime_seconds": runtime_seconds,
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
        checkpoint_path=checkpoint_path,
        save_path=report_path,
        eval_results=eval_results,
        analysis_results=analysis_results,
    )


if __name__ == "__main__":
    main()
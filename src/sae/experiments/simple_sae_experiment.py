"""
Example experiment: Train a simple SAE on Pythia-70M activations.

This demonstrates the library approach:
- You create each component explicitly
- You pass objects around (no magic configuration)
- You control the flow

Run from project root:
    python -m src.sae.experiments.simple_sae_experiment
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import library components
from src.sae.data.loader import load_pile_samples
from src.sae.data.datasets import ActivationDataset, split_activations, create_dataloader
from src.sae.activations import extract_activations
from src.sae.models.simple import SimpleSAE
from src.sae.sparsity import TopKSparsity, L1Sparsity, JumpReLUSparsity
from src.sae.training.train_pipeline import TrainPipeline
from src.sae.training.schedules import WarmupThenLinearSchedule, ConstantSchedule
from src.sae.evaluation import Evaluator, EvalConfig, create_experiment_id
from src.sae.checkpoints import save_checkpoint
from src.sae.util.logging import TeeLogger


def main():
    """Run the experiment."""

    # Start capturing stdout for the report
    logger = TeeLogger()
    logger.start()
    start_time = time.time()

    print("="*60)
    print("SIMPLE SAE EXPERIMENT")
    print("="*60)

    # ========================================================================
    # 1. Load data (you control this)
    # ========================================================================
    print("\n[1/7] Loading data...")
    # Loader automatically finds data file relative to module location
    texts = load_pile_samples(
        num_samples=10000,
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
        batch_size=32
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
    batch_size = 32

    # Create DataLoader for training (shuffle enabled)
    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )

    print(f"Train DataLoader: {len(train_loader)} batches of size {batch_size}")

    # Create test DataLoader for evaluation
    test_loader = create_dataloader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        pin_memory=True,
    )

    # ========================================================================
    # 4. Create SAE (you choose architecture)
    # ========================================================================
    print("\n[4/7] Creating SAE...")
    input_dim = activations.shape[1]
    hidden_dim = input_dim * 32  # 4x expansion

    # Choose sparsity mechanism
    # Option 1: TopK (fixed sparsity)
    sparsity = TopKSparsity(k=128)  # ToDo: if using top_k sparsity no longer use bias_enc

    # Option 2: L1 (adaptive sparsity - better for circuit discovery!)
    # sparsity = L1Sparsity()

    sae = SimpleSAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        sparsity=sparsity
    ).to(device)

    print(f"SAE architecture: {input_dim} → {hidden_dim} → {input_dim}")
    print(f"Sparsity: {sparsity.__class__.__name__}")

    # ========================================================================
    # 4.5 Create Optimizer. The LR doesn't matter, we'll adjust it later at each step.
    # ========================================================================
    print("\n[4.5/7] Creating optimizer...")
    # The learning rate doesn't matter, we'll adjust it later at each step.
    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)

    print(f"SAE optimizer: {optimizer}")

    # ========================================================================
    # 5. Create schedules and initialise the pipeline.
    # ========================================================================
    print("\n[5/7] Creating trainer...")

    # Training configuration
    num_epochs = 20

    # Learning rate schedule (None = use optimizer's default)
    lr_schedule = None
    sparsity_schedule = WarmupThenLinearSchedule(
        warmup_value=1e-2,
        end_value=2.0,
        warmup_steps=2,  # How long to have warmup value
        total_steps=num_epochs  # Reach end value after this many epochs
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
    print("\n[6/7] Training SAE...")
    num_epochs = 20
    torch.manual_seed(53)
    results = train_pipeline.train_sae(
        num_epochs=num_epochs,
        batch_size=32,
        verbose=True
    )

    # ========================================================================
    # 7. Evaluate and Generate Report
    # ========================================================================
    print("\n[7/7] Evaluating SAE and generating report...")

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
    checkpoint_path = f"checkpoints/simple_sae_layer{layer_idx}"
    print(f"\nSaving checkpoint to {checkpoint_path}...")
    save_checkpoint(
        path=checkpoint_path,
        sae=sae,
        activation_mean=activation_mean,
        metadata={
            'model_name': model_name,
            'layer_idx': layer_idx,
            'num_epochs': num_epochs,
            'final_loss': results['final_metrics'].loss,
        }
    )

    # ========================================================================
    # Generate and save report
    # ========================================================================
    runtime_seconds = time.time() - start_time
    print(f"\nTotal runtime: {runtime_seconds:.1f}s ({runtime_seconds/60:.1f} min)")

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE!")
    print("="*60)

    # Stop logging and get captured output
    logger.stop()

    # Generate report using Evaluator (it already has eval results)
    experiment_id = create_experiment_id("simple_sae")
    report_path = f"reports/{experiment_id}"

    evaluator.generate_report(
        experiment_id=experiment_id,
        description=f"SimpleSAE with {sparsity.__class__.__name__} on {model_name} layer {layer_idx}",
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
            "hidden_dim": hidden_dim,
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

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
from src.sae.activations import extract_activations
from src.sae.models.deep import DeepSAE
from src.sae.sparsity import TopKSparsity, L1Sparsity, JumpReLUSparsity
from src.sae.training.train_pipeline import TrainPipeline
from src.sae.training.schedules import WarmupThenLinearSchedule, ConstantSchedule
from src.sae.evaluation.analyzer import analyze_features, get_spectral_stats
from src.sae.evaluation.metrics import compute_dead_features
from src.sae.evaluation.report import ExperimentReport, create_experiment_id
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
    # 3. Extract activations (explicit call to utility)
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
    activations = activations.to(device)

    print(f"Extracted activations: {activations.shape}")
    print(f"  - {activations.shape[0]:,} token activations")
    print(f"  - {activations.shape[1]} dimensions")

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
    # 5. Create trainer (you configure)
    # ========================================================================
    print("\n[5/7] Creating trainer...")

    # Calculate total training steps
    num_epochs = 20
    samples_per_epoch = activations.shape[0]
    batch_size = 32
    steps_per_epoch = (samples_per_epoch + batch_size - 1) // batch_size
    total_steps = num_epochs * steps_per_epoch

    # Learning rate schedule: warmup then linear decay
    # Warmup from 0 to 1.0 over 1000 steps, then decay to 0.1 by end of training
    # lr_schedule = None # ConstantSchedule(1e-3)
    lr_schedule = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,  # Your current LR
        total_steps=total_steps,
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos',
        div_factor=10,  # Start at lr/10
        final_div_factor=10  # End at lr/100
    )
    sparsity_schedule = WarmupThenLinearSchedule(
        warmup_value=1e-2,
        end_value=2.0,
        warmup_steps=2,  # How long to have warmup value
        total_steps=20  # Reach end value after this many steps
    )

    train_pipeline = TrainPipeline(
        sae=sae,
        optimizer=optimizer,
        activations=activations,
        lr_schedule=lr_schedule,       # LR will vary based on schedule
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

    # ========================================================================
    # 7. Evaluate and Generate Report
    # ========================================================================
    print("\n[7/7] Evaluating Deep SAE and generating report...")

    # Test texts for feature analysis
    test_texts = [
        "Dogs are man's best friend.",
        "Neural networks process information.",
        "London is a major city in England.",
    ]

    # Run feature analysis
    analysis_results = analyze_features(
        sae=sae,
        model=model,
        tokenizer=tokenizer,
        texts=test_texts,
        activation_mean=results['activation_mean'],
        layer_idx=layer_idx,
        top_k=10
    )

    # Print feature analysis to stdout (will be captured in log)
    print("\n" + "=" * 60)
    print("FEATURE ANALYSIS")
    print("=" * 60)
    for result in analysis_results:
        print(f"\nText: '{result['text']}'")
        print(f"  Active features: {result['num_active']}/{result['total_features']} ({result['pct_active']:.1f}%)")
        print(f"  Top {len(result['top_features'])} features:")
        for idx, val in result['top_features']:
            print(f"    Feature {idx}: {val:.3f}")
    print("=" * 60)

    # Compute dead features on training activations (centered)
    print("\nComputing dead features...")
    centered_activations = activations - results['activation_mean']
    dead_features = compute_dead_features(
        sae=sae,
        activations=centered_activations,
        threshold=0.01
    )
    print(f"Dead features: {dead_features['count']}/{dead_features['total']} ({dead_features['fraction']:.2%})")

    # Compute spectral stats (ELD)
    print("\nComputing spectral statistics...")
    # Use a subset of activations for efficiency (10k samples max)
    spectral_sample = centered_activations[:10000]
    spectral_stats = get_spectral_stats(sae, spectral_sample)
    print(f"Effective Latent Dimension (ELD): {spectral_stats['ELD']:.2f}")
    print(f"Top Component Explained Variance: {spectral_stats['Top_Component_Explained_Var']:.2%}")

    # ========================================================================
    # Save checkpoint
    # ========================================================================
    checkpoint_path = f"checkpoints/deep_sae_layer{layer_idx}"
    print(f"\nSaving checkpoint to {checkpoint_path}...")
    save_checkpoint(
        path=checkpoint_path,
        sae=sae,
        activation_mean=results['activation_mean'],
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
    # Build and save experiment report
    # ========================================================================
    runtime_seconds = time.time() - start_time
    print(f"\nTotal runtime: {runtime_seconds:.1f}s ({runtime_seconds/60:.1f} min)")

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE!")
    print("="*60)
    print("\nDeep SAE trained successfully!")
    print(f"Architecture: {len(encoder_hidden_dims)}-layer encoder, {len(decoder_hidden_dims) + 1}-layer decoder")
    print(f"Latent dimension: {encoder_hidden_dims[-1]:,}")
    print(f"Final reconstruction loss: {results['final_metrics'].recon_loss:.4f}")
    print(f"Final sparsity: {results['final_metrics'].pct_active:.1f}% active features")

    # Stop logging and get captured output
    logger.stop()

    # Create experiment report
    experiment_id = create_experiment_id("deep_sae")
    report = ExperimentReport(
        experiment_id=experiment_id,
        description=f"DeepSAE with {sparsity.__class__.__name__} on {model_name} layer {layer_idx}"
    )

    # Set metadata
    report.set_metadata(
        runtime_seconds=runtime_seconds,
        device=device,
        model_name=model_name,
        layer_idx=layer_idx,
        num_samples=len(texts),
        num_tokens=activations.shape[0],
        encoder_hidden_dims=encoder_hidden_dims,
        decoder_hidden_dims=decoder_hidden_dims,
    )

    # Set training results
    report.set_training_results(
        final_loss=results['final_metrics'].loss,
        final_recon_loss=results['final_metrics'].recon_loss,
        final_sparsity_loss=results['final_metrics'].sparsity_loss,
        final_num_active=results['final_metrics'].num_active,
        final_pct_active=results['final_metrics'].pct_active,
        loss_history=results['loss_history'],
        recon_loss_history=results['recon_loss_history'],
        sparsity_history=results['sparsity_history'],
        num_epochs=num_epochs,
        batch_size=batch_size,
    )

    # Format feature analysis for report
    feature_analysis_for_report = [
        {
            "text": r['text'],
            "top_features": r['top_features'],
            "num_active": r['num_active'],
            "pct_active": r['pct_active'],
        }
        for r in analysis_results
    ]

    # Set evaluation results
    report.set_eval_results(
        feature_analysis=feature_analysis_for_report,
        dead_features=dead_features,
        spectral_stats=spectral_stats,
        num_eval_samples=len(test_texts),
    )

    # Set artifacts
    report.set_artifacts(checkpoint_path=checkpoint_path)

    # Set captured log
    report.set_log(logger.get_log())

    # Save report
    report_path = f"reports/{experiment_id}"
    report.save(report_path)


if __name__ == "__main__":
    main()
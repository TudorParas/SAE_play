"""
Example experiment: Train a deep (multi-layer) SAE on Pythia-70M activations.

This demonstrates the library approach with a multi-layer architecture:
- You create each component explicitly
- You pass objects around (no magic configuration)
- You control the flow

Run from project root:
    python -m src.sae.experiments.deep_sae_experiment
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import library components
from src.sae.data.loader import load_pile_samples
from src.sae.activations import extract_activations
from src.sae.models.deep import DeepSAE
from src.sae.training.trainer import SAETrainer
from src.sae.training.loop import train_sae
from src.sae.training.schedules import WarmupThenLinearSchedule
from src.sae.evaluation.analyzer import analyze_features, print_feature_analysis
from src.sae.checkpoints import save_checkpoint


def main():
    """Run the deep SAE experiment."""

    print("="*60)
    print("DEEP SAE EXPERIMENT")
    print("="*60)

    # ========================================================================
    # 1. Load data (you control this)
    # ========================================================================
    print("\n[1/7] Loading data...")
    # Loader automatically finds data file relative to module location
    texts = load_pile_samples(
        num_samples=1000,
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
    encoder_hidden_dims = [input_dim * 2, input_dim * 4, input_dim * 8]

    # Decoder progressively contracts: 4096 -> 2048 -> 1024 -> 512
    # Note: Final layer back to input_dim is added automatically
    decoder_hidden_dims = [input_dim * 4, input_dim * 2]

    sae = DeepSAE(
        input_dim=input_dim,
        encoder_hidden_dims=encoder_hidden_dims,
        decoder_hidden_dims=decoder_hidden_dims,
        top_k=0  # TopK activation with k=64
    ).to(device)

    print(f"Deep SAE architecture:")
    print(f"  Encoder: {input_dim} → {' → '.join(map(str, encoder_hidden_dims))}")
    print(f"  Latent: {encoder_hidden_dims[-1]} dimensions (sparse)")
    print(f"  Decoder: {encoder_hidden_dims[-1]} → {' → '.join(map(str, decoder_hidden_dims))} → {input_dim}")
    print(f"  Using TopK activation with k=64")

    # ========================================================================
    # 5. Create trainer (you configure)
    # ========================================================================
    print("\n[5/7] Creating trainer...")

    # Create a sparsity schedule (optional)
    # Since we're using TopK, this won't be used, but shows how you'd do it
    sparsity_schedule = WarmupThenLinearSchedule(
        warmup_value=1e-3,
        end_value=1e-1,
        warmup_steps=5,
        total_steps=20
    )

    trainer = SAETrainer(
        sae=sae,
        lr=1e-3,
        sparsity_penalty=sparsity_schedule
    )

    # ========================================================================
    # 6. Train (library coordinates)
    # ========================================================================
    print("\n[6/7] Training Deep SAE...")

    results = train_sae(
        sae=sae,
        trainer=trainer,
        activations=activations,
        num_epochs=20,
        batch_size=32,
        use_l1_penalty=False,  # We're using TopK, not L1
        verbose=True
    )

    # ========================================================================
    # 7. Evaluate
    # ========================================================================
    print("\n[7/7] Evaluating Deep SAE...")

    test_texts = [
        "Dogs are man's best friend.",
        "Neural networks process information.",
        "London is a major city in England.",
    ]

    analysis_results = analyze_features(
        sae=sae,
        model=model,
        tokenizer=tokenizer,
        texts=test_texts,
        activation_mean=results['activation_mean'],
        layer_idx=layer_idx,
        top_k=10
    )

    print_feature_analysis(analysis_results)

    # ========================================================================
    # Save checkpoint
    # ========================================================================
    print("Saving checkpoint...")
    save_checkpoint(
        path=f"checkpoints/deep_sae_layer{layer_idx}",
        sae=sae,
        activation_mean=results['activation_mean'],
        metadata={
            'model_name': model_name,
            'layer_idx': layer_idx,
            'num_epochs': 20,
            'encoder_hidden_dims': encoder_hidden_dims,
            'decoder_hidden_dims': decoder_hidden_dims,
            'final_loss': results['final_metrics'].loss,
        }
    )

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE!")
    print("="*60)
    print("\nDeep SAE trained successfully!")
    print(f"Architecture: {len(encoder_hidden_dims)}-layer encoder, {len(decoder_hidden_dims)}-layer decoder")
    print(f"Latent dimension: {encoder_hidden_dims[-1]:,}")
    print(f"Final reconstruction loss: {results['final_metrics'].recon_loss:.4f}")
    print(f"Final sparsity: {results['final_metrics'].pct_active:.1f}% active features")


if __name__ == "__main__":
    main()
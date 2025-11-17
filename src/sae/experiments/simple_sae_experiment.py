"""
Example experiment: Train a simple SAE on Pythia-70M activations.

This demonstrates the library approach:
- You create each component explicitly
- You pass objects around (no magic configuration)
- You control the flow

Run from project root:
    python -m src.sae.experiments.simple_sae_experiment
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import library components
from src.sae.data.loader import load_pile_samples
from src.sae.activations import extract_activations
from src.sae.models.simple import SimpleSAE
from src.sae.training.trainer import SAETrainer
from src.sae.training.loop import train_sae
from src.sae.training.schedules import WarmupThenLinearSchedule
from src.sae.evaluation.analyzer import analyze_features, print_feature_analysis
from src.sae.checkpoints import save_checkpoint


def main():
    """Run the experiment."""

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
    # 4. Create SAE (you choose architecture)
    # ========================================================================
    print("\n[4/7] Creating SAE...")
    input_dim = activations.shape[1]
    hidden_dim = input_dim * 4  # 4x expansion

    sae = SimpleSAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        top_k=0  # TopK activation
    ).to(device)

    print(f"SAE architecture: {input_dim} → {hidden_dim} → {input_dim}")
    print(f"Using TopK activation with k=32")

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
        total_steps=50
    )

    trainer = SAETrainer(
        sae=sae,
        lr=1e-3,
        sparsity_penalty=sparsity_schedule
    )

    # ========================================================================
    # 6. Train (library coordinates)
    # ========================================================================
    print("\n[6/7] Training SAE...")

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
    print("\n[7/7] Evaluating SAE...")

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
        path=f"checkpoints/simple_sae_layer{layer_idx}",
        sae=sae,
        activation_mean=results['activation_mean'],
        metadata={
            'model_name': model_name,
            'layer_idx': layer_idx,
            'num_epochs': 50,
            'final_loss': results['final_metrics'].loss,
        }
    )

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()

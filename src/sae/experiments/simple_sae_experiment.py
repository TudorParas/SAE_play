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
from src.sae.sparsity import TopKSparsity, L1Sparsity, JumpReLUSparsity
from src.sae.training.train_pipeline import TrainPipeline
from src.sae.training.schedules import WarmupThenLinearSchedule, ConstantSchedule
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
    # 4. Create SAE (you choose architecture)
    # ========================================================================
    print("\n[4/7] Creating SAE...")
    input_dim = activations.shape[1]
    hidden_dim = input_dim * 32  # 4x expansion

    # Choose sparsity mechanism
    # Option 1: TopK (fixed sparsity)
    # sparsity = TopKSparsity(k=32)

    # Option 2: L1 (adaptive sparsity - better for circuit discovery!)
    sparsity = L1Sparsity()

    # Option 3: JumpRELU
    # sparsity = JumpReLUSparsity(num_features=hidden_dim)

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
    # 5. Create trainer (you configure)
    # ========================================================================
    print("\n[5/7] Creating trainer...")

    # Calculate total training steps
    num_epochs = 20
    samples_per_epoch = activations.shape[0]
    batch_size = 32
    steps_per_epoch = samples_per_epoch // batch_size
    total_steps = num_epochs * steps_per_epoch

    # Learning rate schedule.
    # lr_schedule = ConstantSchedule(5e-3)
    # lr_schedule = torch.optim.lr_scheduler.StepLR(
    #     optimizer=optimizer,
    #     step_size=200,
    #     gamma= 1.0,
    # )
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
        activations=activations,
        lr_schedule=lr_schedule,     # LR will vary: lr * lr_schedule(step)
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
            'num_epochs': num_epochs,
            'final_loss': results['final_metrics'].loss,
        }
    )

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()

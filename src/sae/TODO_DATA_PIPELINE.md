# Data Pipeline & Evaluation Separation Plan

## Goals

1. **Proper train/test split** - Never evaluate on training data
2. **Composable datasets** - Train on A+B, evaluate on A+C
3. **Strong train/eval barrier** - Save model to disk, load fresh for evaluation
4. **DataLoader integration** - Use PyTorch DataLoader for batching/shuffling

## Design Decisions

### Use Pure PyTorch DataLoader (Option A)

Fits library-over-framework philosophy:
- `torch.utils.data.Dataset` for activation datasets
- `torch.utils.data.DataLoader` for batching/shuffling
- `torch.utils.data.random_split` for train/test splits
- `torch.utils.data.ConcatDataset` for combining datasets

### Activation-Level vs Text-Level Datasets

Two approaches:

**A. Text-level datasets** (extract activations on-the-fly)
```python
texts = TextDataset("pile_samples.json")
train_texts, test_texts = random_split(texts, [0.9, 0.1])
# Extract activations during training loop
```

**B. Activation-level datasets** (pre-extract, then split)
```python
activations = extract_activations(model, all_texts, layer_idx)
dataset = ActivationDataset(activations)
train_set, test_set = random_split(dataset, [0.9, 0.1])
```

**Recommendation:** Option B for now (simpler, faster training loops)
- Pre-extract all activations once
- Split the activation tensor
- Train on train split, evaluate on test split

### Train/Eval Separation

**Current flow:**
```
load_data → extract_activations → train_sae → evaluate → save_checkpoint
```

**New flow (separated):**
```
# training_experiment.py
load_data → extract_activations → split → train_sae → save_checkpoint

# evaluation_experiment.py
load_checkpoint → load_eval_data → extract_activations → evaluate → save_report
```

This simulates:
- Training on GPU A
- Saving checkpoint to shared storage
- Loading on GPU B for evaluation

## Implementation Plan

### Phase 1: DataLoader Integration

1. **Create `ActivationDataset`** in `src/sae/data/datasets.py`
   ```python
   class ActivationDataset(Dataset):
       def __init__(self, activations: torch.Tensor):
           self.activations = activations

       def __len__(self):
           return len(self.activations)

       def __getitem__(self, idx):
           return self.activations[idx]
   ```

2. **Create `split_dataset()` helper**
   ```python
   def split_dataset(dataset, train_frac=0.9, seed=42):
       train_size = int(len(dataset) * train_frac)
       test_size = len(dataset) - train_size
       return random_split(dataset, [train_size, test_size],
                          generator=torch.Generator().manual_seed(seed))
   ```

3. **Update `TrainPipeline`** to accept DataLoader instead of raw tensor
   - OR create new `TrainPipelineV2` that uses DataLoader
   - Keep backward compatibility

### Phase 2: Train/Eval Separation

1. **Create `src/sae/experiments/train_sae.py`**
   - Training-only script
   - Saves checkpoint at the end
   - No evaluation code

2. **Create `src/sae/experiments/evaluate_sae.py`**
   - Loads checkpoint from disk
   - Loads evaluation data (can be different from training data)
   - Runs all evaluation metrics
   - Saves report

3. **Update checkpoint format** to include:
   - SAE weights
   - Activation mean
   - Training config (for reproducibility)
   - Data config (which dataset, which split)

### Phase 3: Multi-Dataset Support

1. **Create dataset registry/factory**
   ```python
   datasets = {
       "pile": "src/sae/data/pile_samples.json",
       "wikipedia": "src/sae/data/wiki_samples.json",
       "code": "src/sae/data/code_samples.json",
   }
   ```

2. **Support combining datasets**
   ```python
   train_data = combine_datasets(["pile", "wikipedia"])
   eval_data = combine_datasets(["pile", "code"])
   ```

## File Structure (After Implementation)

```
src/sae/
├── data/
│   ├── datasets.py          # NEW: ActivationDataset, TextDataset
│   ├── loader.py             # UPDATE: add split_dataset()
│   └── pile_samples.json
│
├── experiments/
│   ├── train_sae.py          # NEW: Training-only script
│   ├── evaluate_sae.py       # NEW: Evaluation-only script
│   ├── simple_sae_experiment.py  # KEEP: Combined for quick experiments
│   └── deep_sae_experiment.py
│
├── training/
│   └── train_pipeline.py     # UPDATE: Support DataLoader
```

## Usage Example (After Implementation)

```python
# 1. Training (train_sae.py)
from src.sae.data import load_pile_samples, ActivationDataset, split_dataset

# Load and extract
texts = load_pile_samples(num_samples=10000)
activations = extract_activations(model, tokenizer, texts, layer_idx=3)

# Split
dataset = ActivationDataset(activations)
train_set, test_set = split_dataset(dataset, train_frac=0.9)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

# Train
sae = SimpleSAE(...)
train_pipeline = TrainPipeline(sae, optimizer, train_loader)
results = train_pipeline.train_sae(num_epochs=100)

# Save
save_checkpoint("checkpoints/my_sae", sae, results['activation_mean'], {
    "train_dataset": "pile",
    "train_samples": 9000,
    "layer_idx": 3,
})

# 2. Evaluation (evaluate_sae.py) - SEPARATE SCRIPT
sae, activation_mean, config = load_checkpoint("checkpoints/my_sae")

# Load DIFFERENT data for evaluation
eval_texts = load_pile_samples(num_samples=1000)  # Could be different dataset
eval_activations = extract_activations(model, tokenizer, eval_texts, layer_idx=3)

# Run evaluation
report = evaluate_sae(sae, eval_activations, activation_mean)
report.save("reports/my_sae_eval")
```

## Open Questions

1. Should we support streaming/lazy loading for very large datasets?
2. Should activation extraction happen inside DataLoader (on-the-fly) or pre-computed?
3. How to handle different sequence lengths when batching?

## Priority

1. **HIGH:** Train/test split (we're evaluating on training data right now!)
2. **MEDIUM:** DataLoader integration
3. **MEDIUM:** Train/eval separation
4. **LOW:** Multi-dataset support

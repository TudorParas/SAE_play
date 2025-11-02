#!/usr/bin/env python

"""Train a simple sparse autoencoder on EleutherAI Pythia layer activations."""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_TRAIN_PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Neural networks learn statistical patterns from data.",
    "Mechanistic interpretability studies how models represent concepts.",
    "Pythia is an open language model released by EleutherAI.",
    "Sparse autoencoders encourage features to stay inactive.",
    "Transformers rely on residual streams that mix layer contributions.",
    "Oxford is known for its historic colleges and research centers.",
    "The gradient of a loss function guides parameter updates.",
    "Language models can be probed using textual prompts.",
    "We can analyze activations to understand learned features.",
    "Short stories provide dynamic combinations of words.",
    "Reinforcement learning balances exploration with exploitation.",
    "Computer vision models process pixels to detect structure.",
    "Mathematics offers a precise language for abstract ideas.",
    "Open source tooling accelerates machine learning research.",
    "The weather in London is often cloudy and mild.",
    "Cats and dogs have different behaviors and instincts.",
    "Space exploration inspires advances in engineering.",
    "Probability theory helps quantify uncertainty.",
    "Coding practice builds intuition over time.",
]

DEFAULT_PROBE_PROMPTS = [
    "Dogs are man's best friend.",
    "Neural networks process information.",
    "London is a major city.",
]


def chunked(items: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def resolve_device(device_str: Optional[str]) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SparseAutoencoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_size, hidden_size, bias=True)
        self.decoder = nn.Linear(hidden_size, input_size, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        codes = F.relu(self.encoder(x))
        recon = self.decoder(codes)
        return recon, codes

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.encoder(x))

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return self.decoder(codes)


def load_training_prompts(train_file: Optional[Path], num_prompts: int) -> List[str]:
    prompts: List[str] = []
    if train_file:
        file_text = train_file.read_text(encoding="utf-8")
        prompts.extend([line.strip() for line in file_text.splitlines() if line.strip()])
    prompts.extend(DEFAULT_TRAIN_PROMPTS)
    seen = set()
    deduped: List[str] = []
    for prompt in prompts:
        if prompt in seen:
            continue
        deduped.append(prompt)
        seen.add(prompt)
    if not deduped:
        raise ValueError("No training prompts available.")
    random.shuffle(deduped)
    if num_prompts:
        deduped = deduped[:num_prompts]
    return deduped


def collect_layer_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: Sequence[str],
    *,
    layer_index: int,
    max_length: int,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    num_layers = model.config.num_hidden_layers
    if layer_index < 0 or layer_index >= num_layers:
        raise ValueError(f"Layer index {layer_index} is out of range for model with {num_layers} layers.")
    features: List[torch.Tensor] = []
    model.eval()
    for batch in chunked(texts, batch_size):
        batch_encoding = tokenizer(
            list(batch),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        attention_mask = batch_encoding["attention_mask"].clone()
        batch_inputs = {k: v.to(device) for k, v in batch_encoding.items()}
        with torch.no_grad():
            outputs = model(**batch_inputs, output_hidden_states=True, use_cache=False)
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden states; set output_hidden_states=True.")
        hidden = hidden_states[layer_index + 1].detach().cpu()
        flattened = hidden.view(-1, hidden.size(-1))
        active_positions = attention_mask.view(-1).bool()
        features.append(flattened[active_positions])
    if not features:
        raise ValueError("No activations were captured.")
    activations = torch.cat(features, dim=0).to(torch.float32)
    return activations


def train_autoencoder(
    activations: torch.Tensor,
    hidden_size: int,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    l1_weight: float,
    device: torch.device,
    activity_threshold: float,
) -> tuple[SparseAutoencoder, torch.Tensor]:
    mean = activations.mean(dim=0, keepdim=True)
    centered = activations - mean
    dataset = TensorDataset(centered)
    if len(dataset) == 0:
        raise ValueError("Need at least one activation vector to train the autoencoder.")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    autoencoder = SparseAutoencoder(centered.size(1), hidden_size).to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        epoch_recon = 0.0
        epoch_l1 = 0.0
        epoch_active = 0.0
        total = 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstruction, codes = autoencoder(batch)
            recon_loss = F.mse_loss(reconstruction, batch)
            l1_term = codes.abs().mean()
            loss = recon_loss + l1_weight * l1_term
            loss.backward()
            optimizer.step()
            batch_size_actual = batch.size(0)
            total += batch_size_actual
            epoch_recon += recon_loss.item() * batch_size_actual
            epoch_l1 += l1_term.item() * batch_size_actual
            density = (codes.abs() > activity_threshold).float().mean().item()
            epoch_active += density * batch_size_actual
        avg_recon = epoch_recon / total
        avg_l1 = epoch_l1 / total
        avg_density = epoch_active / total
        logging.info(
            "Epoch %d/%d | recon=%.4f | l1=%.4f | active_frac=%.3f",
            epoch,
            epochs,
            avg_recon,
            avg_l1,
            avg_density,
        )
    return autoencoder, mean


def inspect_prompts(
    prompts: Sequence[str],
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layer_index: int,
    max_length: int,
    device: torch.device,
    autoencoder: SparseAutoencoder,
    mean: torch.Tensor,
    top_k: int,
    activity_threshold: float,
) -> None:
    logging.info("Inspecting %d probe prompts", len(prompts))
    for text in prompts:
        activations = collect_layer_activations(
            model,
            tokenizer,
            [text],
            layer_index=layer_index,
            max_length=max_length,
            device=device,
            batch_size=1,
        )
        centered = activations - mean
        with torch.no_grad():
            codes = autoencoder.encode(centered.to(device))
        codes_cpu = codes.cpu()
        avg_code = codes_cpu.mean(dim=0)
        k = min(top_k, avg_code.numel())
        top_values, top_indices = torch.topk(avg_code, k)
        active_fraction = (codes_cpu.abs() > activity_threshold).float().mean().item()
        print("=" * 72)
        print(f"Prompt: {text}")
        print(f"Tokens analyzed: {codes_cpu.size(0)}")
        print(f"Active fraction @>{activity_threshold:.3f}: {active_fraction:.3f}")
        print("Top features (index -> average activation):")
        for idx, value in zip(top_indices.tolist(), top_values.tolist()):
            print(f"  feature {idx:4d}: {value:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a sparse autoencoder on Pythia hidden activations for mechanistic interpretability experiments."
    )
    parser.add_argument("--model-name", type=str, default="EleutherAI/pythia-70m-deduped", help="Model identifier from Hugging Face Hub.")
    parser.add_argument("--layer-index", type=int, default=3, help="Zero-based transformer block index to probe.")
    parser.add_argument("--max-length", type=int, default=128, help="Maximum tokenized length for prompts.")
    parser.add_argument("--collection-batch-size", type=int, default=4, help="Batch size while collecting activations.")
    parser.add_argument("--activation-limit", type=int, default=2000, help="Optional cap on activation vectors used for training.")
    parser.add_argument("--hidden-features", type=int, default=2048, help="Number of sparse features in the autoencoder.")
    parser.add_argument("--ae-batch-size", type=int, default=128, help="Batch size for autoencoder training.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of autoencoder training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Autoencoder learning rate.")
    parser.add_argument("--l1-weight", type=float, default=1e-3, help="L1 sparsity weight added to the reconstruction loss.")
    parser.add_argument("--activity-threshold", type=float, default=0.05, help="Threshold used to count active SAE features.")
    parser.add_argument("--num-train-prompts", type=int, default=24, help="Number of prompts used to collect activations.")
    parser.add_argument("--train-file", type=Path, help="Optional newline-delimited file containing extra training prompts.")
    parser.add_argument("--probe-text", action="append", dest="probe_texts", help="Additional probe prompt to inspect after training.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top features to display per probe prompt.")
    parser.add_argument("--device", type=str, default=None, help="Torch device string, e.g. 'cuda' or 'cpu'. Defaults to CUDA if available.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    set_seed(args.seed)
    device = resolve_device(args.device)
    logging.info("Using device %s", device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(device)
    model.eval()
    model.config.use_cache = False
    train_prompts = load_training_prompts(args.train_file, args.num_train_prompts)
    logging.info("Collecting layer %d activations from %d prompts", args.layer_index, len(train_prompts))
    activations = collect_layer_activations(
        model,
        tokenizer,
        train_prompts,
        layer_index=args.layer_index,
        max_length=args.max_length,
        device=device,
        batch_size=args.collection_batch_size,
    )
    logging.info("Collected %d activation vectors of width %d", activations.size(0), activations.size(1))
    if args.activation_limit and activations.size(0) > args.activation_limit:
        logging.info("Subsampling activations to %d vectors", args.activation_limit)
        indices = torch.randperm(activations.size(0))[: args.activation_limit]
        activations = activations[indices]
    autoencoder, mean = train_autoencoder(
        activations,
        args.hidden_features,
        epochs=args.epochs,
        batch_size=args.ae_batch_size,
        lr=args.learning_rate,
        l1_weight=args.l1_weight,
        device=device,
        activity_threshold=args.activity_threshold,
    )
    autoencoder.eval()
    probe_prompts: List[str] = []
    if args.probe_texts:
        probe_prompts.extend([text for text in args.probe_texts if text])
    if not probe_prompts:
        probe_prompts = list(DEFAULT_PROBE_PROMPTS)
    inspect_prompts(
        probe_prompts,
        model=model,
        tokenizer=tokenizer,
        layer_index=args.layer_index,
        max_length=args.max_length,
        device=device,
        autoencoder=autoencoder,
        mean=mean,
        top_k=args.top_k,
        activity_threshold=args.activity_threshold,
    )
    logging.info("Finished sparse autoencoder experiment.")


if __name__ == "__main__":
    main()

"""
train.py - Complete Training Pipeline for Tiny-GPT

This script orchestrates the training of Tiny-GPT:
1. Loads training and validation datasets.
2. Tokenizes the text (using tiktoken BPE or built-in character-level tokenizer).
3. Instantiates the GPT model based on hardware-friendly presets.
4. Sets up the AdamW optimizer with selective weight decay and a cosine learning rate schedule.
5. Executes the training loop with periodic validation, checkpointing, and gradient clipping.
6. Visualizes and saves the loss trajectory curve.
"""

import os
import sys
import time
import math
import json
import argparse
from typing import Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from config import GPTConfig, CONFIGS
from model import GPT


# ==============================================================================
# Tokenizer Utilities (BPE or Built-in Character-Level)
# ==============================================================================

class CharTokenizer:
    """A simple, robust character-level tokenizer requiring zero external dependencies."""
    def __init__(self, chars: list):
        self.chars = sorted(list(set(chars)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text: str) -> list:
        # Map unknown characters to space or first character gracefully
        return [self.char_to_idx.get(c, 0) for c in text]

    def decode(self, indices: list) -> str:
        return "".join([self.idx_to_char.get(i, "") for i in indices])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "char",
            "vocab_size": self.vocab_size,
            "chars": self.chars,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CharTokenizer":
        return cls(chars=data["chars"])


class BPETokenizer:
    """Wrapper around OpenAI's tiktoken BPE tokenizer."""
    def __init__(self, encoding_name: str = "gpt2"):
        import tiktoken
        self.enc = tiktoken.get_encoding(encoding_name)
        self.encoding_name = encoding_name
        self.vocab_size = self.enc.n_vocab

    def encode(self, text: str) -> list:
        return self.enc.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, indices: list) -> str:
        return self.enc.decode(indices)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "tiktoken",
            "encoding_name": self.encoding_name,
            "vocab_size": self.vocab_size,
        }


def get_tokenizer(mode: str, sample_text: str):
    """
    Select and initialize the appropriate tokenizer based on user choice and availability.
    """
    if mode in ("auto", "tiktoken"):
        try:
            tok = BPETokenizer("gpt2")
            print(f"[+] Loaded tiktoken BPE tokenizer (vocab size: {tok.vocab_size:,})")
            return tok
        except ImportError:
            if mode == "tiktoken":
                print("[-] Error: tiktoken requested but not installed. Falling back to char tokenizer.")
            else:
                print("[*] tiktoken not installed. Using built-in character tokenizer.")
                
    # Fallback to character-level tokenizer
    unique_chars = sorted(list(set(sample_text)))
    tok = CharTokenizer(unique_chars)
    print(f"[+] Built character tokenizer (vocab size: {tok.vocab_size} unique characters)")
    return tok


# ==============================================================================
# Learning Rate Schedule (Linear Warmup + Cosine Annealing Decay)
# ==============================================================================

def get_lr(it: int, config: GPTConfig) -> float:
    """
    Calculate the learning rate for a given iteration step `it`.
    
    Schedule:
    1. Linear warmup: lr increases linearly from 0 to learning_rate over warmup_iters.
    2. Cosine decay: lr decays along a cosine curve from learning_rate down to min_lr.
    3. Beyond max_iters: lr stays constant at min_lr.
    """
    # 1) Linear warmup phase
    if it < config.warmup_iters:
        return config.learning_rate * (it + 1) / (config.warmup_iters + 1)
        
    # 2) Constant tail phase if trained past max_iters
    if it > config.max_iters:
        return config.min_lr
        
    # 3) Cosine decay phase
    decay_ratio = (it - config.warmup_iters) / (config.max_iters - config.warmup_iters)
    assert 0.0 <= decay_ratio <= 1.0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # Coeff ranges from 1.0 down to 0.0
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


# ==============================================================================
# Validation Loss Estimator
# ==============================================================================

@torch.no_grad()
def estimate_loss(model: nn.Module, get_batch_fn, eval_iters: int) -> Dict[str, float]:
    """
    Estimate the loss on train and validation splits by averaging over eval_iters batches.
    Switches model to eval mode and disables gradients to preserve memory and speed.
    """
    out = {}
    model.eval() # Disable dropout
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch_fn(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = float(losses.mean())
    model.train() # Re-enable dropout
    return out


# ==============================================================================
# Loss Curve Visualizer
# ==============================================================================

def plot_loss_curve(history: Dict[str, list], save_path: str):
    """Generate and save training and validation loss curve using matplotlib."""
    try:
        import matplotlib
        matplotlib.use('Agg') # Headless backend for terminal / server execution
        import matplotlib.pyplot as plt

        steps = history["steps"]
        train_loss = history["train_loss"]
        val_loss = history["val_loss"]

        plt.figure(figsize=(10, 6))
        plt.plot(steps, train_loss, label="Training Loss", color="#1f77b4", alpha=0.8, linewidth=2)
        plt.plot(steps, val_loss, label="Validation Loss", color="#ff7f0e", alpha=0.8, linewidth=2)

        if val_loss:
            min_val = min(val_loss)
            min_step = steps[val_loss.index(min_val)]
            plt.scatter([min_step], [min_val], color="red", s=80, zorder=5,
                        label=f"Best Val: {min_val:.4f} (Step {min_step})")

        plt.title("Tiny-GPT Training Trajectory", fontsize=14, fontweight="bold")
        plt.xlabel("Optimization Step", fontsize=12)
        plt.ylabel("Cross-Entropy Loss", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[✓] Loss trajectory plot saved to: {os.path.abspath(save_path)}")
    except Exception as exc:
        print(f"[-] Could not generate loss plot ({exc}). Skipping visual plot.")


# ==============================================================================
# Main Training Function
# ==============================================================================

def train():
    parser = argparse.ArgumentParser(description="Train Miniature Educational GPT")
    parser.add_argument("--config", type=str, default="tiny", choices=["tiny", "small", "medium"],
                        help="Model size preset (tiny: ~1M params, small: ~10M, medium: ~50M)")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory containing prepared dataset files")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--max_iters", type=int, default=None,
                        help="Override maximum training iterations")
    parser.add_argument("--eval_interval", type=int, default=None,
                        help="Override evaluation interval")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="Override peak learning rate")
    parser.add_argument("--device", type=str, default=None,
                        help="Execution device: cpu, cuda, or mps (default: auto-detect)")
    parser.add_argument("--tokenizer", type=str, default="auto", choices=["auto", "tiktoken", "char"],
                        help="Tokenizer type to use")
    parser.add_argument("--plot_path", type=str, default="loss_curve.png",
                        help="Path to save the training loss plot")
    args = parser.parse_args()

    # 1. Device Selection
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    device_type = "cuda" if "cuda" in device else "cpu"
    print(f"[*] Selected execution device: {device.upper()}")

    # 2. Verify Data Availability
    train_file = os.path.join(args.data_dir, "train.txt")
    val_file = os.path.join(args.data_dir, "val.txt")
    if not (os.path.exists(train_file) and os.path.exists(val_file)):
        print(f"[-] Missing dataset files in '{args.data_dir}'. Auto-running data/prepare.py...")
        from data.prepare import download_corpus, prepare_dataset, DEFAULT_DATA_URL
        raw = download_corpus(DEFAULT_DATA_URL, os.path.join(args.data_dir, "input.txt"))
        prepare_dataset(raw, args.data_dir)

    with open(train_file, "r", encoding="utf-8") as f:
        train_text = f.read()
    with open(val_file, "r", encoding="utf-8") as f:
        val_text = f.read()

    # 3. Setup Tokenizer
    tokenizer = get_tokenizer(args.tokenizer, train_text + val_text)
    
    print("[*] Tokenizing dataset splits into PyTorch tensors...")
    train_tokens = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    val_tokens = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)
    print(f"  • Training tokens:   {len(train_tokens):,}")
    print(f"  • Validation tokens: {len(val_tokens):,}")

    # 4. Load Configuration and Adjust to Tokenizer Vocab Size
    config = CONFIGS[args.config]
    config.vocab_size = tokenizer.vocab_size
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.max_iters is not None:
        config.max_iters = args.max_iters
    if args.eval_interval is not None:
        config.eval_interval = args.eval_interval
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate

    # 5. Educational Guidance: Theoretical Expected Loss
    expected_random_loss = math.log(config.vocab_size)
    print("\n" + "=" * 65)
    print(f"TINY-GPT ARCHITECTURE & TRAINING SETUP ({args.config.upper()} PRESET)")
    print("=" * 65)
    print(f"  • Vocab Size:       {config.vocab_size:,} tokens")
    print(f"  • Context Length:   {config.context_length} tokens (sequence block size)")
    print(f"  • Hidden Dimension: {config.n_embd} (n_embd)")
    print(f"  • Attention Heads:  {config.n_head} (head dimension: {config.n_embd // config.n_head})")
    print(f"  • Transformer Blocks: {config.n_layer} layers")
    print(f"  • Batch Size:       {config.batch_size}")
    print(f"  • Max Iterations:   {config.max_iters}")
    print(f"  • Peak Learning Rate: {config.learning_rate}")
    print("-" * 65)
    print("EDUCATIONAL METRICS TO EXPECT:")
    print(f"  1. Initial Random Guess Loss: -ln(1 / V) = ln({config.vocab_size}) ≈ {expected_random_loss:.2f}")
    print("     (At step 0, if the model guesses uniformly at random, loss will start near this value!)")
    print("  2. Good Convergence Loss:    ≈ 1.5 - 2.5 (Perplexity: e^loss ≈ 4.5 - 12.2)")
    print("=" * 65 + "\n")

    # 6. Instantiate Model
    torch.manual_seed(1337)
    model = GPT(config)
    model.to(device)

    total_params = model.get_num_params()
    non_emb_params = model.get_num_params(non_embedding=True)
    print(f"[+] Model instantiated successfully!")
    print(f"  • Total parameters:          {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  • Non-embedding parameters:  {non_emb_params:,} ({non_emb_params/1e6:.2f}M)")

    # 7. Batch Sampler
    def get_batch(split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        data = train_tokens if split == 'train' else val_tokens
        # Sample random starting indices
        max_start = len(data) - config.context_length - 1
        if max_start <= 0:
            raise ValueError(f"Dataset for {split} is smaller than context_length ({config.context_length}).")
        ix = torch.randint(0, max_start, (config.batch_size,))
        x = torch.stack([data[i:i + config.context_length] for i in ix])
        # Target y is shifted by 1 token forward
        y = torch.stack([data[i + 1:i + 1 + config.context_length] for i in ix])
        if "cuda" in device:
            # Pin and transfer asynchronously for GPU efficiency
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    # 8. Setup Optimizer
    optimizer = model.configure_optimizers(
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        betas=(0.9, 0.95),
        device_type=device_type
    )

    # 9. Training Loop
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')
    history = {"steps": [], "train_loss": [], "val_loss": []}
    
    t0 = time.time()
    step_times = []
    
    print("[*] Commencing training loop...\n")
    try:
        for step in range(config.max_iters + 1):
            t_step_start = time.time()
            
            # --- Evaluation & Checkpointing ---
            if step % config.eval_interval == 0 or step == config.max_iters:
                eval_start = time.time()
                losses = estimate_loss(model, get_batch, config.eval_iters)
                eval_time = time.time() - eval_start
                
                history["steps"].append(step)
                history["train_loss"].append(losses["train"])
                history["val_loss"].append(losses["val"])
                
                is_best = losses["val"] < best_val_loss
                if is_best:
                    best_val_loss = losses["val"]
                    
                # Save checkpoint
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config.to_dict(),
                    "tokenizer": tokenizer.to_dict(),
                    "step": step,
                    "train_loss": losses["train"],
                    "val_loss": losses["val"],
                    "best_val_loss": best_val_loss,
                }
                
                # Always save latest
                latest_path = os.path.join(config.checkpoint_dir, "latest_model.pt")
                torch.save(checkpoint, latest_path)
                
                # Save best checkpoint
                best_marker = ""
                if is_best:
                    best_path = os.path.join(config.checkpoint_dir, "best_model.pt")
                    torch.save(checkpoint, best_path)
                    best_marker = " [★ Best Checkpoint Saved!]"

                curr_lr = get_lr(step, config)
                print(f"Step {step:5d}/{config.max_iters} | "
                      f"Train Loss: {losses['train']:.4f} | "
                      f"Val Loss: {losses['val']:.4f} | "
                      f"LR: {curr_lr:.2e} | "
                      f"Eval Time: {eval_time:.1f}s{best_marker}")

            if step == config.max_iters:
                break

            # --- Update Learning Rate ---
            lr = get_lr(step, config)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # --- Forward Pass & Loss Calculation ---
            X, Y = get_batch('train')
            logits, loss = model(X, Y)

            # --- Backward Pass & Optimization ---
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Gradient Clipping: prevents gradients from exploding in deep transformer layers
            if config.grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            optimizer.step()

            # Record step timing
            t_step = time.time() - t_step_start
            step_times.append(t_step)
            if len(step_times) > 100:
                step_times.pop(0)

            # Quick status log every 100 steps
            if step % 100 == 0 and step > 0 and step % config.eval_interval != 0:
                avg_step_ms = (sum(step_times) / len(step_times)) * 1000
                rem_steps = config.max_iters - step
                rem_seconds = (sum(step_times) / len(step_times)) * rem_steps
                print(f"  step {step:5d} | loss: {loss.item():.4f} | "
                      f"step time: {avg_step_ms:.1f}ms | ETA: {rem_seconds/60:.1f}m")

    except KeyboardInterrupt:
        print("\n[!] Training paused early by user. Saving current state...")

    total_training_time = time.time() - t0
    print("\n" + "=" * 65)
    print(f"[✓] Training concluded in {total_training_time/60:.2f} minutes.")
    print(f"  • Best Validation Loss: {best_val_loss:.4f} (Perplexity: {math.exp(best_val_loss):.2f})")
    print(f"  • Model checkpoints stored in: {os.path.abspath(config.checkpoint_dir)}/")
    print("=" * 65)

    # 10. Generate and save loss curve
    if len(history["steps"]) > 1:
        plot_loss_curve(history, args.plot_path)

    print("\nNext step: Generate text from your trained model!")
    print("Run:")
    print("  python generate.py --prompt \"Once upon a time\" --temperature 0.8\n")


if __name__ == "__main__":
    train()

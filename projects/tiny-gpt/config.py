"""
config.py - Hyperparameter Configurations for Tiny-GPT

This module defines configuration classes and presets for Tiny-GPT.
It is structured to be transparent and educational: every hyperparameter
is documented with its mathematical intuition, role in the Transformer
architecture, and typical resource footprint.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class GPTConfig:
    """
    Configuration dataclass containing all architectural and training hyperparameters.
    
    Attributes:
        vocab_size (int): Total number of unique tokens in the vocabulary.
            - Default 50,257 matches GPT-2 BPE tokenizer.
            - For character-level tokenization, this will automatically adjust
              to the number of distinct characters (typically ~65 for English text).
        context_length (int): Maximum sequence length (also known as block_size or
            context window). The model cannot attend to tokens further apart than this.
        n_embd (int): Dimensionality of the token embeddings and hidden states.
            Must be divisible by `n_head`.
        n_head (int): Number of parallel attention heads.
            Each head operates on a subspace of dimension d_k = n_embd / n_head.
        n_layer (int): Number of stacked Transformer decoder blocks.
        dropout (float): Dropout probability applied across embeddings, attention
            weights, and feed-forward projections to regularize and prevent overfitting.
        bias (bool): Whether to include learnable bias vectors in Linear layers and
            LayerNorms. Following modern practice (like LLaMA and nanoGPT), disabling
            biases (`bias=False`) slightly speeds up computation and improves generalization.
            
        batch_size (int): Number of independent sequences processed simultaneously
            in each forward/backward training step.
        learning_rate (float): Maximum learning rate for the AdamW optimizer.
        min_lr (float): Minimum learning rate at the end of the cosine decay schedule.
            Usually set to 10% of learning_rate.
        max_iters (int): Total number of training optimization steps.
        eval_interval (int): How frequently (in steps) to pause training and evaluate
            validation loss.
        eval_iters (int): Number of random validation batches averaged during evaluation
            to obtain a reliable, low-variance loss estimate.
        warmup_iters (int): Number of initial steps during which learning rate increases
            linearly from 0 to `learning_rate` to stabilize early training.
        weight_decay (float): L2 regularization factor for 2D weight matrices in AdamW.
            Prevents weights from growing too large.
        grad_clip (float): Maximum norm for gradient clipping to prevent gradient explosion.
        
        data_path (str): Path to the primary input text file.
        checkpoint_dir (str): Directory where model checkpoints will be stored.
    """
    # --- Model Architecture ---
    vocab_size: int = 50257      # GPT-2 tokenizer vocab size (adjusted dynamically if char-level)
    context_length: int = 256    # Maximum sequence length (block size)
    n_embd: int = 256            # Embedding dimension (hidden dimension d_model)
    n_head: int = 8              # Attention heads (n_embd must be divisible by n_head)
    n_layer: int = 6             # Number of transformer blocks
    dropout: float = 0.1         # Dropout probability (0.0 to disable during inference)
    bias: bool = False           # Modern standard: bias=False in Linear and LayerNorm

    # --- Training Hyperparameters ---
    batch_size: int = 32         # Batch size (sequences per step)
    learning_rate: float = 3e-4  # Peak learning rate for AdamW
    min_lr: float = 3e-5         # Minimum learning rate (cosine decay target)
    max_iters: int = 5000        # Total optimization steps
    eval_interval: int = 500     # Evaluation step interval
    eval_iters: int = 50         # Number of evaluation batches to average
    warmup_iters: int = 100      # Linear warmup steps
    weight_decay: float = 0.1    # Weight decay coefficient (non-bias/non-norm weights)
    grad_clip: float = 1.0       # Max gradient norm

    # --- Storage & Paths ---
    data_path: str = "data/input.txt"
    checkpoint_dir: str = "checkpoints"

    def __post_init__(self):
        """Validate consistency of hyperparameters upon initialization."""
        assert self.n_embd % self.n_head == 0, (
            f"Embedding dimension n_embd ({self.n_embd}) must be divisible by "
            f"the number of attention heads n_head ({self.n_head})."
        )
        assert 0.0 <= self.dropout < 1.0, f"Dropout ({self.dropout}) must be in [0.0, 1.0)."
        assert self.learning_rate > 0, "Learning rate must be positive."
        assert self.min_lr <= self.learning_rate, "min_lr must be <= learning_rate."

    def to_dict(self) -> Dict[str, Any]:
        """Convert config dataclass to standard Python dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GPTConfig":
        """Instantiate a GPTConfig from a dictionary of parameters."""
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


# ==============================================================================
# Presets for Different Hardware Capacities
# ==============================================================================

# Tiny: ~1.2M parameters.
# Designed for ultra-fast training on modest dual-core CPU laptops (e.g., student laptops).
# Trains in ~5-15 minutes on CPU.
TINY_CONFIG = GPTConfig(
    vocab_size=50257,
    context_length=128,
    n_embd=128,
    n_head=4,
    n_layer=4,
    dropout=0.1,
    batch_size=32,
    learning_rate=5e-4,
    min_lr=5e-5,
    max_iters=2000,
    eval_interval=200,
    eval_iters=30,
    warmup_iters=50,
)

# Small: ~10M parameters.
# Balanced default. Trains comfortably on multi-core modern CPUs or entry-level GPUs.
# Ideal for a 1-hour training run producing coherent grammar and style.
SMALL_CONFIG = GPTConfig(
    vocab_size=50257,
    context_length=256,
    n_embd=256,
    n_head=8,
    n_layer=6,
    dropout=0.1,
    batch_size=32,
    learning_rate=3e-4,
    min_lr=3e-5,
    max_iters=5000,
    eval_interval=500,
    eval_iters=50,
    warmup_iters=100,
)

# Medium: ~45-50M parameters.
# Recommended for laptops with Apple Silicon (M1/M2/M3) or desktop GPUs (NVIDIA GTX/RTX).
# Captures nuanced multi-paragraph structure and rich vocabulary.
MEDIUM_CONFIG = GPTConfig(
    vocab_size=50257,
    context_length=512,
    n_embd=512,
    n_head=8,
    n_layer=8,
    dropout=0.1,
    batch_size=16,
    learning_rate=2e-4,
    min_lr=2e-5,
    max_iters=10000,
    eval_interval=500,
    eval_iters=50,
    warmup_iters=200,
)

# Configuration mapping dictionary for CLI argument parsing
CONFIGS = {
    "tiny": TINY_CONFIG,
    "small": SMALL_CONFIG,
    "medium": MEDIUM_CONFIG,
}

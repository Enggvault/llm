"""
model.py - Educational Miniature GPT Architecture in PyTorch

This module contains a complete, self-contained implementation of an
autoregressive Transformer language model (GPT-2 style).

Architectural Overview:
    Token IDs
       │
    ┌──▼──────────────────────────┐
    │  Token & Positional Embed   │ (wte + wpe)
    └──┬──────────────────────────┘
       │
    ┌──▼──────────────────────────┐
    │  Transformer Block 1        │
    │  - LayerNorm -> Attention   │ (Pre-LN Causal Self-Attention)
    │  - LayerNorm -> MLP         │ (Feed-Forward with GELU)
    └──┬──────────────────────────┘
       │           ...
    ┌──▼──────────────────────────┐
    │  Transformer Block N        │
    └──┬──────────────────────────┘
       │
    ┌──▼──────────────────────────┐
    │  Final LayerNorm (ln_f)     │
    └──┬──────────────────────────┘
       │
    ┌──▼──────────────────────────┐
    │  Linear LM Head             │ (Projection to vocabulary logits)
    └──┬──────────────────────────┘
       │
    Next Token Probabilities (Softmax)
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F

from config import GPTConfig


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention mechanism.
    
    Why "Causal"?
    Language generation is autoregressive: when predicting token t, the model
    must only attend to tokens at indices <= t. Future tokens (> t) are masked
    out by setting their attention weights to -infinity before softmax.
    
    Why "Multi-Head"?
    Multiple attention heads allow the model to jointly attend to information
    from different representation subspaces at different positions. For example,
    one head might track subject-verb agreements, while another tracks punctuation.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        
        # Batch linear projection for Query, Key, and Value in a single combined matrix.
        # Matrix size: [n_embd, 3 * n_embd]. This is mathematically equivalent to 3 separate
        # projections, but significantly more efficient on hardware (GEMM fusion).
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        
        # Output projection back to the embedding dimension
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Regularization dropouts
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Causal mask: Lower triangular matrix of ones.
        # Registered as a persistent buffer so it moves across devices (CPU/GPU) with the module
        # but is NOT treated as a learnable parameter.
        mask = torch.tril(torch.ones(config.context_length, config.context_length))
        self.register_buffer("bias", mask.view(1, 1, config.context_length, config.context_length))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd)
        """
        B, T, C = x.size() # Batch size, Sequence length, Embedding dimension (n_embd)
        
        # 1. Project input to Query, Key, and Value tensors
        # [B, T, C] -> [B, T, 3 * C]
        qkv = self.c_attn(x)
        # Split into separate Q, K, V tensors: each [B, T, C]
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # 2. Reshape for multi-head attention:
        # [B, T, C] -> [B, T, n_head, head_dim] -> [B, n_head, T, head_dim]
        # Transposing puts the head dimension before sequence length for parallel matrix multiplication.
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # 3. Scaled Dot-Product Attention:
        # Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d_k) + Mask ) * V
        if hasattr(F, 'scaled_dot_product_attention'):
            # PyTorch 2.0+ native FlashAttention / memory-efficient attention
            # Automatically applies causal masking when is_causal=True
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            # Pedagogical fallback showing explicit matrix multiplications:
            # Attention scores: [B, nh, T, hs] x [B, nh, hs, T] -> [B, nh, T, T]
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            # Apply causal mask: mask positions where bias is 0 with -infinity
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            # Weighted aggregation of values: [B, nh, T, T] x [B, nh, T, hs] -> [B, nh, T, hs]
            y = att @ v
            
        # 4. Concatenate heads back together:
        # [B, n_head, T, head_dim] -> [B, T, n_head, head_dim] -> [B, T, C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # 5. Output linear projection + residual dropout
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (Feed-Forward Network sublayer).
    
    Standard Transformer architecture expands the hidden dimension by a factor
    of 4 (4 * n_embd) in the intermediate layer, applies non-linear activation,
    and projects back to n_embd.
    
    Uses GELU (Gaussian Error Linear Unit) activation, which provides a smooth,
    non-zero probabilistic gradient for negative inputs.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        # First linear transformation: projects from n_embd to 4 * n_embd
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        # Smooth non-linear activation function
        self.gelu = nn.GELU()
        # Second linear transformation: projects back from 4 * n_embd to n_embd
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flow: Linear -> GELU -> Linear -> Dropout
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    Transformer Decoder Block (Pre-LayerNorm Variant).
    
    Pre-LN formulation (used in GPT-2, GPT-3, LLaMA):
        x = x + Attention(LayerNorm1(x))
        x = x + MLP(LayerNorm2(x))
        
    Compared to the original 2017 Post-LN formulation (LayerNorm(x + Sublayer(x))),
    Pre-LN maintains an unhindered identity highway through the residual stream,
    allowing gradients to flow directly from the final layer back to the input
    without vanishing or exploding.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, elementwise_affine=True)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, elementwise_affine=True)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connection around self-attention
        x = x + self.attn(self.ln_1(x))
        # Residual connection around feed-forward network
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """
    Full Autoregressive Generative Pretrained Transformer (GPT) model.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Model component containers
        self.transformer = nn.ModuleDict(dict(
            # Token Embedding Table: maps token indices [0, vocab_size-1] to vectors of size n_embd
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # Position Embedding Table: maps position indices [0, context_length-1] to vectors of size n_embd
            wpe = nn.Embedding(config.context_length, config.n_embd),
            # Input embedding dropout
            drop = nn.Dropout(config.dropout),
            # Stack of identical Transformer decoder blocks
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # Final Layer Normalization before prediction head
            ln_f = nn.LayerNorm(config.n_embd, elementwise_affine=True),
        ))
        
        # Language Model Head: projects final hidden state back to vocabulary logits
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight Tying: Tie weights between token embeddings (wte) and language model head (lm_head).
        # This reduces parameter count by V * d_model and improves generalization (Press & Wolf, 2017).
        self.transformer.wte.weight = self.lm_head.weight
        
        # Initialize all model weights according to GPT-2 specification
        self.apply(self._init_weights)
        
        # Special scaled initialization for residual projection layers as per GPT-2 paper:
        # Scale by 1 / sqrt(2 * n_layer) to prevent activation variance from growing with depth.
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module):
        """Initialize weights with small normal distribution (mean=0.0, std=0.02)."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            if module.weight is not None:
                torch.nn.init.ones_(module.weight)

    def get_num_params(self, non_embedding: bool = False) -> int:
        """
        Return the total number of parameters in the model.
        Args:
            non_embedding: If True, subtract position embeddings (and token embeddings if untied).
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the GPT model.
        
        Args:
            idx: Tensor of token indices, shape (batch_size, seq_len)
            targets: Optional tensor of ground-truth target token indices,
                     shape (batch_size, seq_len)
                     
        Returns:
            logits: Unnormalized next-token predictions, shape (batch_size, seq_len, vocab_size)
            loss: Cross-entropy loss scalar (None if targets is None)
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.context_length, (
            f"Cannot forward sequence of length {t}, model context length is {self.config.context_length}"
        )
        
        # Construct position indices [0, 1, 2, ..., t-1]
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        # 1. Forward through embeddings:
        tok_emb = self.transformer.wte(idx)           # Token embeddings: (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)           # Position embeddings: (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)  # Combined embeddings: (b, t, n_embd)
        
        # 2. Forward through all Transformer decoder blocks:
        for block in self.transformer.h:
            x = block(x)
            
        # 3. Apply final LayerNorm:
        x = self.transformer.ln_f(x)
        
        # 4. Project to vocabulary logits:
        if targets is not None:
            # Training mode: compute logits for all positions and compute cross-entropy loss
            logits = self.lm_head(x) # (b, t, vocab_size)
            # Flatten predictions and targets for cross_entropy: (b * t, vocab_size) and (b * t)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference mode: optimization trick - only compute logits for the final token position!
            # lm_head on x[:, [-1], :] is much faster than computing over all tokens.
            logits = self.lm_head(x[:, [-1], :]) # (b, 1, vocab_size)
            loss = None
            
        return logits, loss

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
        device_type: str
    ) -> torch.optim.Optimizer:
        """
        Configure AdamW optimizer with selective weight decay.
        
        Best practice (nanoGPT & PyTorch):
        - 2D parameters (matrix multiplications in Linear and Embeddings) -> undergo weight decay
        - 1D parameters (biases, LayerNorm scale and shifts) -> NO weight decay
        """
        # Separate parameters into decay and no_decay groups
        decay_params = []
        nodecay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                nodecay_params.append(param)
                
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        # Check if fused AdamW is available (fast on modern CUDA)
        use_fused = (device_type == 'cuda') and ('fused' in torch.optim.AdamW.__init__.__code__.co_varnames)
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Autoregressively generate new tokens given a conditioning prompt.
        
        Args:
            idx: Tensor of prompt token indices, shape (batch_size, seq_len)
            max_new_tokens: Number of consecutive tokens to generate
            temperature: Scaling factor for logits.
                - temperature = 1.0: standard categorical sampling
                - temperature < 1.0: sharper, more conservative predictions
                - temperature > 1.0: flatter, more creative/diverse predictions
            top_k: If specified, only sample from the top_k most probable tokens.
            
        Returns:
            Tensor of token indices including prompt and generated tokens,
            shape (batch_size, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # If the context sequence exceeds model's context window, crop to the last context_length tokens
            idx_cond = idx if idx.size(1) <= self.config.context_length else idx[:, -self.config.context_length:]
            
            # Forward the model to obtain logits for the last token position
            logits, _ = self(idx_cond)
            # Focus only on the last time step: (b, 1, vocab_size) -> (b, vocab_size)
            logits = logits[:, -1, :]
            
            # Apply temperature scaling:
            if temperature > 0.0:
                logits = logits / temperature
            else:
                # Deterministic greedy decoding when temperature is 0
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
                idx = torch.cat((idx, idx_next), dim=1)
                continue
                
            # Optional top-k filtering:
            if top_k is not None and top_k > 0:
                # Find the value of the k-th highest logit
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # Mask any token with logit value below the top-k cutoff
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            # Apply softmax to convert unnormalized logits to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample next token from the categorical distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append sampled token to running sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx

"""
generate.py - Text Generation and Inference Interface for Tiny-GPT

This script loads a trained checkpoint and generates text autoregressively
using temperature scaling and top-k sampling.

Educational Concepts:
1. Autoregressive Sampling: The model predicts one token at a time. Each generated
   token is appended to the sequence and fed back into the model for the next step.
2. Temperature:
   - High (> 1.0): Flatten probabilities -> more creative, diverse, sometimes chaotic.
   - Low (< 1.0): Peak probabilities -> conservative, grammatically predictable, repetitive.
   - Zero (0.0): Greedy argmax decoding (always picks the highest probability token).
3. Top-K Sampling:
   - Truncates the probability distribution to only the top-K highest-probability candidates.
   - Prevents the model from occasionally sampling nonsensical, long-tail tokens.
"""

import os
import sys
import argparse
import torch

from config import GPTConfig
from model import GPT
from train import CharTokenizer, BPETokenizer


SAMPLE_PROMPTS = [
    "Once upon a time",
    "To be or not to be, that is the",
    "The king spoke unto his people:",
    "ROMEO:\nWhat light through yonder window breaks?",
    "In the ancient hall of the mountain king,",
]


def load_model_and_tokenizer(checkpoint_path: str, device: str):
    """
    Restore the model architecture, weights, and tokenizer from a checkpoint file.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at: {checkpoint_path}\n"
            f"Please run 'python train.py' first to train the model and produce a checkpoint."
        )

    print(f"[*] Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 1. Restore Configuration
    config = GPTConfig.from_dict(checkpoint["config"])

    # 2. Restore Tokenizer
    tok_info = checkpoint.get("tokenizer", {})
    if tok_info.get("type") == "char":
        tokenizer = CharTokenizer.from_dict(tok_info)
    elif tok_info.get("type") == "tiktoken":
        tokenizer = BPETokenizer(tok_info.get("encoding_name", "gpt2"))
    else:
        # Fallback to GPT-2 tokenizer if not specified
        tokenizer = BPETokenizer("gpt2")

    # 3. Instantiate and Load Model Weights
    model = GPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval() # Set to evaluation mode (disables dropout)

    step = checkpoint.get("step", 0)
    val_loss = checkpoint.get("val_loss", None)
    loss_str = f" | Validation Loss: {val_loss:.4f}" if val_loss is not None else ""
    print(f"[+] Loaded model checkpoint from Step {step}{loss_str}")
    print(f"  • Parameters: {model.get_num_params():,}")
    print(f"  • Tokenizer:  {tok_info.get('type', 'default')} (Vocab: {tokenizer.vocab_size:,})")

    return model, tokenizer, config


def generate_text(
    model: GPT,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
    device: str = "cpu"
) -> str:
    """
    Encode prompt, run autoregressive sampling loop, and decode to text.
    """
    # Encode prompt string to list of token IDs
    prompt_tokens = tokenizer.encode(prompt)
    if len(prompt_tokens) == 0:
        prompt_tokens = [0] # Handle empty prompt

    # Create batch tensor: shape (1, seq_len)
    x = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)

    # Autoregressively sample new tokens
    with torch.no_grad():
        y = model.generate(
            idx=x,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k if top_k > 0 else None
        )

    # Extract output sequence and decode back to text
    out_tokens = y[0].tolist()
    generated_text = tokenizer.decode(out_tokens)
    return generated_text


def interactive_mode(model: GPT, tokenizer, config: GPTConfig, device: str):
    """Interactive command-line prompt session for students."""
    print("\n" + "=" * 65)
    print("Tiny-GPT Interactive Generation Console")
    print("Type your prompt and press Enter. (Type 'exit' or 'quit' to stop)")
    print("=" * 65)

    while True:
        try:
            prompt = input("\nPrompt >> ")
            if prompt.strip().lower() in ("exit", "quit", "q"):
                print("Exiting interactive console.")
                break
            if not prompt.strip():
                continue

            print("\nGenerating...")
            output = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=150,
                temperature=0.8,
                top_k=40,
                device=device
            )
            print("-" * 50)
            print(output)
            print("-" * 50)
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break


def main():
    parser = argparse.ArgumentParser(description="Generate text using Tiny-GPT")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt",
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                        help="Initial prompt string to condition the generation")
    parser.add_argument("--max_tokens", type=int, default=200,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (0.0 = greedy, < 1.0 = focused, > 1.0 = creative)")
    parser.add_argument("--top_k", type=int, default=40,
                        help="Top-K token sampling filtering cutoff (0 to disable)")
    parser.add_argument("--device", type=str, default=None,
                        help="Execution device: cpu, cuda, or mps (default: auto)")
    parser.add_argument("--interactive", action="store_true",
                        help="Launch interactive prompt REPL session")
    args = parser.parse_args()

    # Determine execution device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Fallback to latest_model.pt if best_model.pt is not yet available
    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path) and os.path.exists("checkpoints/latest_model.pt"):
        ckpt_path = "checkpoints/latest_model.pt"

    model, tokenizer, config = load_model_and_tokenizer(ckpt_path, device)

    if args.interactive:
        interactive_mode(model, tokenizer, config, device)
        return

    # Single-shot generation
    print("\n" + "=" * 65)
    print("SAMPLING HYPERPARAMETERS:")
    print(f"  • Prompt:        '{args.prompt}'")
    print(f"  • Max Tokens:    {args.max_tokens}")
    print(f"  • Temperature:   {args.temperature} ({'greedy' if args.temperature==0 else 'probabilistic'})")
    print(f"  • Top-K Cutoff:  {args.top_k}")
    print("=" * 65)

    result = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device
    )

    print("\n[GENERATED COMPLETION]:")
    print("-" * 65)
    print(result)
    print("-" * 65 + "\n")


if __name__ == "__main__":
    main()

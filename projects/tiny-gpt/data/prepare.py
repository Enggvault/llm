"""
data/prepare.py - Data Preparation Pipeline for Tiny-GPT

This script downloads or ingests an educational text corpus and formats it
for training Tiny-GPT. It partitions the dataset into training (90%) and
validation (10%) splits, computes dataset statistics, and creates vocabulary
mappings for character-level tokenization.
"""

import os
import sys
import json
import argparse
from typing import Tuple, Dict, Any

# Standard open source corpus URL (Karpathy Tiny Shakespeare dataset)
DEFAULT_DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

# Original fallback educational corpus if offline or network unavailable
OFFLINE_FALLBACK_TEXT = """
In an ancient kingdom nestled between misty mountains and whispering pines,
there lived a young scholar named Ethan who studied the patterns of language.
Every morning, as dawn painted the horizon in hues of rose and amber, Ethan
would climb the steps of the grand observatory to transcribe the ancient scrolls.

"Words," the Elder Sage had once taught him, "are like constellations in the night sky.
Each star holds its place, yet their meaning emerges only when viewed in harmony."

One chilly autumn evening, a traveler arrived carrying a leather-bound journal.
Inside were thousands of verses, dialogues, and historical chronicles gathered from
distant lands across the seas. The traveler presented the tome to Ethan, saying:
"Teach your mechanical counting machines to read these lines, so that the knowledge
of our ancestors may endure when parchment turns to dust."

Ethan set to work upon his workbench. He arranged the symbols into tables, counted
their frequencies, and observed how every question sought an answer, how every noun
sought its verb, and how cadence governed thought. Through long winter nights, the
oil lamp flickered as tokens flowed through sequences and matrices.

When spring arrived, the observatory garden burst into vibrant blossoms of lilac
and jasmine. The scholar gathered his students to demonstrate the working model.
He inscribed three words upon a fresh slate: "Once upon a..."
With quiet precision, the student mechanisms produced the completion: "...time."
Laughter and wonder filled the halls, marking the dawn of a new age of discovery.
""" * 50  # Repeat to provide sufficient volume for multiple training iterations


def download_corpus(url: str, dest_path: str) -> str:
    """
    Download the dataset from a remote URL using requests.
    Falls back to synthetic educational corpus if offline or unavailable.
    """
    print(f"[*] Attempting to download corpus from: {url}")
    try:
        import requests
        response = requests.get(url, timeout=15)
        if response.status_code == 200 and len(response.text) > 1000:
            print(f"[+] Download successful! Size: {len(response.text):,} characters.")
            return response.text
        else:
            print(f"[-] Download returned status {response.status_code}. Using fallback text.")
    except Exception as exc:
        print(f"[-] Network connection failed ({exc}). Using offline fallback text.")
    
    return OFFLINE_FALLBACK_TEXT.strip()


def prepare_dataset(
    input_text: str,
    output_dir: str,
    split_ratio: float = 0.90
) -> Dict[str, Any]:
    """
    Split text into train/val sets, save files, and compute vocabulary statistics.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save complete raw input
    input_path = os.path.join(output_dir, "input.txt")
    with open(input_path, "w", encoding="utf-8") as f:
        f.write(input_text)
        
    total_chars = len(input_text)
    split_idx = int(total_chars * split_ratio)
    
    train_text = input_text[:split_idx]
    val_text = input_text[split_idx:]
    
    train_path = os.path.join(output_dir, "train.txt")
    val_path = os.path.join(output_dir, "val.txt")
    
    with open(train_path, "w", encoding="utf-8") as f:
        f.write(train_text)
    with open(val_path, "w", encoding="utf-8") as f:
        f.write(val_text)
        
    # 2. Extract character-level vocabulary
    unique_chars = sorted(list(set(input_text)))
    char_vocab_size = len(unique_chars)
    
    char_to_idx = {ch: i for i, ch in enumerate(unique_chars)}
    idx_to_char = {i: ch for i, ch in enumerate(unique_chars)}
    
    meta = {
        "vocab_type": "char",
        "vocab_size": char_vocab_size,
        "chars": unique_chars,
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char,
        "total_chars": total_chars,
        "train_chars": len(train_text),
        "val_chars": len(val_text),
    }
    
    # 3. Check for tiktoken BPE availability
    bpe_tokens_est = total_chars // 4  # Standard English rule-of-thumb: ~4 chars per token
    has_tiktoken = False
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        sample_tokens = len(enc.encode(input_text[:50000]))
        bpe_tokens_est = int((sample_tokens / min(50000, total_chars)) * total_chars)
        has_tiktoken = True
    except ImportError:
        pass
        
    meta["has_tiktoken"] = has_tiktoken
    meta["estimated_bpe_tokens"] = bpe_tokens_est
    
    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        
    return meta


def main():
    parser = argparse.ArgumentParser(description="Prepare educational corpus for Tiny-GPT")
    parser.add_argument("--source", type=str, default=DEFAULT_DATA_URL,
                        help="URL or source to download dataset from")
    parser.add_argument("--input_file", type=str, default=None,
                        help="Path to an existing local text file instead of downloading")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Target directory to store processed data")
    parser.add_argument("--split_ratio", type=float, default=0.9,
                        help="Train/val split ratio (default: 0.9 = 90%% train, 10%% val)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Tiny-GPT: Educational Data Preparation")
    print("=" * 60)
    
    if args.input_file and os.path.exists(args.input_file):
        print(f"[*] Reading custom input file: {args.input_file}")
        with open(args.input_file, "r", encoding="utf-8") as f:
            raw_text = f.read()
    else:
        raw_text = download_corpus(args.source, os.path.join(args.output_dir, "input.txt"))
        
    meta = prepare_dataset(raw_text, args.output_dir, split_ratio=args.split_ratio)
    
    print("\nDataset Preparation Summary:")
    print(f"  • Total characters:       {meta['total_chars']:,}")
    print(f"  • Training characters:    {meta['train_chars']:,} ({args.split_ratio*100:.0f}%)")
    print(f"  • Validation characters:  {meta['val_chars']:,} ({(1-args.split_ratio)*100:.0f}%)")
    print(f"  • Unique characters:      {meta['vocab_size']}")
    print(f"  • Estimated BPE tokens:   ~{meta['estimated_bpe_tokens']:,}")
    print(f"  • Saved files to:         {os.path.abspath(args.output_dir)}/")
    print(f"      - input.txt (complete corpus)")
    print(f"      - train.txt (90% training split)")
    print(f"      - val.txt   (10% validation split)")
    print(f"      - meta.json (vocabulary metadata)")
    
    # Training time guidance for students
    est_sec_tiny = (2000 * 0.05) / 60  # ~2000 iters at ~50ms per iter
    est_sec_small = (5000 * 0.20) / 60 # ~5000 iters at ~200ms per iter
    print("\nEstimated Training Times on Standard Laptop CPU:")
    print(f"  • Tiny Config  (~1M params):  ~{est_sec_tiny:.1f} minutes")
    print(f"  • Small Config (~10M params): ~{est_sec_small:.1f} minutes")
    print("=" * 60)
    print("[✓] Data preparation complete. You are ready to run: python train.py\n")


if __name__ == "__main__":
    main()

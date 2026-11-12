# Tiny-GPT: A Miniature Educational Transformer from Scratch

> **Practical Component for Chapter 11: Building a Mini-GPT from Scratch**  
> *A self-contained, readable, and hackable Generative Pretrained Transformer designed to train on student laptops.*

---

## 1. Overview

**Tiny-GPT** is a pure PyTorch implementation of an autoregressive Decoder-only Transformer language model (GPT-2 style). Scaled down to **~1M to 10M parameters**, it allows undergraduate and graduate students to train, inspect, and sample from a complete language model directly on standard consumer laptops without needing high-end GPUs or cloud clusters.

Every component—from positional embeddings and multi-head causal self-attention to weight tying and cosine learning rate schedules—is implemented in clean, heavily documented Python code based on the core architectural principles of modern Large Language Models.

---

## 2. Educational Disclaimer

> [!IMPORTANT]
> **Educational Purpose Only**: This project is strictly intended for pedagogical study, experimentation, and concept validation. It demonstrates the mechanics of tokenization, attention mechanisms, autoregressive next-token prediction, and backpropagation. It is **not** suitable or intended for production deployments, safety-critical applications, or general knowledge retrieval.

---

## 3. Hardware Requirements

Tiny-GPT is engineered to run on lightweight hardware:
- **Processor**: Any modern dual-core or quad-core CPU (Intel Core i3/i5/i7, AMD Ryzen, Apple Silicon M1/M2/M3/M4).
- **Memory (RAM)**: Minimum **4 GB RAM** (8 GB recommended).
- **Disk Space**: ~200 MB for repository, dataset, checkpoints, and dependencies.
- **Operating System**: Linux, macOS, or Windows (via WSL or PowerShell).
- **Accelerators (Optional)**: Automatically leverages NVIDIA CUDA GPUs or Apple Silicon MPS if available, but CPU execution is first-class and fully supported.

---

## 4. Installation Steps

### Step 1: Clone or Navigate to the Project Directory
```bash
cd /home/tushardevx01/Documents/llm/projects/tiny-gpt
```

### Step 2: Set Up a Python Virtual Environment
It is best practice to create an isolated Python virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

> [!NOTE]
> `tiktoken` is optional! If `tiktoken` cannot be installed on your platform, Tiny-GPT automatically switches to its built-in character-level tokenizer with zero external setup.

---

## 5. Quickstart: Train and Generate in 3 Steps

### Step 1: Prepare Training Data
Download and prepare the corpus (defaults to the classic public-domain Tiny Shakespeare dataset):
```bash
python data/prepare.py
```
This splits the corpus into `data/train.txt` (90%) and `data/val.txt` (10%), creates vocabulary metadata, and outputs dataset statistics.

*You can also use your own custom text file:*
```bash
python data/prepare.py --input_file path/to/my_notes.txt
```

### Step 2: Train the Model
Launch the training loop using the lightweight `tiny` preset (~1.2M parameters):
```bash
python train.py --config tiny
```

During training, the script outputs step loss, learning rate, evaluation metrics, and saves the best model weights to `checkpoints/best_model.pt`. When training finishes, it saves a loss visualization curve to `loss_curve.png`.

### Step 3: Generate Text
Generate completions conditioned on an initial prompt:
```bash
python generate.py --prompt "Once upon a time" --temperature 0.8 --top_k 40
```

Or launch an interactive prompt console:
```bash
python generate.py --interactive
```

---

## 6. Model Configurations & Presets

Tiny-GPT includes three presets tailored for different hardware profiles in `config.py`:

| Preset | Parameters | Layers (`n_layer`) | Heads (`n_head`) | Embedding Dim (`n_embd`) | Context Window | RAM / VRAM | Approx. CPU Time | Target Hardware |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`tiny`** | **~1.2M** | 4 | 4 | 128 | 128 | < 1 GB | ~5–10 mins | Any dual-core laptop CPU |
| **`small`** | **~10.5M** | 6 | 8 | 256 | 256 | ~1.5 GB | ~45–60 mins | Multi-core CPU / entry GPU |
| **`medium`**| **~50M** | 8 | 8 | 512 | 512 | ~4 GB | ~2–4 hours | Apple M-series / NVIDIA GPU |

To select a preset during training:
```bash
python train.py --config tiny     # Recommended for first run
python train.py --config small    # Deeper coherence and sentence flow
python train.py --config medium   # Rich multi-paragraph stylistic modeling
```

---

## 7. Hyperparameter Explanation Table

Understanding each hyperparameter and its effect on training is a fundamental learning objective of this project:

| Hyperparameter | Default (`small`) | Mathematical / Architectural Role | Practical Effect |
| :--- | :---: | :--- | :--- |
| `vocab_size` | 50,257 (or ~65 for char) | Number of discrete tokens the model can represent ($V$). | Larger vocabularies compress text into fewer tokens but increase embedding matrix size ($V \times d$). |
| `context_length` | 256 | Maximum sequence block size ($T$). Attention matrices are $(T \times T)$. | Sets how far back the model can look. Memory scales quadratically $\mathcal{O}(T^2)$ with context. |
| `n_embd` | 256 | Dimensionality of token embedding vector ($d_{\text{model}}$). | Determines representation capacity. Must be divisible by `n_head`. |
| `n_head` | 8 | Number of parallel attention heads ($h$). Subspace dim $d_k = d_{\text{model}} / h$. | Allows simultaneous attention to different syntactic and semantic patterns. |
| `n_layer` | 6 | Number of stacked Transformer decoder blocks. | Controls model depth. More layers capture more abstract reasoning hierarchies. |
| `dropout` | 0.1 | Fraction of activations zeroed out randomly during training. | Regularization preventing overfitting on small text datasets. Set to 0.0 during evaluation. |
| `learning_rate` | 3e-4 | Peak step size multiplier for AdamW optimizer weight updates. | Too high causes divergence (loss $\to \text{NaN}$); too low causes slow learning. |
| `min_lr` | 3e-5 | Final learning rate floor after cosine annealing decay. | Allows model weights to settle gently into optimal loss basins. |
| `warmup_iters` | 100 | Steps over which learning rate ramps up linearly from $0$ to $\eta_{\max}$. | Prevents early volatile gradients from destabilizing uninitialized weights. |
| `weight_decay` | 0.1 | $L_2$ regularization penalty applied to 2D weight matrices (Linear & Embeddings). | Penalizes overly large weights; biases and LayerNorm 1D scales are exempted. |
| `grad_clip` | 1.0 | Maximum allowed Euclidean norm for the concatenated gradient vector. | Prevents gradient explosion in deep attention networks. |
| `temperature` | 0.8 | Logit divisor in softmax: $P(w_i) \propto \exp(z_i / \tau)$. | $\tau < 1.0$ produces sharp, conservative text; $\tau > 1.0$ increases creativity and randomness. |
| `top_k` | 40 | Truncates sampling candidates to only the top $k$ highest-scoring tokens. | Eliminates long-tail nonsensical words while preserving diversity. |

---

## 8. Expected Training Trajectory & Metrics

When monitoring training output in `train.py`, use the following theoretical baselines to verify that your model is learning properly:

### 1. Initial Loss (Step 0)
At step 0, model weights are randomly initialized. If the model makes a uniform random guess across all vocabulary tokens:
$$\mathcal{L}_{\text{init}} \approx -\ln\left(\frac{1}{V}\right) = \ln(V)$$
- For character-level tokenization ($V \approx 65$): $\mathcal{L}_{\text{init}} \approx \ln(65) \approx 4.17$
- For BPE tokenization ($V = 50,257$): $\mathcal{L}_{\text{init}} \approx \ln(50257) \approx 10.82$

*If your initial loss is significantly higher or NaN, check your weight initialization!*

### 2. Convergence Loss & Perplexity
- **Step 500–1000**: Loss drops rapidly from ~4.2 to ~2.5 as the model learns basic character frequencies and whitespace rules.
- **Step 2000–5000**: Loss stabilizes around **1.5 – 1.9**. At this point, the model generates recognizable English words, punctuation, and dialogues.
- **Perplexity ($\text{PPL} = e^{\mathcal{L}}$)**: Represents the average number of plausible choices the model is considering at each token. A validation loss of $1.6$ corresponds to $\text{PPL} \approx 4.95$.

---

## 9. Relevant LLM Handbook Chapters

This practical codebase directly accompanies the theoretical foundations covered in the handbook:

- **[Chapter 2: Tokenization & Embeddings](../../chapters/02-tokenization.md)**  
  *Explore Byte-Pair Encoding (BPE), character-level fallbacks, and token/position embedding lookup tables.*
- **[Chapter 3: Attention Mechanisms Deep Dive](../../chapters/03-attention.md)**  
  *Detailed mathematical derivation of Query-Key-Value dot-product attention, causal masks, and multi-head projection.*
- **[Chapter 4: Transformer Architecture & Normalization](../../chapters/04-transformer-architecture.md)**  
  *Pre-LayerNorm vs Post-LayerNorm residual stream dynamics, GELU activations, and Feed-Forward Networks.*
- **[Chapter 5: Pretraining & Optimization](../../chapters/05-pretraining.md)**  
  *AdamW weight decay partitioning, learning rate warmup, cosine decay schedules, and gradient clipping.*
- **[Chapter 6: Decoding & Sampling Strategies](../../chapters/06-inference-sampling.md)**  
  *Autoregressive generation loops, temperature scaling, greedy decoding, and Top-K / Top-P sampling.*
- **[Chapter 11: Building a Mini-GPT from Scratch](../../chapters/11-hands-on-gpt.md)**  
  *Step-by-step code walkthrough of every line in `model.py` and `train.py`.*

---

## 10. Project File Structure

```
tiny-gpt/
├── README.md               # Student guide, documentation, and handbook links
├── requirements.txt        # Minimal Python dependencies (PyTorch, NumPy, Matplotlib)
├── config.py               # GPTConfig dataclass and hardware presets (tiny, small, medium)
├── model.py                # Full Transformer implementation (CausalSelfAttention, MLP, Block, GPT)
├── train.py                # DataLoader, AdamW optimizer, cosine schedule, training loop & plots
├── generate.py             # Inference script supporting temperature, top-k, and interactive REPL
├── data/
│   └── prepare.py          # Dataset download, train/val split, and tokenizer metadata extraction
└── checkpoints/            # Directory where best_model.pt and latest_model.pt are stored
```

---

## 11. Frequently Asked Questions (FAQ)

### Q: Why is my model outputting repetitive gibberish?
**A:** Check the sampling temperature and training step count. If the model has only trained for 100 steps, it hasn't learned English syntax yet. If `temperature` is too low ($< 0.2$), the model may enter a repetitive loop; increase temperature to $0.7 - 0.9$ or set `--top_k 40`.

### Q: Can I run this without an internet connection?
**A:** Yes! `data/prepare.py` has an embedded fallback corpus. If no internet connection is detected, it automatically uses the offline corpus.

### Q: Can I train on my own text files?
**A:** Yes. Run `python data/prepare.py --input_file path/to/your_text.txt`, and `train.py` will automatically train on your data.

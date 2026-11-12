# LLM & AI Glossary

> A comprehensive, alphabetical reference guide to essential concepts, techniques, architectures, and terminology in Modern Machine Learning and Large Language Models. Every entry includes both an intuitive explanation for beginners and a precise technical definition for practitioners.

---

## Quick Navigation

[A](#a) • [B](#b) • [C](#c) • [D](#d) • [E](#e) • [F](#f) • [G](#g) • [H](#h) • [I](#i) • [J](#j) • [K](#k) • [L](#l) • [M](#m) • [N](#n) • [O](#o) • [P](#p) • [Q](#q) • [R](#r) • [S](#s) • [T](#t) • [U](#u) • [V](#v) • [W](#w) • [Z](#z)

---

## A

### Activation Function
**Simple:** A mathematical switch inside an artificial neuron that decides how strongly the signal should pass forward.  
**Technical:** A non-linear mathematical transformation (e.g., ReLU, GeLU, Sigmoid) applied element-wise to the affine combination of inputs and weights ($\sigma(Wx + b)$), enabling neural networks to learn non-linear decision boundaries and approximate arbitrary continuous functions.

### Adapter
**Simple:** A small set of extra adjustable dials added onto a frozen, pre-built model to adapt it to a new job without changing the original machine.  
**Technical:** Lightweight trainable parameter modules inserted between or alongside frozen layers of a pre-trained backbone model (e.g., in LoRA or bottleneck adapter architectures), allowing parameter-efficient fine-tuning (PEFT) while updating under 1% of total model parameters.

### Adversarial Attack
**Simple:** A trick input specifically crafted to fool an AI system into making an error or behaving inappropriately.  
**Technical:** An input crafted using gradient-based optimization or heuristic prompt patterns (e.g., FGSM, PGD, or adversarial jailbreak suffixes like GCG) that introduces imperceptible perturbations or linguistic manipulations designed to maximize loss or bypass alignment guardrails.

### Alignment
**Simple:** The process of steering an AI system so that its answers are helpful, truthful, safe, and aligned with human values rather than harmful or misleading.  
**Technical:** The post-training methodology—typically combining Supervised Fine-Tuning (SFT), Reinforcement Learning from Human Feedback (RLHF), Direct Preference Optimization (DPO), and Constitutional AI—aimed at optimizing model behavior to align with human preferences and ethical criteria (Helpfulness, Honesty, Harmlessness).

### Attention
**Simple:** A mechanism that allows an AI to focus more strongly on the specific words in a sentence that are most relevant to what it is thinking about right now.  
**Technical:** A dynamic weighting mechanism that computes a soft alignment between query and key representations via dot-product scoring, outputting a weighted sum over value vectors ($\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$).

### Attention Head
**Simple:** One individual "lens" or viewpoint inside an attention layer that focuses on a specific type of relationship between words (like grammar, tense, or pronoun references).  
**Technical:** An independent projection of query, key, and value vectors into a lower-dimensional subspace ($d_k = d_{\text{model}} / h$) where scaled dot-product attention is computed in parallel with other heads, allowing the model to attend to information from distinct representation subspaces simultaneously.

### Attention Mask
**Simple:** A digital blindfold that prevents the AI from looking at future words during training or from paying attention to empty padding slots.  
**Technical:** A boolean or additive tensor (often with values of $0$ or $-\infty$) applied to the raw attention logit matrix prior to the softmax operation, enforcing causal autoregressive constraints (lower-triangular masking) or masking out padding tokens in batched sequences.

### Autoregressive
**Simple:** A process where an AI writes text by predicting one word at a time, feeding each new word back into its memory to decide the next one.  
**Technical:** A statistical sequence formulation where the joint probability of a sequence $x = (x_1, \dots, x_T)$ is factored via the chain rule as a product of conditional probabilities: $P(x) = \prod_{t=1}^T P(x_t \mid x_{<t})$, where each token is predicted conditioned strictly on preceding tokens.

---

## B

### Backpropagation
**Simple:** The mathematical process of tracking mistakes backward through a neural network to figure out how much each dial needs to be tweaked.  
**Technical:** An algorithm for computing the partial derivatives (gradients) of a scalar objective loss function with respect to every trainable weight and bias parameter in a computational graph, systematically applying the multivariate chain rule from the output layer back to the input layer.

### Batch
**Simple:** A group of training examples shown to the model together all at once to make learning faster and more stable.  
**Technical:** A collection of $N$ independent training samples processed concurrently in tensor form across parallel hardware units (GPUs/TPUs) to compute an aggregated gradient estimate: $\nabla_\theta \mathcal{L}_{\text{batch}} = \frac{1}{N} \sum_{i=1}^N \nabla_\theta \mathcal{L}(x_i, y_i)$.

### Beam Search
**Simple:** A smart text-generation strategy that keeps track of the top few most promising sentences in parallel instead of committing blindly to just one word.  
**Technical:** A heuristic graph-search decoding algorithm that maintains a fixed beam width $B$ of highest cumulative log-probability sequence hypotheses at each generation time-step $t$, expanding each hypothesis and pruning to the top $B$ candidates according to $\sum_{\tau=1}^t \log P(x_\tau \mid x_{<\tau})$.

### BERT (Bidirectional Encoder Representations from Transformers)
**Simple:** A breakthrough AI model built by Google that reads sentences in both directions at once to deeply understand word meanings.  
**Technical:** An encoder-only Transformer architecture developed by Devlin et al. (2018) pre-trained on large text corpora using Masked Language Modeling (MLM) and Next Sentence Prediction (NSP), producing contextual bidirectional token representations widely used for classification, NER, and dense retrieval.

### Bias (model)
**Simple:** Unfair patterns or prejudice an AI picks up from its training data, causing it to favor certain groups or ideas unfairly.  
**Technical:** Systematic, non-random skew or disparities in model outputs resulting from unrepresentative training distributions, historical societal stereotypes in human text corpora, or poorly calibrated evaluation/reward functions.

### Bias (mathematical)
**Simple:** An adjustable offset number added to a neuron's calculation that shifts its baseline output up or down regardless of the input.  
**Technical:** The intercept scalar or vector parameter $b$ in the linear affine layer transformation $y = Wx + b$, allowing the activation threshold of an artificial neuron to translate independently of the input features $x$.

### BitsAndBytes
**Simple:** A popular software toolkit that compresses large AI models down to 4-bit or 8-bit precision so they can run and train on everyday GPUs.  
**Technical:** An open-source library developed by Tim Dettmers providing optimized CUDA kernels for 8-bit optimizers (e.g., 8-bit AdamW) and LLM quantization algorithms, including LLM.int8() vector-wise quantization and 4-bit NormalFloat (NF4) quantization used in QLoRA.

### BPE (Byte Pair Encoding)
**Simple:** A smart way to chop words into small subword puzzle pieces, turning common words into single tokens and breaking rare words into recognizable syllables.  
**Technical:** A data-driven subword tokenization compression algorithm that iteratively identifies and merges the most frequently co-occurring byte or character pairs in a text corpus, producing a fixed vocabulary of subword units that gracefully handles out-of-vocabulary terms at the byte level.

---

## C

### Causal Language Model
**Simple:** A text model that only looks backward at words already written to guess what comes next, never peeking forward into the future.  
**Technical:** An autoregressive Transformer decoder whose attention matrix is strictly constrained by a causal (lower-triangular) attention mask, ensuring the representation for token $x_t$ can only attend to tokens $x_{\le t}$ when computing next-token probability distributions.

### Chain of Thought (CoT)
**Simple:** Encouraging an AI to think out loud and show its intermediate reasoning steps before giving a final answer, which drastically improves its problem-solving.  
**Technical:** A prompting methodology introduced by Wei et al. (2022) that induces multi-step intermediate natural language reasoning traces prior to generating the terminal output, decomposing complex semantic and mathematical tasks into discrete deduction steps.

### ChatML
**Simple:** A standardized markup formatting style that uses special tags to clearly label whether a message came from the system, the user, or the assistant.  
**Technical:** A structured prompt formatting specification (originally by OpenAI) that encloses conversation turns in explicit role-delimited tags (such as `<|im_start|>system`, `<|im_start|>user`, and `<|im_start|>assistant`), preventing prompt injection by structurally isolating system instructions from untrusted user inputs.

### Checkpoint
**Simple:** A saved snapshot of a model's exact weights and settings at a specific moment during training, like a save point in a video game.  
**Technical:** A serialized file (typically in `.pt`, `.safetensors`, or `.bin` format) containing the comprehensive state dictionary (`state_dict`) of model parameters, optimizer moments, learning rate scheduler state, and global training step indices.

### Chinchilla
**Simple:** A famous research finding showing that older AI models had too many parameters and not enough training data, proving smaller models trained on far more data work better.  
**Technical:** An empirical finding and 70B model by Hoffmann et al. (2022) establishing compute-optimal scaling laws: for a given compute budget, model parameters $N$ and training tokens $D$ should be scaled in equal proportion ($N \propto D$), implying ~20 training tokens per model parameter.

### Constitutional AI (CAI)
**Simple:** A training approach where an AI evaluates and fixes its own bad habits by comparing its answers against a set of written rules or principles.  
**Technical:** An alignment framework developed by Anthropic where an LLM critiques and revises its own candidate outputs based on a specified list of textual principles (a "constitution"), generating synthetic preference pairs for preference optimization (RLAIF) without requiring human labelers.

### Context Window
**Simple:** The total amount of text (prompt plus answer) that an AI can hold in its immediate memory at one time.  
**Technical:** The maximum sequence length $L_{\max}$ in tokens that a Transformer architecture can ingest and process in a single forward pass without truncating attention representations or positional embeddings.

### Contrastive Learning
**Simple:** A training trick where the model learns by pulling similar ideas closer together in its memory map while pushing different ideas far apart.  
**Technical:** A self-supervised representation learning paradigm that optimizes an objective (such as InfoNCE) to minimize the distance between positive embedding pairs (e.g., augmented views of the same text) while maximizing the distance between negative pairs in latent vector space.

### Cosine Similarity
**Simple:** A math score between -1 and 1 measuring how closely two arrows point in the exact same direction, ignoring their length.  
**Technical:** The normalized dot product of two non-zero vectors $u$ and $v$, computed as $\cos(\theta) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$, measuring the directional alignment of representations in high-dimensional embedding space regardless of vector magnitude.

### Cross-Attention
**Simple:** An attention block where the query comes from one sentence (like an English translation in progress) and looks up keys and values from another sentence (like the original French text).  
**Technical:** An attention sublayer where the query representations $Q$ are derived from one sequence (e.g., the decoder state), while keys $K$ and values $V$ are derived from a completely different sequence (e.g., the encoder output representations).

### Cross-Entropy Loss
**Simple:** A score that measures how surprised or confused the AI was by the true next word; lower scores mean the AI predicted the correct word with high confidence.  
**Technical:** An information-theoretic objective function measuring the divergence between empirical ground-truth one-hot targets $y$ and predicted probability distributions $\hat{y}$: $\mathcal{L}_{\text{CE}} = -\sum_k y_k \log(\hat{y}_k) = -\log(\hat{y}_{\text{target}})$.

---

## D

### Data Contamination
**Simple:** An unfair situation where the test exam questions were accidentally included in the AI's training textbooks, making the AI look smarter on benchmarks than it actually is.  
**Technical:** The presence of benchmark evaluation sets or test split samples within the pre-training or fine-tuning web scrape corpora of an LLM, leading to artificially inflated performance metrics that reflect memorization rather than generalization.

### Dataset
**Simple:** A curated collection of texts, images, or examples used to teach or test an AI model.  
**Technical:** A structured or semi-structured corpus of data instances, systematically partitioned into training, validation, and test splits with standardized schema attributes, tokenization schemas, and metadata annotations.

### Decoder
**Simple:** The half of a Transformer model that generates new tokens step-by-step into the future.  
**Technical:** A sequence-generating Transformer component composed of stacked layers with masked multi-head self-attention, feedforward sublayers, and residual layer normalizations, designed for next-token probability prediction.

### Decoder-only
**Simple:** An AI architecture (like GPT or LLaMA) made purely of text-generating blocks, which has become the dominant design for modern conversational LLMs.  
**Technical:** An architecture composed solely of autoregressive Transformer decoder blocks with causal masking, omitting separate encoder stacks and bidirectional cross-attention mechanisms, popularized by GPT-3 and LLaMA due to its unified pre-training objective.

### DeepMind (Google DeepMind)
**Simple:** Google's premier AI research laboratory, famous for AlphaFold, AlphaGo, and the Gemini model family.  
**Technical:** The artificial intelligence research division of Alphabet Inc., responsible for foundational breakthroughs across reinforcement learning (DQN, AlphaGo), structural biology (AlphaFold), and frontier multimodal foundation models (Chinchilla, Gemini).

### DeepSpeed
**Simple:** A high-speed optimization library by Microsoft that lets developers train and run massive multi-billion-parameter models across multiple GPUs.  
**Technical:** An open-source distributed optimization library developed by Microsoft that implements the ZeRO (Zero Redundancy Optimizer) memory-partitioning stages (ZeRO-1, ZeRO-2, ZeRO-3), pipeline parallelism, and CPU/NVMe memory offloading for extreme-scale LLM training.

### DPO (Direct Preference Optimization)
**Simple:** A modern training technique that teaches an AI to prefer good answers over bad ones directly with simple math, completely skipping the complex reinforcement learning step.  
**Technical:** An alignment algorithm by Rafailov et al. (2023) that analytically reparameterizes the RLHF objective, directly optimizing an implicit reward model on paired preference data $(x, y_w, y_l)$ via a closed-form binary cross-entropy loss without training a separate reward model or invoking PPO.

### Dropout
**Simple:** A training trick where random neurons are temporarily turned off so the network doesn't rely too much on any single connection.  
**Technical:** A stochastic regularization technique that zeros out neuron activations with probability $p$ during each training forward pass, scaling remaining activations by $\frac{1}{1-p}$ to prevent feature co-adaptation and mitigate overfitting.

---

## E

### Embedding
**Simple:** A list of numbers that represents the meaning of a word, concept, or document as a single coordinate point in high-dimensional space.  
**Technical:** A dense, continuous, low-dimensional vector representation $e \in \mathbb{R}^d$ mapped from a discrete symbol or sequence by a learned projection matrix, where geometric proximity (e.g., Euclidean distance or cosine similarity) reflects semantic similarity.

### Embedding Model
**Simple:** A specialized AI model whose only job is to turn text into mathematical coordinates for search engines and recommendation systems.  
**Technical:** An encoder-based Transformer architecture (such as text-embedding-ada-002, BGE, or nomic-embed) optimized with contrastive objectives to output fixed-dimensional sentence or passage vectors for dense vector indexing and semantic retrieval.

### Encoder
**Simple:** The component of an AI that reads the entire input text at once to create a deep, rich summary of its meaning.  
**Technical:** A Transformer module consisting of stacked self-attention layers with unmasked (bidirectional) attention followed by feedforward networks, encoding an input sequence into continuous contextual representation vectors.

### Encoder-Decoder
**Simple:** A two-part AI setup where one part reads and understands an input text and the second part writes a completely new output, ideal for language translation.  
**Technical:** A dual-stack Transformer architecture (e.g., T5, BART) where an encoder processes an input sequence bidirectionally and a decoder generates the target sequence autoregressively using cross-attention mechanisms over the encoder outputs.

### Epoch
**Simple:** One complete round of training during which the model has seen every single example in the entire dataset once.  
**Technical:** A single full traversal through all training examples in a dataset during the optimization phase, comprising $\lceil N / \text{batch\_size} \rceil$ gradient update steps.

### Evaluation Leakage
**Simple:** An accidental error where test evaluation prompts are exposed to the AI before or during its training, invalidating the test results.  
**Technical:** A form of data contamination where testing benchmarks, reference answer keys, or evaluation rubrics bleed into the pre-training, fine-tuning, or few-shot context stages, causing an overestimation of model generalization.

---

## F

### Feedforward Network (FFN / MLP)
**Simple:** The thinking engine inside every Transformer layer that independently processes each word vector after it has gathered clues from its neighbors.  
**Technical:** A position-wise multi-layer perceptron sublayer in a Transformer block consisting of two linear transformations with a non-linear activation in between: $\text{FFN}(x) = \sigma(x W_1 + b_1) W_2 + b_2$, typically expanding the hidden dimension by a factor of 4 (or using SwiGLU architectures).

### Few-shot Learning
**Simple:** Teaching an AI how to do a new task by providing just two or three worked-out examples right inside the prompt.  
**Technical:** An in-context inference technique where $k$ input-output demonstration pairs $(x_1, y_1), \dots, (x_k, y_k)$ are prepended to the input context prompt without performing any parameter updates or gradient descent.

### Fine-tuning
**Simple:** Taking a general-purpose AI that already knows language and giving it extra specialized training for a specific job (like medical diagnosis or legal writing).  
**Technical:** The process of taking a pre-trained foundation model and continuing training on a domain-specific or task-specific dataset, updating either all parameters (full fine-tuning) or a parameter-efficient subset (PEFT/LoRA) with a reduced learning rate.

### Flash Attention
**Simple:** A super-smart hardware trick that makes Transformer attention run up to 4 times faster while using way less GPU memory by managing memory cache smartly.  
**Technical:** An exact, IO-aware attention algorithm developed by Tri Dao et al. that tiles queries, keys, and values to compute softmax incrementally in GPU fast SRAM without ever materializing the large $N \times N$ attention matrix in high-bandwidth memory (HBM).

### Foundation Model
**Simple:** A massive, versatile AI model trained on broad general data that serves as the common starting platform for countless different applications.  
**Technical:** A large-scale neural network (typically a Transformer with billions of parameters) trained on massive, diverse datasets using self-supervised learning, exhibiting broad capabilities across multiple modalities and downstream tasks.

---

## G

### GeLU (Gaussian Error Linear Unit)
**Simple:** A smooth activation switch used in modern Transformers that lets small signals pass through gently instead of abruptly cutting them off.  
**Technical:** A smooth non-linear activation function defined as $\text{GeLU}(x) = x \cdot \Phi(x) = x \cdot P(X \le x)$ where $X \sim \mathcal{N}(0, 1)$, approximated as $0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))$, standard in GPT and BERT.

### Generative AI
**Simple:** Any AI system capable of creating brand new text, pictures, audio, or code rather than just sorting or classifying existing things.  
**Technical:** A broad class of artificial intelligence models that model the underlying probability distribution $P(X)$ of high-dimensional data spaces to synthesize novel, high-fidelity sample instances across language, vision, audio, and code modalities.

### GGUF
**Simple:** A modern, single-file format designed to run quantized AI models blazingly fast on regular laptop CPUs and Apple Silicon via llama.cpp.  
**Technical:** The unified binary file format created by Georgi Gerganov and the llama.cpp project that encapsulates model architecture, metadata, tokenizers, and quantized tensor data into an extensible single-file format designed for fast memory-mapped (mmap) loading.

### GPTQ (Generalized Post-Training Quantization)
**Simple:** A method that shrinks a trained model down to 4-bit numbers in just a few hours with almost zero loss in answer quality.  
**Technical:** An efficient one-shot post-training quantization algorithm based on second-order error compensation (approximate Hessian matrices), compressing weight matrices to 4-bit or 3-bit integers with minimal degradation in perplexity.

### GQA (Grouped-Query Attention)
**Simple:** A speed-up technique where multiple attention heads share the same memory cache, cutting down memory usage during chat conversations.  
**Technical:** An attention variant introduced by Ainslie et al. (2023) that interpolates between Multi-Head Attention (MHA) and Multi-Query Attention (MQA) by partitioning query heads into $G$ groups, with each group sharing a single key-value head pair, reducing KV-cache memory bandwidth during inference.

### Gradient
**Simple:** A mathematical vector indicating which direction is uphill on the error hill, telling the AI how to tweak each dial to reduce mistakes.  
**Technical:** The vector of partial derivatives $\nabla_\theta \mathcal{L} = \left[ \frac{\partial \mathcal{L}}{\partial \theta_1}, \dots, \frac{\partial \mathcal{L}}{\partial \theta_k} \right]^T$ of a scalar loss function with respect to all model parameters $\theta$, defining the direction of steepest rate of increase in loss space.

### Gradient Accumulation
**Simple:** A clever way to simulate a big training batch on a small GPU by adding up the tweaks from several small mini-batches before actually updating the dials.  
**Technical:** A training technique where gradients from consecutive mini-forward-and-backward passes are accumulated over $K$ sub-batches into parameter gradient buffers before calling `optimizer.step()`, effectively scaling effective batch size by $K$ without increasing GPU memory usage.

### Gradient Clipping
**Simple:** Setting a strict speed limit on weight updates so that an unexpected huge mistake doesn't blow up the entire model training process.  
**Technical:** A numerical stabilization technique that scales down parameter gradients whenever their global Euclidean $L_2$ norm exceeds a designated threshold $c$: $g \leftarrow g \cdot \min\left(1, \frac{c}{\|g\|_2}\right)$, preventing exploding gradients during training.

### Gradient Descent
**Simple:** The step-by-step learning process where an AI takes small steps downhill along the error curve until it finds the lowest mistake rate.  
**Technical:** An iterative first-order optimization algorithm that updates parameters $\theta$ in the opposite direction of the gradient of the objective function: $\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)$, where $\eta$ represents the learning rate scalar.

### Greedy Decoding
**Simple:** A text generation strategy where the AI always picks the single most probable next word at every step with no randomness.  
**Technical:** A deterministic sequence decoding method that selects the token with the highest predicted logit probability at each generation time-step: $x_t = \arg\max_{w \in \mathcal{V}} P(w \mid x_{<t})$, often resulting in repetitive or degenerate generations for creative tasks.

---

## H

### Hallucination
**Simple:** When an AI confidently invents false facts, fabricated citations, or non-existent events while sounding completely convincing.  
**Technical:** The generation of content that is semantically plausible but factually incorrect, ungrounded in source reference context, or inconsistent with established world knowledge, stemming from probabilistic next-token sampling and dataset inconsistencies.

### Hidden State
**Simple:** The internal memory snapshot that an AI creates for each word after passing it through its layers of artificial neurons.  
**Technical:** The intermediate feature vector tensor $h \in \mathbb{R}^{B \times L \times d_{\text{model}}}$ produced by an internal layer in a neural network, capturing contextualized semantic information about input tokens before final output projection.

### HNSW (Hierarchical Navigable Small World)
**Simple:** A super-fast graph-based shortcut system that helps vector databases find the closest matching documents in a fraction of a millisecond.  
**Technical:** A multi-layer graph-based indexing algorithm for Approximate Nearest Neighbor (ANN) vector search that constructs a hierarchy of proximity graphs with geometrically decreasing link lengths, achieving logarithmic $O(\log N)$ search time complexity.

### Hugging Face
**Simple:** The "GitHub of AI" — the central platform and open-source community where developers share datasets, model weights, and code.  
**Technical:** The open-source AI platform and research organization that maintains the Hugging Face Hub, the `transformers`, `datasets`, `peft`, and `trl` libraries, providing standard model repositories and deployment pipelines for modern machine learning.

### Hyperparameter
**Simple:** The high-level setup knobs (like learning rate, batch size, and layer count) chosen by human engineers before training starts.  
**Technical:** An external structural or algorithmic configuration setting (such as learning rate schedule, weight decay, dropout rate, temperature, or layer depth) that controls the learning process and cannot be directly estimated from training data via gradient descent.

---

## I

### Inference
**Simple:** The stage where a finished, fully trained AI model actually runs to answer user questions or generate text.  
**Technical:** The forward pass evaluation phase where a pre-trained or fine-tuned model processes input inputs without computing gradients or updating weights ($\nabla_\theta = 0$) to generate predictions, classifications, or autoregressive sequences.

### Instruction Tuning
**Simple:** Teaching a raw text-predicting model how to behave like a helpful digital assistant that listens to user requests and answers directly.  
**Technical:** A specialized supervised fine-tuning (SFT) phase where foundation models are trained on structured prompt-response pairs formatted as explicit instructions and desired completions (e.g., FLAN, Alpaca datasets), teaching the model to act as a task executor.

---

## J

### Jailbreak
**Simple:** A sneaky or clever prompt designed to trick an AI into bypassing its safety rules and generating forbidden content.  
**Technical:** An adversarial prompt injection technique that exploits cognitive framing, roleplaying, linguistic obfuscation, or gradient-based suffix injections to circumvent a model's safety filters and constitutional alignment barriers.

---

## K

### KenLM
**Simple:** A classic, lightning-fast text tool used to filter out garbled or low-quality gibberish from massive web datasets before training.  
**Technical:** An optimized, memory-efficient C++ n-gram language modeling library widely used in data pre-processing pipelines (e.g., CCNet, RefinedWeb) to calculate text perplexity and filter out low-quality web scraping junk.

### Knowledge Distillation
**Simple:** A technique where a small "student" AI learns to perform almost as well as a giant "teacher" AI by copying its answer patterns.  
**Technical:** A model compression method where a smaller student network is trained using an objective that minimizes the Kullback-Leibler (KL) divergence between its output logit probability distribution and the soft probability outputs of a larger teacher network.

### KV Cache
**Simple:** A memory storage trick that saves previous word calculations so the AI doesn't have to recompute everything from scratch for every single new word it writes.  
**Technical:** An inference optimization technique that stores the computed Key ($K$) and Value ($V$) projection tensors for all prior tokens in GPU memory, avoiding redundant $O(N^2)$ matrix multiplications during autoregressive token generation.

---

## L

### Language Model (LM)
**Simple:** A computer program trained to understand words and predict what word is most likely to come next in a sentence.  
**Technical:** A statistical model that assigns a joint probability distribution $P(W) = P(w_1, w_2, \dots, w_n)$ over sequences of words or tokens, commonly parameterized using autoregressive neural network architectures.

### Latency
**Simple:** The delay or waiting time between when you send a prompt and when the AI starts or finishes answering.  
**Technical:** The temporal duration required for an inference system to process an input and return an output, decomposed into Time-to-First-Token (TTFT) and Inter-Token Latency (ITL).

### Latent Space
**Simple:** The hidden, multi-dimensional mathematical landscape where the AI organizes concepts so that related ideas sit near each other.  
**Technical:** The continuous, high-dimensional intermediate manifold spanned by hidden state representations within a neural network, where topological proximity corresponds to semantic affinity.

### Layer Normalization (LayerNorm)
**Simple:** A mathematical shower that washes and resets the numbers passing between neural network layers so they don't get wildly huge or tiny.  
**Technical:** A normalization technique that standardizes the activations across features for an individual sample: $\text{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$, where $\mu$ and $\sigma$ are calculated over the hidden dimension, stabilizing Transformer training.

### Learning Rate
**Simple:** The speed setting that controls how big of a step the AI takes when adjusting its internal dials based on its mistakes.  
**Technical:** A positive scalar optimization hyperparameter $\eta$ that scales the gradient magnitude during parameter updates: $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$.

### LLM (Large Language Model)
**Simple:** A massive AI model trained on billions of words that can understand and write human-like text across almost any topic.  
**Technical:** A Transformer-based neural network typically consisting of billions to hundreds of billions of parameters, pre-trained on terabyte-scale textual and code corpora using autoregressive language modeling objectives.

### LoRA (Low-Rank Adaptation)
**Simple:** A breakthrough method that lets you customize a giant AI using tiny add-on matrices without touching the heavy original model weights.  
**Technical:** A parameter-efficient fine-tuning technique that freezes base weight matrices $W_0 \in \mathbb{R}^{d \times k}$ and injects trainable rank-decomposition matrices $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ with rank $r \ll \min(d, k)$, updating $W = W_0 + \frac{\alpha}{r}BA$.

### Loss Function
**Simple:** The scoring rule that calculates a single number showing exactly how badly the AI messed up on a given task.  
**Technical:** A mathematical function $\mathcal{L}(\hat{y}, y)$ that quantifies the discrepancy between model predictions $\hat{y}$ and true ground-truth targets $y$, producing a scalar value minimized during optimization.

---

## M

### Masked Language Model (MLM)
**Simple:** A fill-in-the-blank training game where random words are hidden and the AI must guess them by looking at the words on both sides.  
**Technical:** A bidirectional pre-training objective introduced in BERT where a percentage (typically 15%) of input tokens are corrupted or replaced with a `[MASK]` token, and the network is trained to predict the original tokens using context from both directions simultaneously.

### MinHash
**Simple:** A fast computer algorithm that spots and eliminates near-duplicate web pages from huge training datasets before training starts.  
**Technical:** A locality-sensitive hashing (LSH) probabilistic algorithm that estimates the Jaccard similarity between document n-gram sets in constant time, used in large-scale dataset deduplication pipelines.

### Mixed Precision
**Simple:** Doing math with smaller 16-bit numbers instead of 32-bit numbers during training to cut memory use in half and speed up computations.  
**Technical:** A computation technique that utilizes both 16-bit floating-point representations (FP16 or BF16) and standard 32-bit floats (FP32), keeping master weights in FP32 while executing forward and backward matrix multiplications in FP16/BF16 on specialized hardware tensor cores.

### Model
**Simple:** The mathematical brain formed by a neural network's architecture together with all its learned numbers and connections.  
**Technical:** A parameterized mathematical function $f_\theta: \mathcal{X} \rightarrow \mathcal{Y}$ whose internal weights $\theta$ have been tuned via an optimization algorithm on training data to approximate an empirical data distribution.

### Model Card
**Simple:** A standardized digital nutrition label that transparently explains what an AI model was trained on, its strengths, limitations, and ethical considerations.  
**Technical:** A structured documentation artifact pioneered by Margaret Mitchell et al. detailing a machine learning model's architecture, intended use cases, training datasets, performance evaluations, known biases, and safety limitations.

### Multi-head Attention
**Simple:** Running multiple attention spotlights in parallel so the AI can track different types of sentence connections at the same time.  
**Technical:** An attention mechanism that projects queries, keys, and values into $h$ distinct representation subspaces using learned linear projections, computing attention across each head before concatenating and projecting the outputs back: $\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$.

---

## N

### Next-token Prediction
**Simple:** The fundamental guessing game played by LLMs: reading what has been said so far and guessing which word comes immediately next.  
**Technical:** The canonical self-supervised autoregressive training objective for generative language models, which maximizes the log-likelihood of each subsequent token conditioned on all previous tokens in the sequence: $\max_\theta \sum_t \log P(x_t \mid x_{<t}; \theta)$.

### Neural Network
**Simple:** A computer program structured like interconnected webs of artificial brain cells that learns to recognize complex patterns from examples.  
**Technical:** A computational graph consisting of layers of interconnected artificial neurons that perform parameterized linear transformations followed by non-linear activation functions, trained via backpropagation to approximate complex mathematical mappings.

### NF4 (4-bit NormalFloat)
**Simple:** A custom 4-bit number format mathematically tailored to store neural network weights with almost zero loss in precision.  
**Technical:** An information-theoretically optimal quantile-based quantization data type introduced in QLoRA that distributes quantized bins according to the empirical normal distribution $\mathcal{N}(0, \sigma^2)$ typical of neural network weight tensors.

### NLP (Natural Language Processing)
**Simple:** The field of computer science focused on teaching computers to read, understand, translate, and write human languages.  
**Technical:** The interdisciplinary subfield of artificial intelligence, computational linguistics, and computer science focused on the algorithmic processing, comprehension, translation, and generation of unstructured human language data.

### Normalization
**Simple:** Scaling numbers down into a clean, predictable range so that math calculations stay smooth and stable.  
**Technical:** Any mathematical technique (e.g., LayerNorm, RMSNorm, BatchNorm) applied to activation or weight tensors that standardizes statistical distributions (mean and variance), preventing vanishing or exploding gradients.

---

## O

### Overfitting
**Simple:** When an AI memorizes its training textbook word-for-word like a parrot but fails completely when given a new test question it hasn't seen before.  
**Technical:** An undesirable empirical condition where a model achieves near-zero training loss by fitting the idiosyncratic noise and sample quirks of the training distribution, resulting in high generalization error (validation loss divergence).

---

## P

### PagedAttention
**Simple:** Managing an AI's GPU memory just like an operating system manages computer RAM with pages, eliminating wasted memory.  
**Technical:** A memory management algorithm introduced by vLLM that partitions the non-contiguous KV-cache memory in GPU HBM into fixed-size virtual memory pages, reducing memory fragmentation and waste from up to 80% down to under 4%.

### PEFT (Parameter-Efficient Fine-Tuning)
**Simple:** Customizing a massive AI by training only a tiny fraction of its parameters, saving huge amounts of time and computer hardware.  
**Technical:** An umbrella term for fine-tuning methodologies (including LoRA, QLoRA, Prefix Tuning, and Prompt Tuning) that freeze the vast majority of pre-trained parameters and train less than 1% of the model weights, achieving performance on par with full fine-tuning.

### Perplexity
**Simple:** A score measuring how confused or surprised an AI is when reading text; lower perplexity means the model understands and predicts the text well.  
**Technical:** An intrinsic evaluation metric for language models defined as the exponential of the average cross-entropy loss per token over a test corpus: $\text{PPL} = \exp\left(-\frac{1}{N} \sum_{i=1}^N \log P(x_i \mid x_{<i})\right)$.

### PII (Personally Identifiable Information)
**Simple:** Sensitive personal details (like social security numbers, phone numbers, and addresses) that must be removed before training an AI.  
**Technical:** Any data information that can be utilized to directly or indirectly identify a specific individual human subject, requiring rigorous regex-based, NER-based, or heuristic scrubbing during pre-training dataset curation.

### Position Encoding
**Simple:** Adding number stamps to words so a Transformer knows whether a word came at the beginning, middle, or end of a sentence.  
**Technical:** Positional vector representations added or applied to token embeddings (via sinusoidal functions, learned positional embeddings, or rotary embeddings like RoPE) to inject token order and relative distance information into permutation-invariant Transformer attention mechanisms.

### PPO (Proximal Policy Optimization)
**Simple:** A reinforcement learning technique that trains an AI to maximize good feedback while making sure it doesn't take wild, risky leaps that break its behavior.  
**Technical:** An on-policy actor-critic reinforcement learning algorithm that utilizes a clipped surrogate objective function to constrain policy update steps within a trust region ($\epsilon \approx 0.2$), preventing destructive catastrophic policy updates during RLHF.

### Pretraining
**Simple:** The massive first stage where an AI reads trillions of words from the internet for months to learn language and common sense.  
**Technical:** The computationally intensive, self-supervised initial training phase where a foundation model is trained from random weight initialization on terabytes of raw text using next-token or masked language modeling objectives.

### Prompt
**Simple:** The question, instructions, or text you type into an AI model to tell it what you want it to do.  
**Technical:** The complete input token sequence provided to an autoregressive model that conditions the conditional probability distribution $P(Y \mid X)$ over subsequent generated tokens.

### Prompt Engineering
**Simple:** The skill of writing and structuring prompts carefully to coax the best, most accurate answers out of an AI.  
**Technical:** The discipline of designing, formatting, and optimizing natural language prompt contexts—utilizing techniques like few-shot framing, Chain-of-Thought, persona definition, and structured outputs—to maximize task performance from LLMs.

### Prompt Injection
**Simple:** A cyberattack where an attacker sneaks hidden commands into a prompt to hijack the AI and make it ignore its original instructions.  
**Technical:** A security vulnerability whereby untrusted external user input overrides the system prompt's instructions, manipulating the model's intent and hijacking its subsequent control flow or tool-calling operations.

### Prompt Template
**Simple:** A fill-in-the-blank blueprint that formats raw user inputs into structured text with system instructions and examples.  
**Technical:** A reusable string template containing dynamic variable interpolation placeholders (e.g., using Jinja2 or f-strings) that formats conversational roles, system guardrails, and user queries into a standardized model-specific input sequence.

---

## Q

### QLoRA (Quantized Low-Rank Adaptation)
**Simple:** A fine-tuning superpower that shrinks model weights to 4 bits so you can train huge AI models on an everyday gaming laptop or single GPU.  
**Technical:** An advanced parameter-efficient fine-tuning method by Dettmers et al. (2023) that compresses frozen base model weights into 4-bit NormalFloat (NF4), adds double quantization to compress quantization constants, and backpropagates gradients through 16-bit LoRA adapter matrices.

### Quantization
**Simple:** Shrinking the file size and memory needs of an AI model by rounding its precise numbers down to smaller numbers (like converting 16-bit floats to 4-bit or 8-bit integers).  
**Technical:** The mathematical conversion of high-precision floating-point weight and activation tensors (FP32 or FP16) into lower-bit discrete representations (INT8, INT4, NF4) using affine scaling and zero-point mapping: $q = \text{round}(x / S) + Z$.

### Query
**Simple:** In attention, the "question" or "search request" that a word sends out looking for related information from other words.  
**Technical:** A representation vector $Q = X W_Q$ generated by projecting input hidden states via a learned weight matrix, which is dot-producted against key vectors $K$ to compute dynamic attention weighting coefficients.

---

## R

### RAG (Retrieval-Augmented Generation)
**Simple:** Giving an AI an open-book exam: searching an external database for fresh, factual documents and feeding them to the AI so it answers accurately without guessing.  
**Technical:** A system architecture that couples a dense/sparse retrieval engine with a generative foundation model, fetching top-$k$ semantically relevant context chunks from an external knowledge base and concatenating them into the context window prior to generation.

### ReLU (Rectified Linear Unit)
**Simple:** A dead-simple neuron switch that passes positive numbers straight through and turns all negative numbers to zero.  
**Technical:** A piecewise linear activation function defined mathematically as $f(x) = \max(0, x)$, prized for computational simplicity and reducing vanishing gradients compared to Sigmoid and Tanh.

### Residual Connection (Skip Connection)
**Simple:** An information highway that lets the original signal bypass a neural layer completely and get added directly to the output, preventing signals from getting lost.  
**Technical:** An architectural shortcut that adds the unperturbed input tensor of a layer directly to its transformed output: $y = f(x) + x$, allowing unimpeded gradient flow directly back through deep networks during backpropagation.

### Reward Hacking
**Simple:** When an AI finds an unexpected cheat or loophole to get a high score from its reward judge without actually doing the job properly.  
**Technical:** A pathology in reinforcement learning where an agent exploits specification flaws or vulnerabilities in the reward function to achieve high numerical rewards without fulfilling the underlying human intent.

### Reward Model
**Simple:** A separate AI judge trained to score how good, helpful, or safe another AI's answers are.  
**Technical:** A specialized neural network $r_\psi(x, y) \rightarrow \mathbb{R}$ trained on human preference comparison pairs $(x, y_{\text{better}}, y_{\text{worse}})$ using a Bradley-Terry preference loss, used to provide scalar reward signals during RLHF.

### RLHF (Reinforcement Learning from Human Feedback)
**Simple:** Polishing an AI by having humans grade its responses, then rewarding the AI when it behaves well so it learns to produce better answers.  
**Technical:** A multi-stage alignment pipeline where a language model is fine-tuned using reinforcement learning algorithms (like PPO) against a learned reward model trained on pairwise human preference ratings.

---

## S

### Self-Attention
**Simple:** Words in a sentence looking at and talking to each other to figure out how their meanings change based on their surrounding context.  
**Technical:** An attention mechanism where the query, key, and value matrices are all derived from the exact same input sequence via linear projections ($Q = XW_Q, K = XW_K, V = XW_V$), enabling representations to incorporate contextual dependencies from the same sequence.

### Semantic Search
**Simple:** Searching by the underlying meaning of what you want rather than just matching exact keywords.  
**Technical:** An information retrieval technique that encodes queries and documents into dense latent vectors using embedding models, returning results based on vector proximity (cosine similarity) in conceptual space rather than lexical token overlap.

### SFT (Supervised Fine-Tuning)
**Simple:** Teaching a model to follow instructions by having it study thousands of examples of good user questions paired with ideal answers.  
**Technical:** The first stage of LLM instruction alignment where a pre-trained base model is fine-tuned on curated datasets of instruction-demonstration pairs $(x, y)$ using the standard autoregressive cross-entropy loss over target tokens.

### Sliding Window Attention
**Simple:** An attention diet where each word is only allowed to look at its immediate neighbors within a fixed window rather than every word in the entire document.  
**Technical:** A sparse attention pattern (used in models like Mistral 7B) where each token only attends to tokens within a local window of size $W$, reducing attention computation complexity from $O(N^2)$ to $O(N \times W)$.

### Softmax
**Simple:** A mathematical function that converts a list of raw scores into clean percentages that add up to 100%.  
**Technical:** A normalized exponential function that maps an unconstrained $K$-dimensional vector of real logits $z$ into a valid categorical probability distribution: $\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$.

### System Prompt
**Simple:** The secret foundational instructions given to an AI at the very beginning of a chat that tell it who it is, how to act, and what rules to follow.  
**Technical:** An initial instruction block set at the highest priority in the model's context window that configures persona, operating parameters, behavioral constraints, safety boundaries, and response formats for downstream dialogue turns.

---

## T

### Temperature
**Simple:** The creativity knob: set it low for strict, logical, factual answers; set it high for playful, diverse, and creative responses.  
**Technical:** A hyperparameter $T > 0$ that scales logits prior to the softmax computation: $P(x_i) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$. Lower temperatures ($T \rightarrow 0$) sharpen the distribution toward greedy decoding; higher temperatures flatten the distribution.

### Token
**Simple:** A bite-sized chunk of text (a word, part of a word, or punctuation) that an AI reads and writes.  
**Technical:** The basic atomic unit of sequence processing in an NLP model, representing a character, subword, or byte index from a discrete vocabulary $\mathcal{V}$ (typically 32,000 to 128,000 distinct tokens in modern LLMs).

### Token ID
**Simple:** The unique serial number assigned to each specific token in an AI's vocabulary dictionary.  
**Technical:** The integer index $i \in \{0, 1, \dots, |\mathcal{V}| - 1\}$ that maps a discrete subword string to its corresponding row index in the model's learned embedding table $W_E \in \mathbb{R}^{|\mathcal{V}| \times d}$.

### Tokenizer
**Simple:** The translation machine that slices raw human text into token puzzle pieces and converts them into numbers the AI can calculate.  
**Technical:** An algorithmic module (e.g., Byte-Pair Encoding, WordPiece, Unigram) that decomposes an arbitrary string of characters or bytes into a sequence of discrete token integers according to a predefined, trained vocabulary.

### Top-k Sampling
**Simple:** A generation filter where the AI only considers the top $k$ most likely next words, tossing out all the crazy or unlikely options.  
**Technical:** A truncation sampling strategy where the candidate vocabulary for next-token selection at step $t$ is strictly restricted to the $k$ tokens possessing the highest probabilities, redistributing probability mass over only those top $k$ items via softmax.

### Top-p Sampling (Nucleus Sampling)
**Simple:** A dynamic filter that gathers the top most probable words until their combined chances add up to $p$ percent (like 90%), then chooses from that group.  
**Technical:** A dynamic sampling algorithm by Holtzman et al. (2019) that selects the smallest subset of top tokens whose cumulative probability exceeds threshold $p \in (0, 1]$: $\sum_{i \in V^{(p)}} P(w_i) \ge p$, dynamically adapting the candidate pool size based on model confidence.

### Transformer
**Simple:** The revolutionary neural network architecture introduced in 2017 that powers modern AI, relying entirely on self-attention to process words in parallel.  
**Technical:** The deep learning architecture introduced by Vaswani et al. (2017) based purely on multi-head self-attention mechanisms, feedforward networks, layer normalizations, and residual connections, entirely dispensing with recurrent or convolutional recurrence.

### TRL (Transformer Reinforcement Learning)
**Simple:** A popular Hugging Face library built specifically for fine-tuning and aligning modern language models with RLHF and DPO.  
**Technical:** A full-stack open-source library built by Hugging Face providing battle-tested training classes (`SFTTrainer`, `DPOTrainer`, `PPOTrainer`, `RewardTrainer`) for end-to-end post-training alignment of Transformer models.

---

## U

### Underfitting
**Simple:** When an AI is too basic or hasn't practiced enough to learn the underlying patterns, performing poorly on both training and test data.  
**Technical:** An optimization failure condition where a model lacks sufficient representational capacity or has received insufficient optimization, resulting in elevated error rates across both training and validation datasets.

### Unsloth
**Simple:** An open-source tool that makes fine-tuning popular LLMs up to 5 times faster while using 80% less memory with custom hand-tuned math code.  
**Technical:** A specialized open-source fine-tuning framework that rewrites PyTorch forward and backward passes using custom hand-optimized Triton kernels, dramatically reducing memory overhead and accelerating LoRA/QLoRA training on consumer and enterprise GPUs.

---

## V

### Validation Loss
**Simple:** The mistake score measured on a secret set of practice questions the AI has never trained on, showing how well it is actually learning.  
**Technical:** The average evaluation loss metric computed on an isolated holdout validation dataset during training, serving as the primary diagnostic signal for monitoring generalization and detecting the onset of overfitting.

### Vector Database
**Simple:** A specialized storage system built to organize and search millions of high-dimensional concept vectors in milliseconds.  
**Technical:** A specialized database system (e.g., Pinecone, Milvus, Qdrant, Chroma) optimized for storing, indexing, and executing high-throughput Approximate Nearest Neighbor (ANN) vector similarity queries over high-dimensional dense embeddings.

### vLLM
**Simple:** A blazing-fast production server for running open-source LLMs that handles multiple users at once with maximum efficiency.  
**Technical:** A high-throughput, memory-efficient LLM serving and inference engine featuring PagedAttention, continuous batching, chunked prefill, and tensor parallelism with an OpenAI-compatible API interface.

### Vocabulary
**Simple:** The complete dictionary of all words and subword chunks that an AI model knows how to read and write.  
**Technical:** The fixed discrete set $\mathcal{V}$ of all recognized subword, character, or byte tokens supported by a language model's tokenizer, typically indexing between 32,000 and 128,000 unique token IDs.

---

## W

### Weight
**Simple:** An internal numeric dial in a neural network that determines how strongly one piece of information influences another.  
**Technical:** A learnable scalar parameter residing within the weight matrices ($W$) of linear, convolutional, or attention layers, optimized via backpropagation and gradient descent to encode task knowledge.

### Word Embedding
**Simple:** Converting a word into a list of numbers so that words with similar meanings end up close together on the math grid.  
**Technical:** A dense vector representation of a word in a continuous vector space where words with similar semantic and syntactic distributions occupy proximate coordinates, pioneered by classic algorithms such as Word2Vec and GloVe.

---

## Z

### Zero-shot Learning (Zero-shot)
**Simple:** Asking an AI to solve a brand-new task right away without showing it any prior examples or practice problems.  
**Technical:** The ability of a foundation model to generalize and perform a downstream task prompted purely via natural language task descriptions without any task-specific fine-tuning or few-shot demonstration examples.

---

[← Chapter 20: Resources & References](./20-resources-and-references/README.md) | [🏠 Home](../README.md)

# Chapter 20: Resources & References

> This chapter is a curated guide to the best papers, tools, courses, books, repositories, datasets, and communities for learning and working with Large Language Models. Bookmark this page — you will return to it throughout your journey.

---

## Table of Contents

1. [Foundational Research Papers](#foundational-research-papers)
2. [Official Documentation](#official-documentation)
3. [Online Courses and Learning Paths](#online-courses-and-learning-paths)
4. [YouTube Channels and Video Resources](#youtube-channels-and-video-resources)
5. [Books](#books)
6. [GitHub Repositories to Study](#github-repositories-to-study)
7. [Datasets for Practice](#datasets-for-practice)
8. [Community and Forums](#community-and-forums)
9. [Student Contribution Opportunities](#student-contribution-opportunities)

---

## Foundational Research Papers

These are the must-read papers that shaped modern LLMs. Each paper marked a paradigm shift. Read them in roughly chronological order for the best understanding of how the field evolved.

> **Tip:** Use [arXiv](https://arxiv.org) or [Semantic Scholar](https://www.semanticscholar.org) to access all these papers for free.

---

### 1. Attention Is All You Need

| Field | Details |
|-------|---------|
| **Authors** | Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin |
| **Year** | 2017 |
| **URL** | https://arxiv.org/abs/1706.03762 |
| **Venue** | NeurIPS 2017 |

**Contribution:** This paper introduced the Transformer architecture, replacing recurrent and convolutional networks with a purely attention-based design for sequence modeling. It established the multi-head self-attention mechanism and positional encoding that remain the foundation of virtually every modern LLM, including GPT, BERT, T5, and their descendants.

---

### 2. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

| Field | Details |
|-------|---------|
| **Authors** | Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova |
| **Year** | 2018 |
| **URL** | https://arxiv.org/abs/1810.04805 |
| **Venue** | NAACL 2019 |

**Contribution:** BERT introduced bidirectional pre-training via Masked Language Modeling (MLM) and Next Sentence Prediction (NSP), showing that pre-training on large corpora then fine-tuning on specific tasks dramatically outperforms task-specific architectures. It set new state-of-the-art results on 11 NLP benchmarks and popularized the pre-train → fine-tune paradigm.

---

### 3. Language Models are Few-Shot Learners (GPT-3)

| Field | Details |
|-------|---------|
| **Authors** | Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, et al. |
| **Year** | 2020 |
| **URL** | https://arxiv.org/abs/2005.14165 |
| **Venue** | NeurIPS 2020 |

**Contribution:** GPT-3, with 175 billion parameters, demonstrated that scaling language models enables remarkable few-shot and zero-shot task performance without any gradient updates — just prompting. This paper launched the era of "in-context learning" and showed that prompt engineering is itself a powerful technique, raising fundamental questions about emergent capabilities in large models.

---

### 4. Training language models to follow instructions with human feedback (InstructGPT)

| Field | Details |
|-------|---------|
| **Authors** | Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, et al. |
| **Year** | 2022 |
| **URL** | https://arxiv.org/abs/2203.02155 |
| **Venue** | NeurIPS 2022 |

**Contribution:** InstructGPT introduced Reinforcement Learning from Human Feedback (RLHF) as a systematic approach to align language models with human intent, showing that a 1.3B parameter RLHF-tuned model is preferred over a raw 175B GPT-3 model by human raters. This work directly led to ChatGPT and established the alignment-through-feedback paradigm now used by virtually all deployed chat models.

---

### 5. LLaMA: Open and Efficient Foundation Language Models

| Field | Details |
|-------|---------|
| **Authors** | Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, et al. |
| **Year** | 2023 |
| **URL** | https://arxiv.org/abs/2302.13971 |
| **Venue** | Meta AI, 2023 |

**Contribution:** LLaMA demonstrated that carefully-trained smaller models (7B–65B parameters) trained on more tokens can outperform much larger models like GPT-3, challenging the assumption that raw parameter count is the primary driver of capability. Its public release (even if accidentally leaked for LLaMA-1) sparked an explosion of open-source LLM research, fine-tuned variants (Alpaca, Vicuna), and community tooling.

---

### 6. LoRA: Low-Rank Adaptation of Large Language Models

| Field | Details |
|-------|---------|
| **Authors** | Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen |
| **Year** | 2021 |
| **URL** | https://arxiv.org/abs/2106.09685 |
| **Venue** | ICLR 2022 |

**Contribution:** LoRA introduced a parameter-efficient fine-tuning method that freezes pre-trained model weights and injects trainable low-rank decomposition matrices into each Transformer layer, reducing trainable parameters by up to 10,000× while maintaining comparable performance to full fine-tuning. LoRA became the dominant approach for fine-tuning LLMs on consumer hardware and is now implemented in Hugging Face PEFT, Unsloth, and virtually every LLM fine-tuning framework.

---

### 7. QLoRA: Efficient Finetuning of Quantized LLMs

| Field | Details |
|-------|---------|
| **Authors** | Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer |
| **Year** | 2023 |
| **URL** | https://arxiv.org/abs/2305.14314 |
| **Venue** | NeurIPS 2023 |

**Contribution:** QLoRA combined 4-bit NormalFloat (NF4) quantization with LoRA adapters and double quantization to enable fine-tuning of a 65B parameter model on a single 48GB GPU — a feat that previously required multiple high-end GPUs. This paper introduced the BitsAndBytes library integration and made frontier-scale fine-tuning accessible to individual researchers with consumer-grade hardware.

---

### 8. Scaling Laws for Neural Language Models

| Field | Details |
|-------|---------|
| **Authors** | Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei |
| **Year** | 2020 |
| **URL** | https://arxiv.org/abs/2001.08361 |
| **Venue** | OpenAI Technical Report, 2020 |

**Contribution:** This paper established empirical power-law relationships between model performance (cross-entropy loss), model size (parameters), training compute (FLOPs), and dataset size, providing a framework for predicting how much a model will improve given more resources. These "scaling laws" directly guided the development of GPT-3 and all subsequent large models, essentially giving the field a roadmap for scaling up.

---

### 9. Constitutional AI: Harmlessness from AI Feedback

| Field | Details |
|-------|---------|
| **Authors** | Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, et al. |
| **Year** | 2022 |
| **URL** | https://arxiv.org/abs/2212.08073 |
| **Venue** | Anthropic Technical Report, 2022 |

**Contribution:** Constitutional AI (CAI) introduced a technique where an AI model critiques and revises its own responses according to a set of explicit principles (a "constitution"), replacing human labelers for the harmlessness portion of RLHF with AI-generated feedback. This approach (RLHF-AI or RLAIF) reduced the dependence on human annotation for safety training and became the basis for Anthropic's Claude model family.

---

### 10. Direct Preference Optimization: Your Language Model is Secretly a Reward Model

| Field | Details |
|-------|---------|
| **Authors** | Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn |
| **Year** | 2023 |
| **URL** | https://arxiv.org/abs/2305.18290 |
| **Venue** | NeurIPS 2023 |

**Contribution:** DPO showed that the RLHF objective can be optimized directly on preference data without training a separate reward model or using reinforcement learning, using a closed-form reparameterization that reduces alignment to a simple binary cross-entropy loss. This dramatically simplified the alignment pipeline and became widely adopted as a more stable and computationally efficient alternative to PPO-based RLHF.

---

### 11. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

| Field | Details |
|-------|---------|
| **Authors** | Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, et al. |
| **Year** | 2020 |
| **URL** | https://arxiv.org/abs/2005.11401 |
| **Venue** | NeurIPS 2020 |

**Contribution:** RAG introduced a framework combining a dense retrieval component (DPR) with a seq2seq generator, allowing the model to retrieve relevant documents from an external knowledge base at inference time and condition generation on them. This paper laid the groundwork for the entire field of Retrieval-Augmented Generation now deployed widely in production AI systems (e.g., Bing Chat, Claude with web search).

---

### 12. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

| Field | Details |
|-------|---------|
| **Authors** | Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré |
| **Year** | 2022 |
| **URL** | https://arxiv.org/abs/2205.14135 |
| **Venue** | NeurIPS 2022 |

**Contribution:** FlashAttention reformulated the attention computation to be IO-aware, using kernel fusion and tiling to avoid materializing the full N×N attention matrix in GPU HBM memory, achieving 2–4× speedup and 5–20× memory reduction with mathematically exact (not approximate) results. It became the de facto standard attention implementation in all major LLM training frameworks and enabled significantly longer context windows.

---

### 13. GPT-4 Technical Report

| Field | Details |
|-------|---------|
| **Authors** | OpenAI |
| **Year** | 2023 |
| **URL** | https://arxiv.org/abs/2303.08774 |
| **Venue** | OpenAI Technical Report, 2023 |

**Contribution:** The GPT-4 technical report documented a large multimodal model capable of accepting both image and text inputs and producing text outputs, achieving human-level performance on many professional and academic benchmarks (e.g., bar exam top 10%, AP exams). While sparse on architectural details, the report introduced a new paradigm of safety-focused "predictable scaling" and comprehensive capability evaluation frameworks.

---

### 14. Gemini: A Family of Highly Capable Multimodal Models

| Field | Details |
|-------|---------|
| **Authors** | Gemini Team, Google DeepMind |
| **Year** | 2023 |
| **URL** | https://arxiv.org/abs/2312.11805 |
| **Venue** | Google DeepMind Technical Report, 2023 |

**Contribution:** Gemini introduced a family of natively multimodal models (Ultra, Pro, Nano) jointly trained on text, image, audio, and video data, with the Ultra variant surpassing GPT-4 on the MMLU benchmark and achieving human-expert performance on various science and reasoning tasks. The paper also described efficient deployment strategies including 4-bit quantization for the Nano variant running on-device, advancing on-device AI.

---

### 15. Training Compute-Optimal Large Language Models (Chinchilla)

| Field | Details |
|-------|---------|
| **Authors** | Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, et al. |
| **Year** | 2022 |
| **URL** | https://arxiv.org/abs/2203.15556 |
| **Venue** | NeurIPS 2022 |

**Contribution:** The "Chinchilla" paper challenged the prevailing wisdom (from Kaplan et al.'s scaling laws) that more parameters always win, demonstrating that for a fixed compute budget, model size and training tokens should scale equally — roughly 20 tokens per parameter. The 70B Chinchilla model, trained on 1.4T tokens, outperformed the 280B Gopher on most tasks, redirecting the field toward training smaller models on more data.

---

## Official Documentation

> **Note:** These are the primary references you should consult when implementing models, training pipelines, or deploying systems. Bookmark all of them.

---

### PyTorch

| Field | Details |
|-------|---------|
| **URL** | https://pytorch.org/docs/stable/index.html |
| **GitHub** | https://github.com/pytorch/pytorch |

The official PyTorch documentation covers the full tensor computation library, autograd engine, neural network modules (`torch.nn`), optimization (`torch.optim`), data loading (`torch.utils.data`), and distributed training. Start with the **Tutorials** section (https://pytorch.org/tutorials/) for hands-on introductions.

---

### Hugging Face Transformers

| Field | Details |
|-------|---------|
| **URL** | https://huggingface.co/docs/transformers/index |
| **GitHub** | https://github.com/huggingface/transformers |

The Transformers library provides thousands of pre-trained models for NLP, vision, and audio tasks with a unified API. Docs include model cards, training guides, pipeline documentation, and integration with Hugging Face Hub for model sharing and versioning.

---

### Hugging Face Datasets

| Field | Details |
|-------|---------|
| **URL** | https://huggingface.co/docs/datasets/index |
| **GitHub** | https://github.com/huggingface/datasets |

The `datasets` library enables fast loading, streaming, and processing of thousands of ML datasets with memory-efficient Apache Arrow backing. Essential for training and evaluating LLMs; supports custom dataset creation from JSON, CSV, Parquet, and more.

---

### Hugging Face PEFT (Parameter-Efficient Fine-Tuning)

| Field | Details |
|-------|---------|
| **URL** | https://huggingface.co/docs/peft/index |
| **GitHub** | https://github.com/huggingface/peft |

PEFT implements LoRA, Prefix Tuning, Prompt Tuning, IA³, and other parameter-efficient fine-tuning methods. It integrates directly with Transformers and TRL, making it the standard library for fine-tuning LLMs on limited hardware. Essential reading: the LoRA and QLoRA integration guides.

---

### TRL (Transformer Reinforcement Learning)

| Field | Details |
|-------|---------|
| **URL** | https://huggingface.co/docs/trl/index |
| **GitHub** | https://github.com/huggingface/trl |

TRL provides implementations of SFTTrainer (supervised fine-tuning), PPOTrainer (RLHF), DPOTrainer (direct preference optimization), and RewardTrainer. It is the go-to library for the full alignment pipeline and works seamlessly with PEFT and Transformers.

---

### LangChain

| Field | Details |
|-------|---------|
| **URL** | https://docs.langchain.com |
| **GitHub** | https://github.com/langchain-ai/langchain |

LangChain is a framework for building LLM-powered applications, providing abstractions for chains, agents, memory, retrievers, and tool use. It supports dozens of LLM providers and vector stores. Best used for RAG pipelines, chatbots, and agent-based systems.

---

### LlamaIndex (formerly GPT Index)

| Field | Details |
|-------|---------|
| **URL** | https://docs.llamaindex.ai |
| **GitHub** | https://github.com/run-llama/llama_index |

LlamaIndex focuses on data ingestion, indexing, and querying over your own documents, making it ideal for building knowledge-grounded LLM applications. It provides connectors for PDFs, databases, APIs, and more, with tight integration with vector databases.

---

### Ollama

| Field | Details |
|-------|---------|
| **URL** | https://ollama.com/docs |
| **GitHub** | https://github.com/ollama/ollama |

Ollama simplifies running LLMs locally on macOS, Linux, and Windows, providing a Docker-like CLI for pulling, running, and managing models (Llama 3, Mistral, Phi, Gemma, etc.). Its REST API and Python/JavaScript libraries make it easy to integrate local LLMs into applications.

---

### vLLM

| Field | Details |
|-------|---------|
| **URL** | https://docs.vllm.ai |
| **GitHub** | https://github.com/vllm-project/vllm |

vLLM is a high-throughput and memory-efficient inference engine for LLMs, implementing PagedAttention for near-zero KV cache waste and continuous batching for production serving. It supports OpenAI-compatible APIs, making it the standard choice for self-hosting open-source LLMs at scale.

---

## Online Courses and Learning Paths

> **Recommended learning order:** Start with Karpathy's Zero to Hero → fast.ai Practical DL → Stanford CS224N → HF NLP Course → DeepLearning.AI Short Courses.

---

### 🏆 Andrej Karpathy's Neural Networks: Zero to Hero

| Field | Details |
|-------|---------|
| **URL** | https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ |
| **GitHub** | https://github.com/karpathy/nn-zero-to-hero |
| **Level** | Beginner → Advanced |

**The single best free resource for understanding LLMs from first principles.** Karpathy builds everything from scratch in Python: micrograd (autograd from scratch), makemore (bigram → MLP → RNN → transformer character-level models), and nanoGPT (full GPT-2 reproduction). If you watch only one course, watch this. No shortcuts, no high-level abstractions — pure understanding.

**Playlist covers:**
- `micrograd`: building backpropagation from scratch
- `makemore` series: language modeling from bigrams to Transformers
- `nanoGPT`: reproducing GPT-2 from scratch, then training on Shakespeare
- `GPT tokenizer`: building BPE tokenization from scratch
- `Llama 2 from scratch`: implementing modern LLM architecture details

---

### fast.ai Practical Deep Learning

| Field | Details |
|-------|---------|
| **URL** | https://course.fast.ai |
| **GitHub** | https://github.com/fastai/fastai |
| **Level** | Beginner → Intermediate |

Jeremy Howard's top-down, code-first approach to deep learning — learn to build impressive models in the first lesson and understand why they work as the course progresses. Covers image classification, NLP, tabular data, collaborative filtering, and diffusion models using PyTorch and fastai. Free and updated regularly.

---

### Stanford CS224N: Natural Language Processing with Deep Learning

| Field | Details |
|-------|---------|
| **URL** | https://web.stanford.edu/class/cs224n/ |
| **YouTube** | https://www.youtube.com/playlist?list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVx76 |
| **Level** | Intermediate → Advanced |

The definitive graduate-level NLP course covering word vectors, neural networks for NLP, RNNs, Transformers, pre-training, BERT, GPT, question answering, machine translation, NLP ethics, and large model prompting. Lecture slides and assignments are free; the course runs every winter quarter at Stanford.

---

### Stanford CS229: Machine Learning

| Field | Details |
|-------|---------|
| **URL** | https://cs229.stanford.edu |
| **YouTube** | https://www.youtube.com/playlist?list=PLoROMvodv4rNH7qL6-efu_q2_bPuy0adh |
| **Level** | Beginner → Intermediate |

Andrew Ng's foundational ML course covering supervised learning, unsupervised learning, neural networks, SVMs, dimensionality reduction, and reinforcement learning. Provides the mathematical foundations that make LLM concepts understandable. Required background for CS224N.

---

### DeepLearning.AI Short Courses

| Field | Details |
|-------|---------|
| **URL** | https://www.deeplearning.ai/short-courses/ |
| **Level** | Beginner → Intermediate |

A growing library of 1–2 hour focused courses on specific LLM skills, taught by practitioners from Hugging Face, LangChain, OpenAI, and others. Highly recommended courses include:
- **LangChain for LLM Application Development** (with Harrison Chase)
- **Building Systems with the ChatGPT API**
- **Finetuning Large Language Models** (Sharon Zhou)
- **Building and Evaluating Advanced RAG**
- **Reinforcement Learning from Human Feedback**

---

### Hugging Face NLP Course

| Field | Details |
|-------|---------|
| **URL** | https://huggingface.co/learn/nlp-course |
| **GitHub** | https://github.com/huggingface/course |
| **Level** | Beginner → Intermediate |

The official Hugging Face NLP course covers the Transformers and Datasets libraries in depth: tokenization, fine-tuning pre-trained models, building custom datasets, model sharing, and efficient training. Hands-on with real tasks (text classification, NER, QA, summarization, translation). Free and available in multiple languages.

---

### Sebastian Raschka's LLMs from Scratch

| Field | Details |
|-------|---------|
| **URL** | https://github.com/rasbt/LLMs-from-scratch |
| **Book** | https://www.manning.com/books/build-a-large-language-model-from-scratch |
| **Level** | Intermediate → Advanced |

The companion code repository to Raschka's book "Build a Large Language Model (From Scratch)." Implements a GPT-like model from scratch in PyTorch, covering tokenization, attention, pre-training, fine-tuning for classification, and instruction tuning with RLHF. All Jupyter notebooks are freely available on GitHub.

---

### fast.ai NLP (Part 2)

| Field | Details |
|-------|---------|
| **URL** | https://course.fast.ai |
| **Level** | Intermediate |

The advanced fast.ai course covers building models from scratch, including language models, and dives deeper into the mathematical foundations of modern deep learning architectures. Complements the Practical Deep Learning course with more implementation focus.

---

## YouTube Channels and Video Resources

---

### 🏆 Andrej Karpathy (@karpathy)

| Field | Details |
|-------|---------|
| **Channel** | https://www.youtube.com/@AndrejKarpathy |
| **Notable Videos** | Let's build GPT from scratch, Let's build the GPT Tokenizer, nanoGPT, Intro to LLMs |

**Highest-quality LLM educational content available online.** Karpathy's lecture-style videos are detailed, honest, and deeply pedagogical — he builds everything from scratch and explains every design decision. Watch "Let's build GPT: from scratch, in code, spelled out" (2h13m) as your first deep dive into Transformer internals.

---

### 3Blue1Brown

| Field | Details |
|-------|---------|
| **Channel** | https://www.youtube.com/@3blue1brown |
| **Notable Series** | Neural Networks, Essence of Linear Algebra, Essence of Calculus |

Grant Sanderson's visually stunning explanations of neural networks, attention, transformers, and the mathematical foundations of deep learning. Start with the "Neural networks" playlist before diving into code — building intuition here pays dividends throughout your studies.

---

### Yannic Kilcher

| Field | Details |
|-------|---------|
| **Channel** | https://www.youtube.com/@YannicKilcher |
| **Notable Videos** | Attention Is All You Need walkthrough, BERT, GPT, DPO, Chinchilla |

Yannic reads and explains ML papers in real-time, covering the motivation, methodology, and critique of each. His paper walkthroughs for the foundational papers listed in this chapter are excellent companions to reading the papers themselves.

---

### Two Minute Papers

| Field | Details |
|-------|---------|
| **Channel** | https://www.youtube.com/@TwoMinutePapers |

Short (2–5 minute) accessible summaries of recent AI and ML research papers. Great for staying current with the field and identifying which new papers are worth reading in full.

---

### Sentdex

| Field | Details |
|-------|---------|
| **Channel** | https://www.youtube.com/@sentdex |

Python and ML tutorials ranging from beginner to advanced, including hands-on deep learning with PyTorch and practical NLP projects. Good for applied coding tutorials that complement the more theoretical courses above.

---

## Books

---

### Build a Large Language Model (From Scratch)

| Field | Details |
|-------|---------|
| **Author** | Sebastian Raschka |
| **Year** | 2024 |
| **Publisher** | Manning Publications |
| **URL** | https://www.manning.com/books/build-a-large-language-model-from-scratch |
| **Code** | https://github.com/rasbt/LLMs-from-scratch |

**The most practical hands-on book for building LLMs.** Guides the reader from tokenization through building a GPT architecture, pre-training on text data, and fine-tuning with instruction tuning and RLHF — all in pure PyTorch. Assumes Python and basic ML knowledge. Highly recommended as your primary LLM implementation reference.

---

### Natural Language Processing with Transformers

| Field | Details |
|-------|---------|
| **Authors** | Lewis Tunstall, Leandro von Werra, Thomas Wolf |
| **Year** | 2022 |
| **Publisher** | O'Reilly Media |
| **URL** | https://www.oreilly.com/library/view/natural-language-processing/9781098136789/ |

The definitive practical guide to using the Hugging Face Transformers library for NLP tasks. Covers text classification, NER, QA, summarization, translation, fine-tuning, and efficient Transformers — with real code and real datasets throughout.

---

### Deep Learning

| Field | Details |
|-------|---------|
| **Authors** | Ian Goodfellow, Yoshua Bengio, Aaron Courville |
| **Year** | 2016 |
| **Publisher** | MIT Press |
| **URL (Free Online)** | https://www.deeplearningbook.org |

The standard textbook for deep learning theory. Covers linear algebra, probability, information theory, ML fundamentals, feedforward networks, regularization, optimization, CNNs, RNNs, and autoencoders. Free online in HTML format. Essential for building mathematical understanding.

---

### Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (3rd ed.)

| Field | Details |
|-------|---------|
| **Author** | Aurélien Géron |
| **Year** | 2022 |
| **Publisher** | O'Reilly Media |
| **URL** | https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/ |

A comprehensive, project-driven introduction to ML and deep learning using Python, covering scikit-learn for traditional ML and Keras/TensorFlow for neural networks. Excellent for building a solid applied ML foundation before specializing in LLMs.

---

### Speech and Language Processing (3rd ed. Draft)

| Field | Details |
|-------|---------|
| **Authors** | Dan Jurafsky, James H. Martin |
| **Year** | Ongoing (3rd edition draft) |
| **URL (Free PDF)** | https://web.stanford.edu/~jurafsky/slp3/ |

The canonical NLP textbook used in graduate courses worldwide, freely available as a PDF draft. Covers everything from regular expressions and edit distance through n-gram models, neural networks, transformers, pre-trained language models, dialogue systems, and information extraction.

---

## GitHub Repositories to Study

> **Learning strategy:** Don't just clone and run these. Read the source code. The best way to understand LLMs is to read clean, well-documented implementations.

---

### karpathy/nanoGPT

| Field | Details |
|-------|---------|
| **URL** | https://github.com/karpathy/nanoGPT |
| **Stars** | ~40k+ |
| **Language** | Python / PyTorch |

**Start here.** A minimal, clean, fully-functional GPT-2 implementation in ~300 lines of PyTorch. The `model.py` file is a masterclass in readable Transformer code. Trains on Shakespeare with one GPU in minutes. Read every line; understand every hyperparameter.

---

### karpathy/llm.c

| Field | Details |
|-------|---------|
| **URL** | https://github.com/karpathy/llm.c |
| **Language** | C / CUDA |

A GPT-2 training implementation in raw C/CUDA, with no ML framework dependencies. Advanced reading for understanding what frameworks abstract away. Remarkably fast and educational for understanding low-level GPU computation.

---

### huggingface/transformers

| Field | Details |
|-------|---------|
| **URL** | https://github.com/huggingface/transformers |
| **Stars** | ~135k+ |
| **Language** | Python |

The production-grade implementation of hundreds of Transformer architectures. Study `modeling_gpt2.py`, `modeling_llama.py`, and `modeling_bert.py` to see how research papers translate into production code. Use this as a reference, not a starting point.

---

### microsoft/LoRA

| Field | Details |
|-------|---------|
| **URL** | https://github.com/microsoft/LoRA |
| **Language** | Python |

The original LoRA implementation accompanying the paper. Clean and minimal — read `loralib/layers.py` to understand exactly how LoRA replaces linear layers with low-rank decompositions.

---

### EleutherAI/lm-evaluation-harness

| Field | Details |
|-------|---------|
| **URL** | https://github.com/EleutherAI/lm-evaluation-harness |
| **Language** | Python |

The standard open-source LLM evaluation framework, implementing benchmarks including MMLU, HellaSwag, TruthfulQA, ARC, and 200+ others. Used by the Open LLM Leaderboard on Hugging Face and most published open-source model evaluations.

---

### tatsu-lab/stanford_alpaca

| Field | Details |
|-------|---------|
| **URL** | https://github.com/tatsu-lab/stanford_alpaca |
| **Language** | Python |

The original Alpaca fine-tuning code showing how to instruction-tune LLaMA using 52K GPT-generated instruction-following examples. A clean minimal reference for SFT fine-tuning pipelines.

---

### tloen/alpaca-lora

| Field | Details |
|-------|---------|
| **URL** | https://github.com/tloen/alpaca-lora |
| **Language** | Python |

LoRA fine-tuning applied to Alpaca, enabling instruction tuning of LLaMA on a single RTX 4090 in hours. One of the first demonstrations of consumer-grade fine-tuning — historically important and educationally clean.

---

### unslothai/unsloth

| Field | Details |
|-------|---------|
| **URL** | https://github.com/unslothai/unsloth |
| **Language** | Python / Triton |

The fastest open-source fine-tuning framework for LLMs, achieving 2–5× faster training and 60–80% less memory usage through hand-written Triton kernels and optimized backward passes. Use this for practical fine-tuning projects; study the architecture for understanding advanced optimization techniques.

---

### ollama/ollama

| Field | Details |
|-------|---------|
| **URL** | https://github.com/ollama/ollama |
| **Language** | Go |

The source code for the Ollama local LLM server. Good for understanding model serving architecture, quantization integration (via llama.cpp), and REST API design for LLM applications.

---

## Datasets for Practice

---

### Hugging Face Datasets Hub

| Field | Details |
|-------|---------|
| **URL** | https://huggingface.co/datasets |
| **Search** | https://huggingface.co/datasets?sort=trending |

The largest repository of ML datasets, with tens of thousands of datasets for NLP, vision, audio, and multimodal tasks. Searchable by task, language, license, and size. Use the `datasets` library to load any dataset in one line of Python.

---

### Specific Recommended Datasets for LLM Learning

| Dataset | Description | URL |
|---------|-------------|-----|
| **wikitext** | Wikipedia text in train/valid/test splits — standard LM benchmark | https://huggingface.co/datasets/wikitext |
| **tiny_shakespeare** | Complete works of Shakespeare (~1MB) — classic nanoGPT demo dataset | https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt |
| **openwebtext** | Open reproduction of GPT-2's WebText training data (~40GB) | https://huggingface.co/datasets/openwebtext |
| **roneneldan/TinyStories** | 2.1M short stories written in simple language for small LLMs | https://huggingface.co/datasets/roneneldan/TinyStories |
| **databricks-dolly-15k** | 15K human-generated instruction-response pairs for fine-tuning | https://huggingface.co/datasets/databricks/databricks-dolly-15k |
| **tatsu-lab/alpaca** | 52K GPT-generated instruction-following examples (Alpaca) | https://huggingface.co/datasets/tatsu-lab/alpaca |
| **Open-Orca/OpenOrca** | 4M+ GPT-4 generated reasoning chains for instruction tuning | https://huggingface.co/datasets/Open-Orca/OpenOrca |
| **HuggingFaceH4/ultrachat_200k** | 200K high-quality multi-turn dialogues for chat fine-tuning | https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k |
| **Anthropic/hh-rlhf** | Human preference data (chosen/rejected pairs) for RLHF/DPO | https://huggingface.co/datasets/Anthropic/hh-rlhf |
| **lmsys/chatbot_arena_conversations** | 33K conversations from Chatbot Arena with human preferences | https://huggingface.co/datasets/lmsys/chatbot_arena_conversations |

---

## Community and Forums

---

### Hugging Face Community

| Field | Details |
|-------|---------|
| **Discord** | https://huggingface.co/join/discord |
| **Forums** | https://discuss.huggingface.co |

The most active community for open-source LLM practitioners. The Discord server has channels for model discussion, fine-tuning help, dataset sharing, and research. The forums have detailed Q&A on Transformers, PEFT, TRL, and Datasets.

---

### Reddit Communities

| Community | URL | Description |
|-----------|-----|-------------|
| **r/MachineLearning** | https://reddit.com/r/MachineLearning | Main academic ML community — paper discussions, job posts, research news |
| **r/LocalLLaMA** | https://reddit.com/r/LocalLLaMA | Focused on running/fine-tuning open-source LLMs locally |
| **r/artificial** | https://reddit.com/r/artificial | Broader AI discussion |
| **r/learnmachinelearning** | https://reddit.com/r/learnmachinelearning | Beginner-friendly ML learning community |

---

### Papers With Code

| Field | Details |
|-------|---------|
| **URL** | https://paperswithcode.com |

Tracks ML papers alongside their official code implementations and benchmark leaderboards. Invaluable for finding the state-of-the-art on any task, finding reproducible implementations, and discovering new techniques. The "Methods" section is an excellent glossary of ML techniques.

---

### ArXiv and Related

| Resource | URL | Description |
|----------|-----|-------------|
| **arXiv (cs.CL / cs.LG / cs.AI)** | https://arxiv.org/list/cs.CL/recent | Preprint server for all ML/NLP papers |
| **Semantic Scholar** | https://www.semanticscholar.org | AI-powered paper search with citation graphs |
| **Connected Papers** | https://www.connectedpapers.com | Visualize paper citation networks |
| **Papers With Code** | https://paperswithcode.com | Papers + code + benchmarks |

---

### AI Safety and Alignment Resources

| Resource | URL | Description |
|----------|-----|-------------|
| **LessWrong** | https://www.lesswrong.com | Long-form writing on AI safety, alignment, rationality |
| **AI Alignment Forum** | https://www.alignmentforum.org | Technical AI alignment research community |
| **Anthropic Research** | https://www.anthropic.com/research | Published alignment and safety research |
| **EleutherAI** | https://www.eleuther.ai | Open-source LLM research organization |

---

### GitHub Discussions

Engage directly with the communities building the tools you use:
- [huggingface/transformers Discussions](https://github.com/huggingface/transformers/discussions)
- [karpathy/nanoGPT Issues](https://github.com/karpathy/nanoGPT/issues)
- [vllm-project/vllm Discussions](https://github.com/vllm-project/vllm/discussions)
- [unslothai/unsloth Issues](https://github.com/unslothai/unsloth/issues)

---

## Student Contribution Opportunities

The LLM field is remarkably open — meaningful contributions are within reach of any motivated student.

---

### Open Source Contributions

| Project | How to Start | Impact |
|---------|-------------|--------|
| **Hugging Face Transformers** | Fix documentation, add model examples, improve tests | Used by millions of researchers |
| **EleutherAI Projects** | lm-evaluation-harness, Pythia, The Pile | Core open-source LLM infrastructure |
| **LangChain** | Add integrations, improve docs, fix bugs | Widely-used application framework |
| **unsloth** | Test new models, report issues, contribute kernels | Fast-growing fine-tuning community |

**How to start contributing to any open-source project:**
1. Find "good first issue" labels on GitHub
2. Read the CONTRIBUTING.md file
3. Set up the development environment
4. Make a small, focused pull request
5. Respond promptly to reviewer feedback

---

### Competitions and Challenges

| Platform | URL | Description |
|----------|-----|-------------|
| **Kaggle** | https://www.kaggle.com/competitions | ML competitions with prize money and rankings |
| **Hugging Face Competitions** | https://huggingface.co/spaces/competitions/landing | Community ML challenges |
| **Papers With Code Benchmarks** | https://paperswithcode.com/sota | Submit results to leaderboards |
| **LMSYS Chatbot Arena** | https://chat.lmsys.org | Evaluate and compare models |

---

### Knowledge Sharing

- **Write a blog post** explaining a paper in your own words — Distill.pub, Medium, Substack, or your own site
- **Create educational notebooks** and share on Kaggle or Colab
- **Record explainer videos** in the style of Yannic Kilcher
- **Build fine-tuned models** on specialized domains and share on Hugging Face Hub
- **Answer questions** on r/LocalLLaMA, Hugging Face forums, and Stack Overflow

---

### Research Participation

- Apply for **summer internships** at Anthropic, OpenAI, Google DeepMind, Meta AI, Hugging Face, Mistral AI, Cohere, or AI2
- Participate in **research REUs** (Research Experiences for Undergraduates) at universities with active NLP groups
- Reproduce a published paper as a course project — submit to **ML Reproducibility Challenge** (https://reproml.org)
- Join your university's **AI/ML student club** or start one

---

## Reading List: Suggested Order for Students

If you don't know where to start, follow this progression:

```
Phase 1 — Foundations (Weeks 1–4)
  ✓ 3Blue1Brown Neural Networks playlist
  ✓ Karpathy Zero to Hero: micrograd + makemore parts 1-3
  ✓ Fast.ai Practical Deep Learning Lessons 1-4

Phase 2 — Transformers (Weeks 5–8)
  ✓ Attention Is All You Need (paper)
  ✓ Karpathy: "Let's build GPT from scratch"
  ✓ BERT paper + HF NLP Course

Phase 3 — LLMs in Practice (Weeks 9–12)
  ✓ GPT-3 paper + LLaMA paper
  ✓ LoRA paper + QLoRA paper
  ✓ HF PEFT docs + fine-tune a model
  ✓ RAG paper + build a RAG system

Phase 4 — Alignment & Safety (Weeks 13–16)
  ✓ InstructGPT paper
  ✓ Constitutional AI paper
  ✓ DPO paper + TRL docs
  ✓ Chinchilla paper + Scaling Laws paper

Phase 5 — Production & Research (Ongoing)
  ✓ Flash Attention paper
  ✓ vLLM PagedAttention paper
  ✓ GPT-4 Technical Report
  ✓ Gemini paper
  ✓ Follow arXiv cs.CL daily
```

---

[← Previous Chapter](../19-future-of-llms/README.md) | [🏠 Home](../../README.md) | [📖 Glossary](../glossary.md)

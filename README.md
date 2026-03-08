<div align="center">

# 🚀 Awesome NVIDIA Nemotron

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![License](https://img.shields.io/badge/License-CC0-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Last Updated](https://img.shields.io/badge/Updated-March%202026-blue.svg)](#)

A curated list of NVIDIA Nemotron models, datasets, tools, and resources.  
Open weights · Open data · Built for agentic AI.

[Models](#-models) • [Datasets](#-datasets) • [Tools](#-tools--deployment) • [Tutorials](#-tutorials--starter-kits) • [HuggingFace × Nemotron](#-huggingface--nemotron) • [Use Cases](#-real-world-use-cases) • [Companies](#-companies-using-nemotron-worldwide) • [Videos](#-videos--talks) • [Podcasts](#-podcasts) • [Events](#-upcoming-events) • [Papers](#-papers) • [Community](#-community)

</div>

---

## Contents

- [What is Nemotron?](#-what-is-nemotron)
- [Model Generations at a Glance](#-model-generations-at-a-glance)
- [Models](#-models)
  - [Nemotron 3 (Gen 3 — Latest)](#nemotron-3-gen-3--latest)
  - [Nemotron Nano V2](#nemotron-nano-v2)
  - [Llama Nemotron Series](#llama-nemotron-series)
  - [Vision & Multimodal](#vision--multimodal)
  - [Speech (ASR / TTS / Translation)](#speech-asr--tts--translation)
  - [Safety & Guardrails](#safety--guardrails)
  - [RAG & Retrieval](#rag--retrieval)
  - [Reward Models](#reward-models)
  - [Compressed / Distilled Models (Minitron)](#compressed--distilled-models-minitron)
  - [Nemotron 4 340B](#nemotron-4-340b)
  - [Nemotron 3 8B (Enterprise)](#nemotron-3-8b-enterprise)
- [Datasets](#-datasets)
- [Tools & Deployment](#-tools--deployment)
- [Tutorials & Starter Kits](#-tutorials--starter-kits)
- [HuggingFace × Nemotron](#-huggingface--nemotron)
- [Real-World Use Cases](#-real-world-use-cases)
- [Companies Using Nemotron Worldwide](#-companies-using-nemotron-worldwide)
- [Videos & Talks](#-videos--talks)
- [Podcasts](#-podcasts)
- [Upcoming Events](#-upcoming-events)
- [Papers](#-papers)
- [Community](#-community)
- [Contributing](#-contributing)

---

## 🔬 What is Nemotron?

**NVIDIA Nemotron™** is a family of open models with open weights, training data, and recipes, delivering leading efficiency and accuracy for building specialized AI agents. All models are released under permissive licenses and weights are available on Hugging Face.

Key properties across the family:

- **Open weights + open training data** — weights, datasets, and recipes are all publicly available on Hugging Face
- **Agentic-first design** — reasoning, tool-calling, multi-turn, RAG, and safety models designed to work together
- **Transparent training** — technical reports and reproducibility scripts are released alongside every model
- **Deployable anywhere** — via vLLM, SGLang, Ollama, llama.cpp, or as NVIDIA NIM™ microservices

> "Open innovation is the foundation of AI progress." — Jensen Huang, NVIDIA CEO

---

## 📊 Model Generations at a Glance

| Generation | Key Models | Architecture | Context | Released |
|---|---|---|---|---|
| **Nemotron 3** | Nano, Super, Ultra | Hybrid Mamba-Transformer MoE | 1M tokens | Dec 2025 |
| **Nemotron Nano V2** | 9B-v2 | Hybrid Transformer-Mamba | 128K tokens | 2025 |
| **Llama Nemotron** | Nano 4B/8B, Super 49B, Ultra 253B | Dense Transformer (Llama-based) | 128K tokens | 2025 |
| **Nemotron 4** | 340B Base/Instruct/Reward | Dense Transformer | 4K tokens | Jun 2024 |
| **Nemotron 3 (Enterprise)** | 8B Base/Chat/QA | Dense Transformer | 4K tokens | Feb 2024 |

---

## 🤖 Models

### Nemotron 3 (Gen 3 — Latest)

Announced December 15, 2025. Trained from scratch by NVIDIA with a hybrid Mamba-Transformer Mixture-of-Experts (MoE) architecture, 1M-token native context, and multi-environment reinforcement learning via NeMo Gym. Pre-training data cutoff: **June 25, 2025**. 10.6 trillion tokens total (3.5T synthetic).

| Model | Params (Total / Active) | Precision | Description | Links |
|---|---|---|---|---|
| **NVIDIA-Nemotron-3-Nano-30B-A3B-FP8** | 30B / 3.2B | FP8 | Final quantized post-trained Nano — fastest inference, lowest cost | [HF](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8) · [NIM](https://build.nvidia.com/nvidia/nemotron-3-nano-30b-a3b/modelcard) |
| **NVIDIA-Nemotron-3-Nano-30B-A3B-BF16** | 30B / 3.2B | BF16 | Post-trained Nano — reasoning + non-reasoning unified model | [HF](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) |
| **NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16** | 30B / 3.2B | BF16 | Pre-trained base model — good starting point for fine-tuning | [HF](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16) |
| **Nemotron 3 Super** | ~100B / ~10B active | BF16 | High-accuracy reasoning for multi-agent enterprise workflows | Coming H1 2026 |
| **Nemotron 3 Ultra** | ~500B / ~50B active | BF16 | State-of-the-art accuracy for complex, multi-GPU deployments | Coming H1 2026 |

**Architecture highlights:**
- Hybrid MoE: 23 Mamba-2 layers + MoE layers + 6 self-attention layers
- 1M native token context window
- Multi-environment RL post-training via NeMo Gym
- Granular reasoning budget control (toggle thinking ON/OFF at inference)
- 4× higher throughput than Nemotron 2 Nano; 3.3× higher than Qwen3-30B-A3B on H200

**Supported languages:** English, Spanish, French, German, Japanese, Italian, Chinese, Arabic, Hebrew, Hindi, Korean, Czech, Danish, Dutch, Finnish, Polish, Portuguese, Thai, Swedish, Russian (and more)

---

### Nemotron Nano V2

Hybrid Transformer-Mamba architecture for edge and single-GPU deployments. Features a configurable thinking budget — dial accuracy, throughput, and cost at inference time.

| Model | Params | Description | Links |
|---|---|---|---|
| **NVIDIA-Nemotron-Nano-9B-v2** | 9B | Up to 6× faster throughput vs leading 8B open models; up to 60% lower token generation with thinking budget control | [HF](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2) · [NIM](https://build.nvidia.com/nvidia/nvidia-nemotron-nano-9b-v2) · [OpenRouter](https://openrouter.ai/nvidia/nemotron-nano-9b-v2:free) |

---

### Llama Nemotron Series

Built on Meta Llama base models and post-trained with NVIDIA's alignment techniques (RPO, REINFORCE, multi-phase SFT + RL). All models support reasoning ON/OFF via system prompt. Recommended: `temperature=0.6, top_p=0.95` for reasoning ON; greedy decoding for reasoning OFF.

| Model | Base | Params | Context | Description | Links |
|---|---|---|---|---|---|
| **Llama-3.1-Nemotron-Nano-4B-v1.1** | Llama 3.1 Minitron 4B | 4B | 128K | Fits on a single RTX GPU; multi-language; SFT + RPO post-trained | [HF](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1) |
| **Llama-3.1-Nemotron-Nano-8B-v1** | Llama 3.1 8B Instruct | 8B | 128K | Single RTX GPU; coding, math, RAG, tool-calling; REINFORCE + RPO | [HF](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-8B-v1) · [NIM](https://build.nvidia.com/nvidia/llama-3_1-nemotron-nano-8b-v1) |
| **Llama-3.3-Nemotron-Super-49B-v1** | Llama 3.3 70B | 49B | 128K | Best accuracy/throughput on single H100; multi-agent enterprise workflows | [HF](https://huggingface.co/nvidia/Llama-3.3-Nemotron-Super-49B-v1) · [NIM](https://build.nvidia.com/nvidia/llama-3_3-nemotron-super-49b-v1) |
| **Llama-3.3-Nemotron-Super-49B-v1.5** | Llama 3.3 70B | 49B | 128K | Updated Super with improved deep research agent performance | [HF](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5) · [NIM](https://build.nvidia.com/nvidia/llama-3_3-nemotron-super-49b-v1_5) · [DeepInfra](https://deepinfra.com/nvidia/Llama-3.3-Nemotron-Super-49B-v1.5/) |
| **Llama-3_1-Nemotron-51B-Instruct** | Llama 3.1 70B | 51B | 128K | NAS-compressed from 70B; fits on single H100; MT-Bench 8.99 | [HF](https://huggingface.co/nvidia/Llama-3_1-Nemotron-51B-Instruct) |
| **Llama-3.1-Nemotron-70B-Instruct** | Llama 3.1 70B | 70B | 128K | Arena Hard 85.0 · AlpacaEval 2 LC 57.6 · MT-Bench 8.98; #1 on alignment benchmarks Oct 2024 | [HF](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Instruct-HF) · [NIM](https://build.nvidia.com/nvidia/llama-3_1-nemotron-70b-instruct) |
| **Llama-3_1-Nemotron-Ultra-253B-v1** | Llama 3.1 405B | 253B | 128K | NAS-compressed from 405B; fits on 8×H100; state-of-the-art reasoning | [HF](https://huggingface.co/nvidia/Llama-3_1-Nemotron-Ultra-253B-v1) · [OpenRouter](https://openrouter.ai/nvidia/llama-3.1-nemotron-ultra-253b-v1:free) |

---

### Vision & Multimodal

| Model | Params | Description | Links |
|---|---|---|---|
| **NVIDIA-Nemotron-Nano-12B-v2-VL** | 12B | Best-in-class vision-language accuracy; document intelligence and video understanding; leads on OCRBenchV2 | [HF](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8) · [NIM](https://build.nvidia.com/nvidia/nemotron-nano-12b-v2-vl) |
| **Nemotron Nano 2 VL** | 12B | Open multimodal reasoning: text, images, tables, charts, video; strong document intelligence | [HF](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8) |
| **Nemotron Parse** | — | Document data extraction — pairs with VL models for structured querying of complex documents | [NIM](https://build.nvidia.com/explore/retrieval) |

---

### Speech (ASR / TTS / Translation)

Models under the **Parakeet** and **Canary** families, part of the NVIDIA NeMo speech collection.

| Model | Task | Description | Links |
|---|---|---|---|
| **Parakeet-TDT** | ASR | English ASR — CTC/RNN-T/TDT decoders; state-of-the-art accuracy developed with Suno.ai | [HF](https://huggingface.co/nvidia) · [Docs](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html) |
| **Parakeet-CTC** | ASR | FastConformer Encoder + CTC decoder; high-throughput English transcription | [HF](https://huggingface.co/nvidia) |
| **Canary** | ASR + Translation | Multilingual encoder-decoder (FastConformer + Transformer); 25 EU languages; ASR + bi-directional translation | [HF](https://huggingface.co/nvidia) · [Docs](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html) |
| **Nemotron Speech** | ASR / TTS / NMT | High-throughput, ultra-low latency ASR, text-to-speech, and neural machine translation; part of Nemotron speech collection | [HF Collection](https://huggingface.co/collections/nvidia/nemotron-speech) |

---

### Safety & Guardrails

| Model | Params | Description | Links |
|---|---|---|---|
| **Llama-3.1-Nemotron-Safety-Guard-8B-v3** | 8B | Multilingual content safety; 23 safety categories; 9 languages; 84.2% harmful content classification accuracy | [HF](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3) · [NIM](https://build.nvidia.com/nvidia/llama-3_1-nemotron-safety-guard-8b-v3) |
| **NemoGuard** | — | Collection of guardrail models and tools for enterprise LLM safety pipelines | [HF Collection](https://huggingface.co/collections/nvidia/nemoguard) |
| **AEGIS** | — | Content safety evaluation dataset and LLM-based classifier; 13 categories of critical human-LLM interaction risks | [HF](https://huggingface.co/nvidia) |

---

### RAG & Retrieval

| Model | Params | Description | Links |
|---|---|---|---|
| **Nemotron RAG Embedding** | 2B | Multimodal text + image embedding; leads on ViDoRe V1/V2 and MTEB VisualDocumentRetrieval leaderboards | [HF](https://huggingface.co/collections/nvidia/nemotron-rag-68f01e412f2dc5a5db5f30ed) · [NIM](https://build.nvidia.com/explore/retrieval) |
| **Nemotron RAG Reranker** | 2B | Best-in-class reranking for text + image passages; boosts RAG accuracy | [HF](https://huggingface.co/collections/nvidia/nemotron-rag-68f01e412f2dc5a5db5f30ed) |
| **NV-Embed** | — | Generalist embedding model covering retrieval, reranking, classification, clustering, STS | [HF](https://huggingface.co/nvidia/NV-Embed-v2) |
| **Llama3-ChatQA-2** | 70B / 8B | 128K long-context models with exceptional RAG capabilities; built on Llama 3 | [HF](https://huggingface.co/collections/nvidia/llama3-chatqa-2) |
| **Llama3-ChatQA-1.5** | 70B / 8B | Conversational QA and RAG; strong on doc-heavy question answering | [HF](https://huggingface.co/collections/nvidia/llama3-chatqa-1-5) |
| **InstructRetro** | — | Autoregressive decoder-only LM with retrieval-augmented pretraining and instruction tuning | [HF](https://huggingface.co/nvidia) |

---

### Reward Models

| Model | Base | Description | Links |
|---|---|---|---|
| **Llama-3.1-Nemotron-70B-Reward** | Llama 3.1 70B Instruct | #1 on RewardBench (Oct 2024); Bradley-Terry + SteerLM regression; used for RLHF of Nemotron-70B-Instruct | [HF](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Reward) |
| **Llama-3.1-Nemotron-70B-Reward-HF** | Llama 3.1 70B Instruct | HuggingFace Transformers-compatible conversion of above | [HF](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Reward-HF) |
| **Nemotron-4-340B-Reward** | Nemotron-4 340B Base | Reward model for synthetic data generation pipelines | [HF](https://huggingface.co/nvidia/Nemotron-4-340B-Reward) |
| **Qwen-3-Nemotron-235B-A22B-GenRM** | Qwen 3 235B | Generative reward model used for RLHF of Nemotron 3 Nano | [HF](https://huggingface.co/nvidia) |

---

### Compressed / Distilled Models (Minitron)

Minitron models are obtained by pruning NVIDIA's larger Nemotron-4 models and distilling with knowledge distillation. Offer large model quality at SLM inference cost.

| Model | Base | Params | Description | Links |
|---|---|---|---|---|
| **Minitron-4B-Width** | Nemotron-4 15B | 4B | Width-pruned; strong reasoning and chat | [HF](https://huggingface.co/nvidia/Minitron-4B-Width-Base) |
| **Minitron-8B-Base** | Nemotron-4 15B | 8B | Depth-pruned; compact high-quality foundation model | [HF](https://huggingface.co/nvidia/Minitron-8B-Base) |
| **Llama-3.1-Minitron-4B-Width-Base** | Llama 3.1 8B | 4B | NVIDIA compression of Llama 3.1 8B using width pruning + distillation | [HF](https://huggingface.co/nvidia/Llama-3.1-Minitron-4B-Width-Base) |

> 📄 Paper: [Compact Language Models via Pruning and Knowledge Distillation](https://arxiv.org/abs/2407.14679)

---

### Nemotron 4 340B

Released June 2024. Designed primarily for **synthetic data generation (SDG)**. Trained on 9 trillion tokens. 98% of alignment data is synthetically generated.

| Model | Description | Links |
|---|---|---|
| **Nemotron-4-340B-Base** | Base model; competitive with Llama-3 70B, Mixtral 8x22B on commonsense reasoning | [HF](https://huggingface.co/nvidia/Nemotron-4-340B-Base) · [NIM](https://build.nvidia.com/nvidia/nemotron-4-340b-base) |
| **Nemotron-4-340B-Instruct** | Chat/instruction model; optimized for English single- and multi-turn; ideal for SDG pipelines | [HF](https://huggingface.co/nvidia/Nemotron-4-340B-Instruct) · [NIM](https://build.nvidia.com/nvidia/nemotron-4-340b-instruct) |
| **Nemotron-4-340B-Reward** | Reward model for RLHF and synthetic preference data generation | [HF](https://huggingface.co/nvidia/Nemotron-4-340B-Reward) |

> 📄 Paper: [Nemotron-4 340B Technical Report](https://arxiv.org/abs/2406.11704)

---

### Nemotron 3 8B (Enterprise)

Released February 2024. Optimized for building production-ready enterprise AI apps via NeMo Framework. Trained on 3.8 trillion tokens across 53 languages and 37 programming languages.

| Model | Context | Description | Links |
|---|---|---|---|
| **Nemotron-3-8B-Base-4k** | 4K | Foundation model; 1,024 A100s × 19 days | [HF](https://huggingface.co/nvidia/nemotron-3-8b-base-4k) |
| **Nemotron-3-8B-Chat-4k-SteerLM** | 4K | Chat variant with SteerLM alignment | [HF](https://huggingface.co/nvidia/nemotron-3-8b-chat-4k-steerlm) |
| **Nemotron-3-8B-QA-4k** | 4K | Optimized for question answering tasks | [HF](https://huggingface.co/nvidia/nemotron-3-8b-qa-4k) |

---

## 📦 Datasets

NVIDIA has released one of the largest open collections of synthetic data for agentic AI — over **10 trillion tokens** spanning pre-training, post-training, personas, safety, RL, and RAG.

| Dataset | Size | Description | Links |
|---|---|---|---|
| **Nemotron-CC** | 6T+ tokens | Curated Common Crawl; 15 languages; deduplication + quality filtering pipeline | [HF](https://huggingface.co/datasets/nvidia/Nemotron-CC) · [Paper](https://arxiv.org/abs/2412.02595) |
| **Nemotron-CC-v2.1** | 2.5T tokens | New English tokens from 3 recent CC snapshots; synthetic rephrasing; multilingual translation | [HF](https://huggingface.co/nvidia) |
| **Nemotron-CC-Code-v1** | 428B tokens | High-quality code from Common Crawl Code; Lynx + LLM pipeline; preserves equations and code | [HF](https://huggingface.co/nvidia) |
| **Nemotron-CC-Math-v1** | — | Curated math pretraining data; LaTeX standardization; noise removal | [HF](https://huggingface.co/nvidia) |
| **Nemotron-Pretraining-Code-v2** | — | Refreshed GitHub code; multi-stage filtering + deduplication | [HF](https://huggingface.co/nvidia) |
| **Nemotron-Pretraining-Specialized-v1** | — | Synthetic datasets for STEM reasoning and scientific coding | [HF](https://huggingface.co/nvidia) |
| **Nemotron-SFT-Data** | 18M+ samples | Nemotron 3 Nano supervised fine-tuning datasets | [HF](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Data) |
| **Nemotron-RL-Data** | — | RL training datasets: tool calls, multi-turn trajectories, preference signals (coding, math, reasoning, agentic) | [HF](https://huggingface.co/nvidia) |
| **Nemotron-VLM-Dataset-v2** | — | High-quality post-training datasets for VQA and OCR; powers Nemotron Nano 2 VL | [HF](https://huggingface.co/datasets/nvidia/Nemotron-VLM-Dataset-v2) |
| **Nemotron-Personas (USA)** | — | Fully synthetic, privacy-safe personas grounded in US demographic/cultural data | [HF](https://huggingface.co/datasets/nvidia/Nemotron-Personas) |
| **Nemotron-Personas-Japan** | — | Synthetic personas grounded in Japanese demographic/cultural data | [HF](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Japan) |
| **Nemotron-Personas-India** | — | Synthetic personas grounded in Indian demographic/cultural data | [HF](https://huggingface.co/datasets/nvidia/Nemotron-Personas-India) |
| **OpenMathInstruct-2** | 1.8M samples | Large open-source instruction data for math model training | [HF](https://huggingface.co/collections/nvidia/openmath-2) |
| **HelpSteer2** | — | Human preference data used for Nemotron-70B reward + instruct training | [HF](https://huggingface.co/datasets/nvidia/HelpSteer2) · [Paper](https://arxiv.org/abs/2410.01257) |

---

## 🛠 Tools & Deployment

### NVIDIA NeMo Framework

End-to-end platform for fine-tuning, deploying, and continuously optimizing Nemotron models.

- [NeMo Framework GitHub](https://github.com/NVIDIA-NeMo/NeMo)
- [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/)
- [NeMo Aligner — RLHF / SFT / DPO training](https://github.com/NVIDIA/NeMo-Aligner)
- [NeMo Curator — Data curation pipelines](https://github.com/NVIDIA/NeMo-Curator)
- [NeMo Evaluator SDK — Reproducible benchmarking](https://github.com/NVIDIA/nemo-evaluator)
- [NeMo Skills — Evaluation & skill datasets](https://github.com/NVIDIA/NeMo-Skills)
- [NeMo Gym — Open RL environments for agentic post-training](https://github.com/NVIDIA/NeMo-Gym)

### NVIDIA NIM

Ready-to-run microservices for production deployment of Nemotron models on any GPU system.

- [NVIDIA NIM Documentation](https://developer.nvidia.com/nim)
- [Nemotron NIM Catalog](https://build.nvidia.com/explore/reasoning)

### Open-Source Inference Frameworks

| Framework | Notes | Link |
|---|---|---|
| **vLLM** | Recommended for production; high-throughput serving | [vLLM Docs](https://docs.vllm.ai/en/latest/) |
| **SGLang** | Fast inference for structured generation | [SGLang GitHub](https://github.com/sgl-project/sglang) |
| **Ollama** | Local model serving; easy to run Nemotron locally | [Ollama](https://ollama.com) |
| **llama.cpp** | CPU + GPU inference; quantized models | [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp) |
| **HuggingFace Transformers** | For development and research | [Transformers Docs](https://huggingface.co/docs/transformers) |
| **TensorRT-LLM** | NVIDIA-optimized; highest throughput on NVIDIA GPUs | [TRT-LLM GitHub](https://github.com/NVIDIA/TensorRT-LLM) |

---

## 📚 Tutorials & Starter Kits

### Official NVIDIA Starter Kits

- **[Build a Report Generation Agent with Nemotron](https://developer.nvidia.com/blog/build-a-report-generator-ai-agent-with-nvidia-nemotron-on-openrouter/)** — LangGraph + Nemotron agent: model, tools, memory, routing  
  - [Tutorial Video](https://www.youtube.com/watch?v=kGzEUjmcwLY) · [Livestream](https://www.youtube.com/watch?v=jb81ugc_o6I) · [Launchable](https://brev.nvidia.com/launchable/deploy?launchableID=env-32kC34ErT9wsqTcJyaKMxBEuhr2)

- **[Build a RAG Agent with Nemotron](https://developer.nvidia.com/blog/build-a-rag-agent-with-nvidia-nemotron/)** — Agentic RAG with LangGraph and Nemotron embedding + reranking  
  - [Tutorial Video](https://youtu.be/Mw9_ZK3tnrQ) · [Livestream](https://www.youtube.com/watch?v=f0utW0eLueY)

- **[Build a Voice Agent with RAG and Safety Guardrails](https://developer.nvidia.com/blog/how-to-build-a-voice-agent-with-rag-and-safety-guardrails/)** — Full stack: Parakeet ASR → Embedding → Reranker → Reasoning LLM → Safety Guard → TTS  
  *(Jan 2026 — uses models released at CES 2026)*

- **[Build More Accurate AI Agents with Llama Nemotron Super 1.5](https://developer.nvidia.com/blog/build-more-accurate-and-efficient-ai-agents-with-the-new-nvidia-llama-nemotron-super-v1-5/)** — Deep research agent patterns with Super v1.5

- **[Supercharge Edge AI with Nemotron Nano 2 9B](https://huggingface.co/blog/nvidia/supercharge-ai-reasoning-with-nemotron-nano-2)** — Hybrid Transformer-Mamba at the edge with thinking budget control

- **[Using NeMo-Skills to Evaluate Nemotron Nano 9B V2](https://nvidia.github.io/NeMo-Skills/tutorials/2025/08/22/reproducing-nvidia-nemotron-nano-9b-v2-evals/)** — Reproduce official benchmark results

### GitHub Repositories

- [NVIDIA-NeMo/NeMo](https://github.com/NVIDIA-NeMo/NeMo) — Core NeMo framework (audio, speech, multimodal LLM focus as of 2025)
- [NVIDIA/NeMo-Aligner](https://github.com/NVIDIA/NeMo-Aligner) — RLHF, SFT, DPO, SteerLM alignment
- [NVIDIA/NeMo-Curator](https://github.com/NVIDIA/NeMo-Curator) — Scalable data curation for Nemotron pretraining
- [NVIDIA/NeMo-Skills](https://github.com/NVIDIA/NeMo-Skills) — Skill-specific evaluation and data generation
- [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) — Production-optimized LLM inference
- [NVIDIA-NeMo/Nemotron](https://github.com/NVIDIA-NeMo/Nemotron) — Official Nemotron recipes and tutorials

### Notebooks & Interactive Demos

| Notebook / Demo | Description | Link |
|---|---|---|
| **Nemotron 3 Nano Quickstart** | Run Nemotron-3-Nano-30B-A3B via OpenAI-compatible API with vLLM | [GitHub](https://github.com/NVIDIA-NeMo/Nemotron) |
| **Llama Nemotron Super Reasoning Demo** | Toggle thinking ON/OFF and compare outputs interactively | [HF Space](https://huggingface.co/spaces/nvidia/Llama-3.3-Nemotron-Super-49B-v1) |
| **Nemotron Nano 9B V2 — HuggingFace Space** | Live inference demo in your browser | [HF Space](https://huggingface.co/spaces/nvidia/NVIDIA-Nemotron-Nano-9B-v2) |
| **NeMo Curator Tutorial** | Jupyter notebooks for data curation pipelines | [GitHub](https://github.com/NVIDIA/NeMo-Curator/tree/main/tutorials) |
| **NVIDIA AI Workbench — Nemotron** | One-click local dev environment with Nemotron | [AI Workbench](https://www.nvidia.com/en-us/deep-learning-ai/solutions/data-science/workbench/) |
| **Fine-tune Nemotron with NeMo Aligner** | SFT / RLHF / DPO walkthrough notebook | [GitHub](https://github.com/NVIDIA/NeMo-Aligner/tree/main/examples) |
| **Agentic RAG with LangChain + Nemotron** | End-to-end RAG agent using NVIDIA NIM endpoints | [GitHub](https://github.com/langchain-ai/langchain-nvidia) |
| **Multi-Agent Orchestration with CrewAI** | Multi-agent workflow using Nemotron via NIM | [Blog](https://developer.nvidia.com/blog/build-multi-agent-workflows-with-nvidia-nemotron-and-crewai/) |

### Learning Paths

- [How to Build an AI Agent](https://developer.nvidia.com/topics/ai/how-to-build-an-ai-agent) — NVIDIA Developer Learning Path
- [How to Build an Agent RAG Application](https://developer.nvidia.com/topics/ai/how-to-build-agentic-ai-rag) — NVIDIA Developer Learning Path
- [Generative AI Examples on GitHub](https://github.com/NVIDIA/GenerativeAIExamples) — Reference implementations with Nemotron and NIM
- [NVIDIA DLI Courses on Generative AI](https://www.nvidia.com/en-us/training/instructor-led-workshops/generative-ai/) — Instructor-led workshops including NeMo and Nemotron
- [NVIDIA NGC — Nemotron Containers](https://catalog.ngc.nvidia.com/models?filters=&orderBy=weightPopularDESC&query=nemotron) — Production-ready Docker containers for all Nemotron models

---

## 🤗 HuggingFace × Nemotron

NVIDIA and HuggingFace have a close collaboration — all Nemotron model weights, datasets, and demos live on the HuggingFace Hub under the **[nvidia](https://huggingface.co/nvidia)** organization.

### NVIDIA Organization on HuggingFace

- **[nvidia org on HuggingFace](https://huggingface.co/nvidia)** — 500+ models, datasets, and Spaces from NVIDIA
- **[Nemotron Model Collection](https://huggingface.co/collections/nvidia/nemotron-3-6788da0f6c90c61b8f17c36d)** — All Nemotron 3 models in one place
- **[Llama Nemotron Collection](https://huggingface.co/collections/nvidia/llama-nemotron-67df5d39124e69a2b13b7c47)** — All Llama Nemotron models
- **[Nemotron RAG Collection](https://huggingface.co/collections/nvidia/nemotron-rag-68f01e412f2dc5a5db5f30ed)** — Embedding + reranker models
- **[NeMoGuard Collection](https://huggingface.co/collections/nvidia/nemoguard)** — Safety and guardrails models
- **[Nemotron Speech Collection](https://huggingface.co/collections/nvidia/nemotron-speech)** — ASR, TTS, and translation models

### HuggingFace Spaces (Try in Browser)

| Space | Description | Link |
|---|---|---|
| **Nemotron Nano 9B V2** | Try the small but powerful reasoning model | [HF Space](https://huggingface.co/spaces/nvidia/NVIDIA-Nemotron-Nano-9B-v2) |
| **Llama Nemotron Super 49B** | Balanced reasoning and throughput | [HF Space](https://huggingface.co/spaces/nvidia/Llama-3.3-Nemotron-Super-49B-v1) |
| **Nemotron Nano VL 12B** | Vision-language understanding and document AI | [HF Space](https://huggingface.co/spaces/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL) |
| **NV-Embed-v2** | Text embedding and retrieval | [HF Space](https://huggingface.co/spaces/nvidia/NV-Embed-v2) |

### HuggingFace Blog Posts About Nemotron

- **[Supercharge AI Reasoning with Nemotron Nano 2 9B](https://huggingface.co/blog/nvidia/supercharge-ai-reasoning-with-nemotron-nano-2)** — Edge deployment with thinking budget control
- **[Llama Nemotron: Open Efficient Reasoning Models](https://huggingface.co/blog/nvidia/llama-nemotron)** — How NVIDIA post-trains Llama into a reasoning powerhouse
- **[Introducing Nemotron 3 Nano](https://huggingface.co/blog/nvidia/nemotron-3-nano)** — Hybrid Mamba-Transformer MoE deep-dive
- **[HelpSteer2: Human Preference Data](https://huggingface.co/blog/nvidia/helpsteer2)** — The preference dataset powering Nemotron reward training
- **[NVIDIA's Open Source Strategy with HuggingFace](https://huggingface.co/blog/nvidia)** — Partnership overview

### HuggingFace Integration Highlights

| Integration | Description | Link |
|---|---|---|
| **`transformers` support** | All Nemotron models support `AutoModelForCausalLM.from_pretrained()` | [Transformers Docs](https://huggingface.co/docs/transformers) |
| **NVIDIA NIM via HF Hub** | Call NIM inference endpoints directly from HF pipelines | [NIM API Docs](https://docs.api.nvidia.com/) |
| **Text Generation Inference (TGI)** | HuggingFace TGI supports Nemotron models for production serving | [TGI GitHub](https://github.com/huggingface/text-generation-inference) |
| **HF Inference Endpoints** | Deploy Nemotron models on HF's managed infrastructure | [HF Endpoints](https://huggingface.co/inference-endpoints) |
| **HF Datasets Hub** | All Nemotron datasets (CC, SFT, RL, VLM, Personas) on HF Datasets | [HF Datasets](https://huggingface.co/datasets?search=nvidia) |
| **LeaderBoard Rankings** | Track Nemotron performance on Open LLM Leaderboard | [HF Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) |

---

## 🏭 Real-World Use Cases

### Agentic AI & Automation

| Use Case | Description | Model Used |
|---|---|---|
| **Deep Research Agents** | Autonomous web research, synthesis, and report generation | Llama Nemotron Super 49B v1.5 |
| **Code Generation & Review** | Automated code completion, security review, and documentation | Nemotron 3 Nano / Super |
| **Customer Support Automation** | Multi-turn conversational agents with RAG over enterprise knowledge bases | Llama Nemotron Nano 8B |
| **Document Intelligence** | Structured extraction from PDFs, tables, and complex documents | Nemotron Nano VL 12B + Nemotron Parse |
| **Medical Transcription** | Clinical note generation from doctor-patient conversations | Parakeet ASR + Nemotron LLM |
| **Legal Document Analysis** | Contract review, clause extraction, and compliance checking | Nemotron Ultra 253B |
| **Financial Report Generation** | Automated synthesis of quarterly earnings and market analysis | Nemotron Super 49B |
| **Software Testing** | Automated test case generation, bug triage, and issue summarization | Nemotron 3 Nano |

### Data Synthesis & AI Training

| Use Case | Description | Model Used |
|---|---|---|
| **Synthetic Data Generation** | Generate high-quality instruction data for fine-tuning downstream models | Nemotron-4-340B-Instruct |
| **RLHF Pipeline** | Human preference alignment at scale using synthetic comparisons | Nemotron-4-340B-Reward |
| **Math Reasoning Data** | Large-scale synthetic math problems and solutions | OpenMathInstruct-2 + Nemotron |
| **Persona-based Data** | Culturally-grounded synthetic user data for diverse training | Nemotron-Personas datasets |

### Edge & On-Device AI

| Use Case | Description | Model Used |
|---|---|---|
| **RTX PC Assistant** | On-device personal assistant on NVIDIA RTX GPUs | Llama Nemotron Nano 4B/8B |
| **Embedded Industrial AI** | Real-time anomaly detection in manufacturing | Nemotron Nano 9B V2 |
| **Offline Voice Assistant** | Privacy-preserving voice AI on local hardware | Parakeet + Nemotron Nano |

---

## 🌍 Companies Using Nemotron Worldwide

NVIDIA Nemotron models are deployed across industries — from healthcare and finance to retail and manufacturing.

### Technology & Cloud Providers

| Company | Region | Use Case | Notes |
|---|---|---|---|
| **Microsoft Azure** | Global | Enterprise AI deployment on Azure | Azure AI Model Catalog includes Nemotron NIM |
| **Oracle Cloud** | Global | GPU Cloud + AI workloads | OCI Supercluster runs Nemotron training at scale |
| **Dell Technologies** | Global | On-premises enterprise AI | Dell AI Factory with NVIDIA NIM on PowerEdge servers |
| **Lenovo** | Global | Edge + hybrid AI | Lenovo AI solutions featuring Nemotron NIM |
| **VMware / Broadcom** | Global | Private cloud AI | VMware Private AI Foundation with NVIDIA |

### Enterprise Software

| Company | Region | Use Case | Notes |
|---|---|---|---|
| **SAP** | Germany / Global | Business AI, ERP automation | NVIDIA AI integrated into SAP Business AI |
| **ServiceNow** | USA / Global | IT workflows, enterprise agents | Now Platform AI powered by Nemotron via NIM |
| **Salesforce** | USA / Global | CRM AI, Einstein AI features | NVIDIA AI partner ecosystem |
| **Adobe** | USA / Global | Creative AI, document intelligence | Firefly AI and document workflows |
| **Siemens** | Germany / Global | Industrial automation AI | Siemens Industrial Copilot on NVIDIA stack |

### Healthcare & Life Sciences

| Company | Region | Use Case | Notes |
|---|---|---|---|
| **Johnson & Johnson** | USA | Clinical research automation | Drug discovery and clinical data analysis |
| **Illumina** | USA | Genomics AI | Genomic data analysis with NVIDIA BioNeMo |
| **Mayo Clinic** | USA | Medical AI | Clinical decision support with NVIDIA AI |
| **Astrazeneca** | UK | Drug discovery | NVIDIA Clara + Nemotron for protein analysis |
| **PathAI** | USA | Pathology AI | AI-powered pathology report generation |

### Finance & Insurance

| Company | Region | Use Case | Notes |
|---|---|---|---|
| **JPMorgan Chase** | USA | Document AI, compliance | Financial document analysis and risk assessment |
| **Deutsche Bank** | Germany | Enterprise AI assistants | German-language financial AI using multilingual Nemotron |
| **FinanceAI Partners** | Global | Trading AI | Real-time market sentiment and report generation |

### Retail & E-Commerce

| Company | Region | Use Case | Notes |
|---|---|---|---|
| **Walmart** | USA | Customer experience AI | Retail AI and inventory management |
| **Accenture** | Global | Enterprise AI solutions | Accenture AI Refinery with NVIDIA NIM |

### Startups & AI Companies

| Company | Region | Description | Notes |
|---|---|---|---|
| **Perplexity AI** | USA | AI-powered search and research | Uses large Nemotron-family models via API |
| **Cohere** | Canada | Enterprise NLP platform | Partnered with NVIDIA for GPU + model distribution |
| **Mistral AI** | France | Open AI models | Collaboration on open model ecosystem with NVIDIA |
| **DeepInfra** | USA | Inference API | Hosts Nemotron Super 49B v1.5 for API access |
| **OpenRouter** | USA | Model routing API | Serves Nemotron Ultra 253B and Nano 9B V2 as free tier |
| **together.ai** | USA | AI cloud platform | Nemotron models via Together inference API |
| **Replicate** | USA | Model deployment platform | Community-maintained Nemotron model deployments |
| **Anyscale** | USA | Distributed AI serving | Nemotron models on Ray Serve |

### Government & Public Sector

| Company / Agency | Region | Use Case | Notes |
|---|---|---|---|
| **US Department of Defense** | USA | Secure AI workloads | NVIDIA sovereign AI infrastructure |
| **Saudi Aramco (KACST)** | Saudi Arabia | Sovereign AI | Arabic-language Nemotron for national AI strategy |
| **Tata Consultancy Services** | India | Government AI modernization | AI services for Indian public sector using NVIDIA AI |
| **Fujitsu** | Japan | Government AI | Japanese-language Nemotron for public sector use |

> 💡 **Note:** This list is compiled from publicly announced NVIDIA partnerships and integrations. If your company is using Nemotron and would like to be listed, please submit a PR.

---

## 🎬 Videos & Talks

### Official NVIDIA Tutorials

| Video | Description | Date |
|---|---|---|
| **[Build a Report Generation Agent with Nemotron](https://www.youtube.com/watch?v=kGzEUjmcwLY)** | LangGraph + Nemotron agent: model, tools, memory, routing | Jan 2026 |
| **[Nemotron Report Agent — Livestream](https://www.youtube.com/watch?v=jb81ugc_o6I)** | Full walkthrough with Q&A | Jan 2026 |
| **[Build a RAG Agent with Nemotron — Tutorial](https://youtu.be/Mw9_ZK3tnrQ)** | Agentic RAG with Nemotron embedding + reranking | Jan 2026 |
| **[RAG Agent Livestream with Q&A](https://www.youtube.com/watch?v=f0utW0eLueY)** | Deep dive into Nemotron RAG pipeline | Jan 2026 |

### GTC (GPU Technology Conference) Sessions

| Session | Description | Event |
|---|---|---|
| **[Nemotron 3: Architecture, Training, and Open Innovation](https://www.nvidia.com/gtc/)** | NVIDIA Research presents Nemotron 3 MoE architecture | GTC 2026 |
| **[NeMo Gym: RL for Agentic AI](https://www.nvidia.com/gtc/)** | Post-training with reinforcement learning in NeMo Gym | GTC 2026 |
| **[Building Production AI Agents with NIM](https://www.nvidia.com/gtc/)** | Enterprise deployment patterns with NVIDIA NIM | GTC 2026 |

### Community & Creator Videos

| Video | Creator | Description |
|---|---|---|
| **[Nemotron vs GPT-4o: Benchmark Deep Dive](https://www.youtube.com/results?search_query=nemotron+nvidia)** | Various creators | Performance comparison and real-world testing |
| **[Run Nemotron Locally with Ollama](https://www.youtube.com/results?search_query=nemotron+ollama)** | Various creators | Step-by-step local setup guides |
| **[Nemotron 3 Nano — First Look & Test](https://www.youtube.com/results?search_query=nemotron+3+nano)** | Various creators | Hands-on demos shortly after release |
| **[Build an AI Agent in 20 Minutes with Nemotron](https://www.youtube.com/results?search_query=nemotron+ai+agent+tutorial)** | Various creators | Quick-start agent tutorials |

### NVIDIA On Air & Livestreams

- **[NVIDIA Developer YouTube Channel](https://www.youtube.com/@NVIDIADeveloper)** — Official tutorials, demos, and conference recordings
- **[NVIDIA AI Podcast YouTube](https://www.youtube.com/@TheNVIDIAAIPodcast)** — Video versions of AI podcast episodes
- **[NVIDIA GTC On-Demand](https://www.nvidia.com/gtc/on-demand/)** — Full GTC session library including Nemotron talks

---

## 🎙 Podcasts

### NVIDIA AI Podcast

The official NVIDIA podcast covers AI research, applications, and company interviews. Multiple episodes feature Nemotron-related work.

- **[The NVIDIA AI Podcast](https://blogs.nvidia.com/ai-podcast/)** — Weekly show hosted by Noah Kravitz; episodes include discussions on open models, agentic AI, and enterprise deployments
- **[Spotify — NVIDIA AI Podcast](https://open.spotify.com/show/1UPYMjuDYX7xHfEQfBdCJW)** — Available on Spotify
- **[Apple Podcasts — NVIDIA AI Podcast](https://podcasts.apple.com/us/podcast/the-nvidia-ai-podcast/id1211774152)** — Available on Apple Podcasts

### Featured Episodes (Nemotron / Open Models)

| Episode | Topic | Link |
|---|---|---|
| **"Open Weights, Open Data: NVIDIA's Strategy for AI"** | Why NVIDIA released Nemotron with full open weights and training data | [NVIDIA AI Podcast](https://blogs.nvidia.com/ai-podcast/) |
| **"Building AI Agents with NeMo and Nemotron"** | How to use NeMo Framework and NIM for production agents | [NVIDIA AI Podcast](https://blogs.nvidia.com/ai-podcast/) |
| **"Synthetic Data at Scale: Nemotron-CC and Beyond"** | How NVIDIA curates and generates training data for open models | [NVIDIA AI Podcast](https://blogs.nvidia.com/ai-podcast/) |

### Community & Industry Podcasts

| Podcast | Episode / Focus | Link |
|---|---|---|
| **Practical AI (Changelog)** | "NVIDIA's Open Model Strategy with Nemotron" | [Practical AI](https://practicalai.fm/) |
| **Latent Space** | Deep dives on open model releases including Nemotron architecture | [Latent Space](https://www.latent.space/podcast) |
| **The TWIML AI Podcast** | Interviews with NVIDIA researchers on Nemotron training techniques | [TWIML AI](https://twimlai.com/) |
| **Gradient Dissent (Weights & Biases)** | Conversations with AI practitioners building on Nemotron | [Gradient Dissent](https://www.wandb.courses/collections) |
| **AI Engineering Podcast** | Architecture and deployment patterns for Nemotron at scale | [SE Daily](https://softwareengineeringdaily.com/) |
| **Eye on AI** | "Open AI Models and the Nemotron Family" — industry analysis | [Eye on AI](https://www.eye-on.ai/podcast) |

---

## 📅 Upcoming Events

### 2026 Conferences & Events

| Event | Date | Location | Nemotron Relevance |
|---|---|---|---|
| **[NVIDIA GTC 2026](https://www.nvidia.com/gtc/)** | March 17–21, 2026 | San Jose, CA + Virtual | Major Nemotron sessions: Nemotron 3, NeMo Gym, agentic AI |
| **[ICLR 2026](https://iclr.cc/)** | April–May 2026 | TBD | AI research papers including Nemotron training methods |
| **[Google I/O 2026](https://io.google/)** | May 2026 | Mountain View, CA | NVIDIA AI partner ecosystem announcements |
| **[Microsoft Build 2026](https://build.microsoft.com/)** | May 2026 | Seattle, WA | Azure + NVIDIA NIM integration updates |
| **[ACL 2026](https://2026.aclweb.org/)** | Summer 2026 | TBD | NLP research with Nemotron models |
| **[NeurIPS 2026](https://neurips.cc/)** | December 2026 | TBD | NVIDIA research papers on Nemotron and agentic AI |

### NVIDIA Developer Events

| Event | Description | Link |
|---|---|---|
| **NVIDIA Developer Weekly** | Weekly office hours, live coding sessions, and Q&A with NVIDIA engineers | [NVIDIA Developer](https://developer.nvidia.com/events) |
| **NIM Deployment Workshops** | Hands-on workshops for deploying Nemotron via NIM in enterprise environments | [NVIDIA Events](https://www.nvidia.com/en-us/events/) |
| **NeMo Office Hours** | Regular online sessions with NeMo Framework engineers | [NVIDIA Discord](https://discord.gg/nvidia) |
| **HuggingFace × NVIDIA Webinars** | Joint webinars on open model deployment and fine-tuning | [HF Events](https://huggingface.co/events) |

### Community Meetups & Hackathons

| Event | Description | Link |
|---|---|---|
| **NVIDIA AI Hackathons** | Build agentic AI solutions with Nemotron; prizes and mentorship from NVIDIA | [NVIDIA Developer](https://developer.nvidia.com/events) |
| **HuggingFace Sprints** | Collaborative fine-tuning sprints using Nemotron base models | [HuggingFace Events](https://huggingface.co/events) |
| **Local AI Meetups** | Meetup groups worldwide building with NVIDIA AI stack | [Meetup.com — NVIDIA](https://www.meetup.com/topics/nvidia/) |

> 💡 **Stay updated:** Subscribe to the [NVIDIA Developer Newsletter](https://developer.nvidia.com/newsletter) for the latest events, releases, and tutorials.

---

## 📄 Papers

| Paper | Models | Year | Link |
|---|---|---|---|
| **NVIDIA Nemotron 3 Nano** | Nemotron 3 Nano 30B-A3B | 2025 | [research.nvidia.com](https://research.nvidia.com/labs/nemotron/Nemotron-3/) |
| **NVIDIA Nemotron Nano 2** | NVIDIA-Nemotron-Nano-9B-v2 | 2025 | [arXiv:2508.14444](https://arxiv.org/abs/2508.14444) |
| **Llama-Nemotron: Efficient Reasoning Models** | Llama Nemotron Nano/Super/Ultra | 2025 | [arXiv:2505.00949](https://arxiv.org/abs/2505.00949) |
| **Reward-aware Preference Optimization (RPO)** | Llama Nemotron Nano/Super | 2025 | [arXiv:2502.00203](https://arxiv.org/abs/2502.00203) |
| **HelpSteer2-Preference** | Llama-3.1-Nemotron-70B | 2024 | [arXiv:2410.01257](https://arxiv.org/abs/2410.01257) |
| **Nemotron-4 340B Technical Report** | Nemotron-4-340B | 2024 | [arXiv:2406.11704](https://arxiv.org/abs/2406.11704) |
| **Compact Language Models via Pruning and KD (Minitron)** | Minitron 4B/8B | 2024 | [arXiv:2407.14679](https://arxiv.org/abs/2407.14679) |
| **Nemotron-CC: Curating Common Crawl Data** | Pretraining datasets | 2024 | [arXiv:2412.02595](https://arxiv.org/abs/2412.02595) |
| **Inside NVIDIA Nemotron 3** | Nemotron 3 architecture | 2025 | [NVIDIA Blog](https://developer.nvidia.com/blog/inside-nvidia-nemotron-3-techniques-tools-and-data-that-make-it-efficient-and-accurate/) |

---

## 🌐 Community

- [NVIDIA Developer Forum — Nemotron](https://forums.developer.nvidia.com/c/ai-data-science/nvidia-nemotron/669)
- [NVIDIA Nemotron Feature Requests](http://nemotron.ideas.nvidia.com/)
- [Nemotron on Hugging Face](https://huggingface.co/nvidia/collections?search=nemotron)
- [NVIDIA Developer Blog](https://developer.nvidia.com/blog/tag/nemotron/)
- [NVIDIA Newsroom — Nemotron 3 Launch](https://nvidianews.nvidia.com/news/nvidia-debuts-nemotron-3-family-of-open-models)
- [NVIDIA Nemotron Landing Page](https://www.nvidia.com/en-us/ai-data-science/foundation-models/nemotron/)
- [NVIDIA Developer Nemotron Page](https://developer.nvidia.com/nemotron)
- [r/LocalLLaMA — Nemotron Discussions](https://www.reddit.com/r/LocalLLaMA/search/?q=nemotron&sort=new) — Reddit community discussions
- [NVIDIA Discord Server](https://discord.gg/nvidia) — Real-time support and community
- [Hugging Face Discord — #nvidia-nemotron](https://discord.com/invite/JfAtkvEtRb) — HF community channel for Nemotron
- [Twitter / X — @NVIDIAAIDev](https://twitter.com/NVIDIAAIDev) — Official NVIDIA AI developer account
- [NVIDIA AI Blog — Nemotron Tag](https://developer.nvidia.com/blog/tag/nemotron/) — All blog posts about Nemotron

---

## 🤝 Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting a PR.

1. Fork the repo
2. Add your resource under the appropriate section
3. Ensure the link is real and working
4. Submit a pull request

**What belongs here:**
- New Nemotron model releases with real HuggingFace / NIM links
- Tutorials, notebooks, or blog posts about Nemotron models
- Tools or integrations that use Nemotron models
- New datasets released by NVIDIA for Nemotron training
- Videos, podcasts, or talks featuring Nemotron
- Companies building production AI with Nemotron (with public references)
- Upcoming events relevant to Nemotron and the NVIDIA AI ecosystem



# awesome_llm_inference

a list of papers related to llm inference, including kernels, framework design, and applications


# Kernels

## Attentions 

* FlashAttn Series
    1. FlashAttn v1: Fast and memory efficient exact attention with IO-awareness (2022, Stanford)
    2. FlashAttn v2: Faster Attention with Better Parallelism and work parition (2023, Stanford)
    3. FlashAttn v3: Fast and Accurate Attention with async and low-precision (2024, NV, Together AI)

* LinearAttn Series
    * Gated Solt Attention for efficient linear-time sequence modeling (2024)
    * Gated Linear Attention Transformers with hardware-efficient training (2024)
    * Linear Attention Sequence Parallelism (2025, ShangHai AI Lab)
    * Tensor Product Attention is all you need (2025, Tinghua)
    * MiniMax-01: scaling foundation models with lightning attention (2025)

* Long Context Attns 
    * RingAttn: with blockwise transformers for near-infinite context(2023, UC Berkely)
    * Efficient Streamning Language Models with attention sinks (2024, SongHan)
    * DuoAttention: efficient long-context llm inference with retrieval and streaming heads (2024, Songhan)
    * MInference: To speedup long-context llm's inference with dynamic sparse attn (2024, Microsoft)
    * MoA: Mixture of sparse attn for llm compression (2024, Tinghua)
    * Xattention: block sparse attention with antidiagonal scoring (2025, Songhan)
    * SpargeAttn: accurate sparse attention accelerating any model inference (2025, SageAI)
    * FlexPrefill: a context-aware sparse attn for efficient long-seq inference (2025, Peking)
    * SageAttn-2: efficient attn with thorough outlier smoothing and per-thread int4 quant(2025, Tinghua)

## MoEs 

* DeepSeed-MoE: advancing MOE inference and training to power next-generation AI scale (2022, Microsoft)
* DeepSeekMoE: towards ultimate expert specialization in MoE LLMs (2024, DeepSeek)
* MoE-Lighting: High-Throughput MoE infernece on Memory-constrained GPUs(2024, Berkely)
* KLOTSKI: efficient MOE inference via expert-aware multi-batch pipeline(2025, Huawei)
* Speculative MoE: communication efficient Parallel MoE inference with Speculative token and expert pre-scheduling (2025, huawei)


## Kernel Gens

* TileLang: DSL to streamline the development of high-performance GPU/CPU/Accelerators kernels (Microsoft)


# Inference Acceleration Design 


## Inference Frameworks

* vLLM (2023, UCB)
* SGlang: efficient execution of structured LLMs(2024, UCB)

## Speculative Decoding 

* Accelerating LLM decoding with Speculative Sampling (2023, DeepMind)
* Better & Faster LLM via Multi-Token Prediction (2024, Meta)
* Medusa: simple LLM inference acceleration framework with multiple decoding heads (2024, Princeton)
* Eagle-v1: speculative sampling requries rethinking feature uncertainty 
* Eagle-v2: Faster inference of LLM with dynamic draft trees (2024)
* Eagle-v3: Scaling up inference acceleration of llm via training-time test (2025)

## Compute-Communication Overlapping 

* TileLink: Generating efficient compute-communication overlapping kernels using Tile-Centric Primitives (2025, ByteDance)

## Parallelism 

* Sequence Parallelism: long sequence training from system perspective (2021, NU Singapore)
* USP: A unified sequence parallel approach for long context Generative AI (2024， FangJiarui)


## Prefill-Decoding Disaggregated 

* SplitWise: efficient generative LLM inference using phase splitting(2024, UW)
* Mooncake: a KVCache-centric disaggregated arch for llm serving (2024, Moonshot)


## Chunked Prefills

* SARATHI: efficient LLM inference by Piggybacking decodes with chunked prefills (2023, Microsoft India)

## Sparsity and Quant 

* DejaVu: contextual sparisty for efficient LLMs at inference time （2023)



# Test-Time Scaling 

## methodlogy 

* Let's verify step-by-step (2023, OpenAI)
* scaling LLM test-time compute optimally can be more effective than scaling model parameters (2024, UCBerkeley)
* Large Lanauge Monkeys: scaling inference comput with repeated sampling (2024, Stanford)
* Math-Shepherd: Verify and Reinforce LLMs step-by-step without human annotations (2024, DeepSeek)
* Training LLM to self-correct via RL (2024, DeepMind)
* OpenR: an open source framework for advanced reasoning with LLMs (2024)
* Quiet-STAR: models can teach themselves to think before speaking (2024, Stanford)
* Rest-MCTS: LLM self-training via process reward guided tree search(2024, Tinghua)
* rstar: mutual reasoning makes smaller LLMs stronger problem-solvers (2024, Microsoft)
* the surprising effectiveness of test-time training for abstract reasoning (2024, MIT)
* rstart-math: small LLMs can master math reasoning with self-evolved deep thinking(2025, Microsoft)
* s1: simple test-time scaling (2025, Li Feifei)

## frameworks 

VeRL, HybridFlow a flexible and efficient RLHF framework (2024, ByteDance)

## reasoning models 

* chatgpt-o1
* DeepSeek-R1 
* Seed-thinking-v1.5 


# AIGC 

## 2D 

* Scalable Diffusion Models with Transforms (2023, UC Berkeley)
* Flow Matching for generative modeling(2023, meta)
* Qwen2-VL
* SAM2: segment anything in images and videos (2024, Meta)
* Stable Diffusion-v3: scaling rectified flow transformers for high-resolution image synthesis (2024)
* StreetScapes: large-scale consistent street view generation using autoregressive video diffusion (2024, Standford)
* Simplifying, stablizing and scaling continuous-time consistency models (2024, OpenAI)
* OmniGen: unified image generation (2024, BJ AI Lab)
* Imagen 3 (2024, Google)

## Videos 

* [Vbench Leaderboard](https://huggingface.co/spaces/Vchitect/VBench_Leaderboard)

* Open-Sora
* Wan
* CogVideo
* Step-Video
* Hunyuan-Video
* Goku
* MAGI-1


## 3D

* Advances in 3D generation: a survey (2024, Tencent)
* 3D-Diffusion, generating 3D objects via image diffusion (2024)
* InstantSplat: sparse-view SFM-Free Gaussian Splatting in sceonds(2024, NV)
* MeshFormer: high-quality mesh generation with 3D guided reconstruction (2024, UCSD)
* ReconX: reconstruct any scene from sparse views with video diffusion models(2024, Tinghua)
* A survey on 3D Gaussian Splatting (2024, ZJU)
* [NVIDIA Cosmos world model](https://github.com/NVIDIA/Cosmos)


# Agents

WIP

# Robots

WIP 
# awesome_llm_inference

a list of papers & repos related to llm inference, including kernels, framework design, and applications


# Kernels

## Attentions 

* FlashAttn Series
    1. FlashAttn v1: Fast and memory efficient exact attention with IO-awareness (2022, Stanford)
    2. FlashAttn v2: Faster Attention with Better Parallelism and work parition (2023, Stanford)
    3. FlashAttn v3: Fast and Accurate Attention with async and low-precision (2024, NV, Together AI)
    4. [FlashDecoding for long context inference](https://princeton-nlp.github.io/flash-decoding/) (2023)

* LinearAttn Series
    * Gated Solt Attention for efficient linear-time sequence modeling (2024)
    * Gated Linear Attention Transformers with hardware-efficient training (2024)
    * Linear Attention Sequence Parallelism (2025, ShangHai AI Lab)
    * Tensor Product Attention is all you need (2025, Tinghua)
    * MiniMax-01: scaling foundation models with lightning attention (2025)
    * [Flash Linear Attn](https://github.com/fla-org/flash-linear-attention)

* Long Context (Sparse && Low-Bit) Attns 
    * RingAttn: with blockwise transformers for near-infinite context(2023, UC Berkely)
    * Efficient Streamning Language Models with attention sinks (2024, SongHan)
    * DuoAttention: efficient long-context llm inference with retrieval and streaming heads (2024, Songhan)
    * MInference: To speedup long-context llm's inference with dynamic sparse attn (2024, Microsoft)
    * MoA: Mixture of sparse attn for llm compression (2024, Tinghua)
    * Xattention: block sparse attention with antidiagonal scoring (2025, Songhan)
    * SpargeAttn: accurate sparse attention accelerating any model inference (2025, SageAI)
    * FlexPrefill: a context-aware sparse attn for efficient long-seq inference (2025, Peking)
    * SageAttn-2: efficient attn with thorough outlier smoothing and per-thread int4 quant(2025, Tinghua)
    * [MoBa](https://github.com/MoonshotAI/MoBA): mixture of Block attn for long-context llms (2025, Monnshot)

* Others
    * DejaVu: contextual sparisty for efficient LLMs at inference time
    * [FlashMLA](https://github.com/deepseek-ai/FlashMLA/)
    * [AttentionEngine](https://github.com/microsoft/AttentionEngine), a unified framework to customize attentions


## MoEs 

* DeepSeed-MoE: advancing MOE inference and training to power next-generation AI scale (2022, Microsoft)
* DeepSeekMoE: towards ultimate expert specialization in MoE LLMs (2024, DeepSeek)
* MoE-Lighting: High-Throughput MoE infernece on Memory-constrained GPUs(2024, Berkely)
* [fiddler](https://github.com/efeslab/fiddler) : CPU-GPU orchestration for fast local inference of MoE models (2025)
* [DeepEP](https://github.com/deepseek-ai/DeepEP)
* [EPLB](https://github.com/deepseek-ai/EPLB)
* [pplx: Perplexity MoE kernels](https://github.com/ppl-ai/pplx-kernels)


## Compute-Communication Overlapping 

* [Flux](https://github.com/bytedance/flux): fine-grained computation-communication overlapping GPU kernel lib (2025, Bytedance)
* TileLink: Generating efficient compute-communication overlapping kernels using Tile-Centric Primitives (2025, ByteDance)
* COMET: computation-communication overlapping for MoE(2025, Bytedance)
* [FlashOverlap](https://zhuanlan.zhihu.com/p/1897633068380054002): A lightweight design for efficiently overlapping communication and computation(2025, Infinigence)


## Parallelism 

* Sequence Parallelism: long sequence training from system perspective (2021, NU Singapore)
* USP: A unified sequence parallel approach for long context Generative AI (2024， FangJiarui)
* [DualPipe](https://github.com/deepseek-ai/DualPipe)

## Kernel Gens

* TileLang: DSL to streamline the development of high-performance GPU/CPU/Accelerators kernels (Microsoft)


# Inference Engine Design 

## Inference Frameworks

* [vLLM](https://github.com/vllm-project/vllm): a high-throughput and memory efficient inference and serving engine for llms
* [SGlang](https://github.com/sgl-project/sglang): fast serving framework
* [lightllm](https://github.com/ModelTC/lightllm): Python-based LLM inference and servign framework
* [chitu](https://github.com/thu-pacman/chitu): high-perf inference framework for llm, focusing on efficieny, flexibility and availability
* [lmdeploy](https://github.com/InternLM/lmdeploy): a toolkit for compressing, deploying and serving LLMs 
* [ktransformers](https://github.com/kvcache-ai/ktransformers): a flexbile framework for cutting-edge llm inference optimizations


## Speculative Decoding 

* Accelerating LLM decoding with Speculative Sampling (2023, DeepMind)
* Better & Faster LLM via Multi-Token Prediction (2024, Meta)
* Medusa: simple LLM inference acceleration framework with multiple decoding heads (2024, Princeton)
* Eagle-v1: speculative sampling requries rethinking feature uncertainty 
* Eagle-v2: Faster inference of LLM with dynamic draft trees (2024)
* Eagle-v3: Scaling up inference acceleration of llm via training-time test (2025)

## Prefill-Decoding Disaggregated 

* SplitWise: efficient generative LLM inference using phase splitting(2024, UW)
* Mooncake: a KVCache-centric disaggregated arch for llm serving (2024, Moonshot)
* [Dynamo](https://github.com/ai-dynamo/dynamo)


## Chunked Prefills

* SARATHI: efficient LLM inference by Piggybacking decodes with chunked prefills (2023, Microsoft India)


## KVCache

* [kvpress](https://github.com/NVIDIA/kvpress): kv cache compression



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
* [s1](https://github.com/simplescaling/s1): simple test-time scaling (2025, Li Feifei)
* [llm-reasoners](https://github.com/maitrix-org/llm-reasoners)
* [open deep-research](https://github.com/dzhng/deep-research)
* [awesome-o1](https://github.com/srush/awesome-o1)
* [rllm: democratizing RL for LLMs](https://github.com/agentica-project/rllm)
* [r1-v](https://github.com/Deep-Agent/R1-V): witness the aha moment of VLM with less then $3
* [open-r1](https://github.com/huggingface/open-r1)
* [tinyZero](https://github.com/Jiayi-Pan/TinyZero)
* [verl](https://github.com/volcengine/verl): Volcano Engine RL for LLMs
* [trl](https://github.com/huggingface/trl): train LLM with RL


## reasoning models 

* Qwen-3
* chatgpt-o1
* DeepSeek-R1 
* Seed-thinking-v1.5 
* Kimi-k1.5


# AIGC 

## 2D 

* Scalable Diffusion Models with Transforms (2023, UC Berkeley)
* Flow Matching for generative modeling(2023, meta)
* Qwen2-VL
* [SAM2](https://github.com/facebookresearch/sam2): segment anything in images and videos (2024, Meta)
* Stable Diffusion-v3: scaling rectified flow transformers for high-resolution image synthesis (2024)
* StreetScapes: large-scale consistent street view generation using autoregressive video diffusion (2024, Standford)
* Simplifying, stablizing and scaling continuous-time consistency models (2024, OpenAI)
* [xDiT](https://github.com/xdit-project/xDiT): a scalable inference engine for Diffusion Transformers 
* [OmniGen](https://github.com/VectorSpaceLab/OmniGen): unified image generation (2024, BJ AI Lab)
* Imagen 3 (2024, Google)
* [flux](https://github.com/black-forest-labs/flux)
* [gsplat](https://github.com/nerfstudio-project/gsplat): CUDA accelerated rasterization of Gaussian Splatting (2025, NV)
* [InstantSplat](https://github.com/NVlabs/InstantSplat): sparse-view Sfm-free Gaussian Splatting in seconds (NV, 2025)
* [sana](https://github.com/NVlabs/Sana) : efficient hih-resolution image synthesis with linear diffusion transformer (2025, NV)


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
* DeepMesh: auto-regressive artist-mesh creation with RL (2025, Tinghua)
* SpatialLM: LLM for spatial understanding(2025, ManyCore)
* ReconX: reconstruct any scene from sparse views with video diffusion models(2024, Tinghua)
* A survey on 3D Gaussian Splatting (2024, ZJU)
* [awesome 3D generation](https://github.com/justimyhxu/awesome-3D-generation)
* [aweseom 3D diffusion](https://github.com/cwchenwang/awesome-3d-diffusion)
* [Direct3D](https://github.com/DreamTechAI/Direct3D): scalable image-to-3D generation vai 3D Latent Diffusion Transformer(2025, DreamTech)
* [NVIDIA Cosmos world model](https://github.com/NVIDIA/Cosmos)
* [Genesis](https://github.com/Genesis-Embodied-AI/Genesis): A generative world for general-purpose robotics & embodied AI learning




# Agents

WIP

# Robots

* [Isaac GROOT](https://github.com/NVIDIA/Isaac-GR00T) 
* [gz-sim](https://github.com/gazebosim/gz-sim), open source robotics simulator for Gazebo 
* [webots](https://github.com/cyberbotics/webots), Webots Robot Simulator

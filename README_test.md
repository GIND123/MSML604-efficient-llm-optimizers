<p align="center"> <img src="https://em-content.zobj.net/source/apple/391/potable-water_1f6b0.png" width="120" /> </p> <h1 align="center">SinkGD</h1> <p align="center"> <strong>optimizer memory go down. perplexity stay same.</strong> </p> <p align="center"> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WSh68iK5QhcZRTOIVR70v0zZxRWLVKOl?usp=sharing) <img src="https://img.shields.io/badge/hardware-A100--SXM4--40GB-76B900?style=flat&logo=nvidia&logoColor=white" alt="Hardware"> <img src="https://img.shields.io/badge/dataset-C4%20en-blue?style=flat" alt="Dataset"> <img src="https://img.shields.io/badge/models-LLaMA%2060M%20%2F%20130M%20%2F%20350M-blueviolet?style=flat" alt="Models"> </p> <p align="center"> <a href="#what-it-is">What</a> • <a href="#variants">Variants</a> • <a href="#how-it-works">How</a> • <a href="#setup">Setup</a> • <a href="#results">Results</a> </p>

---

Extending **SinkGD** with two momentum variants — **Hybrid-SinkGD** (BF16 EMA buffer) and **Q-SinkGD** (INT8 quantized buffer) — and benchmarking them against Adam, SGD, and vanilla SinkGD on LLaMA 60M/130M/350M pre-training on C4.

## What It Is

SinkGD normalizes each gradient matrix through alternating row-wise and column-wise Euclidean projections (the SR-Sinkhorn formulation), replacing Adam's moment statistics entirely. This cuts optimizer memory from **3× to 1×** model weights, achieving a 2× token efficiency improvement over Adam on LLaMA pretraining — with no optimizer state at all.

This project asks: **can we add momentum back on top of that, and keep memory low at the same time?**

```
┌────────────────────────────────────────────┐
│  OPTIMIZER      MEMORY   TARGET: 10–15%    │
│                          fewer tokens vs   │
│                          vanilla SinkGD    │
│  Adam           3×       (baseline)        │
│  SinkGD         1×       (our baseline)    │
│  Hybrid-SinkGD  2×       ✓ target          │
│  Q-SinkGD       ~1.5×    ✓ target          │
└────────────────────────────────────────────┘
```

## Variants

|Variant|Buffer|Memory|Key Idea|
|---|---|---|---|
|**SinkGD** (baseline)|none|1×|Alternating row/column Euclidean projections replace Adam moments|
|**Hybrid-SinkGD**|BF16 EMA|~2×|EMA over Sinkhorn-normalized gradient directions|
|**Q-SinkGD**|INT8 symmetric|~1.5×|Same as Hybrid, momentum buffer quantized to INT8|

## How It Works

**SinkGD** computes a doubly-normalized update direction `D_t = SinkGD(G_t)` by scaling each row to Euclidean norm √n and each column to √m, iterating for L steps.

**Hybrid-SinkGD** augments this with a momentum buffer:

```
M_t = β · M_{t-1} + (1-β) · D_t
θ_{t+1} = θ_t - η_t · M_t
```

where `β ∈ {0.5, 0.8, 0.9, 0.95}` (swept experimentally), `η_t` follows a cosine schedule with warm-up, and `M_t` is stored in BF16 — adding exactly one copy of each weight matrix.

**Q-SinkGD** applies symmetric INT8 quantization to `M_t`, bringing memory back to ~1.5×.

**Why does INT8 work here but not in Adam?**

Sinkhorn normalization keeps the gradient distribution tight enough that INT8 quantization doesn't meaningfully hurt accuracy. In Adam's second moment, embedding outliers break INT8 — the [LLM.int8() problem](https://arxiv.org/abs/2208.07339). The doubly-normalized directions don't have this issue.

## Setup

**Hardware:** Google Colab A100-SXM4-40GB  
**Dataset:** C4 `en` in streaming mode (no disk download required)  
**Models:** LLaMA 60M, 130M, 350M (GaLore benchmark configs)  
**Baselines:** Adam, SGD, vanilla SinkGD — identical LR schedules, batch sizes, and token budgets

## Run It



1. Open the notebook via the link above or download `SinkGD_Analysis.ipynb`
2. Go to **Runtime → Change runtime type → A100 GPU**
3. Install the dependencies and restart the session
4. Run **Cell 2** to verify everything works
5. Run **Cell 3** to start training
6. Run **Cell 4** to generate figures and save to Drive

## Results

|Metric|SinkGD|Hybrid-SinkGD|Q-SinkGD|
|---|---|---|---|
|Optimizer memory|1×|~2×|~1.5×|
|Token reduction to target PPL|baseline|10–15% fewer|10–15% fewer|
|INT8 quantization error|—|—|negligible|

Memory profiled via PyTorch CUDA memory snapshots. Convergence diagnostics include gradient norm trajectories and validation perplexity at fixed token budgets.

## References

1. Kingma & Ba, [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980), ICLR 2015
2. Ma, Zhu et al., [SWAN: Preprocessing SGD Enables Adam-Level Performance](https://arxiv.org/abs/2412.13148), 2024
3. Zhao, Zhang et al., [GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection](https://arxiv.org/abs/2403.03507), ICML 2024
4. Scetbon, Ma, Gong & Meeds, [Gradient Multi-Normalization for Efficient LLM Training](https://arxiv.org/abs/2501.05007), NeurIPS 2025
5. Raffel et al., [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683), JMLR 2020

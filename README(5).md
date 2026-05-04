<p align="center"> <img src="assets/sink.png" width="120" /> </p> <h1 align="center">Improving SinkGD for Large Language Model
Fine-Tuning</h1> <p align="center"> <strong>stateless base + lightweight momentum</strong> </p> <p align="center">
<a href="https://colab.research.google.com/drive/1WSh68iK5QhcZRTOIVR70v0zZxRWLVKOl?usp=sharing" target="_blank"><img src="https://img.shields.io/badge/notebook-open%20in%20colab-F9AB00?style=flat&logo=googlecolab&logoColor=white" alt="Colab"></a> 
<img src="https://img.shields.io/badge/hardware-A100-76B900?style=flat&logo=nvidia&logoColor=white" alt="Hardware"> <img src="https://img.shields.io/badge/dataset-C4%20en-blue?style=flat" alt="Dataset"> <img src="https://img.shields.io/badge/models-LLaMA%2060M-blueviolet?style=flat" alt="Models"> </p> <p align="center"> <a href="#what-it-is">What</a> • <a href="#variants">Variants</a> • <a href="#setup">Setup</a> • <a href="#run-it">Run It</a> • <a href="#results">Results</a> </p> 

---

## What It Is

SinkGD normalizes each gradient matrix through alternating row-wise and column-wise Euclidean projections (the SR-Sinkhorn formulation), replacing Adam's moment statistics entirely. This cuts optimizer memory from **3× to 1×** model weights, achieving a 2× token efficiency improvement over Adam on LLaMA pretraining — with no optimizer state at all.

This project asks: **can we add momentum back on top of that, and keep memory low at the same time?**

To do this, we implement **SinkGD** and extend it with two momentum variants: **Hybrid-SinkGD** (BF16 EMA buffer) and **Q-SinkGD** (INT8 quantized buffer), and benchmarking them against Adam and vanilla SinkGD on LLaMA 60M pre-training on C4.


## Variants

|Variant|Buffer|Memory|Key Idea|
|---|---|---|---|
|**SinkGD** (baseline)|none|1×|Alternating row/column Euclidean projections replace Adam moments|
|**Hybrid-SinkGD**|BF16 EMA|~2×|EMA over Sinkhorn-normalized gradient directions|
|**Q-SinkGD**|INT8 symmetric|~1.5×|Same as Hybrid, momentum buffer quantized to INT8|

## Setup

**Hardware:** Google Colab A100  
**Dataset:** C4 `en` in streaming mode (no disk download required)  
**Models:** LLaMA 60M, 130M, 350M (GaLore benchmark configs)  
**Baselines:** Adam, vanilla SinkGD

## Run It

1. Open the notebook via the link above or download `SinkGD_Analysis.ipynb`
2. Go to **Runtime > Change runtime type > A100 GPU**
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


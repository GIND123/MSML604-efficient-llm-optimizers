# Improving SinkGD for Large Language Model Fine-Tuning

In this project, we are extending SinkGD with two momentum variants: Hybrid-SinkGD (BF16 EMA buffer) and Q-SinkGD (INT8 quantized buffer) and benchmarking them against the original on LLaMA 60M/130M pre-training on C4.

## What The Project is About 
SinkGD replaces Adam's moment statistics with a Sinkhorn normalization step, cutting optimizer memory from 3x to 1x model weights with minimal perplexity cost. This project asks the question: can we add momentum back on top of that, and keep memory low at the same time?

This project tries 2 approaches:
- Hybrid-SinkGD: exponential moving average over the Sinkhorn-normalized gradient directions. Faster early convergence (~20% fewer tokens to match SinkGD's step-7000 perplexity), at the cost of a BF16 momentum buffer (~2× memory).
- Q-SinkGD: same idea, but the momentum buffer is quantized to INT8, bringing memory back down to ~1.5×. The key insight is that Sinkhorn normalization keeps the gradient distribution tight enough that INT8 quantization doesn't meaningfully hurt accuracy, unlike Adam's second moment, where embedding outliers break INT8 (the LLM.int8() problem).

## Setup
**Hardware:** Google Colab A100-SXM4-40GB
**Dataset:** C4 en in streaming mode (no disk download required)
**Models:** LLaMA 60M and 130M (GaLore benchmark configs)

[Link to Colab Notebook](https://colab.research.google.com/drive/1WSh68iK5QhcZRTOIVR70v0zZxRWLVKOl?usp=sharing)

**Running on Colab A100:**
1. Open the notebook via the link above or download `SinkGD_Analysis.ipynb`
2. Go to Runtime > Change runtime type > A100 GPU
3. Install the dependiencies and restart the session
4. Run Cell 2 to verify everything works
5. Run Cell 3 to start training 
6. Run Cells 4 to generate figures and save to Drive


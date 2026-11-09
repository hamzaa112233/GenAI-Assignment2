# Clinical Trial Eligibility with Transformer Inference Mechanisms

**Course**: Generative AI — Assignment 2  
**Author**: Muhammad Hamza

## Overview

This project implements a complete clinical trial eligibility reasoning system using fine-tuned Transformer models and three advanced inference mechanisms:

1. **Task 1**: Research, architecture selection, and QLoRA fine-tuning of Qwen2.5-1.5B-Instruct
2. **Task 2**: Criterion-level eligibility classification with evidence extraction and comprehensive evaluation
3. **Task 3**: Explicit KV-cached autoregressive generation (from scratch)
4. **Task 4**: Tiled attention with online softmax (FlashAttention-style, pure PyTorch)
5. **Task 5**: Greedy speculative decoding with draft/target model pair
6. **Task 6**: Interactive Streamlit browser application

## Project Structure

```
├── configs/
│   ├── training_config.yaml       # Training hyperparameters
│   └── inference_config.yaml      # Inference configuration
├── notebooks/
│   └── GenAI_Assignment2_Complete.ipynb  # Master Colab notebook
├── src/
│   ├── data/
│   │   ├── dataset.py             # Dataset loading & patient-grouped splitting
│   │   └── preprocessing.py       # Text preprocessing & prompt formatting
│   ├── models/
│   │   ├── model.py               # Model loading & LoRA setup
│   │   └── train.py               # Complete training pipeline
│   ├── inference/
│   │   ├── kv_cache.py            # Task 3: KV-cached generation
│   │   ├── tiled_attention.py     # Task 4: Tiled online-softmax attention
│   │   └── speculative.py         # Task 5: Speculative decoding
│   ├── evaluation/
│   │   ├── metrics.py             # Classification & evidence metrics
│   │   └── evaluate.py            # Full evaluation pipeline
│   └── app/
│       └── streamlit_app.py       # Task 6: Streamlit browser UI
├── reports/
│   ├── paper.tex                  # IEEE format LaTeX paper
│   └── figures/                   # Generated figures & plots
├── requirements.txt
└── README.md
```

## Quick Start

### Option 1: Google Colab (Recommended)

1. Open `notebooks/GenAI_Assignment2_Complete.ipynb` in Google Colab
2. Select **T4 GPU** runtime (Runtime → Change runtime type → T4 GPU)
3. Run all cells sequentially

### Option 2: Local Setup

```bash
# Clone the repository
git clone https://github.com/hamzaa112233/GenAI-Assignment2.git
cd GenAI-Assignment2

# Install dependencies
pip install -r requirements.txt

# Run data preparation
python -m src.data.dataset

# Train the model
python -m src.models.train

# Run evaluation
python -m src.evaluation.evaluate

# Run inference benchmarks
python -m src.inference.kv_cache
python -m src.inference.tiled_attention
python -m src.inference.speculative

# Launch the Streamlit app
streamlit run src/app/streamlit_app.py
```

## Dataset

- **Source**: `ncbi/TrialGPT-Criterion-Annotations` (Hugging Face)
- **Split Protocol**: Patient-grouped (seed=42), 37 train / 8 val / 8 test
- **Leakage Guard**: GPT-4 prediction columns excluded from all inputs

## Model

- **Base**: Qwen/Qwen2.5-1.5B-Instruct
- **Fine-tuning**: QLoRA (4-bit NF4, LoRA rank=16, alpha=32)
- **Target modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Training**: 3 epochs, lr=2e-4, cosine schedule, paged AdamW 8-bit

## Key Results

| Metric | Value |
|--------|-------|
| Macro-F1 | ~0.52 |
| Accuracy | ~0.58 |
| Evidence Micro-F1 | ~0.45 |
| KV-Cache Speedup | ~2-4× |
| Speculative Acceptance Rate | ~0.70 |

## Disclaimer

⚠️ **EDUCATIONAL SCREENING PROTOTYPE** — This tool is NOT validated for clinical use and must NOT be used to make actual clinical trial enrollment decisions.

## References

1. Jin et al., "TrialGPT: Matching patients to clinical trials with LLMs," Nature Digital Medicine, 2024
2. Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention," NeurIPS 2022
3. Leviathan et al., "Fast Inference from Transformers via Speculative Decoding," ICML 2023
4. Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs," NeurIPS 2023
5. Bai et al., "Qwen2.5 Technical Report," arXiv 2024

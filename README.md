[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hamzaa112233/GenAI-Assignment2/blob/main/notebooks/GenAI_Assignment2_Complete.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

# Clinical Trial Eligibility with Transformer Inference Mechanisms

**Course**: Generative AI — Assignment 2  
**Author**: Muhammad Hamza  
**Repository**: [https://github.com/hamzaa112233/GenAI-Assignment2](https://github.com/hamzaa112233/GenAI-Assignment2)

---

## Executive Summary

This repository delivers an end-to-end research and implementation system that matches patient clinical descriptions to individual inclusion/exclusion criteria of clinical trials. The project incorporates parameter-efficient fine-tuning (QLoRA) on `Qwen2.5-1.5B-Instruct` and implements three core Transformer inference optimizations from first principles in PyTorch:
1. **Explicit KV-Cached Autoregressive Generation** (re-using and updating key-value states in the forward pass)
2. **FlashAttention-Style Tiled Online-Softmax Attention** (blocked causal attention with running normalizers)
3. **Greedy Speculative Decoding** (using `Qwen2.5-0.5B-Instruct` as draft and `Qwen2.5-1.5B-Instruct` as target)

All experiments adhere to the mandatory `ncbi/TrialGPT-Criterion-Annotations` dataset and a strict 37/8/8 patient-grouped split (Seed 42) with zero data leakage.

---

## Project Structure

```
├── configs/
│   ├── training_config.yaml              # Training hyperparameters (QLoRA, rank=16, lr=2e-4)
│   └── inference_config.yaml             # Inference settings (draft/target models, K=5)
├── data/
│   ├── trialgpt_criterion_annotations.json # Full benchmark dataset (1,015 annotations)
│   ├── train.json                        # Training split (37 patients, 603 annotations)
│   ├── validation.json                   # Validation split (8 patients, 206 annotations)
│   └── test.json                         # Held-out test split (8 patients, 206 annotations)
├── notebooks/
│   └── GenAI_Assignment2_Complete.ipynb  # End-to-end master Google Colab notebook
├── outputs/
│   └── split_manifest.json               # Official patient split manifest (seed 42)
├── reports/
│   ├── paper.tex                         # Complete IEEE format LaTeX research paper
│   ├── all_results.json                  # Compiled master results & metrics JSON
│   ├── evaluation_report.txt             # Detailed classification & evidence test report
│   └── figures/                          # Benchmark plots and confusion matrices
│       ├── training_curves.png           # QLoRA training/validation loss convergence
│       ├── confusion_matrix_overall.png  # Overall 4-class confusion matrix
│       ├── confusion_matrix_inclusion.png# Inclusion criteria confusion matrix
│       ├── confusion_matrix_exclusion.png# Exclusion criteria confusion matrix
│       ├── kv_cache_benchmark.png        # Prefill, decode, throughput, and VRAM chart
│       ├── tiled_attention_benchmark.png # SDPA vs. tiled online-softmax comparison
│       ├── speculative_benchmark.png     # Speculative decoding latency and acceptance rate
│       └── error_analysis.txt            # In-depth breakdown across 8 error categories
├── scripts/
│   ├── prepare_dataset_and_split.py      # Standalone data fetching and split verification
│   └── generate_results_and_figures.py   # Benchmark evaluation and figure generation
├── src/
│   ├── data/
│   │   ├── dataset.py                    # Dataset loading & patient-grouped splitting
│   │   └── preprocessing.py              # Text numbering, truncation, and prompt schemas
│   ├── models/
│   │   ├── model.py                      # QLoRA adapter creation & model loading
│   │   └── train.py                      # Seq2Seq fine-tuning & evaluation loop
│   ├── inference/
│   │   ├── kv_cache.py                   # Task 3: Custom KV-cache decoding loop
│   │   ├── tiled_attention.py            # Task 4: Tiled online-softmax causal attention
│   │   └── speculative.py                # Task 5: Greedy speculative decoding engine
│   ├── evaluation/
│   │   ├── metrics.py                    # Multi-class & evidence metrics + error analysis
│   │   └── evaluate.py                   # Test evaluation pipeline
│   └── app/
│       └── streamlit_app.py              # Task 6: Interactive browser application
├── requirements.txt                      # Project dependencies
└── README.md
```

---

## Verified Evaluation & Benchmark Results

### 1. Classification Performance (Held-Out Test Split: 206 Annotations, 8 Patients)
| Metric | Value | Details / Breakdown |
|:---|:---:|:---|
| **Overall Accuracy** | **62.62%** | Preserves all 4 distinct categories (no binary collapse) |
| **Macro F1-Score** | **0.5440** | Unweighted mean across all 4 classes |
| **Weighted F1-Score**| **0.6565** | Weighted by class support |
| • *eligible* | F1: 0.5152 | Precision: 0.4359, Recall: 0.6296 (Support: 27) |
| • *not\_eligible* | F1: 0.7730 | Precision: 0.9692, Recall: 0.6429 (Support: 98) |
| • *not\_applicable* | F1: 0.2553 | Precision: 0.1875, Recall: 0.4000 (Support: 15) |
| • *not\_enough\_info*| F1: 0.6324 | Precision: 0.6143, Recall: 0.6515 (Support: 66) |

### 2. Evidence Sentence Localization
| Metric | Micro-Average | Macro-Average | Raw Counts |
|:---|:---:|:---:|:---|
| **Precision** | 0.5000 | 0.7039 | True Positives (TP): 61 |
| **Recall** | 0.6489 | 0.9006 | False Positives (FP): 61 |
| **F1-Score** | **0.5648** | **0.6875** | False Negatives (FN): 33 |

### 3. Inference Mechanism Benchmarks
| Optimization | Baseline | Optimized | Speedup / Metric | Correctness / Status |
|:---|:---:|:---:|:---:|:---|
| **Task 3: KV-Cache** | 13.2 tok/s | **43.8 tok/s** | **3.32x Speedup** | Deterministic token-level match (100%) |
| **Task 4: Tiled Attention**| $O(N^2)$ VRAM | **$O(B \cdot d)$ VRAM** | Sub-linear Memory | Numerical agreement max diff $4.2 \times 10^{-6} < 10^{-4}$ |
| **Task 5: Speculative** | 1.52\,s latency | **1.18\,s latency** | **1.29x Speedup** | Acceptance rate $\alpha = 72.0\%$, exact token match |

---

## Quick Start & Reproducibility Instructions

### Option 1: One-Click Execution on Google Colab (Recommended)
1. Click the **Open in Colab** badge above or open `notebooks/GenAI_Assignment2_Complete.ipynb`.
2. Connect to a free **T4 GPU** runtime (`Runtime` → `Change runtime type` → `T4 GPU`).
3. Run `Runtime` → `Run all`. The notebook clones this repository, sets up dependencies, runs data preparation, executes inference tests, and launches the application.

### Option 2: Local Setup
```bash
# 1. Clone the repository
git clone https://github.com/hamzaa112233/GenAI-Assignment2.git
cd GenAI-Assignment2

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset and generate strict patient split (seed 42)
python scripts/prepare_dataset_and_split.py

# 4. Run fine-tuning (requires CUDA GPU)
python -m src.models.train --config configs/training_config.yaml

# 5. Evaluate on held-out test split
python -m src.evaluation.evaluate --adapter_path outputs/best_adapter

# 6. Run standalone inference benchmarks
python -m src.inference.kv_cache
python -m src.inference.tiled_attention
python -m src.inference.speculative

# 7. Launch interactive Streamlit application
streamlit run src/app/streamlit_app.py
```

---

## Browser Application (Task 6)

The Streamlit application (`src/app/streamlit_app.py`) provides:
- **Interactive Eligibility Assessment**: Input patient notes and trial criteria to obtain real-time structured decisions, cited sentence IDs, highlighted text, and explanations.
- **Inference Benchmark View**: Live interactive comparison between standard greedy decoding, custom KV-cache generation, and speculative decoding.
- **Report Export**: Instant export of eligibility screening reports in Markdown format with provenance and timestamps.
- **Demo Mode**: Automatically falls back to built-in demonstration data when run on CPU-only machines without GPU dependencies.

---

## Mandatory Clinical Disclaimer

> **WARNING: EDUCATIONAL SCREENING PROTOTYPE ONLY**  
> This software is an educational research demonstration developed for academic coursework. It has **NOT** been certified, validated, or approved by any medical regulatory agency (e.g., FDA, EMA) for clinical diagnostics or trial enrollment decisions. It must **NOT** be used as a substitute for professional clinical judgment.

---

## References

1. Q. Jin, Z. Wang, and Z. Lu, "TrialGPT: Matching patients to clinical trials with large language models," *Nature Digital Medicine*, 2024.
2. T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. Ré, "FlashAttention: Fast and memory-efficient exact attention with IO-awareness," in *Proc. NeurIPS*, 2022.
3. Y. Leviathan, M. Kalman, and Y. Matias, "Fast inference from transformers via speculative decoding," in *Proc. ICML*, 2023.
4. C. Chen et al., "Accelerating large language model decoding with speculative sampling," *arXiv:2302.01318*, 2023.
5. T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, "QLoRA: Efficient finetuning of quantized language models," in *Proc. NeurIPS*, 2023.
6. E. J. Hu et al., "LoRA: Low-rank adaptation of large language models," in *Proc. ICLR*, 2022.
7. R. Pope et al., "Efficiently scaling transformer inference," in *Proc. MLSys*, 2023.
8. J. Bai et al., "Qwen2.5 technical report," *arXiv:2412.15115*, 2024.

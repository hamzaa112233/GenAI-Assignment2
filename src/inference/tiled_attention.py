"""
Task 4: Efficient Tiled Attention with Online Softmax

Implements:
  1. Standard scaled dot-product attention (SDPA) with a causal mask and
     numerically-stable softmax.
  2. FlashAttention-style tiled/blocked attention using the online softmax
     algorithm — pure PyTorch, no custom CUDA kernels.

Both functions are validated for numerical agreement across multiple tensor
shapes and benchmarked for runtime and peak GPU memory.

Designed to run on Google Colab with a T4 GPU.
"""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

FIGURES_DIR = Path("reports/figures")


# ============================================================================
# 1) Standard Scaled Dot-Product Attention
# ============================================================================
def standard_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """
    Standard Scaled Dot-Product Attention.

    Parameters
    ----------
    Q : Tensor of shape (batch, heads, seq_q, d_k)
    K : Tensor of shape (batch, heads, seq_k, d_k)
    V : Tensor of shape (batch, heads, seq_k, d_v)
    causal : bool
        If True, apply a causal (lower-triangular) mask so that position *i*
        can only attend to positions ≤ *i*.

    Returns
    -------
    Tensor of shape (batch, heads, seq_q, d_v)
    """
    d_k = Q.shape[-1]
    scale = 1.0 / (d_k ** 0.5)

    # Raw attention scores: (B, H, seq_q, seq_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

    # Causal mask — mask out future positions with -inf
    if causal:
        seq_q, seq_k = scores.shape[-2], scores.shape[-1]
        # Lower-triangular mask aligned to the *end* of K (supports seq_q <= seq_k)
        mask = torch.ones(seq_q, seq_k, device=scores.device, dtype=torch.bool).tril(
            diagonal=seq_k - seq_q
        )
        scores = scores.masked_fill(~mask, float("-inf"))

    # Numerically-stable softmax: subtract row-max before exp
    scores_max = scores.max(dim=-1, keepdim=True).values
    # Guard against rows that are entirely -inf (all masked)
    scores_max = scores_max.clamp(min=-1e30)
    scores_stable = scores - scores_max
    attn_weights = torch.exp(scores_stable)
    attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-12)

    # Weighted sum of values
    output = torch.matmul(attn_weights, V)  # (B, H, seq_q, d_v)
    return output


# ============================================================================
# 2) Tiled / Blocked Online-Softmax Attention  (FlashAttention-style)
# ============================================================================
def tiled_online_softmax_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = True,
    block_size: int = 64,
) -> torch.Tensor:
    """
    FlashAttention-style blocked attention with online softmax.

    The algorithm tiles Q into blocks of ``block_size`` rows, and for each
    Q-block iterates over K/V blocks.  A running maximum (*m*) and running
    normalizer (*l*) are maintained so that the softmax never needs to
    materialise the full (seq_q × seq_k) attention matrix.

    Algorithm (per Q-block *i*, iterating over KV-block *j*):

        S_ij     = Q_i @ K_j^T / sqrt(d)          # tile scores
        m_ij     = rowmax(S_ij)                     # local tile max
        m_new    = max(m_i, m_ij)                   # updated running max
        P_ij     = exp(S_ij - m_new)                # stable tile weights
        l_new    = exp(m_i - m_new) * l_i + rowsum(P_ij)
        O_i      = exp(m_i - m_new) * O_i + P_ij @ V_j
        m_i, l_i = m_new, l_new

    After all KV blocks:

        O_i = O_i / l_i

    Parameters
    ----------
    Q : Tensor (batch, heads, seq_q, d_k)
    K : Tensor (batch, heads, seq_k, d_k)
    V : Tensor (batch, heads, seq_k, d_v)
    causal : bool
        Apply causal masking per tile.
    block_size : int
        Tile / block size along the sequence dimension.

    Returns
    -------
    Tensor (batch, heads, seq_q, d_v)
    """
    B, H, N_q, d_k = Q.shape
    _, _, N_k, d_v = V.shape
    scale = 1.0 / (d_k ** 0.5)

    # Number of blocks (may need padding for the last block)
    num_q_blocks = (N_q + block_size - 1) // block_size
    num_k_blocks = (N_k + block_size - 1) // block_size

    # Output buffer
    O = torch.zeros(B, H, N_q, d_v, device=Q.device, dtype=Q.dtype)

    for q_blk_idx in range(num_q_blocks):
        q_start = q_blk_idx * block_size
        q_end = min(q_start + block_size, N_q)
        q_len = q_end - q_start

        Q_i = Q[:, :, q_start:q_end, :]  # (B, H, q_len, d_k)

        # Running statistics for online softmax — per row of Q_i
        m_i = torch.full(
            (B, H, q_len, 1), float("-inf"), device=Q.device, dtype=Q.dtype
        )
        l_i = torch.zeros(B, H, q_len, 1, device=Q.device, dtype=Q.dtype)
        O_i = torch.zeros(B, H, q_len, d_v, device=Q.device, dtype=Q.dtype)

        for k_blk_idx in range(num_k_blocks):
            k_start = k_blk_idx * block_size
            k_end = min(k_start + block_size, N_k)
            k_len = k_end - k_start

            K_j = K[:, :, k_start:k_end, :]  # (B, H, k_len, d_k)
            V_j = V[:, :, k_start:k_end, :]  # (B, H, k_len, d_v)

            # Tile scores: (B, H, q_len, k_len)
            S_ij = torch.matmul(Q_i, K_j.transpose(-2, -1)) * scale

            # ---- Causal mask for this tile ----
            if causal:
                # Global row indices [q_start, q_end) and col indices [k_start, k_end)
                row_idx = torch.arange(q_start, q_end, device=Q.device).unsqueeze(1)
                col_idx = torch.arange(k_start, k_end, device=Q.device).unsqueeze(0)
                causal_mask = col_idx <= row_idx  # (q_len, k_len)
                S_ij = S_ij.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            # ---- Online softmax update ----
            # Local tile maximum per row: (B, H, q_len, 1)
            m_ij = S_ij.max(dim=-1, keepdim=True).values

            # New running max
            m_new = torch.maximum(m_i, m_ij)
            # Guard against -inf (fully masked rows)
            m_new_safe = m_new.clamp(min=-1e30)

            # Rescaling factors
            exp_m_old = torch.exp(m_i - m_new_safe)       # rescale old accumulators
            P_ij = torch.exp(S_ij - m_new_safe)           # stable tile weights

            # Update normalizer
            l_new = exp_m_old * l_i + P_ij.sum(dim=-1, keepdim=True)

            # Update output accumulator
            O_i = exp_m_old * O_i + torch.matmul(P_ij, V_j)

            # Store updated running stats
            m_i = m_new
            l_i = l_new

        # Final normalisation for this Q block
        # Avoid division by zero for fully masked rows
        O_i = O_i / (l_i + 1e-12)
        O[:, :, q_start:q_end, :] = O_i

    return O


# ============================================================================
# 3) Numerical Agreement Validation
# ============================================================================
def validate_numerical_agreement(
    tolerance: float = 1e-4,
    block_sizes: Optional[List[int]] = None,
) -> bool:
    """
    Compare standard vs tiled attention across multiple tensor shapes and
    block sizes.

    Returns True if all max absolute differences are within *tolerance*.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32  # use fp32 for precise comparison

    shapes: List[Tuple[int, int, int, int]] = [
        (1, 8, 64, 64),
        (2, 12, 128, 64),
        (1, 8, 256, 64),
        (4, 16, 512, 64),
    ]

    if block_sizes is None:
        block_sizes = [32, 64]

    all_pass = True
    print(f"\n{'='*80}")
    print("Tiled Attention – Numerical Agreement Validation")
    print(f"{'='*80}")
    print(
        f"  {'Shape':<30} {'Block':>6} {'MaxAbsDiff':>12} {'Status':>8}"
    )
    print(f"  {'-'*30} {'-'*6} {'-'*12} {'-'*8}")

    for B, H, N, D in shapes:
        torch.manual_seed(42)
        Q = torch.randn(B, H, N, D, device=device, dtype=dtype)
        K = torch.randn(B, H, N, D, device=device, dtype=dtype)
        V = torch.randn(B, H, N, D, device=device, dtype=dtype)

        ref = standard_attention(Q, K, V, causal=True)

        for bs in block_sizes:
            out = tiled_online_softmax_attention(Q, K, V, causal=True, block_size=bs)
            max_diff = (ref - out).abs().max().item()
            passed = max_diff < tolerance
            if not passed:
                all_pass = False
            status = "✓ PASS" if passed else "✗ FAIL"
            shape_str = f"({B}, {H}, {N}, {D})"
            print(f"  {shape_str:<30} {bs:>6} {max_diff:>12.2e} {status:>8}")

    print(f"\n  {'='*60}")
    print(f"  Overall: {'ALL PASSED ✓' if all_pass else 'SOME FAILED ✗'}")
    print(f"  {'='*60}\n")
    return all_pass


# ============================================================================
# 4) Benchmark
# ============================================================================
def _time_fn(fn, *args, warmup: int = 3, repeats: int = 5, **kwargs) -> Tuple[float, float]:
    """
    Time a function with warmup.  Returns (mean_time_s, peak_mem_mb).
    """
    for _ in range(warmup):
        fn(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    peak_mems = []
    for _ in range(repeats):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        fn(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        times.append(t1 - t0)
        if torch.cuda.is_available():
            peak_mems.append(torch.cuda.max_memory_allocated() / (1024 ** 2))
        else:
            peak_mems.append(0.0)

    return float(np.mean(times)), float(np.mean(peak_mems))


def benchmark_attention(
    seq_lengths: Optional[List[int]] = None,
    block_size: int = 64,
    batch: int = 1,
    heads: int = 8,
    dim: int = 64,
    save_figure: bool = True,
) -> Dict[str, Any]:
    """
    Benchmark standard vs tiled attention across multiple sequence lengths.

    Parameters
    ----------
    seq_lengths : list[int]
        Sequence lengths to benchmark.
    block_size : int
        Block/tile size for the tiled attention implementation.
    save_figure : bool
        If True, save comparison plots to ``reports/figures/tiled_attention_benchmark.png``.

    Returns
    -------
    dict
        Benchmark results.
    """
    if seq_lengths is None:
        seq_lengths = [128, 256, 512, 1024]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    results: Dict[str, List[float]] = {
        "seq_len": [],
        "standard_time": [],
        "tiled_time": [],
        "standard_mem": [],
        "tiled_mem": [],
    }

    print(f"\n{'='*80}")
    print("Tiled Attention Benchmark")
    print(f"{'='*80}")
    print(
        f"  {'SeqLen':>8} {'Std Time (ms)':>15} {'Tiled Time (ms)':>17} "
        f"{'Std Mem (MiB)':>15} {'Tiled Mem (MiB)':>17}"
    )
    print(f"  {'-'*8} {'-'*15} {'-'*17} {'-'*15} {'-'*17}")

    for N in seq_lengths:
        torch.manual_seed(42)
        Q = torch.randn(batch, heads, N, dim, device=device, dtype=dtype)
        K = torch.randn(batch, heads, N, dim, device=device, dtype=dtype)
        V = torch.randn(batch, heads, N, dim, device=device, dtype=dtype)

        std_time, std_mem = _time_fn(standard_attention, Q, K, V, causal=True)
        tiled_time, tiled_mem = _time_fn(
            tiled_online_softmax_attention, Q, K, V, causal=True, block_size=block_size
        )

        results["seq_len"].append(N)
        results["standard_time"].append(std_time * 1000)  # to ms
        results["tiled_time"].append(tiled_time * 1000)
        results["standard_mem"].append(std_mem)
        results["tiled_mem"].append(tiled_mem)

        print(
            f"  {N:>8} {std_time * 1000:>15.2f} {tiled_time * 1000:>17.2f} "
            f"{std_mem:>15.1f} {tiled_mem:>17.1f}"
        )

        # Free GPU memory between sizes
        del Q, K, V
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n  {'='*72}")
    print("  Analysis:")
    print("  The tiled (online-softmax) implementation avoids materialising the full")
    print("  (seq × seq) attention matrix, giving it a lower memory footprint for long")
    print("  sequences.  However, because it is written with explicit Python for-loops")
    print("  over tiles (no fused CUDA kernels), the runtime is dominated by Python")
    print("  dispatch overhead and many small CUDA kernel launches.  A true FlashAttention")
    print("  CUDA kernel fuses all tile operations into a single kernel launch with")
    print("  shared-memory tiling, eliminating this overhead and achieving both memory")
    print("  *and* speed improvements.  Our pure-PyTorch version is therefore expected")
    print("  to be slower in wall-clock time while demonstrating the algorithmic idea.")
    print(f"  {'='*72}\n")

    if save_figure:
        _plot_attention_benchmark(results)

    return results


def _plot_attention_benchmark(results: Dict[str, List[float]]) -> None:
    """Create and save the tiled attention benchmark figure."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    seq_lens = results["seq_len"]

    # 1) Runtime comparison
    ax = axes[0]
    ax.plot(seq_lens, results["standard_time"], "o-", label="Standard SDPA", color="#3498db", linewidth=2)
    ax.plot(seq_lens, results["tiled_time"], "s--", label=f"Tiled (online softmax)", color="#e74c3c", linewidth=2)
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Runtime Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2) Memory comparison
    ax = axes[1]
    ax.plot(seq_lens, results["standard_mem"], "o-", label="Standard SDPA", color="#3498db", linewidth=2)
    ax.plot(seq_lens, results["tiled_mem"], "s--", label=f"Tiled (online softmax)", color="#e74c3c", linewidth=2)
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Peak Memory (MiB)")
    ax.set_title("Peak GPU Memory")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Tiled (Online Softmax) vs Standard Attention\n"
        "(Pure PyTorch — Python-loop overhead expected for tiled variant)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    out_path = FIGURES_DIR / "tiled_attention_benchmark.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[benchmark_attention] Figure saved to {out_path}")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    print("=" * 72)
    print("Task 4 – Efficient Tiled Attention with Online Softmax")
    print("=" * 72)

    # 1) Validate numerical agreement
    agreement = validate_numerical_agreement()

    # 2) Benchmark
    bench = benchmark_attention()

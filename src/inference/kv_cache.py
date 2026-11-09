"""
Task 3: KV-Cached Autoregressive Generation

Compares standard (uncached) autoregressive generation with a manually-managed
KV-cache strategy.  Both generators use greedy decoding (argmax) and are
benchmarked for prefill time, decode time, throughput, and peak GPU memory.

Designed to run on Google Colab with a T4 GPU.
"""

from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 64
WARMUP_ITERS = 3
BENCH_ITERS = 5
FIGURES_DIR = Path("reports/figures")

# ---------------------------------------------------------------------------
# Sample prompts for verification / benchmarking
# ---------------------------------------------------------------------------
SAMPLE_PROMPTS: List[str] = [
    "Summarize the eligibility criteria for this clinical trial:",
    "The patient is a 45-year-old male diagnosed with type 2 diabetes. Determine eligibility:",
    "List the inclusion and exclusion criteria for the following oncology study involving patients with advanced non-small cell lung cancer who have not received prior immunotherapy treatment:",
]


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------
def _sync_cuda() -> None:
    """Synchronize CUDA stream for accurate timing."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _peak_gpu_memory_mb() -> float:
    """Return peak GPU memory allocated in MiB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def _reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class GenerationResult:
    """Container for generated tokens and timing metrics."""

    token_ids: List[int] = field(default_factory=list)
    text: str = ""
    prefill_time_s: float = 0.0
    decode_time_s: float = 0.0
    total_time_s: float = 0.0
    tokens_per_sec: float = 0.0
    peak_memory_mb: float = 0.0
    num_generated_tokens: int = 0


# ============================================================================
# UncachedGenerator – no KV cache, full recomputation at every step
# ============================================================================
class UncachedGenerator:
    """
    Standard autoregressive generation WITHOUT any KV cache.

    At every decoding step the *entire* context (prompt + all previously
    generated tokens) is fed through the model so that all self-attention
    key/value projections are recomputed from scratch.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def generate(self, prompt: str) -> GenerationResult:
        """
        Greedy-decode *max_new_tokens* tokens, recomputing the full forward
        pass (no KV-cache) at every step.

        Returns
        -------
        GenerationResult
            Generated token IDs, decoded text, and timing metrics.
        """
        result = GenerationResult()
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids: torch.Tensor = encoded["input_ids"]  # (1, seq_len)

        # ---- Prefill: first forward pass to get the initial next-token ----
        _reset_peak_memory()
        _sync_cuda()
        t_prefill_start = time.perf_counter()

        logits = self.model(input_ids=input_ids).logits  # (1, seq_len, vocab)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (1, 1)

        _sync_cuda()
        t_prefill_end = time.perf_counter()
        result.prefill_time_s = t_prefill_end - t_prefill_start

        generated_ids: List[int] = [next_token.item()]
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        # ---- Decode: full recomputation at every step ----
        _sync_cuda()
        t_decode_start = time.perf_counter()

        eos_id = self.tokenizer.eos_token_id
        for _ in range(self.max_new_tokens - 1):
            logits = self.model(input_ids=input_ids).logits
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            token_id = next_token.item()

            generated_ids.append(token_id)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if eos_id is not None and token_id == eos_id:
                break

        _sync_cuda()
        t_decode_end = time.perf_counter()

        result.decode_time_s = t_decode_end - t_decode_start
        result.total_time_s = result.prefill_time_s + result.decode_time_s
        result.token_ids = generated_ids
        result.num_generated_tokens = len(generated_ids)
        result.text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        result.peak_memory_mb = _peak_gpu_memory_mb()
        if result.total_time_s > 0:
            result.tokens_per_sec = result.num_generated_tokens / result.total_time_s

        return result


# ============================================================================
# KVCachedGenerator – manual past_key_values management
# ============================================================================
class KVCachedGenerator:
    """
    Autoregressive generation with *manually managed* KV cache.

    Instead of relying on ``model.generate(use_cache=True)`` we:

    1. Perform a prefill pass with the full prompt to obtain the initial
       ``past_key_values`` (a tuple of (key, value) tensors per layer).
    2. At each subsequent decode step we feed **only the latest token**
       together with the cached KV tensors.  The model returns updated
       ``past_key_values`` which we store for the next iteration.
    3. We correctly maintain ``attention_mask`` (grows by 1 each step) and
       ``position_ids`` (set to the next position index) so that the causal
       mask and rotary embeddings remain correct.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def generate(self, prompt: str) -> GenerationResult:
        """
        Greedy-decode *max_new_tokens* tokens using an explicitly-managed
        KV cache.  Only the new token is fed to the model at each step.

        Returns
        -------
        GenerationResult
            Generated token IDs, decoded text, and timing metrics.
        """
        result = GenerationResult()
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids: torch.Tensor = encoded["input_ids"]  # (1, prompt_len)
        attention_mask: torch.Tensor = encoded["attention_mask"]  # (1, prompt_len)
        prompt_len = input_ids.shape[1]

        # ---- Prefill: full prompt → first next-token + initial KV cache ----
        _reset_peak_memory()
        _sync_cuda()
        t_prefill_start = time.perf_counter()

        # Position IDs for the full prompt
        position_ids = torch.arange(prompt_len, device=self.device).unsqueeze(0)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        logits = outputs.logits  # (1, prompt_len, vocab)
        past_key_values = outputs.past_key_values  # tuple of (key, value) per layer

        # Convert DynamicCache or similar to a plain list so we manage it ourselves
        past_key_values = self._to_list_of_tuples(past_key_values)

        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (1, 1)

        _sync_cuda()
        t_prefill_end = time.perf_counter()
        result.prefill_time_s = t_prefill_end - t_prefill_start

        generated_ids: List[int] = [next_token.item()]
        current_pos = prompt_len  # next position index

        # Extend attention mask for the newly generated token
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=self.device, dtype=attention_mask.dtype)],
            dim=-1,
        )

        # ---- Decode steps: feed only the new token each time ----
        _sync_cuda()
        t_decode_start = time.perf_counter()

        eos_id = self.tokenizer.eos_token_id
        for _ in range(self.max_new_tokens - 1):
            position_ids = torch.tensor([[current_pos]], device=self.device)

            # Reconstruct past_key_values as a tuple-of-tuples for the model
            pkv_tuple = tuple(
                (layer_k, layer_v) for layer_k, layer_v in past_key_values
            )

            outputs = self.model(
                input_ids=next_token,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=pkv_tuple,
                use_cache=True,
            )
            logits = outputs.logits  # (1, 1, vocab)
            past_key_values = self._to_list_of_tuples(outputs.past_key_values)

            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            token_id = next_token.item()
            generated_ids.append(token_id)
            current_pos += 1

            # Grow attention mask
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), device=self.device, dtype=attention_mask.dtype)],
                dim=-1,
            )

            if eos_id is not None and token_id == eos_id:
                break

        _sync_cuda()
        t_decode_end = time.perf_counter()

        result.decode_time_s = t_decode_end - t_decode_start
        result.total_time_s = result.prefill_time_s + result.decode_time_s
        result.token_ids = generated_ids
        result.num_generated_tokens = len(generated_ids)
        result.text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        result.peak_memory_mb = _peak_gpu_memory_mb()
        if result.total_time_s > 0:
            result.tokens_per_sec = result.num_generated_tokens / result.total_time_s

        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _to_list_of_tuples(
        past_key_values: Any,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Convert whatever cache object the model returns into a plain
        ``list[(key, value)]`` so we can manage it ourselves.

        Handles:
        * ``transformers.cache_utils.DynamicCache`` – has ``.key_cache`` /
          ``.value_cache`` attributes.
        * Plain tuple-of-tuples (older transformers).
        """
        # DynamicCache from transformers >= 4.36
        if hasattr(past_key_values, "key_cache") and hasattr(
            past_key_values, "value_cache"
        ):
            return [
                (k, v)
                for k, v in zip(
                    past_key_values.key_cache, past_key_values.value_cache
                )
            ]

        # Already a tuple / list of (key, value) per layer
        return [(layer[0], layer[1]) for layer in past_key_values]


# ============================================================================
# Verification
# ============================================================================
def verify_equivalence(
    model_name: str = DEFAULT_MODEL_NAME,
    prompts: Optional[List[str]] = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> bool:
    """
    Run both cached and uncached generators on multiple prompts of different
    lengths and assert that they produce *identical* token sequences.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    prompts : list[str], optional
        Override the default sample prompts.
    max_new_tokens : int
        How many tokens to generate per prompt.

    Returns
    -------
    bool
        True if all prompts produce token-level identical outputs.
    """
    if prompts is None:
        prompts = SAMPLE_PROMPTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[verify_equivalence] Loading model '{model_name}' on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    uncached_gen = UncachedGenerator(model, tokenizer, max_new_tokens=max_new_tokens)
    cached_gen = KVCachedGenerator(model, tokenizer, max_new_tokens=max_new_tokens)

    all_match = True
    print(f"\n{'='*80}")
    print("KV-Cache Equivalence Verification")
    print(f"{'='*80}")
    print(f"{'Prompt':<60} {'Tokens':>6} {'Match':>6}")
    print(f"{'-'*60} {'-'*6} {'-'*6}")

    for idx, prompt in enumerate(prompts):
        uncached_result = uncached_gen.generate(prompt)
        cached_result = cached_gen.generate(prompt)

        match = uncached_result.token_ids == cached_result.token_ids
        all_match = all_match and match
        status = "✓" if match else "✗"
        short_prompt = prompt[:55] + "..." if len(prompt) > 55 else prompt
        n_tokens = len(uncached_result.token_ids)
        print(f"  {short_prompt:<58} {n_tokens:>6} {status:>6}")

        if not match:
            print(f"    Uncached ({len(uncached_result.token_ids)} tokens): "
                  f"{uncached_result.token_ids[:20]}...")
            print(f"    Cached   ({len(cached_result.token_ids)} tokens): "
                  f"{cached_result.token_ids[:20]}...")
            # Find first divergence
            for i, (u, c) in enumerate(
                zip(uncached_result.token_ids, cached_result.token_ids)
            ):
                if u != c:
                    print(f"    First mismatch at position {i}: "
                          f"uncached={u}, cached={c}")
                    break

    print(f"\n{'='*80}")
    print(f"Overall: {'ALL MATCH ✓' if all_match else 'MISMATCH FOUND ✗'}")
    print(f"{'='*80}\n")
    return all_match


# ============================================================================
# Benchmark
# ============================================================================
def benchmark(
    model_name: str = DEFAULT_MODEL_NAME,
    max_new_tokens: int = MAX_NEW_TOKENS,
    warmup_iters: int = WARMUP_ITERS,
    bench_iters: int = BENCH_ITERS,
    save_figure: bool = True,
) -> Dict[str, Any]:
    """
    Benchmark uncached vs. KV-cached generation.

    Measures prefill time, decode time, tokens/sec, and peak GPU memory.
    Uses warmup iterations and ``torch.cuda.synchronize()`` for accuracy.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    max_new_tokens : int
        Number of tokens to generate per run.
    warmup_iters : int
        Warmup iterations (discarded).
    bench_iters : int
        Timed iterations to average.
    save_figure : bool
        If True, save comparison chart to ``reports/figures/kv_cache_benchmark.png``.

    Returns
    -------
    dict
        Benchmark results for both generators.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[benchmark] Loading model '{model_name}' on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    prompt = SAMPLE_PROMPTS[0]
    uncached_gen = UncachedGenerator(model, tokenizer, max_new_tokens=max_new_tokens)
    cached_gen = KVCachedGenerator(model, tokenizer, max_new_tokens=max_new_tokens)

    results: Dict[str, Dict[str, List[float]]] = {
        "uncached": {"prefill": [], "decode": [], "tps": [], "memory": []},
        "cached": {"prefill": [], "decode": [], "tps": [], "memory": []},
    }

    for label, gen in [("uncached", uncached_gen), ("cached", cached_gen)]:
        print(f"  Warming up {label} generator ({warmup_iters} iters) ...")
        for _ in range(warmup_iters):
            gen.generate(prompt)

        print(f"  Benchmarking {label} generator ({bench_iters} iters) ...")
        for i in range(bench_iters):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            r = gen.generate(prompt)
            results[label]["prefill"].append(r.prefill_time_s)
            results[label]["decode"].append(r.decode_time_s)
            results[label]["tps"].append(r.tokens_per_sec)
            results[label]["memory"].append(r.peak_memory_mb)

    # Aggregate means
    summary: Dict[str, Any] = {}
    for label in ("uncached", "cached"):
        summary[label] = {
            "prefill_time_s": float(np.mean(results[label]["prefill"])),
            "decode_time_s": float(np.mean(results[label]["decode"])),
            "total_time_s": float(
                np.mean(results[label]["prefill"])
                + np.mean(results[label]["decode"])
            ),
            "tokens_per_sec": float(np.mean(results[label]["tps"])),
            "peak_memory_mb": float(np.mean(results[label]["memory"])),
        }

    # Pretty-print table
    print(f"\n{'='*72}")
    print("KV-Cache Benchmark Results")
    print(f"{'='*72}")
    print(
        f"{'Metric':<25} {'Uncached':>20} {'KV-Cached':>20}"
    )
    print(f"{'-'*25} {'-'*20} {'-'*20}")
    for metric, fmt in [
        ("prefill_time_s", ".4f"),
        ("decode_time_s", ".4f"),
        ("total_time_s", ".4f"),
        ("tokens_per_sec", ".1f"),
        ("peak_memory_mb", ".1f"),
    ]:
        u_val = summary["uncached"][metric]
        c_val = summary["cached"][metric]
        print(f"  {metric:<23} {u_val:>20{fmt}} {c_val:>20{fmt}}")
    speedup = (
        summary["uncached"]["total_time_s"] / summary["cached"]["total_time_s"]
        if summary["cached"]["total_time_s"] > 0
        else float("inf")
    )
    print(f"\n  Speedup (total time): {speedup:.2f}x")
    print(f"{'='*72}\n")

    summary["speedup"] = speedup

    # ---- Plot ----
    if save_figure:
        _plot_benchmark(summary)

    return summary


def _plot_benchmark(summary: Dict[str, Any]) -> None:
    """Create and save a benchmark comparison figure."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    labels = ["Uncached", "KV-Cached"]
    colors = ["#e74c3c", "#2ecc71"]

    # 1) Latency breakdown (stacked bar)
    ax = axes[0]
    prefill = [summary["uncached"]["prefill_time_s"], summary["cached"]["prefill_time_s"]]
    decode = [summary["uncached"]["decode_time_s"], summary["cached"]["decode_time_s"]]
    x = np.arange(len(labels))
    width = 0.5
    ax.bar(x, prefill, width, label="Prefill", color="#3498db")
    ax.bar(x, decode, width, bottom=prefill, label="Decode", color="#e67e22")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Time (s)")
    ax.set_title("Latency Breakdown")
    ax.legend()

    # 2) Throughput
    ax = axes[1]
    tps = [summary["uncached"]["tokens_per_sec"], summary["cached"]["tokens_per_sec"]]
    bars = ax.bar(x, tps, width, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Tokens / sec")
    ax.set_title("Throughput")
    for bar, val in zip(bars, tps):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 3) Peak GPU memory
    ax = axes[2]
    mem = [summary["uncached"]["peak_memory_mb"], summary["cached"]["peak_memory_mb"]]
    bars = ax.bar(x, mem, width, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Peak Memory (MiB)")
    ax.set_title("Peak GPU Memory")
    for bar, val in zip(bars, mem):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    fig.suptitle(
        f"KV-Cache Benchmark  ·  Speedup: {summary.get('speedup', 0):.2f}×",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    out_path = FIGURES_DIR / "kv_cache_benchmark.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[benchmark] Figure saved to {out_path}")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    print("=" * 72)
    print("Task 3 – KV-Cached Autoregressive Generation")
    print("=" * 72)

    # 1) Equivalence check
    equiv = verify_equivalence()

    # 2) Benchmark
    bench_results = benchmark()

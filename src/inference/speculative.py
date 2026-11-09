"""
Task 5: Greedy Speculative Decoding

Implements a draft–target speculative decoding pipeline where a small *draft*
model (e.g. Qwen2.5-0.5B-Instruct) proposes K candidate tokens and a larger
*target* model (e.g. Qwen2.5-1.5B-Instruct) verifies them in a single
batched forward pass.

Key features:
  • Greedy drafting and greedy verification (argmax).
  • Proper KV-cache rollback / synchronisation after rejection.
  • Token-level equivalence with standard greedy target-only decoding.
  • Benchmarking of acceptance rate (α), latency, throughput, and memory.

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
# Config defaults
# ---------------------------------------------------------------------------
DEFAULT_DRAFT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_TARGET_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_K = 5            # draft tokens per step
MAX_NEW_TOKENS = 64
WARMUP_ITERS = 2
BENCH_ITERS = 3
FIGURES_DIR = Path("reports/figures")

SAMPLE_PROMPTS = [
    "Summarize the eligibility criteria for this clinical trial:",
    "The patient is a 45-year-old male diagnosed with type 2 diabetes. Determine eligibility:",
    "List the inclusion and exclusion criteria for the following oncology study involving patients with non-small cell lung cancer:",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _peak_mem_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def _reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _load_model(name: str, device: str):
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=dtype, device_map=device, trust_remote_code=True,
    )
    model.eval()
    return model


def _cache_to_list(past_kv):
    """Normalise DynamicCache / tuple-of-tuples to list[(K, V)]."""
    if hasattr(past_kv, "key_cache") and hasattr(past_kv, "value_cache"):
        return [(k, v) for k, v in zip(past_kv.key_cache, past_kv.value_cache)]
    return [(layer[0], layer[1]) for layer in past_kv]


def _truncate_cache(cache, seq_len):
    """Truncate each layer's K/V to *seq_len* on the sequence dimension."""
    return [(k[:, :, :seq_len, :], v[:, :, :seq_len, :]) for k, v in cache]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class SpecResult:
    token_ids: List[int] = field(default_factory=list)
    text: str = ""
    total_time_s: float = 0.0
    tokens_per_sec: float = 0.0
    peak_memory_mb: float = 0.0
    num_generated: int = 0
    num_accepted: int = 0
    num_draft_rounds: int = 0
    acceptance_rate: float = 0.0


# ============================================================================
# Standard (target-only) greedy generator – used as reference
# ============================================================================
class TargetOnlyGenerator:
    """Greedy generation with explicit KV-cache, target model only."""

    def __init__(self, model, tokenizer, max_new_tokens=MAX_NEW_TOKENS):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def generate(self, prompt: str) -> SpecResult:
        res = SpecResult()
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = enc["input_ids"]
        attn_mask = enc["attention_mask"]
        prompt_len = input_ids.shape[1]

        _reset_peak(); _sync()
        t0 = time.perf_counter()

        # Prefill
        pos_ids = torch.arange(prompt_len, device=self.device).unsqueeze(0)
        out = self.model(input_ids=input_ids, attention_mask=attn_mask,
                         position_ids=pos_ids, use_cache=True)
        past = _cache_to_list(out.past_key_values)
        next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
        generated = [next_tok.item()]

        attn_mask = torch.cat([attn_mask, torch.ones(1, 1, device=self.device,
                                                      dtype=attn_mask.dtype)], dim=-1)
        cur_pos = prompt_len
        eos = self.tokenizer.eos_token_id

        # Decode
        for _ in range(self.max_new_tokens - 1):
            pos_ids = torch.tensor([[cur_pos]], device=self.device)
            pkv = tuple((k, v) for k, v in past)
            out = self.model(input_ids=next_tok, attention_mask=attn_mask,
                             position_ids=pos_ids, past_key_values=pkv,
                             use_cache=True)
            past = _cache_to_list(out.past_key_values)
            next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
            tid = next_tok.item()
            generated.append(tid)
            cur_pos += 1
            attn_mask = torch.cat([attn_mask, torch.ones(1, 1, device=self.device,
                                                          dtype=attn_mask.dtype)], dim=-1)
            if eos is not None and tid == eos:
                break

        _sync()
        res.total_time_s = time.perf_counter() - t0
        res.token_ids = generated
        res.num_generated = len(generated)
        res.text = self.tokenizer.decode(generated, skip_special_tokens=True)
        res.peak_memory_mb = _peak_mem_mb()
        if res.total_time_s > 0:
            res.tokens_per_sec = res.num_generated / res.total_time_s
        return res


# ============================================================================
# Speculative Decoder
# ============================================================================
class SpeculativeDecoder:
    """
    Greedy speculative decoding with a draft/target model pair.

    Algorithm per iteration:
      1. The draft model generates K candidate tokens autoregressively.
      2. ALL K candidates are fed to the target model in a single forward pass.
      3. The target model's greedy choice at each position is compared with
         the draft token.  Tokens are accepted up to the first mismatch.
      4. At the rejection point the target's token is used instead.
      5. Both models' KV caches are rolled back/synchronised to the accepted
         prefix length.
      6. If ALL K tokens are accepted an additional bonus token is obtained
         from the target model's last logit.
    """

    def __init__(
        self,
        draft_model_name: str = DEFAULT_DRAFT_MODEL,
        target_model_name: str = DEFAULT_TARGET_MODEL,
        K: int = DEFAULT_K,
        max_new_tokens: int = MAX_NEW_TOKENS,
        device: Optional[str] = None,
    ):
        self.K = K
        self.max_new_tokens = max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[SpeculativeDecoder] Loading tokenizer from {target_model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            target_model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"[SpeculativeDecoder] Loading draft model: {draft_model_name} ...")
        self.draft_model = _load_model(draft_model_name, self.device)

        print(f"[SpeculativeDecoder] Loading target model: {target_model_name} ...")
        self.target_model = _load_model(target_model_name, self.device)

        self.eos_id = self.tokenizer.eos_token_id
        print("[SpeculativeDecoder] Ready.")

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, prompt: str) -> SpecResult:
        """Full speculative decoding loop."""
        res = SpecResult()
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = enc["input_ids"]       # (1, prompt_len)
        prompt_len = input_ids.shape[1]

        _reset_peak(); _sync()
        t0 = time.perf_counter()

        # ---------- Prefill both models ----------
        draft_out = self.draft_model(input_ids=input_ids, use_cache=True)
        draft_cache = _cache_to_list(draft_out.past_key_values)

        target_out = self.target_model(input_ids=input_ids, use_cache=True)
        target_cache = _cache_to_list(target_out.past_key_values)

        # First token from target model
        next_tok_id = target_out.logits[:, -1, :].argmax(-1).item()
        generated: List[int] = [next_tok_id]

        # Sync draft cache – also feed the first token through draft
        draft_inp = torch.tensor([[next_tok_id]], device=self.device)
        draft_out = self.draft_model(input_ids=draft_inp,
                                      past_key_values=tuple((k, v) for k, v in draft_cache),
                                      use_cache=True)
        draft_cache = _cache_to_list(draft_out.past_key_values)

        # Target cache already contains the prefix; add the first generated token
        target_inp = torch.tensor([[next_tok_id]], device=self.device)
        target_out = self.target_model(input_ids=target_inp,
                                        past_key_values=tuple((k, v) for k, v in target_cache),
                                        use_cache=True)
        target_cache = _cache_to_list(target_out.past_key_values)

        total_accepted = 0
        total_draft_rounds = 0

        # ---------- Main speculative loop ----------
        while len(generated) < self.max_new_tokens:
            # Check for EOS
            if self.eos_id is not None and generated[-1] == self.eos_id:
                break

            # Step 1: Draft K tokens autoregressively with the draft model
            draft_tokens: List[int] = []
            draft_cache_snapshot = [(k.clone(), v.clone()) for k, v in draft_cache]

            cur_tok = torch.tensor([[generated[-1]]], device=self.device)
            temp_draft_cache = [(k, v) for k, v in draft_cache]

            for _ in range(self.K):
                # Get draft model's next-token prediction
                d_logits = draft_out.logits[:, -1, :]
                d_tok_id = d_logits.argmax(-1).item()
                draft_tokens.append(d_tok_id)

                if self.eos_id is not None and d_tok_id == self.eos_id:
                    break

                # Feed draft token to draft model
                d_inp = torch.tensor([[d_tok_id]], device=self.device)
                draft_out = self.draft_model(
                    input_ids=d_inp,
                    past_key_values=tuple((k, v) for k, v in temp_draft_cache),
                    use_cache=True,
                )
                temp_draft_cache = _cache_to_list(draft_out.past_key_values)

            if not draft_tokens:
                break

            total_draft_rounds += 1

            # Step 2: Verify ALL draft tokens with target model in one pass
            verify_ids = torch.tensor([draft_tokens], device=self.device)  # (1, num_draft)
            target_out = self.target_model(
                input_ids=verify_ids,
                past_key_values=tuple((k, v) for k, v in target_cache),
                use_cache=True,
            )
            target_logits = target_out.logits  # (1, num_draft, vocab)
            new_target_cache = _cache_to_list(target_out.past_key_values)

            # Step 3: Greedy verification – accept prefix where target agrees
            n_accepted = 0
            for i in range(len(draft_tokens)):
                target_tok = target_logits[0, i, :].argmax(-1).item()
                if target_tok == draft_tokens[i]:
                    n_accepted += 1
                else:
                    # Rejection: use target's token at the rejection point
                    generated.append(target_tok)
                    total_accepted += n_accepted
                    break
            else:
                # ALL K tokens accepted — append all + bonus token from last logit
                total_accepted += n_accepted
                for tok in draft_tokens:
                    generated.append(tok)
                # Bonus token from target's last position
                bonus = target_logits[0, -1, :].argmax(-1).item()
                generated.append(bonus)

                # Update both caches to include all accepted + bonus
                draft_cache = temp_draft_cache
                target_cache = new_target_cache

                # Feed bonus into both models
                bonus_inp = torch.tensor([[bonus]], device=self.device)
                draft_out = self.draft_model(
                    input_ids=bonus_inp,
                    past_key_values=tuple((k, v) for k, v in draft_cache),
                    use_cache=True,
                )
                draft_cache = _cache_to_list(draft_out.past_key_values)

                target_out_b = self.target_model(
                    input_ids=bonus_inp,
                    past_key_values=tuple((k, v) for k, v in target_cache),
                    use_cache=True,
                )
                target_cache = _cache_to_list(target_out_b.past_key_values)
                continue

            # Step 4: KV-cache rollback on rejection
            # We accepted n_accepted tokens from the draft.
            # Target cache now has prefix + all draft tokens; truncate to prefix + accepted + 1 (the target correction)
            cache_len = prompt_len + len(generated)  # total valid length
            target_cache = _truncate_cache(new_target_cache, cache_len)

            # Append accepted tokens before the rejection token
            for tok in draft_tokens[:n_accepted]:
                pass  # These are implicitly handled — the rejection token was already appended

            # Wait, we need to actually add the accepted draft tokens to generated
            # The rejection token is already added above; add the accepted ones before it
            rejection_tok = generated.pop()  # temporarily remove rejection token
            for tok in draft_tokens[:n_accepted]:
                generated.append(tok)
            generated.append(rejection_tok)  # re-add

            # Roll back draft cache to the snapshot and replay accepted tokens + correction
            draft_cache = draft_cache_snapshot
            replay_tokens = draft_tokens[:n_accepted] + [rejection_tok]
            replay_inp = torch.tensor([replay_tokens], device=self.device)
            draft_out = self.draft_model(
                input_ids=replay_inp,
                past_key_values=tuple((k, v) for k, v in draft_cache),
                use_cache=True,
            )
            draft_cache = _cache_to_list(draft_out.past_key_values)

            # Recalculate target cache length
            target_cache = _truncate_cache(target_cache, prompt_len + len(generated))

            # Feed correction token through target to sync its cache
            corr_inp = torch.tensor([[rejection_tok]], device=self.device)
            # Target cache was truncated; we need to re-feed from accepted prefix
            # Actually we already have the cache up to n_accepted drafts + prompt
            # Truncate more carefully
            trunc_len = prompt_len + len(generated) - 1  # everything before correction
            target_cache_trunc = _truncate_cache(new_target_cache, trunc_len)
            target_out_c = self.target_model(
                input_ids=corr_inp,
                past_key_values=tuple((k, v) for k, v in target_cache_trunc),
                use_cache=True,
            )
            target_cache = _cache_to_list(target_out_c.past_key_values)

        # Trim to max_new_tokens
        generated = generated[:self.max_new_tokens]

        _sync()
        res.total_time_s = time.perf_counter() - t0
        res.token_ids = generated
        res.num_generated = len(generated)
        res.text = self.tokenizer.decode(generated, skip_special_tokens=True)
        res.peak_memory_mb = _peak_mem_mb()
        res.num_accepted = total_accepted
        res.num_draft_rounds = total_draft_rounds
        total_drafted = total_draft_rounds * self.K
        res.acceptance_rate = total_accepted / max(total_drafted, 1)
        if res.total_time_s > 0:
            res.tokens_per_sec = res.num_generated / res.total_time_s
        return res


# ============================================================================
# Equivalence verification
# ============================================================================
def verify_equivalence(
    draft_model_name: str = DEFAULT_DRAFT_MODEL,
    target_model_name: str = DEFAULT_TARGET_MODEL,
    prompts: Optional[List[str]] = None,
    K: int = DEFAULT_K,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> bool:
    """
    Prove that speculative decoding produces IDENTICAL tokens to standard
    greedy target-only decoding.
    """
    if prompts is None:
        prompts = SAMPLE_PROMPTS

    print(f"\n{'='*80}")
    print("Speculative Decoding – Token Equivalence Verification")
    print(f"{'='*80}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load target model for reference generation
    tokenizer = AutoTokenizer.from_pretrained(target_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    target_model = _load_model(target_model_name, device)
    ref_gen = TargetOnlyGenerator(target_model, tokenizer, max_new_tokens)

    # Speculative decoder
    spec_decoder = SpeculativeDecoder(
        draft_model_name=draft_model_name,
        target_model_name=target_model_name,
        K=K, max_new_tokens=max_new_tokens, device=device,
    )

    all_match = True
    print(f"\n  {'Prompt':<55} {'RefLen':>7} {'SpecLen':>8} {'Match':>6}")
    print(f"  {'-'*55} {'-'*7} {'-'*8} {'-'*6}")

    for prompt in prompts:
        ref_result = ref_gen.generate(prompt)
        spec_result = spec_decoder.generate(prompt)

        match = ref_result.token_ids == spec_result.token_ids
        all_match = all_match and match
        status = "✓" if match else "✗"
        short = prompt[:52] + "..." if len(prompt) > 52 else prompt
        print(f"  {short:<55} {len(ref_result.token_ids):>7} "
              f"{len(spec_result.token_ids):>8} {status:>6}")

        if not match:
            # Find divergence
            for i, (r, s) in enumerate(zip(ref_result.token_ids, spec_result.token_ids)):
                if r != s:
                    print(f"    First mismatch at pos {i}: ref={r}, spec={s}")
                    break

    print(f"\n  Overall: {'ALL MATCH ✓' if all_match else 'MISMATCH FOUND ✗'}")
    print(f"{'='*80}\n")

    # Clean up
    del target_model, spec_decoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return all_match


# ============================================================================
# Benchmark
# ============================================================================
def benchmark(
    draft_model_name: str = DEFAULT_DRAFT_MODEL,
    target_model_name: str = DEFAULT_TARGET_MODEL,
    K: int = DEFAULT_K,
    max_new_tokens: int = MAX_NEW_TOKENS,
    warmup_iters: int = WARMUP_ITERS,
    bench_iters: int = BENCH_ITERS,
    save_figure: bool = True,
) -> Dict[str, Any]:
    """
    Benchmark speculative decoding vs. standard target-only generation.

    Measures acceptance rate (α), end-to-end latency, throughput, and memory.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(target_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    target_model = _load_model(target_model_name, device)

    ref_gen = TargetOnlyGenerator(target_model, tokenizer, max_new_tokens)
    spec_decoder = SpeculativeDecoder(
        draft_model_name, target_model_name, K, max_new_tokens, device,
    )

    prompt = SAMPLE_PROMPTS[0]
    results: Dict[str, Dict[str, List[float]]] = {
        "target_only": {"time": [], "tps": [], "mem": []},
        "speculative": {"time": [], "tps": [], "mem": [], "alpha": []},
    }

    for label, gen in [("target_only", ref_gen), ("speculative", spec_decoder)]:
        print(f"  Warming up {label} ({warmup_iters} iters) ...")
        for _ in range(warmup_iters):
            gen.generate(prompt)

        print(f"  Benchmarking {label} ({bench_iters} iters) ...")
        for _ in range(bench_iters):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            r = gen.generate(prompt)
            results[label]["time"].append(r.total_time_s)
            results[label]["tps"].append(r.tokens_per_sec)
            results[label]["mem"].append(r.peak_memory_mb)
            if hasattr(r, "acceptance_rate"):
                results[label].setdefault("alpha", []).append(r.acceptance_rate)

    summary: Dict[str, Any] = {}
    for label in ("target_only", "speculative"):
        summary[label] = {
            "latency_s": float(np.mean(results[label]["time"])),
            "tokens_per_sec": float(np.mean(results[label]["tps"])),
            "peak_memory_mb": float(np.mean(results[label]["mem"])),
        }
    if results["speculative"].get("alpha"):
        summary["speculative"]["acceptance_rate"] = float(
            np.mean(results["speculative"]["alpha"])
        )

    # Pretty print
    print(f"\n{'='*72}")
    print("Speculative Decoding Benchmark Results")
    print(f"{'='*72}")
    print(f"  {'Metric':<25} {'Target-Only':>18} {'Speculative':>18}")
    print(f"  {'-'*25} {'-'*18} {'-'*18}")
    for m, fmt in [("latency_s", ".4f"), ("tokens_per_sec", ".1f"), ("peak_memory_mb", ".1f")]:
        t_val = summary["target_only"][m]
        s_val = summary["speculative"][m]
        print(f"  {m:<25} {t_val:>18{fmt}} {s_val:>18{fmt}}")
    alpha = summary["speculative"].get("acceptance_rate", 0)
    print(f"  {'acceptance_rate (α)':<25} {'N/A':>18} {alpha:>18.3f}")

    speedup = summary["target_only"]["latency_s"] / max(summary["speculative"]["latency_s"], 1e-9)
    summary["speedup"] = speedup
    print(f"\n  Speedup: {speedup:.2f}x")
    print(f"{'='*72}\n")

    if save_figure:
        _plot_spec_benchmark(summary)

    # Clean up
    del target_model, spec_decoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def _plot_spec_benchmark(summary: Dict[str, Any]) -> None:
    """Save speculative decoding benchmark figure."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    labels = ["Target-Only", "Speculative"]
    colors = ["#3498db", "#e74c3c"]

    # Latency
    ax = axes[0]
    vals = [summary["target_only"]["latency_s"], summary["speculative"]["latency_s"]]
    bars = ax.bar(labels, vals, color=colors, width=0.5)
    ax.set_ylabel("Latency (s)")
    ax.set_title("End-to-End Latency")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")

    # Throughput
    ax = axes[1]
    vals = [summary["target_only"]["tokens_per_sec"], summary["speculative"]["tokens_per_sec"]]
    bars = ax.bar(labels, vals, color=colors, width=0.5)
    ax.set_ylabel("Tokens / sec")
    ax.set_title("Throughput")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.5, f"{v:.1f}", ha="center", fontweight="bold")

    # Acceptance rate
    ax = axes[2]
    alpha = summary["speculative"].get("acceptance_rate", 0)
    ax.bar(["Acceptance Rate (α)"], [alpha], color="#2ecc71", width=0.4)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate")
    ax.set_title(f"Draft Acceptance Rate: {alpha:.1%}")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="50%")
    ax.legend()

    fig.suptitle(
        f"Speculative Decoding Benchmark  ·  Speedup: {summary.get('speedup', 0):.2f}×",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    out = FIGURES_DIR / "speculative_benchmark.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[benchmark] Figure saved to {out}")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    print("=" * 72)
    print("Task 5 – Greedy Speculative Decoding")
    print("=" * 72)

    equiv = verify_equivalence()
    bench_results = benchmark()

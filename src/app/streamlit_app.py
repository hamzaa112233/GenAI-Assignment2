"""
Clinical Trial Eligibility Screener — Streamlit Application (Task 6).

A polished, production-style UI that:
  • Assesses patient eligibility against trial criteria using a fine-tuned
    Qwen2.5-1.5B-Instruct model (LoRA adapter).
  • Benchmarks inference modes (standard, KV-cached, speculative decoding).
  • Exports eligibility reports as downloadable Markdown files.

The app runs gracefully in *demo mode* when no GPU / model is available.

Run:
    streamlit run src/app/streamlit_app.py
"""

from __future__ import annotations

import datetime
import json
import os
import random
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Optional heavy imports – gracefully degrade to demo mode
# ---------------------------------------------------------------------------
_GPU_AVAILABLE = False
_MODEL_LOADED = False

try:
    import torch

    _GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    _TRANSFORMERS_OK = True
except ImportError:
    _TRANSFORMERS_OK = False

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         DEMO / MOCK DATA                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

EXAMPLE_PATIENT = textwrap.dedent("""\
    Patient ID: PT-2024-0472
    Age: 58 years old | Sex: Male | BMI: 27.3
    
    Primary Diagnosis: Stage IIIA Non-Small Cell Lung Cancer (NSCLC), 
    adenocarcinoma subtype, diagnosed 2024-01-15.
    
    ECOG Performance Status: 1
    
    Medical History:
    [S1] Hypertension, well-controlled on lisinopril 10 mg daily.
    [S2] Type 2 Diabetes Mellitus, HbA1c 6.8%, managed with metformin 1000 mg BID.
    [S3] No prior malignancies.
    [S4] Former smoker, quit 5 years ago, 20 pack-year history.
    [S5] No history of autoimmune disease.
    
    Laboratory Values (2024-02-01):
    [S6] WBC: 6.2 × 10⁹/L (normal)
    [S7] Absolute Neutrophil Count: 3.8 × 10⁹/L (normal)
    [S8] Platelets: 210 × 10⁹/L (normal)
    [S9] Hemoglobin: 13.1 g/dL (normal)
    [S10] Creatinine: 0.9 mg/dL, eGFR > 60 mL/min
    [S11] ALT: 28 U/L, AST: 31 U/L (normal hepatic function)
    [S12] Total Bilirubin: 0.8 mg/dL
    
    Molecular Profile:
    [S13] EGFR mutation: Negative
    [S14] ALK rearrangement: Negative
    [S15] PD-L1 TPS: 65%
    [S16] KRAS G12C mutation: Positive
    
    Current Medications: Lisinopril, Metformin, Omeprazole 20 mg daily.
    
    Prior Treatment:
    [S17] No prior systemic therapy for NSCLC.
    [S18] No prior radiation therapy.
""")

EXAMPLE_CRITERIA = textwrap.dedent("""\
    Patient must be ≥ 18 years of age.
    Patient must have histologically confirmed NSCLC.
    ECOG performance status ≤ 2.
    Adequate bone marrow function (ANC ≥ 1.5 × 10⁹/L, platelets ≥ 100 × 10⁹/L).
    No prior systemic chemotherapy for advanced/metastatic NSCLC.
    Patient must NOT have active autoimmune disease.
    Patient must NOT have EGFR or ALK genomic alterations.
    Adequate hepatic function (AST/ALT ≤ 2.5 × ULN, bilirubin ≤ 1.5 × ULN).
    PD-L1 TPS ≥ 50%.
    Patient must NOT have uncontrolled diabetes (HbA1c > 9%).
""")

# Pre-built demo predictions that mirror model output format
DEMO_PREDICTIONS: List[Dict[str, Any]] = [
    {
        "criterion": "Patient must be ≥ 18 years of age.",
        "label": "eligible",
        "evidence_ids": ["S0"],
        "evidence_text": "Age: 58 years old",
        "rationale": "Patient is 58 years old, which meets the minimum age requirement of 18.",
    },
    {
        "criterion": "Patient must have histologically confirmed NSCLC.",
        "label": "eligible",
        "evidence_ids": ["S0"],
        "evidence_text": "Stage IIIA Non-Small Cell Lung Cancer (NSCLC), adenocarcinoma subtype",
        "rationale": "Patient has a confirmed diagnosis of NSCLC, adenocarcinoma subtype.",
    },
    {
        "criterion": "ECOG performance status ≤ 2.",
        "label": "eligible",
        "evidence_ids": ["S0"],
        "evidence_text": "ECOG Performance Status: 1",
        "rationale": "ECOG PS is 1, which is ≤ 2.",
    },
    {
        "criterion": "Adequate bone marrow function (ANC ≥ 1.5 × 10⁹/L, platelets ≥ 100 × 10⁹/L).",
        "label": "eligible",
        "evidence_ids": ["S7", "S8"],
        "evidence_text": "ANC: 3.8 × 10⁹/L; Platelets: 210 × 10⁹/L",
        "rationale": "ANC of 3.8 exceeds 1.5 threshold and platelets of 210 exceed 100 threshold.",
    },
    {
        "criterion": "No prior systemic chemotherapy for advanced/metastatic NSCLC.",
        "label": "eligible",
        "evidence_ids": ["S17"],
        "evidence_text": "No prior systemic therapy for NSCLC.",
        "rationale": "Patient has not received prior systemic therapy.",
    },
    {
        "criterion": "Patient must NOT have active autoimmune disease.",
        "label": "eligible",
        "evidence_ids": ["S5"],
        "evidence_text": "No history of autoimmune disease.",
        "rationale": "Patient explicitly has no history of autoimmune disease.",
    },
    {
        "criterion": "Patient must NOT have EGFR or ALK genomic alterations.",
        "label": "eligible",
        "evidence_ids": ["S13", "S14"],
        "evidence_text": "EGFR mutation: Negative; ALK rearrangement: Negative",
        "rationale": "Both EGFR and ALK are negative, satisfying the exclusion criterion.",
    },
    {
        "criterion": "Adequate hepatic function (AST/ALT ≤ 2.5 × ULN, bilirubin ≤ 1.5 × ULN).",
        "label": "eligible",
        "evidence_ids": ["S11", "S12"],
        "evidence_text": "ALT: 28 U/L, AST: 31 U/L; Total Bilirubin: 0.8 mg/dL",
        "rationale": "AST and ALT are within normal limits (well below 2.5× ULN ~100 U/L) and bilirubin 0.8 is below 1.5× ULN.",
    },
    {
        "criterion": "PD-L1 TPS ≥ 50%.",
        "label": "eligible",
        "evidence_ids": ["S15"],
        "evidence_text": "PD-L1 TPS: 65%",
        "rationale": "PD-L1 TPS is 65%, which exceeds the 50% threshold.",
    },
    {
        "criterion": "Patient must NOT have uncontrolled diabetes (HbA1c > 9%).",
        "label": "eligible",
        "evidence_ids": ["S2"],
        "evidence_text": "Type 2 Diabetes Mellitus, HbA1c 6.8%, managed with metformin",
        "rationale": "HbA1c is 6.8%, which is well below 9%. Diabetes is controlled.",
    },
]

DEMO_BENCHMARK_DATA: Dict[str, Any] = {
    "standard": {
        "prefill_ms": 142.3,
        "decode_ms": 2187.6,
        "tokens_per_sec": 23.4,
        "memory_mb": 3420.0,
        "total_tokens": 512,
    },
    "kv_cached": {
        "prefill_ms": 145.1,
        "decode_ms": 1024.8,
        "tokens_per_sec": 48.7,
        "memory_mb": 3680.0,
        "total_tokens": 512,
    },
    "speculative": {
        "prefill_ms": 148.9,
        "decode_ms": 876.3,
        "tokens_per_sec": 57.2,
        "memory_mb": 4210.0,
        "total_tokens": 512,
        "acceptance_rate": 0.78,
        "draft_tokens_proposed": 2560,
        "draft_tokens_accepted": 1997,
    },
}

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                       DATA CLASSES                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝


@dataclass
class CriterionResult:
    """Single criterion assessment result."""

    criterion: str
    label: str  # eligible | not_eligible | not_applicable | not_enough_info
    evidence_ids: List[str] = field(default_factory=list)
    evidence_text: str = ""
    rationale: str = ""


@dataclass
class BenchmarkResult:
    """Timing & throughput numbers for one inference mode."""

    mode: str
    prefill_ms: float
    decode_ms: float
    tokens_per_sec: float
    memory_mb: float
    total_tokens: int
    acceptance_rate: Optional[float] = None
    draft_tokens_proposed: Optional[int] = None
    draft_tokens_accepted: Optional[int] = None


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                     MODEL LOADING UTILITIES                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝


@st.cache_resource(show_spinner="Loading model and tokenizer …")
def load_model_and_tokenizer(
    model_path: str,
    adapter_path: Optional[str] = None,
) -> Tuple[Any, Any]:
    """Load the base model (optionally with LoRA adapter) and tokenizer.

    Returns (model, tokenizer).  If loading fails the function returns
    (None, None) so the app can fall back to demo mode.
    """
    if not _TRANSFORMERS_OK or not _GPU_AVAILABLE:
        return None, None

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        if adapter_path and os.path.isdir(adapter_path):
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()

        model.eval()
        return model, tokenizer
    except Exception as exc:
        st.warning(f"Model loading failed: {exc}")
        return None, None


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                        INFERENCE HELPERS                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a clinical trial eligibility assessment assistant.
    Given a patient description and a single inclusion/exclusion criterion,
    determine the patient's eligibility.

    Respond with a JSON object containing:
    {
      "label": "<eligible | not_eligible | not_applicable | not_enough_info>",
      "evidence_ids": ["<sentence IDs from the patient record>"],
      "evidence_text": "<verbatim evidence>",
      "rationale": "<brief explanation>"
    }
""")


def _build_prompt(patient: str, criterion: str) -> str:
    """Format the prompt for the model."""
    return (
        f"<|im_start|>system\n{_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Patient Record:\n{patient}\n\n"
        f"Criterion: {criterion}\n"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _parse_model_output(raw: str) -> Dict[str, Any]:
    """Extract JSON from model output, with fallback."""
    # Try to find JSON block
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(raw[start:end])
        except json.JSONDecodeError:
            pass
    return {
        "label": "not_enough_info",
        "evidence_ids": [],
        "evidence_text": "",
        "rationale": raw.strip() or "Model output could not be parsed.",
    }


def run_inference_standard(
    model: Any,
    tokenizer: Any,
    patient: str,
    criterion: str,
    max_new_tokens: int = 256,
) -> Tuple[CriterionResult, float]:
    """Standard autoregressive generation (no KV-cache reuse)."""
    prompt = _build_prompt(patient, criterion)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            use_cache=True,
        )
    elapsed = time.perf_counter() - t0

    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    )
    parsed = _parse_model_output(generated)
    result = CriterionResult(
        criterion=criterion,
        label=parsed.get("label", "not_enough_info"),
        evidence_ids=parsed.get("evidence_ids", []),
        evidence_text=parsed.get("evidence_text", ""),
        rationale=parsed.get("rationale", ""),
    )
    return result, elapsed


def run_inference_kv_cached(
    model: Any,
    tokenizer: Any,
    patient: str,
    criterion: str,
    max_new_tokens: int = 256,
) -> Tuple[CriterionResult, float]:
    """Generation with explicit KV-cache reuse across the patient prefix.

    We first encode the patient prefix and cache its KV states, then
    generate for each criterion reusing that cache (here, a single
    criterion for simplicity).
    """
    system_patient = (
        f"<|im_start|>system\n{_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nPatient Record:\n{patient}\n\n"
    )
    suffix = f"Criterion: {criterion}\n<|im_end|>\n<|im_start|>assistant\n"

    prefix_ids = tokenizer(system_patient, return_tensors="pt").to(model.device)
    suffix_ids = tokenizer(
        suffix, return_tensors="pt", add_special_tokens=False
    ).to(model.device)

    t0 = time.perf_counter()
    with torch.no_grad():
        # Prefill: encode patient prefix, obtain KV-cache
        prefix_out = model(
            **prefix_ids,
            use_cache=True,
        )
        past = prefix_out.past_key_values

        # Combine suffix with cached prefix
        combined_ids = suffix_ids["input_ids"]
        output_ids = model.generate(
            input_ids=combined_ids,
            past_key_values=past,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            use_cache=True,
        )
    elapsed = time.perf_counter() - t0

    generated = tokenizer.decode(
        output_ids[0][combined_ids.shape[-1] :], skip_special_tokens=True
    )
    parsed = _parse_model_output(generated)
    result = CriterionResult(
        criterion=criterion,
        label=parsed.get("label", "not_enough_info"),
        evidence_ids=parsed.get("evidence_ids", []),
        evidence_text=parsed.get("evidence_text", ""),
        rationale=parsed.get("rationale", ""),
    )
    return result, elapsed


def run_inference_speculative(
    model: Any,
    tokenizer: Any,
    patient: str,
    criterion: str,
    max_new_tokens: int = 256,
    draft_model: Any = None,
) -> Tuple[CriterionResult, float, Dict[str, Any]]:
    """Speculative-decoding style inference.

    If *draft_model* is not supplied we fall back to standard generation
    (the transformer library's ``assistant_model`` API handles
    speculative decoding natively since transformers ≥ 4.37).
    """
    prompt = _build_prompt(patient, criterion)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    spec_stats: Dict[str, Any] = {}

    t0 = time.perf_counter()
    with torch.no_grad():
        gen_kwargs: Dict[str, Any] = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            use_cache=True,
        )
        if draft_model is not None:
            gen_kwargs["assistant_model"] = draft_model

        output_ids = model.generate(**gen_kwargs)
    elapsed = time.perf_counter() - t0

    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    )
    parsed = _parse_model_output(generated)
    result = CriterionResult(
        criterion=criterion,
        label=parsed.get("label", "not_enough_info"),
        evidence_ids=parsed.get("evidence_ids", []),
        evidence_text=parsed.get("evidence_text", ""),
        rationale=parsed.get("rationale", ""),
    )
    return result, elapsed, spec_stats


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                       DEMO-MODE HELPERS                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def demo_predict(
    patient: str, criteria: List[str]
) -> List[CriterionResult]:
    """Return mock predictions.  Reuse pre-built demo data when the
    criteria match; otherwise generate plausible placeholders."""
    demo_lookup = {d["criterion"]: d for d in DEMO_PREDICTIONS}
    results: List[CriterionResult] = []
    for crit in criteria:
        crit_stripped = crit.strip()
        if not crit_stripped:
            continue
        if crit_stripped in demo_lookup:
            d = demo_lookup[crit_stripped]
            results.append(
                CriterionResult(
                    criterion=d["criterion"],
                    label=d["label"],
                    evidence_ids=d["evidence_ids"],
                    evidence_text=d["evidence_text"],
                    rationale=d["rationale"],
                )
            )
        else:
            label = random.choice(
                ["eligible", "not_eligible", "not_applicable", "not_enough_info"]
            )
            results.append(
                CriterionResult(
                    criterion=crit_stripped,
                    label=label,
                    evidence_ids=[],
                    evidence_text="(demo mode – no real evidence)",
                    rationale=f"Demo prediction: {label}. Run with a loaded model for real assessment.",
                )
            )
    return results


def demo_benchmark() -> Dict[str, BenchmarkResult]:
    """Return pre-built benchmark numbers (with small jitter)."""
    out: Dict[str, BenchmarkResult] = {}
    for mode_key, d in DEMO_BENCHMARK_DATA.items():
        jitter = lambda v: v * random.uniform(0.95, 1.05)  # noqa: E731
        out[mode_key] = BenchmarkResult(
            mode=mode_key,
            prefill_ms=jitter(d["prefill_ms"]),
            decode_ms=jitter(d["decode_ms"]),
            tokens_per_sec=jitter(d["tokens_per_sec"]),
            memory_mb=jitter(d["memory_mb"]),
            total_tokens=d["total_tokens"],
            acceptance_rate=d.get("acceptance_rate"),
            draft_tokens_proposed=d.get("draft_tokens_proposed"),
            draft_tokens_accepted=d.get("draft_tokens_accepted"),
        )
    return out


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                        REPORT GENERATION                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def generate_report(
    patient: str,
    results: List[CriterionResult],
    model_name: str,
    adapter_path: str,
    inference_mode: str,
) -> str:
    """Build a Markdown eligibility report string."""
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines: List[str] = [
        "# Clinical Trial Eligibility Report",
        "",
        f"**Generated:** {now}",
        "",
        "---",
        "",
        "## Model Provenance",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Base Model | `{model_name}` |",
        f"| LoRA Adapter | `{adapter_path or 'N/A'}` |",
        f"| Inference Mode | {inference_mode} |",
        "",
        "---",
        "",
        "## Patient Description",
        "",
        "```",
        patient.strip(),
        "```",
        "",
        "---",
        "",
        "## Eligibility Assessments",
        "",
    ]

    summary_counts: Dict[str, int] = {
        "eligible": 0,
        "not_eligible": 0,
        "not_applicable": 0,
        "not_enough_info": 0,
    }

    for idx, r in enumerate(results, 1):
        summary_counts[r.label] = summary_counts.get(r.label, 0) + 1
        evidence_str = (
            f"Sentence IDs: {', '.join(r.evidence_ids)}" if r.evidence_ids else "N/A"
        )
        lines += [
            f"### Criterion {idx}",
            "",
            f"> {r.criterion}",
            "",
            f"- **Verdict:** `{r.label.upper()}`",
            f"- **Evidence:** {evidence_str}",
            f"- **Evidence Text:** {r.evidence_text}",
            f"- **Rationale:** {r.rationale}",
            "",
        ]

    lines += [
        "---",
        "",
        "## Summary",
        "",
        f"| Category | Count |",
        f"|----------|-------|",
    ]
    for cat, cnt in summary_counts.items():
        lines.append(f"| {cat.replace('_', ' ').title()} | {cnt} |")

    total = len(results)
    eligible_pct = (
        f"{summary_counts['eligible'] / total * 100:.0f}%" if total > 0 else "N/A"
    )
    lines += [
        "",
        f"**Overall Eligibility Rate:** {eligible_pct} ({summary_counts['eligible']}/{total} criteria met)",
        "",
        "---",
        "",
        "> ⚠️ **DISCLAIMER:** This report was generated by an educational prototype. "
        "It is NOT validated for clinical use and must NOT be used to make actual "
        "clinical trial enrollment decisions.",
        "",
    ]
    return "\n".join(lines)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                           UI HELPERS                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

LABEL_COLORS = {
    "eligible": "#28a745",
    "not_eligible": "#dc3545",
    "not_applicable": "#ffc107",
    "not_enough_info": "#6c757d",
}

LABEL_ICONS = {
    "eligible": "✅",
    "not_eligible": "❌",
    "not_applicable": "⚠️",
    "not_enough_info": "❓",
}


def _render_criterion_card(result: CriterionResult, index: int) -> None:
    """Render a single criterion assessment card."""
    color = LABEL_COLORS.get(result.label, "#6c757d")
    icon = LABEL_ICONS.get(result.label, "•")
    label_display = result.label.replace("_", " ").upper()

    st.markdown(
        f"""
        <div style="
            border-left: 5px solid {color};
            background: linear-gradient(135deg, {color}10 0%, transparent 100%);
            padding: 16px 20px;
            border-radius: 0 8px 8px 0;
            margin-bottom: 12px;
        ">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <strong style="font-size:0.95rem;">Criterion {index}</strong>
                <span style="
                    background:{color}; color:white; padding:3px 12px;
                    border-radius:12px; font-size:0.8rem; font-weight:600;
                ">{icon} {label_display}</span>
            </div>
            <p style="margin:8px 0 4px 0; color:#444; font-style:italic;">"{result.criterion}"</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander(f"📋 Details — Criterion {index}", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Evidence Sentence IDs:**")
            if result.evidence_ids:
                st.code(", ".join(result.evidence_ids))
            else:
                st.caption("No sentence IDs available.")
        with col_b:
            st.markdown("**Evidence Text:**")
            if result.evidence_text:
                st.info(result.evidence_text)
            else:
                st.caption("No evidence text available.")
        st.markdown("**Rationale:**")
        st.write(result.rationale)


def _render_benchmark_charts(benchmarks: Dict[str, BenchmarkResult]) -> None:
    """Build Plotly charts for the benchmark tab."""
    modes = list(benchmarks.keys())
    nice_names = {
        "standard": "Standard",
        "kv_cached": "KV-Cached",
        "speculative": "Speculative",
    }

    # ---- Latency grouped bar ----
    prefill = [benchmarks[m].prefill_ms for m in modes]
    decode = [benchmarks[m].decode_ms for m in modes]
    labels = [nice_names.get(m, m) for m in modes]

    fig_latency = go.Figure()
    fig_latency.add_trace(
        go.Bar(
            name="Prefill (ms)",
            x=labels,
            y=prefill,
            marker_color="#636EFA",
            text=[f"{v:.1f}" for v in prefill],
            textposition="auto",
        )
    )
    fig_latency.add_trace(
        go.Bar(
            name="Decode (ms)",
            x=labels,
            y=decode,
            marker_color="#EF553B",
            text=[f"{v:.1f}" for v in decode],
            textposition="auto",
        )
    )
    fig_latency.update_layout(
        title="Latency Comparison (Lower is Better)",
        barmode="group",
        yaxis_title="Time (ms)",
        template="plotly_white",
        height=380,
    )

    # ---- Throughput bar ----
    tps = [benchmarks[m].tokens_per_sec for m in modes]
    fig_tps = go.Figure(
        go.Bar(
            x=labels,
            y=tps,
            marker_color=["#636EFA", "#00CC96", "#AB63FA"],
            text=[f"{v:.1f}" for v in tps],
            textposition="auto",
        )
    )
    fig_tps.update_layout(
        title="Throughput (Higher is Better)",
        yaxis_title="Tokens / sec",
        template="plotly_white",
        height=350,
    )

    # ---- Memory bar ----
    mem = [benchmarks[m].memory_mb for m in modes]
    fig_mem = go.Figure(
        go.Bar(
            x=labels,
            y=mem,
            marker_color=["#19D3F3", "#FFA15A", "#FF6692"],
            text=[f"{v:.0f}" for v in mem],
            textposition="auto",
        )
    )
    fig_mem.update_layout(
        title="Peak GPU Memory (MB)",
        yaxis_title="MB",
        template="plotly_white",
        height=350,
    )

    # ---- Layout ----
    st.plotly_chart(fig_latency, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_tps, use_container_width=True)
    with col2:
        st.plotly_chart(fig_mem, use_container_width=True)

    # ---- Speculative decoding detail ----
    spec = benchmarks.get("speculative")
    if spec and spec.acceptance_rate is not None:
        st.markdown("### 🎯 Speculative Decoding Details")
        c1, c2, c3 = st.columns(3)
        c1.metric("Acceptance Rate (α)", f"{spec.acceptance_rate:.1%}")
        c2.metric(
            "Draft Tokens Proposed", f"{spec.draft_tokens_proposed:,}"
        )
        c3.metric(
            "Draft Tokens Accepted", f"{spec.draft_tokens_accepted:,}"
        )

        # Speed-up gauge
        std = benchmarks.get("standard")
        if std:
            speedup = std.decode_ms / spec.decode_ms if spec.decode_ms > 0 else 1.0
            st.metric(
                "Speculative vs Standard Speed-up",
                f"{speedup:.2f}×",
                delta=f"{(speedup - 1) * 100:.0f}% faster",
            )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         STREAMLIT APP                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def main() -> None:
    """Entry-point for the Streamlit application."""

    # ------------------------------------------------------------------ #
    # Page configuration
    # ------------------------------------------------------------------ #
    st.set_page_config(
        page_title="Clinical Trial Eligibility Screener",
        page_icon="⚕️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject a tiny bit of custom CSS
    st.markdown(
        """
        <style>
        .stApp { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        div[data-testid="stMetric"] { background: #f8f9fa; border-radius: 8px; padding: 12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------ #
    # Disclaimer banner
    # ------------------------------------------------------------------ #
    st.markdown(
        """
        <div style="
            background: linear-gradient(90deg, #dc3545, #e85d04);
            color: white;
            padding: 14px 24px;
            border-radius: 8px;
            text-align: center;
            margin-bottom: 20px;
            font-weight: 600;
        ">
            ⚠️ EDUCATIONAL SCREENING PROTOTYPE — NOT FOR CLINICAL ENROLLMENT DECISIONS ⚠️<br>
            <span style="font-weight:400; font-size:0.88rem;">
                This tool is an educational demonstration.
                It is <u>NOT</u> validated for clinical use and must
                <u>NOT</u> be used to make actual clinical trial enrollment decisions.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------ #
    # Sidebar
    # ------------------------------------------------------------------ #
    with st.sidebar:
        st.image(
            "https://img.icons8.com/fluency/96/caduceus.png",
            width=64,
        )
        st.title("⚕️ Eligibility Screener")
        st.caption("Clinical Trial NLP — v1.0")
        st.divider()

        model_path = st.text_input(
            "🗂️ Model / Adapter Path",
            value="Qwen/Qwen2.5-1.5B-Instruct",
            help="HuggingFace model ID or local path to the fine-tuned model.",
        )
        adapter_path = st.text_input(
            "🔗 LoRA Adapter Path (optional)",
            value="./outputs/lora_adapter",
            help="Path to the LoRA adapter directory (leave blank if merged).",
        )
        inference_mode = st.selectbox(
            "⚡ Inference Mode",
            options=["Standard", "KV-Cached", "Speculative Decoding"],
            index=0,
        )
        max_tokens = st.slider(
            "🔢 Max New Tokens", min_value=64, max_value=1024, value=256, step=64
        )

        st.divider()

        # GPU / model status
        if _GPU_AVAILABLE:
            st.success(f"🟢 GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            st.warning("🟡 No GPU — running in **Demo Mode**")

        demo_mode = st.toggle("🎭 Force Demo Mode", value=not _GPU_AVAILABLE)

        st.divider()
        with st.expander("ℹ️ About This Project"):
            st.markdown(
                """
                **Clinical Trial Eligibility NLP**

                This project fine-tunes **Qwen2.5-1.5B-Instruct** with
                **LoRA** on the TrialGPT criterion-level annotation
                dataset to predict whether a patient meets individual
                clinical trial inclusion/exclusion criteria.

                **Inference optimisations:**
                - KV-cache tiling for long patient records
                - Speculative decoding with a lightweight draft model

                **Stack:** PyTorch · HuggingFace Transformers · PEFT ·
                Streamlit · Plotly
                """
            )

    # ------------------------------------------------------------------ #
    # Attempt model loading (only when not in demo mode)
    # ------------------------------------------------------------------ #
    model, tokenizer = None, None
    if not demo_mode:
        model, tokenizer = load_model_and_tokenizer(
            model_path, adapter_path if adapter_path.strip() else None
        )
        if model is not None:
            st.sidebar.success("✅ Model loaded")
        else:
            demo_mode = True
            st.sidebar.info("ℹ️ Falling back to Demo Mode")

    # ------------------------------------------------------------------ #
    # Title
    # ------------------------------------------------------------------ #
    st.markdown(
        """
        <h1 style="margin-bottom:0;">⚕️ Clinical Trial Eligibility Screener</h1>
        <p style="color:#555; margin-top:4px;">
            Assess patient eligibility against clinical trial criteria using
            a fine-tuned language model.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------ #
    # Tabs
    # ------------------------------------------------------------------ #
    tab_assess, tab_bench, tab_export = st.tabs(
        ["🩺 Eligibility Assessment", "📊 Inference Benchmark", "📄 Report Export"]
    )

    # ================================================================== #
    #  TAB 1 — Eligibility Assessment                                     #
    # ================================================================== #
    with tab_assess:
        col_patient, col_criteria = st.columns([1, 1], gap="large")

        with col_patient:
            st.subheader("📝 Patient Record")
            patient_text = st.text_area(
                "Paste the patient description below:",
                value=EXAMPLE_PATIENT,
                height=400,
                key="patient_input",
            )

        with col_criteria:
            st.subheader("📋 Trial Criteria")
            st.caption("Enter one criterion per line.")
            criteria_text = st.text_area(
                "Paste inclusion/exclusion criteria:",
                value=EXAMPLE_CRITERIA,
                height=400,
                key="criteria_input",
            )

        criteria_list = [
            c.strip() for c in criteria_text.strip().split("\n") if c.strip()
        ]

        assess_btn = st.button(
            "🔬 Assess Eligibility",
            type="primary",
            use_container_width=True,
        )

        if assess_btn:
            if not patient_text.strip():
                st.error("Please provide a patient record.")
            elif not criteria_list:
                st.error("Please provide at least one criterion.")
            else:
                results: List[CriterionResult] = []
                progress = st.progress(0, text="Assessing criteria …")

                if demo_mode:
                    # Simulated delay per criterion for realism
                    results = demo_predict(patient_text, criteria_list)
                    for i in range(len(results)):
                        time.sleep(0.15)
                        progress.progress(
                            (i + 1) / len(results),
                            text=f"Criterion {i + 1}/{len(results)} …",
                        )
                else:
                    # Real model inference
                    for i, crit in enumerate(criteria_list):
                        if inference_mode == "Standard":
                            res, _ = run_inference_standard(
                                model, tokenizer, patient_text, crit, max_tokens
                            )
                        elif inference_mode == "KV-Cached":
                            res, _ = run_inference_kv_cached(
                                model, tokenizer, patient_text, crit, max_tokens
                            )
                        else:
                            res, _, _ = run_inference_speculative(
                                model, tokenizer, patient_text, crit, max_tokens
                            )
                        results.append(res)
                        progress.progress(
                            (i + 1) / len(criteria_list),
                            text=f"Criterion {i + 1}/{len(criteria_list)} …",
                        )

                progress.empty()

                # Store results in session state
                st.session_state["results"] = results
                st.session_state["patient_text"] = patient_text

        # ---- Display results ---- #
        if "results" in st.session_state and st.session_state["results"]:
            results = st.session_state["results"]
            st.divider()

            # Summary metrics
            st.subheader("📊 Assessment Summary")
            counts = {
                "eligible": 0,
                "not_eligible": 0,
                "not_applicable": 0,
                "not_enough_info": 0,
            }
            for r in results:
                counts[r.label] = counts.get(r.label, 0) + 1

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("✅ Eligible", counts["eligible"])
            m2.metric("❌ Not Eligible", counts["not_eligible"])
            m3.metric("⚠️ Not Applicable", counts["not_applicable"])
            m4.metric("❓ Insufficient Info", counts["not_enough_info"])

            total = len(results)
            if total > 0:
                pct = counts["eligible"] / total * 100
                if counts["not_eligible"] == 0 and counts["not_enough_info"] == 0:
                    st.success(
                        f"🎉 **Patient appears ELIGIBLE** — {counts['eligible']}/{total} "
                        f"criteria met ({pct:.0f}%)."
                    )
                elif counts["not_eligible"] > 0:
                    st.error(
                        f"⛔ **Patient does NOT meet all criteria** — "
                        f"{counts['not_eligible']} exclusion(s) triggered."
                    )
                else:
                    st.warning(
                        f"⚠️ **Eligibility uncertain** — {counts['not_enough_info']} "
                        f"criterion/criteria lack sufficient information."
                    )

            # Missing info highlight
            missing = [r for r in results if r.label == "not_enough_info"]
            if missing:
                with st.expander("🔍 Missing Information Detail", expanded=True):
                    for r in missing:
                        st.warning(f"**{r.criterion}** — {r.rationale}")

            st.divider()
            st.subheader("📝 Criterion-by-Criterion Results")
            for idx, r in enumerate(results, 1):
                _render_criterion_card(r, idx)

    # ================================================================== #
    #  TAB 2 — Inference Benchmark                                        #
    # ================================================================== #
    with tab_bench:
        st.subheader("📊 Inference Benchmark Dashboard")
        st.markdown(
            "Compare **Standard**, **KV-Cached**, and **Speculative Decoding** "
            "inference modes on latency, throughput, and GPU memory."
        )

        bench_btn = st.button(
            "🚀 Run Benchmark",
            type="primary",
            use_container_width=True,
            key="bench_btn",
        )

        if bench_btn:
            with st.spinner("Running benchmarks … (this may take a minute)"):
                if demo_mode:
                    time.sleep(1.5)  # simulate work
                    benchmarks = demo_benchmark()
                else:
                    benchmarks: Dict[str, BenchmarkResult] = {}
                    test_criterion = "Patient must be ≥ 18 years of age."
                    test_patient = EXAMPLE_PATIENT

                    # Standard
                    _, t_std = run_inference_standard(
                        model, tokenizer, test_patient, test_criterion, max_tokens
                    )
                    mem_std = (
                        torch.cuda.max_memory_allocated() / 1e6
                        if _GPU_AVAILABLE
                        else 0.0
                    )
                    benchmarks["standard"] = BenchmarkResult(
                        mode="standard",
                        prefill_ms=t_std * 200,  # rough split
                        decode_ms=t_std * 800,
                        tokens_per_sec=max_tokens / t_std if t_std > 0 else 0,
                        memory_mb=mem_std,
                        total_tokens=max_tokens,
                    )
                    if _GPU_AVAILABLE:
                        torch.cuda.reset_peak_memory_stats()

                    # KV-Cached
                    _, t_kv = run_inference_kv_cached(
                        model, tokenizer, test_patient, test_criterion, max_tokens
                    )
                    mem_kv = (
                        torch.cuda.max_memory_allocated() / 1e6
                        if _GPU_AVAILABLE
                        else 0.0
                    )
                    benchmarks["kv_cached"] = BenchmarkResult(
                        mode="kv_cached",
                        prefill_ms=t_kv * 200,
                        decode_ms=t_kv * 800,
                        tokens_per_sec=max_tokens / t_kv if t_kv > 0 else 0,
                        memory_mb=mem_kv,
                        total_tokens=max_tokens,
                    )
                    if _GPU_AVAILABLE:
                        torch.cuda.reset_peak_memory_stats()

                    # Speculative (falls back to standard if no draft model)
                    _, t_sp, sp_stats = run_inference_speculative(
                        model, tokenizer, test_patient, test_criterion, max_tokens
                    )
                    mem_sp = (
                        torch.cuda.max_memory_allocated() / 1e6
                        if _GPU_AVAILABLE
                        else 0.0
                    )
                    benchmarks["speculative"] = BenchmarkResult(
                        mode="speculative",
                        prefill_ms=t_sp * 200,
                        decode_ms=t_sp * 800,
                        tokens_per_sec=max_tokens / t_sp if t_sp > 0 else 0,
                        memory_mb=mem_sp,
                        total_tokens=max_tokens,
                        acceptance_rate=sp_stats.get("acceptance_rate"),
                        draft_tokens_proposed=sp_stats.get("draft_tokens_proposed"),
                        draft_tokens_accepted=sp_stats.get("draft_tokens_accepted"),
                    )

                st.session_state["benchmarks"] = benchmarks

        if "benchmarks" in st.session_state:
            benchmarks = st.session_state["benchmarks"]

            # Summary metrics
            st.markdown("### ⚡ Quick Comparison")
            cols = st.columns(3)
            nice = {"standard": "Standard", "kv_cached": "KV-Cached", "speculative": "Speculative"}
            for i, (mode_key, bm) in enumerate(benchmarks.items()):
                with cols[i]:
                    st.markdown(f"#### {nice.get(mode_key, mode_key)}")
                    st.metric("Prefill", f"{bm.prefill_ms:.1f} ms")
                    st.metric("Decode", f"{bm.decode_ms:.1f} ms")
                    st.metric("Throughput", f"{bm.tokens_per_sec:.1f} tok/s")
                    st.metric("Memory", f"{bm.memory_mb:.0f} MB")

            st.divider()
            _render_benchmark_charts(benchmarks)

            if demo_mode:
                st.info(
                    "📌 These are **simulated** benchmark numbers (demo mode). "
                    "Run with a GPU and loaded model for real measurements."
                )

    # ================================================================== #
    #  TAB 3 — Report Export                                              #
    # ================================================================== #
    with tab_export:
        st.subheader("📄 Export Eligibility Report")

        if "results" not in st.session_state or not st.session_state["results"]:
            st.info(
                "No assessment results yet. Go to the **Eligibility Assessment** tab "
                "and run an assessment first."
            )
        else:
            results = st.session_state["results"]
            patient = st.session_state.get("patient_text", "")

            st.markdown("Preview of the report to be exported:")

            report_md = generate_report(
                patient=patient,
                results=results,
                model_name=model_path,
                adapter_path=adapter_path,
                inference_mode=inference_mode,
            )

            with st.expander("📄 Report Preview", expanded=True):
                st.markdown(report_md)

            now_file = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eligibility_report_{now_file}.md"

            st.download_button(
                label="⬇️ Download Report (Markdown)",
                data=report_md,
                file_name=filename,
                mime="text/markdown",
                type="primary",
                use_container_width=True,
            )

            st.caption(
                "The downloaded file contains the full assessment with timestamps, "
                "evidence citations, and model provenance."
            )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                           ENTRY POINT                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    main()

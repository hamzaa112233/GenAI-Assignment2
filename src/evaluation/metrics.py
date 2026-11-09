# src/evaluation/metrics.py
"""
Evaluation metrics for Clinical Trial Eligibility NLP (Task 2).

Computes classification metrics (multi-class, NOT binary), evidence retrieval
metrics, confusion matrices, error analysis, and a full evaluation report.

Label space:  eligible | not_eligible | not_applicable | not_enough_info
"""

from __future__ import annotations

import json
import os
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for Colab / headless

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABEL_NAMES: List[str] = [
    "eligible",
    "not_eligible",
    "not_applicable",
    "not_enough_info",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def _ensure_dirs() -> None:
    """Create output directories if they do not already exist."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ===================================================================
# (a)  Classification Metrics  –  4-class, NOT collapsed to binary
# ===================================================================


def compute_classification_metrics(
    y_true: List[str],
    y_pred: List[str],
    label_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute multi-class classification metrics.

    Parameters
    ----------
    y_true : list[str]
        Ground-truth labels (one of the four eligibility categories).
    y_pred : list[str]
        Predicted labels.
    label_names : list[str] | None
        Ordered label names.  Defaults to ``LABEL_NAMES``.

    Returns
    -------
    dict
        {
            "accuracy": float,
            "macro_f1": float,
            "macro_precision": float,
            "macro_recall": float,
            "weighted_f1": float,
            "per_class": {
                "<label>": {"precision": …, "recall": …, "f1": …, "support": …}
            },
            "classification_report_text": str,
            "n_samples": int,
        }
    """
    if label_names is None:
        label_names = LABEL_NAMES

    # Validate inputs
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("y_true and y_pred must be non-empty.")
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} samples, "
            f"y_pred has {len(y_pred)}."
        )

    accuracy = accuracy_score(y_true, y_pred)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=label_names, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        y_true, y_pred, labels=label_names, average="weighted", zero_division=0
    )

    per_class_p, per_class_r, per_class_f, per_class_s = (
        precision_recall_fscore_support(
            y_true, y_pred, labels=label_names, average=None, zero_division=0
        )
    )

    per_class: Dict[str, Dict[str, float]] = {}
    for i, label in enumerate(label_names):
        per_class[label] = {
            "precision": float(per_class_p[i]),
            "recall": float(per_class_r[i]),
            "f1": float(per_class_f[i]),
            "support": int(per_class_s[i]),
        }

    report_text = classification_report(
        y_true,
        y_pred,
        labels=label_names,
        target_names=label_names,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "classification_report_text": report_text,
        "n_samples": len(y_true),
    }


# ===================================================================
# (b)  Confusion Matrices – overall, inclusion, exclusion
# ===================================================================


def _plot_confusion_matrix(
    cm: np.ndarray,
    label_names: List[str],
    title: str,
    save_path: Path,
) -> None:
    """Plot and save a single confusion matrix as a seaborn heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
        linewidths=0.5,
        linecolor="grey",
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def compute_confusion_matrices(
    y_true: List[str],
    y_pred: List[str],
    criteria_types: Optional[List[str]] = None,
    label_names: Optional[List[str]] = None,
    save_dir: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    """Compute and save overall / inclusion / exclusion confusion matrices.

    Parameters
    ----------
    y_true : list[str]
        Ground-truth labels.
    y_pred : list[str]
        Predicted labels.
    criteria_types : list[str] | None
        Parallel list with ``"inclusion"`` or ``"exclusion"`` for each sample.
        If *None*, only the overall matrix is produced.
    label_names : list[str] | None
        Ordered label names.  Defaults to ``LABEL_NAMES``.
    save_dir : Path | None
        Directory for saving figures.  Defaults to ``FIGURES_DIR``.

    Returns
    -------
    dict
        ``{"overall": np.ndarray, "inclusion": np.ndarray | None,
           "exclusion": np.ndarray | None}``
    """
    _ensure_dirs()
    if label_names is None:
        label_names = LABEL_NAMES
    if save_dir is None:
        save_dir = FIGURES_DIR

    # Overall
    cm_overall = confusion_matrix(y_true, y_pred, labels=label_names)
    _plot_confusion_matrix(
        cm_overall,
        label_names,
        "Confusion Matrix – Overall",
        save_dir / "confusion_matrix_overall.png",
    )

    result: Dict[str, Any] = {"overall": cm_overall}

    # Inclusion / Exclusion splits
    for split_name in ("inclusion", "exclusion"):
        if criteria_types is None:
            result[split_name] = None
            continue
        mask = [ct == split_name for ct in criteria_types]
        if not any(mask):
            result[split_name] = None
            continue
        yt = [y for y, m in zip(y_true, mask) if m]
        yp = [y for y, m in zip(y_pred, mask) if m]
        cm = confusion_matrix(yt, yp, labels=label_names)
        _plot_confusion_matrix(
            cm,
            label_names,
            f"Confusion Matrix – {split_name.capitalize()} Criteria",
            save_dir / f"confusion_matrix_{split_name}.png",
        )
        result[split_name] = cm

    return result


# ===================================================================
# (c)  Evidence Retrieval Metrics
# ===================================================================


def compute_evidence_metrics(
    gold_evidence: List[Set[str]],
    pred_evidence: List[Set[str]],
) -> Dict[str, Any]:
    """Compute micro and macro precision / recall / F1 for evidence sentences.

    Parameters
    ----------
    gold_evidence : list[set[str]]
        Ground-truth sentence IDs per sample.
    pred_evidence : list[set[str]]
        Predicted sentence IDs per sample.

    Returns
    -------
    dict
        Contains micro_precision, micro_recall, micro_f1,
        macro_precision, macro_recall, macro_f1, per_sample details,
        and counts.

    Notes
    -----
    Edge-case semantics:
    * gold=∅, pred=∅  →  TP=0, FP=0, FN=0  (correct – no evidence needed)
    * gold=∅, pred≠∅  →  TP=0, FP=|pred|, FN=0
    * gold≠∅, pred=∅  →  TP=0, FP=0, FN=|gold|
    """
    if len(gold_evidence) != len(pred_evidence):
        raise ValueError(
            f"Length mismatch: gold has {len(gold_evidence)} samples, "
            f"pred has {len(pred_evidence)}."
        )

    # Accumulators for micro
    total_tp = 0
    total_fp = 0
    total_fn = 0

    # Per-sample metrics for macro
    per_sample_p: List[float] = []
    per_sample_r: List[float] = []
    per_sample_f1: List[float] = []

    for gold, pred in zip(gold_evidence, pred_evidence):
        gold = set(gold)
        pred = set(pred)

        tp = len(gold & pred)
        fp = len(pred - gold)
        fn = len(gold - pred)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Per-sample precision / recall / F1
        if len(pred) == 0 and len(gold) == 0:
            # Nothing to retrieve, nothing retrieved → perfect
            p, r, f1 = 1.0, 1.0, 1.0
        elif len(pred) == 0:
            # Missed everything
            p, r, f1 = 0.0, 0.0, 0.0
        elif len(gold) == 0:
            # Predicted spurious evidence
            p, r, f1 = 0.0, 1.0, 0.0  # recall is vacuously 1, but F1=0
        else:
            p = tp / len(pred)
            r = tp / len(gold)
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        per_sample_p.append(p)
        per_sample_r.append(r)
        per_sample_f1.append(f1)

    # Micro-averaged
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (
        (2 * micro_p * micro_r / (micro_p + micro_r))
        if (micro_p + micro_r) > 0
        else 0.0
    )

    # Macro-averaged
    n = len(per_sample_p)
    macro_p = sum(per_sample_p) / n if n > 0 else 0.0
    macro_r = sum(per_sample_r) / n if n > 0 else 0.0
    macro_f1 = sum(per_sample_f1) / n if n > 0 else 0.0

    return {
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "total_tp": int(total_tp),
        "total_fp": int(total_fp),
        "total_fn": int(total_fn),
        "n_samples": n,
    }


# ===================================================================
# (d)  Error Analysis  –  8 distinct error categories
# ===================================================================


def error_analysis(
    samples: List[Dict[str, Any]],
    max_examples_per_category: int = 5,
    save_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Perform structured error analysis over model predictions.

    Parameters
    ----------
    samples : list[dict]
        Each dict has at minimum:
            - ``true_label`` (str)
            - ``pred_label`` (str)
            - ``gold_evidence`` (set[str] | list[str])
            - ``pred_evidence`` (set[str] | list[str])
            - ``criteria_type`` (str): ``"inclusion"`` or ``"exclusion"``
            - ``patient_description`` (str)
            - ``criterion_text`` (str)
            - ``criterion_id`` (str | int, optional)
    max_examples_per_category : int
        Maximum number of examples to store per error category.
    save_dir : Path | None
        Directory for saving the error analysis text file.

    Returns
    -------
    dict
        Structured error analysis report keyed by category name.
    """
    _ensure_dirs()
    if save_dir is None:
        save_dir = FIGURES_DIR

    # ------------------------------------------------------------------
    # Define the 8 error categories + collectors
    # ------------------------------------------------------------------
    categories: Dict[str, Dict[str, Any]] = {
        "eligible_vs_not_eligible": {
            "description": (
                "Model confused 'eligible' with 'not_eligible' or vice-versa.  "
                "This is the most clinically dangerous error because it directly "
                "affects whether a patient would be enrolled."
            ),
            "examples": [],
            "suggested_fixes": [
                "Add contrastive training examples that differ only in the negation of a criterion.",
                "Incorporate attention supervision to force focus on negation cues.",
                "Use chain-of-thought prompting to explicitly reason about eligibility.",
            ],
        },
        "not_applicable_confusion": {
            "description": (
                "Model either missed a 'not_applicable' label or wrongly predicted it.  "
                "This indicates the model struggles to recognise when a criterion "
                "simply does not apply to a patient."
            ),
            "examples": [],
            "suggested_fixes": [
                "Add explicit training examples where the criterion topic is absent from the patient note.",
                "Use keyword-overlap features as auxiliary signals.",
                "Prompt the model to first determine topic relevance before eligibility.",
            ],
        },
        "not_enough_info_confusion": {
            "description": (
                "Model either missed a 'not_enough_info' label or wrongly predicted it.  "
                "This error occurs when the model hallucinates information that is "
                "not present in the patient description."
            ),
            "examples": [],
            "suggested_fixes": [
                "Train with explicit 'information gap' annotations.",
                "Add a verification step asking 'Is this information present in the note?'.",
                "Increase dropout to reduce over-confident predictions on sparse evidence.",
            ],
        },
        "missing_evidence_recall": {
            "description": (
                "Model failed to retrieve one or more gold evidence sentences.  "
                "Indicates insufficient sensitivity to supporting text."
            ),
            "examples": [],
            "suggested_fixes": [
                "Lower the evidence selection threshold during inference.",
                "Use BM25 as a recall-boosting pre-filter before neural re-ranking.",
                "Train a dedicated evidence extractor with focal loss on rare positives.",
            ],
        },
        "spurious_evidence_precision": {
            "description": (
                "Model retrieved evidence sentences not in the gold set.  "
                "This can mislead clinicians reviewing the system output."
            ),
            "examples": [],
            "suggested_fixes": [
                "Raise the evidence selection threshold.",
                "Add a post-hoc entailment verifier to filter irrelevant sentences.",
                "Use contrastive evidence training with hard-negative sentences.",
            ],
        },
        "inclusion_exclusion_confusion": {
            "description": (
                "Model shows systematically different error patterns on inclusion "
                "vs. exclusion criteria.  E.g., tends to predict 'eligible' for "
                "exclusion criteria that should be 'not_eligible'."
            ),
            "examples": [],
            "suggested_fixes": [
                "Prepend the criterion type (inclusion / exclusion) as a special token.",
                "Use separate classification heads or adapters for inclusion vs. exclusion.",
                "Add explicit reasoning instructions about polarity reversal for exclusion criteria.",
            ],
        },
        "long_patient_truncation": {
            "description": (
                "Errors on samples whose patient description exceeds the model's "
                "effective context window, suggesting information loss from truncation."
            ),
            "examples": [],
            "suggested_fixes": [
                "Increase max_length or switch to a model with longer context.",
                "Summarise the patient description before feeding to the model.",
                "Use retrieval-augmented generation to select only relevant note sections.",
            ],
        },
        "complex_multi_condition": {
            "description": (
                "Errors on criteria containing multiple logical conditions "
                "(AND / OR / ranges / nested clauses).  The model fails to "
                "evaluate all conditions correctly."
            ),
            "examples": [],
            "suggested_fixes": [
                "Decompose multi-condition criteria into atomic sub-criteria.",
                "Train with chain-of-thought annotations that reason over each sub-condition.",
                "Use a rule-based post-processor for numeric range checks.",
            ],
        },
    }

    # Heuristic thresholds
    LONG_DESCRIPTION_THRESHOLD = 1500  # characters
    MULTI_CONDITION_KEYWORDS = [
        " and ",
        " or ",
        " AND ",
        " OR ",
        ">=",
        "<=",
        ">",
        "<",
        "between",
        "range",
        "either",
        "both",
        "none of",
        "all of",
        "at least",
        "no more than",
    ]

    # ------------------------------------------------------------------
    # Iterate over samples and classify errors
    # ------------------------------------------------------------------
    total_errors = 0
    label_error_counter: Counter = Counter()

    for idx, s in enumerate(samples):
        true_label = s.get("true_label", "")
        pred_label = s.get("pred_label", "")
        gold_ev = set(s.get("gold_evidence", []))
        pred_ev = set(s.get("pred_evidence", []))
        crit_type = s.get("criteria_type", "unknown")
        patient_desc = s.get("patient_description", "")
        crit_text = s.get("criterion_text", "")
        crit_id = s.get("criterion_id", idx)

        is_label_error = true_label != pred_label
        has_missing_ev = bool(gold_ev - pred_ev) and len(gold_ev) > 0
        has_spurious_ev = bool(pred_ev - gold_ev)

        if not is_label_error and not has_missing_ev and not has_spurious_ev:
            continue  # correct prediction

        total_errors += 1
        if is_label_error:
            label_error_counter[(true_label, pred_label)] += 1

        example_record = {
            "sample_index": idx,
            "criterion_id": crit_id,
            "true_label": true_label,
            "pred_label": pred_label,
            "gold_evidence": sorted(gold_ev),
            "pred_evidence": sorted(pred_ev),
            "criteria_type": crit_type,
            "patient_description_snippet": patient_desc[:300],
            "criterion_text_snippet": crit_text[:300],
        }

        # --- Category 1: eligible ↔ not_eligible ---
        if is_label_error and {true_label, pred_label} == {
            "eligible",
            "not_eligible",
        }:
            cat = categories["eligible_vs_not_eligible"]
            if len(cat["examples"]) < max_examples_per_category:
                cat["examples"].append(example_record)

        # --- Category 2: not_applicable confusion ---
        if is_label_error and (
            true_label == "not_applicable" or pred_label == "not_applicable"
        ):
            cat = categories["not_applicable_confusion"]
            if len(cat["examples"]) < max_examples_per_category:
                cat["examples"].append(example_record)

        # --- Category 3: not_enough_info confusion ---
        if is_label_error and (
            true_label == "not_enough_info" or pred_label == "not_enough_info"
        ):
            cat = categories["not_enough_info_confusion"]
            if len(cat["examples"]) < max_examples_per_category:
                cat["examples"].append(example_record)

        # --- Category 4: missing evidence (recall) ---
        if has_missing_ev:
            cat = categories["missing_evidence_recall"]
            rec = dict(example_record)
            rec["missing_sentences"] = sorted(gold_ev - pred_ev)
            if len(cat["examples"]) < max_examples_per_category:
                cat["examples"].append(rec)

        # --- Category 5: spurious evidence (precision) ---
        if has_spurious_ev:
            cat = categories["spurious_evidence_precision"]
            rec = dict(example_record)
            rec["spurious_sentences"] = sorted(pred_ev - gold_ev)
            if len(cat["examples"]) < max_examples_per_category:
                cat["examples"].append(rec)

        # --- Category 6: inclusion / exclusion confusion pattern ---
        if is_label_error and crit_type in ("inclusion", "exclusion"):
            # Check for polarity confusion: e.g., predicting eligible on an
            # exclusion criterion that should be not_eligible.
            if (
                (crit_type == "exclusion" and pred_label == "eligible")
                or (crit_type == "inclusion" and pred_label == "not_eligible")
            ):
                cat = categories["inclusion_exclusion_confusion"]
                if len(cat["examples"]) < max_examples_per_category:
                    cat["examples"].append(example_record)

        # --- Category 7: long patient description truncation ---
        if (
            is_label_error or has_missing_ev
        ) and len(patient_desc) > LONG_DESCRIPTION_THRESHOLD:
            cat = categories["long_patient_truncation"]
            rec = dict(example_record)
            rec["patient_description_length"] = len(patient_desc)
            if len(cat["examples"]) < max_examples_per_category:
                cat["examples"].append(rec)

        # --- Category 8: complex multi-condition criteria ---
        if is_label_error and any(
            kw in crit_text for kw in MULTI_CONDITION_KEYWORDS
        ):
            cat = categories["complex_multi_condition"]
            rec = dict(example_record)
            rec["detected_keywords"] = [
                kw for kw in MULTI_CONDITION_KEYWORDS if kw in crit_text
            ]
            if len(cat["examples"]) < max_examples_per_category:
                cat["examples"].append(rec)

    # ------------------------------------------------------------------
    # Build the report dictionary
    # ------------------------------------------------------------------
    report: Dict[str, Any] = {
        "total_samples": len(samples),
        "total_errors": total_errors,
        "error_rate": total_errors / len(samples) if len(samples) > 0 else 0.0,
        "label_confusion_counts": {
            f"{t} -> {p}": c for (t, p), c in label_error_counter.most_common()
        },
        "categories": {},
    }

    for cat_name, cat_data in categories.items():
        report["categories"][cat_name] = {
            "description": cat_data["description"],
            "n_examples_found": len(cat_data["examples"]),
            "examples": cat_data["examples"],
            "suggested_fixes": cat_data["suggested_fixes"],
        }

    # ------------------------------------------------------------------
    # Save human-readable text report
    # ------------------------------------------------------------------
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("ERROR ANALYSIS REPORT – Clinical Trial Eligibility")
    lines.append("=" * 80)
    lines.append(f"Total samples analysed : {report['total_samples']}")
    lines.append(f"Total errors           : {report['total_errors']}")
    lines.append(f"Error rate             : {report['error_rate']:.4f}")
    lines.append("")
    lines.append("--- Label confusion counts (true -> pred) ---")
    for pair, cnt in report["label_confusion_counts"].items():
        lines.append(f"  {pair}: {cnt}")
    lines.append("")

    for cat_name, cat_info in report["categories"].items():
        lines.append("-" * 80)
        lines.append(f"Category: {cat_name}")
        lines.append(f"  Description : {cat_info['description']}")
        lines.append(f"  Examples found: {cat_info['n_examples_found']}")
        if cat_info["examples"]:
            for i, ex in enumerate(cat_info["examples"][:3], 1):
                lines.append(f"    Example {i}:")
                lines.append(f"      True: {ex['true_label']}  Pred: {ex['pred_label']}")
                lines.append(
                    f"      Criterion: {ex.get('criterion_text_snippet', 'N/A')[:120]}"
                )
                lines.append(
                    f"      Patient: {ex.get('patient_description_snippet', 'N/A')[:120]}"
                )
        lines.append("  Suggested fixes:")
        for fix in cat_info["suggested_fixes"]:
            lines.append(f"    - {fix}")
        lines.append("")

    text_report = "\n".join(lines)
    report_path = save_dir / "error_analysis.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(text_report)

    report["report_path"] = str(report_path)
    return report


# ===================================================================
# (e)  Full Evaluation Report
# ===================================================================


def generate_evaluation_report(
    y_true: List[str],
    y_pred: List[str],
    gold_evidence: List[Set[str]],
    pred_evidence: List[Set[str]],
    criteria_types: Optional[List[str]] = None,
    samples: Optional[List[Dict[str, Any]]] = None,
    label_names: Optional[List[str]] = None,
    save_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run *all* evaluation metrics and produce a complete text report.

    Parameters
    ----------
    y_true, y_pred : label lists
    gold_evidence, pred_evidence : evidence sentence-id sets
    criteria_types : ``"inclusion"`` / ``"exclusion"`` per sample (optional)
    samples : full sample dicts for error analysis (optional)
    label_names : ordered label set
    save_dir : where to write reports and figures

    Returns
    -------
    dict
        Combined metrics from every sub-routine.
    """
    _ensure_dirs()
    if label_names is None:
        label_names = LABEL_NAMES
    if save_dir is None:
        save_dir = REPORTS_DIR

    figures_dir = save_dir / "figures" if (save_dir / "figures").exists() or save_dir == REPORTS_DIR else save_dir
    figures_dir.mkdir(parents=True, exist_ok=True)

    # (a) Classification
    cls_metrics = compute_classification_metrics(y_true, y_pred, label_names)

    # (b) Confusion matrices
    cm_results = compute_confusion_matrices(
        y_true, y_pred, criteria_types, label_names, save_dir=figures_dir
    )

    # (c) Evidence
    ev_metrics = compute_evidence_metrics(gold_evidence, pred_evidence)

    # (d) Error analysis
    ea_report: Optional[Dict[str, Any]] = None
    if samples is not None:
        ea_report = error_analysis(samples, save_dir=figures_dir)

    # ------------------------------------------------------------------
    # Build combined text report
    # ------------------------------------------------------------------
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("EVALUATION REPORT – Clinical Trial Eligibility NLP")
    lines.append("=" * 80)
    lines.append("")

    # Classification
    lines.append("1.  CLASSIFICATION METRICS (4-class)")
    lines.append("-" * 50)
    lines.append(f"  Samples evaluated : {cls_metrics['n_samples']}")
    lines.append(f"  Accuracy          : {cls_metrics['accuracy']:.4f}")
    lines.append(f"  Macro F1          : {cls_metrics['macro_f1']:.4f}")
    lines.append(f"  Macro Precision   : {cls_metrics['macro_precision']:.4f}")
    lines.append(f"  Macro Recall      : {cls_metrics['macro_recall']:.4f}")
    lines.append(f"  Weighted F1       : {cls_metrics['weighted_f1']:.4f}")
    lines.append("")
    lines.append("  Per-class breakdown:")
    for label, m in cls_metrics["per_class"].items():
        lines.append(
            f"    {label:20s}  P={m['precision']:.4f}  R={m['recall']:.4f}  "
            f"F1={m['f1']:.4f}  support={m['support']}"
        )
    lines.append("")

    # Confusion matrices
    lines.append("2.  CONFUSION MATRICES")
    lines.append("-" * 50)
    lines.append("  Overall confusion matrix:")
    lines.append(str(cm_results["overall"]))
    for split in ("inclusion", "exclusion"):
        if cm_results.get(split) is not None:
            lines.append(f"  {split.capitalize()} confusion matrix:")
            lines.append(str(cm_results[split]))
    lines.append(
        f"  Figures saved to: {figures_dir / 'confusion_matrix_*.png'}"
    )
    lines.append("")

    # Evidence
    lines.append("3.  EVIDENCE RETRIEVAL METRICS")
    lines.append("-" * 50)
    lines.append(f"  Micro Precision : {ev_metrics['micro_precision']:.4f}")
    lines.append(f"  Micro Recall    : {ev_metrics['micro_recall']:.4f}")
    lines.append(f"  Micro F1        : {ev_metrics['micro_f1']:.4f}")
    lines.append(f"  Macro Precision : {ev_metrics['macro_precision']:.4f}")
    lines.append(f"  Macro Recall    : {ev_metrics['macro_recall']:.4f}")
    lines.append(f"  Macro F1        : {ev_metrics['macro_f1']:.4f}")
    lines.append(f"  Total TP={ev_metrics['total_tp']}  FP={ev_metrics['total_fp']}  FN={ev_metrics['total_fn']}")
    lines.append("")

    # Error analysis summary
    if ea_report is not None:
        lines.append("4.  ERROR ANALYSIS SUMMARY")
        lines.append("-" * 50)
        lines.append(f"  Total errors : {ea_report['total_errors']}")
        lines.append(f"  Error rate   : {ea_report['error_rate']:.4f}")
        for cat_name, cat_info in ea_report["categories"].items():
            lines.append(
                f"  [{cat_name}] {cat_info['n_examples_found']} examples found"
            )
        lines.append(
            f"  Full error analysis: {ea_report.get('report_path', 'N/A')}"
        )
        lines.append("")

    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    report_text = "\n".join(lines)
    report_path = save_dir / "evaluation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    all_metrics: Dict[str, Any] = {
        "classification": cls_metrics,
        "evidence": ev_metrics,
        "confusion_matrices": {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in cm_results.items()
        },
        "error_analysis": ea_report,
        "report_path": str(report_path),
    }

    return all_metrics

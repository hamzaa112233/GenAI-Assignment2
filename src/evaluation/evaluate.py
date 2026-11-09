"""
Complete evaluation pipeline for Task 2.

Loads the fine-tuned model, runs inference on the test set, parses
predictions, computes all classification and evidence metrics, generates
confusion matrices, performs error analysis, and writes a full report.

Usage:
    python -m src.evaluation.evaluate --adapter_path outputs/best_adapter
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a clinical trial eligibility assessment system. "
    "Given a patient description (numbered sentences), a criterion type "
    "(inclusion or exclusion), and the criterion text, determine whether "
    "the patient meets the criterion. Respond with a JSON object containing: "
    '"eligibility" (one of "eligible", "not_eligible", "not_applicable", '
    '"not_enough_info"), "evidence_sentence_ids" (list of integers referring '
    "to the patient description sentences that support your decision), and "
    '"rationale" (a brief explanation).'
)


def _build_prompt(example: Dict, tokenizer) -> str:
    """Build a chat prompt for inference."""
    user_content = (
        f"Patient Description: {example.get('note', example.get('patient_description', ''))}\n"
        f"Criterion Type: {example.get('criterion_type', '')}\n"
        f"Criterion: {example.get('criterion_text', example.get('criterion', ''))}\n"
        "Determine eligibility and cite evidence."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def run_inference_on_dataset(
    model,
    tokenizer,
    dataset,
    max_new_tokens: int = 512,
    batch_size: int = 1,
) -> List[str]:
    """Run inference on every example in *dataset*, return raw outputs."""
    model.eval()
    tokenizer.padding_side = "left"
    outputs = []

    for i in tqdm(range(len(dataset)), desc="Inference"):
        example = dataset[i]
        prompt = _build_prompt(example, tokenizer)
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1536,
        ).to(model.device)

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_ids = gen_ids[0, inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        outputs.append(text)

    return outputs


def parse_prediction(text: str) -> Dict[str, Any]:
    """Parse a model output into eligibility + evidence."""
    import json as _json
    import re

    result = {
        "eligibility": "unknown",
        "evidence_sentence_ids": [],
        "rationale": "",
        "parse_success": False,
    }
    if not text or not text.strip():
        return result

    # Try JSON
    try:
        parsed = _json.loads(text.strip())
        if isinstance(parsed, dict):
            result["eligibility"] = parsed.get("eligibility", "unknown").strip().lower()
            result["evidence_sentence_ids"] = [
                int(x) for x in parsed.get("evidence_sentence_ids",
                                             parsed.get("supporting_sentences", []))
            ]
            result["rationale"] = parsed.get("rationale", "")
            result["parse_success"] = True
            return result
    except Exception:
        pass

    # Try extracting JSON from text
    m = re.search(r"\{[^{}]+\}", text)
    if m:
        try:
            parsed = _json.loads(m.group())
            result["eligibility"] = parsed.get("eligibility", "unknown").strip().lower()
            result["evidence_sentence_ids"] = [
                int(x) for x in parsed.get("evidence_sentence_ids",
                                             parsed.get("supporting_sentences", []))
            ]
            result["rationale"] = parsed.get("rationale", "")
            result["parse_success"] = True
            return result
        except Exception:
            pass

    # Fallback: look for label keywords
    lower = text.lower()
    for label in ["not_enough_info", "not_applicable", "not_eligible", "eligible",
                   "not enough info", "not applicable", "not eligible"]:
        if label in lower:
            result["eligibility"] = label.replace(" ", "_")
            result["parse_success"] = True
            break

    return result


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------

def run_full_evaluation(
    model=None,
    tokenizer=None,
    test_dataset=None,
    adapter_path: Optional[str] = None,
    base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    output_dir: str = "reports",
) -> Dict[str, Any]:
    """Run the complete evaluation pipeline.

    If *model* and *tokenizer* are provided, uses them directly.
    Otherwise loads from *adapter_path*.
    """
    from src.evaluation.metrics import (
        compute_classification_metrics,
        compute_confusion_matrices,
        compute_evidence_metrics,
        error_analysis,
        generate_evaluation_report,
    )
    from pathlib import Path

    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # Load model if not provided
    if model is None or tokenizer is None:
        if adapter_path is None:
            adapter_path = "outputs/best_adapter"
        from src.models.model import load_finetuned_model
        logger.info("Loading fine-tuned model from %s", adapter_path)
        model, tokenizer = load_finetuned_model(base_model_name, adapter_path)

    # Load test dataset if not provided
    if test_dataset is None:
        from src.data.dataset import prepare_datasets
        ds_dict, _ = prepare_datasets()
        test_dataset = ds_dict["test"]

    logger.info("Running inference on %d test examples ...", len(test_dataset))
    raw_outputs = run_inference_on_dataset(model, tokenizer, test_dataset)

    # Parse predictions
    predictions = [parse_prediction(out) for out in raw_outputs]
    pred_labels = [p["eligibility"] for p in predictions]
    pred_evidence = [set(p["evidence_sentence_ids"]) for p in predictions]

    # Extract ground truth
    col_names = getattr(test_dataset, "column_names", None)
    if col_names is None and len(test_dataset) > 0:
        col_names = list(test_dataset[0].keys())
    col_names = col_names or []
    label_col = "expert_eligibility" if "expert_eligibility" in col_names else "label"
    evidence_col = "expert_sentences" if "expert_sentences" in col_names else "evidence_sentence_ids"

    true_labels = []
    true_evidence = []
    criterion_types = []

    for i in range(len(test_dataset)):
        ex = test_dataset[i]
        true_labels.append(ex.get(label_col, "unknown"))
        ev = ex.get(evidence_col, "[]")
        if isinstance(ev, str):
            import re
            ids = [int(x) for x in re.findall(r"\d+", ev)]
        elif isinstance(ev, list):
            ids = [int(x) for x in ev]
        else:
            ids = []
        true_evidence.append(set(ids))
        criterion_types.append(ex.get("criterion_type", "unknown"))

    # Compute all metrics
    logger.info("Computing classification metrics ...")
    class_metrics = compute_classification_metrics(true_labels, pred_labels)

    logger.info("Computing confusion matrices ...")
    compute_confusion_matrices(true_labels, pred_labels, criterion_types, figures_dir)

    logger.info("Computing evidence metrics ...")
    ev_metrics = compute_evidence_metrics(
        [list(s) for s in true_evidence],
        [list(s) for s in pred_evidence],
    )

    logger.info("Running error analysis ...")
    samples_for_analysis = []
    for i in range(len(test_dataset)):
        samples_for_analysis.append({
            "index": i,
            "true_label": true_labels[i],
            "pred_label": pred_labels[i],
            "true_evidence": list(true_evidence[i]),
            "pred_evidence": list(pred_evidence[i]),
            "criterion_type": criterion_types[i],
            "raw_output": raw_outputs[i],
            "parse_success": predictions[i]["parse_success"],
        })
    err_analysis = error_analysis(samples_for_analysis, output_dir=output_dir)

    # Full report
    all_results = {
        "classification": class_metrics,
        "evidence": ev_metrics,
        "error_analysis_summary": err_analysis,
        "num_test_samples": len(test_dataset),
        "parse_success_rate": sum(1 for p in predictions if p["parse_success"]) / max(len(predictions), 1),
    }

    logger.info("Generating evaluation report ...")
    # Build sample dicts for error analysis
    err_samples = []
    for i in range(len(test_dataset)):
        ex = test_dataset[i]
        err_samples.append({
            "true_label": true_labels[i],
            "pred_label": pred_labels[i],
            "gold_evidence": list(true_evidence[i]),
            "pred_evidence": list(pred_evidence[i]),
            "criteria_type": criterion_types[i],
            "patient_description": ex.get("note", ex.get("patient_description", "")),
            "criterion_text": ex.get("criterion_text", ex.get("criterion", "")),
            "criterion_id": i,
        })

    full_report = generate_evaluation_report(
        y_true=true_labels,
        y_pred=pred_labels,
        gold_evidence=[set(s) for s in [[int(x) for x in te] for te in true_evidence]],
        pred_evidence=[set(s) for s in [[int(x) for x in pe] for pe in pred_evidence]],
        criteria_types=criterion_types,
        samples=err_samples,
        save_dir=Path(output_dir),
    )
    all_results.update(full_report)

    # Save results JSON
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Results saved to %s", results_path)

    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Test samples:        {len(test_dataset)}")
    print(f"  Parse success rate:  {all_results['parse_success_rate']:.1%}")
    print(f"  Macro-F1:            {class_metrics.get('macro_f1', 0):.4f}")
    print(f"  Accuracy:            {class_metrics.get('accuracy', 0):.4f}")
    print(f"  Evidence Micro-F1:   {ev_metrics.get('micro_f1', 0):.4f}")
    print(f"{'='*60}\n")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model on test set")
    parser.add_argument("--adapter_path", default="outputs/best_adapter")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output_dir", default="reports")
    args = parser.parse_args()

    run_full_evaluation(
        adapter_path=args.adapter_path,
        base_model_name=args.base_model,
        output_dir=args.output_dir,
    )

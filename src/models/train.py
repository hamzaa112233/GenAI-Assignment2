# src/models/train.py
"""
Complete training script for clinical trial eligibility fine-tuning.

Usage (command line):
    python -m src.models.train                          # uses default config
    python -m src.models.train --config path/to/cfg.yaml

Usage (notebook / programmatic):
    from src.models.train import main
    main("configs/training_config.yaml")

The script:
1. Loads config from ``configs/training_config.yaml``
2. Loads and tokenises the dataset via ``src.data.dataset``
3. Builds a QLoRA model via ``src.models.model``
4. Trains with HuggingFace ``Trainer`` + ``DataCollatorForSeq2Seq``
5. Saves training curves to ``reports/figures/training_curves.png``
6. Runs before-vs-after inference comparison on 5 validation samples
7. Saves the best adapter checkpoint to ``<output_dir>/best_adapter``
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for Colab / headless
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)

from src.models.model import (
    load_model_for_training,
    load_tokenizer,
    load_base_model,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt & formatting helpers
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


def format_user_message(
    patient_description: str,
    criterion_type: str,
    criterion_text: str,
) -> str:
    """Build the user-turn content for the chat template.

    Args:
        patient_description: Numbered patient sentences.
        criterion_type: ``"inclusion"`` or ``"exclusion"``.
        criterion_text: The eligibility criterion.

    Returns:
        Formatted user message string.
    """
    return (
        f"Patient Description: {patient_description}\n"
        f"Criterion Type: {criterion_type}\n"
        f"Criterion: {criterion_text}\n"
        "Determine eligibility and cite evidence."
    )


def format_assistant_message(
    eligibility: str,
    evidence_sentence_ids: List[int],
    rationale: str,
) -> str:
    """Build the expected assistant JSON response.

    Args:
        eligibility: One of eligible / not_eligible / not_applicable / not_enough_info.
        evidence_sentence_ids: List of 1-based sentence indices.
        rationale: Brief explanation.

    Returns:
        JSON string.
    """
    return json.dumps(
        {
            "eligibility": eligibility,
            "evidence_sentence_ids": evidence_sentence_ids,
            "rationale": rationale,
        },
        ensure_ascii=False,
    )


def build_chat_messages(example: Dict[str, Any]) -> List[Dict[str, str]]:
    """Convert a raw dataset example into Qwen2.5 chat-format messages.

    Supports both TrialGPT native columns (note, criterion_text, expert_eligibility, expert_sentences)
    and canonical alias keys.

    Returns:
        List of message dicts with roles ``system``, ``user``, ``assistant``.
    """
    patient_desc = example.get("patient_description", example.get("note", ""))
    crit_type = example.get("criterion_type", "inclusion")
    crit_text = example.get("criterion", example.get("criterion_text", ""))
    label = example.get("label", example.get("expert_eligibility", "not_enough_info"))
    ev_ids = example.get("evidence_sentence_ids", example.get("expert_sentences", []))
    if isinstance(ev_ids, str):
        import re
        ev_ids = [int(x) for x in re.findall(r"\d+", ev_ids)]

    user_content = format_user_message(
        patient_description=patient_desc,
        criterion_type=crit_type,
        criterion_text=crit_text,
    )
    assistant_content = format_assistant_message(
        eligibility=label,
        evidence_sentence_ids=ev_ids,
        rationale=example.get("rationale", f"Criterion assessment based on evidence sentences {ev_ids}."),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


# ---------------------------------------------------------------------------
# Tokenisation with proper label masking
# ---------------------------------------------------------------------------

def tokenize_chat_example(
    example: Dict[str, Any],
    tokenizer: AutoTokenizer,
    max_length: int = 2048,
) -> Dict[str, List[int]]:
    """Tokenise a single example using the Qwen2.5 chat template.

    The prompt tokens (system + user turns) are masked in the labels
    (set to -100) so that loss is computed only on the assistant response.

    Args:
        example: Raw dataset row.
        tokenizer: The loaded tokenizer.
        max_length: Maximum sequence length.

    Returns:
        Dict with ``input_ids``, ``attention_mask``, ``labels``.
    """
    messages = build_chat_messages(example)

    # --- Full conversation (with assistant) ---
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # --- Prompt only (system + user, with generation prompt) ---
    prompt_messages = messages[:2]  # system + user
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenise both
    full_enc = tokenizer(
        full_text,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors=None,
    )
    prompt_enc = tokenizer(
        prompt_text,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors=None,
    )

    input_ids = full_enc["input_ids"]
    attention_mask = full_enc["attention_mask"]

    # Mask the prompt portion in labels
    prompt_len = len(prompt_enc["input_ids"])
    labels = [-100] * prompt_len + input_ids[prompt_len:]

    # Safety: ensure labels length matches input_ids
    labels = labels[: len(input_ids)]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def prepare_dataset(
    tokenizer: AutoTokenizer,
    max_length: int = 2048,
    seed: int = 42,
) -> Dict[str, Dataset]:
    """Load the clinical trial dataset and tokenise for training.

    This function tries to import ``src.data.dataset`` which should expose
    a ``load_dataset()`` function returning a ``DatasetDict`` with at least
    ``train`` and ``validation`` splits. Each example must have the keys
    expected by :func:`build_chat_messages`.

    If ``src.data.dataset`` is not available yet, a small synthetic dataset
    is generated for testing the training pipeline.

    Args:
        tokenizer: Tokenizer to use.
        max_length: Max sequence length.
        seed: Random seed for shuffling.

    Returns:
        Dict with ``"train"`` and ``"validation"`` Hugging Face Datasets.
    """
    try:
        from src.data.dataset import load_dataset as load_ct_dataset

        logger.info("Loading dataset via src.data.dataset.load_dataset()")
        dataset_dict = load_ct_dataset()
        train_ds = dataset_dict["train"]
        val_ds = dataset_dict["validation"]
    except (ImportError, Exception) as exc:
        logger.warning(
            "Could not load dataset from src.data.dataset (%s). "
            "Generating synthetic examples for pipeline testing.",
            exc,
        )
        train_ds, val_ds = _generate_synthetic_dataset(seed=seed)

    def _tokenize(example):
        return tokenize_chat_example(example, tokenizer, max_length)

    logger.info("Tokenising training set (%d examples)...", len(train_ds))
    train_tokenized = train_ds.map(
        _tokenize,
        remove_columns=train_ds.column_names,
        desc="Tokenising train",
    )

    logger.info("Tokenising validation set (%d examples)...", len(val_ds))
    val_tokenized = val_ds.map(
        _tokenize,
        remove_columns=val_ds.column_names,
        desc="Tokenising validation",
    )

    return {"train": train_tokenized, "validation": val_tokenized}


def _generate_synthetic_dataset(
    n_train: int = 50,
    n_val: int = 10,
    seed: int = 42,
) -> tuple:
    """Create a small synthetic dataset for pipeline testing.

    Returns:
        Tuple of (train_dataset, val_dataset) as Hugging Face Datasets.
    """
    rng = random.Random(seed)
    labels = ["eligible", "not_eligible", "not_applicable", "not_enough_info"]
    criterion_types = ["inclusion", "exclusion"]

    conditions = [
        "Type 2 diabetes mellitus",
        "Hypertension",
        "Chronic kidney disease stage 3",
        "Heart failure with reduced ejection fraction",
        "Major depressive disorder",
        "Non-small cell lung cancer",
        "Rheumatoid arthritis",
        "Atrial fibrillation",
    ]
    patient_templates = [
        (
            "1. Patient is a {age}-year-old {sex}. "
            "2. Diagnosed with {cond} for {years} years. "
            "3. Currently on {med}. "
            "4. No history of liver disease. "
            "5. ECOG performance status {ecog}."
        ),
    ]

    def _make_example():
        cond = rng.choice(conditions)
        age = rng.randint(30, 80)
        sex = rng.choice(["male", "female"])
        years = rng.randint(1, 20)
        med = rng.choice(["metformin", "lisinopril", "atorvastatin", "aspirin"])
        ecog = rng.randint(0, 2)
        desc = patient_templates[0].format(
            age=age, sex=sex, cond=cond, years=years, med=med, ecog=ecog
        )
        ct = rng.choice(criterion_types)
        criterion = f"Patient must have {cond}" if ct == "inclusion" else f"Patient must not have severe hepatic impairment"
        label = rng.choice(labels)
        ev_ids = sorted(rng.sample([1, 2, 3, 4, 5], k=rng.randint(1, 3)))
        rationale = f"Based on sentences {ev_ids}, the patient is assessed as {label}."
        return {
            "patient_description": desc,
            "criterion_type": ct,
            "criterion": criterion,
            "label": label,
            "evidence_sentence_ids": ev_ids,
            "rationale": rationale,
        }

    train_examples = [_make_example() for _ in range(n_train)]
    val_examples = [_make_example() for _ in range(n_val)]

    train_ds = Dataset.from_list(train_examples)
    val_ds = Dataset.from_list(val_examples)
    logger.info(
        "Synthetic dataset created: %d train, %d val", len(train_ds), len(val_ds)
    )
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Training-curve callback
# ---------------------------------------------------------------------------

class LossLoggerCallback(TrainerCallback):
    """Record training and eval losses for plotting."""

    def __init__(self):
        super().__init__()
        self.train_losses: List[Dict[str, float]] = []
        self.eval_losses: List[Dict[str, float]] = []

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        if logs is None:
            return
        step = state.global_step
        if "loss" in logs:
            self.train_losses.append({"step": step, "loss": logs["loss"]})
        if "eval_loss" in logs:
            self.eval_losses.append({"step": step, "loss": logs["eval_loss"]})


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_curves(
    loss_callback: LossLoggerCallback,
    output_path: str,
) -> None:
    """Save training and eval loss curves as a PNG figure.

    Args:
        loss_callback: The callback that recorded losses during training.
        output_path: File path for the saved figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if loss_callback.train_losses:
        steps = [d["step"] for d in loss_callback.train_losses]
        losses = [d["loss"] for d in loss_callback.train_losses]
        ax.plot(steps, losses, label="Train Loss", color="steelblue", linewidth=1.5)

    if loss_callback.eval_losses:
        steps = [d["step"] for d in loss_callback.eval_losses]
        losses = [d["loss"] for d in loss_callback.eval_losses]
        ax.plot(
            steps,
            losses,
            label="Eval Loss",
            color="darkorange",
            linewidth=2,
            marker="o",
            markersize=5,
        )

    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Curves — Clinical Trial Eligibility Fine-Tuning", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Training curves saved to %s", output_path)


# ---------------------------------------------------------------------------
# Before / After inference comparison
# ---------------------------------------------------------------------------

def run_inference_comparison(
    model,
    tokenizer: AutoTokenizer,
    val_examples: List[Dict[str, Any]],
    tag: str = "BEFORE",
    max_new_tokens: int = 256,
) -> List[str]:
    """Run inference on a few validation examples and print results.

    Args:
        model: The (P)EFT or base model.
        tokenizer: Tokenizer.
        val_examples: Raw validation examples (dicts).
        tag: Label printed before outputs (e.g. "BEFORE" / "AFTER").
        max_new_tokens: Max tokens to generate.

    Returns:
        List of generated strings.
    """
    model.eval()
    results = []

    # Switch to left-padding for generation
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    print(f"\n{'='*80}")
    print(f"  INFERENCE {tag} TRAINING")
    print(f"{'='*80}")

    for i, example in enumerate(val_examples):
        patient_desc = example.get("patient_description", example.get("note", ""))
        crit_type = example.get("criterion_type", "inclusion")
        crit_text = example.get("criterion", example.get("criterion_text", ""))
        label = example.get("label", example.get("expert_eligibility", "not_enough_info"))

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_user_message(
                patient_desc,
                crit_type,
                crit_text,
            )},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1536,
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only newly generated tokens
        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        results.append(generated_text)

        print(f"\n--- Sample {i+1} [{tag}] ---")
        print(f"Criterion ({crit_type}): {crit_text[:100]}...")
        print(f"Ground truth: {label}")
        print(f"Generated:\n{generated_text[:500]}")

    print(f"\n{'='*80}\n")

    # Restore original padding side
    tokenizer.padding_side = original_padding_side
    return results


# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Configuration dictionary.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.info("Loaded config from %s", config_path)
    return cfg


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def main(
    config_path: str = "configs/training_config.yaml",
    model=None,
    tokenizer=None,
) -> Dict[str, Any]:
    """End-to-end training pipeline.

    Args:
        config_path: Path to the YAML training configuration.
        model: Optional pre-loaded model.
        tokenizer: Optional pre-loaded tokenizer.
    """
    # ------------------------------------------------------------------
    # 1. Load configuration
    # ------------------------------------------------------------------
    cfg = load_config(config_path)

    model_name: str = cfg["model_name"]
    max_length: int = cfg.get("max_length", 2048)
    lora_r: int = cfg.get("lora_r", 16)
    lora_alpha: int = cfg.get("lora_alpha", 32)
    lora_dropout: float = cfg.get("lora_dropout", 0.05)
    lr: float = cfg.get("learning_rate", 2e-4)
    num_epochs: int = cfg.get("num_epochs", 3)
    batch_size: int = cfg.get("batch_size", 4)
    grad_accum: int = cfg.get("gradient_accumulation_steps", 4)
    warmup_ratio: float = cfg.get("warmup_ratio", 0.1)
    weight_decay: float = cfg.get("weight_decay", 0.01)
    seed: int = cfg.get("seed", 42)
    output_dir: str = cfg.get("output_dir", "./outputs")
    logging_steps: int = cfg.get("logging_steps", 10)
    save_steps: int = cfg.get("save_steps", 100)
    eval_steps: int = cfg.get("eval_steps", 100)

    figures_dir = "reports/figures"
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Also add a file handler for logging
    log_file = os.path.join(output_dir, "training.log")
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    )
    logging.getLogger().addHandler(file_handler)
    logger.info("Training log will be written to %s", log_file)

    # ------------------------------------------------------------------
    # 2. Load model + tokenizer
    # ------------------------------------------------------------------
    if model is None or tokenizer is None:
        logger.info("Loading model and tokenizer...")
        model, tokenizer = load_model_for_training(
            model_name=model_name,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

    # ------------------------------------------------------------------
    # 3. Prepare dataset
    # ------------------------------------------------------------------
    logger.info("Preparing dataset...")
    datasets = prepare_dataset(tokenizer, max_length=max_length, seed=seed)
    train_dataset = datasets["train"]
    val_dataset = datasets["validation"]

    logger.info("Train samples: %d, Validation samples: %d",
                len(train_dataset), len(val_dataset))

    # ------------------------------------------------------------------
    # 4. Get raw validation examples for before/after comparison
    # ------------------------------------------------------------------
    try:
        from src.data.dataset import load_dataset as load_ct_dataset
        raw_dataset = load_ct_dataset()
        raw_val = [raw_dataset["validation"][i] for i in range(min(5, len(raw_dataset["validation"])))]
    except Exception:
        # Use synthetic examples for comparison
        _, raw_val_ds = _generate_synthetic_dataset(n_train=5, n_val=5, seed=seed)
        raw_val = [raw_val_ds[i] for i in range(min(5, len(raw_val_ds)))]

    # ------------------------------------------------------------------
    # 5. Before-training inference
    # ------------------------------------------------------------------
    logger.info("Running BEFORE-training inference on %d samples...", len(raw_val))
    before_results = run_inference_comparison(
        model, tokenizer, raw_val, tag="BEFORE", max_new_tokens=256
    )

    # ------------------------------------------------------------------
    # 6. Training arguments
    # ------------------------------------------------------------------
    import dataclasses
    import inspect

    # Inspect valid fields of TrainingArguments across transformers versions (v4 vs v5)
    if dataclasses.is_dataclass(TrainingArguments):
        valid_fields = {f.name for f in dataclasses.fields(TrainingArguments)}
    else:
        valid_fields = set(inspect.signature(TrainingArguments.__init__).parameters.keys())

    eval_strategy_key = "eval_strategy" if "eval_strategy" in valid_fields else "evaluation_strategy"

    is_ampere_or_newer = (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] >= 8
    )
    use_bf16 = (
        is_ampere_or_newer
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )
    use_fp16 = torch.cuda.is_available() and not use_bf16

    args_dict = {
        "output_dir": output_dir,
        "num_train_epochs": num_epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "lr_scheduler_type": "cosine",
        "logging_steps": logging_steps,
        "logging_first_step": True,
        eval_strategy_key: "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "bf16": use_bf16,
        "fp16": use_fp16,
        "gradient_checkpointing": True,
        "optim": "paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        "report_to": "none",
        "seed": seed,
        "dataloader_pin_memory": torch.cuda.is_available(),
        "remove_unused_columns": False,
        "label_names": ["labels"],
    }

    # Handle warmup: in transformers v5, warmup_ratio was removed in favor of warmup_steps (which accepts float ratio)
    if "warmup_ratio" in valid_fields:
        args_dict["warmup_ratio"] = warmup_ratio
    elif "warmup_steps" in valid_fields:
        args_dict["warmup_steps"] = warmup_ratio

    if "gradient_checkpointing_kwargs" in valid_fields:
        args_dict["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    if "tf32" in valid_fields:
        args_dict["tf32"] = is_ampere_or_newer

    # Strictly keep only fields that exist in the active transformers version
    final_args = {k: v for k, v in args_dict.items() if k in valid_fields}

    training_args = TrainingArguments(**final_args)

    # Data collator: pads input_ids, attention_mask, and labels
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=max_length,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    # Loss logger callback
    loss_callback = LossLoggerCallback()

    # ------------------------------------------------------------------
    # 7. Trainer
    # ------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[loss_callback],
    )

    # ------------------------------------------------------------------
    # 8. Train!
    # ------------------------------------------------------------------
    logger.info("Starting training...")
    train_result = trainer.train()

    # Log final metrics
    metrics = train_result.metrics
    logger.info("Training complete. Metrics: %s", json.dumps(metrics, indent=2))
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Evaluate on validation set
    eval_metrics = trainer.evaluate()
    logger.info("Evaluation metrics: %s", json.dumps(eval_metrics, indent=2))
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    # ------------------------------------------------------------------
    # 9. Save best adapter
    # ------------------------------------------------------------------
    best_adapter_path = os.path.join(output_dir, "best_adapter")
    logger.info("Saving best adapter to %s", best_adapter_path)
    model.save_pretrained(best_adapter_path)
    tokenizer.save_pretrained(best_adapter_path)

    # ------------------------------------------------------------------
    # 10. Plot training curves
    # ------------------------------------------------------------------
    curves_path = os.path.join(figures_dir, "training_curves.png")
    plot_training_curves(loss_callback, curves_path)

    # ------------------------------------------------------------------
    # 11. After-training inference
    # ------------------------------------------------------------------
    logger.info("Running AFTER-training inference on %d samples...", len(raw_val))
    after_results = run_inference_comparison(
        model, tokenizer, raw_val, tag="AFTER", max_new_tokens=256
    )

    # ------------------------------------------------------------------
    # 12. Print comparison summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  BEFORE vs AFTER TRAINING COMPARISON")
    print("=" * 80)
    for i, (before, after, example) in enumerate(zip(before_results, after_results, raw_val)):
        print(f"\n--- Sample {i+1} ---")
        ground_truth = example.get("expert_eligibility", example.get("label", "N/A"))
        print(f"Ground truth label: {ground_truth}")
        print(f"BEFORE: {before[:300]}")
        print(f"AFTER:  {after[:300]}")
    print("\n" + "=" * 80)

    logger.info("Training pipeline complete!")
    logger.info("  Best adapter:     %s", best_adapter_path)
    logger.info("  Training curves:  %s", curves_path)
    logger.info("  Training log:     %s", log_file)

    return {
        "best_adapter_path": best_adapter_path,
        "training_curves_path": curves_path,
        "train_metrics": metrics,
        "eval_metrics": eval_metrics,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen2.5-1.5B for clinical trial eligibility"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    main(config_path=args.config)

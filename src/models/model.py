# src/models/model.py
"""
Model loading and LoRA configuration for clinical trial eligibility NLP.

Provides functions to:
- Load Qwen2.5-1.5B-Instruct with 4-bit QLoRA quantization
- Apply LoRA adapters to attention + MLP projection layers
- Load a fine-tuned adapter checkpoint for inference
- Print trainable parameter statistics
"""

import os
import logging
from typing import Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType,
    prepare_model_for_kbit_training,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quantization config
# ---------------------------------------------------------------------------

def get_compute_dtype() -> torch.dtype:
    """Return appropriate compute dtype: bfloat16 on Ampere+, else float16."""
    if (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] >= 8
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    ):
        return torch.bfloat16
    return torch.float16 if torch.cuda.is_available() else torch.float32


def get_bnb_config() -> BitsAndBytesConfig:
    """Return a 4-bit NormalFloat quantization config for QLoRA."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=get_compute_dtype(),
        bnb_4bit_use_double_quant=True,
    )


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def load_tokenizer(model_name: str) -> AutoTokenizer:
    """Load the tokenizer and ensure pad token is set.

    Args:
        model_name: Hugging Face model identifier.

    Returns:
        Configured tokenizer with left-padding for generation.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    # Qwen2.5 uses <|endoftext|> as eos; set pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Left-pad so generation works correctly in batch mode
    tokenizer.padding_side = "right"  # right-pad for training
    return tokenizer


# ---------------------------------------------------------------------------
# Base model loading (quantised)
# ---------------------------------------------------------------------------

def load_base_model(
    model_name: str,
    quantize: bool = True,
) -> AutoModelForCausalLM:
    """Load the base causal-LM, optionally with 4-bit quantization.

    Args:
        model_name: Hugging Face model identifier.
        quantize: Whether to apply 4-bit NF4 quantization.

    Returns:
        The loaded model on GPU.
    """
    kwargs = dict(
        pretrained_model_name_or_path=model_name,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=get_compute_dtype(),
    )
    if quantize:
        kwargs["quantization_config"] = get_bnb_config()

    logger.info("Loading base model: %s (quantize=%s)", model_name, quantize)
    model = AutoModelForCausalLM.from_pretrained(**kwargs)
    logger.info("Model loaded. Memory footprint: %.2f MB",
                model.get_memory_footprint() / 1e6)
    return model


# ---------------------------------------------------------------------------
# LoRA adapter
# ---------------------------------------------------------------------------

def get_lora_config(
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
) -> LoraConfig:
    """Build a LoRA configuration targeting all major projection layers.

    Targets: q_proj, k_proj, v_proj, o_proj (attention) and
             gate_proj, up_proj, down_proj (MLP / SwiGLU).

    Args:
        lora_r: Rank of the low-rank matrices.
        lora_alpha: Scaling factor.
        lora_dropout: Dropout probability on LoRA layers.

    Returns:
        A ``LoraConfig`` instance.
    """
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
    )


def apply_lora(
    model: AutoModelForCausalLM,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
) -> AutoModelForCausalLM:
    """Wrap *model* with LoRA adapters and print trainable parameter stats.

    Args:
        model: A base causal-LM (possibly quantised).
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha scaling.
        lora_dropout: LoRA dropout.

    Returns:
        The PEFT-wrapped model.
    """
    # Prepare model for k-bit training (freeze, cast layer norms, etc.)
    model = prepare_model_for_kbit_training(model)

    lora_cfg = get_lora_config(lora_r, lora_alpha, lora_dropout)
    model = get_peft_model(model, lora_cfg)

    print_trainable_parameters(model)
    return model


# ---------------------------------------------------------------------------
# Load a fine-tuned adapter for inference
# ---------------------------------------------------------------------------

def load_finetuned_model(
    base_model_name: str,
    adapter_path: str,
    quantize: bool = True,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the base model and merge a saved LoRA adapter on top.

    Args:
        base_model_name: Hugging Face model identifier for the base model.
        adapter_path: Local path to the saved adapter directory.
        quantize: Whether to load the base model in 4-bit.

    Returns:
        Tuple of (model, tokenizer) ready for inference.
    """
    logger.info("Loading base model for inference: %s", base_model_name)
    base_model = load_base_model(base_model_name, quantize=quantize)

    logger.info("Loading LoRA adapter from: %s", adapter_path)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    tokenizer = load_tokenizer(base_model_name)
    # Switch to left-padding for generation
    tokenizer.padding_side = "left"

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Fine-tuned model loaded. Total parameters: %s", f"{total_params:,}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Convenience: load model for training (base + LoRA)
# ---------------------------------------------------------------------------

def load_model_for_training(
    model_name: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """High-level helper: load quantised model, apply LoRA, return with tokenizer.

    Args:
        model_name: Hugging Face model identifier.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha.
        lora_dropout: LoRA dropout.

    Returns:
        Tuple of (peft_model, tokenizer).
    """
    tokenizer = load_tokenizer(model_name)
    base_model = load_base_model(model_name, quantize=True)
    peft_model = apply_lora(base_model, lora_r, lora_alpha, lora_dropout)
    return peft_model, tokenizer


# ---------------------------------------------------------------------------
# Parameter stats
# ---------------------------------------------------------------------------

def print_trainable_parameters(model) -> None:
    """Print the number and percentage of trainable parameters."""
    total_params = 0
    trainable_params = 0
    for _, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    pct = 100.0 * trainable_params / total_params if total_params > 0 else 0.0
    print(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} "
        f"({pct:.2f}%)"
    )
    logger.info(
        "Trainable parameters: %s / %s (%.2f%%)",
        f"{trainable_params:,}",
        f"{total_params:,}",
        pct,
    )

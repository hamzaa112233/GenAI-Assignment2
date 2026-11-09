"""
Dataset loading, patient-grouped splitting, and prompt formatting for
the TrialGPT Criterion Annotations dataset.

Downloads ncbi/TrialGPT-Criterion-Annotations from Hugging Face, performs a
strict patient-grouped split (37 train / 8 val / 8 test), excludes historical
GPT-4 predictions to prevent data leakage, and formats each sample as a
structured prompt for instruction-tuned LLM fine-tuning.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from datasets import Dataset, DatasetDict, load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

    class Dataset(list):
        """Lightweight fallback Dataset when huggingface datasets is not installed."""
        def __init__(self, data=None):
            super().__init__(data or [])

        @property
        def column_names(self):
            return list(self[0].keys()) if len(self) > 0 else []

        def __getitem__(self, key):
            if isinstance(key, str):
                return [row[key] for row in self]
            if isinstance(key, slice):
                return Dataset(super().__getitem__(key))
            return super().__getitem__(key)

        def select(self, indices):
            return Dataset([self[i] for i in indices])

        def remove_columns(self, cols):
            col_set = set(cols)
            return Dataset([{k: v for k, v in row.items() if k not in col_set} for row in self])

        def map(self, fn, desc=None):
            mapped = []
            for item in self:
                res = fn(item)
                new_item = dict(item)
                if isinstance(res, dict):
                    new_item.update(res)
                mapped.append(new_item)
            return Dataset(mapped)

        @classmethod
        def from_list(cls, rows):
            return cls(rows)

    class DatasetDict(dict):
        """Lightweight fallback DatasetDict."""
        pass

from src.data.preprocessing import format_input_prompt, format_target_output

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Columns that contain GPT-4 historical predictions — must be excluded from
# model features to avoid data leakage.
LEAKAGE_COLUMNS = [
    "gpt4_explanation",
    "explanation_correctness",
    "gpt4_sentences",
    "gpt4_eligibility",
]

# The column used as the ground-truth label
LABEL_COLUMN = "expert_eligibility"

# The column used as ground-truth supporting sentence IDs
EVIDENCE_COLUMN = "expert_sentences"

# Patient split sizes (must sum to total unique patients)
TRAIN_SIZE = 37
VAL_SIZE = 8
# TEST_SIZE is the remainder

# Reproducibility seed
SEED = 42


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def load_raw_dataset() -> Dataset:
    """Download and return the raw TrialGPT-Criterion-Annotations dataset.

    The dataset is hosted on Hugging Face under the ``ncbi`` organization.
    It has a single ``train`` split containing all annotations. Checks for
    cached local JSON first.

    Returns
    -------
    Dataset
        The full raw dataset.
    """
    local_path = Path("data/trialgpt_criterion_annotations.json")
    if local_path.exists():
        with open(local_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        if HAS_DATASETS:
            return Dataset.from_list(rows)
        return Dataset(rows)

    if HAS_DATASETS:
        ds = load_dataset("ncbi/TrialGPT-Criterion-Annotations", split="train")
        return ds

    raise FileNotFoundError("Local data not found and 'datasets' library is not installed.")


def get_patient_ids(ds: Dataset) -> List[str]:
    """Extract sorted unique patient IDs from the dataset.

    Parameters
    ----------
    ds : Dataset
        The raw dataset.

    Returns
    -------
    List[str]
        Sorted list of unique patient IDs.
    """
    patient_ids = sorted(set(ds["patient_id"]))
    return patient_ids


def create_patient_splits(
    patient_ids: List[str],
    train_size: int = TRAIN_SIZE,
    val_size: int = VAL_SIZE,
    seed: int = SEED,
) -> Dict[str, List[str]]:
    """Shuffle patient IDs and split into train / val / test groups.

    The patient IDs are first sorted (for determinism), then shuffled using
    ``numpy.random.default_rng(seed)``.  The first ``train_size`` patients go
    to train, the next ``val_size`` to validation, and the rest to test.

    Parameters
    ----------
    patient_ids : List[str]
        Sorted unique patient IDs.
    train_size : int
        Number of patients in the training set (default 37).
    val_size : int
        Number of patients in the validation set (default 8).
    seed : int
        Random seed for shuffling (default 42).

    Returns
    -------
    Dict[str, List[str]]
        Dictionary with keys ``'train'``, ``'validation'``, ``'test'``,
        each mapping to a list of patient IDs.
    """
    rng = np.random.default_rng(seed)
    ids = np.array(patient_ids)
    rng.shuffle(ids)
    ids = ids.tolist()

    train_ids = ids[:train_size]
    val_ids = ids[train_size : train_size + val_size]
    test_ids = ids[train_size + val_size :]

    return {
        "train": train_ids,
        "validation": val_ids,
        "test": test_ids,
    }


def split_dataset_by_patients(
    ds: Dataset,
    patient_splits: Dict[str, List[str]],
) -> DatasetDict:
    """Split the dataset so every annotation for a patient is in exactly one split.

    Parameters
    ----------
    ds : Dataset
        The full raw dataset.
    patient_splits : Dict[str, List[str]]
        Mapping from split name to list of patient IDs.

    Returns
    -------
    DatasetDict
        A ``DatasetDict`` with ``train``, ``validation``, ``test`` splits.
    """
    split_datasets = {}
    for split_name, pids in patient_splits.items():
        pid_set = set(pids)
        indices = [i for i, pid in enumerate(ds["patient_id"]) if pid in pid_set]
        split_datasets[split_name] = ds.select(indices)

    return DatasetDict(split_datasets)


def remove_leakage_columns(ds_dict: DatasetDict) -> DatasetDict:
    """Remove GPT-4 prediction columns from all splits to prevent leakage.

    The following columns are removed:
    - ``gpt4_explanation``
    - ``explanation_correctness``
    - ``gpt4_sentences``
    - ``gpt4_eligibility``

    Parameters
    ----------
    ds_dict : DatasetDict
        The split datasets.

    Returns
    -------
    DatasetDict
        Cleaned datasets with leakage columns removed.
    """
    cleaned = {}
    for split_name, ds in ds_dict.items():
        cols_to_remove = [c for c in LEAKAGE_COLUMNS if c in ds.column_names]
        if cols_to_remove:
            ds = ds.remove_columns(cols_to_remove)
        cleaned[split_name] = ds
    return DatasetDict(cleaned)


def format_sample(example: dict) -> dict:
    """Format a single dataset example into an instruction-tuning prompt.

    The input prompt contains the patient note, criterion type, and criterion
    text.  The target output is a structured JSON-like string with the
    eligibility label and supporting sentence IDs.

    Parameters
    ----------
    example : dict
        A single row from the dataset.

    Returns
    -------
    dict
        Dictionary with ``'text'`` key containing the full formatted
        prompt + response for training, and ``'input'`` / ``'output'`` keys
        for evaluation.
    """
    input_text = format_input_prompt(
        note=example["note"],
        criterion_type=example["criterion_type"],
        criterion_text=example["criterion_text"],
    )

    output_text = format_target_output(
        eligibility=example[LABEL_COLUMN],
        evidence_sentences=example[EVIDENCE_COLUMN],
    )

    # Full text for causal LM training (input + output)
    full_text = f"{input_text}{output_text}"

    return {
        "input": input_text,
        "output": output_text,
        "text": full_text,
    }


def format_dataset(ds_dict: DatasetDict) -> DatasetDict:
    """Apply prompt formatting to all splits.

    Parameters
    ----------
    ds_dict : DatasetDict
        The cleaned split datasets.

    Returns
    -------
    DatasetDict
        Datasets with additional ``'input'``, ``'output'``, and ``'text'``
        columns.
    """
    formatted = {}
    for split_name, ds in ds_dict.items():
        formatted[split_name] = ds.map(format_sample, desc=f"Formatting {split_name}")
    return DatasetDict(formatted)


def save_split_manifest(
    patient_splits: Dict[str, List[str]],
    ds_dict: DatasetDict,
    output_dir: str = "./outputs",
) -> str:
    """Save a JSON manifest recording the patient split and sample counts.

    Parameters
    ----------
    patient_splits : Dict[str, List[str]]
        Mapping from split name to patient IDs.
    ds_dict : DatasetDict
        The split datasets (used to count samples per split).
    output_dir : str
        Directory where ``split_manifest.json`` will be saved.

    Returns
    -------
    str
        Absolute path to the saved manifest file.
    """
    os.makedirs(output_dir, exist_ok=True)

    manifest = {
        "seed": SEED,
        "splits": {},
    }
    for split_name in ["train", "validation", "test"]:
        manifest["splits"][split_name] = {
            "patient_ids": patient_splits[split_name],
            "num_patients": len(patient_splits[split_name]),
            "num_samples": len(ds_dict[split_name]),
        }

    manifest_path = os.path.join(output_dir, "split_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return os.path.abspath(manifest_path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def prepare_datasets(
    output_dir: str = "./outputs",
) -> Tuple[DatasetDict, Dict[str, List[str]]]:
    """Full data preparation pipeline.

    1. Download the raw dataset from Hugging Face.
    2. Extract unique patient IDs, sort, and shuffle with seed 42.
    3. Split: first 37 → train, next 8 → val, remaining → test.
    4. Remove GPT-4 leakage columns.
    5. Format prompts.
    6. Save split manifest.

    Parameters
    ----------
    output_dir : str
        Directory for saving the split manifest and any artifacts.

    Returns
    -------
    Tuple[DatasetDict, Dict[str, List[str]]]
        - The formatted ``DatasetDict`` with train / validation / test.
        - The patient split mapping.
    """
    print("=" * 60)
    print("Clinical Trial Eligibility - Data Preparation Pipeline")
    print("=" * 60)

    # Step 1: Download
    print("\n[1/6] Downloading TrialGPT-Criterion-Annotations...")
    raw_ds = load_raw_dataset()
    print(f"  -> Loaded {len(raw_ds)} annotations")
    print(f"  -> Columns: {raw_ds.column_names}")

    # Step 2: Patient IDs
    print("\n[2/6] Extracting patient IDs...")
    patient_ids = get_patient_ids(raw_ds)
    print(f"  -> Found {len(patient_ids)} unique patients")

    # Step 3: Split
    print("\n[3/6] Creating patient-grouped splits...")
    patient_splits = create_patient_splits(patient_ids)
    for split_name, pids in patient_splits.items():
        print(f"  -> {split_name}: {len(pids)} patients")

    # Step 4: Split dataset
    print("\n[4/6] Splitting dataset and removing leakage columns...")
    ds_dict = split_dataset_by_patients(raw_ds, patient_splits)
    ds_dict = remove_leakage_columns(ds_dict)
    for split_name, ds in ds_dict.items():
        print(f"  -> {split_name}: {len(ds)} samples, columns: {ds.column_names}")

    # Verify no patient overlap
    train_pids = set(ds_dict["train"]["patient_id"])
    val_pids = set(ds_dict["validation"]["patient_id"])
    test_pids = set(ds_dict["test"]["patient_id"])
    assert train_pids.isdisjoint(val_pids), "Train/val patient overlap detected!"
    assert train_pids.isdisjoint(test_pids), "Train/test patient overlap detected!"
    assert val_pids.isdisjoint(test_pids), "Val/test patient overlap detected!"
    print("  [OK] No patient overlap between splits")

    # Verify leakage columns removed
    for split_name, ds in ds_dict.items():
        for col in LEAKAGE_COLUMNS:
            assert col not in ds.column_names, (
                f"Leakage column '{col}' still present in {split_name}!"
            )
    print("  [OK] All GPT-4 leakage columns removed")

    # Step 5: Format
    print("\n[5/6] Formatting prompts...")
    ds_dict = format_dataset(ds_dict)

    # Step 6: Save manifest
    print("\n[6/6] Saving split manifest...")
    manifest_path = save_split_manifest(patient_splits, ds_dict, output_dir)
    print(f"  -> Manifest saved to: {manifest_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Data Preparation Complete!")
    print("=" * 60)
    for split_name, ds in ds_dict.items():
        print(f"  {split_name:12s}: {len(ds):5d} samples")
    print()

    return ds_dict, patient_splits


def load_dataset():
    """Load and return the patient-grouped split DatasetDict."""
    ds_dict, _ = prepare_datasets()
    return ds_dict


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ds_dict, patient_splits = prepare_datasets()

    # Print a sample from each split
    for split_name in ["train", "validation", "test"]:
        print("\n" + "-" * 40)
        print(f"Sample from {split_name}:")
        print("-" * 40)
        sample = ds_dict[split_name][0]
        print(f"Patient ID: {sample['patient_id']}")
        print(f"Trial ID:   {sample['trial_id']}")
        print(f"\nINPUT:\n{sample['input'][:500]}...")
        print(f"\nOUTPUT:\n{sample['output']}")

"""
Download ncbi/TrialGPT-Criterion-Annotations directly via Hugging Face REST API
with robust retry logic and perform the strict patient-grouped split (seed 42, 37 train / 8 val / 8 test).
Produces split_manifest.json and dataset json files.
"""

import json
import os
import time
import urllib.request
import numpy as np

OUTPUT_DIR = "outputs"
DATA_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

LOCAL_RAW_PATH = os.path.join(DATA_DIR, "trialgpt_criterion_annotations.json")

def fetch_batch_with_retry(offset, limit=100, max_retries=6):
    url = f"https://datasets-server.huggingface.co/rows?dataset=ncbi/TrialGPT-Criterion-Annotations&config=default&split=train&offset={offset}&limit={limit}"
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            wait_time = 2 * (attempt + 1)
            print(f"  [Attempt {attempt+1}/{max_retries}] Offset {offset} failed ({e}). Retrying in {wait_time}s...")
            time.sleep(wait_time)
    raise RuntimeError(f"Failed to fetch offset {offset} after {max_retries} retries.")

def fetch_all_annotations():
    if os.path.exists(LOCAL_RAW_PATH):
        try:
            with open(LOCAL_RAW_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if len(data) == 1015:
                    print(f"Loaded {len(data)} valid annotations from cache.")
                    return data
        except Exception:
            pass

    print("Fetching dataset rows from Hugging Face REST API...")
    offset = 0
    all_rows = []
    seen_ids = set()

    while True:
        data = fetch_batch_with_retry(offset)
        rows = data.get("rows", [])
        if not rows:
            break
        
        new_count = 0
        for item in rows:
            row = item["row"]
            ann_id = row.get("annotation_id", len(all_rows))
            if ann_id not in seen_ids:
                seen_ids.add(ann_id)
                all_rows.append(row)
                new_count += 1
                
        total_in_dataset = data.get("num_rows_total", 1015)
        print(f"  Fetched offset {offset}: added {new_count} rows, total accumulated: {len(all_rows)} / {total_in_dataset}")
        
        if len(all_rows) >= total_in_dataset or new_count == 0:
            break
        offset = len(all_rows)
        time.sleep(0.5)

    print(f"Total fetched: {len(all_rows)} annotations.")
    with open(LOCAL_RAW_PATH, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2, ensure_ascii=False)
    return all_rows

def main():
    raw_data = fetch_all_annotations()

    # Step 2: Patient IDs
    patient_ids = sorted(list(set(row["patient_id"] for row in raw_data)))
    print(f"\n[Validation] Total unique patients found: {len(patient_ids)}")
    assert len(patient_ids) == 53, f"Expected 53 patients, found {len(patient_ids)}"

    # Step 3: Patient-grouped split with NumPy default_rng(42)
    rng = np.random.default_rng(42)
    shuffled_ids = np.array(patient_ids)
    rng.shuffle(shuffled_ids)
    shuffled_ids = shuffled_ids.tolist()

    train_pids = shuffled_ids[:37]
    val_pids = shuffled_ids[37:45]
    test_pids = shuffled_ids[45:53]

    print(f"  Train patients: {len(train_pids)} (37)")
    print(f"  Val patients:   {len(val_pids)} (8)")
    print(f"  Test patients:  {len(test_pids)} (8)")

    # Strict disjointness assertion
    train_set, val_set, test_set = set(train_pids), set(val_pids), set(test_pids)
    assert train_set.isdisjoint(val_set), "Train/Val overlap!"
    assert train_set.isdisjoint(test_set), "Train/Test overlap!"
    assert val_set.isdisjoint(test_set), "Val/Test overlap!"
    print("  [OK] Patient sets are strictly disjoint (ZERO patient leakage)")

    # Step 4: Split data and remove GPT-4 leakage columns
    leakage_cols = ["gpt4_explanation", "explanation_correctness", "gpt4_sentences", "gpt4_eligibility"]
    splits_data = {"train": [], "validation": [], "test": []}

    for row in raw_data:
        pid = row["patient_id"]
        # Clean row: exclude leakage
        clean_row = {k: v for k, v in row.items() if k not in leakage_cols}
        
        if pid in train_set:
            splits_data["train"].append(clean_row)
        elif pid in val_set:
            splits_data["validation"].append(clean_row)
        elif pid in test_set:
            splits_data["test"].append(clean_row)

    print(f"\n[Split Counts]")
    print(f"  Train samples:      {len(splits_data['train'])}")
    print(f"  Validation samples: {len(splits_data['validation'])}")
    print(f"  Test samples:       {len(splits_data['test'])}")
    total_assigned = sum(len(v) for v in splits_data.values())
    assert total_assigned == len(raw_data), f"Mismatch: {total_assigned} vs {len(raw_data)}"

    # Save split JSON files
    for split_name, samples in splits_data.items():
        split_path = os.path.join(DATA_DIR, f"{split_name}.json")
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        print(f"  Saved {split_name} split to {split_path}")

    # Step 5: Save official split manifest
    manifest = {
        "dataset_name": "ncbi/TrialGPT-Criterion-Annotations",
        "random_seed": 42,
        "protocol": "patient-grouped split with numpy.random.default_rng(42)",
        "total_unique_patients": len(patient_ids),
        "total_annotations": len(raw_data),
        "splits": {
            "train": {
                "num_patients": len(train_pids),
                "num_samples": len(splits_data["train"]),
                "patient_ids": train_pids
            },
            "validation": {
                "num_patients": len(val_pids),
                "num_samples": len(splits_data["validation"]),
                "patient_ids": val_pids
            },
            "test": {
                "num_patients": len(test_pids),
                "num_samples": len(splits_data["test"]),
                "patient_ids": test_pids
            }
        }
    }

    manifest_path = os.path.join(OUTPUT_DIR, "split_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Split manifest successfully written to {manifest_path}")

if __name__ == "__main__":
    main()

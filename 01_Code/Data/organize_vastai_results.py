"""
organize_vastai_results.py
==========================
Run after downloading vast.ai results into a dated folder.

Usage:
    py organize_vastai_results.py "C:/Users/Tristan Leiter/Documents/MT/03_Output/VastAI_2026-03-28"

What it does:
    For each of the 12 models, creates a single folder named by model ID
    (e.g. M1_fund/) and copies both the Tables output and the AutoGluon
    model artifacts into it:

    VastAI_2026-03-28/
    ├── M1_fund/
    │   ├── tables/         ← from Tables/ag_fund/
    │   └── model/          ← from Models/AutoGluon/ag_fund/
    ├── M2_latent_fund/
    │   ├── tables/
    │   └── model/
    ...
"""

import sys
import shutil
from pathlib import Path

MODEL_MAP = {
    "M1": "fund",
    "M2": "latent_fund",
    "M3": "raw",
    "M4": "latent_raw",
    "B1": "bucket",
    "B2": "bucket_latent_fund",
    "B3": "bucket_raw",
    "B4": "bucket_latent_raw",
    "S1": "structural",
    "S2": "structural_latent_fund",
    "S3": "structural_raw",
    "S4": "structural_latent_raw",
}

def organize(root: Path):
    tables_root = root / "Tables"
    models_root = root / "Models" / "AutoGluon"

    if not tables_root.exists() and not models_root.exists():
        print(f"ERROR: Neither Tables/ nor Models/AutoGluon/ found under {root}")
        sys.exit(1)

    for model_id, key in MODEL_MAP.items():
        dest = root / f"{model_id}_{key}"
        dest.mkdir(exist_ok=True)

        # Tables output
        tables_src = tables_root / f"ag_{key}"
        if tables_src.exists():
            dest_tables = dest / "tables"
            if dest_tables.exists():
                shutil.rmtree(dest_tables)
            shutil.copytree(tables_src, dest_tables)
            n = len(list(dest_tables.iterdir()))
            print(f"[{model_id}] tables/ <- {tables_src.name}  ({n} files)")
        else:
            print(f"[{model_id}] tables/ — MISSING (ag_{key} not found, model may have failed)")

        # Model artifacts
        models_src = models_root / f"ag_{key}"
        if models_src.exists():
            size_mb = sum(f.stat().st_size for f in models_src.rglob("*") if f.is_file()) / 1e6
            dest_model = dest / "model"
            if dest_model.exists():
                shutil.rmtree(dest_model)
            shutil.copytree(models_src, dest_model)
            print(f"[{model_id}] model/  <- {models_src.name}  ({size_mb:.0f} MB)")
        else:
            print(f"[{model_id}] model/  — MISSING (ag_{key} not found, model may have failed)")

    print(f"\nDone. Results organized under: {root}")
    print("Each folder contains tables/ and model/ subdirectories.")
    print("Original Tables/ and Models/ subdirectories are preserved.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: py organize_vastai_results.py <path_to_VastAI_folder>")
        print('Example: py organize_vastai_results.py "C:/Users/Tristan Leiter/Documents/MT/03_Output/VastAI_2026-03-28"')
        sys.exit(1)

    root = Path(sys.argv[1])
    if not root.exists():
        print(f"ERROR: Folder not found: {root}")
        sys.exit(1)

    organize(root)

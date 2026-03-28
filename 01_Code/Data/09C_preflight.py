"""
09C_preflight.py
================
Diagnostic checks to run on the vast.ai instance BEFORE starting 09C_AutoGluon.py.
Verifies environment, data files, label integrity, and GPU availability.

Run from /workspace/MT/01_Code/Data:
    python 09C_preflight.py

A clean run prints only [OK] lines and a final READY summary.
Any [FAIL] or [WARN] must be resolved before running 09C.
"""

import os
import sys
from pathlib import Path

PASS = "[OK]  "
WARN = "[WARN]"
FAIL = "[FAIL]"
failures = []
warnings = []

def ok(msg):   print(f"{PASS} {msg}")
def warn(msg): print(f"{WARN} {msg}"); warnings.append(msg)
def fail(msg): print(f"{FAIL} {msg}"); failures.append(msg)

# ==============================================================================
# 1. Paths
# ==============================================================================
print("\n── 1. Environment & Paths ─────────────────────────────────────────────")

is_vast = "VAST_CONTAINERLABEL" in os.environ
if is_vast:
    DATA_ROOT = Path("/workspace/MT")
    ok("vast.ai environment detected")
else:
    if os.name == "nt":
        DATA_ROOT = Path(r"C:\Users\Tristan Leiter\Documents\MT")
    else:
        DATA_ROOT = Path("./MT")
    warn(f"Not on vast.ai — using local DATA_ROOT: {DATA_ROOT}")

if DATA_ROOT.exists():
    ok(f"DATA_ROOT exists: {DATA_ROOT}")
else:
    fail(f"DATA_ROOT not found: {DATA_ROOT}")

DIR_DATA     = DATA_ROOT / "02_Data"
DIR_FEATURES = DIR_DATA  / "Features"
DIR_LABELS   = DIR_DATA  / "Labels"
DIR_OUTPUT   = DATA_ROOT / "03_Output"
DIR_MODELS   = DIR_OUTPUT / "Models" / "AutoGluon"
DIR_TABLES   = DIR_OUTPUT / "Tables"

# ==============================================================================
# 2. Required input files
# ==============================================================================
print("\n── 2. Required Input Files ────────────────────────────────────────────")

PHASE1_FILES = {
    "split_labels_oot.parquet" : DIR_FEATURES / "split_labels_oot.parquet",
    "features_fund.rds"        : DIR_FEATURES / "features_fund.rds",
    "features_raw.rds"         : DIR_FEATURES / "features_raw.rds",
    "labels_bucket.rds"        : DIR_LABELS   / "labels_bucket.rds",
    "labels_structural.rds"    : DIR_LABELS   / "labels_structural.rds",
}

PHASE2_FILES = {
    "features_latent_fund.parquet" : DIR_FEATURES / "features_latent_fund.parquet",
    "features_latent_raw.parquet"  : DIR_FEATURES / "features_latent_raw.parquet",
}

phase1_ready = True
for name, path in PHASE1_FILES.items():
    if path.exists():
        size_mb = path.stat().st_size / 1e6
        ok(f"{name}  ({size_mb:.1f} MB)")
    else:
        fail(f"{name} MISSING — {path}")
        phase1_ready = False

phase2_ready = True
for name, path in PHASE2_FILES.items():
    if path.exists():
        size_mb = path.stat().st_size / 1e6
        ok(f"{name}  ({size_mb:.1f} MB)")
    else:
        warn(f"{name} not yet present — Phase 2 models will fail until 08B completes")
        phase2_ready = False

# ==============================================================================
# 3. Imports
# ==============================================================================
print("\n── 3. Python Imports ──────────────────────────────────────────────────")

import importlib
REQUIRED_PACKAGES = ["numpy", "pandas", "pyreadr", "pyarrow", "sklearn"]
for pkg in REQUIRED_PACKAGES:
    try:
        importlib.import_module(pkg)
        ok(pkg)
    except ImportError:
        fail(f"{pkg} not installed — run: pip install {pkg}")

try:
    from autogluon.tabular import TabularDataset, TabularPredictor
    ok("autogluon.tabular")
except ImportError:
    fail("autogluon not installed — run: pip install autogluon.tabular")

# ==============================================================================
# 4. GPU
# ==============================================================================
print("\n── 4. GPU / CUDA ──────────────────────────────────────────────────────")

try:
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        ok(f"CUDA available — {gpu_name}  ({vram_gb:.1f} GB VRAM)")
    else:
        warn("CUDA not available — AutoGluon will run on CPU (slower but functional)")
except ImportError:
    warn("torch not installed — cannot check GPU. AutoGluon will auto-detect.")

# ==============================================================================
# 5. Disk space
# ==============================================================================
print("\n── 5. Disk Space ──────────────────────────────────────────────────────")

import shutil
total, used, free = shutil.disk_usage(DATA_ROOT if DATA_ROOT.exists() else "/")
free_gb = free / 1e9
if free_gb >= 20:
    ok(f"Free disk space: {free_gb:.1f} GB")
elif free_gb >= 5:
    warn(f"Low disk space: {free_gb:.1f} GB free — AutoGluon model artifacts can be large")
else:
    fail(f"Very low disk space: {free_gb:.1f} GB — likely to fail during training")

# ==============================================================================
# 6. Data integrity spot-checks
# ==============================================================================
print("\n── 6. Data Integrity ──────────────────────────────────────────────────")

if phase1_ready:
    try:
        import pyreadr
        import pandas as pd

        # Check split file
        splits = pd.read_parquet(DIR_FEATURES / "split_labels_oot.parquet")
        expected_cols = {"permno", "year", "eval_split"}
        missing = expected_cols - set(splits.columns)
        if missing:
            fail(f"split_labels_oot.parquet missing columns: {missing}")
        else:
            ok(f"split_labels_oot.parquet  shape={splits.shape}  splits={sorted(splits['eval_split'].unique())}")

        # Check features_fund
        fund_rds = pyreadr.read_r(str(DIR_FEATURES / "features_fund.rds"))
        fund = list(fund_rds.values())[0]
        if fund.shape[0] < 1000:
            warn(f"features_fund.rds has only {fund.shape[0]} rows — unexpectedly small")
        else:
            ok(f"features_fund.rds  shape={fund.shape}")

        # Check y column exists (y_next is created at runtime via shift(-1))
        if "y" in fund.columns:
            vc = fund["y"].value_counts(dropna=False)
            ok(f"y (CSI label, pre-shift) counts: {vc.to_dict()}")
        else:
            fail("y column not found in features_fund.rds — CSI models will fail")

        # Check features_raw
        raw_rds = pyreadr.read_r(str(DIR_FEATURES / "features_raw.rds"))
        raw = list(raw_rds.values())[0]
        ok(f"features_raw.rds  shape={raw.shape}")

        # Check bucket labels
        lbl_path = DIR_LABELS / "labels_bucket.rds"
        if lbl_path.exists():
            lbl = list(pyreadr.read_r(str(lbl_path)).values())[0]
            if "y_loser" in lbl.columns:
                ok(f"labels_bucket.rds — y_loser counts: {lbl['y_loser'].value_counts(dropna=False).to_dict()}")
            else:
                fail("y_loser column not found in labels_bucket.rds")

        # Check structural labels
        lbl_path = DIR_LABELS / "labels_structural.rds"
        if lbl_path.exists():
            lbl = list(pyreadr.read_r(str(lbl_path)).values())[0]
            if "y_structural" in lbl.columns:
                ok(f"labels_structural.rds — y_structural counts: {lbl['y_structural'].value_counts(dropna=False).to_dict()}")
            else:
                fail("y_structural column not found in labels_structural.rds")

    except Exception as e:
        fail(f"Data integrity check error: {e}")
else:
    warn("Skipping data integrity checks — Phase 1 files incomplete")

if phase2_ready:
    try:
        import pandas as pd
        for name, path in PHASE2_FILES.items():
            df = pd.read_parquet(path)
            ok(f"{name}  shape={df.shape}")
    except Exception as e:
        fail(f"Phase 2 parquet check error: {e}")

# ==============================================================================
# 7. Output directory writability
# ==============================================================================
print("\n── 7. Output Directories ──────────────────────────────────────────────")

for d in [DIR_MODELS, DIR_TABLES]:
    try:
        d.mkdir(parents=True, exist_ok=True)
        test_file = d / ".write_test"
        test_file.touch()
        test_file.unlink()
        ok(f"Writable: {d}")
    except Exception as e:
        fail(f"Cannot write to {d}: {e}")

# ==============================================================================
# Summary
# ==============================================================================
print("\n" + "=" * 70)
if not failures:
    phase_str = "Phase 1 + Phase 2" if phase2_ready else "Phase 1 only (Phase 2 awaits 08B)"
    print(f"READY — {phase_str} models can run.")
    if warnings:
        print(f"  {len(warnings)} warning(s) noted above — review before proceeding.")
else:
    print(f"NOT READY — {len(failures)} failure(s) must be resolved:")
    for f in failures:
        print(f"  • {f}")
print("=" * 70)
sys.exit(0 if not failures else 1)

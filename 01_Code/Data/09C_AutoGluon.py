"""
09C_AutoGluon.py  (v7 — M1/M2/M3/M4 · B1/B2/B3/B4 · S1/S2/S3/S4)
====================================================================
AutoGluon AutoML for CSI prediction, Bucket classification,
and Structural (combined) classification.

══ MODEL MATRIX ═══════════════════════════════════════════════════════════════

  Track 1 — CSI (acute crash prediction, label shift y(t+1))
    M1  fund          Fundamentals only          → ag_fund/
    M2  latent_fund   VAE latent on fund          → ag_latent_fund/
    M3  raw           Full engineered features    → ag_raw/
    M4  latent_raw    VAE latent on full raw       → ag_latent_raw/

  Track 2 — Bucket (5yr terminal loser, no label shift)
    B1  bucket                fund features       → ag_bucket/
    B2  bucket_latent_fund    VAE latent on fund   → ag_bucket_latent_fund/
    B3  bucket_raw            full features        → ag_bucket_raw/
    B4  bucket_latent_raw     VAE latent on raw    → ag_bucket_latent_raw/

  Track 3 — Structural (combined CSI + 5yr CAGR, no label shift)
    S1  structural                 fund features  → ag_structural/
    S2  structural_latent_fund     VAE on fund     → ag_structural_latent_fund/
    S3  structural_raw             full features   → ag_structural_raw/
    S4  structural_latent_raw      VAE on raw      → ag_structural_latent_raw/

  Set MODEL = one of the keys above and run. Results go to separate dirs.

══ EVALUATION DESIGN ══════════════════════════════════════════════════════════

  Train          : 1993–2015
  Test (metrics) : 2016–2019  ← SOLE basis for cross-model comparison
  Live OOS       : 2020–2024  ← index construction ONLY, never model selection

  All 12 models evaluated on identical 2016–2019 window so comparisons
  are apples-to-apples across tracks.

  For B1–S4: no labels exist beyond 2019 (5yr forward window requires
  data through 2024–2029). OOS predictions are generated from 2020–2024
  features and used as exclusion scores for the index backtest.
  No OOS labels needed — the index backtest compares returns, not AUC.

══ KEY FIXES vs v5 ════════════════════════════════════════════════════════════

  1. Split boundary leakage:
       CSI train year 2015 → y_next = y(2016) [test label] → excluded from metrics.
       CSI test year 2019  → y_next = y(2020) [OOS label]  → excluded from metrics.
       eval_split column from 08_Split.R drives exclusion.
       Both years STILL receive predictions for index construction.

  2. year_cat created and passed to AutoGluon (bucket/structural models).
       Previously referenced but never instantiated.

  3. Per-fold CV metric averaging (not pooling).
       Pooling conflates data-volume effects with model quality.

  4. OOS metrics for M1–M4 stored but flagged as index-only.
       Model selection uses test 2016–2019 exclusively.

  5. Full M1–M4 / B1–B4 / S1–S4 MODEL_CONFIG (was M1/M3/B1/structural only).

  6. 08B encode() fix (apply separately in 08B_Autoencoder.py):
       Use z_mu not z_samp for reconstruction error in encode().

══ RUN ORDER ═══════════════════════════════════════════════════════════════════

  Phase 1 (no VAE dependency):
    M1 → M3 → B1 → B3 → S1 → S3

  Phase 2 (after 08B completes for both VAE_INPUT settings):
    M2 → M4 → B2 → B4 → S2 → S4
"""

# ==============================================================================
# 0. Imports
# ==============================================================================

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import QuantileTransformer

warnings.filterwarnings("ignore")

try:
    from autogluon.tabular import TabularDataset, TabularPredictor
except ImportError:
    raise ImportError("AutoGluon not installed. Run: pip install autogluon.tabular")

# ==============================================================================
# 1. Paths
# ==============================================================================

import os # Ensure os is imported here if not already at the top

# Check for a vast.ai specific environment variable to switch environments
is_vast = "VAST_CONTAINERLABEL" in os.environ

if is_vast:
    print("[09C] Environment: vast.ai detected.")
    DATA_ROOT = Path("/workspace/MT")
else:
    print("[09C] Environment: Local (default).")
    # Default local paths based on OS
    if os.name == "nt":
        DATA_ROOT = Path(r"C:\Users\Tristan Leiter\Documents\MT")
    else:
        # Update this if you ever run locally on Mac/Linux
        DATA_ROOT = Path("./MT")

print(f"[09C] DATA_ROOT: {DATA_ROOT}")
assert DATA_ROOT.exists(), f"DATA_ROOT not found: {DATA_ROOT}"

DIR_DATA     = DATA_ROOT / "02_Data"
DIR_FEATURES = DIR_DATA  / "Features"
DIR_LABELS   = DIR_DATA  / "Labels"
DIR_OUTPUT   = DATA_ROOT / "03_Output"
DIR_MODELS   = DIR_OUTPUT / "Models" / "AutoGluon"
DIR_TABLES   = DIR_OUTPUT / "Tables"

PATH_SPLIT_LABELS      = DIR_FEATURES / "split_labels_oot.parquet"
PATH_LABELS_BUCKET     = DIR_LABELS   / "labels_bucket.rds"
PATH_LABELS_STRUCTURAL = DIR_LABELS   / "labels_structural.rds"

assert PATH_SPLIT_LABELS.exists(), \
    f"split_labels_oot.parquet not found. Run revised 08_Split.R first.\n{PATH_SPLIT_LABELS}"

# ==============================================================================
# 2. ── MODEL SELECTION ────────────────────────────────────────────────────────
#
#   Set MODEL to one of the 12 keys listed in the header.
#   Run this script once per model.
#
# Phase 1(no VAE dependency):
# M1 → M3 → B1 → B3 → S1 → S3
#
# Phase 2(after 08B completes for both VAE_INPUT settings):
# M2 → M4 → B2 → B4 → S2 → S4
# ==============================================================================

MODEL = "M1"   # <- CHANGE THIS

# ==============================================================================
# 3. Training hyperparameters
# ==============================================================================

SEED          = 123
TIME_LIMIT    = 1800    # seconds, Stage 1 full training
PRESET        = "good_quality"
CV_TIME_LIMIT = 900     # seconds per CV fold
CV_PRESET     = "medium_quality"
EVAL_METRIC   = "average_precision"

# Period constants — must match config.R
TRAIN_END_YEAR  = 2015
TEST_START_YEAR = 2016
TEST_END_YEAR   = 2019
OOS_START_YEAR  = 2020

# ==============================================================================
# 4. MODEL_CONFIG — one entry per model key
# ==============================================================================

LABEL_CSI        = "y_next"
LABEL_BUCKET     = "y_loser"
LABEL_STRUCTURAL = "y_structural"

MODEL_CONFIG = {

    # ── Track 1: CSI ──────────────────────────────────────────────────────────
    "fund": {
        "description": "M1 — Fundamentals only (no price features)",
        "path"       : DIR_FEATURES / "features_fund.rds",
        "loader"     : "rds",
        "label_col"  : LABEL_CSI,
        "is_bucket"  : False,
        "prereq"     : "Run 06B_Feature_Eng.R first.",
    },
    "latent_fund": {
        "description": "M2 — VAE latent on fundamentals",
        "path"       : DIR_FEATURES / "features_latent_fund.parquet",
        "loader"     : "parquet",
        "label_col"  : LABEL_CSI,
        "is_bucket"  : False,
        "prereq"     : "Run 08B_Autoencoder.py with VAE_INPUT='fund' first.",
    },
    "raw": {
        "description": "M3 — Full engineered features",
        "path"       : DIR_FEATURES / "features_raw.rds",
        "loader"     : "rds",
        "label_col"  : LABEL_CSI,
        "is_bucket"  : False,
        "prereq"     : "Run 06B_Feature_Eng.R first.",
    },
    "latent_raw": {
        "description": "M4 — VAE latent on full raw features",
        "path"       : DIR_FEATURES / "features_latent_raw.parquet",
        "loader"     : "parquet",
        "label_col"  : LABEL_CSI,
        "is_bucket"  : False,
        "prereq"     : "Run 08B_Autoencoder.py with VAE_INPUT='raw' first.",
    },

    # ── Track 2: Bucket ───────────────────────────────────────────────────────
    "bucket": {
        "description": "B1 — 5yr forward CAGR bucket, fundamentals only",
        "path"       : DIR_FEATURES / "features_fund.rds",
        "loader"     : "rds",
        "label_col"  : LABEL_BUCKET,
        "is_bucket"  : True,
        "prereq"     : "Run 05B_Bucket_Labels.R first.",
    },
    "bucket_latent_fund": {
        "description": "B2 — 5yr bucket, VAE latent on fundamentals",
        "path"       : DIR_FEATURES / "features_latent_fund.parquet",
        "loader"     : "parquet",
        "label_col"  : LABEL_BUCKET,
        "is_bucket"  : True,
        "prereq"     : "Run 08B with VAE_INPUT='fund' and 05B first.",
    },
    "bucket_raw": {
        "description": "B3 — 5yr bucket, full engineered features",
        "path"       : DIR_FEATURES / "features_raw.rds",
        "loader"     : "rds",
        "label_col"  : LABEL_BUCKET,
        "is_bucket"  : True,
        "prereq"     : "Run 05B_Bucket_Labels.R and 06B_Feature_Eng.R first.",
    },
    "bucket_latent_raw": {
        "description": "B4 — 5yr bucket, VAE latent on full raw features",
        "path"       : DIR_FEATURES / "features_latent_raw.parquet",
        "loader"     : "parquet",
        "label_col"  : LABEL_BUCKET,
        "is_bucket"  : True,
        "prereq"     : "Run 08B with VAE_INPUT='raw' and 05B first.",
    },

    # ── Track 3: Structural ───────────────────────────────────────────────────
    "structural": {
        "description": "S1 — Structural (CSI + 5yr CAGR), fundamentals only",
        "path"       : DIR_FEATURES / "features_fund.rds",
        "loader"     : "rds",
        "label_col"  : LABEL_STRUCTURAL,
        "is_bucket"  : True,
        "prereq"     : "Run 05C_Structural_Labels.R first.",
    },
    "structural_latent_fund": {
        "description": "S2 — Structural, VAE latent on fundamentals",
        "path"       : DIR_FEATURES / "features_latent_fund.parquet",
        "loader"     : "parquet",
        "label_col"  : LABEL_STRUCTURAL,
        "is_bucket"  : True,
        "prereq"     : "Run 08B with VAE_INPUT='fund' and 05C first.",
    },
    "structural_raw": {
        "description": "S3 — Structural, full engineered features",
        "path"       : DIR_FEATURES / "features_raw.rds",
        "loader"     : "rds",
        "label_col"  : LABEL_STRUCTURAL,
        "is_bucket"  : True,
        "prereq"     : "Run 05C_Structural_Labels.R and 06B_Feature_Eng.R first.",
    },
    "structural_latent_raw": {
        "description": "S4 — Structural, VAE latent on full raw features",
        "path"       : DIR_FEATURES / "features_latent_raw.parquet",
        "loader"     : "parquet",
        "label_col"  : LABEL_STRUCTURAL,
        "is_bucket"  : True,
        "prereq"     : "Run 08B with VAE_INPUT='raw' and 05C first.",
    },
}

assert MODEL in MODEL_CONFIG, \
    f"Unknown MODEL '{MODEL}'. Valid keys:\n  {list(MODEL_CONFIG.keys())}"

cfg       = MODEL_CONFIG[MODEL]
LABEL_COL = cfg["label_col"]
IS_BUCKET = cfg["is_bucket"]

assert cfg["path"].exists(), \
    f"Feature file not found: {cfg['path']}\n  {cfg['prereq']}"

## For bucket/structural models, verify label file exists
if IS_BUCKET:
    lbl_path = (PATH_LABELS_STRUCTURAL
                if LABEL_COL == LABEL_STRUCTURAL
                else PATH_LABELS_BUCKET)
    assert lbl_path.exists(), \
        f"Label file not found: {lbl_path}\n  {cfg['prereq']}"

## Output directories — one per model key, nothing ever overwrites another
DIR_MODELS_RUN = DIR_MODELS / f"ag_{MODEL}"
DIR_TABLES_RUN = DIR_TABLES / f"ag_{MODEL}"
DIR_MODELS_RUN.mkdir(parents=True, exist_ok=True)
DIR_TABLES_RUN.mkdir(parents=True, exist_ok=True)

print(f"\n[09C] {'='*54}")
print(f"  MODEL        : {MODEL}")
print(f"  Description  : {cfg['description']}")
print(f"  Features     : {cfg['path'].name}")
print(f"  Label        : {LABEL_COL}")
print(f"  Track        : {'Bucket/Structural (no label shift)' if IS_BUCKET else 'CSI (label shift y→y_next)'}")
print(f"  Output       : {DIR_TABLES_RUN.name}/")
print(f"[09C] {'='*54}\n")

# ==============================================================================
# 5. ID column exclusion list
# ==============================================================================

ID_COLS = [
    "permno", "year", "y", "censored", "param_id",
    "gvkey", "datadate", "lifetime_years", "fiscal_year_end_month",
    "split", "eval_split", "vae_split", "split_oot",
    "y_loser", "y_structural", "fwd_cagr", "n_months", "bucket",
    "year_cat",
]

LATENT_COLS = [f"z{i}" for i in range(1, 25)] + ["vae_recon_error"]

# ==============================================================================
# 6. Load feature data
# ==============================================================================

print(f"[09C] Loading features: {cfg['path'].name} ...")

if cfg["loader"] == "rds":
    features_input = pyreadr.read_r(str(cfg["path"]))[None]
elif cfg["loader"] == "parquet":
    features_input = pd.read_parquet(cfg["path"])
    ## VAE parquet uses 'split' for the VAE train/val/test split —
    ## rename to avoid collision with OOT split labels
    if "split" in features_input.columns:
        features_input = features_input.rename(columns={"split": "vae_split"})

print(f"  Shape: {features_input.shape[0]:,} × {features_input.shape[1]}")

# ==============================================================================
# 7A. CSI models — merge OOT split labels, apply label shift, mark boundaries
# ==============================================================================

if not IS_BUCKET:

    print("[09C] Loading OOT split labels ...")
    split_labels = pd.read_parquet(PATH_SPLIT_LABELS)

    ## Expect both 'split' and 'eval_split' from revised 08_Split.R
    if "eval_split" not in split_labels.columns:
        print("  WARNING: eval_split column missing — re-run revised 08_Split.R.")
        print("  Falling back to hardcoded boundary exclusion (2015, 2019).")
        split_labels["eval_split"] = np.where(
            (split_labels["split"] == "train") & (split_labels["year"] == TRAIN_END_YEAR),
            "train_boundary",
            np.where(
                (split_labels["split"] == "test") & (split_labels["year"] == TEST_END_YEAR),
                "test_boundary",
                split_labels["split"]
            )
        )

    df = (features_input
          .merge(split_labels, on=["permno", "year"], how="left")
          .query("split.notna()")
          .sort_values(["permno", "year"])
          .reset_index(drop=True))

    ## Label shift: y(t) → y_next = y(t+1)
    ## Applied within each firm so the shift never crosses firm boundaries
    df["y_next"] = df.groupby("permno")["y"].shift(-1)

    ## Rows with a valid shifted label
    df_with_label = df[df["y_next"].notna()].copy()
    df_with_label["y_next"] = df_with_label["y_next"].astype(int)

    ## Eval-safe: exclude boundary years from AUC/AP computation
    ## (their y_next comes from the next period after label shift)
    BOUNDARY_FLAGS = {"train_boundary", "test_boundary"}
    df_eval = df_with_label[
        ~df_with_label["eval_split"].isin(BOUNDARY_FLAGS)
    ].copy()

    print(f"\n[09C] Label shift applied:")
    print(f"  Rows with y_next       : {len(df_with_label):,}")
    print(f"  Eval-safe (no boundary): {len(df_eval):,}  "
          f"({len(df_with_label)-len(df_eval):,} boundary rows excluded from metrics)")
    print(f"  y_next prevalence      : {df_eval['y_next'].mean():.4f}")

    ## Full splits — for prediction generation (index construction)
    ## Includes boundary years because we need predictions for every year
    train_all = df_with_label[df_with_label["split"] == "train"].copy()
    test_all  = df_with_label[df_with_label["split"] == "test"].copy()
    oos_all   = df_with_label[df_with_label["split"] == "oos"].copy()

    ## Eval-safe splits — for AUC/AP metrics only
    train_eval = df_eval[df_eval["split"] == "train"].copy()
    test_eval  = df_eval[df_eval["split"] == "test"].copy()
    oos_eval   = df_eval[df_eval["split"] == "oos"].copy()

    print(f"\n[09C] Split sizes  [all rows / eval-safe rows]:")
    print(f"  Train : {len(train_all):,} / {len(train_eval):,}  "
          f"prev={train_eval['y_next'].mean():.4f}")
    print(f"  Test  : {len(test_all):,} / {len(test_eval):,}  "
          f"prev={test_eval['y_next'].mean():.4f}")
    print(f"  OOS   : {len(oos_all):,} / {len(oos_eval):,}  "
          f"(OOS metrics stored but not used for model selection)")

    ## Convenience aliases used downstream
    train_df = train_eval   ## model is trained on eval-safe train set
    test_df  = test_eval
    oos_df   = oos_eval

# ==============================================================================
# 7B. Bucket / Structural models — merge label file, no label shift
# ==============================================================================

else:

    lbl_path  = (PATH_LABELS_STRUCTURAL
                 if LABEL_COL == LABEL_STRUCTURAL
                 else PATH_LABELS_BUCKET)
    lbl_raw   = pyreadr.read_r(str(lbl_path))[None]

    ## Keep only rows with a valid (non-NA) label
    lbl_clean = lbl_raw[lbl_raw[LABEL_COL].notna()].copy()
    lbl_clean[LABEL_COL] = lbl_clean[LABEL_COL].astype(int)

    print(f"[09C] Labels loaded: {len(lbl_clean):,} usable rows  "
          f"prevalence={lbl_clean[LABEL_COL].mean():.4f}  "
          f"years {lbl_clean['year'].min()}–{lbl_clean['year'].max()}")

    df_labelled = (features_input
                   .merge(lbl_clean[["permno", "year", LABEL_COL]],
                          on=["permno", "year"], how="inner")
                   .copy())

    ## year_cat: categorical year feature for period fixed-effect absorption.
    ## The 5yr forward CAGR label is period-dependent — bull markets produce
    ## fewer terminal losers mechanically. year_cat lets AutoGluon absorb this
    ## calendar-time trend so firm features predict the cross-sectional deviation.
    df_labelled["year_cat"] = df_labelled["year"].astype(str).astype("category")

    ## ── Generate OOS features for 2020–2024 (no labels, inference only) ──
    ## We apply the trained model to 2020–2024 features to get exclusion scores
    ## for the index backtest. Labels are not needed for this step.
    ## For feature rows in 2020–2024, merge features_input directly.
    oos_features = features_input[
        features_input["year"] >= OOS_START_YEAR
    ].copy()
    oos_features["year_cat"] = oos_features["year"].astype(str).astype("category")
    ## No label column for OOS — added as NaN placeholder so make_preds handles it
    oos_features[LABEL_COL] = np.nan

    ## Splits — train/test use only labelled rows; OOS uses feature-only rows
    train_df = df_labelled[df_labelled["year"] <= TRAIN_END_YEAR].copy()
    test_df  = df_labelled[
        (df_labelled["year"] >= TEST_START_YEAR) &
        (df_labelled["year"] <= TEST_END_YEAR)
    ].copy()

    ## All/eval distinction does not apply for bucket models (no label shift)
    train_all = train_df
    test_all  = test_df
    oos_all   = oos_features   ## feature-only, no labels
    oos_df    = test_df        ## placeholder — oos metrics N/A for bucket

    print(f"\n[09C] Split sizes (bucket/structural):")
    print(f"  Train (≤{TRAIN_END_YEAR})       : {len(train_df):,}  "
          f"prev={train_df[LABEL_COL].mean():.4f}")
    print(f"  Test  ({TEST_START_YEAR}–{TEST_END_YEAR})  : {len(test_df):,}  "
          f"prev={test_df[LABEL_COL].mean():.4f}")
    print(f"  OOS   ({OOS_START_YEAR}–2024)  : {len(oos_features):,}  "
          f"(features only — no labels, used for index construction)")

# ==============================================================================
# 8. Feature columns
# ==============================================================================

## Latent models: use only VAE latent dimensions + reconstruction error
if MODEL in ("latent_fund", "latent_raw",
             "bucket_latent_fund", "bucket_latent_raw",
             "structural_latent_fund", "structural_latent_raw"):
    feature_cols = [c for c in LATENT_COLS if c in train_df.columns]
    print(f"\n[09C] Latent feature columns: {len(feature_cols)}")

else:
    feature_cols = [
        c for c in train_df.columns
        if c not in ID_COLS
        and c not in [LABEL_COL, "y_next", "y", "y_loser", "y_structural"]
        and pd.api.types.is_numeric_dtype(train_df[c])
    ]
    print(f"\n[09C] Feature columns: {len(feature_cols)}"
          + (f" + 1 categorical (year_cat)" if IS_BUCKET else ""))

feature_cols_model = feature_cols

## AutoGluon column set — includes year_cat for bucket/structural
ag_cols = feature_cols_model + ([LABEL_COL] if not IS_BUCKET
                                  else ["year_cat", LABEL_COL])

# ==============================================================================
# 9. Preprocessing — winsorise / impute / quantile transform
#
#   All statistics (percentiles, medians, QT quantiles) fitted on the
#   eval-safe train set ONLY. Applied consistently to all other sets.
# ==============================================================================

print("\n[09C] Preprocessing (winsorise → impute → quantile transform)...")

def to_float64(df, cols):
    arr = df[cols].values.astype(np.float64)
    arr[np.isinf(arr)] = np.nan
    return arr

X_tr = to_float64(train_df, feature_cols)

## Collect all arrays that need the same transforms
## OOS for bucket uses oos_features (feature-only); for CSI uses oos_eval
if IS_BUCKET:
    transform_sets = {
        "train": (train_df,    X_tr),
        "test" : (test_df,     to_float64(test_df,     feature_cols)),
        "oos"  : (oos_features, to_float64(oos_features, feature_cols)),
    }
else:
    transform_sets = {
        "train_eval" : (train_eval, X_tr),
        "test_eval"  : (test_eval,  to_float64(test_eval,  feature_cols)),
        "oos_eval"   : (oos_eval,   to_float64(oos_eval,   feature_cols)),
        "train_all"  : (train_all,  to_float64(train_all,  feature_cols)),
        "test_all"   : (test_all,   to_float64(test_all,   feature_cols)),
        "oos_all"    : (oos_all,    to_float64(oos_all,    feature_cols)),
    }

## Winsorise using train percentiles
lo = np.nanpercentile(X_tr, 0.1,  axis=0)
hi = np.nanpercentile(X_tr, 99.9, axis=0)

## Median imputation using train medians
medians = np.nanmedian(np.clip(X_tr, lo, hi), axis=0)
medians = np.where(np.isnan(medians), 0.0, medians)

def preprocess(X, lo, hi, medians):
    X = np.clip(X, lo, hi)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        if mask.any():
            X[mask, j] = medians[j]
    return X

## Quantile transform fitted on train only
qt = QuantileTransformer(
    output_distribution="uniform",
    n_quantiles=min(1000, len(X_tr)),
    random_state=SEED
)
qt.fit(preprocess(X_tr.copy(), lo, hi, medians))

## Apply to all sets
processed = {}
for key, (src_df, X_raw) in transform_sets.items():
    X_pp  = preprocess(X_raw.copy(), lo, hi, medians)
    X_qt  = qt.transform(X_pp)
    out   = pd.DataFrame(X_qt, columns=feature_cols, index=src_df.index)
    out[LABEL_COL] = src_df[LABEL_COL].values
    out["year"]    = src_df["year"].values
    out["permno"]  = src_df["permno"].values
    if IS_BUCKET and "year_cat" in src_df.columns:
        out["year_cat"] = src_df["year_cat"].values
    processed[key] = out

assert not np.isnan(processed["train" if IS_BUCKET else "train_eval"]
                    [feature_cols].values).any(), \
    "NaN remains in train set after preprocessing"

train_qt = processed["train" if IS_BUCKET else "train_eval"]
test_qt  = processed["test"  if IS_BUCKET else "test_eval"]
oos_qt   = processed["oos"   if IS_BUCKET else "oos_eval"]

if not IS_BUCKET:
    train_all_qt = processed["train_all"]
    test_all_qt  = processed["test_all"]
    oos_all_qt   = processed["oos_all"]

print(f"  Train shape : {train_qt.shape}")
print(f"  QT mean (~0.5) : {train_qt[feature_cols].values.mean():.4f}")
print(f"  QT std  (~0.29): {train_qt[feature_cols].values.std():.4f}")

# ==============================================================================
# 10. Stage 1 fold boundaries
#
#   Holdout = 2011–2014 for CSI (2015 excluded — boundary year after shift)
#           = 2011–2015 for bucket/structural (no label shift)
#   CV folds: expanding window starting from fold 2.
#   Fold 1 years (1993–earliest block) always serve as initial training data.
# ==============================================================================

HOLDOUT_START = 2011
HOLDOUT_END   = 2014 if not IS_BUCKET else 2015

## CV fold boundaries: (fold_id, train_end_year, val_start_year, val_end_year)
## For CSI models: last year of training window is a boundary year and is
## excluded from fold training (its y_next = label from val period).
FOLD_BOUNDARIES = [
    (2, 2001, 2002, 2006),
    (3, 2006, 2007, 2010),
    (4, 2010, 2011, HOLDOUT_END),
]

print(f"\n[09C] Holdout: {HOLDOUT_START}–{HOLDOUT_END}")
print(f"[09C] CV fold structure (expanding window, folds 2–4):")
for fid, tend, vst, vend in FOLD_BOUNDARIES:
    n_tr  = (train_qt["year"] <= tend).sum()
    n_val = ((train_qt["year"] >= vst) & (train_qt["year"] <= vend)).sum()
    bnd   = f"  [CSI: year {tend} excluded from fold fit]" if not IS_BUCKET else ""
    print(f"  Fold {fid}: train ≤{tend} n={n_tr:,} | val {vst}–{vend} n={n_val:,}{bnd}")

# ==============================================================================
# 11. Stage 1 — AutoGluon training
# ==============================================================================

print(f"\n[09C] ══ STAGE 1 [{MODEL}] ══  preset={PRESET}  time={TIME_LIMIT}s\n")

holdout_mask  = (train_qt["year"] >= HOLDOUT_START) & (train_qt["year"] <= HOLDOUT_END)
ag_train_data = TabularDataset(train_qt[~holdout_mask][ag_cols].reset_index(drop=True))
ag_holdout    = TabularDataset(train_qt[ holdout_mask][ag_cols].reset_index(drop=True))
ag_test_data  = TabularDataset(test_qt[ag_cols].reset_index(drop=True)) \
                if len(test_qt) > 0 else None

print(f"  Training rows : {len(ag_train_data):,}")
print(f"  Holdout rows  : {len(ag_holdout):,}")
if ag_test_data is not None:
    print(f"  Test rows     : {len(ag_test_data):,}  (eval-safe 2016–2018 for CSI, 2016–2019 for bucket)")

fit_kwargs = dict(
    train_data       = ag_train_data,
    tuning_data      = ag_holdout,
    presets          = PRESET,
    time_limit       = TIME_LIMIT,
    num_bag_folds    = 0,
    num_stack_levels = 0,
)
if IS_BUCKET:
    fit_kwargs["feature_metadata"] = {"year_cat": "category"}

predictor = TabularPredictor(
    label        = LABEL_COL,
    problem_type = "binary",
    eval_metric  = EVAL_METRIC,
    path         = str(DIR_MODELS_RUN / "ag_predictor"),
    verbosity    = 2,
).fit(**fit_kwargs)

print(f"\n[09C] Stage 1 training complete.")

# ==============================================================================
# 12. Leaderboard and feature importance
# ==============================================================================

print(f"\n[09C] Leaderboard (holdout):\n")
lb = predictor.leaderboard(ag_holdout, silent=True)
print(lb[["model", "score_val", "pred_time_val", "fit_time"]].to_string(index=False))
lb.to_csv(DIR_TABLES_RUN / "ag_leaderboard.csv", index=False)

print(f"\n[09C] Feature importance ...")
try:
    fi = predictor.feature_importance(ag_holdout, silent=True)
    fi.to_csv(DIR_TABLES_RUN / "ag_feature_importance.csv")
    print(f"  Top 10:\n{fi.head(10).to_string()}")
except Exception as e:
    print(f"  Skipped: {e}")

# ==============================================================================
# 13. Predictions
#
#   Eval predictions  → eval-safe rows  → AUC/AP metrics in thesis
#   Full predictions  → all split rows  → index construction in 11_Results.R
#
#   For bucket/structural OOS: feature-only rows, no labels.
#   p_csi is the exclusion score used in 11_Results.R regardless of labels.
# ==============================================================================

print(f"\n[09C] Generating predictions ...")

def make_preds(predictor, qt_df, ag_cols, source_df, label_col):
    """Generate probability predictions and package with identifiers."""
    if len(qt_df) == 0:
        return pd.DataFrame()
    proba = predictor.predict_proba(
        TabularDataset(qt_df[ag_cols].reset_index(drop=True)),
        as_multiclass=False
    )
    return pd.DataFrame({
        "permno": source_df["permno"].values,
        "year"  : source_df["year"].values,
        "y"     : source_df[label_col].values,   ## NaN for bucket OOS rows
        "p_csi" : proba.values,                   ## exclusion score for 11_Results.R
    })

## ── Eval predictions (for AUC/AP metrics) ───────────────────────────────────
preds_test_eval = make_preds(predictor, test_qt,  ag_cols, test_df,  LABEL_COL)
## OOS eval: only meaningful for CSI models (bucket/structural have no OOS labels)
preds_oos_eval  = make_preds(predictor, oos_qt, ag_cols,
                              oos_df if not IS_BUCKET else pd.DataFrame(),
                              LABEL_COL) if not IS_BUCKET else pd.DataFrame()

## ── Full predictions (for index construction) ────────────────────────────────
if IS_BUCKET:
    ## For bucket/structural: full = eval for test; OOS uses feature-only rows
    preds_test_full = preds_test_eval
    preds_oos_full  = make_preds(predictor, oos_qt, ag_cols, oos_features, LABEL_COL)
else:
    ## For CSI: full includes boundary years (2015 in train, 2019 in test)
    preds_test_full = make_preds(predictor, test_all_qt, ag_cols, test_all, LABEL_COL)
    preds_oos_full  = make_preds(predictor, oos_all_qt,  ag_cols, oos_all,  LABEL_COL)
    ## Train boundary year (2015) predictions — needed for full index history
    bnd_rows = train_all[train_all["year"] == TRAIN_END_YEAR].copy()
    bnd_qt   = processed.get("train_all", train_all_qt)
    bnd_qt   = bnd_qt[bnd_qt["year"] == TRAIN_END_YEAR]
    preds_train_boundary = make_preds(predictor, bnd_qt, ag_cols, bnd_rows, LABEL_COL)

## ── Save all prediction files ────────────────────────────────────────────────
def save_preds(df, fname, note=""):
    if len(df) > 0:
        df.to_parquet(DIR_TABLES_RUN / fname, index=False)
        print(f"  {fname:<45} {len(df):>7,} rows  {note}")

save_preds(preds_test_eval, "ag_preds_test_eval.parquet",  "← eval-safe metrics")
save_preds(preds_test_full, "ag_preds_test.parquet",       "← index construction")
save_preds(preds_oos_eval,  "ag_preds_oos_eval.parquet",   "← OOS metrics (CSI only)")
save_preds(preds_oos_full,  "ag_preds_oos.parquet",        "← index construction")
if not IS_BUCKET:
    save_preds(preds_train_boundary, "ag_preds_train_boundary.parquet",
               f"← year={TRAIN_END_YEAR} for index construction")

# ==============================================================================
# 14. Evaluation helpers
# ==============================================================================

def fn_recall_at_fpr(y_true, y_pred, fpr_target):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    idx = np.where(fpr <= fpr_target)[0]
    return 0.0 if len(idx) == 0 else float(tpr[idx].max())

def fn_eval(y_true, y_pred, label):
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    ok = ~np.isnan(yt) & ~np.isnan(yp)
    yt, yp = yt[ok], yp[ok]
    if len(np.unique(yt)) < 2:
        return None
    return {
        "set"         : label,
        "n_obs"       : int(len(yt)),
        "n_pos"       : int(yt.sum()),
        "prevalence"  : round(float(yt.mean()), 4),
        "auc_roc"     : round(float(roc_auc_score(yt, yp)), 4),
        "avg_precision": round(float(average_precision_score(yt, yp)), 4),
        "recall_fpr1" : round(fn_recall_at_fpr(yt, yp, 0.01), 4),
        "recall_fpr3" : round(fn_recall_at_fpr(yt, yp, 0.03), 4),
        "recall_fpr5" : round(fn_recall_at_fpr(yt, yp, 0.05), 4),
        "recall_fpr10": round(fn_recall_at_fpr(yt, yp, 0.10), 4),
        "brier"       : round(float(np.mean((yp - yt)**2)), 4),
    }

# ==============================================================================
# 15. Stage 1 evaluation
#
#   Test metrics (2016–2018 for CSI, 2016–2019 for bucket) = model selection.
#   OOS metrics (2020–2024, CSI only) = stored for completeness,
#   NOT used for model selection or cross-model comparison.
# ==============================================================================

print(f"\n[09C] ── Stage 1 Evaluation [{MODEL}] ──\n")
print(f"  (Test 2016–{'2018' if not IS_BUCKET else '2019'} = model selection basis)")
print(f"  (OOS 2020–2024 = index construction, not model selection)\n")

test_label = f"test_{TEST_START_YEAR}_{'2018' if not IS_BUCKET else TEST_END_YEAR}"
metrics_test = fn_eval(preds_test_eval["y"], preds_test_eval["p_csi"], test_label) \
               if len(preds_test_eval) > 0 else None

metrics_oos = fn_eval(preds_oos_eval["y"], preds_oos_eval["p_csi"], "oos_2020_2024") \
              if len(preds_oos_eval) > 0 else None

for m in [metrics_test, metrics_oos]:
    if m:
        tag = " ← MODEL SELECTION" if "test" in m["set"] else " ← index use only"
        print(f"  {m['set']:<28}  AP={m['avg_precision']:.4f}  "
              f"AUC={m['auc_roc']:.4f}  R@FPR3={m['recall_fpr3']:.4f}  "
              f"R@FPR5={m['recall_fpr5']:.4f}{tag}")

# ==============================================================================
# 16. Stage 2 — Expanding window CV with per-fold metric averaging
#
#   Metrics averaged across folds — NOT pooled.
#   Rationale: pooling mixes predictions from models trained on very
#   different data volumes and conflates data-size effects with model quality.
#   Per-fold averaging gives equal weight to each time regime.
#
#   CSI models: last training year of each fold excluded from fold fitting
#   (its y_next = label from the validation period).
# ==============================================================================

print(f"\n[09C] ══ STAGE 2 — CV [{MODEL}] ══  "
      f"preset={CV_PRESET}  time={CV_TIME_LIMIT}s/fold\n")

fold_ap_list  = []
fold_auc_list = []
fold_r3_list  = []
cv_preds_list = []

for fid, tend, vst, vend in FOLD_BOUNDARIES:

    print(f"[09C] Fold {fid}: train ≤{tend} → val {vst}–{vend}")

    fold_tr  = train_qt[train_qt["year"] <= tend].copy()
    fold_val = train_qt[(train_qt["year"] >= vst) &
                        (train_qt["year"] <= vend)].copy()

    ## CSI: exclude last training year from fitting (boundary year)
    fold_fit = fold_tr[fold_tr["year"] < tend].copy() if not IS_BUCKET else fold_tr

    n_pos = int(fold_fit[LABEL_COL].sum())
    n_neg = int((fold_fit[LABEL_COL] == 0).sum())
    print(f"  Fit rows: {len(fold_fit):,}  pos={n_pos}  neg={n_neg}")
    print(f"  Val rows: {len(fold_val):,}")

    if n_pos < 10:
        print(f"  SKIPPED — fewer than 10 positives in fold {fid}")
        continue

    fold_fit_kw = dict(
        train_data       = TabularDataset(fold_fit[ag_cols].reset_index(drop=True)),
        tuning_data      = TabularDataset(fold_val[ag_cols].reset_index(drop=True)),
        presets          = CV_PRESET,
        time_limit       = CV_TIME_LIMIT,
        num_bag_folds    = 0,
        num_stack_levels = 0,
    )
    if IS_BUCKET:
        fold_fit_kw["feature_metadata"] = {"year_cat": "category"}

    fold_pred = TabularPredictor(
        label        = LABEL_COL,
        problem_type = "binary",
        eval_metric  = EVAL_METRIC,
        path         = str(DIR_MODELS_RUN / f"ag_cv_fold{fid}"),
        verbosity    = 1,
    ).fit(**fold_fit_kw)

    fold_proba = fold_pred.predict_proba(
        TabularDataset(fold_val[ag_cols].reset_index(drop=True)),
        as_multiclass=False
    )
    fold_y = fold_val[LABEL_COL].values

    cv_df = pd.DataFrame({
        "fold_id": fid,
        "permno" : fold_val["permno"].values,
        "year"   : fold_val["year"].values,
        "y"      : fold_y,
        "p_csi"  : fold_proba.values,
    })
    cv_preds_list.append(cv_df)

    if len(np.unique(fold_y)) < 2:
        print(f"  Fold {fid}: only one class present — metric skipped")
        continue

    ## Per-fold metrics — computed independently, then averaged across folds
    f_ap  = average_precision_score(fold_y, cv_df["p_csi"])
    f_auc = roc_auc_score(fold_y, cv_df["p_csi"])
    f_r3  = fn_recall_at_fpr(fold_y, cv_df["p_csi"].values, 0.03)

    fold_ap_list.append(f_ap)
    fold_auc_list.append(f_auc)
    fold_r3_list.append(f_r3)

    print(f"  Fold {fid}: AP={f_ap:.4f}  AUC={f_auc:.4f}  "
          f"R@FPR3={f_r3:.4f}  n={len(fold_y):,}  prev={fold_y.mean():.4f}")

## Average across folds
if fold_ap_list:
    cv_ap   = float(np.mean(fold_ap_list))
    cv_auc  = float(np.mean(fold_auc_list))
    cv_r3   = float(np.mean(fold_r3_list))
    print(f"\n[09C] CV averaged across {len(fold_ap_list)} folds (not pooled):")
    print(f"  AP      : {cv_ap:.4f}  per-fold: {[round(x,4) for x in fold_ap_list]}")
    print(f"  AUC     : {cv_auc:.4f}  per-fold: {[round(x,4) for x in fold_auc_list]}")
    print(f"  R@FPR3  : {cv_r3:.4f}  per-fold: {[round(x,4) for x in fold_r3_list]}")
    if cv_preds_list:
        pd.concat(cv_preds_list, ignore_index=True).to_parquet(
            DIR_TABLES_RUN / "ag_cv_results.parquet", index=False)
else:
    cv_ap = cv_auc = cv_r3 = None
    print("[09C] No CV folds produced valid metrics.")

# ==============================================================================
# 17. Save evaluation summary JSON
# ==============================================================================

eval_summary = {
    "model"              : MODEL,
    "description"        : cfg["description"],
    "preset"             : PRESET,
    "time_limit_s"       : TIME_LIMIT,
    "label_col"          : LABEL_COL,
    "is_bucket"          : IS_BUCKET,
    "label_shift"        : ("none — labels forward-aligned in 05B/05C"
                             if IS_BUCKET else "y(t) → y_next = y(t+1)"),
    "eval_period"        : f"{TEST_START_YEAR}–{'2018' if not IS_BUCKET else TEST_END_YEAR}",
    "oos_period"         : f"{OOS_START_YEAR}–2024",
    "oos_role"           : ("index construction only — no OOS labels for bucket"
                             if IS_BUCKET
                             else "index construction; OOS metrics stored but not used for model selection"),
    "boundary_exclusion" : ("N/A" if IS_BUCKET
                            else f"year {TRAIN_END_YEAR} (train) and "
                                 f"year {TEST_END_YEAR} (test) excluded from metrics"),
    "cv_method"          : "per-fold averaging (not pooled)",
    "n_features"         : len(feature_cols_model),
    "year_cat"           : IS_BUCKET,
    "cv_ap"              : round(cv_ap,   4) if cv_ap   else None,
    "cv_auc"             : round(cv_auc,  4) if cv_auc  else None,
    "cv_r3"              : round(cv_r3,   4) if cv_r3   else None,
    "cv_n_folds"         : len(fold_ap_list),
    "cv_per_fold_ap"     : [round(x, 4) for x in fold_ap_list],
    "cv_per_fold_auc"    : [round(x, 4) for x in fold_auc_list],
    "test"               : metrics_test,
    "oos"                : metrics_oos,
}

with open(DIR_TABLES_RUN / "ag_eval_summary.json", "w") as fh:
    json.dump(eval_summary, fh, indent=2)
print(f"\n[09C] Summary: {DIR_TABLES_RUN / 'ag_eval_summary.json'}")

# ==============================================================================
# 18. Results table
# ==============================================================================

print(f"\n[09C] ══ RESULTS [{MODEL}] ══\n")

hdr_csi = f"  {'Model':<30} | {'CV AP':<8} | {'Test AP':<8} | {'AUC':<8} | {'R@FPR3'}"
hdr_bkt = f"  {'Model':<30} | {'CV AP':<8} | {'Test AP':<8} | {'Test AUC':<10}"
sep     = f"  {'-'*30} | {'-'*8} | {'-'*8} | {'-'*10}"

print(hdr_bkt if IS_BUCKET else hdr_csi)
print(sep)

if metrics_test:
    cv_s = f"{cv_ap:.4f}" if cv_ap else "—"
    if IS_BUCKET:
        print(f"  {MODEL:<30} | {cv_s:<8} | "
              f"{metrics_test['avg_precision']:<8} | "
              f"{metrics_test['auc_roc']:<10}")
    else:
        print(f"  {MODEL:<30} | {cv_s:<8} | "
              f"{metrics_test['avg_precision']:<8} | "
              f"{metrics_test['auc_roc']:<8} | "
              f"{metrics_test['recall_fpr3']}")

## Reference baselines
print(sep)
if IS_BUCKET:
    print(f"  {'Logistic baseline':<30} | {'—':<8} | {'—':<8} | {'0.7440':<10}")
else:
    print(f"  {'M1 prior run':<30} | {'—':<8} | {'0.6546':<8} | {'0.9337':<8} | 0.4936")
    print(f"  {'M3 prior run':<30} | {'—':<8} | {'0.7576':<8} | {'0.9601':<8} | 0.6397")

print(f"\n[09C] DONE [{MODEL}]")
print(f"  Tables : {DIR_TABLES_RUN}")
print(f"  Model  : {DIR_MODELS_RUN / 'ag_predictor'}")
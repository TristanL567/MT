"""
09C_AutoGluon.py  (v6 — M1/M2/M3/M4/B1/structural)
====================================================
AutoGluon AutoML for CSI prediction + Bucket/Structural classification.

CHANGES FROM v5:
  1. Split boundary leakage fix:
       Train rows with year == TRAIN_END_YEAR (2015) are excluded from
       AUC/AP metric computation (their y_next = y(2016), a test-period label).
       Test rows with year == TEST_END_YEAR (2019) are similarly excluded.
       eval_split column in split_labels_oot.parquet drives this — rows
       flagged "train_boundary" or "test_boundary" are excluded from metrics
       but STILL receive predictions for index construction in 11_Results.R.

  2. year_cat fix (B1/structural):
       year_cat column is now explicitly created and passed to AutoGluon
       as a categorical feature, enabling period fixed-effect absorption.
       Previously the column was referenced but never created.

  3. Per-fold CV metric averaging (not pooling):
       CV AUC and AP are computed per fold independently, then averaged.
       Pooling all fold predictions into one DataFrame and computing a single
       metric is wrong — it gives undue weight to larger folds and conflates
       time-regime effects with model quality.

  4. 08B encode() fix (reference — change is in 08B_Autoencoder.py):
       Reconstruction error in encode() should use z_mu (deterministic),
       not z_samp (stochastic). Ensures reproducible vae_recon_error.
       This script is unaffected but documents the dependency.

MODEL SELECTION:
    Set MODEL in Section 2.
    "raw"         → M3  full engineered features
    "fund"        → M1  fundamentals only
    "latent_fund" → M2  VAE latent on fund features
    "latent_raw"  → M4  VAE latent on full raw features
    "bucket"      → B1  5yr forward CAGR bucket classifier
    "structural"  → B1-structural  combined CSI + 5yr CAGR label

LABEL SHIFT:
    M1/M2/M3/M4: y_next = y(t+1) — applied in section 6A.
    B1/structural: no shift — labels already forward-aligned in 05B/05C.

EXPANDING WINDOW CV:
    Stage 1: Train on years [train_start .. 2010], holdout = 2011–2014.
    Stage 2: 3-fold expanding window CV within the training set.
             Fold boundaries defined in FOLD_BOUNDARIES.
             Last training year of each fold excluded from fold metrics
             (same boundary logic as Stage 1).
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

DATA_ROOT    = Path(r"C:\Users\Tristan Leiter\Documents\MT")
DIR_DATA     = DATA_ROOT / "02_Data"
DIR_FEATURES = DIR_DATA  / "Features"
DIR_LABELS   = DIR_DATA  / "Labels"
DIR_OUTPUT   = DATA_ROOT / "03_Output"
DIR_MODELS   = DIR_OUTPUT / "Models" / "AutoGluon"
DIR_TABLES   = DIR_OUTPUT / "Tables"

PATH_SPLIT_LABELS      = DIR_FEATURES / "split_labels_oot.parquet"
PATH_LABELS_BUCKET     = DIR_LABELS / "labels_bucket.rds"
PATH_LABELS_STRUCTURAL = DIR_LABELS / "labels_structural.rds"

assert PATH_SPLIT_LABELS.exists(), f"Not found: {PATH_SPLIT_LABELS}"

# ==============================================================================
# 2. ── MODEL SELECTION ────────────────────────────────────────────────────────
# ==============================================================================

MODEL = "fund"    # <- CHANGE THIS

# ==============================================================================
# 3. Model configuration
# ==============================================================================

SEED          = 123
TIME_LIMIT    = 3600
PRESET        = "good_quality"
CV_TIME_LIMIT = 900
CV_PRESET     = "medium_quality"
EVAL_METRIC   = "average_precision"

LABEL_COL_CSI    = "y_next"
LABEL_COL_BUCKET = "y_loser"

MODEL_CONFIG = {
    "raw": {
        "path"       : DIR_FEATURES / "features_raw.rds",
        "loader"     : "rds",
        "description": "M3 — Full engineered features (~463)",
        "prereq"     : "Run 06B_Feature_Eng.R first.",
        "label_col"  : LABEL_COL_CSI,
        "is_bucket"  : False,
    },
    "fund": {
        "path"       : DIR_FEATURES / "features_fund.rds",
        "loader"     : "rds",
        "description": "M1 — Fundamentals only (no price features)",
        "prereq"     : "Run 06B_Feature_Eng.R first.",
        "label_col"  : LABEL_COL_CSI,
        "is_bucket"  : False,
    },
    "latent_fund": {
        "path"       : DIR_FEATURES / "features_latent_fund.parquet",
        "loader"     : "parquet",
        "description": "M2 — VAE latent on fundamentals",
        "prereq"     : "Run 08B_Autoencoder.py with VAE_INPUT='fund' first.",
        "label_col"  : LABEL_COL_CSI,
        "is_bucket"  : False,
    },
    "latent_raw": {
        "path"       : DIR_FEATURES / "features_latent_raw.parquet",
        "loader"     : "parquet",
        "description": "M4 — VAE latent on full raw features",
        "prereq"     : "Run 08B_Autoencoder.py with VAE_INPUT='raw' first.",
        "label_col"  : LABEL_COL_CSI,
        "is_bucket"  : False,
    },
    "bucket": {
        "path"       : DIR_FEATURES / "features_fund.rds",
        "loader"     : "rds",
        "description": "B1 — 5yr forward CAGR bucket classifier",
        "prereq"     : "Run 05B_Bucket_Labels.R first.",
        "label_col"  : LABEL_COL_BUCKET,
        "is_bucket"  : True,
    },
    "structural": {
        "path"       : DIR_FEATURES / "features_fund.rds",
        "loader"     : "rds",
        "description": "B1-structural — Combined CSI + 5yr CAGR label",
        "prereq"     : "Run 05C_Structural_Labels.R first.",
        "label_col"  : "y_structural",
        "is_bucket"  : True,
    },
}

assert MODEL in MODEL_CONFIG, \
    f"Unknown MODEL '{MODEL}'. Choose from: {list(MODEL_CONFIG.keys())}"

cfg       = MODEL_CONFIG[MODEL]
LABEL_COL = cfg["label_col"]
IS_BUCKET = cfg["is_bucket"]

assert cfg["path"].exists(), \
    f"Feature file not found: {cfg['path']}\n{cfg['prereq']}"

if IS_BUCKET:
    lbl_path = PATH_LABELS_STRUCTURAL if MODEL == "structural" \
               else PATH_LABELS_BUCKET
    assert lbl_path.exists(), \
        f"Labels not found: {lbl_path}\n{cfg['prereq']}"

DIR_MODELS_RUN = DIR_MODELS / f"ag_{MODEL}"
DIR_TABLES_RUN = DIR_TABLES / f"ag_{MODEL}"
DIR_MODELS_RUN.mkdir(parents=True, exist_ok=True)
DIR_TABLES_RUN.mkdir(parents=True, exist_ok=True)

print(f"[09C] ══════════════════════════════════════")
print(f"  MODEL        : {MODEL.upper()}")
print(f"  Description  : {cfg['description']}")
print(f"  Label column : {LABEL_COL}")
print(f"  Output dir   : {DIR_TABLES_RUN}")
print(f"[09C] ══════════════════════════════════════\n")

# ==============================================================================
# 4. ID column definitions
# ==============================================================================

ID_COLS = [
    "permno", "year", "y", "censored", "param_id",
    "gvkey", "datadate", "lifetime_years",
    "fiscal_year_end_month", "split", "eval_split", "vae_split", "split_oot",
    "y_loser", "y_structural", "fwd_cagr", "n_months", "bucket",
    "year_cat",
]

LATENT_FEATURE_NAMES = [f"z{i}" for i in range(1, 25)] + ["vae_recon_error"]

# ==============================================================================
# 5. Load feature data
# ==============================================================================

print(f"[09C] Loading {cfg['path'].name} ({cfg['loader']})...")

if cfg["loader"] == "rds":
    result         = pyreadr.read_r(str(cfg["path"]))
    features_input = result[None]
elif cfg["loader"] == "parquet":
    features_input = pd.read_parquet(cfg["path"])
    if "split" in features_input.columns:
        features_input = features_input.rename(columns={"split": "vae_split"})

print(f"  Shape: {features_input.shape[0]:,} rows × {features_input.shape[1]} cols")

# ==============================================================================
# 6A. CSI models — merge split labels, apply label shift, mark boundary rows
# ==============================================================================

if not IS_BUCKET:

    print("[09C] Loading split labels (OOT)...")
    split_labels = pd.read_parquet(PATH_SPLIT_LABELS)
    ## split_labels now contains both 'split' and 'eval_split' columns
    ## (from revised 08_Split.R)
    print(f"  Split distribution:\n"
          f"{split_labels['split'].value_counts().to_string()}")
    if "eval_split" in split_labels.columns:
        print(f"  Eval-split distribution:\n"
              f"{split_labels['eval_split'].value_counts().to_string()}")
    else:
        ## Backward compatibility: if running against old split file,
        ## construct eval_split by excluding known boundary years
        print("  WARNING: eval_split column not found in parquet. "
              "Re-run 08_Split.R to generate it. "
              "Falling back to manual boundary exclusion (2015, 2019).")
        split_labels["eval_split"] = split_labels.apply(
            lambda r: "train_boundary" if (r["split"] == "train" and r["year"] == 2015)
                 else "test_boundary"  if (r["split"] == "test"  and r["year"] == 2019)
                 else r["split"],
            axis=1
        )

    df = features_input.merge(split_labels, on=["permno", "year"], how="left")
    df = df[df["split"].notna()].reset_index(drop=True)
    df = df.sort_values(["permno", "year"]).reset_index(drop=True)

    ## Label shift: y(t) → y_next = y(t+1), within each firm
    df["y_next"] = df.groupby("permno")["y"].shift(-1)

    ## All rows — used for prediction generation (index construction)
    df_all       = df[df["y_next"].notna()].copy()
    df_all["y_next"] = df_all["y_next"].astype(int)

    ## Eval-safe rows — used for AUC/AP metric computation
    ## Excludes boundary years whose y_next label comes from the next period
    BOUNDARY_FLAGS  = {"train_boundary", "test_boundary"}
    df_labelled = df_all[~df_all["eval_split"].isin(BOUNDARY_FLAGS)].copy()

    print(f"\n[09C] After label shift:")
    print(f"  All rows (for predictions) : {len(df_all):,}")
    print(f"  Eval-safe rows (for metrics): {len(df_labelled):,}")
    print(f"  Boundary rows excluded      : "
          f"{len(df_all) - len(df_labelled):,}")
    print(f"  y_next prevalence (eval)   : "
          f"{df_labelled['y_next'].mean():.3f}")

    ## Full splits for prediction generation (includes boundary years)
    train_df_all = df_all[df_all["split"] == "train"].copy()
    test_df_all  = df_all[df_all["split"] == "test"].copy()
    oos_df_all   = df_all[df_all["split"] == "oos"].copy()

    ## Eval-safe splits for metric computation
    train_df = df_labelled[df_labelled["split"] == "train"].copy()
    test_df  = df_labelled[df_labelled["split"] == "test"].copy()
    oos_df   = df_labelled[df_labelled["split"] == "oos"].copy()

    print(f"\n[09C] Split sizes:")
    print(f"  Train (all/eval) : {len(train_df_all):,} / "
          f"{len(train_df):,}  (prev: {train_df['y_next'].mean():.3f})")
    print(f"  Test  (all/eval) : {len(test_df_all):,} / "
          f"{len(test_df):,}  (prev: {test_df['y_next'].mean():.3f})")
    print(f"  OOS   (all/eval) : {len(oos_df_all):,} / "
          f"{len(oos_df):,}  (prev: {oos_df['y_next'].mean():.3f})")

# ==============================================================================
# 6B. B1/structural models — no label shift, no boundary issue
# ==============================================================================

else:

    lbl_path = PATH_LABELS_STRUCTURAL if MODEL == "structural" \
               else PATH_LABELS_BUCKET
    lbl_name = ("structural labels from 05C" if MODEL == "structural"
                else "bucket labels from 05B")
    print(f"[09C] Loading {lbl_name}...")
    labels_r    = pyreadr.read_r(str(lbl_path))
    labels_tbl  = labels_r[None]

    raw_label_col = ("y_structural" if MODEL == "structural" else "y_loser")

    labels_tbl = labels_tbl[labels_tbl[raw_label_col].notna()].copy()
    labels_tbl[raw_label_col] = labels_tbl[raw_label_col].astype(int)

    print(f"  Labels: {len(labels_tbl):,} usable | "
          f"{raw_label_col} prevalence: {labels_tbl[raw_label_col].mean():.3f}")
    print(f"  Years: {labels_tbl['year'].min()} – {labels_tbl['year'].max()}")

    df_labelled = features_input.merge(
        labels_tbl[["permno", "year", raw_label_col]],
        on=["permno", "year"],
        how="inner"
    ).copy()

    if raw_label_col != LABEL_COL:
        df_labelled = df_labelled.rename(columns={raw_label_col: LABEL_COL})

    ## ── Create year_cat for period fixed-effect absorption ─────────────────
    ## B1/structural labels are period-dependent (bull/bear cycles affect the
    ## 5yr forward CAGR distribution). year_cat lets AutoGluon learn a fixed
    ## effect per calendar year, so firm features predict the deviation from
    ## the cross-sectional mean rather than the calendar-time trend.
    df_labelled["year_cat"] = df_labelled["year"].astype(str).astype("category")

    ## Split — same years as M1 for comparability; OOS ends at 2019
    train_df = df_labelled[df_labelled["year"] <= 2015].copy()
    test_df  = df_labelled[(df_labelled["year"] >= 2016) &
                            (df_labelled["year"] <= 2019)].copy()
    oos_df   = pd.DataFrame()

    ## For B1, all/eval distinction does not apply (no label shift)
    train_df_all = train_df
    test_df_all  = test_df
    oos_df_all   = oos_df

    print(f"\n[09C] Split sizes (B1):")
    print(f"  Train (≤2015)    : {len(train_df):,}  "
          f"(prev: {train_df[LABEL_COL].mean():.3f})")
    print(f"  Test  (2016-2019): {len(test_df):,}  "
          f"(prev: {test_df[LABEL_COL].mean():.3f})")
    print(f"  OOS              : N/A — labels censored after 2019")

# ==============================================================================
# 7. Feature columns
# ==============================================================================

if MODEL in ("latent_fund", "latent_raw"):
    feature_cols = [c for c in LATENT_FEATURE_NAMES if c in train_df.columns]
    print(f"\n[09C] Latent feature columns: {len(feature_cols)}")

elif IS_BUCKET:
    feature_cols = [
        c for c in train_df.columns
        if c not in ID_COLS
        and c not in [LABEL_COL, "y_next"]
        and pd.api.types.is_numeric_dtype(train_df[c])
    ]
    print(f"\n[09C] Feature columns (B1): {len(feature_cols)} numeric "
          f"+ 1 categorical (year_cat)")

else:
    feature_cols = [
        c for c in train_df.columns
        if c not in ID_COLS
        and c not in [LABEL_COL, "y_next"]
        and pd.api.types.is_numeric_dtype(train_df[c])
        and c != "y"
    ]
    print(f"\n[09C] Feature columns: {len(feature_cols)}")

feature_cols_model = feature_cols

# ==============================================================================
# 8. Quantile transform — fitted on eval-safe train set
# ==============================================================================

print("\n[09C] Applying quantile transform (fitted on train only)...")

X_train = train_df[feature_cols].values.astype(np.float64)
X_test  = test_df[feature_cols].values.astype(np.float64)  \
          if len(test_df) > 0 else np.empty((0, len(feature_cols)))
X_oos   = oos_df[feature_cols].values.astype(np.float64)   \
          if len(oos_df) > 0 else np.empty((0, len(feature_cols)))

## Also prepare full-split arrays for prediction generation (CSI models)
if not IS_BUCKET:
    X_train_all = train_df_all[feature_cols].values.astype(np.float64)
    X_test_all  = test_df_all[feature_cols].values.astype(np.float64)
    X_oos_all   = oos_df_all[feature_cols].values.astype(np.float64)  \
                  if len(oos_df_all) > 0 else np.empty((0, len(feature_cols)))

for arr in ([X_train, X_test, X_oos] +
            ([] if IS_BUCKET else [X_train_all, X_test_all, X_oos_all])):
    arr[np.isinf(arr)] = np.nan

## Winsorise using train percentiles only
lo = np.nanpercentile(X_train, 0.1, axis=0)
hi = np.nanpercentile(X_train, 99.9, axis=0)

def clip_arr(arr, lo, hi):
    return np.clip(arr, lo, hi)

X_train = clip_arr(X_train, lo, hi)
X_test  = clip_arr(X_test,  lo, hi) if len(X_test) > 0 else X_test
X_oos   = clip_arr(X_oos,   lo, hi) if len(X_oos)  > 0 else X_oos

if not IS_BUCKET:
    X_train_all = clip_arr(X_train_all, lo, hi)
    X_test_all  = clip_arr(X_test_all,  lo, hi) if len(X_test_all) > 0 else X_test_all
    X_oos_all   = clip_arr(X_oos_all,   lo, hi) if len(X_oos_all)  > 0 else X_oos_all

## Median imputation — train medians only
train_medians = np.nanmedian(X_train, axis=0)
train_medians = np.where(np.isnan(train_medians), 0.0, train_medians)

def impute_median(X, medians):
    X_out = X.copy()
    for j in range(X_out.shape[1]):
        mask = np.isnan(X_out[:, j])
        if mask.any():
            X_out[mask, j] = medians[j]
    return X_out

X_train_imp = impute_median(X_train, train_medians)
X_test_imp  = impute_median(X_test,  train_medians) if len(X_test) > 0 else X_test
X_oos_imp   = impute_median(X_oos,   train_medians) if len(X_oos)  > 0 else X_oos

if not IS_BUCKET:
    X_train_all_imp = impute_median(X_train_all, train_medians)
    X_test_all_imp  = impute_median(X_test_all,  train_medians) if len(X_test_all) > 0 else X_test_all
    X_oos_all_imp   = impute_median(X_oos_all,   train_medians) if len(X_oos_all)  > 0 else X_oos_all

assert not np.isnan(X_train_imp).any(), "NaN remains after imputation"

## Quantile transform
qt = QuantileTransformer(
    output_distribution="uniform",
    n_quantiles=min(1000, len(X_train_imp)),
    random_state=SEED
)
qt.fit(X_train_imp)

def qt_transform(X):
    return qt.transform(X) if len(X) > 0 else X

X_train_qt     = qt_transform(X_train_imp)
X_test_qt      = qt_transform(X_test_imp)
X_oos_qt       = qt_transform(X_oos_imp)

if not IS_BUCKET:
    X_train_all_qt = qt_transform(X_train_all_imp)
    X_test_all_qt  = qt_transform(X_test_all_imp)
    X_oos_all_qt   = qt_transform(X_oos_all_imp)

def rebuild_df(X_qt, source_df, feature_cols, label_col, add_year_cat=False):
    out = pd.DataFrame(X_qt, columns=feature_cols, index=source_df.index)
    out[label_col]  = source_df[label_col].values
    out["year"]     = source_df["year"].values
    out["permno"]   = source_df["permno"].values
    if add_year_cat and "year_cat" in source_df.columns:
        out["year_cat"] = source_df["year_cat"].values
    return out

add_yc = IS_BUCKET
train_qt     = rebuild_df(X_train_qt, train_df, feature_cols, LABEL_COL, add_yc)
test_qt      = rebuild_df(X_test_qt,  test_df,  feature_cols, LABEL_COL, add_yc) \
               if len(test_df) > 0 else pd.DataFrame()
oos_qt       = rebuild_df(X_oos_qt,   oos_df,   feature_cols, LABEL_COL, add_yc) \
               if len(oos_df) > 0 else pd.DataFrame()

if not IS_BUCKET:
    train_all_qt = rebuild_df(X_train_all_qt, train_df_all, feature_cols, LABEL_COL)
    test_all_qt  = rebuild_df(X_test_all_qt,  test_df_all,  feature_cols, LABEL_COL) \
                   if len(test_df_all) > 0 else pd.DataFrame()
    oos_all_qt   = rebuild_df(X_oos_all_qt,   oos_df_all,   feature_cols, LABEL_COL) \
                   if len(oos_df_all) > 0 else pd.DataFrame()

print(f"  QT applied. Train shape: {train_qt.shape}")
print(f"  Train mean (~0.5): {X_train_qt.mean():.4f}")
print(f"  Train std  (~0.29): {X_train_qt.std():.4f}")

# ==============================================================================
# 9. AutoGluon column set
# ==============================================================================

ag_train_cols = feature_cols_model + [LABEL_COL]
if IS_BUCKET:
    ag_train_cols = feature_cols_model + ["year_cat", LABEL_COL]

# ==============================================================================
# 10. Fold boundaries for CV
# ==============================================================================

## Stage 1 holdout: 2011–2014 (2015 excluded — boundary year for CSI models)
## For B1/structural: 2015 is fine (no label shift), keep 2011–2015
STAGE1_HOLDOUT_START = 2011
STAGE1_HOLDOUT_END   = 2014 if not IS_BUCKET else 2015

## CV folds: expanding window, fold 2 onward
## Each tuple: (fold_id, train_end, val_start, val_end)
## train_end is the last year of training — its y_next is from the val period
## so it is excluded from fold-level metric computation (CSI models only)
FOLD_BOUNDARIES = [
    (2, 2001, 2002, 2006),
    (3, 2006, 2007, 2010),
    (4, 2010, 2011, STAGE1_HOLDOUT_END),
]

print(f"\n[09C] Stage 1 holdout: {STAGE1_HOLDOUT_START}–{STAGE1_HOLDOUT_END}")
print(f"  (year {2015 if not IS_BUCKET else 2015} "
      f"{'excluded from metrics (boundary)' if not IS_BUCKET else 'included (no shift)'})")
print(f"\n[09C] CV fold structure:")
for fold_id, train_end, val_start, val_end in FOLD_BOUNDARIES:
    n_train = len(train_qt[train_qt["year"] <= train_end])
    n_val   = len(train_qt[(train_qt["year"] >= val_start) &
                            (train_qt["year"] <= val_end)])
    boundary_note = f" | boundary year={train_end} excluded from fold metrics" \
                    if not IS_BUCKET else ""
    print(f"  Fold {fold_id}: train [≤{train_end}] n={n_train:,} "
          f"| val [{val_start}–{val_end}] n={n_val:,}{boundary_note}")

# ==============================================================================
# 11. Stage 1 — AutoGluon training
# ==============================================================================

print(f"\n[09C] ══ STAGE 1: AutoGluon [{MODEL.upper()}] ══")
print(f"  Preset: {PRESET} | Time limit: {TIME_LIMIT}s\n")

holdout_mask = (
    (train_qt["year"] >= STAGE1_HOLDOUT_START) &
    (train_qt["year"] <= STAGE1_HOLDOUT_END)
)

ag_train_data = TabularDataset(
    train_qt[~holdout_mask][ag_train_cols].reset_index(drop=True))
ag_holdout    = TabularDataset(
    train_qt[holdout_mask][ag_train_cols].reset_index(drop=True))

## Test data for Stage 1 evaluation uses eval-safe rows
ag_test_data  = TabularDataset(
    test_qt[ag_train_cols].reset_index(drop=True)) \
    if len(test_qt) > 0 else None

print(f"  AG training rows : {len(ag_train_data):,}")
print(f"  AG holdout rows  : {len(ag_holdout):,}")
if ag_test_data is not None:
    print(f"  AG test rows     : {len(ag_test_data):,}  (eval-safe)")

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
    verbosity    = 2
).fit(**fit_kwargs)

print(f"\n[09C] Stage 1 complete.")

# ==============================================================================
# 12. Leaderboard and feature importance
# ==============================================================================

print(f"\n[09C] ── Leaderboard ──\n")
leaderboard = predictor.leaderboard(ag_holdout, silent=True)
print(leaderboard[["model", "score_val", "pred_time_val",
                    "fit_time"]].to_string(index=False))
leaderboard.to_csv(DIR_TABLES_RUN / "ag_leaderboard.csv", index=False)

print(f"\n[09C] Feature importance...")
try:
    feat_imp = predictor.feature_importance(ag_holdout, silent=True)
    feat_imp.to_csv(DIR_TABLES_RUN / "ag_feature_importance.csv")
    print(f"  Top 10:\n{feat_imp.head(10).to_string()}")
except Exception as e:
    print(f"  Feature importance failed: {e}")
    feat_imp = None

# ==============================================================================
# 13. Predictions
#
#   Two prediction outputs per period:
#   (a) Eval predictions — on eval-safe rows — for AUC/AP metrics.
#   (b) Full predictions — on all split rows including boundary years
#       — for index construction in 11_Results.R.
#
#   For B1/structural: no boundary issue, so eval == full.
# ==============================================================================

print(f"\n[09C] Generating predictions...")

def make_preds(predictor, qt_df, ag_cols, source_df, label_col):
    if len(qt_df) == 0:
        return pd.DataFrame()
    proba = predictor.predict_proba(
        TabularDataset(qt_df[ag_cols].reset_index(drop=True)),
        as_multiclass=False
    )
    return pd.DataFrame({
        "permno": source_df["permno"].values,
        "year"  : source_df["year"].values,
        "y"     : source_df[label_col].values,
        "p_csi" : proba.values
    })

## Eval-safe predictions (for metrics)
preds_test_eval = make_preds(predictor, test_qt, ag_train_cols,
                              test_df, LABEL_COL)
preds_oos_eval  = make_preds(predictor, oos_qt,  ag_train_cols,
                              oos_df,  LABEL_COL)

## Full predictions (for index construction — includes boundary years)
if IS_BUCKET:
    preds_test_full = preds_test_eval
    preds_oos_full  = preds_oos_eval
else:
    preds_test_full = make_preds(predictor, test_all_qt, ag_train_cols,
                                  test_df_all, LABEL_COL)
    preds_oos_full  = make_preds(predictor, oos_all_qt,  ag_train_cols,
                                  oos_df_all,  LABEL_COL)
    ## Also generate predictions for train boundary year (2015) for completeness
    train_boundary = train_df_all[train_df_all["year"] == 2015].copy() \
                     if not IS_BUCKET else pd.DataFrame()
    if len(train_boundary) > 0:
        X_bnd = train_boundary[feature_cols].values.astype(np.float64)
        X_bnd[np.isinf(X_bnd)] = np.nan
        X_bnd = clip_arr(X_bnd, lo, hi)
        X_bnd = impute_median(X_bnd, train_medians)
        X_bnd_qt = qt_transform(X_bnd)
        bnd_qt = rebuild_df(X_bnd_qt, train_boundary, feature_cols, LABEL_COL)
        preds_train_boundary = make_preds(
            predictor, bnd_qt, ag_train_cols, train_boundary, LABEL_COL)
    else:
        preds_train_boundary = pd.DataFrame()

## Save eval predictions (for metric reporting in thesis)
if len(preds_test_eval) > 0:
    preds_test_eval.to_parquet(
        DIR_TABLES_RUN / "ag_preds_test_eval.parquet", index=False)
    print(f"  ag_preds_test_eval.parquet  : {len(preds_test_eval):,} rows "
          f"(eval-safe, for metrics)")

## Save full predictions (for index construction)
if len(preds_test_full) > 0:
    preds_test_full.to_parquet(
        DIR_TABLES_RUN / "ag_preds_test.parquet", index=False)
    print(f"  ag_preds_test.parquet       : {len(preds_test_full):,} rows "
          f"(full, for index construction)")

if len(preds_oos_eval) > 0:
    preds_oos_eval.to_parquet(
        DIR_TABLES_RUN / "ag_preds_oos_eval.parquet", index=False)
    print(f"  ag_preds_oos_eval.parquet   : {len(preds_oos_eval):,} rows")

if len(preds_oos_full) > 0:
    preds_oos_full.to_parquet(
        DIR_TABLES_RUN / "ag_preds_oos.parquet", index=False)
    print(f"  ag_preds_oos.parquet        : {len(preds_oos_full):,} rows "
          f"(full, for index construction)")

if not IS_BUCKET and len(preds_train_boundary) > 0:
    preds_train_boundary.to_parquet(
        DIR_TABLES_RUN / "ag_preds_train_boundary.parquet", index=False)
    print(f"  ag_preds_train_boundary.parquet : {len(preds_train_boundary):,} "
          f"rows (year=2015, for index construction)")

# ==============================================================================
# 14. Evaluation helpers
# ==============================================================================

def fn_recall_at_fpr(y_true, y_pred, fpr_target):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    eligible = np.where(fpr <= fpr_target)[0]
    return 0.0 if len(eligible) == 0 else float(tpr[eligible].max())

def fn_eval_metrics(y_true, y_pred, set_name):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    valid  = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt, yp = y_true[valid], y_pred[valid]
    if len(np.unique(yt)) < 2:
        return None
    return {
        "set"          : set_name,
        "n_obs"        : int(len(yt)),
        "n_pos"        : int(yt.sum()),
        "prevalence"   : round(float(yt.mean()), 4),
        "auc_roc"      : round(float(roc_auc_score(yt, yp)), 4),
        "avg_precision": round(float(average_precision_score(yt, yp)), 4),
        "recall_fpr1"  : round(fn_recall_at_fpr(yt, yp, 0.01), 4),
        "recall_fpr3"  : round(fn_recall_at_fpr(yt, yp, 0.03), 4),
        "recall_fpr5"  : round(fn_recall_at_fpr(yt, yp, 0.05), 4),
        "recall_fpr10" : round(fn_recall_at_fpr(yt, yp, 0.10), 4),
        "brier"        : round(float(np.mean((yp - yt)**2)), 4),
    }

# ==============================================================================
# 15. Stage 1 evaluation — on eval-safe rows
# ==============================================================================

print(f"\n[09C] ── Stage 1 Evaluation [{MODEL.upper()}] (eval-safe rows) ──\n")

metrics_test = fn_eval_metrics(
    preds_test_eval["y"], preds_test_eval["p_csi"],
    f"test_{2016}_{2018 if not IS_BUCKET else 2019}"
) if len(preds_test_eval) > 0 else None

metrics_oos = fn_eval_metrics(
    preds_oos_eval["y"], preds_oos_eval["p_csi"],
    "oos_2020_2024"
) if len(preds_oos_eval) > 0 else None

for m in [metrics_test, metrics_oos]:
    if m:
        print(f"  {m['set']:<25} | AP={m['avg_precision']:.4f} | "
              f"AUC={m['auc_roc']:.4f} | R@FPR3={m['recall_fpr3']:.4f} | "
              f"R@FPR5={m['recall_fpr5']:.4f}")

# ==============================================================================
# 16. Stage 2 — Expanding window CV with per-fold metric averaging
#
#   CV metrics are computed per fold independently, then averaged.
#   Rationale: pooling predictions from different models (each fold trains
#   a different-sized model) conflates data-volume effects with model quality
#   and gives undue weight to larger folds. Per-fold averaging gives equal
#   weight to each time regime.
#
#   Boundary exclusion: for CSI models, the last training year of each fold
#   carries y_next from the validation period — excluded from fold metrics.
# ==============================================================================

print(f"\n[09C] ══ STAGE 2: Expanding window CV [{MODEL.upper()}] ══\n")

cv_preds_list     = []   ## for saving to parquet (diagnostic use only)
fold_auc_list     = []
fold_ap_list      = []
fold_r3_list      = []

for fold_id, train_end, val_start, val_end in FOLD_BOUNDARIES:

    print(f"\n[09C] Fold {fold_id}: train [≤{train_end}] → val [{val_start}–{val_end}]")

    fold_train_mask = train_qt["year"] <= train_end
    fold_val_mask   = (
        (train_qt["year"] >= val_start) &
        (train_qt["year"] <= val_end)
    )

    fold_train_qt = train_qt[fold_train_mask].copy()
    fold_val_qt   = train_qt[fold_val_mask].copy()

    ## For CSI models: exclude last training year from fold_train for model
    ## fitting (its label comes from val period after shift), and exclude it
    ## from fold metrics as a boundary year
    if not IS_BUCKET:
        ## Training: omit the boundary year
        fold_train_for_fit = fold_train_qt[
            fold_train_qt["year"] < train_end].copy()
        ## Metrics: use full val set (val labels are clean)
        fold_val_for_metrics = fold_val_qt.copy()
    else:
        fold_train_for_fit   = fold_train_qt
        fold_val_for_metrics = fold_val_qt

    fold_train_ag = TabularDataset(
        fold_train_for_fit[ag_train_cols].reset_index(drop=True))
    fold_val_ag   = TabularDataset(
        fold_val_qt[ag_train_cols].reset_index(drop=True))

    n_pos = int(fold_train_ag[LABEL_COL].sum())
    n_neg = int((fold_train_ag[LABEL_COL] == 0).sum())
    print(f"  Train (for fit): {len(fold_train_ag):,} | "
          f"pos={n_pos} neg={n_neg}")
    print(f"  Val  : {len(fold_val_ag):,}")

    if n_pos < 10:
        print(f"  Fold {fold_id} SKIPPED — fewer than 10 positives.")
        continue

    fold_fit_kwargs = dict(
        train_data       = fold_train_ag,
        tuning_data      = fold_val_ag,
        presets          = CV_PRESET,
        time_limit       = CV_TIME_LIMIT,
        num_bag_folds    = 0,
        num_stack_levels = 0,
    )
    if IS_BUCKET:
        fold_fit_kwargs["feature_metadata"] = {"year_cat": "category"}

    fold_predictor = TabularPredictor(
        label        = LABEL_COL,
        problem_type = "binary",
        eval_metric  = EVAL_METRIC,
        path         = str(DIR_MODELS_RUN / f"ag_cv_fold{fold_id}"),
        verbosity    = 1
    ).fit(**fold_fit_kwargs)

    ## ── Per-fold metrics (not pooled) ────────────────────────────────────
    ## Compute AUC and AP for this fold independently.
    ## Do NOT concatenate and compute pooled metric — see rationale in header.
    fold_proba = fold_predictor.predict_proba(fold_val_ag, as_multiclass=False)
    fold_y     = fold_val_for_metrics[LABEL_COL].values

    fold_cv_df = pd.DataFrame({
        "fold_id": fold_id,
        "permno" : fold_val_qt["permno"].values,
        "year"   : fold_val_qt["year"].values,
        "y"      : fold_y,
        "p_csi"  : fold_proba.values
    })

    ## Compute per-fold metrics on clean validation rows only
    if len(np.unique(fold_y)) < 2:
        print(f"  Fold {fold_id}: only one class in validation — skipping metric")
        cv_preds_list.append(fold_cv_df)
        continue

    fold_ap  = average_precision_score(fold_y, fold_cv_df["p_csi"])
    fold_auc = roc_auc_score(fold_y, fold_cv_df["p_csi"])
    fold_r3  = fn_recall_at_fpr(fold_y, fold_cv_df["p_csi"].values, 0.03)

    fold_ap_list.append(fold_ap)
    fold_auc_list.append(fold_auc)
    fold_r3_list.append(fold_r3)
    cv_preds_list.append(fold_cv_df)

    print(f"  Fold {fold_id}: AP={fold_ap:.4f} | AUC={fold_auc:.4f} | "
          f"R@FPR3={fold_r3:.4f} | n={len(fold_y):,} | "
          f"prev={fold_y.mean():.3f}")

## ── Average per-fold metrics ────────────────────────────────────────────────
if fold_ap_list:
    cv_ap_mean  = float(np.mean(fold_ap_list))
    cv_auc_mean = float(np.mean(fold_auc_list))
    cv_r3_mean  = float(np.mean(fold_r3_list))

    print(f"\n[09C] CV summary [{MODEL.upper()}] — AVERAGED across "
          f"{len(fold_ap_list)} folds (not pooled):")
    print(f"  CV AP     : {cv_ap_mean:.4f}  "
          f"(per-fold: {[round(x,4) for x in fold_ap_list]})")
    print(f"  CV AUC    : {cv_auc_mean:.4f}  "
          f"(per-fold: {[round(x,4) for x in fold_auc_list]})")
    print(f"  CV R@FPR3 : {cv_r3_mean:.4f}  "
          f"(per-fold: {[round(x,4) for x in fold_r3_list]})")

    ## Save concatenated fold predictions for diagnostic use
    if cv_preds_list:
        cv_preds_all = pd.concat(cv_preds_list, ignore_index=True)
        cv_preds_all.to_parquet(
            DIR_TABLES_RUN / "ag_cv_results.parquet", index=False)
else:
    cv_ap_mean = cv_auc_mean = cv_r3_mean = None
    print(f"[09C] No CV folds completed.")

# ==============================================================================
# 17. Save evaluation summary
# ==============================================================================

eval_summary = {
    "model"             : f"autogluon_{MODEL}",
    "model_label"       : cfg["description"],
    "preset"            : PRESET,
    "time_limit_s"      : TIME_LIMIT,
    "cv_time_limit_s"   : CV_TIME_LIMIT,
    "label_col"         : LABEL_COL,
    "is_bucket"         : IS_BUCKET,
    "label_shift"       : "none — labels forward-aligned in 05B/05C" if IS_BUCKET
                          else "y(t+1) — shift applied in 6A",
    "boundary_exclusion": "N/A" if IS_BUCKET
                          else f"train year {2015} and test year {2019} "
                               f"excluded from eval metrics; predictions "
                               f"still generated for index construction",
    "cv_method"         : "per-fold averaging (not pooled)",
    "eval_metric"       : EVAL_METRIC,
    "n_features"        : len(feature_cols_model),
    "year_cat_included" : IS_BUCKET,
    "cv_avg_precision"  : round(cv_ap_mean,  4) if cv_ap_mean  else None,
    "cv_auc_roc"        : round(cv_auc_mean, 4) if cv_auc_mean else None,
    "cv_recall_fpr3"    : round(cv_r3_mean,  4) if cv_r3_mean  else None,
    "cv_n_folds_used"   : len(fold_ap_list),
    "cv_per_fold_ap"    : [round(x, 4) for x in fold_ap_list],
    "cv_per_fold_auc"   : [round(x, 4) for x in fold_auc_list],
    "test"              : metrics_test,
    "oos"               : metrics_oos,
    "baselines"         : {
        "logistic_auc"  : 0.7440 if IS_BUCKET else None,
        "paper_r_fpr3"  : None   if IS_BUCKET else 0.61,
        "ag_m3_cv_ap"   : None   if IS_BUCKET else 0.6656,
        "ag_m1_test_ap" : None   if IS_BUCKET else 0.6546,
    }
}

with open(DIR_TABLES_RUN / "ag_eval_summary.json", "w") as f:
    json.dump(eval_summary, f, indent=2)
print(f"\n[09C] Summary saved: {DIR_TABLES_RUN / 'ag_eval_summary.json'}")

# ==============================================================================
# 18. Final results table
# ==============================================================================

print(f"\n[09C] ══ RESULTS: [{MODEL.upper()}] ══\n")

if IS_BUCKET:
    print(f"  {'Model':<30} | {'CV AP (avg)':<12} | {'Test AP':<8} | {'AUC':<8}")
    print(f"  {'-'*30} | {'------------':<12} | {'--------':<8} | {'--------'}")
    if metrics_test:
        cv_str = f"{cv_ap_mean:.4f}" if cv_ap_mean else "—"
        print(f"  {f'B1-{MODEL}':<30} | {cv_str:<12} | "
              f"{metrics_test['avg_precision']:<8} | {metrics_test['auc_roc']:<8}")
    print(f"  {'Logistic baseline (OOS)':<30} | {'—':<12} | "
          f"{'—':<8} | {'0.7440':<8}")
else:
    print(f"  {'Model':<25} | {'CV AP (avg)':<12} | {'Test AP':<8} | "
          f"{'AUC':<8} | {'R@FPR3':<8}")
    print(f"  {'-'*25} | {'------------':<12} | {'--------':<8} | "
          f"{'--------':<8} | {'--------'}")
    if metrics_test:
        cv_str = f"{cv_ap_mean:.4f}" if cv_ap_mean else "—"
        print(f"  {f'AG {MODEL}':<25} | {cv_str:<12} | "
              f"{metrics_test['avg_precision']:<8} | "
              f"{metrics_test['auc_roc']:<8} | "
              f"{metrics_test['recall_fpr3']:<8}")
    print(f"  {'AG M3 (raw, prior run)':<25} | {'—':<12} | "
          f"{'0.7576':<8} | {'0.9601':<8} | {'0.6397':<8}")
    print(f"  {'AG M1 (fund, prior run)':<25} | {'—':<12} | "
          f"{'0.6546':<8} | {'0.9337':<8} | {'0.4936':<8}")

print(f"\n[09C] DONE [{MODEL.upper()}]")
print(f"  Tables : {DIR_TABLES_RUN}")
print(f"  Model  : {DIR_MODELS_RUN / 'ag_predictor'}")
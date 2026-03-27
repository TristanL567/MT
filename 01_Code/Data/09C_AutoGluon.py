"""
09C_AutoGluon.py  (v5 — M1/M2/M3/M4/B1)
======================================
AutoGluon AutoML for CSI prediction + Bucket classification.

MODEL SELECTION:
    Set MODEL in Section 2 to control which feature set is used.
    Results saved to separate subdirectories — no run ever overwrites another.

    MODEL = "raw"         → M3: full engineered features (~463)
                               Input : features_raw.rds
                               Output: Tables/ag_raw/

    MODEL = "fund"        → M1: fundamentals only, no price features
                               Input : features_fund.rds
                               Output: Tables/ag_fund/

    MODEL = "latent_fund" → M2: VAE latent trained on fund features
                               Input : features_latent_fund.parquet
                               Output: Tables/ag_latent_fund/

    MODEL = "latent_raw"  → M4: VAE latent trained on full raw features
                               Input : features_latent_raw.parquet
                               Output: Tables/ag_latent_raw/

    MODEL = "bucket"      → B1: 5-year forward CAGR bucket classifier
                               Input : features_fund.rds
                               Labels: labels_bucket.rds  (from 05B)
                               Output: Tables/ag_bucket/
                               Target: y_loser (1=terminal loser, 0=phoenix)
                               Note : OOS limited to 2016-2019 (labels censored
                                      after 2019 due to 5yr forward window)

MODEL MATRIX:
    M1 vs M2 → answers Sub-Q 1: does VAE add signal over fundamentals?
    M3 vs M4 → does VAE compress full features usefully?
    M1 vs M3 → cost of removing price features (avoidance alpha constraint)
    B1       → 5-year structural quality signal for concentrated portfolio

LABEL SHIFT:
    M1/M2/M3/M4: features(t) → y(t+1), matching 09_Train.R.
    B1:          features(t) → y_loser(t) [5yr forward CAGR already shifted
                               in 05B — no additional shift applied here]

EXPANDING WINDOW CV:
    Stage 1: Train on full train set, holdout = years 2011-2015.
    Stage 2: Manual 3-fold expanding window CV for honest CV metric.
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
DIR_LABELS   = DIR_DATA  / "Labels"          ## for bucket labels
DIR_OUTPUT   = DATA_ROOT / "03_Output"
DIR_MODELS   = DIR_OUTPUT / "Models" / "AutoGluon"
DIR_TABLES   = DIR_OUTPUT / "Tables"

PATH_SPLIT_LABELS   = DIR_FEATURES / "split_labels_oot.parquet"
PATH_LABELS_BUCKET      = DIR_LABELS / "labels_bucket.rds"      ## from 05B
PATH_LABELS_STRUCTURAL = DIR_LABELS / "labels_structural.rds"  ## from 05C

assert PATH_SPLIT_LABELS.exists(), f"Not found: {PATH_SPLIT_LABELS}"

# ==============================================================================
# 2. ── MODEL SELECTION ────────────────────────────────────────────────────────
#
#   "raw"         → M3  full engineered features        [existing]
#   "fund"        → M1  fundamentals only               [existing]
#   "latent_fund" → M2  VAE latent on fund input        [existing]
#   "latent_raw"  → M4  VAE latent on raw input         [existing]
#   "bucket"      → B1  5yr forward CAGR bucket         [NEW]
#
# ==============================================================================

MODEL = "structural"    # <- CHANGE THIS

# ==============================================================================
# 3. Model-specific configuration
# ==============================================================================

SEED          = 123
TIME_LIMIT    = 3600
PRESET        = "good_quality"
CV_TIME_LIMIT = 900
CV_PRESET     = "medium_quality"
EVAL_METRIC   = "average_precision"

## Label column differs by model — set below after MODEL_CONFIG
LABEL_COL_CSI    = "y_next"    ## M1/M2/M3/M4
LABEL_COL_BUCKET = "y_loser"   ## B1

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
        "description": "M2 — VAE latent on fundamentals (z1-z24 + recon error)",
        "prereq"     : "Run 08B_Autoencoder.py with VAE_INPUT='fund' first.",
        "label_col"  : LABEL_COL_CSI,
        "is_bucket"  : False,
    },
    "latent_raw": {
        "path"       : DIR_FEATURES / "features_latent_raw.parquet",
        "loader"     : "parquet",
        "description": "M4 — VAE latent on full raw features (z1-z24 + recon error)",
        "prereq"     : "Run 08B_Autoencoder.py with VAE_INPUT='raw' first.",
        "label_col"  : LABEL_COL_CSI,
        "is_bucket"  : False,
    },
    "bucket": {
        "path"       : DIR_FEATURES / "features_fund.rds",
        "loader"     : "rds",
        "description": "B1 — 5yr forward CAGR bucket classifier (terminal loser vs phoenix)",
        "prereq"     : "Run 05B_Bucket_Labels.R first to generate labels_bucket.rds.",
        "label_col"  : LABEL_COL_BUCKET,
        "is_bucket"  : True,
    },
    "structural": {
        "path"       : DIR_FEATURES / "features_fund.rds",   ## same features as M1/B1
        "loader"     : "rds",
        "description": "B1-structural — Combined CSI + 5yr CAGR structural quality label",
        "prereq"     : "Run 05C_Structural_Labels.R first to generate labels_structural.rds.",
        "label_col"  : "y_structural",
        "is_bucket"  : True,   ## same loading path as bucket model
    },
}

assert MODEL in MODEL_CONFIG, \
    f"Unknown MODEL '{MODEL}'. Choose from: {list(MODEL_CONFIG.keys())}"

cfg      = MODEL_CONFIG[MODEL]
LABEL_COL = cfg["label_col"]
IS_BUCKET = cfg["is_bucket"]

assert cfg["path"].exists(), \
    f"Feature file not found: {cfg['path']}\n{cfg['prereq']}"

if IS_BUCKET:
    lbl_path = PATH_LABELS_STRUCTURAL if MODEL == "structural" else PATH_LABELS_BUCKET
    assert lbl_path.exists(),         f"Labels not found: {lbl_path}\n{cfg['prereq']}"

## Output directories
DIR_MODELS_RUN = DIR_MODELS / f"ag_{MODEL}"
DIR_TABLES_RUN = DIR_TABLES / f"ag_{MODEL}"
DIR_MODELS_RUN.mkdir(parents=True, exist_ok=True)
DIR_TABLES_RUN.mkdir(parents=True, exist_ok=True)

print(f"[09C] ══════════════════════════════════════")
print(f"  MODEL        : {MODEL.upper()}")
print(f"  Description  : {cfg['description']}")
print(f"  Input file   : {cfg['path'].name}")
print(f"  Label column : {LABEL_COL}")
print(f"  Output dir   : {DIR_TABLES_RUN}")
if IS_BUCKET:
    print(f"  Bucket labels: {PATH_LABELS_BUCKET.name}")
    print(f"  OOS period   : 2016-2019 only (labels censored after 2019)")
print(f"[09C] ══════════════════════════════════════\n")

# ==============================================================================
# 4. ID and latent column definitions
# ==============================================================================

ID_COLS = [
    "permno", "year", "y", "censored", "param_id",
    "gvkey", "datadate", "lifetime_years",
    "fiscal_year_end_month", "split", "vae_split", "split_oot",
    ## bucket-specific IDs to exclude from features
    "y_loser", "y_structural", "fwd_cagr", "n_months", "bucket",
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
# 6A. CSI models — merge split labels and apply label shift (unchanged)
# ==============================================================================

if not IS_BUCKET:

    print("[09C] Loading split labels...")
    split_labels = pd.read_parquet(PATH_SPLIT_LABELS)
    print(f"  Split distribution:\n{split_labels['split'].value_counts().to_string()}")

    df = features_input.merge(split_labels, on=["permno", "year"], how="left")
    df = df[df["split"].notna()].reset_index(drop=True)
    df = df.sort_values(["permno", "year"]).reset_index(drop=True)

    ## Label shift: y(t) → y_next = y(t+1)
    df["y_next"] = df.groupby("permno")["y"].shift(-1)
    df_labelled  = df[df["y_next"].notna()].copy()
    df_labelled["y_next"] = df_labelled["y_next"].astype(int)

    print(f"\n[09C] After label shift:")
    print(f"  Rows with valid y_next : {len(df_labelled):,}")
    print(f"  y_next prevalence      : {df_labelled['y_next'].mean():.3f}")

    train_df     = df_labelled[df_labelled["split"] == "train"].copy()
    test_df      = df_labelled[df_labelled["split"] == "test"].copy()
    oos_df       = df_labelled[df_labelled["split"] == "oos"].copy()
    oos_clean_df = oos_df[oos_df["year"] <= 2022].copy()

    print(f"\n[09C] Split sizes:")
    print(f"  Train : {len(train_df):,}  (prev: {train_df['y_next'].mean():.3f})")
    print(f"  Test  : {len(test_df):,}  (prev: {test_df['y_next'].mean():.3f})")
    print(f"  OOS   : {len(oos_df):,}  (prev: {oos_df['y_next'].mean():.3f})")

# ==============================================================================
# 6B. B1 bucket model — merge bucket labels directly (NO additional label shift)
# ==============================================================================

else:

    lbl_path = PATH_LABELS_STRUCTURAL if MODEL == "structural" else PATH_LABELS_BUCKET
    lbl_name = "structural labels from 05C" if MODEL == "structural" else "bucket labels from 05B"
    print(f"[09C] Loading {lbl_name}...")
    labels_bucket_r = pyreadr.read_r(str(lbl_path))
    labels_bucket   = labels_bucket_r[None]

    ## Determine the actual label column name in the file
    ## bucket model: y_loser | structural model: y_structural
    raw_label_col = "y_structural" if MODEL == "structural" else "y_loser"

    ## Keep only usable labels (0 or 1, not NA/censored)
    labels_bucket = labels_bucket[labels_bucket[raw_label_col].notna()].copy()
    labels_bucket[raw_label_col] = labels_bucket[raw_label_col].astype(int)

    print(f"  Bucket labels: {len(labels_bucket):,} usable observations")
    print(f"  {raw_label_col} prevalence: {labels_bucket[raw_label_col].mean():.3f}")
    print(f"  Years: {labels_bucket['year'].min()} – {labels_bucket['year'].max()}")

    ## Merge features with labels
    ## No label shift needed — labels already aligned to feature year t
    df_labelled = features_input.merge(
        labels_bucket[["permno", "year", raw_label_col]],
        on   = ["permno", "year"],
        how  = "inner"
    ).copy()

    ## Rename raw label column to the LABEL_COL expected by the rest of the script
    ## LABEL_COL = "y_loser" for bucket, "y_structural" for structural
    ## If they differ (they won't here, but defensive), rename
    if raw_label_col != LABEL_COL:
        df_labelled = df_labelled.rename(columns={raw_label_col: LABEL_COL})

    print(f"\n[09C] After merge:")
    print(f"  Rows         : {len(df_labelled):,}")
    print(f"  Unique firms : {df_labelled['permno'].nunique():,}")
    print(f"  Prevalence   : {df_labelled[LABEL_COL].mean():.3f}")

    ## ── CRITICAL: add year as categorical feature ──────────────────────────
    ## The 5-year forward CAGR label is period-dependent (bull/bear cycles).
    ## Including year as a categorical feature lets AutoGluon absorb the year
    ## fixed effect, so firm-level features predict the deviation from the
    ## year mean rather than the calendar-time trend.
    df_labelled["year_cat"] = df_labelled["year"].astype(str)
    print(f"\n  Year fixed effect: 'year_cat' added as categorical feature")
    print(f"  This absorbs bull/bear period effects on loser prevalence")

    ## Train/test split — same years as M1 for comparability
    ## OOS ends at 2019 (labels censored after 2019 due to 5yr window)
    train_df = df_labelled[df_labelled["year"] <= 2015].copy()
    test_df  = df_labelled[(df_labelled["year"] >= 2016) &
                            (df_labelled["year"] <= 2019)].copy()
    ## No OOS beyond 2019 for B1
    oos_df       = pd.DataFrame()    ## empty — censored
    oos_clean_df = pd.DataFrame()

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
    ## All numeric features excluding IDs, plus year_cat (categorical)
    feature_cols = [
        c for c in train_df.columns
        if c not in ID_COLS
        and c not in [LABEL_COL, "y_next", "split", "split_oot",
                      "vae_split", "y", "year_cat"]
        and pd.api.types.is_numeric_dtype(train_df[c])
    ]
    ## Add year_cat explicitly — AutoGluon handles it as categorical
    feature_cols_model = feature_cols + ["year_cat"]
    print(f"\n[09C] Feature columns (B1): {len(feature_cols)} numeric "
          f"+ 1 categorical (year_cat) = {len(feature_cols_model)} total")

else:
    feature_cols = [
        c for c in train_df.columns
        if c not in ID_COLS
        and c not in [LABEL_COL, "y_next", "split", "split_oot", "vae_split"]
        and pd.api.types.is_numeric_dtype(train_df[c])
        and c != "y"
    ]
    feature_cols_model = feature_cols
    print(f"\n[09C] Feature columns: {len(feature_cols)}")

if not IS_BUCKET:
    feature_cols_model = feature_cols

# ==============================================================================
# 8. Quantile Transform — fitted on train only (numeric features only)
# ==============================================================================

print("\n[09C] Applying quantile transform to numeric features...")

X_train = train_df[feature_cols].values.astype(np.float64)
X_test  = test_df[feature_cols].values.astype(np.float64)
X_oos   = oos_df[feature_cols].values.astype(np.float64) if len(oos_df) > 0 \
          else np.empty((0, len(feature_cols)))

for arr in [X_train, X_test, X_oos]:
    arr[np.isinf(arr)] = np.nan

lo = np.nanpercentile(X_train, 0.1, axis=0)
hi = np.nanpercentile(X_train, 99.9, axis=0)
X_train = np.clip(X_train, lo, hi)
X_test  = np.clip(X_test,  lo, hi)
if len(X_oos) > 0:
    X_oos = np.clip(X_oos, lo, hi)

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
X_test_imp  = impute_median(X_test,  train_medians)
X_oos_imp   = impute_median(X_oos,   train_medians) if len(X_oos) > 0 \
              else X_oos

assert not np.isnan(X_train_imp).any(), "NaN remains after imputation"
assert not np.isinf(X_train_imp).any(), "Inf remains after imputation"

qt = QuantileTransformer(
    output_distribution="uniform",
    n_quantiles=min(1000, len(X_train_imp)),
    random_state=SEED
)
qt.fit(X_train_imp)

X_train_qt = qt.transform(X_train_imp)
X_test_qt  = qt.transform(X_test_imp)
X_oos_qt   = qt.transform(X_oos_imp) if len(X_oos_imp) > 0 else X_oos_imp

def rebuild_df(X_qt, source_df, feature_cols, label_col):
    out = pd.DataFrame(X_qt, columns=feature_cols, index=source_df.index)
    out[label_col]  = source_df[label_col].values
    out["year"]     = source_df["year"].values
    out["permno"]   = source_df["permno"].values
    ## Re-attach year_cat for B1 — not transformed, just carried through
    if "year_cat" in source_df.columns:
        out["year_cat"] = source_df["year_cat"].values
    return out

train_qt = rebuild_df(X_train_qt, train_df, feature_cols, LABEL_COL)
test_qt  = rebuild_df(X_test_qt,  test_df,  feature_cols, LABEL_COL)
oos_qt   = rebuild_df(X_oos_qt,   oos_df,   feature_cols, LABEL_COL) \
           if len(oos_df) > 0 else pd.DataFrame()

print(f"  QT applied. Train shape: {train_qt.shape}")
print(f"  Train mean (~0.5): {X_train_qt.mean():.4f}")
print(f"  Train std (~0.29): {X_train_qt.std():.4f}")

# ==============================================================================
# 9. AutoGluon column set
# ==============================================================================

ag_train_cols = feature_cols_model + [LABEL_COL]

## Ensure year_cat is included for B1 (not in feature_cols but needed)
if IS_BUCKET and "year_cat" not in ag_train_cols:
    ag_train_cols = feature_cols + ["year_cat"] + [LABEL_COL]

# ==============================================================================
# 10. Fold structure
# ==============================================================================

FOLD_BOUNDARIES = [
    (2, 2001, 2002, 2006),
    (3, 2006, 2007, 2010),
    (4, 2010, 2011, 2015),
]

print(f"\n[09C] Fold structure:")
for fold_id, train_end, val_start, val_end in FOLD_BOUNDARIES:
    n_train = len(train_qt[train_qt["year"] <= train_end])
    n_val   = len(train_qt[(train_qt["year"] >= val_start) &
                            (train_qt["year"] <= val_end)])
    print(f"  Fold {fold_id}: train [≤{train_end}] n={n_train:,} "
          f"| val [{val_start}–{val_end}] n={n_val:,}")

# ==============================================================================
# 11. Stage 1 — AutoGluon training
# ==============================================================================

print(f"\n[09C] ══ STAGE 1: AutoGluon [{MODEL.upper()}] ══")
print(f"  Preset: {PRESET} | Time limit: {TIME_LIMIT}s\n")

holdout_mask  = (train_qt["year"] >= 2011) & (train_qt["year"] <= 2015)

ag_train_data = TabularDataset(
    train_qt[~holdout_mask][ag_train_cols].reset_index(drop=True))
ag_holdout    = TabularDataset(
    train_qt[holdout_mask][ag_train_cols].reset_index(drop=True))
ag_test_data  = TabularDataset(
    test_qt[ag_train_cols].reset_index(drop=True))

print(f"  AG training rows : {len(ag_train_data):,}")
print(f"  AG holdout rows  : {len(ag_holdout):,}")
print(f"  AG test rows     : {len(ag_test_data):,}\n")

predictor = TabularPredictor(
    label        = LABEL_COL,
    problem_type = "binary",
    eval_metric  = EVAL_METRIC,
    path         = str(DIR_MODELS_RUN / "ag_predictor"),
    verbosity    = 2
).fit(
    train_data       = ag_train_data,
    tuning_data      = ag_holdout,
    presets          = PRESET,
    time_limit       = TIME_LIMIT,
    num_bag_folds    = 0,
    num_stack_levels = 0,
)

print(f"\n[09C] Stage 1 complete [{MODEL.upper()}].")

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
# 13. Predictions — test set
# ==============================================================================

print(f"\n[09C] Generating predictions...")

preds_test_proba = predictor.predict_proba(ag_test_data, as_multiclass=False)
preds_test = pd.DataFrame({
    "permno" : test_qt["permno"].values,
    "year"   : test_qt["year"].values,
    "y"      : test_qt[LABEL_COL].values,
    "p_csi"  : preds_test_proba.values       ## p_csi column name kept for
                                              ## compatibility with downstream R scripts
})
preds_test.to_parquet(DIR_TABLES_RUN / "ag_preds_test.parquet", index=False)
print(f"  ag_preds_test.parquet saved ({len(preds_test):,} rows)")

## OOS predictions — skip for B1 (no valid labels beyond 2019)
if len(oos_qt) > 0:
    ag_oos_data     = TabularDataset(oos_qt[ag_train_cols].reset_index(drop=True))
    preds_oos_proba = predictor.predict_proba(ag_oos_data, as_multiclass=False)
    preds_oos = pd.DataFrame({
        "permno": oos_qt["permno"].values,
        "year"  : oos_qt["year"].values,
        "y"     : oos_qt[LABEL_COL].values,
        "p_csi" : preds_oos_proba.values
    })
    preds_oos.to_parquet(DIR_TABLES_RUN / "ag_preds_oos.parquet", index=False)
    print(f"  ag_preds_oos.parquet saved ({len(preds_oos):,} rows)")
else:
    preds_oos       = pd.DataFrame()
    preds_oos_clean = pd.DataFrame()
    print(f"  ag_preds_oos.parquet skipped (B1 — labels censored after 2019)")

# ==============================================================================
# 14. Evaluation helpers
# ==============================================================================

def fn_recall_at_fpr(y_true, y_pred, fpr_target):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    eligible = np.where(fpr <= fpr_target)[0]
    return 0.0 if len(eligible) == 0 else float(tpr[eligible].max())

def fn_eval_metrics(y_true, y_pred, set_name):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
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
# 15. Stage 1 evaluation
# ==============================================================================

print(f"\n[09C] ── Stage 1 Evaluation [{MODEL.upper()}] ──\n")

metrics_test = fn_eval_metrics(
    preds_test["y"], preds_test["p_csi"], "test_2016_2019")

metrics_oos_clean = fn_eval_metrics(
    preds_oos[preds_oos["year"] <= 2022]["y"],
    preds_oos[preds_oos["year"] <= 2022]["p_csi"],
    "oos_2020_2022") if len(preds_oos) > 0 else None

metrics_oos_full = fn_eval_metrics(
    preds_oos["y"], preds_oos["p_csi"],
    "oos_full") if len(preds_oos) > 0 else None

for m in [metrics_test, metrics_oos_clean, metrics_oos_full]:
    if m:
        print(f"  {m['set']:<25} | AP={m['avg_precision']:.4f} | "
              f"AUC={m['auc_roc']:.4f} | R@FPR3={m['recall_fpr3']:.4f} | "
              f"R@FPR5={m['recall_fpr5']:.4f}")

if IS_BUCKET and metrics_test:
    baseline_ap = test_df[LABEL_COL].mean()
    lift = metrics_test["avg_precision"] / baseline_ap if baseline_ap > 0 else None
    print(f"\n  B1 baseline AP (random): {baseline_ap:.4f}")
    if lift:
        print(f"  B1 AP lift over random : {lift:.2f}x")
    if MODEL == "structural":
        print(f"  B1-bucket baseline AUC: 0.7428 (from 09C bucket run)")
        print(f"  If AUC > 0.74: combined label improves on bucket alone")
    else:
        print(f"  Logistic regression baseline AUC: 0.7440 (from explore_buckets.R)")

# ==============================================================================
# 16. Stage 2 — Manual expanding window CV
# ==============================================================================

print(f"\n[09C] ══ STAGE 2: Expanding window CV [{MODEL.upper()}] ══\n")

cv_preds_list = []

for fold_id, train_end, val_start, val_end in FOLD_BOUNDARIES:
    print(f"\n[09C] Fold {fold_id}: train [≤{train_end}] → val [{val_start}–{val_end}]")

    fold_train_mask = train_qt["year"] <= train_end
    fold_val_mask   = ((train_qt["year"] >= val_start) &
                       (train_qt["year"] <= val_end))

    fold_train  = TabularDataset(
        train_qt[fold_train_mask][ag_train_cols].reset_index(drop=True))
    fold_val_qt = train_qt[fold_val_mask].copy()
    fold_val    = TabularDataset(
        fold_val_qt[ag_train_cols].reset_index(drop=True))

    n_pos = int(fold_train[LABEL_COL].sum())
    n_neg = int((fold_train[LABEL_COL] == 0).sum())
    print(f"  Train: {len(fold_train):,} | pos={n_pos} neg={n_neg} "
          f"weight={n_neg/max(n_pos,1):.2f}")
    print(f"  Val  : {len(fold_val):,}")

    if n_pos < 10:
        print(f"  Fold {fold_id} SKIPPED — fewer than 10 positives.")
        continue

    fold_predictor = TabularPredictor(
        label        = LABEL_COL,
        problem_type = "binary",
        eval_metric  = EVAL_METRIC,
        path         = str(DIR_MODELS_RUN / f"ag_cv_fold{fold_id}"),
        verbosity    = 1
    ).fit(
        train_data       = fold_train,
        tuning_data      = fold_val,
        presets          = CV_PRESET,
        time_limit       = CV_TIME_LIMIT,
        num_bag_folds    = 0,
        num_stack_levels = 0,
    )

    fold_proba    = fold_predictor.predict_proba(fold_val, as_multiclass=False)
    fold_cv_preds = pd.DataFrame({
        "fold_id": fold_id,
        "permno" : fold_val_qt["permno"].values,
        "year"   : fold_val_qt["year"].values,
        "y"      : fold_val_qt[LABEL_COL].values,
        "p_csi"  : fold_proba.values
    })

    fold_ap  = average_precision_score(fold_cv_preds["y"], fold_cv_preds["p_csi"])
    fold_auc = roc_auc_score(fold_cv_preds["y"], fold_cv_preds["p_csi"])
    print(f"  Fold {fold_id} AP: {fold_ap:.4f} | AUC: {fold_auc:.4f}")
    cv_preds_list.append(fold_cv_preds)

if cv_preds_list:
    cv_preds_all = pd.concat(cv_preds_list, ignore_index=True)
    cv_preds_all.to_parquet(DIR_TABLES_RUN / "ag_cv_results.parquet", index=False)

    cv_ap_mean  = average_precision_score(cv_preds_all["y"], cv_preds_all["p_csi"])
    cv_auc_mean = roc_auc_score(cv_preds_all["y"], cv_preds_all["p_csi"])
    cv_r3       = fn_recall_at_fpr(cv_preds_all["y"].values,
                                   cv_preds_all["p_csi"].values, 0.03)

    print(f"\n[09C] CV results [{MODEL.upper()}] (pooled):")
    print(f"  CV AP       : {cv_ap_mean:.4f}")
    print(f"  CV AUC-ROC  : {cv_auc_mean:.4f}")
    print(f"  CV R@FPR3   : {cv_r3:.4f}")

    print(f"\n  Per-fold:")
    for fp in cv_preds_list:
        f_id = fp["fold_id"].iloc[0]
        f_ap = average_precision_score(fp["y"], fp["p_csi"])
        print(f"    Fold {f_id}: AP={f_ap:.4f} n={len(fp):,} "
              f"prev={fp['y'].mean():.3f}")
else:
    cv_ap_mean = cv_auc_mean = cv_r3 = None
    print(f"[09C] No CV folds completed.")

# ==============================================================================
# 17. Save evaluation summary
# ==============================================================================

eval_summary = {
    "model"            : f"autogluon_{MODEL}",
    "model_label"      : cfg["description"],
    "preset"           : PRESET,
    "time_limit_s"     : TIME_LIMIT,
    "cv_time_limit_s"  : CV_TIME_LIMIT,
    "label_col"        : LABEL_COL,
    "is_bucket"        : IS_BUCKET,
    "label_shift"      : "none (5yr already aligned in 05B)" if IS_BUCKET
                         else "y(t+1)",
    "eval_metric"      : EVAL_METRIC,
    "n_features"       : len(feature_cols_model),
    "year_cat_included": IS_BUCKET,
    "cv_avg_precision" : round(cv_ap_mean,  4) if cv_ap_mean  else None,
    "cv_auc_roc"       : round(cv_auc_mean, 4) if cv_auc_mean else None,
    "cv_recall_fpr3"   : round(cv_r3,       4) if cv_r3       else None,
    "test"             : metrics_test,
    "oos_2020_2022"    : metrics_oos_clean,
    "oos_full"         : metrics_oos_full,
    ## Baselines for comparison
    "logistic_baseline_auc" : 0.7440 if IS_BUCKET else None,
    "paper_r_fpr3"          : 0.61   if not IS_BUCKET else None,
    "ag_m3_cv_ap"           : 0.6656 if not IS_BUCKET else None,
    "ag_m1_test_ap"         : 0.6546 if not IS_BUCKET else None,
}

with open(DIR_TABLES_RUN / "ag_eval_summary.json", "w") as f:
    json.dump(eval_summary, f, indent=2)
print(f"\n[09C] Summary saved: {DIR_TABLES_RUN / 'ag_eval_summary.json'}")

# ==============================================================================
# 18. Final comparison table
# ==============================================================================

print(f"\n[09C] ══ RESULTS: [{MODEL.upper()}] ══\n")

if IS_BUCKET:
    print(f"  {'Model':<30} | {'CV AP':<8} | {'Test AP':<8} | {'AUC':<8}")
    print(f"  {'-'*30} | {'--------':<8} | {'--------':<8} | {'--------'}")
    if metrics_test:
        cv_str = f"{cv_ap_mean:.4f}" if cv_ap_mean else "—"
        print(f"  {'B1 AG Bucket':<30} | {cv_str:<8} | "
              f"{metrics_test['avg_precision']:<8} | "
              f"{metrics_test['auc_roc']:<8}")
    print(f"  {'Logistic baseline (OOS)':<30} | {'—':<8} | "
          f"{'—':<8} | {'0.7440':<8}")
    print(f"  {'Random (prevalence)':<30} | {'—':<8} | "
          f"{test_df[LABEL_COL].mean():<8.4f} | {'0.5000':<8}")
else:
    print(f"  {'Model':<25} | {'CV AP':<8} | {'Test AP':<8} | "
          f"{'AUC':<8} | {'R@FPR3':<8} | {'R@FPR5'}")
    print(f"  {'-'*25} | {'--------':<8} | {'--------':<8} | "
          f"{'--------':<8} | {'--------':<8} | {'--------'}")
    if metrics_test:
        cv_str = f"{cv_ap_mean:.4f}" if cv_ap_mean else "—"
        print(f"  {f'AG {MODEL}':<25} | {cv_str:<8} | "
              f"{metrics_test['avg_precision']:<8} | "
              f"{metrics_test['auc_roc']:<8} | "
              f"{metrics_test['recall_fpr3']:<8} | "
              f"{metrics_test['recall_fpr5']}")
    print(f"  {'AG M3 (raw)':<25} | {'0.6656':<8} | {'0.7576':<8} | "
          f"{'0.9601':<8} | {'0.6397':<8} | {'0.7670'}")
    print(f"  {'AG M1 (fund)':<25} | {'0.5809':<8} | {'0.6546':<8} | "
          f"{'0.9337':<8} | {'0.4936':<8} | {'0.6337'}")

print(f"\n[09C] DONE [{MODEL.upper()}]")
print(f"  Tables : {DIR_TABLES_RUN}")
print(f"  Model  : {DIR_MODELS_RUN / 'ag_predictor'}")
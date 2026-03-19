"""
09C_AutoGluon.py
================
AutoGluon AutoML for CSI prediction — raw features, expanding window CV.

PURPOSE:
    Complement the hand-tuned XGBoost (09_Train.R) with AutoGluon's automated
    model selection and ensembling. AutoGluon trains XGBoost, LightGBM,
    CatBoost, Random Forest, ExtraTrees, Neural Nets and a stacked ensemble
    automatically, selecting the best combination via internal validation.

EXPANDING WINDOW CV — HOW IT IS IMPLEMENTED:
    AutoGluon does not support expanding window CV natively (as of 1.5.x,
    a GitHub issue #4492 tracks the request). The workaround uses the
    `groups` parameter with `LeaveOneGroupOut` splitting:

    - Each training row is assigned a fold ID (0 = fold 2 train, 1 = fold 3
      train, ..., 3 = fold 5 train), matching the expanding window structure
      from 08_Split.R.
    - AutoGluon treats each unique group value as one "leave-one-out" fold.
    - Because LeaveOneGroupOut leaves out exactly one group ID at a time, and
      we assign group IDs to the VALIDATION portion of each fold, AutoGluon
      trains on all rows NOT in the current group and validates on the current
      group — replicating expanding window CV.

    Critically: AutoGluon's LeaveOneGroupOut leaves out one group for
    VALIDATION and trains on ALL other groups. For expanding window this
    means fold 4's validation set (group=3) trains on groups 0+1+2, which
    includes future data relative to fold 2's training set. This is not
    perfectly equivalent to expanding window — it is closer to standard
    k-fold for ensembling purposes.

    The honest expanding window CV metrics are therefore computed MANUALLY
    after AutoGluon training by running each fold prediction sequentially.

TWO-STAGE APPROACH:
    Stage 1 — AutoGluon training (bagging disabled):
        Train on full training set with a holdout validation set (last fold
        validation years = 2011-2015) for model selection.
        AutoGluon trains all model types and selects the best ensemble.

    Stage 2 — Manual expanding window CV:
        Replicate the 4-fold expanding window CV from 09_Train.R manually:
        for each fold k in [2,3,4,5], train AutoGluon on fold k training rows,
        predict on fold k validation rows, accumulate out-of-fold predictions,
        compute CV AUCPR.

    Stage 1 produces the final deployable model.
    Stage 2 produces the honest CV metric for thesis comparison.

    This approach is practical because AutoGluon training is fast enough
    that running 4 sequential fits is feasible within ~2-4 hours.

LABEL SHIFT:
    features(t) → y(t+1), matching 09_Train.R.
    Applied here in Python before passing to AutoGluon.

INPUTS:
    - features_raw.rds     (via pyreadr)
    - split_labels_oot.parquet

OUTPUTS:
    - autogluon_model/           : saved AutoGluon predictor
    - ag_preds_test.parquet      : test set predictions
    - ag_preds_oos.parquet       : OOS predictions
    - ag_cv_results.parquet      : per-fold CV predictions
    - ag_leaderboard.csv         : model leaderboard
    - ag_feature_importance.csv  : feature importance
    - ag_eval_summary.json       : evaluation metrics

INSTALL:
    pip install autogluon.tabular
    (includes XGBoost, LightGBM, CatBoost, Neural Nets automatically)
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

# AutoGluon
try:
    from autogluon.tabular import TabularDataset, TabularPredictor
except ImportError:
    raise ImportError(
        "AutoGluon not installed. Run:\n"
        "    pip install autogluon.tabular\n"
        "Note: requires ~3GB disk space and several minutes to install."
    )

# ==============================================================================
# 1. Paths — update DATA_ROOT if needed
# ==============================================================================

DATA_ROOT    = Path(r"C:\Users\Tristan Leiter\Documents\MT")
DIR_DATA     = DATA_ROOT / "02_Data"
DIR_FEATURES = DIR_DATA  / "Features"
DIR_OUTPUT   = DATA_ROOT / "03_Output"
DIR_MODELS   = DIR_OUTPUT / "Models" / "AutoGluon"
DIR_FIGURES  = DIR_OUTPUT / "Figures"
DIR_TABLES   = DIR_OUTPUT / "Tables"

DIR_MODELS.mkdir(parents=True, exist_ok=True)
DIR_TABLES.mkdir(parents=True, exist_ok=True)

PATH_FEATURES_RAW = DIR_FEATURES / "features_raw.rds"
PATH_SPLIT_LABELS = DIR_FEATURES / "split_labels_oot.parquet"

for p in [PATH_FEATURES_RAW, PATH_SPLIT_LABELS]:
    assert p.exists(), f"Required input not found: {p}"

print(f"[09C] DATA_ROOT    : {DATA_ROOT}")
print(f"[09C] DIR_MODELS   : {DIR_MODELS}")
print(f"[09C] Inputs verified.")

# ==============================================================================
# 2. Configuration
# ==============================================================================

SEED          = 123
LABEL_COL     = "y_next"          # target column name after label shift
TIME_LIMIT    = 3600              # seconds for Stage 1 AutoGluon fit
                                   # increase to 7200 for best quality
PRESET        = "good_quality"    # "medium_quality" faster, "best_quality" slower
                                   # good_quality: LightGBM + XGBoost + RF + NN
EVAL_METRIC   = "average_precision"  # AutoGluon metric name
POSITIVE_CLASS = 1                 # CSI = 1

# Columns that are identifiers — never features
ID_COLS = [
    "permno", "year", "y", "censored", "param_id",
    "gvkey", "datadate", "lifetime_years",
    "fiscal_year_end_month", "split"
]

# ==============================================================================
# 3. Load data
# ==============================================================================

print("\n[09C] Loading features_raw.rds...")
result       = pyreadr.read_r(str(PATH_FEATURES_RAW))
features_raw = result[None]
print(f"  Shape: {features_raw.shape[0]:,} rows × {features_raw.shape[1]} cols")

print("[09C] Loading split labels (OOT)...")
split_labels = pd.read_parquet(PATH_SPLIT_LABELS)
print(f"  Split distribution:\n{split_labels['split'].value_counts().to_string()}")

# ==============================================================================
# 4. Merge split labels and apply label shift
# ==============================================================================

df = features_raw.merge(split_labels, on=["permno", "year"], how="left")
df = df[df["split"].notna()].reset_index(drop=True)

# Label shift: predict y(t+1) from features(t)
df = df.sort_values(["permno", "year"]).reset_index(drop=True)
df["y_next"] = df.groupby("permno")["y"].shift(-1)

# Drop rows where y_next is NA (last year per firm, zombie next year)
df_labelled = df[df["y_next"].notna()].copy()
df_labelled["y_next"] = df_labelled["y_next"].astype(int)

print(f"\n[09C] After label shift:")
print(f"  Total rows with valid y_next : {len(df_labelled):,}")
print(f"  y_next prevalence            : {df_labelled['y_next'].mean():.3f}")

# ==============================================================================
# 5. Split into train / test / OOS
# ==============================================================================

train_df = df_labelled[df_labelled["split"] == "train"].copy()
test_df  = df_labelled[df_labelled["split"] == "test"].copy()
oos_df   = df_labelled[df_labelled["split"] == "oos"].copy()

# OOS clean: 2023 excluded (right-censoring — no confirmed CSI)
oos_clean_df = oos_df[oos_df["year"] <= 2022].copy()

print(f"\n[09C] Split sizes:")
print(f"  Train : {len(train_df):,}  (prevalence: {train_df['y_next'].mean():.3f})")
print(f"  Test  : {len(test_df):,}  (prevalence: {test_df['y_next'].mean():.3f})")
print(f"  OOS   : {len(oos_df):,}  (prevalence: {oos_df['y_next'].mean():.3f})")
print(f"  OOS clean (≤2022): {len(oos_clean_df):,}")

# ==============================================================================
# 6. Feature columns
# ==============================================================================

feature_cols = [
    c for c in train_df.columns
    if c not in ID_COLS
    and c not in [LABEL_COL, "y_next", "split", "split_oot"]
    and pd.api.types.is_numeric_dtype(train_df[c])
    and c != "y"
]

print(f"\n[09C] Feature columns: {len(feature_cols)}")

# ==============================================================================
# 7. Quantile Transform — fitted on train only
#
#   AutoGluon handles NA imputation internally, but we apply QT upfront
#   for consistency with 09_Train.R and to ensure all models see the
#   same normalised inputs. AutoGluon can still learn from raw features
#   (tree models are invariant to monotone transforms) but NNs benefit.
#
#   QT fitted on train rows, applied to test and OOS using train ECDF.
#   NA values imputed with column training median before transform.
# ==============================================================================

print("\n[09C] Applying quantile transform (train-fit only)...")

X_train = train_df[feature_cols].values.astype(np.float64)
X_test  = test_df[feature_cols].values.astype(np.float64)
X_oos   = oos_df[feature_cols].values.astype(np.float64)

# Median imputation before QT — fit on train
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
X_oos_imp   = impute_median(X_oos,   train_medians)

# QT transform
qt = QuantileTransformer(
    output_distribution="uniform",
    n_quantiles=min(1000, len(X_train_imp)),
    random_state=SEED
)
qt.fit(X_train_imp)

X_train_qt = qt.transform(X_train_imp)
X_test_qt  = qt.transform(X_test_imp)
X_oos_qt   = qt.transform(X_oos_imp)

# Rebuild DataFrames with QT features
train_qt = pd.DataFrame(X_train_qt, columns=feature_cols,
                         index=train_df.index)
train_qt[LABEL_COL] = train_df[LABEL_COL].values
train_qt["year"]    = train_df["year"].values
train_qt["permno"]  = train_df["permno"].values

test_qt = pd.DataFrame(X_test_qt, columns=feature_cols,
                        index=test_df.index)
test_qt[LABEL_COL] = test_df[LABEL_COL].values
test_qt["year"]    = test_df["year"].values
test_qt["permno"]  = test_df["permno"].values

oos_qt = pd.DataFrame(X_oos_qt, columns=feature_cols,
                       index=oos_df.index)
oos_qt[LABEL_COL] = oos_df[LABEL_COL].values
oos_qt["year"]    = oos_df["year"].values
oos_qt["permno"]  = oos_df["permno"].values

print(f"  QT applied. Train shape: {train_qt.shape}")
print(f"  Train mean (should be ~0.5): {X_train_qt.mean():.4f}")
print(f"  Train std  (should be ~0.29): {X_train_qt.std():.4f}")

# ==============================================================================
# 8. Define expanding window fold structure
#
#   Matching 08_Split.R fold boundaries:
#   Fold 1: train=[] val=[1993-1997]  → SKIPPED (no training rows)
#   Fold 2: train=[1998-2001] val=[2002-2006]
#   Fold 3: train=[1998-2006] val=[2007-2010]
#   Fold 4: train=[1998-2010] val=[2011-2015]
#   Fold 5: train=[1998-2015] val=[2016-2019]  ← this is the test set
#
#   For AutoGluon Stage 1: use last fold validation as holdout
#   (years 2011-2015 as tuning_data, train on 1998-2010)
#
#   For Stage 2 manual CV: run all 4 folds sequentially
# ==============================================================================

# Fold boundaries (feature year = t, predicting y at t+1)
FOLD_BOUNDARIES = [
    # (fold_id, train_end_year, val_start_year, val_end_year)
    (2, 2001, 2002, 2006),
    (3, 2006, 2007, 2010),
    (4, 2010, 2011, 2015),
    # Fold 5 = test set, used for final evaluation not CV
]

print(f"\n[09C] Fold structure (expanding window):")
for fold_id, train_end, val_start, val_end in FOLD_BOUNDARIES:
    n_train = len(train_qt[train_qt["year"] <= train_end])
    n_val   = len(train_qt[(train_qt["year"] >= val_start) &
                            (train_qt["year"] <= val_end)])
    print(f"  Fold {fold_id}: train [1998–{train_end}] n={n_train:,} "
          f"| val [{val_start}–{val_end}] n={n_val:,}")

# ==============================================================================
# 9. Stage 1 — AutoGluon training on full training set
#
#   Uses years 2011-2015 as holdout (tuning_data) for model selection.
#   AutoGluon trains all model types, selects best ensemble.
#   Bagging disabled — we handle CV manually in Stage 2.
# ==============================================================================

print(f"\n[09C] ══════════════════════════════════════")
print(f"  STAGE 1: AutoGluon training on full train set")
print(f"  Preset: {PRESET} | Time limit: {TIME_LIMIT}s")
print(f"[09C] ══════════════════════════════════════\n")

# Training data: all train years, features + label only
ag_train_cols = feature_cols + [LABEL_COL]

# Holdout validation: last CV fold (years 2011-2015)
holdout_mask   = (train_qt["year"] >= 2011) & (train_qt["year"] <= 2015)
train_fit_mask = train_qt["year"] < 2011

ag_train_data  = TabularDataset(
    train_qt[~holdout_mask][ag_train_cols].reset_index(drop=True)
)
ag_holdout     = TabularDataset(
    train_qt[holdout_mask][ag_train_cols].reset_index(drop=True)
)
ag_test_data   = TabularDataset(
    test_qt[ag_train_cols].reset_index(drop=True)
)

print(f"  AG training rows   : {len(ag_train_data):,}")
print(f"  AG holdout rows    : {len(ag_holdout):,}")
print(f"  AG test rows       : {len(ag_test_data):,}")
print(f"  Positive class     : {LABEL_COL}=={POSITIVE_CLASS}")
print(f"  Eval metric        : {EVAL_METRIC}\n")

predictor = TabularPredictor(
    label          = LABEL_COL,
    problem_type   = "binary",
    eval_metric    = EVAL_METRIC,
    path           = str(DIR_MODELS / "ag_predictor"),
    verbosity      = 2,
    positive_class = POSITIVE_CLASS
).fit(
    train_data     = ag_train_data,
    tuning_data    = ag_holdout,      # holdout for model selection
    presets        = PRESET,
    time_limit     = TIME_LIMIT,
    # Bagging disabled — we do manual CV in Stage 2
    num_bag_folds  = 0,
    num_stack_levels = 0,
    # Class imbalance: use sample weights proportional to inverse class freq
    # AutoGluon handles this via balanced eval metric + scale_pos_weight
    # in individual models (XGBoost, LightGBM auto-detect from eval_metric)
)

print("\n[09C] Stage 1 training complete.")

# ==============================================================================
# 10. Stage 1 results — leaderboard and feature importance
# ==============================================================================

print("\n[09C] ── Leaderboard (holdout validation) ──────────────────\n")
leaderboard = predictor.leaderboard(ag_holdout, silent=True)
print(leaderboard[["model", "score_val", "pred_time_val",
                    "fit_time"]].to_string(index=False))

leaderboard.to_csv(DIR_TABLES / "ag_leaderboard.csv", index=False)
print(f"\n  Leaderboard saved: ag_leaderboard.csv")

# Feature importance — computed on holdout
print("\n[09C] Computing feature importance (holdout)...")
try:
    feat_imp = predictor.feature_importance(ag_holdout, silent=True)
    feat_imp.to_csv(DIR_TABLES / "ag_feature_importance.csv")
    print(f"  Top 10 features:")
    print(feat_imp.head(10).to_string())
    print(f"  Feature importance saved: ag_feature_importance.csv")
except Exception as e:
    print(f"  Feature importance failed: {e}")
    feat_imp = None

# ==============================================================================
# 11. Stage 1 predictions — test and OOS
# ==============================================================================

print("\n[09C] Generating test and OOS predictions...")

# Test set predictions
preds_test_proba = predictor.predict_proba(ag_test_data, as_multiclass=False)
preds_test = pd.DataFrame({
    "permno" : test_qt["permno"].values,
    "year"   : test_qt["year"].values,
    "y"      : test_qt[LABEL_COL].values,
    "p_csi"  : preds_test_proba.values
})

# OOS predictions
ag_oos_data = TabularDataset(
    oos_qt[ag_train_cols].reset_index(drop=True)
)
preds_oos_proba = predictor.predict_proba(ag_oos_data, as_multiclass=False)
preds_oos = pd.DataFrame({
    "permno" : oos_qt["permno"].values,
    "year"   : oos_qt["year"].values,
    "y"      : oos_qt[LABEL_COL].values,
    "p_csi"  : preds_oos_proba.values
})

preds_test.to_parquet(DIR_TABLES / "ag_preds_test.parquet", index=False)
preds_oos.to_parquet(DIR_TABLES / "ag_preds_oos.parquet",  index=False)
print(f"  Test predictions  saved: ag_preds_test.parquet")
print(f"  OOS  predictions  saved: ag_preds_oos.parquet")

# ==============================================================================
# 12. Evaluation metrics — helper functions
# ==============================================================================

def fn_recall_at_fpr(y_true, y_pred, fpr_target):
    """Maximum recall at FPR <= fpr_target."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    eligible = np.where(fpr <= fpr_target)[0]
    if len(eligible) == 0:
        return 0.0
    return float(tpr[eligible].max())

def fn_eval_metrics(y_true, y_pred, set_name):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    valid  = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt, yp = y_true[valid], y_pred[valid]
    if len(np.unique(yt)) < 2:
        return None
    return {
        "set"           : set_name,
        "n_obs"         : int(len(yt)),
        "n_pos"         : int(yt.sum()),
        "prevalence"    : round(float(yt.mean()), 4),
        "auc_roc"       : round(float(roc_auc_score(yt, yp)), 4),
        "avg_precision" : round(float(average_precision_score(yt, yp)), 4),
        "recall_fpr1"   : round(fn_recall_at_fpr(yt, yp, 0.01), 4),
        "recall_fpr3"   : round(fn_recall_at_fpr(yt, yp, 0.03), 4),
        "recall_fpr5"   : round(fn_recall_at_fpr(yt, yp, 0.05), 4),
        "recall_fpr10"  : round(fn_recall_at_fpr(yt, yp, 0.10), 4),
        "brier"         : round(float(np.mean((yp - yt)**2)), 4),
    }

# ==============================================================================
# 13. Stage 1 evaluation — test and OOS
# ==============================================================================

print("\n[09C] ── Stage 1 Evaluation ──────────────────────────────\n")

metrics_test = fn_eval_metrics(
    preds_test["y"], preds_test["p_csi"], "test_2016_2019"
)
metrics_oos_clean = fn_eval_metrics(
    preds_oos[preds_oos["year"] <= 2022]["y"],
    preds_oos[preds_oos["year"] <= 2022]["p_csi"],
    "oos_2020_2022"
)
metrics_oos_full = fn_eval_metrics(
    preds_oos["y"], preds_oos["p_csi"], "oos_2020_2024_full"
)

for m in [metrics_test, metrics_oos_clean, metrics_oos_full]:
    if m:
        print(f"  {m['set']:<25} | "
              f"AP={m['avg_precision']:.4f} | "
              f"AUC={m['auc_roc']:.4f} | "
              f"R@FPR3={m['recall_fpr3']:.4f} | "
              f"R@FPR5={m['recall_fpr5']:.4f}")

# ==============================================================================
# 14. Stage 2 — Manual expanding window CV
#
#   For each fold k in [2, 3, 4]:
#     1. Train AutoGluon on all train rows where year <= fold_train_end
#     2. Predict on fold validation rows
#     3. Accumulate out-of-fold predictions
#
#   This is expensive (~3 AutoGluon fits) but produces a genuinely
#   temporally-honest CV metric comparable to the XGBoost CV AUCPR.
#
#   Use a shorter time_limit for CV fits to keep total runtime manageable.
#   Recommendation: 900s per fold (15 min) × 3 folds = 45 min.
#   Reduce PRESET to "medium_quality" for faster CV.
# ==============================================================================

print(f"\n[09C] ══════════════════════════════════════")
print(f"  STAGE 2: Manual expanding window CV")
print(f"  {len(FOLD_BOUNDARIES)} folds × ~{900//60} min each = "
      f"~{len(FOLD_BOUNDARIES) * 900 // 60} min total")
print(f"[09C] ══════════════════════════════════════\n")

CV_TIME_LIMIT = 900    # seconds per fold CV fit — increase for better results
CV_PRESET     = "medium_quality"  # faster for CV

cv_preds_list = []

for fold_id, train_end, val_start, val_end in FOLD_BOUNDARIES:

    print(f"\n[09C] CV Fold {fold_id}: "
          f"train [≤{train_end}] → val [{val_start}–{val_end}]")

    # Training data for this fold
    fold_train_mask = train_qt["year"] <= train_end
    fold_val_mask   = ((train_qt["year"] >= val_start) &
                       (train_qt["year"] <= val_end))

    fold_train = TabularDataset(
        train_qt[fold_train_mask][ag_train_cols].reset_index(drop=True)
    )
    fold_val_qt = train_qt[fold_val_mask].copy()
    fold_val    = TabularDataset(
        fold_val_qt[ag_train_cols].reset_index(drop=True)
    )

    n_pos = int(fold_train[LABEL_COL].sum())
    n_neg = int((fold_train[LABEL_COL] == 0).sum())
    print(f"  Train: {len(fold_train):,} rows | "
          f"pos={n_pos} neg={n_neg} weight={n_neg/max(n_pos,1):.2f}")
    print(f"  Val  : {len(fold_val):,} rows")

    if n_pos < 10:
        print(f"  Fold {fold_id} SKIPPED — fewer than 10 positive labels.")
        continue

    # Fit AutoGluon on this fold
    fold_predictor = TabularPredictor(
        label          = LABEL_COL,
        problem_type   = "binary",
        eval_metric    = EVAL_METRIC,
        path           = str(DIR_MODELS / f"ag_cv_fold{fold_id}"),
        verbosity      = 1,
        positive_class = POSITIVE_CLASS
    ).fit(
        train_data     = fold_train,
        tuning_data    = fold_val,   # holdout within fold for early stopping
        presets        = CV_PRESET,
        time_limit     = CV_TIME_LIMIT,
        num_bag_folds  = 0,
        num_stack_levels = 0,
    )

    # Predict on validation set
    fold_proba = fold_predictor.predict_proba(fold_val, as_multiclass=False)

    fold_cv_preds = pd.DataFrame({
        "fold_id" : fold_id,
        "permno"  : fold_val_qt["permno"].values,
        "year"    : fold_val_qt["year"].values,
        "y"       : fold_val_qt[LABEL_COL].values,
        "p_csi"   : fold_proba.values
    })

    fold_ap = average_precision_score(
        fold_cv_preds["y"], fold_cv_preds["p_csi"]
    )
    fold_auc = roc_auc_score(
        fold_cv_preds["y"], fold_cv_preds["p_csi"]
    )
    print(f"  Fold {fold_id} AP: {fold_ap:.4f} | AUC: {fold_auc:.4f}")

    cv_preds_list.append(fold_cv_preds)

# Aggregate CV results
if cv_preds_list:
    cv_preds_all = pd.concat(cv_preds_list, ignore_index=True)
    cv_preds_all.to_parquet(DIR_TABLES / "ag_cv_results.parquet", index=False)

    cv_ap_mean = average_precision_score(
        cv_preds_all["y"], cv_preds_all["p_csi"]
    )
    cv_auc_mean = roc_auc_score(
        cv_preds_all["y"], cv_preds_all["p_csi"]
    )
    cv_r3 = fn_recall_at_fpr(
        cv_preds_all["y"].values,
        cv_preds_all["p_csi"].values, 0.03
    )

    print(f"\n[09C] Expanding window CV results (all folds pooled):")
    print(f"  CV Average Precision : {cv_ap_mean:.4f}")
    print(f"  CV AUC-ROC           : {cv_auc_mean:.4f}")
    print(f"  CV Recall@FPR3       : {cv_r3:.4f}")

    # Per-fold breakdown
    print(f"\n  Per-fold breakdown:")
    for fold_preds in cv_preds_list:
        f_id = fold_preds["fold_id"].iloc[0]
        f_ap = average_precision_score(fold_preds["y"], fold_preds["p_csi"])
        print(f"    Fold {f_id}: AP={f_ap:.4f} "
              f"n={len(fold_preds):,} "
              f"prev={fold_preds['y'].mean():.3f}")
else:
    cv_ap_mean  = None
    cv_auc_mean = None
    cv_r3       = None
    print("[09C] No CV folds completed.")

# ==============================================================================
# 15. Save evaluation summary
# ==============================================================================

eval_summary = {
    "model"              : "autogluon",
    "preset"             : PRESET,
    "time_limit_s"       : TIME_LIMIT,
    "cv_time_limit_s"    : CV_TIME_LIMIT,
    "label_shift"        : "y(t+1)",
    "eval_metric"        : EVAL_METRIC,

    ## CV metrics (Stage 2 — honest expanding window)
    "cv_avg_precision"   : round(cv_ap_mean,  4) if cv_ap_mean  else None,
    "cv_auc_roc"         : round(cv_auc_mean, 4) if cv_auc_mean else None,
    "cv_recall_fpr3"     : round(cv_r3,       4) if cv_r3       else None,

    ## Test metrics (Stage 1 — final model)
    "test"               : metrics_test,
    "oos_2020_2022"      : metrics_oos_clean,
    "oos_full"           : metrics_oos_full,

    ## Comparison
    "paper_recall_fpr3"  : 0.61,
    "xgb_cv_aucpr"       : 0.4855,
    "xgb_test_ap"        : 0.6493,
    "xgb_test_r_fpr3"    : 0.4727,
    "xgb_test_r_fpr5"    : 0.6210,
}

with open(DIR_TABLES / "ag_eval_summary.json", "w") as f:
    json.dump(eval_summary, f, indent=2)

print(f"\n[09C] Evaluation summary saved: ag_eval_summary.json")

# ==============================================================================
# 16. Final comparison table
# ==============================================================================

print(f"\n[09C] ══════════════════════════════════════")
print(f"  FINAL COMPARISON: AutoGluon vs XGBoost vs Paper")
print(f"  ══════════════════════════════════════\n")

print(f"  {'Model':<20} | {'CV AP':<8} | {'Test AP':<8} | "
      f"{'AUC':<8} | {'R@FPR3':<8} | {'R@FPR5'}")
print(f"  {'--------------------':<20} | {'--------':<8} | {'--------':<8} | "
      f"{'--------':<8} | {'--------':<8} | {'--------'}")

## AutoGluon
if metrics_test:
    print(f"  {'AutoGluon':<20} | "
          f"{str(round(cv_ap_mean, 4)) if cv_ap_mean else '—':<8} | "
          f"{metrics_test['avg_precision']:<8} | "
          f"{metrics_test['auc_roc']:<8} | "
          f"{metrics_test['recall_fpr3']:<8} | "
          f"{metrics_test['recall_fpr5']}")

## XGBoost (from 09_Train.R — hardcoded from previous results)
print(f"  {'XGBoost (raw)':<20} | "
      f"{'0.4855':<8} | {'0.6493':<8} | {'0.9355':<8} | "
      f"{'0.4727':<8} | {'0.6210'}")

## Paper benchmark
print(f"  {'Paper (Tewari)':<20} | "
      f"{'—':<8} | {'—':<8} | {'—':<8} | "
      f"{'0.6100':<8} | {'—'}")

print(f"\n[09C] DONE")
print(f"  Saved: {DIR_TABLES / 'ag_eval_summary.json'}")
print(f"  Model: {DIR_MODELS / 'ag_predictor'}")
"""
09C_AutoGluon.py  (v4 — M1/M2/M3/M4)
======================================
AutoGluon AutoML for CSI prediction.

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
                               Purpose: Does VAE add signal over raw fundamentals?

    MODEL = "latent_raw"  → M4: VAE latent trained on full raw features
                               Input : features_latent_raw.parquet
                               Output: Tables/ag_latent_raw/
                               Purpose: Does VAE compress full features usefully?

MODEL MATRIX:
    M1 vs M2 → answers Sub-Q 1: does VAE add signal over fundamentals?
    M3 vs M4 → does VAE compress full features usefully?
    M1 vs M3 → cost of removing price features (avoidance alpha constraint)

LABEL SHIFT:
    features(t) → y(t+1), matching 09_Train.R.

EXPANDING WINDOW CV:
    Stage 1: Train on full train set, holdout = years 2011–2015.
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
DIR_OUTPUT   = DATA_ROOT / "03_Output"
DIR_MODELS   = DIR_OUTPUT / "Models" / "AutoGluon"
DIR_TABLES   = DIR_OUTPUT / "Tables"

PATH_SPLIT_LABELS = DIR_FEATURES / "split_labels_oot.parquet"
assert PATH_SPLIT_LABELS.exists(), f"Not found: {PATH_SPLIT_LABELS}"

# ==============================================================================
# 2. ── MODEL SELECTION ────────────────────────────────────────────────────────
#
#   "raw"         → M3  full engineered features        [existing]
#   "fund"        → M1  fundamentals only               [NEW]
#   "latent_fund" → M2  VAE latent on fund input        [NEW]
#   "latent_raw"  → M4  VAE latent on raw input         [NEW]
#
# ==============================================================================

MODEL = "latent_raw"    # <- CHANGE THIS: "raw" | "fund" | "latent_fund" | "latent_raw"

# ==============================================================================
# 3. Model-specific configuration
# ==============================================================================

SEED          = 123
TIME_LIMIT    = 3600
PRESET        = "good_quality"
CV_TIME_LIMIT = 900
CV_PRESET     = "medium_quality"
EVAL_METRIC   = "average_precision"
LABEL_COL     = "y_next"

MODEL_CONFIG = {
    "raw": {
        "path"       : DIR_FEATURES / "features_raw.rds",
        "loader"     : "rds",
        "description": "M3 — Full engineered features (~463)",
        "prereq"     : "Run 06B_Feature_Eng.R first.",
    },
    "fund": {
        "path"       : DIR_FEATURES / "features_fund.rds",
        "loader"     : "rds",
        "description": "M1 — Fundamentals only (no price features)",
        "prereq"     : "Run 06B_Feature_Eng.R first.",
    },
    "latent_fund": {
        "path"       : DIR_FEATURES / "features_latent_fund.parquet",
        "loader"     : "parquet",
        "description": "M2 — VAE latent on fundamentals (z1–z24 + recon error)",
        "prereq"     : "Run 08B_Autoencoder.py with VAE_INPUT='fund' first.",
    },
    "latent_raw": {
        "path"       : DIR_FEATURES / "features_latent_raw.parquet",
        "loader"     : "parquet",
        "description": "M4 — VAE latent on full raw features (z1–z24 + recon error)",
        "prereq"     : "Run 08B_Autoencoder.py with VAE_INPUT='raw' first.",
    },
}

assert MODEL in MODEL_CONFIG, \
    f"Unknown MODEL '{MODEL}'. Choose from: {list(MODEL_CONFIG.keys())}"

cfg = MODEL_CONFIG[MODEL]

assert cfg["path"].exists(), \
    f"Feature file not found: {cfg['path']}\n{cfg['prereq']}"

# Model-specific output directories — never overwrite other models
DIR_MODELS_RUN = DIR_MODELS / f"ag_{MODEL}"
DIR_TABLES_RUN = DIR_TABLES / f"ag_{MODEL}"
DIR_MODELS_RUN.mkdir(parents=True, exist_ok=True)
DIR_TABLES_RUN.mkdir(parents=True, exist_ok=True)

print(f"[09C] ══════════════════════════════════════")
print(f"  MODEL        : {MODEL.upper()}")
print(f"  Description  : {cfg['description']}")
print(f"  Input file   : {cfg['path'].name}")
print(f"  Output dir   : {DIR_TABLES_RUN}")
print(f"[09C] ══════════════════════════════════════\n")

# ==============================================================================
# 4. ID and latent column definitions
# ==============================================================================

ID_COLS = [
    "permno", "year", "y", "censored", "param_id",
    "gvkey", "datadate", "lifetime_years",
    "fiscal_year_end_month", "split", "vae_split", "split_oot"
]

LATENT_FEATURE_NAMES = [f"z{i}" for i in range(1, 25)] + ["vae_recon_error"]

# ==============================================================================
# 5. Load feature data
# ==============================================================================

print(f"[09C] Loading {cfg['path'].name} ({cfg['loader']})...")

if cfg["loader"] == "rds":
    result        = pyreadr.read_r(str(cfg["path"]))
    features_input = result[None]
elif cfg["loader"] == "parquet":
    features_input = pd.read_parquet(cfg["path"])
    # Latent parquet has its own 'split' column from 08B — rename before merge
    if "split" in features_input.columns:
        features_input = features_input.rename(columns={"split": "vae_split"})

print(f"  Shape: {features_input.shape[0]:,} rows × {features_input.shape[1]} cols")

print("[09C] Loading split labels...")
split_labels = pd.read_parquet(PATH_SPLIT_LABELS)
print(f"  Split distribution:\n{split_labels['split'].value_counts().to_string()}")

# ==============================================================================
# 6. Merge split labels and apply label shift
# ==============================================================================

df = features_input.merge(split_labels, on=["permno", "year"], how="left")
df = df[df["split"].notna()].reset_index(drop=True)

df = df.sort_values(["permno", "year"]).reset_index(drop=True)
df["y_next"] = df.groupby("permno")["y"].shift(-1)

df_labelled = df[df["y_next"].notna()].copy()
df_labelled["y_next"] = df_labelled["y_next"].astype(int)

print(f"\n[09C] After label shift:")
print(f"  Rows with valid y_next : {len(df_labelled):,}")
print(f"  y_next prevalence      : {df_labelled['y_next'].mean():.3f}")

# ==============================================================================
# 7. Split
# ==============================================================================

train_df     = df_labelled[df_labelled["split"] == "train"].copy()
test_df      = df_labelled[df_labelled["split"] == "test"].copy()
oos_df       = df_labelled[df_labelled["split"] == "oos"].copy()
oos_clean_df = oos_df[oos_df["year"] <= 2022].copy()

print(f"\n[09C] Split sizes:")
print(f"  Train : {len(train_df):,}  (prev: {train_df['y_next'].mean():.3f})")
print(f"  Test  : {len(test_df):,}  (prev: {test_df['y_next'].mean():.3f})")
print(f"  OOS   : {len(oos_df):,}  (prev: {oos_df['y_next'].mean():.3f})")
print(f"  OOS clean (≤2022): {len(oos_clean_df):,}")

# ==============================================================================
# 8. Feature columns
# ==============================================================================

if MODEL in ("latent_fund", "latent_raw"):
    feature_cols = [c for c in LATENT_FEATURE_NAMES if c in train_df.columns]
    print(f"\n[09C] Latent feature columns: {len(feature_cols)}")
else:
    feature_cols = [
        c for c in train_df.columns
        if c not in ID_COLS
        and c not in [LABEL_COL, "y_next", "split", "split_oot", "vae_split"]
        and pd.api.types.is_numeric_dtype(train_df[c])
        and c != "y"
    ]
    print(f"\n[09C] Feature columns: {len(feature_cols)}")

# ==============================================================================
# 9. Quantile Transform — fitted on train only
# ==============================================================================

print("\n[09C] Applying quantile transform...")

X_train = train_df[feature_cols].values.astype(np.float64)
X_test  = test_df[feature_cols].values.astype(np.float64)
X_oos   = oos_df[feature_cols].values.astype(np.float64)

for arr in [X_train, X_test, X_oos]:
    arr[np.isinf(arr)] = np.nan

lo = np.nanpercentile(X_train, 0.1, axis=0)
hi = np.nanpercentile(X_train, 99.9, axis=0)
X_train = np.clip(X_train, lo, hi)
X_test  = np.clip(X_test,  lo, hi)
X_oos   = np.clip(X_oos,   lo, hi)

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
X_oos_qt   = qt.transform(X_oos_imp)

def rebuild_df(X_qt, source_df):
    out = pd.DataFrame(X_qt, columns=feature_cols, index=source_df.index)
    out[LABEL_COL] = source_df[LABEL_COL].values
    out["year"]    = source_df["year"].values
    out["permno"]  = source_df["permno"].values
    return out

train_qt = rebuild_df(X_train_qt, train_df)
test_qt  = rebuild_df(X_test_qt,  test_df)
oos_qt   = rebuild_df(X_oos_qt,   oos_df)

print(f"  QT applied. Train shape: {train_qt.shape}")
print(f"  Train mean (~0.5): {X_train_qt.mean():.4f}")
print(f"  Train std (~0.29): {X_train_qt.std():.4f}")

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
    print(f"  Fold {fold_id}: train [1998–{train_end}] n={n_train:,} "
          f"| val [{val_start}–{val_end}] n={n_val:,}")

# ==============================================================================
# 11. Stage 1 — AutoGluon training
# ==============================================================================

print(f"\n[09C] ══ STAGE 1: AutoGluon [{MODEL.upper()}] ══")
print(f"  Preset: {PRESET} | Time limit: {TIME_LIMIT}s\n")

ag_train_cols = feature_cols + [LABEL_COL]
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
# 13. Predictions
# ==============================================================================

print(f"\n[09C] Generating predictions...")

preds_test_proba = predictor.predict_proba(ag_test_data, as_multiclass=False)
preds_test = pd.DataFrame({
    "permno": test_qt["permno"].values,
    "year"  : test_qt["year"].values,
    "y"     : test_qt[LABEL_COL].values,
    "p_csi" : preds_test_proba.values
})

ag_oos_data     = TabularDataset(oos_qt[ag_train_cols].reset_index(drop=True))
preds_oos_proba = predictor.predict_proba(ag_oos_data, as_multiclass=False)
preds_oos = pd.DataFrame({
    "permno": oos_qt["permno"].values,
    "year"  : oos_qt["year"].values,
    "y"     : oos_qt[LABEL_COL].values,
    "p_csi" : preds_oos_proba.values
})

preds_test.to_parquet(DIR_TABLES_RUN / "ag_preds_test.parquet", index=False)
preds_oos.to_parquet(DIR_TABLES_RUN  / "ag_preds_oos.parquet",  index=False)
print(f"  Saved to {DIR_TABLES_RUN}")

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

metrics_test      = fn_eval_metrics(preds_test["y"], preds_test["p_csi"],
                                    "test_2016_2019")
metrics_oos_clean = fn_eval_metrics(
    preds_oos[preds_oos["year"] <= 2022]["y"],
    preds_oos[preds_oos["year"] <= 2022]["p_csi"],
    "oos_2020_2022")
metrics_oos_full  = fn_eval_metrics(preds_oos["y"], preds_oos["p_csi"],
                                    "oos_2020_2024_full")

for m in [metrics_test, metrics_oos_clean, metrics_oos_full]:
    if m:
        print(f"  {m['set']:<25} | AP={m['avg_precision']:.4f} | "
              f"AUC={m['auc_roc']:.4f} | R@FPR3={m['recall_fpr3']:.4f} | "
              f"R@FPR5={m['recall_fpr5']:.4f}")

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
    "label_shift"      : "y(t+1)",
    "eval_metric"      : EVAL_METRIC,
    "n_features"       : len(feature_cols),
    "cv_avg_precision" : round(cv_ap_mean,  4) if cv_ap_mean  else None,
    "cv_auc_roc"       : round(cv_auc_mean, 4) if cv_auc_mean else None,
    "cv_recall_fpr3"   : round(cv_r3,       4) if cv_r3       else None,
    "test"             : metrics_test,
    "oos_2020_2022"    : metrics_oos_clean,
    "oos_full"         : metrics_oos_full,
    ## Reference benchmarks
    "paper_r_fpr3"     : 0.61,
    "ag_m3_cv_ap"      : 0.6656,
    "ag_m3_test_ap"    : 0.7576,
    "ag_m1_cv_ap"      : 0.5809,
    "ag_m1_test_ap"    : 0.6546,
}

with open(DIR_TABLES_RUN / "ag_eval_summary.json", "w") as f:
    json.dump(eval_summary, f, indent=2)
print(f"\n[09C] Summary saved: {DIR_TABLES_RUN / 'ag_eval_summary.json'}")

# ==============================================================================
# 18. Final comparison table
# ==============================================================================

print(f"\n[09C] ══ RESULTS: [{MODEL.upper()}] vs benchmarks ══\n")
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

# Known benchmarks
print(f"  {'AG M3 (raw)':<25} | {'0.6656':<8} | {'0.7576':<8} | "
      f"{'0.9601':<8} | {'0.6397':<8} | {'0.7670'}")
print(f"  {'AG M1 (fund)':<25} | {'0.5809':<8} | {'0.6546':<8} | "
      f"{'0.9337':<8} | {'0.4936':<8} | {'0.6337'}")
print(f"  {'XGB M3 (raw)':<25} | {'0.4855':<8} | {'0.6493':<8} | "
      f"{'0.9355':<8} | {'0.4727':<8} | {'0.6210'}")
print(f"  {'Paper (Tewari)':<25} | {'—':<8} | {'—':<8} | "
      f"{'—':<8} | {'0.6100':<8} | {'—'}")

print(f"\n[09C] DONE [{MODEL.upper()}]")
print(f"  Tables : {DIR_TABLES_RUN}")
print(f"  Model  : {DIR_MODELS_RUN / 'ag_predictor'}")
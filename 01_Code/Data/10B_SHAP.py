"""
10B_SHAP.py
===========
Companion to 10_Evaluate.R.
Computes SHAP values and Partial Dependence data from AutoGluon predictors.
Outputs parquet files that 10_Evaluate.R loads for plotting.

OUTPUTS per model:
  DIR_TABLES/ag_{MODEL}/shap_values.parquet       — SHAP matrix (n_obs × n_features)
  DIR_TABLES/ag_{MODEL}/shap_meta.parquet         — permno, year, y, p_csi, split
  DIR_TABLES/ag_{MODEL}/shap_importance.parquet   — mean |SHAP| per feature (ranked)
  DIR_TABLES/ag_{MODEL}/pdp_1d.parquet            — 1D PDP for top 10 features
  DIR_TABLES/ag_{MODEL}/pdp_2d.parquet            — 2D PDP for top 2 feature pairs

DESIGN:
  AutoGluon ensembles multiple models. For SHAP we extract the best-performing
  individual model (by holdout score) that supports SHAP natively — typically
  XGBoost or LightGBM. If neither is available, falls back to the weighted
  ensemble via TreeExplainer on the best tree model.

  PDP is computed by marginalising over the full test set (ICE-style mean),
  using the full AutoGluon predictor (not just the best base model) so the
  PDP reflects the actual deployed predictions.

RUN ORDER:
  After 09C_AutoGluon.py completes for each model.
  Before 10_Evaluate.R.

USAGE:
  Set MODEL to any of the 12 model keys. Run once per model.
  Phase 1 models (no VAE): fund, raw, bucket, bucket_raw, structural, structural_raw
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr
import shap

warnings.filterwarnings("ignore")

try:
    from autogluon.tabular import TabularPredictor
except ImportError:
    raise ImportError("AutoGluon not installed.")

# ==============================================================================
# 1. Paths
# ==============================================================================

if os.name == "nt":
    DATA_ROOT = Path(r"C:\Users\Tristan Leiter\Documents\MT")
else:
    DATA_ROOT = Path("/workspace/MT")

DIR_DATA     = DATA_ROOT / "02_Data"
DIR_FEATURES = DIR_DATA  / "Features"
DIR_LABELS   = DIR_DATA  / "Labels"
DIR_OUTPUT   = DATA_ROOT / "03_Output"
DIR_MODELS   = DIR_OUTPUT / "Models" / "AutoGluon"
DIR_TABLES   = DIR_OUTPUT / "Tables"

PATH_SPLIT_LABELS      = DIR_FEATURES / "split_labels_oot.parquet"
PATH_LABELS_BUCKET     = DIR_LABELS   / "labels_bucket.rds"
PATH_LABELS_STRUCTURAL = DIR_LABELS   / "labels_structural.rds"

# ==============================================================================
# 2. Model selection — must match 09C_AutoGluon.py
# ==============================================================================

MODEL = "fund"   # <- CHANGE THIS (same keys as 09C)

# ==============================================================================
# 3. Config — mirrors 09C
# ==============================================================================

SEED           = 123
TRAIN_END_YEAR = 2015
TEST_START     = 2016
TEST_END       = 2019
OOS_START      = 2020

N_PDP_POINTS   = 50    ## grid points per feature for 1D PDP
N_PDP_2D       = 20    ## grid points per axis for 2D PDP
TOP_K_FEATURES = 10    ## number of features for PDP
TOP_K_2D_PAIRS = 3     ## number of feature pairs for 2D PDP
N_SHAP_SAMPLE  = 2000  ## max rows for SHAP (subsampled if test set larger)

LABEL_CSI        = "y_next"
LABEL_BUCKET     = "y_loser"
LABEL_STRUCTURAL = "y_structural"

MODEL_CONFIG = {
    "fund"                    : {"label_col": LABEL_CSI,        "is_bucket": False, "feat": "features_fund.rds",             "loader": "rds"},
    "latent_fund"             : {"label_col": LABEL_CSI,        "is_bucket": False, "feat": "features_latent_fund.parquet",  "loader": "parquet"},
    "raw"                     : {"label_col": LABEL_CSI,        "is_bucket": False, "feat": "features_raw.rds",              "loader": "rds"},
    "latent_raw"              : {"label_col": LABEL_CSI,        "is_bucket": False, "feat": "features_latent_raw.parquet",   "loader": "parquet"},
    "bucket"                  : {"label_col": LABEL_BUCKET,     "is_bucket": True,  "feat": "features_fund.rds",             "loader": "rds"},
    "bucket_latent_fund"      : {"label_col": LABEL_BUCKET,     "is_bucket": True,  "feat": "features_latent_fund.parquet",  "loader": "parquet"},
    "bucket_raw"              : {"label_col": LABEL_BUCKET,     "is_bucket": True,  "feat": "features_raw.rds",              "loader": "rds"},
    "bucket_latent_raw"       : {"label_col": LABEL_BUCKET,     "is_bucket": True,  "feat": "features_latent_raw.parquet",   "loader": "parquet"},
    "structural"              : {"label_col": LABEL_STRUCTURAL, "is_bucket": True,  "feat": "features_fund.rds",             "loader": "rds"},
    "structural_latent_fund"  : {"label_col": LABEL_STRUCTURAL, "is_bucket": True,  "feat": "features_latent_fund.parquet",  "loader": "parquet"},
    "structural_raw"          : {"label_col": LABEL_STRUCTURAL, "is_bucket": True,  "feat": "features_raw.rds",              "loader": "rds"},
    "structural_latent_raw"   : {"label_col": LABEL_STRUCTURAL, "is_bucket": True,  "feat": "features_latent_raw.parquet",   "loader": "parquet"},
}

cfg       = MODEL_CONFIG[MODEL]
LABEL_COL = cfg["label_col"]
IS_BUCKET = cfg["is_bucket"]
DIR_OUT   = DIR_TABLES / f"ag_{MODEL}"

print(f"\n[10B] ══════════════════════════════════════")
print(f"  MODEL    : {MODEL}")
print(f"  Label    : {LABEL_COL}")
print(f"  Output   : {DIR_OUT}")
print(f"[10B] ══════════════════════════════════════\n")

# ==============================================================================
# 4. Load test set predictions (from 09C output)
# ==============================================================================

preds_path = DIR_OUT / "ag_preds_test_eval.parquet"
assert preds_path.exists(), f"Run 09C for MODEL='{MODEL}' first.\n{preds_path}"

preds_df = pd.read_parquet(preds_path)
print(f"[10B] Test predictions loaded: {len(preds_df):,} rows")

# ==============================================================================
# 5. Reconstruct test feature matrix
# ==============================================================================

print(f"[10B] Reconstructing test feature matrix...")

feat_path = DIR_FEATURES / cfg["feat"]
if cfg["loader"] == "rds":
    features_input = pyreadr.read_r(str(feat_path))[None]
else:
    features_input = pd.read_parquet(feat_path)
    if "split" in features_input.columns:
        features_input = features_input.rename(columns={"split": "vae_split"})

## Merge to get test rows with features
if not IS_BUCKET:
    split_labels = pd.read_parquet(PATH_SPLIT_LABELS)
    df = features_input.merge(split_labels, on=["permno", "year"], how="left")
    df = df[df["split"] == "test"].copy()
    df["y_next"] = df.groupby("permno")["y"].shift(-1)
    df = df[df["y_next"].notna() & ~df["eval_split"].isin({"train_boundary", "test_boundary"})].copy()
    df["y_next"] = df["y_next"].astype(int)
else:
    lbl_path = PATH_LABELS_STRUCTURAL if LABEL_COL == LABEL_STRUCTURAL else PATH_LABELS_BUCKET
    lbl = pyreadr.read_r(str(lbl_path))[None]
    lbl = lbl[lbl[LABEL_COL].notna()].copy()
    lbl[LABEL_COL] = lbl[LABEL_COL].astype(int)
    df = features_input.merge(lbl[["permno", "year", LABEL_COL]], on=["permno", "year"], how="inner")
    df = df[(df["year"] >= TEST_START) & (df["year"] <= TEST_END)].copy()

print(f"  Test rows: {len(df):,}")

## Identify feature columns (same logic as 09C section 8)
LATENT_COLS = [f"z{i}" for i in range(1, 25)] + ["vae_recon_error"]
ID_COLS = ["permno", "year", "y", "censored", "param_id", "gvkey", "datadate",
           "lifetime_years", "fiscal_year_end_month", "split", "eval_split",
           "vae_split", "split_oot", "y_loser", "y_structural", "fwd_cagr",
           "n_months", "bucket", "year_cat", "y_next"]

if MODEL in ("latent_fund", "latent_raw", "bucket_latent_fund",
             "bucket_latent_raw", "structural_latent_fund", "structural_latent_raw"):
    feature_cols = [c for c in LATENT_COLS if c in df.columns]
else:
    feature_cols = [
        c for c in df.columns
        if c not in ID_COLS
        and pd.api.types.is_numeric_dtype(df[c])
    ]

print(f"  Feature columns: {len(feature_cols)}")

## Align with prediction output by permno+year
test_df = df.merge(preds_df[["permno", "year", "p_csi"]],
                   on=["permno", "year"], how="inner")
print(f"  Aligned test rows: {len(test_df):,}")

X_test    = test_df[feature_cols].copy()
meta_df   = test_df[["permno", "year", LABEL_COL, "p_csi"]].copy()
meta_df   = meta_df.rename(columns={LABEL_COL: "y"})

# ==============================================================================
# 6. Load AutoGluon predictor
# ==============================================================================

predictor_path = DIR_MODELS / f"ag_{MODEL}" / "ag_predictor"
assert predictor_path.exists(), f"AutoGluon predictor not found: {predictor_path}"

print(f"\n[10B] Loading AutoGluon predictor...")
predictor = TabularPredictor.load(str(predictor_path))

## Identify best tree model for SHAP
lb = predictor.leaderboard(silent=True)
print(f"  Leaderboard (top 5):\n{lb.head(5)[['model','score_val']].to_string(index=False)}")

## Prefer XGBoost or LightGBM for SHAP (TreeExplainer)
tree_models = lb[lb["model"].str.contains("XGB|LightGBM|GBM|CatBoost",
                                           case=False, na=False)]
if len(tree_models) > 0:
    best_tree_model_name = tree_models.iloc[0]["model"]
else:
    best_tree_model_name = lb.iloc[0]["model"]

print(f"  Using model for SHAP: {best_tree_model_name}")

# ==============================================================================
# 7. SHAP values — TreeExplainer on best tree model
# ==============================================================================

print(f"\n[10B] Computing SHAP values...")

## Subsample for speed if test set is large
np.random.seed(SEED)
if len(X_test) > N_SHAP_SAMPLE:
    shap_idx = np.random.choice(len(X_test), N_SHAP_SAMPLE, replace=False)
    X_shap   = X_test.iloc[shap_idx].reset_index(drop=True)
    meta_shap = meta_df.iloc[shap_idx].reset_index(drop=True)
    print(f"  Subsampled to {N_SHAP_SAMPLE} rows for SHAP")
else:
    X_shap    = X_test.reset_index(drop=True)
    meta_shap = meta_df.reset_index(drop=True)

## Get the underlying sklearn/xgb model from AutoGluon
try:
    ag_model    = predictor._trainer.load_model(best_tree_model_name)
    skl_model   = ag_model.model   ## underlying sklearn/xgb estimator
    explainer   = shap.TreeExplainer(skl_model)
    shap_vals   = explainer.shap_values(X_shap)

    ## For binary classification some models return list [class0, class1]
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    shap_df = pd.DataFrame(shap_vals, columns=feature_cols)
    print(f"  SHAP matrix: {shap_df.shape}")

except Exception as e:
    print(f"  TreeExplainer failed ({e}), falling back to KernelExplainer (slow)...")
    ## KernelExplainer fallback — very slow, limit to 200 rows
    bg_idx      = np.random.choice(len(X_shap), min(100, len(X_shap)), replace=False)
    background  = X_shap.iloc[bg_idx]

    def ag_predict(X_arr):
        X_df  = pd.DataFrame(X_arr, columns=feature_cols)
        proba = predictor.predict_proba(X_df, as_multiclass=False)
        return proba.values

    explainer = shap.KernelExplainer(ag_predict, background)
    shap_rows = X_shap.iloc[:200]
    shap_vals = explainer.shap_values(shap_rows, nsamples=100)
    shap_df   = pd.DataFrame(shap_vals, columns=feature_cols)
    meta_shap = meta_shap.iloc[:200].reset_index(drop=True)
    print(f"  KernelExplainer SHAP matrix: {shap_df.shape}")

## SHAP importance (mean |SHAP| per feature)
shap_importance = pd.DataFrame({
    "feature"     : feature_cols,
    "mean_abs_shap": np.abs(shap_df.values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
shap_importance["rank"] = range(1, len(shap_importance) + 1)

print(f"\n  Top 10 features by mean |SHAP|:")
print(shap_importance.head(10)[["rank", "feature", "mean_abs_shap"]].to_string(index=False))

## Save
shap_df.to_parquet(DIR_OUT / "shap_values.parquet",     index=False)
meta_shap.to_parquet(DIR_OUT / "shap_meta.parquet",     index=False)
shap_importance.to_parquet(DIR_OUT / "shap_importance.parquet", index=False)
print(f"\n  Saved: shap_values.parquet | shap_meta.parquet | shap_importance.parquet")

# ==============================================================================
# 8. 1D Partial Dependence — top K features, full predictor
# ==============================================================================

print(f"\n[10B] Computing 1D PDPs for top {TOP_K_FEATURES} features...")

top_features = shap_importance["feature"].head(TOP_K_FEATURES).tolist()
pdp_rows     = []

## Use train quantiles as PDP grid (avoids extrapolation)
## Load training predictions to reconstruct train feature distribution
train_preds_path = DIR_OUT / "ag_preds_test.parquet"   ## proxy — use test distribution
X_grid_base = X_test[top_features].copy()

for feat in top_features:
    feat_vals    = X_test[feat].dropna()
    grid         = np.linspace(feat_vals.quantile(0.01),
                               feat_vals.quantile(0.99),
                               N_PDP_POINTS)
    feat_pdp     = []

    for gv in grid:
        X_mod             = X_test[feature_cols].copy()
        X_mod[feat]       = gv
        if IS_BUCKET:
            X_mod_df          = X_mod.copy()
            X_mod_df["year_cat"] = test_df["year"].astype(str).astype("category")
            proba = predictor.predict_proba(X_mod_df, as_multiclass=False)
        else:
            proba = predictor.predict_proba(X_mod, as_multiclass=False)
        feat_pdp.append(float(proba.mean()))

    for i, (gv, pv) in enumerate(zip(grid, feat_pdp)):
        pdp_rows.append({
            "feature"     : feat,
            "feature_val" : gv,
            "pdp_mean"    : pv,
            "grid_idx"    : i,
            "shap_rank"   : shap_importance[shap_importance["feature"] == feat]["rank"].iloc[0]
        })

    print(f"  PDP done: {feat}")

pdp_1d = pd.DataFrame(pdp_rows)
pdp_1d.to_parquet(DIR_OUT / "pdp_1d.parquet", index=False)
print(f"  Saved: pdp_1d.parquet ({len(pdp_1d):,} rows)")

# ==============================================================================
# 9. 2D Partial Dependence — top pairs by SHAP importance
# ==============================================================================

print(f"\n[10B] Computing 2D PDPs for top {TOP_K_2D_PAIRS} feature pairs...")

top2 = shap_importance["feature"].head(TOP_K_FEATURES).tolist()
pairs = [(top2[i], top2[j]) for i in range(len(top2))
         for j in range(i+1, len(top2))][:TOP_K_2D_PAIRS]

pdp2d_rows = []

for (feat1, feat2) in pairs:
    grid1 = np.linspace(X_test[feat1].quantile(0.05), X_test[feat1].quantile(0.95), N_PDP_2D)
    grid2 = np.linspace(X_test[feat2].quantile(0.05), X_test[feat2].quantile(0.95), N_PDP_2D)

    for g1 in grid1:
        for g2 in grid2:
            X_mod         = X_test[feature_cols].copy()
            X_mod[feat1]  = g1
            X_mod[feat2]  = g2
            if IS_BUCKET:
                X_mod_df              = X_mod.copy()
                X_mod_df["year_cat"]  = test_df["year"].astype(str).astype("category")
                proba = predictor.predict_proba(X_mod_df, as_multiclass=False)
            else:
                proba = predictor.predict_proba(X_mod, as_multiclass=False)
            pdp2d_rows.append({
                "feat1"    : feat1,
                "feat2"    : feat2,
                "feat1_val": g1,
                "feat2_val": g2,
                "pdp_mean" : float(proba.mean())
            })
    print(f"  2D PDP done: {feat1} × {feat2}")

pdp_2d = pd.DataFrame(pdp2d_rows)
pdp_2d.to_parquet(DIR_OUT / "pdp_2d.parquet", index=False)
print(f"  Saved: pdp_2d.parquet ({len(pdp_2d):,} rows)")

# ==============================================================================
# 10. Exemplary high-risk firm — highest mean p_csi across test years
# ==============================================================================

print(f"\n[10B] Identifying exemplary high-risk firm...")

mean_risk = preds_df.groupby("permno")["p_csi"].mean().reset_index()
top_firm  = mean_risk.sort_values("p_csi", ascending=False).iloc[0]
top_permno = int(top_firm["permno"])
top_mean   = float(top_firm["p_csi"])

print(f"  Top firm: permno={top_permno}  mean p_csi={top_mean:.4f}")

## Find the year with highest p_csi for this firm
firm_preds = preds_df[preds_df["permno"] == top_permno].sort_values("p_csi", ascending=False)
top_year   = int(firm_preds.iloc[0]["year"])
print(f"  Peak risk year: {top_year}  p_csi={firm_preds.iloc[0]['p_csi']:.4f}")

## Extract features for that single observation
firm_row = X_test[
    (test_df["permno"] == top_permno) &
    (test_df["year"]   == top_year)
]
if len(firm_row) == 0:
    ## Fall back to closest year in test set
    firm_rows_all = test_df[test_df["permno"] == top_permno]
    if len(firm_rows_all) > 0:
        top_year  = int(firm_rows_all.iloc[0]["year"])
        firm_row  = X_test.iloc[[firm_rows_all.index[0]]]
        print(f"  Adjusted to year {top_year} (closest available in test set)")

if len(firm_row) > 0:
    ## SHAP waterfall for this single firm
    try:
        sv_firm = explainer.shap_values(firm_row)
        if isinstance(sv_firm, list):
            sv_firm = sv_firm[1]
        firm_shap_df = pd.DataFrame({
            "feature"   : feature_cols,
            "shap_value": sv_firm.flatten(),
            "feat_value": firm_row.values.flatten()
        }).sort_values("shap_value", key=abs, ascending=False).reset_index(drop=True)
        firm_shap_df["permno"] = top_permno
        firm_shap_df["year"]   = top_year
        firm_shap_df["p_csi"]  = float(firm_preds.iloc[0]["p_csi"])
        firm_shap_df.to_parquet(DIR_OUT / "shap_waterfall_firm.parquet", index=False)
        print(f"  Saved: shap_waterfall_firm.parquet (permno={top_permno}, year={top_year})")
    except Exception as e:
        print(f"  Waterfall computation failed: {e}")

# ==============================================================================
# 11. Summary
# ==============================================================================

print(f"\n[10B] ══ DONE [{MODEL}] ══")
print(f"  Output files in: {DIR_OUT}")
for fname in ["shap_values.parquet", "shap_meta.parquet", "shap_importance.parquet",
              "pdp_1d.parquet", "pdp_2d.parquet", "shap_waterfall_firm.parquet"]:
    path = DIR_OUT / fname
    status = f"{path.stat().st_size // 1024:,} KB" if path.exists() else "NOT FOUND"
    print(f"  {fname:<40} {status}")
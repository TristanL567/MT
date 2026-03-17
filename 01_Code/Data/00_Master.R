#==============================================================================#
#==== 00_Master.R =============================================================#
#==== Pipeline Orchestrator ===================================================#
#==============================================================================#
#
# PROJECT:
#   "The Agony and the Ecstasy" — Constructing a Crash-Filtered Equity Index
#   using Machine Learning. Master's Thesis, WU Wien.
#   Supervisor: Prof. Kurt Hornik
#
# AUTHOR:  Tristan Leiter
# UPDATED: 2026
#
# PURPOSE:
#   Single entry point for the entire pipeline. Sources config.R first,
#   establishes the WRDS connection, then runs each stage in sequence.
#   Every stage can also be run independently by sourcing it directly —
#   all inputs are read from disk via PATH_* constants in config.R.
#
# PIPELINE OVERVIEW:
#   00_Master.R           Orchestrator (this file)
#   config.R              Single source of truth — all parameters and paths
#   ──────────────────────────────────────────────────────────────────────
#   01_Universe.R         CRSP stock universe construction
#   02_Prices.R           CRSP price and return download
#   03_Fundamentals.R     Compustat accounting variable download + CCM merge
#   04_Macro.R            FRED macroeconomic variable download
#   05_CSI_Label.R        Catastrophic Stock Implosion label construction
#   ──────────────────────────────────────────────────────────────────────
#   06_Merge.R            Master panel assembly + reporting lag + validation
#   06B_Feature_Eng.R     Feature engineering and rolling aggregations
#   ──────────────────────────────────────────────────────────────────────
#   07_Feature_Sel.R      Feature selection (Boruta) on raw features
#   08_Split.R            Train / test / OOS split construction
#   08B_Autoencoder.py    Autoencoder — PIT normalisation + VAE (Python/PyCharm)
#   ──────────────────────────────────────────────────────────────────────
#   09_Train.R            Model training with cross-validated HPO
#   10_Evaluate.R         Model evaluation — Recall at fixed FPR
#   11_Results.R          Index construction and backtest
#   12_Robustness.R       Sensitivity analysis across CSI parameter grid
#
# STAGE GROUPINGS:
#   DATA ACQUISITION   (01–04) : Requires live WRDS connection
#   LABEL CONSTRUCTION (05)    : Offline — reads from disk
#   DATA PREPARATION   (06–06B): Offline — merge and engineer features
#   MODELLING          (07–12) : Offline — select, split, encode, train,
#                                evaluate, backtest
#
# WHY AUTOENCODER IS AT 08B (NOT 06C):
#   The autoencoder uses PIT normalisation fitted on the TRAINING set only.
#   This requires the train/test split (08_Split.R) to exist before training.
#   Moving it to 08B ensures:
#     - Boruta (07) runs on raw untransformed features
#     - Split indices exist before PIT fitting — no leakage
#     - Autoencoder is trained only on training data
#     - Latent features are available for 09_Train.R alongside raw features
#
# RUNTIME NOTES:
#   - Stages 01–04 require a live WRDS connection (wrds object).
#   - Stages 05–12 are fully offline — they read from disk only.
#   - Stage 08B runs in Python (PyCharm) — see 08B_Autoencoder.py.
#     R pipeline waits for features_latent.rds to be written by Python
#     before 09_Train.R proceeds.
#   - Each stage saves its outputs to disk before the next stage begins.
#     A crash in any stage does not corrupt prior outputs.
#   - To re-run from a specific stage, comment out earlier source() calls.
#
# DATA FLOW:
#   WRDS ──► 01 ──► universe.rds
#        ──► 02 ──► prices_weekly.rds, prices_monthly.rds
#        ──► 03 ──► fundamentals.rds
#        ──► 04 ──► macro_monthly.rds
#                    │
#                    ▼
#             05 ──► labels_base.rds, labels_all_grid.rds
#                    │
#                    ▼
#             06 ──► panel_raw.rds          (joined, lagged, filtered)
#            06B ──► features_raw.rds       (ratios + rolling aggregations)
#                    │
#                    ▼
#             07 ──► features_selected.rds  (Boruta on raw features)
#             08 ──► splits.rds             (train/test/OOS indices)
#            08B ──► features_latent.rds    (Python: PIT + VAE latent space)
#                    │
#                    ▼
#             09 ──► models/*.rds           (raw features + latent features)
#             10 ──► evaluation_results.rds
#             11 ──► index_returns.rds, backtest_summary.rds
#             12 ──► robustness_results.rds
#
#==============================================================================#

#==============================================================================#
# 1. Environment Setup
#==============================================================================#

## Working directory — set to the script's own location
## Requires: install.packages("this.path")
Directory <- this.path::this.dir()
setwd(Directory)

## Load all required packages
## Auto-installs any missing package before loading
packages <- c(
  ## Core data manipulation
  "here", "dplyr", "tidyr", "tidyverse", "lubridate", "data.table",
  ## Database / WRDS
  "RPostgres", "RSQLite", "dbplyr", "tidyfinance",
  ## FRED
  "fredr",
  ## Time series
  "xts", "slider",
  ## Machine learning
  "xgboost", "lightgbm", "randomForest",
  ## Visualisation
  "ggplot2", "scales",
  ## Utilities
  "purrr", "stringr"
)

for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
    cat(sprintf("Installed: %s\n", pkg))
  }
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}

cat("[00_Master.R] Packages loaded.\n")

#==============================================================================#
# 2. Configuration
#==============================================================================#

source("config.R")

#==============================================================================#
# 3. WRDS Connection
#==============================================================================#

wrds <- get_wrds_connection()
cat("[00_Master.R] WRDS connection established.\n")
print(wrds)

#==============================================================================#
# DATA ACQUISITION (stages 01–04 — requires WRDS)
#==============================================================================#

#==============================================================================#
# Stage 01 — Universe Construction
#
#   Key output : universe.rds
#   Status     : COMPLETE ✓
#==============================================================================#

source("01_Universe.R")

#==============================================================================#
# Stage 02 — Price & Return Download
#
#   Key outputs: prices_weekly.rds, prices_monthly.rds
#   Status     : COMPLETE ✓
#==============================================================================#

source("02_Prices.R")

#==============================================================================#
# Stage 03 — Fundamentals Download
#
#   Downloads ~60 Compustat variables. Point-in-time CCM merge.
#
#   Key output : fundamentals.rds
#   Status     : COMPLETE ✓ — 176,495 rows, 18,526 permno
#==============================================================================#

source("03_Fundamentals.R")

#==============================================================================#
# Stage 04 — Macro Variables
#
#   Downloads 9 FRED series. Forward-fills to monthly. Computes derived
#   features: GDP growth, CPI inflation, INDPRO growth, term spread.
#
#   Key output : macro_monthly.rds
#   Status     : COMPLETE ✓
#==============================================================================#

source("04_Macro.R")

#==============================================================================#
# LABEL CONSTRUCTION (stage 05 — offline)
#==============================================================================#

#==============================================================================#
# Stage 05 — CSI Label Construction
#
#   Drawdown from rolling peak. CSI: drawdown < -80%, zombie T=18 months,
#   max recovery M=-20%. Annual labels. 27-combination sensitivity grid.
#
#   Key outputs: labels_base.rds, labels_all_grid.rds, csi_diagnostics.rds
#   Status     : COMPLETE ✓ — 9.66% base case prevalence, 16,466 events
#==============================================================================#

source("05_CSI_Label.R")

#==============================================================================#
# DATA PREPARATION (stages 06–06B — offline)
#==============================================================================#

#==============================================================================#
# Stage 06 — Master Panel Assembly
#
#   Joins labels + fundamentals + prices + macro → (permno, year) panel.
#   Applies 3-month Compustat reporting lag. Lifetime filter (>=5 years).
#
#   Key output : panel_raw.rds
#   Status     : COMPLETE ✓ — 127,649 rows, 12,895 permno, 90 columns
#==============================================================================#

source("06_Merge.R")

#==============================================================================#
# Stage 06B — Feature Engineering
#
#   ~463 features across 11 families: point-in-time ratios, YoY changes,
#   acceleration, expanding mean/vol, peak deterioration, consecutive
#   declines, accounting momentum, rolling 3yr/5yr stats, price momentum,
#   macro interactions. Altman Z-score components included.
#
#   Key output : features_raw.rds
#   Status     : COMPLETE ✓ — 127,649 rows, 463 features
#==============================================================================#

source("06B_Feature_Eng.R")

#==============================================================================#
# MODELLING PREPARATION (stages 07–08B)
#==============================================================================#

#==============================================================================#
# Stage 07 — Feature Selection (Boruta)
#
#   Applies Boruta algorithm on the TRAINING set (raw features only).
#   Identifies confirmed-important features above the shadow-feature threshold.
#   Run before the split exists — uses temporal split boundary directly.
#   Reduces 463 features to ~80–150 confirmed features.
#
#   Key output : features_selected.rds
#   Status     : NOT YET WRITTEN
#==============================================================================#

# source("07_Feature_Sel.R")

#==============================================================================#
# Stage 08 — Train / Validation / OOS Split
#
#   Three-way split:
#     Train  : 1993–2015 (cross-validated HPO)
#     Test   : 2016–2019 (model selection)
#     OOS    : 2020–2024 (backtest — never touched during training)
#   Split at firm-year level — no firm spans multiple sets.
#   Outputs row indices and pre-split feature matrices for 09_Train.R.
#
#   Key output : splits.rds
#   Status     : NOT YET WRITTEN
#==============================================================================#

# source("08_Split.R")

#==============================================================================#
# Stage 08B — Autoencoder / VAE (Python — PyCharm)
#
#   !! RUN IN PYTHON — see 08B_Autoencoder.py !!
#
#   Reads: features_raw.rds (via reticulate or saved as parquet/csv)
#          splits.rds (train indices for PIT fitting and VAE training)
#
#   Steps:
#     1. Load features_raw and split indices
#     2. Apply QuantileTransform (uniform [0,1]) fitted on TRAIN set only
#        — prevents test/OOS distribution from leaking into normalisation
#     3. Classify features: continuous, bounded [0,1], binary (recession flag)
#     4. Train VAE on normalised train set:
#        - Manual training loop (KL warmup, early stopping, weight restore)
#        - Mixed loss: MSE for continuous, BCE for binary
#        - Beta-VAE with β = 3.0
#     5. Extract z_mean as latent features for train/test/OOS
#     6. Compute reconstruction error (anomaly score) per firm-year
#     7. Save features_latent.rds — latent dims + recon error
#        (readable by R via arrow::read_parquet or reticulate)
#
#   Architecture (auto-derived from input dimensions):
#     Input      : ~463 features (or Boruta-selected subset)
#     Encoder    : 926 → 694 → min_last → latent_dim (≈ √463 ≈ 21)
#     Decoder    : mirror of encoder
#     Latent dim : max(4, floor(√n_features))
#
#   Addresses thesis subquestion 1:
#     Does autoencoder feature extraction improve Average Precision vs
#     raw features alone?
#
#   Key output : features_latent.rds  (or features_latent.parquet)
#   Status     : NOT YET WRITTEN
#   Language   : Python 3.10 | TensorFlow 2.15 | Keras
#   Location   : 08B_Autoencoder.py  (same project directory)
#
#   NOTE: 09_Train.R will not run until features_latent.rds exists.
#         Implement a file-existence check at the top of 09_Train.R.
#==============================================================================#

## Stage 08B runs in Python — no source() call here.
## After running 08B_Autoencoder.py in PyCharm, verify output exists:
if (!file.exists(PATH_FEATURES_LATENT)) {
  warning(paste(
    "[00_Master.R] WARNING: features_latent.rds not found.",
    "Run 08B_Autoencoder.py in PyCharm before proceeding to 09_Train.R.",
    "Downstream stages requiring latent features will be skipped."
  ))
}

#==============================================================================#
# MODELLING (stages 09–12 — offline)
#==============================================================================#

#==============================================================================#
# Stage 09 — Model Training
#
#   Trains five model families × two feature sets = 10 models:
#     Logistic Regression (interpretable baseline)
#     Random Forest, XGBoost, CatBoost, LightGBM
#     × raw features (from 07) and latent features (from 08B)
#   HPO via 5-fold time-series CV, optimising Average Precision.
#   Quantile transformation applied per-feature inside training loop
#   (QuantileTransformUniform, fitted on train set only).
#
#   Key outputs: models/model_<name>_<feature_set>.rds (10 files)
#   Status     : NOT YET WRITTEN
#==============================================================================#

# source("09_Train.R")

#==============================================================================#
# Stage 10 — Evaluation
#
#   Evaluates all 10 models on the test set (2016–2019).
#   Primary: Recall at fixed FPR (3% and 5% — FPR_CONSTRAINTS in config.R).
#   Secondary: AUC-ROC, Average Precision, MCC.
#   Selects best model for index construction.
#   Benchmarks vs: MinVol, Low-Beta, Altman Z-score classifier.
#
#   Key output : evaluation_results.rds
#   Status     : NOT YET WRITTEN
#==============================================================================#

# source("10_Evaluate.R")

#==============================================================================#
# Stage 11 — Index Construction & Backtest
#
#   OOS period: 2020–2024.
#   Crash-Filtered index: excludes firms where P(CSI) > theta.
#   theta calibrated to FPR <= 5% on test set.
#   Annual rebalancing at calendar year-end.
#   Benchmarks: equal-weight market, MinVol, Low-Beta, Altman Z-score.
#
#   Key outputs: index_returns.rds, backtest_summary.rds
#   Status     : NOT YET WRITTEN
#==============================================================================#

# source("11_Results.R")

#==============================================================================#
# Stage 12 — Robustness Analysis
#
#   Re-runs modelling pipeline across all 27 CSI_GRID combinations.
#   Label prevalence stability, AUC stability, index return stability.
#
#   Key output : robustness_results.rds
#   Status     : NOT YET WRITTEN
#==============================================================================#

# source("12_Robustness.R")

#==============================================================================#
# Pipeline Complete
#==============================================================================#

cat("\n[00_Master.R] ══════════════════════════════════════════\n")
cat("  Pipeline run complete.\n")
cat(sprintf("  Timestamp : %s\n", format(Sys.time())))
cat(sprintf("  Period    : %s to %s\n", format(START_DATE), format(END_DATE)))
cat(sprintf("  Seed      : %d\n", SEED))
cat("[00_Master.R] All active stages completed successfully.\n")
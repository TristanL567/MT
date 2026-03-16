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
#   00_Master.R          Orchestrator (this file)
#   config.R             Single source of truth — all parameters and paths
#   ─────────────────────────────────────────────────────────────────────
#   01_Universe.R        CRSP stock universe construction
#   02_Prices.R          CRSP price and return download
#   03_Fundamentals.R    Compustat accounting variable download
#   04_Macro.R           FRED macroeconomic variable download
#   05_CSI_Label.R       Catastrophic Stock Implosion label construction
#   06_Feature_Eng.R     Feature engineering and rolling aggregations
#   06B_Autoencoder.R    Autoencoder for latent feature extraction
#   07_Feature_Sel.R     Feature selection (Boruta / importance screening)
#   08_Split.R           Train / test / OOS split construction
#   09_Train.R           Model training with cross-validated HPO
#   10_Evaluate.R        Model evaluation — Recall at fixed FPR
#   11_Results.R         Index construction and backtest
#   12_Robustness.R      Sensitivity analysis across CSI parameter grid
#
# RUNTIME NOTES:
#   - Stages 01–04 require a live WRDS connection (wrds object).
#   - Stages 05–12 are fully offline — they read from disk only.
#   - Each stage saves its outputs to disk before the next stage begins.
#     A crash in any stage does not corrupt prior outputs.
#   - To re-run from a specific stage, comment out earlier source() calls.
#
# DATA FLOW:
#   WRDS ──► 01 ──► universe.rds
#        ──► 02 ──► prices_weekly.rds, prices_monthly.rds
#        ──► 03 ──► fundamentals.rds
#        ──► 04 ──► macro.rds
#                    │
#                    ▼
#             05 ──► labels_base.rds, labels_all_grid.rds
#                    │
#                    ▼
#             06 ──► features_raw.rds
#            06B ──► features_latent.rds
#             07 ──► features_selected.rds
#             08 ──► splits.rds
#                    │
#                    ▼
#             09 ──► models/*.rds  (one per model × feature set)
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
#
#   Sources config.R — defines every parameter, threshold, date, and PATH_*
#   constant used by all downstream scripts. No magic numbers anywhere else.
#==============================================================================#

source("config.R")

#==============================================================================#
# 3. WRDS Connection
#
#   Establishes a single persistent PostgreSQL connection to WRDS.
#   Passed implicitly to stages 01–04 via the global `wrds` object.
#   Stages 05–12 do not require WRDS and can run fully offline.
#
#   Credentials are stored in .Renviron — never hardcoded.
#   To set credentials: tidyfinance::set_wrds_credentials()
#==============================================================================#

wrds <- get_wrds_connection()
cat("[00_Master.R] WRDS connection established.\n")
print(wrds)

#==============================================================================#
# DATA ACQUISITION (requires WRDS)
#==============================================================================#

#==============================================================================#
# Stage 01 — Universe Construction
#
#   Builds the investable stock universe from CRSP stksecurityinfohist.
#   Filters to NYSE / AMEX / NASDAQ common equity (shrcd 10/11).
#   Deduplicates to one permno per economic entity (permco).
#   Critically uses FIRST valid exchange record — not the terminal record —
#   to avoid silently excluding all delisted stocks (survivor bias fix).
#
#   Key output: universe.rds (~8,000–14,000 permno before lifetime filter)
#==============================================================================#

source("01_Universe.R")

#==============================================================================#
# Stage 02 — Price & Return Download
#
#   Downloads daily prices from crsp_a_stock.dsf → aggregates to weekly.
#   Downloads monthly prices from crsp_a_stock.msf separately.
#   Applies delisting returns (dlret) from crsp_a_stock.msedelist at
#   terminal observations to correct understated crash-period returns.
#   Removes CRSP sentinel values (-66/-77/-88/-99) from return fields.
#
#   Key outputs: prices_weekly.rds (for CSI labels)
#                prices_monthly.rds (for feature engineering)
#==============================================================================#

source("02_Prices.R")

#==============================================================================#
# Stage 03 — Fundamentals Download
#
#   Downloads Compustat accounting variables via the CCM link table
#   (crsp_a_ccm.ccmxpf_linktable) to map permno → gvkey.
#   Pulls balance sheet and income statement items at quarterly frequency,
#   forward-filled to monthly to align with the price panel.
#   Includes zombie-precursor variables: emp (employees), xrent (rental expenses).
#
#   Key output: fundamentals.rds
#
#   STATUS: NOT YET WRITTEN
#==============================================================================#

# source("03_Fundamentals.R")

#==============================================================================#
# Stage 04 — Macro Variables
#
#   Downloads macroeconomic variables from FRED via the tidyfinance package.
#   Includes: federal funds rate, term spread, credit spread, VIX, CPI.
#   Merged to the panel at monthly frequency.
#   Macro variables capture systematic risk environment — Tewari et al.
#   show interactions between accounting distress and macro conditions.
#
#   Key output: macro.rds
#
#   STATUS: NOT YET WRITTEN
#==============================================================================#

# source("04_Macro.R")

#==============================================================================#
# LABEL CONSTRUCTION (offline — no WRDS required)
#==============================================================================#

#==============================================================================#
# Stage 05 — CSI Label Construction
#
#   Computes drawdown from rolling wealth index peak for each stock.
#   Identifies Catastrophic Stock Implosion events: drawdown crosses C = -80%,
#   followed by a zombie period of T = 18 months where the stock never
#   recovers more than M = -20% from the crash low.
#   Collapses monthly status to annual (permno, year) binary labels.
#   Runs all 27 parameter combinations of the CSI sensitivity grid.
#   Right-censors events where zombie window extends beyond END_DATE.
#
#   Key outputs: labels_base.rds, labels_all_grid.rds, csi_diagnostics.rds
#==============================================================================#

source("05_CSI_Label.R")

#==============================================================================#
# FEATURE ENGINEERING (offline — no WRDS required)
#==============================================================================#

#==============================================================================#
# Stage 06 — Feature Engineering
#
#   Merges price panel (monthly), fundamentals, and macro into a single
#   (permno, year) feature matrix aligned to the label panel.
#   Applies the MIN_LIFETIME_YEARS = 5 filter here (not at universe construction)
#   to avoid survivor bias in the label universe.
#   Computes rolling aggregations over 1-year and 5-year windows:
#   mean, min, max, sd, variance, autocorrelation, etc. per ROLLING_STATS.
#   Constructs ratio features: earnings yield, OCF/share, B/P, leverage, etc.
#   Applies leakage-free train/test split at the firm level.
#
#   Key output: features_raw.rds
#
#   STATUS: NOT YET WRITTEN
#==============================================================================#

# source("06_Feature_Eng.R")

#==============================================================================#
# Stage 06B — Autoencoder (Latent Feature Extraction)
#
#   Trains a denoising autoencoder on the raw feature matrix (train set only).
#   Encodes features to a lower-dimensional latent representation.
#   Produces two parallel feature sets for downstream modelling:
#     - Raw features (from 06_Feature_Eng.R)
#     - Latent features (encoder output from this stage)
#   Addresses thesis subquestion 1: does autoencoder feature extraction
#   improve Average Precision compared to raw features alone?
#
#   Key output: features_latent.rds
#
#   STATUS: NOT YET WRITTEN
#==============================================================================#

# source("06B_Autoencoder.R")

#==============================================================================#
# Stage 07 — Feature Selection
#
#   Applies Boruta algorithm on the training set to identify features
#   with confirmed importance above the shadow-feature threshold.
#   Run separately on raw features and latent features.
#   Reduces dimensionality before model training to improve
#   generalisation and reduce HPO search space.
#
#   Key output: features_selected.rds
#
#   STATUS: NOT YET WRITTEN
#==============================================================================#

# source("07_Feature_Sel.R")

#==============================================================================#
# MODELLING (offline — no WRDS required)
#==============================================================================#

#==============================================================================#
# Stage 08 — Train / Validation / OOS Split
#
#   Constructs the three-way split defined in config.R:
#     Train  : 1998–2015 (cross-validated HPO)
#     Test   : 2016–2019 (model selection)
#     OOS    : 2020–2024 (final backtest — never touched during training)
#   Split is at the firm-year level; no firm appears in multiple sets.
#   Computes class weights for imbalanced learning if CLASS_WEIGHT != NULL.
#
#   Key output: splits.rds
#
#   STATUS: NOT YET WRITTEN
#==============================================================================#

# source("08_Split.R")

#==============================================================================#
# Stage 09 — Model Training
#
#   Trains five model families on both raw and latent feature sets:
#     Logistic Regression (interpretable baseline)
#     Random Forest
#     XGBoost
#     CatBoost
#     LightGBM
#   Hyperparameter optimisation via 5-fold time-series CV,
#   optimising Average Precision (HPO_METRIC) over HPO_ITER iterations.
#   Produces 10 trained models (5 algorithms × 2 feature sets).
#
#   Key outputs: models/model_<name>_<feature_set>.rds (10 files)
#
#   STATUS: NOT YET WRITTEN
#==============================================================================#

# source("09_Train.R")

#==============================================================================#
# Stage 10 — Evaluation
#
#   Evaluates all 10 models on the test set (2016–2019).
#   Primary metric: Recall at fixed FPR (3% and 5% — see FPR_CONSTRAINTS).
#   Also reports: AUC-ROC, Average Precision, MCC for completeness.
#   Selects the best-performing model for index construction in Stage 11.
#   Addresses thesis subquestion 2: do ensemble methods reduce FPR
#   vs traditional volatility-based exclusion strategies?
#
#   Key output: evaluation_results.rds
#
#   STATUS: NOT YET WRITTEN
#==============================================================================#

# source("10_Evaluate.R")

#==============================================================================#
# INDEX CONSTRUCTION & RESULTS (offline — no WRDS required)
#==============================================================================#

#==============================================================================#
# Stage 11 — Index Construction & Backtest
#
#   Uses best model from Stage 10 to predict CSI probability for each firm
#   in the OOS period (2020–2024) one year ahead.
#   Constructs the "Crash-Filtered" index: excludes firms where
#   predicted P(CSI) > theta, where theta is calibrated to satisfy
#   FPR <= 5% (dynamically, not the default 0.5 classifier threshold).
#   Index is rebalanced annually at end of each calendar year.
#   Compares risk-adjusted returns vs:
#     - Market benchmark (equal-weight universe)
#     - Minimum-volatility strategy
#     - Low-beta strategy
#     - Altman Z-score exclusion strategy
#
#   Key outputs: index_returns.rds, backtest_summary.rds
#
#   STATUS: NOT YET WRITTEN
#==============================================================================#

# source("11_Results.R")

#==============================================================================#
# Stage 12 — Robustness Analysis
#
#   Re-runs the full modelling pipeline for all 27 CSI parameter combinations
#   from CSI_GRID to assess sensitivity of results to label definition.
#   Reports: label prevalence stability, AUC stability, index return stability.
#   Confirms that main findings are not an artefact of a specific C/M/T choice.
#
#   Key output: robustness_results.rds
#
#   STATUS: NOT YET WRITTEN
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
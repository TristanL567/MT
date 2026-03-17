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
#   06C_Autoencoder.R     Autoencoder for latent feature extraction
#   ──────────────────────────────────────────────────────────────────────
#   07_Feature_Sel.R      Feature selection (Boruta)
#   08_Split.R            Train / test / OOS split construction
#   09_Train.R            Model training with cross-validated HPO
#   10_Evaluate.R         Model evaluation — Recall at fixed FPR
#   11_Results.R          Index construction and backtest
#   12_Robustness.R       Sensitivity analysis across CSI parameter grid
#
# STAGE GROUPINGS:
#   DATA ACQUISITION  (01–04) : Requires live WRDS connection
#   LABEL CONSTRUCTION (05)   : Offline — reads from disk
#   DATA PREPARATION   (06)   : Offline — merge, engineer, encode
#   MODELLING         (07–12) : Offline — select, train, evaluate, backtest
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
#        ──► 04 ──► macro_monthly.rds
#                    │
#                    ▼
#             05 ──► labels_base.rds, labels_all_grid.rds
#                    │
#                    ▼
#             06 ──► panel_raw.rds          (joined, lagged, filtered)
#            06B ──► features_raw.rds       (ratios + rolling aggregations)
#            06C ──► features_latent.rds    (autoencoder latent space)
#                    │
#                    ▼
#             07 ──► features_selected.rds
#             08 ──► splits.rds
#             09 ──► models/*.rds
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
# DATA ACQUISITION (stages 01–04 — requires WRDS)
#==============================================================================#

#==============================================================================#
# Stage 01 — Universe Construction
#
#   Builds the investable stock universe from CRSP stksecurityinfohist.
#   Filters to NYSE / AMEX / NASDAQ common equity.
#   Deduplicates to one permno per economic entity (permco).
#   Uses FIRST valid exchange record to avoid silently excluding all
#   delisted stocks (survivor bias fix — see script header for full detail).
#
#   Key output : universe.rds
#   Status     : COMPLETE
#==============================================================================#

source("01_Universe.R")

#==============================================================================#
# Stage 02 — Price & Return Download
#
#   Downloads daily prices (crsp_a_stock.dsf) → aggregates to weekly.
#   Downloads monthly prices (crsp_a_stock.msf) separately.
#   Applies delisting returns (dlret) at terminal observations.
#   Removes CRSP sentinel values (-66/-77/-88/-99).
#
#   Key outputs: prices_weekly.rds, prices_monthly.rds
#   Status     : COMPLETE
#==============================================================================#

source("02_Prices.R")

#==============================================================================#
# Stage 03 — Fundamentals Download
#
#   Downloads ~60 Compustat accounting variables from comp.funda.
#   Links to CRSP permno via point-in-time CCM merge
#   (crsp_a_ccm.ccmxpf_linktable) to prevent look-ahead bias.
#   Covers: balance sheet, income statement, cash flow, market data,
#   and zombie-precursor variables (emp, xrent).
#
#   Key output : fundamentals.rds
#   Status     : RUNNING
#==============================================================================#

source("03_Fundamentals.R")

#==============================================================================#
# Stage 04 — Macro Variables
#
#   Downloads 9 macroeconomic series from FRED via the fredr package.
#   Series: GDP, UNRATE, FEDFUNDS, GS10, HY spread, VIX, CPI, INDPRO, USREC.
#   Forward-fills all series to monthly frequency aligned to month-end dates.
#   Computes derived features: GDP growth, CPI inflation, INDPRO growth,
#   term spread (GS10 − FEDFUNDS).
#
#   Key output : macro_monthly.rds
#   Status     : COMPLETE (pending run)
#==============================================================================#

# source("04_Macro.R")

#==============================================================================#
# LABEL CONSTRUCTION (stage 05 — offline)
#==============================================================================#

#==============================================================================#
# Stage 05 — CSI Label Construction
#
#   Computes drawdown from rolling wealth index peak for each stock.
#   Identifies CSI events: drawdown < C = -80%, zombie period T = 18 months,
#   maximum recovery M = -20% from crash low.
#   Collapses monthly status to annual (permno, year) binary labels.
#   Runs full 27-combination sensitivity grid (CSI_GRID in config.R).
#   Right-censors events where zombie window extends beyond END_DATE.
#
#   Key outputs: labels_base.rds, labels_all_grid.rds, csi_diagnostics.rds
#   Status     : COMPLETE ✓ — validated, 9.66% base case prevalence
#==============================================================================#

source("05_CSI_Label.R")

#==============================================================================#
# DATA PREPARATION (stages 06–06C — offline)
#==============================================================================#

#==============================================================================#
# Stage 06 — Master Panel Assembly
#
#   Joins labels + fundamentals + prices_monthly + macro into a single
#   (permno, year) analytical panel.
#   Applies ~3-month reporting lag to Compustat fundamentals to prevent
#   look-ahead bias (datadate ≠ public availability date).
#   Applies MIN_LIFETIME_YEARS = 5 lifetime filter (deferred from stage 01
#   to preserve full label universe for CSI diagnostics).
#   Validates join coverage, class balance, and feature missingness.
#   This stage produces the definitive modelling dataset before any
#   feature construction — use it to validate the data foundation.
#
#   Key output : panel_raw.rds
#   Status     : NOT YET WRITTEN
#==============================================================================#

# source("06_Merge.R")

#==============================================================================#
# Stage 06B — Feature Engineering
#
#   Constructs the full feature matrix from panel_raw.rds.
#   Ratio features: earnings yield, OCF/share, ROIC, leverage, etc.
#   All 25 paper ff_* features derived here (see compustat_variable_reference.xlsx).
#   Rolling aggregations over 12-month and 60-month windows per ROLLING_STATS:
#   mean, min, max, sd, variance, autocorrelation, etc.
#   Log return from prices_monthly appended as a feature.
#
#   Key output : features_raw.rds
#   Status     : NOT YET WRITTEN
#==============================================================================#

# source("06B_Feature_Eng.R")

#==============================================================================#
# Stage 06C — Autoencoder (Latent Feature Extraction)
#
#   Trains a denoising autoencoder on features_raw.rds (train set only —
#   no data leakage from test/OOS into the encoder).
#   Encodes features to a lower-dimensional latent representation.
#   Produces two parallel feature sets for downstream modelling:
#     - Raw features    (from 06B_Feature_Eng.R)
#     - Latent features (encoder output from this stage)
#   Addresses thesis subquestion 1: does autoencoder feature extraction
#   improve Average Precision compared to raw features alone?
#
#   Key output : features_latent.rds
#   Status     : NOT YET WRITTEN
#==============================================================================#

# source("06C_Autoencoder.R")

#==============================================================================#
# MODELLING (stages 07–12 — offline)
#==============================================================================#

#==============================================================================#
# Stage 07 — Feature Selection
#
#   Applies Boruta algorithm on the training set.
#   Run separately on raw features (from 06B) and latent features (from 06C).
#   Confirmed-important features passed to stage 08.
#   Reduces dimensionality before HPO search.
#
#   Key output : features_selected.rds
#   Status     : NOT YET WRITTEN
#==============================================================================#

# source("07_Feature_Sel.R")

#==============================================================================#
# Stage 08 — Train / Validation / OOS Split
#
#   Constructs the three-way split defined in config.R:
#     Train  : 1998–2015 (cross-validated HPO)
#     Test   : 2016–2019 (model selection)
#     OOS    : 2020–2024 (final backtest — never touched during training)
#   Split is firm-year level — no firm spans multiple sets.
#
#   Key output : splits.rds
#   Status     : NOT YET WRITTEN
#==============================================================================#

# source("08_Split.R")

#==============================================================================#
# Stage 09 — Model Training
#
#   Trains five model families × two feature sets = 10 models:
#     Logistic Regression (interpretable baseline)
#     Random Forest, XGBoost, CatBoost, LightGBM
#     × raw features and latent features
#   HPO via 5-fold time-series CV, optimising Average Precision.
#
#   Key outputs: models/model_<name>_<feature_set>.rds (10 files)
#   Status     : NOT YET WRITTEN
#==============================================================================#

# source("09_Train.R")

#==============================================================================#
# Stage 10 — Evaluation
#
#   Evaluates all 10 models on the test set (2016–2019).
#   Primary metric: Recall at fixed FPR (3% and 5%).
#   Secondary: AUC-ROC, Average Precision, MCC.
#   Selects best model for index construction in stage 11.
#   Addresses subquestion 2: do ensemble methods reduce FPR vs
#   traditional volatility-based exclusion strategies?
#
#   Key output : evaluation_results.rds
#   Status     : NOT YET WRITTEN
#==============================================================================#

# source("10_Evaluate.R")

#==============================================================================#
# Stage 11 — Index Construction & Backtest
#
#   Uses best model to predict CSI probability for OOS period (2020–2024).
#   Constructs Crash-Filtered index: excludes firms where P(CSI) > theta,
#   theta dynamically calibrated to satisfy FPR <= 5%.
#   Rebalances annually at end of each calendar year.
#   Benchmarks against: equal-weight market, MinVol, Low-Beta, Altman Z-score.
#
#   Key outputs: index_returns.rds, backtest_summary.rds
#   Status     : NOT YET WRITTEN
#==============================================================================#

# source("11_Results.R")

#==============================================================================#
# Stage 12 — Robustness Analysis
#
#   Re-runs full modelling pipeline across all 27 CSI_GRID combinations.
#   Reports: label prevalence stability, AUC stability, index return stability.
#   Confirms findings are not artefacts of a specific C/M/T choice.
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
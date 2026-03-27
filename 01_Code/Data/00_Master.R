#==============================================================================#
#==== 00_Master.R =============================================================#
#==== Pipeline Orchestration — "The Agony and the Ecstasy" ===================#
#==============================================================================#
#
# PURPOSE:
#   Single entry point for the full pipeline. Run sections selectively or
#   end-to-end. Each section is guarded by a RUN_* flag so you can skip
#   completed steps without re-running expensive operations.
#
# THESIS:
#   "The Agony and the Ecstasy" — Crash-Filtered Equity Index Construction
#   via Machine Learning. WU Wien, supervised by Prof. Kurt Hornik.
#
# DATA SOURCES:
#   CRSP    : prices, returns, delisting (via WRDS)
#   Compustat: annual fundamentals (via WRDS CCM link)
#   FRED    : macro variables (via fredr API)
#
# MODEL VARIANTS:
#   M1  : CSI prediction, fundamentals only (features_fund.rds)
#   M3  : CSI prediction, full features (features_raw.rds)
#   B1  : 5-yr CAGR bucket classifier (features_fund.rds, labels_bucket.rds)
#   BS  : B1-structural combined label (features_fund.rds, labels_structural.rds)
#
# INDEX STRATEGIES:
#   S1  : M1 exclusion (top 5% by score)
#   S4  : M1 + zombie recovery filter
#   S5  : Tiered threshold (3%/8% + altman_z2)
#   S6  : B1 exclusion (top 20% by score)
#   C1  : B1-structural concentrated long 200
#   C3  : C1 + M1 veto (two-layer filter)
#
#==============================================================================#

## ── Working directory ────────────────────────────────────────────────────────
## Requires: install.packages("this.path")
suppressPackageStartupMessages(library(this.path))
Directory <- this.path::this.dir()
setwd(Directory)

## ── Package loading ──────────────────────────────────────────────────────────
.packages <- c(
  ## Core
  "here", "data.table", "dplyr", "tidyr", "lubridate",
  ## Database / WRDS
  "RPostgres", "RSQLite", "dbplyr", "tidyfinance",
  ## FRED
  "fredr",
  ## Time series
  "xts", "slider",
  ## Machine learning (R-side)
  "xgboost", "lightgbm", "randomForest",
  ## Visualisation
  "ggplot2", "scales", "patchwork",
  ## Model evaluation
  "pROC", "PRROC",
  ## Tree models
  "rpart", "rpart.plot",
  ## Parquet
  "arrow",
  ## Utilities
  "purrr", "stringr", "forcats"
)

for (.pkg in .packages) {
  if (!requireNamespace(.pkg, quietly = TRUE)) {
    install.packages(.pkg, quiet = TRUE)
    cat(sprintf("[00_Master] Installed: %s\n", .pkg))
  }
  suppressPackageStartupMessages(library(.pkg, character.only = TRUE))
}
rm(.packages, .pkg)
cat("[00_Master] All packages loaded.\n")

## ── Config ───────────────────────────────────────────────────────────────────
source("config.R")

## ── Initialise all figure directories ────────────────────────────────────────
FIGS <- fn_setup_figure_dirs()
cat(sprintf("[00_Master] Figure directories ready under %s\n", DIR_FIGURES))

#==============================================================================#
# RUN FLAGS — set TRUE to execute, FALSE to skip
#==============================================================================#

## Data pipeline (01–04): run once, outputs cached
RUN_01_UNIVERSE     <- FALSE  ## CRSP universe construction
RUN_02_PRICES       <- FALSE  ## monthly/weekly prices + delisting
RUN_03_FUNDAMENTALS <- FALSE  ## Compustat fundamentals + CCM link
RUN_04_MACRO        <- FALSE  ## FRED macro variables

## Label construction (05): run when data changes or parameters change
RUN_05_CSI          <- FALSE  ## CSI base case + 27-parameter grid
RUN_05B_BUCKET      <- FALSE  ## 5-year forward CAGR bucket labels
RUN_05C_STRUCTURAL  <- FALSE  ## Combined CSI + bucket structural labels

## Feature engineering (06): run when panel or features change
RUN_06_MERGE        <- FALSE  ## panel merge
RUN_06B_FEATURES    <- FALSE  ## feature engineering

## Model preparation (07–08)
RUN_07_FEATSEL      <- FALSE  ## feature selection diagnostics
RUN_08_SPLIT        <- FALSE  ## train/test/OOS split construction
## 08B_Autoencoder.py is run separately in Python (VAE — M2/M4)

## AutoML training — run in Python (09C_AutoGluon.py), not from here
## MODEL = "fund"        → M1
## MODEL = "raw"         → M3
## MODEL = "bucket"      → B1-bucket
## MODEL = "structural"  → B1-structural

## R-side evaluation and index construction
RUN_10_EVALUATE     <- FALSE  ## model evaluation (AUC, AP, decile tables)
RUN_11_RESULTS      <- FALSE  ## index construction (S1–S6, benchmark)
RUN_12_EVALUATION   <- FALSE  ## index diagnostics

## Robustness and comparison
RUN_13_ROBUSTNESS   <- FALSE  ## Parts A–E
RUN_14_COMPARISON   <- FALSE  ## vs low-vol and quality benchmarks

#==============================================================================#
# Helper: run a script with timing and error handling
#==============================================================================#

fn_run_script <- function(script_name, description) {
  path <- file.path(DIR_CODE, script_name)
  if (!file.exists(path)) {
    cat(sprintf("[00_Master] WARNING: %s not found — skipping\n", script_name))
    return(invisible(FALSE))
  }
  cat(sprintf("\n[00_Master] ════ Running: %s (%s) ════\n",
              script_name, description))
  t0 <- proc.time()
  tryCatch(
    source(path, local = FALSE),
    error = function(e) {
      cat(sprintf("[00_Master] ERROR in %s:\n  %s\n", script_name, e$message))
      stop(e)
    }
  )
  elapsed <- round((proc.time() - t0)["elapsed"], 1)
  cat(sprintf("[00_Master] ✓ %s complete (%.1fs)\n", script_name, elapsed))
  invisible(TRUE)
}

#==============================================================================#
# 01 — Universe Construction
#==============================================================================#

if (RUN_01_UNIVERSE) {
  fn_run_script("Data/01_Universe.R",
                "CRSP universe — valid exchanges, share types, lifetime filter")
}

#==============================================================================#
# 02 — Prices
#==============================================================================#

if (RUN_02_PRICES) {
  fn_run_script("Data/02_Prices.R",
                "Monthly + weekly prices, delisting returns, CRSP adjustments")
}

#==============================================================================#
# 03 — Fundamentals
#==============================================================================#

if (RUN_03_FUNDAMENTALS) {
  fn_run_script("Data/03_Fundamentals.R",
                "Compustat annual fundamentals, CCM permno-gvkey link")
}

#==============================================================================#
# 04 — Macro
#==============================================================================#

if (RUN_04_MACRO) {
  fn_run_script("Data/04_Macro.R",
                "FRED: term spread, HY spread, VIX, unemployment, recession")
}

#==============================================================================#
# 05 — Label Construction
#==============================================================================#

if (RUN_05_CSI) {
  fn_run_script("Data/05_CSI_Label.R",
                "CSI labels: base case (C=-0.8, M=-0.2, T=18) + 27-param grid")
}

if (RUN_05B_BUCKET) {
  fn_run_script("Data/05B_Bucket_Labels.R",
                "Bucket labels: 5-yr forward CAGR < -2% vs >= 0%")
}

if (RUN_05C_STRUCTURAL) {
  fn_run_script("Data/05C_Structural_Labels.R",
                "Structural labels: combined CSI + bucket (phoenix reclassification)")
}

#==============================================================================#
# 06 — Panel + Feature Engineering
#==============================================================================#

if (RUN_06_MERGE) {
  fn_run_script("Data/06_Merge.R",
                "Panel merge: CRSP prices + Compustat fundamentals + macro")
}

if (RUN_06B_FEATURES) {
  fn_run_script("Data/06B_Feature_Eng.R",
                paste0("Feature engineering → features_raw.rds (~463 features) ",
                       "and features_fund.rds (fundamentals + macro, no price)"))
}

#==============================================================================#
# 07–08 — Preparation
#==============================================================================#

if (RUN_07_FEATSEL) {
  fn_run_script("Data/07_Feature_Sel.R",
                "Feature selection diagnostics (VIF, correlation, importance)")
}

if (RUN_08_SPLIT) {
  fn_run_script("Data/08_Split.R",
                "Train/test/OOS split: train <= 2015, test 2016-2019, OOS 2020+")
}

#==============================================================================#
# 09 — AutoML Training (Python — run separately)
#==============================================================================#

cat("\n[00_Master] ── AutoML Training (Python) ─────────────────────────────\n")
cat("  Run 09C_AutoGluon.py separately with MODEL set to:\n")
cat("    'fund'       → M1  (CSI, fundamentals)       → Tables/ag_fund/\n")
cat("    'raw'        → M3  (CSI, full features)       → Tables/ag_raw/\n")
cat("    'bucket'     → B1  (5yr CAGR bucket)          → Tables/ag_bucket/\n")
cat("    'structural' → BS  (combined structural label) → Tables/ag_structural/\n")
cat("  NOTE: Remove year_cat, use term_spread/hy_spread/vix/recession instead.\n")

#==============================================================================#
# 10 — Model Evaluation
#==============================================================================#

if (RUN_10_EVALUATE) {
  fn_run_script("ML/10_Evaluate.R",
                "AUC, AP, PR curves, decile tables for M1/M3/B1/BS")
}

#==============================================================================#
# 11 — Index Construction
#==============================================================================#

if (RUN_11_RESULTS) {
  fn_run_script("ML/11_Results.R",
                "Index construction: benchmark, S1-S6, C1-C3 backtest")
}

#==============================================================================#
# 12 — Index Diagnostics
#==============================================================================#

if (RUN_12_EVALUATION) {
  fn_run_script("ML/12_Evaluation.R",
                "Exclusion diagnostics, sector concentration, TE, turnover")
}

#==============================================================================#
# 13 — Robustness
#==============================================================================#

if (RUN_13_ROBUSTNESS) {
  fn_run_script("ML/13_Robustness.R",
                "Parts A-E: grid sensitivity, recovery classifier, S4, tiered, C1/C3")
}

#==============================================================================#
# 14 — Comparison vs Naive Benchmarks
#==============================================================================#

if (RUN_14_COMPARISON) {
  fn_run_script("ML/14_Comparison.R",
                "C1-structural vs low-vol 200 vs quality 200 (Altman Z)")
}

#==============================================================================#
# Pipeline Summary
#==============================================================================#

cat("\n[00_Master] ════════════════════════════════════════════════════\n")
cat("  PIPELINE STATUS\n")
cat("  ════════════════════════════════════════════════════════════\n\n")

## Check which key output files exist
.check <- function(path, label) {
  status <- if (file.exists(path)) "✓" else "✗ MISSING"
  cat(sprintf("  %s  %-30s  %s\n", status, label, basename(path)))
}

cat("  Data:\n")
.check(PATH_UNIVERSE,          "Universe")
.check(PATH_PRICES_MONTHLY,    "Prices monthly")
.check(PATH_FUNDAMENTALS,      "Fundamentals")
.check(PATH_MACRO_MONTHLY,     "Macro monthly")

cat("\n  Labels:\n")
.check(PATH_LABELS_BASE,       "CSI base labels")
.check(PATH_LABELS_BUCKET,     "Bucket labels (5yr)")
.check(PATH_LABELS_STRUCTURAL, "Structural labels (combined)")

cat("\n  Features:\n")
.check(PATH_FEATURES_FUND,     "features_fund.rds (M1/B1 input)")
.check(PATH_FEATURES_RAW,      "features_raw.rds  (M3 input)")

cat("\n  Model predictions:\n")
.check(file.path(DIR_TABLES_M1,        "ag_preds_test.parquet"), "M1 test preds")
.check(file.path(DIR_TABLES_M3,        "ag_preds_test.parquet"), "M3 test preds")
.check(file.path(DIR_TABLES_B1_BUCKET, "ag_preds_test.parquet"), "B1-bucket test preds")
.check(file.path(DIR_TABLES_B1_STRUCT, "ag_preds_test.parquet"), "B1-structural test preds")

cat("\n  Index outputs:\n")
.check(PATH_INDEX_RETURNS,     "Index returns")
.check(PATH_ROBUST_CONC_P,     "Concentrated portfolio perf")
.check(PATH_COMPARISON_PERF,   "Comparison performance")

cat("\n")
rm(.check)

cat(sprintf("[00_Master] DONE: %s\n", format(Sys.time())))
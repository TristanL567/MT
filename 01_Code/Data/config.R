#==============================================================================#
#==== config.R ================================================================#
#==== Single Source of Truth — All Pipeline Parameters ========================#
#==============================================================================#
#
# PURPOSE:
#   Every threshold, date, seed, and path used anywhere in the pipeline
#   is declared here and nowhere else. No magic numbers in any other script.
#
# USAGE:
#   source("config.R") at the top of every pipeline script.
#   00_Master.R sources this first, before any stage runs.
#
#==============================================================================#

#==============================================================================#
# 1. Reproducibility
#==============================================================================#

SEED <- 123
set.seed(SEED)

#==============================================================================#
# 2. Directories & Output Paths
#==============================================================================#

DIR_ROOT   <- here::here()
DIR_CODE   <- file.path(DIR_ROOT, "01_Code")
DIR_DATA   <- file.path(DIR_ROOT, "02_Data")
DIR_OUTPUT <- file.path(DIR_ROOT, "03_Output")

## CRSP
## Note: Data_CRSP_Directory kept as alias for 01_Universe.R compatibility
Data_CRSP_Directory <- file.path(DIR_DATA, "CRSP")
DIR_CRSP_RAW        <- file.path(Data_CRSP_Directory, "Raw")
DIR_CRSP_PROC       <- file.path(Data_CRSP_Directory, "Processed")

## Compustat
DIR_COMP      <- file.path(DIR_DATA, "Compustat")
DIR_COMP_RAW  <- file.path(DIR_COMP, "Raw")
DIR_COMP_PROC <- file.path(DIR_COMP, "Processed")

## Downstream
DIR_MACRO    <- file.path(DIR_DATA, "Macro")
DIR_LABELS   <- file.path(DIR_DATA, "Labels")
DIR_FEATURES <- file.path(DIR_DATA, "Features")
DIR_PANEL    <- file.path(DIR_DATA, "Panel")

## Output
DIR_FIGURES <- file.path(DIR_OUTPUT, "Figures")
DIR_TABLES  <- file.path(DIR_OUTPUT, "Tables")
DIR_MODELS  <- file.path(DIR_OUTPUT, "Models")

## Create all directories if they don't exist
invisible(lapply(
  c(DIR_CRSP_RAW, DIR_CRSP_PROC,
    DIR_COMP_RAW, DIR_COMP_PROC,
    DIR_MACRO, DIR_LABELS, DIR_FEATURES, DIR_PANEL,
    DIR_FIGURES, DIR_TABLES, DIR_MODELS),
  dir.create, showWarnings = FALSE, recursive = TRUE
))

#------------------------------------------------------------------------------#
# File paths — 01_Universe.R
#------------------------------------------------------------------------------#

PATH_UNIVERSE_RAW <- file.path(DIR_CRSP_RAW,  "universe_raw.rds")
PATH_UNIVERSE     <- file.path(DIR_CRSP_PROC, "universe.rds")

#------------------------------------------------------------------------------#
# File paths — 02_Prices.R
#------------------------------------------------------------------------------#

PATH_PRICES_DAILY_RAW   <- file.path(DIR_CRSP_RAW,  "prices_daily_raw.rds")
PATH_PRICES_MONTHLY_RAW <- file.path(DIR_CRSP_RAW,  "prices_monthly_raw.rds")
PATH_PRICES_WEEKLY      <- file.path(DIR_CRSP_PROC, "prices_weekly.rds")
PATH_PRICES_MONTHLY     <- file.path(DIR_CRSP_PROC, "prices_monthly.rds")
PATH_DELISTING          <- file.path(DIR_CRSP_RAW,  "delisting_raw.rds")

#------------------------------------------------------------------------------#
# File paths — 03_Fundamentals.R
#------------------------------------------------------------------------------#

PATH_FUNDAMENTALS_RAW <- file.path(DIR_COMP_RAW,  "fundamentals_raw.rds")
PATH_FUNDAMENTALS     <- file.path(DIR_COMP_PROC, "fundamentals.rds")
PATH_CCM_LINK         <- file.path(DIR_COMP_RAW,  "ccm_link_raw.rds")

#------------------------------------------------------------------------------#
# File paths — 04_Macro.R
#------------------------------------------------------------------------------#

PATH_MACRO_RAW     <- file.path(DIR_MACRO, "macro_raw.rds")
PATH_MACRO_MONTHLY <- file.path(DIR_MACRO, "macro_monthly.rds")

#------------------------------------------------------------------------------#
# File paths — 05_CSI_Label.R
#------------------------------------------------------------------------------#

PATH_LABELS_BASE <- file.path(DIR_LABELS,  "labels_base.rds")
PATH_LABELS_GRID <- file.path(DIR_LABELS,  "labels_all_grid.rds")
PATH_LABELS_DIAG <- file.path(DIR_LABELS,  "csi_diagnostics.rds")
PATH_FIGURE_CSI  <- file.path(DIR_FIGURES, "csi_events_per_year_base.png")
## Note: individual grid files labels_<param_id>.rds use inline file.path()
## in 05_CSI_Label.R since param_id is dynamic — this is acceptable.

#------------------------------------------------------------------------------#
# File paths — 06_Merge.R
#------------------------------------------------------------------------------#

PATH_PANEL_RAW <- file.path(DIR_PANEL, "panel_raw.rds")

#------------------------------------------------------------------------------#
# File paths — 06B_Feature_Eng.R
#------------------------------------------------------------------------------#

PATH_FEATURES_RAW <- file.path(DIR_FEATURES, "features_raw.rds")

#------------------------------------------------------------------------------#
# File paths — 06C_Autoencoder.R
#------------------------------------------------------------------------------#

PATH_FEATURES_LATENT <- file.path(DIR_FEATURES, "features_latent.parquet")
PATH_FEATURES_FUND <- file.path(DIR_FEATURES, "features_fund.rds")   # ADD THIS — M1

#------------------------------------------------------------------------------#
# File paths — 07_Feature_Sel.R
#------------------------------------------------------------------------------#

PATH_FEATURES_SELECTED <- file.path(DIR_FEATURES, "features_selected.rds")

#------------------------------------------------------------------------------#
# File paths — 08_Split.R
#------------------------------------------------------------------------------#

PATH_SPLITS <- file.path(DIR_FEATURES, "splits.rds")

#------------------------------------------------------------------------------#
# File paths — 10_Evaluate.R
#------------------------------------------------------------------------------#

PATH_EVAL_RESULTS <- file.path(DIR_TABLES, "evaluation_results.rds")

#------------------------------------------------------------------------------#
# File paths — 11_Results.R
#------------------------------------------------------------------------------#

PATH_INDEX_RETURNS    <- file.path(DIR_TABLES,  "index_returns.rds")
PATH_BACKTEST_SUMMARY <- file.path(DIR_TABLES,  "backtest_summary.rds")

#------------------------------------------------------------------------------#
# File paths — 12_Robustness.R
#------------------------------------------------------------------------------#

PATH_ROBUSTNESS <- file.path(DIR_TABLES, "robustness_results.rds")

#==============================================================================#
# 3. Date Range
#==============================================================================#

START_DATE <- as.Date("1993-01-01")
END_DATE   <- as.Date("2024-12-31")

#==============================================================================#
# 4. Universe Filters
#==============================================================================#

VALID_EXCHANGES    <- c("N", "A", "Q")   # NYSE, AMEX, NASDAQ
EXCLUDE_SECTYPES   <- c("FUND")
MIN_LIFETIME_YEARS <- 5                  # Applied in 06_Feature_Engineering.R only
VALID_SHARETYPES   <- c("NS", NA)        # Normal shares; NA = legacy pre-classification
VALID_SUBTYPES     <- c("COM", NA)       # Common stock; NA = legacy

#==============================================================================#
# 5. CSI Label Parameters
#
#   DEFINITION USED IN THIS PIPELINE:
#   ---------------------------------
#   Implemented as drawdown from rolling peak:
#
#     Drawdown(t) = Price(t) / max(Price(s), s <= t) - 1
#
#   Economically cleaner than cumulative-from-first-observation:
#   reference-point agnostic, captures what a portfolio manager observes,
#   and avoids penalising stocks with long prior appreciation.
#
#   THREE PARAMETERS — ALL T VALUES ARE IN MONTHS:
#
#     CSI_C : Drawdown threshold — initial crash triggers when
#             Drawdown(t) < CSI_C.
#             Paper equivalent: C = -0.80
#
#     CSI_M : Maximum recovery allowed during the zombie period.
#             Measured as return from crash low:
#               Recovery(t) = Price(t) / Price(crash_low) - 1
#             Zombie broken if Recovery(t) > CSI_M at any point in T months.
#             CSI_M = -0.20: stock must stay >= 20% below crash low.
#             Paper equivalent: M = -0.20
#
#     CSI_T : Minimum zombie period length in MONTHS.
#             18 months ≈ 78 weeks ≈ 1.5 years.
#             Paper equivalent: T = 78 weeks ≈ 18 months.
#             Unit is MONTHS here because the pipeline runs on monthly prices.
#
#   BASE CASE (replicates paper's chosen parameters):
#   Do not change for primary analysis. Use CSI_GRID for robustness checks.
#==============================================================================#

CSI_BASE <- list(
  C = -0.80,
  M = -0.20,
  T = 18L    # MONTHS — equivalent to paper's 78 weeks
)

#==============================================================================#
# 5B. CSI Sensitivity Grid
#
#   Full factorial grid: 3 x 3 x 3 = 27 parameter combinations.
#   ALL T VALUES ARE IN MONTHS.
#
#   C: -0.60 (moderate crash) to -0.90 (near-wipeout)
#   M:  0.00 (any recovery breaks zombie) to -0.30 (strict non-recovery)
#   T:    12 (1 year) to 24 (2 years) in months
#
#   Used by:
#     05_CSI_Label.R  — loops over all 27, saves labels_<param_id>.rds each
#     12_Robustness.R — compares label stability and model sensitivity
#
#   MAX_IMPLOSION_RATE: hard ceiling on CSI prevalence per parameterisation.
#   Paper achieved 6.6% under base case. 15% allows headroom for loose params.
#==============================================================================#

CSI_GRID <- expand.grid(
  C = c(-0.60, -0.80, -0.90),
  M = c( 0.00, -0.20, -0.30),
  T = c(  12L,   18L,   24L),   # MONTHS: 12=1yr, 18=1.5yr, 24=2yr
  stringsAsFactors = FALSE
) |>
  dplyr::arrange(C, M, T) |>
  dplyr::mutate(
    param_id = sprintf(
      "C%s_M%s_T%s",
      sub("\\.", "", formatC(abs(C), format = "f", digits = 2)),
      sub("\\.", "", formatC(abs(M), format = "f", digits = 2)),
      formatC(T, width = 3, flag = "0")
    )
  )

MAX_IMPLOSION_RATE <- 0.15

#==============================================================================#
# 5C. Data Quality Thresholds — 05_CSI_Label.R
#==============================================================================#

## Maximum number of consecutive missing monthly returns allowed per stock.
## Stocks exceeding this trigger a pipeline halt — data quality assumption violated.
MAX_CONSECUTIVE_NA <- 3L

#==============================================================================#
# 6. Feature Engineering
#==============================================================================#

## Rolling window lengths in MONTHS (pipeline runs on monthly data)
WINDOW_SHORT <- 36L   # 3 years in months — minimum for meaningful rolling stats
WINDOW_LONG  <- 60L   # 5 years unchanged

## Reporting lag: months between fiscal year-end and public filing availability
## Applied in 06_Merge.R when joining Compustat to the label panel.
## Standard assumption: 3 months (10-K filed ~90 days after fiscal year-end).
REPORTING_LAG_MONTHS <- 3L

## Aggregation functions applied per fundamental over the rolling window.
ROLLING_STATS <- c(
  "mean",
  "min",
  "max",
  "median",
  "sd",
  "var",
  "mean_abs_diff",
  "median_abs_diff",
  "autocorr_lag1"
)

#==============================================================================#
# 7. Train / Validation / Out-of-Sample Split
#
#   Three-way split required for proper out-of-sample backtesting:
#     TRAIN  : used for cross-validated hyperparameter optimisation
#     TEST   : held out for model selection among all trained models
#     OOS    : never touched until final backtest — true out-of-sample
#
#   OOS period (2020–2024) deliberately includes:
#     - COVID crash (2020) — genuine market stress
#     - Rate cycle (2022)  — macro regime shift
#     - Recovery (2023–24) — full cycle coverage
#==============================================================================#

TRAIN_END  <- as.Date("2015-12-31")
TEST_START <- as.Date("2016-01-01")
TEST_END   <- as.Date("2019-12-31")
OOS_START  <- as.Date("2020-01-01")

SPLIT_GAP_MONTHS <- 0L

CV_FOLDS         <- 5L
CV_MIN_TRAIN_YRS <- 3L

#==============================================================================#
# 8. Modelling
#==============================================================================#

HPO_ITER        <- 50L
HPO_METRIC      <- "average_precision"   # Threshold-free, robust to imbalance
CLASS_WEIGHT    <- NULL
FPR_CONSTRAINTS <- c(0.03, 0.05)        # Recall evaluated at these fixed FPR levels

MODELS_TO_RUN <- c(
  "logistic_regression",   # Interpretable baseline
  "random_forest",
  "xgboost",
  "catboost",
  "lightgbm"
)

#==============================================================================#
# 9. Output / Plotting
#==============================================================================#

PLOT_WIDTH  <- 10
PLOT_HEIGHT <- 6
PLOT_DPI    <- 150

#==============================================================================#
# Confirm load
#==============================================================================#

cat("[config.R] Loaded.\n")
cat("  Period              :", format(START_DATE), "to", format(END_DATE), "\n")
cat("  Min lifetime        :", MIN_LIFETIME_YEARS, "years (applied in 06_Feature_Engineering.R)\n")
cat("  CSI base case       : C =", CSI_BASE$C,
    "| M =", CSI_BASE$M,
    "| T =", CSI_BASE$T, "months\n")
cat("  CSI grid size       :", nrow(CSI_GRID), "parameter combinations\n")
cat("  Data split          : Train <=", format(TRAIN_END),
    "| Test", format(TEST_START), "–", format(TEST_END),
    "| OOS >=", format(OOS_START), "\n")
cat("  Seed                :", SEED, "\n")
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
# 2. Directories
#==============================================================================#

DIR_ROOT      <- here::here()
DIR_CODE      <- file.path(DIR_ROOT, "01_Code")
DIR_DATA      <- file.path(DIR_ROOT, "02_Data")
DIR_OUTPUT    <- file.path(DIR_ROOT, "03_Output")

DIR_CRSP      <- file.path(DIR_DATA, "CRSP")
DIR_CRSP_RAW  <- file.path(DIR_CRSP, "Raw")
DIR_CRSP_PROC <- file.path(DIR_CRSP, "Processed")

DIR_COMP      <- file.path(DIR_DATA, "Compustat")
DIR_COMP_RAW  <- file.path(DIR_COMP, "Raw")
DIR_COMP_PROC <- file.path(DIR_COMP, "Processed")

DIR_MACRO     <- file.path(DIR_DATA, "Macro")
DIR_LABELS    <- file.path(DIR_DATA, "Labels")
DIR_FEATURES  <- file.path(DIR_DATA, "Features")

DIR_FIGURES   <- file.path(DIR_OUTPUT, "Figures")
DIR_TABLES    <- file.path(DIR_OUTPUT, "Tables")
DIR_MODELS    <- file.path(DIR_OUTPUT, "Models")

invisible(lapply(
  c(DIR_CRSP_RAW, DIR_CRSP_PROC,
    DIR_COMP_RAW, DIR_COMP_PROC,
    DIR_MACRO, DIR_LABELS, DIR_FEATURES,
    DIR_FIGURES, DIR_TABLES, DIR_MODELS),
  dir.create, showWarnings = FALSE, recursive = TRUE
))

#==============================================================================#
# 3. Date Range
#==============================================================================#

START_DATE <- as.Date("1993-01-01")
END_DATE   <- as.Date("2024-12-31")

#==============================================================================#
# 4. Universe Filters
#==============================================================================#

VALID_EXCHANGES    <- c("N", "A", "Q")    # NYSE, AMEX, NASDAQ
EXCLUDE_SECTYPES   <- c("FUND")
MIN_LIFETIME_YEARS <- 5
VALID_SHARETYPES   <- c("NS", NA)         # Normal shares; NA = legacy pre-classification
VALID_SUBTYPES     <- c("COM", NA)        # Common stock; NA = legacy

#==============================================================================#
# 5. CSI Label Parameters
#
#   DEFINITION USED IN THIS PIPELINE:
#   ---------------------------------
#   The paper defines a Catastrophic Stock Implosion via cumulative returns
#   but never specifies the reference point. We implement it as a DRAWDOWN
#   FROM PEAK, which is:
#
#     Drawdown(t) = Price(t) / max(Price(s), s <= t) - 1
#
#   This is economically cleaner than cumulative-from-first-observation:
#   it is reference-point agnostic, captures what a portfolio manager
#   observes, and avoids penalising stocks with long prior appreciation.
#
#   THREE PARAMETERS:
#
#     CSI_C : Drawdown threshold — initial crash triggers when
#             Drawdown(t) < CSI_C.
#             Interpretation: stock has fallen >= |CSI_C| from its all-time
#             peak up to and including week t.
#             Paper equivalent: C = -0.80
#
#     CSI_M : Maximum recovery allowed during the zombie period.
#             Measured as a return from the crash low:
#               Recovery(t) = Price(t) / Price(crash_low) - 1
#             The zombie period is broken (event disqualified) if
#               Recovery(t) > CSI_M at any point during the T-week window.
#             CSI_M = -0.20 means the stock must stay at least 20% below
#             the crash low price. Verify against Figure 1 of the paper.
#             Paper equivalent: M = -0.20
#
#     CSI_T : Minimum zombie period length in WEEKS.
#             Paper equivalent: T = 78 (~18 months)
#
#   BASE CASE (used for main results and model training):
#   -----------------------------------------------------
#   These replicate the paper's chosen parameters. Do not change these
#   for the primary analysis. Use the sensitivity grid below for
#   robustness checks.
#==============================================================================#

CSI_BASE <- list(
  C = -0.80,
  M = -0.20,
  T = 78L
)

#==============================================================================#
# 5B. CSI Sensitivity Grid
#
#   Full factorial grid across C, M, T.
#   Used by:
#     05_CSI_Label.R  — loops over all 27 combinations, saves one labelled
#                       dataset per parameterisation to DIR_LABELS
#     12_Robustness.R — compares label stability and downstream model
#                       sensitivity across the grid
#
#   Parameter ranges and rationale:
#
#   C: -0.60 (moderate crash) to -0.90 (near-wipeout).
#      Tighter (more negative) = fewer but cleaner CSI events.
#
#   M:  0.00 (any recovery breaks zombie period) to -0.30 (strict:
#      stock must stay 30% below crash low to remain in zombie state).
#      Less negative = more permissive = easier to break zombie period.
#
#   T: 52 weeks (1 year) to 104 weeks (2 years).
#      Longer = fewer events but higher confidence of true non-recovery.
#
#   Total: 3 x 3 x 3 = 27 parameter sets.
#   Each produces one label file: labels_<param_id>.rds
#==============================================================================#

CSI_GRID <- expand.grid(
  C = c(-0.60, -0.80, -0.90),
  M = c( 0.00, -0.20, -0.30),
  T = c(  52L,   78L,  104L),
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

## If any parameterisation labels more than this share of stocks CSI=1, assert.
## Paper achieved 6.6% under base case. 15% allows headroom for looser params.
MAX_IMPLOSION_RATE <- 0.15

#==============================================================================#
# 6. Feature Engineering
#==============================================================================#

WINDOW_SHORT <- 52L     # 1-year rolling window in weeks  (Approach 1)
WINDOW_LONG  <- 260L    # 5-year rolling window in weeks  (Approach 2 — paper best)

## Aggregation functions applied per fundamental over the rolling window.
## Each must be implemented in fn_rolling_stats.R
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
# 7. Train / Test Split
#==============================================================================#

TRAIN_END        <- as.Date("2018-12-31")
TEST_START       <- as.Date("2019-01-01")
SPLIT_GAP_MONTHS <- 0L

CV_FOLDS         <- 5L
CV_MIN_TRAIN_YRS <- 3L

#==============================================================================#
# 8. Modelling
#==============================================================================#

HPO_ITER     <- 50L
HPO_METRIC   <- "mcc"
CLASS_WEIGHT <- NULL

MODELS_TO_RUN <- c(
  "logistic_regression",
  "random_forest",
  "xgboost"
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
cat("  Min lifetime        :", MIN_LIFETIME_YEARS, "years\n")
cat("  CSI base case       : C =", CSI_BASE$C,
    "| M =", CSI_BASE$M,
    "| T =", CSI_BASE$T, "weeks\n")
cat("  CSI grid size       :", nrow(CSI_GRID), "parameter combinations\n")
cat("  Train / Test split  :", format(TRAIN_END), "/", format(TEST_START), "\n")
cat("  Seed                :", SEED, "\n")


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
  c(DIR_CRSP_RAW, DIR_CRSP_PROC, DIR_COMP_RAW, DIR_COMP_PROC,
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

#------------------------------------------------------------------------------#
# File paths — 06_Merge.R
#------------------------------------------------------------------------------#

PATH_PANEL_RAW <- file.path(DIR_PANEL, "panel_raw.rds")

#------------------------------------------------------------------------------#
# File paths — 06B_Feature_Eng.R
#
#   features_raw.rds  : full engineered features (~463)       [M3 input]
#   features_fund.rds : fundamentals only, no price features  [M1 + M2 VAE input]
#------------------------------------------------------------------------------#

PATH_FEATURES_RAW  <- file.path(DIR_FEATURES, "features_raw.rds")
PATH_FEATURES_FUND <- file.path(DIR_FEATURES, "features_fund.rds")

#------------------------------------------------------------------------------#
# File paths — 08B_Autoencoder.py
#
#   Run 08B with VAE_INPUT="fund" → features_latent_fund.parquet  [M2]
#   Run 08B with VAE_INPUT="raw"  → features_latent_raw.parquet   [M4]
#
#   M2: VAE trained on fund features only — tests whether VAE adds signal
#       over raw fundamentals. Compare M1 vs M2.
#   M4: VAE trained on full raw features — tests VAE compression of all
#       signals. Compare M3 vs M4.
#------------------------------------------------------------------------------#

PATH_FEATURES_LATENT_FUND <- file.path(DIR_FEATURES, "features_latent_fund.parquet")
PATH_FEATURES_LATENT_RAW  <- file.path(DIR_FEATURES, "features_latent_raw.parquet")

## Legacy alias for any downstream script using PATH_FEATURES_LATENT directly
## Points to fund latent (M2) as the primary/default VAE output
PATH_FEATURES_LATENT <- PATH_FEATURES_LATENT_FUND

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

PATH_INDEX_RETURNS    <- file.path(DIR_TABLES, "index_returns.rds")
PATH_BACKTEST_SUMMARY <- file.path(DIR_TABLES, "backtest_summary.rds")

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

VALID_EXCHANGES    <- c("N", "A", "Q")
EXCLUDE_SECTYPES   <- c("FUND")
MIN_LIFETIME_YEARS <- 5
VALID_SHARETYPES   <- c("NS", NA)
VALID_SUBTYPES     <- c("COM", NA)

#==============================================================================#
# 5. CSI Label Parameters
#==============================================================================#

CSI_BASE <- list(C = -0.80, M = -0.20, T = 18L)

#==============================================================================#
# 5B. CSI Sensitivity Grid
#==============================================================================#

CSI_GRID <- expand.grid(
  C = c(-0.60, -0.80, -0.90),
  M = c( 0.00, -0.20, -0.30),
  T = c(  12L,   18L,   24L),
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
# 5C. Data Quality Thresholds
#==============================================================================#

MAX_CONSECUTIVE_NA <- 3L

#==============================================================================#
# 6. Feature Engineering
#==============================================================================#

WINDOW_SHORT         <- 36L
WINDOW_LONG          <- 60L
REPORTING_LAG_MONTHS <- 3L

ROLLING_STATS <- c(
  "mean", "min", "max", "median", "sd", "var",
  "mean_abs_diff", "median_abs_diff", "autocorr_lag1"
)

#==============================================================================#
# 7. Train / Validation / Out-of-Sample Split
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
HPO_METRIC      <- "average_precision"
CLASS_WEIGHT    <- NULL
FPR_CONSTRAINTS <- c(0.03, 0.05)

MODELS_TO_RUN <- c(
  "logistic_regression",
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
cat("  Period         :", format(START_DATE), "to", format(END_DATE), "\n")
cat("  CSI base case  : C =", CSI_BASE$C, "| M =", CSI_BASE$M,
    "| T =", CSI_BASE$T, "months\n")
cat("  Data split     : Train <=", format(TRAIN_END),
    "| Test", format(TEST_START), "–", format(TEST_END),
    "| OOS >=", format(OOS_START), "\n")
cat("  Feature paths  :\n")
cat("    RAW          :", basename(PATH_FEATURES_RAW),  "\n")
cat("    FUND         :", basename(PATH_FEATURES_FUND), "\n")
cat("    LATENT FUND  :", basename(PATH_FEATURES_LATENT_FUND), "\n")
cat("    LATENT RAW   :", basename(PATH_FEATURES_LATENT_RAW),  "\n")
cat("  Seed           :", SEED, "\n")

## ============================================================================
## Figure directory structure — add this block to config.R
## ============================================================================
## Call this once at the start of any script that saves figures.
## All subdirectories are created automatically if they don't exist.

fn_setup_figure_dirs <- function(base_dir = DIR_FIGURES) {
  
  dirs <- c(
    ## Label diagnostics
    file.path(base_dir, "05_labels"),
    
    ## Model outputs — one subfolder per model
    file.path(base_dir, "09_models", "m1"),
    file.path(base_dir, "09_models", "m3"),
    file.path(base_dir, "09_models", "b1"),
    
    ## Index construction and backtest
    file.path(base_dir, "11_index"),
    
    ## Exclusion diagnostics
    file.path(base_dir, "12_evaluation"),
    
    ## Robustness checks — one subfolder per part
    file.path(base_dir, "13_robustness", "partA"),
    file.path(base_dir, "13_robustness", "partB"),
    file.path(base_dir, "13_robustness", "partC"),
    file.path(base_dir, "13_robustness", "partD"),
    file.path(base_dir, "13_robustness", "partE"),
    
    ## Exploratory / validation scripts
    file.path(base_dir, "explore")
  )
  
  created <- 0L
  for (d in dirs) {
    if (!dir.exists(d)) {
      dir.create(d, recursive = TRUE, showWarnings = FALSE)
      created <- created + 1L
    }
  }
  
  if (created > 0L)
    cat(sprintf("  [figures] Created %d new subdirectories under %s\n",
                created, base_dir))
  
  ## Return named list of paths for use in scripts
  invisible(list(
    labels      = file.path(base_dir, "05_labels"),
    m1          = file.path(base_dir, "09_models", "m1"),
    m3          = file.path(base_dir, "09_models", "m3"),
    b1          = file.path(base_dir, "09_models", "b1"),
    index       = file.path(base_dir, "11_index"),
    evaluation  = file.path(base_dir, "12_evaluation"),
    robustness  = file.path(base_dir, "13_robustness"),
    rob_a       = file.path(base_dir, "13_robustness", "partA"),
    rob_b       = file.path(base_dir, "13_robustness", "partB"),
    rob_c       = file.path(base_dir, "13_robustness", "partC"),
    rob_d       = file.path(base_dir, "13_robustness", "partD"),
    rob_e       = file.path(base_dir, "13_robustness", "partE"),
    explore     = file.path(base_dir, "explore")
  ))
}

## Path helpers — use these instead of file.path(DIR_FIGURES, "xxx.png")
## Usage: FIG$index("index_cumulative_ew.png")
##        FIG$m1("m1_pr_curve.png")
##        FIG$rob_b("robust_tree_plot.png")

## Also define PATH_LABELS_BUCKET for 05B output
PATH_LABELS_BUCKET <- file.path(DIR_LABELS, "labels_bucket.rds")

## B1 prediction output paths (parallel to ag_fund/)
DIR_TABLES_B1 <- file.path(DIR_TABLES, "ag_bucket")
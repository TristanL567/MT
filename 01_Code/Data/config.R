#==============================================================================#
#==== config.R ================================================================#
#==== Single Source of Truth — All Pipeline Parameters ========================#
#==============================================================================#
#
# PURPOSE:
#   Every threshold, date, seed, path, and figure directory used anywhere in
#   the pipeline is declared here and nowhere else. Source this file at the
#   top of every pipeline script with: source("config.R")
#
# PIPELINE OVERVIEW:
#   01_Universe.R            CRSP universe construction
#   02_Prices.R              Monthly/weekly prices + delisting returns
#   03_Fundamentals.R        Compustat annual fundamentals + CCM link
#   04_Macro.R               FRED macro variables
#   05_CSI_Label.R           CSI event labels (base + 27-parameter grid)
#   05B_Bucket_Labels.R      5-year forward CAGR bucket labels
#   05C_Structural_Labels.R  Combined CSI + bucket structural quality labels
#   06_Merge.R               Panel merge (prices + fundamentals + macro)
#   06B_Feature_Eng.R        Feature engineering → features_raw / features_fund
#   07_Feature_Sel.R         Feature selection diagnostics
#   08_Split.R               Train / test / OOS split construction
#   08B_Autoencoder.py       Beta-VAE latent features (M2/M4)
#   09C_AutoGluon.py         AutoML training (M1/M3/B1-bucket/B1-structural)
#   10_Evaluate.R            Model evaluation (AUC, AP, decile tables)
#   11_Results.R             Index construction (S1–S6, benchmark)
#   12_Evaluation.R          Index diagnostics (exclusion, sector, TE, turnover)
#   13_Robustness.R          Robustness checks Parts A–E
#   14_Comparison.R          Comparison vs naive benchmarks (low-vol, quality)
#
#==============================================================================#

suppressPackageStartupMessages(library(lubridate))

#==============================================================================#
# 1. Reproducibility
#==============================================================================#

SEED <- 123L
set.seed(SEED)

#==============================================================================#
# 2. Root Directories
#==============================================================================#

DIR_ROOT   <- here::here()
DIR_CODE   <- file.path(DIR_ROOT, "01_Code")
DIR_DATA   <- file.path(DIR_ROOT, "02_Data")
DIR_OUTPUT <- file.path(DIR_ROOT, "03_Output")

#==============================================================================#
# 3. Data Directories
#==============================================================================#

DIR_CRSP      <- file.path(DIR_DATA, "CRSP")
DIR_CRSP_RAW  <- file.path(DIR_CRSP, "Raw")
DIR_CRSP_PROC <- file.path(DIR_CRSP, "Processed")

DIR_COMP      <- file.path(DIR_DATA, "Compustat")
DIR_COMP_RAW  <- file.path(DIR_COMP, "Raw")
DIR_COMP_PROC <- file.path(DIR_COMP, "Processed")

DIR_MACRO    <- file.path(DIR_DATA, "Macro")
DIR_LABELS   <- file.path(DIR_DATA, "Labels")
DIR_FEATURES <- file.path(DIR_DATA, "Features")
DIR_PANEL    <- file.path(DIR_DATA, "Panel")

#==============================================================================#
# 4. Output Directories
#==============================================================================#

DIR_FIGURES <- file.path(DIR_OUTPUT, "Figures")
DIR_TABLES  <- file.path(DIR_OUTPUT, "Tables")
DIR_MODELS  <- file.path(DIR_OUTPUT, "Models")

## AutoGluon model subdirectories — one per model variant
DIR_TABLES_M1         <- file.path(DIR_TABLES, "ag_fund")
DIR_TABLES_M3         <- file.path(DIR_TABLES, "ag_raw")
DIR_TABLES_B1_BUCKET  <- file.path(DIR_TABLES, "ag_bucket")
DIR_TABLES_B1_STRUCT  <- file.path(DIR_TABLES, "ag_structural")

#==============================================================================#
# 5. Create All Directories
#==============================================================================#

.dirs_to_create <- c(
  DIR_CRSP_RAW, DIR_CRSP_PROC, DIR_COMP_RAW, DIR_COMP_PROC,
  DIR_MACRO, DIR_LABELS, DIR_FEATURES, DIR_PANEL,
  DIR_FIGURES, DIR_TABLES, DIR_MODELS,
  DIR_TABLES_M1, DIR_TABLES_M3, DIR_TABLES_B1_BUCKET, DIR_TABLES_B1_STRUCT
)
invisible(lapply(.dirs_to_create, dir.create,
                 showWarnings = FALSE, recursive = TRUE))
rm(.dirs_to_create)

#==============================================================================#
# 6. Figure Directory Helper
#==============================================================================#
#
# LAYOUT:
#   Figures/
#   ├── 01_universe/
#   ├── 02_prices/
#   ├── 03_fundamentals/
#   ├── 04_macro/
#   ├── 05_labels/
#   │   ├── csi/               CSI label diagnostics
#   │   ├── bucket/            5yr CAGR bucket diagnostics
#   │   └── structural/        combined label diagnostics
#   ├── 06_features/
#   ├── 09_models/
#   │   ├── m1/                M1 (CSI, fundamentals only)
#   │   ├── m3/                M3 (CSI, full features)
#   │   ├── b1_bucket/         B1 (5yr CAGR bucket)
#   │   └── b1_structural/     B1 (combined structural label)
#   ├── 11_index/
#   ├── 12_evaluation/
#   ├── 13_robustness/
#   │   ├── partA/ … partE/
#   ├── 14_comparison/
#   └── explore/
#
# USAGE:  FIGS <- fn_setup_figure_dirs()
#         ggsave(file.path(FIGS$csi,  "csi_events_per_year.png"), p, ...)
#         ggsave(file.path(FIGS$m1,   "m1_pr_curve.png"), p, ...)
#         ggsave(file.path(FIGS$rob_e,"conc_cumulative_oos.png"), p, ...)
#
#==============================================================================#

fn_setup_figure_dirs <- function(base_dir = DIR_FIGURES) {
  
  dirs <- c(
    file.path(base_dir, "01_universe"),
    file.path(base_dir, "02_prices"),
    file.path(base_dir, "03_fundamentals"),
    file.path(base_dir, "04_macro"),
    file.path(base_dir, "05_labels", "csi"),
    file.path(base_dir, "05_labels", "bucket"),
    file.path(base_dir, "05_labels", "structural"),
    file.path(base_dir, "06_features"),
    file.path(base_dir, "09_models", "m1"),
    file.path(base_dir, "09_models", "m3"),
    file.path(base_dir, "09_models", "b1_bucket"),
    file.path(base_dir, "09_models", "b1_structural"),
    file.path(base_dir, "11_index"),
    file.path(base_dir, "12_evaluation"),
    file.path(base_dir, "13_robustness", "partA"),
    file.path(base_dir, "13_robustness", "partB"),
    file.path(base_dir, "13_robustness", "partC"),
    file.path(base_dir, "13_robustness", "partD"),
    file.path(base_dir, "13_robustness", "partE"),
    file.path(base_dir, "14_comparison"),
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
    cat(sprintf("  [figures] %d new subdirectories created under %s\n",
                created, base_dir))
  
  invisible(list(
    universe      = file.path(base_dir, "01_universe"),
    prices        = file.path(base_dir, "02_prices"),
    fundamentals  = file.path(base_dir, "03_fundamentals"),
    macro         = file.path(base_dir, "04_macro"),
    labels        = file.path(base_dir, "05_labels"),
    csi           = file.path(base_dir, "05_labels", "csi"),
    bucket        = file.path(base_dir, "05_labels", "bucket"),
    structural    = file.path(base_dir, "05_labels", "structural"),
    features      = file.path(base_dir, "06_features"),
    models        = file.path(base_dir, "09_models"),
    m1            = file.path(base_dir, "09_models", "m1"),
    m3            = file.path(base_dir, "09_models", "m3"),
    b1_bucket     = file.path(base_dir, "09_models", "b1_bucket"),
    b1_structural = file.path(base_dir, "09_models", "b1_structural"),
    index         = file.path(base_dir, "11_index"),
    evaluation    = file.path(base_dir, "12_evaluation"),
    robustness    = file.path(base_dir, "13_robustness"),
    rob_a         = file.path(base_dir, "13_robustness", "partA"),
    rob_b         = file.path(base_dir, "13_robustness", "partB"),
    rob_c         = file.path(base_dir, "13_robustness", "partC"),
    rob_d         = file.path(base_dir, "13_robustness", "partD"),
    rob_e         = file.path(base_dir, "13_robustness", "partE"),
    comparison    = file.path(base_dir, "14_comparison"),
    explore       = file.path(base_dir, "explore")
  ))
}

#==============================================================================#
# 7. File Paths — Data Pipeline
#==============================================================================#

## 01_Universe.R
PATH_UNIVERSE_RAW <- file.path(DIR_CRSP_RAW,  "universe_raw.rds")
PATH_UNIVERSE     <- file.path(DIR_CRSP_PROC, "universe.rds")

## 02_Prices.R
PATH_PRICES_DAILY_RAW   <- file.path(DIR_CRSP_RAW,  "prices_daily_raw.rds")
PATH_PRICES_MONTHLY_RAW <- file.path(DIR_CRSP_RAW,  "prices_monthly_raw.rds")
PATH_PRICES_WEEKLY      <- file.path(DIR_CRSP_PROC, "prices_weekly.rds")
PATH_PRICES_MONTHLY     <- file.path(DIR_CRSP_PROC, "prices_monthly.rds")
PATH_DELISTING          <- file.path(DIR_CRSP_RAW,  "delisting_raw.rds")

## 03_Fundamentals.R
PATH_FUNDAMENTALS_RAW <- file.path(DIR_COMP_RAW,  "fundamentals_raw.rds")
PATH_FUNDAMENTALS     <- file.path(DIR_COMP_PROC, "fundamentals.rds")
PATH_CCM_LINK         <- file.path(DIR_COMP_RAW,  "ccm_link_raw.rds")

## 04_Macro.R
PATH_MACRO_RAW     <- file.path(DIR_MACRO, "macro_raw.rds")
PATH_MACRO_MONTHLY <- file.path(DIR_MACRO, "macro_monthly.rds")

## 05_CSI_Label.R
PATH_LABELS_BASE <- file.path(DIR_LABELS, "labels_base.rds")
PATH_LABELS_GRID <- file.path(DIR_LABELS, "labels_all_grid.rds")
PATH_LABELS_DIAG <- file.path(DIR_LABELS, "csi_diagnostics.rds")

## 05B_Bucket_Labels.R
PATH_LABELS_BUCKET <- file.path(DIR_LABELS, "labels_bucket.rds")

## 05C_Structural_Labels.R
PATH_LABELS_STRUCTURAL <- file.path(DIR_LABELS, "labels_structural.rds")

## 06_Merge.R
PATH_PANEL_RAW <- file.path(DIR_PANEL, "panel_raw.rds")

## 06B_Feature_Eng.R
PATH_FEATURES_RAW  <- file.path(DIR_FEATURES, "features_raw.rds")
PATH_FEATURES_FUND <- file.path(DIR_FEATURES, "features_fund.rds")

## 08B_Autoencoder.py
PATH_FEATURES_LATENT_FUND <- file.path(DIR_FEATURES, "features_latent_fund.parquet")
PATH_FEATURES_LATENT_RAW  <- file.path(DIR_FEATURES, "features_latent_raw.parquet")
PATH_FEATURES_LATENT      <- PATH_FEATURES_LATENT_FUND

## 07_Feature_Sel.R
PATH_FEATURES_SELECTED <- file.path(DIR_FEATURES, "features_selected.rds")

## 08_Split.R
PATH_SPLITS <- file.path(DIR_FEATURES, "splits.rds")

#==============================================================================#
# 8. File Paths — Output
#==============================================================================#

## 10_Evaluate.R
PATH_EVAL_RESULTS <- file.path(DIR_TABLES, "evaluation_results.rds")

## 11_Results.R
PATH_INDEX_WEIGHTS <- file.path(DIR_TABLES, "index_weights.rds")
PATH_INDEX_RETURNS <- file.path(DIR_TABLES, "index_returns.rds")
PATH_INDEX_PERF    <- file.path(DIR_TABLES, "index_performance.rds")

## 12_Evaluation.R
PATH_INDEX_EXCLUSION <- file.path(DIR_TABLES, "index_csi_avoidance.rds")

## 13_Robustness.R
PATH_ROBUST_GRID   <- file.path(DIR_TABLES, "robust_grid_performance.rds")
PATH_ROBUST_TREE   <- file.path(DIR_TABLES, "robust_recovery_classifier.rds")
PATH_ROBUST_INDEX  <- file.path(DIR_TABLES, "robust_index_returns.rds")
PATH_ROBUST_TIERED <- file.path(DIR_TABLES, "robust_tiered_results.rds")
PATH_ROBUST_CONC   <- file.path(DIR_TABLES, "robust_conc_returns.rds")
PATH_ROBUST_CONC_P <- file.path(DIR_TABLES, "robust_conc_performance.rds")

## 14_Comparison.R
PATH_COMPARISON_RETURNS <- file.path(DIR_TABLES, "comparison_returns.rds")
PATH_COMPARISON_PERF    <- file.path(DIR_TABLES, "comparison_performance.rds")

#==============================================================================#
# 9. Date Range
#==============================================================================#

START_DATE <- as.Date("1993-01-01")
END_DATE   <- as.Date("2024-12-31")

#==============================================================================#
# 10. Universe Construction Parameters
#==============================================================================#

VALID_EXCHANGES     <- c("N", "A", "Q")
EXCLUDE_SECTYPES    <- c("FUND")
VALID_SHARETYPES    <- c("NS", NA)
VALID_SUBTYPES      <- c("COM", NA)
MIN_LIFETIME_YEARS  <- 5L

UNIVERSE_SIZE       <- 3000L
UNIVERSE_MIN_MKTCAP <- 100      ## $M

#==============================================================================#
# 11. CSI Label Parameters
#==============================================================================#

CSI_BASE <- list(C = -0.80, M = -0.20, T = 18L)

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
      formatC(T, width = 3L, flag = "0")
    )
  )

MAX_IMPLOSION_RATE <- 0.15

#==============================================================================#
# 12. Bucket Label Parameters (05B / 05C)
#==============================================================================#

BUCKET_FWD_YEARS      <- 5L
BUCKET_MIN_MONTHS     <- 48L
BUCKET_LOSER_THRESH   <- -0.02   ## CAGR < -2%  → terminal loser (y=1)
BUCKET_PHOENIX_THRESH <-  0.00   ## CAGR >= 0%  → phoenix        (y=0)
BUCKET_LAST_YEAR      <- year(END_DATE) - BUCKET_FWD_YEARS  ## 2019

#==============================================================================#
# 13. Data Quality Thresholds
#==============================================================================#

MAX_CONSECUTIVE_NA <- 3L

#==============================================================================#
# 14. Feature Engineering Parameters
#==============================================================================#

WINDOW_SHORT         <- 36L
WINDOW_LONG          <- 60L
REPORTING_LAG_MONTHS <- 3L

ROLLING_STATS <- c(
  "mean", "min", "max", "median", "sd", "var",
  "mean_abs_diff", "median_abs_diff", "autocorr_lag1"
)

#==============================================================================#
# 15. Train / Test / OOS Split
#==============================================================================#

TRAIN_END  <- as.Date("2015-12-31")
TEST_START <- as.Date("2016-01-01")
TEST_END   <- as.Date("2019-12-31")
OOS_START  <- as.Date("2020-01-01")

TRAIN_END_YR  <- 2015L
TEST_START_YR <- 2016L
TEST_END_YR   <- 2019L
OOS_START_YR  <- 2020L

SPLIT_GAP_MONTHS <- 0L
CV_FOLDS         <- 5L
CV_MIN_TRAIN_YRS <- 3L

#==============================================================================#
# 16. Modelling Parameters
#==============================================================================#

HPO_ITER        <- 50L
HPO_METRIC      <- "average_precision"
CLASS_WEIGHT    <- NULL
FPR_CONSTRAINTS <- c(0.03, 0.05)

MODELS_TO_RUN <- c(
  "logistic_regression", "random_forest",
  "xgboost", "catboost", "lightgbm"
)

AG_TIME_LIMIT_MAIN <- 3600L
AG_TIME_LIMIT_CV   <- 900L

#==============================================================================#
# 17. Portfolio Construction Parameters
#==============================================================================#

EXCLUSION_RATE      <- 0.05    ## M1 top 5% flagged for exclusion

PORT_CONC_SIZE_C1   <- 200L    ## C1 portfolio size
PORT_CONC_SIZE_C2   <- 100L    ## C2 portfolio size
PORT_M1_VETO_RATE   <- 0.10    ## C3 M1 veto threshold
PORT_S6_EXCL_RATE   <- 0.20    ## S6 B1 exclusion rate
PORT_RF_ANNUAL      <- 0.03    ## risk-free rate for Sharpe

## Altman Z-score zombie threshold (from recovery classifier Part B)
ZOMBIE_Z2_THRESH    <- -2.768814

#==============================================================================#
# 18. Plotting Parameters
#==============================================================================#

PLOT_WIDTH  <- 10
PLOT_HEIGHT <- 6
PLOT_DPI    <- 150

## Consistent strategy colours across all plots
STRAT_COLOURS <- c(
  bench         = "#9E9E9E",
  s1            = "#2196F3",
  s4            = "#4CAF50",
  s5            = "#9C27B0",
  s6            = "#FF9800",
  c1_bucket     = "#9C27B0",
  c1_structural = "#E91E63",
  c2            = "#CE93D8",
  c3            = "#F44336",
  low_vol       = "#00BCD4",
  quality       = "#FF5722"
)

STRAT_LABELS <- c(
  bench         = "Benchmark (EW 3000)",
  s1            = "S1: M1 Exclusion (5%)",
  s4            = "S4: M1 + Zombie Filter",
  s5            = "S5: Tiered Threshold",
  s6            = "S6: B1 Exclusion (20%)",
  c1_bucket     = "C1: B1-Bucket Long 200",
  c1_structural = "C1: B1-Structural Long 200",
  c2            = "C2: B1 Long 100",
  c3            = "C3: B1-Structural + M1 Veto",
  low_vol       = "Low-Vol 200",
  quality       = "Quality 200 (Altman Z)"
)

#==============================================================================#
# 19. Confirm Load
#==============================================================================#

cat("[config.R] Loaded.\n")
cat(sprintf("  Root           : %s\n", DIR_ROOT))
cat(sprintf("  Period         : %s to %s\n",
            format(START_DATE), format(END_DATE)))
cat(sprintf("  CSI base case  : C = %.2f | M = %.2f | T = %d months\n",
            CSI_BASE$C, CSI_BASE$M, CSI_BASE$T))
cat(sprintf("  Data split     : Train <= %s | Test %s–%s | OOS >= %s\n",
            format(TRAIN_END), format(TEST_START),
            format(TEST_END),  format(OOS_START)))
cat(sprintf("  Bucket         : %d-yr fwd | loser < %.0f%% | phoenix >= %.0f%%\n",
            BUCKET_FWD_YEARS,
            BUCKET_LOSER_THRESH*100, BUCKET_PHOENIX_THRESH*100))
cat(sprintf("  Universe       : Top %d | min $%dM mktcap\n",
            UNIVERSE_SIZE, UNIVERSE_MIN_MKTCAP))
cat(sprintf("  Seed           : %d\n", SEED))
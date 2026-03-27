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
#   08B_Autoencoder.py       Beta-VAE latent features (M2/M4/B2/B4/S2/S4)
#   09C_AutoGluon.py         AutoML training (M1–M4, B1–B4, S1–S4)
#   10_Evaluate.R            Model evaluation (AUC, AP, decile tables, PR curves)
#   11_Results.R             Index construction (all strategies)
#   12_Evaluation.R          Index diagnostics (exclusion, sector, TE, turnover)
#   13_Robustness.R          Robustness checks Parts A–E
#   14_Comparison.R          Comparison vs naive benchmarks (low-vol, quality)
#
# OUTPUT DIRECTORY STRUCTURE:
#
#   03_Output/
#   ├── Tables/
#   │   ├── ag_fund/               M1 predictions + leaderboard
#   │   ├── ag_latent_fund/        M2
#   │   ├── ag_raw/                M3
#   │   ├── ag_latent_raw/         M4
#   │   ├── ag_bucket/             B1
#   │   ├── ag_bucket_latent_fund/ B2
#   │   ├── ag_bucket_raw/         B3
#   │   ├── ag_bucket_latent_raw/  B4
#   │   ├── ag_structural/         S1
#   │   ├── ag_structural_latent_fund/ S2
#   │   ├── ag_structural_raw/     S3
#   │   └── ag_structural_latent_raw/  S4
#   │
#   ├── Figures/
#   │   ├── 01_universe/
#   │   ├── 02_prices/
#   │   ├── 03_fundamentals/
#   │   ├── 04_macro/
#   │   ├── 05_labels/
#   │   │   ├── csi/
#   │   │   ├── bucket/
#   │   │   └── structural/
#   │   ├── 06_features/
#   │   │
#   │   ├── 09_models/             ← all model evaluation figures
#   │   │   ├── comparison/        cross-model comparison (all 12 side by side)
#   │   │   ├── csi/               Track 1 overview
#   │   │   │   ├── m1/
#   │   │   │   ├── m2/
#   │   │   │   ├── m3/
#   │   │   │   └── m4/
#   │   │   ├── bucket/            Track 2 overview
#   │   │   │   ├── b1/
#   │   │   │   ├── b2/
#   │   │   │   ├── b3/
#   │   │   │   └── b4/
#   │   │   └── structural/        Track 3 overview
#   │   │       ├── s1/
#   │   │       ├── s2/
#   │   │       ├── s3/
#   │   │       └── s4/
#   │   │
#   │   ├── 11_index/              ← index construction + returns figures
#   │   │   ├── general/           overall benchmark vs all strategies
#   │   │   ├── csi_track/         CSI-model-based index strategies
#   │   │   ├── bucket_track/      bucket-model-based index strategies
#   │   │   ├── structural_track/  structural-model-based index strategies
#   │   │   └── concentrated/      C1/C2/C3 concentrated portfolios
#   │   │
#   │   ├── 12_evaluation/         index diagnostics (exclusion, sector, TE)
#   │   ├── 13_robustness/
#   │   │   ├── partA/
#   │   │   ├── partB/
#   │   │   ├── partC/
#   │   │   ├── partD/
#   │   │   └── partE/
#   │   ├── 14_comparison/         vs naive benchmarks
#   │   └── explore/
#   │
#   └── Models/
#       ├── AutoGluon/             ag_fund/, ag_raw/, ... (one per model)
#       └── VAE/
#           ├── fund/
#           └── raw/
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
# 4. Output Directories — Tables
#==============================================================================#

DIR_FIGURES <- file.path(DIR_OUTPUT, "Figures")
DIR_TABLES  <- file.path(DIR_OUTPUT, "Tables")
DIR_MODELS  <- file.path(DIR_OUTPUT, "Models")

## ── AutoGluon prediction + leaderboard output — one directory per model ─────
## Naming mirrors MODEL key in 09C_AutoGluon.py (ag_{MODEL})

## Track 1: CSI
DIR_TABLES_M1 <- file.path(DIR_TABLES, "ag_fund")
DIR_TABLES_M2 <- file.path(DIR_TABLES, "ag_latent_fund")
DIR_TABLES_M3 <- file.path(DIR_TABLES, "ag_raw")
DIR_TABLES_M4 <- file.path(DIR_TABLES, "ag_latent_raw")

## Track 2: Bucket
DIR_TABLES_B1 <- file.path(DIR_TABLES, "ag_bucket")
DIR_TABLES_B2 <- file.path(DIR_TABLES, "ag_bucket_latent_fund")
DIR_TABLES_B3 <- file.path(DIR_TABLES, "ag_bucket_raw")
DIR_TABLES_B4 <- file.path(DIR_TABLES, "ag_bucket_latent_raw")

## Track 3: Structural
DIR_TABLES_S1 <- file.path(DIR_TABLES, "ag_structural")
DIR_TABLES_S2 <- file.path(DIR_TABLES, "ag_structural_latent_fund")
DIR_TABLES_S3 <- file.path(DIR_TABLES, "ag_structural_raw")
DIR_TABLES_S4 <- file.path(DIR_TABLES, "ag_structural_latent_raw")

## Lookup list — index by model key for programmatic access
## Usage: DIR_TABLES_MODEL[["m1"]]  or  DIR_TABLES_MODEL[[MODEL_KEY]]
DIR_TABLES_MODEL <- list(
  m1 = DIR_TABLES_M1, m2 = DIR_TABLES_M2,
  m3 = DIR_TABLES_M3, m4 = DIR_TABLES_M4,
  b1 = DIR_TABLES_B1, b2 = DIR_TABLES_B2,
  b3 = DIR_TABLES_B3, b4 = DIR_TABLES_B4,
  s1 = DIR_TABLES_S1, s2 = DIR_TABLES_S2,
  s3 = DIR_TABLES_S3, s4 = DIR_TABLES_S4
)

#==============================================================================#
# 5. Create All Directories
#==============================================================================#

.dirs_to_create <- c(
  ## Data
  DIR_CRSP_RAW, DIR_CRSP_PROC,
  DIR_COMP_RAW, DIR_COMP_PROC,
  DIR_MACRO, DIR_LABELS, DIR_FEATURES, DIR_PANEL,
  ## Output root
  DIR_FIGURES, DIR_TABLES, DIR_MODELS,
  ## AutoGluon table directories — all 12 models
  DIR_TABLES_M1, DIR_TABLES_M2, DIR_TABLES_M3, DIR_TABLES_M4,
  DIR_TABLES_B1, DIR_TABLES_B2, DIR_TABLES_B3, DIR_TABLES_B4,
  DIR_TABLES_S1, DIR_TABLES_S2, DIR_TABLES_S3, DIR_TABLES_S4
)
invisible(lapply(.dirs_to_create, dir.create,
                 showWarnings = FALSE, recursive = TRUE))
rm(.dirs_to_create)

#==============================================================================#
# 6. Figure Directory Setup
#==============================================================================#
#
# fn_setup_figure_dirs() creates the full figure subdirectory tree and returns
# a named list of paths for use in ggsave() calls throughout the pipeline.
#
# LAYOUT:
#
#   Figures/
#   ├── 01_universe/
#   ├── 02_prices/
#   ├── 03_fundamentals/
#   ├── 04_macro/
#   ├── 05_labels/
#   │   ├── csi/
#   │   ├── bucket/
#   │   └── structural/
#   ├── 06_features/
#   │
#   ├── 09_models/
#   │   ├── comparison/          all-model comparison plots (AUC/AP table, PR curves)
#   │   ├── csi/                 Track 1 summary
#   │   │   ├── m1/              M1 individual model figures
#   │   │   ├── m2/
#   │   │   ├── m3/
#   │   │   └── m4/
#   │   ├── bucket/              Track 2 summary
#   │   │   ├── b1/
#   │   │   ├── b2/
#   │   │   ├── b3/
#   │   │   └── b4/
#   │   └── structural/          Track 3 summary
#   │       ├── s1/
#   │       ├── s2/
#   │       ├── s3/
#   │       └── s4/
#   │
#   ├── 11_index/
#   │   ├── general/             benchmark vs all strategies, summary tables
#   │   ├── csi_track/           CSI-model index strategies (S1–S4 type)
#   │   ├── bucket_track/        bucket-model index strategies
#   │   ├── structural_track/    structural-model index strategies
#   │   └── concentrated/        C1/C2/C3 concentrated portfolios
#   │
#   ├── 12_evaluation/           exclusion diagnostics, sector, TE, turnover
#   ├── 13_robustness/
#   │   ├── partA/ … partE/
#   ├── 14_comparison/           vs naive benchmarks (low-vol, quality)
#   └── explore/
#
# USAGE:
#   FIGS <- fn_setup_figure_dirs()
#   ggsave(file.path(FIGS$m1,           "pr_curve.png"),         ...)
#   ggsave(file.path(FIGS$model_compare,"auc_ap_table.png"),     ...)
#   ggsave(file.path(FIGS$index_general,"cumulative_returns.png"),...)
#   ggsave(file.path(FIGS$index_conc,   "c1_vs_benchmark.png"),  ...)
#   ggsave(file.path(FIGS$rob_e,        "conc_oos_returns.png"), ...)
#
#==============================================================================#

fn_setup_figure_dirs <- function(base_dir = DIR_FIGURES) {
  
  dirs <- c(
    ## Data pipeline
    file.path(base_dir, "01_universe"),
    file.path(base_dir, "02_prices"),
    file.path(base_dir, "03_fundamentals"),
    file.path(base_dir, "04_macro"),
    
    ## Labels
    file.path(base_dir, "05_labels", "csi"),
    file.path(base_dir, "05_labels", "bucket"),
    file.path(base_dir, "05_labels", "structural"),
    
    ## Features
    file.path(base_dir, "06_features"),
    
    ## Models — cross-model comparison
    file.path(base_dir, "09_models", "comparison"),
    
    ## Models — Track 1: CSI
    file.path(base_dir, "09_models", "csi"),
    file.path(base_dir, "09_models", "csi", "m1"),
    file.path(base_dir, "09_models", "csi", "m2"),
    file.path(base_dir, "09_models", "csi", "m3"),
    file.path(base_dir, "09_models", "csi", "m4"),
    
    ## Models — Track 2: Bucket
    file.path(base_dir, "09_models", "bucket"),
    file.path(base_dir, "09_models", "bucket", "b1"),
    file.path(base_dir, "09_models", "bucket", "b2"),
    file.path(base_dir, "09_models", "bucket", "b3"),
    file.path(base_dir, "09_models", "bucket", "b4"),
    
    ## Models — Track 3: Structural
    file.path(base_dir, "09_models", "structural"),
    file.path(base_dir, "09_models", "structural", "s1"),
    file.path(base_dir, "09_models", "structural", "s2"),
    file.path(base_dir, "09_models", "structural", "s3"),
    file.path(base_dir, "09_models", "structural", "s4"),
    
    ## Index construction + returns
    file.path(base_dir, "11_index", "general"),
    file.path(base_dir, "11_index", "csi_track"),
    file.path(base_dir, "11_index", "bucket_track"),
    file.path(base_dir, "11_index", "structural_track"),
    file.path(base_dir, "11_index", "concentrated"),
    
    ## Index diagnostics
    file.path(base_dir, "12_evaluation"),
    
    ## Robustness
    file.path(base_dir, "13_robustness", "partA"),
    file.path(base_dir, "13_robustness", "partB"),
    file.path(base_dir, "13_robustness", "partC"),
    file.path(base_dir, "13_robustness", "partD"),
    file.path(base_dir, "13_robustness", "partE"),
    
    ## Comparison vs benchmarks
    file.path(base_dir, "14_comparison"),
    
    ## Exploration / ad hoc
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
    
    ## Data pipeline
    universe     = file.path(base_dir, "01_universe"),
    prices       = file.path(base_dir, "02_prices"),
    fundamentals = file.path(base_dir, "03_fundamentals"),
    macro        = file.path(base_dir, "04_macro"),
    
    ## Labels
    labels       = file.path(base_dir, "05_labels"),
    csi_labels   = file.path(base_dir, "05_labels", "csi"),
    bucket_labels= file.path(base_dir, "05_labels", "bucket"),
    struct_labels= file.path(base_dir, "05_labels", "structural"),
    
    ## Features
    features     = file.path(base_dir, "06_features"),
    
    ## Models — cross-track comparison
    model_compare = file.path(base_dir, "09_models", "comparison"),
    
    ## Models — Track 1: CSI track overview + individual models
    csi_track    = file.path(base_dir, "09_models", "csi"),
    m1           = file.path(base_dir, "09_models", "csi", "m1"),
    m2           = file.path(base_dir, "09_models", "csi", "m2"),
    m3           = file.path(base_dir, "09_models", "csi", "m3"),
    m4           = file.path(base_dir, "09_models", "csi", "m4"),
    
    ## Models — Track 2: Bucket track overview + individual models
    bucket_track = file.path(base_dir, "09_models", "bucket"),
    b1           = file.path(base_dir, "09_models", "bucket", "b1"),
    b2           = file.path(base_dir, "09_models", "bucket", "b2"),
    b3           = file.path(base_dir, "09_models", "bucket", "b3"),
    b4           = file.path(base_dir, "09_models", "bucket", "b4"),
    
    ## Models — Track 3: Structural track overview + individual models
    struct_track = file.path(base_dir, "09_models", "structural"),
    s1           = file.path(base_dir, "09_models", "structural", "s1"),
    s2           = file.path(base_dir, "09_models", "structural", "s2"),
    s3           = file.path(base_dir, "09_models", "structural", "s3"),
    s4           = file.path(base_dir, "09_models", "structural", "s4"),
    
    ## Lookup list for programmatic access: FIGS$models[["m1"]]
    models       = list(
      m1 = file.path(base_dir, "09_models", "csi",        "m1"),
      m2 = file.path(base_dir, "09_models", "csi",        "m2"),
      m3 = file.path(base_dir, "09_models", "csi",        "m3"),
      m4 = file.path(base_dir, "09_models", "csi",        "m4"),
      b1 = file.path(base_dir, "09_models", "bucket",     "b1"),
      b2 = file.path(base_dir, "09_models", "bucket",     "b2"),
      b3 = file.path(base_dir, "09_models", "bucket",     "b3"),
      b4 = file.path(base_dir, "09_models", "bucket",     "b4"),
      s1 = file.path(base_dir, "09_models", "structural", "s1"),
      s2 = file.path(base_dir, "09_models", "structural", "s2"),
      s3 = file.path(base_dir, "09_models", "structural", "s3"),
      s4 = file.path(base_dir, "09_models", "structural", "s4")
    ),
    
    ## Index construction
    index_general  = file.path(base_dir, "11_index", "general"),
    index_csi      = file.path(base_dir, "11_index", "csi_track"),
    index_bucket   = file.path(base_dir, "11_index", "bucket_track"),
    index_struct   = file.path(base_dir, "11_index", "structural_track"),
    index_conc     = file.path(base_dir, "11_index", "concentrated"),
    
    ## Diagnostics and robustness
    evaluation   = file.path(base_dir, "12_evaluation"),
    robustness   = file.path(base_dir, "13_robustness"),
    rob_a        = file.path(base_dir, "13_robustness", "partA"),
    rob_b        = file.path(base_dir, "13_robustness", "partB"),
    rob_c        = file.path(base_dir, "13_robustness", "partC"),
    rob_d        = file.path(base_dir, "13_robustness", "partD"),
    rob_e        = file.path(base_dir, "13_robustness", "partE"),
    comparison   = file.path(base_dir, "14_comparison"),
    explore      = file.path(base_dir, "explore")
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
PATH_LABELS_BASE   <- file.path(DIR_LABELS, "labels_base.rds")
PATH_LABELS_GRID   <- file.path(DIR_LABELS, "labels_all_grid.rds")
PATH_LABELS_DIAG   <- file.path(DIR_LABELS, "csi_diagnostics.rds")
PATH_FIGURE_CSI    <- file.path(DIR_FIGURES, "05_labels", "csi",
                                "csi_events_per_year.png")

## 05B_Bucket_Labels.R
PATH_LABELS_BUCKET <- file.path(DIR_LABELS, "labels_bucket.rds")

## 05C_Structural_Labels.R
PATH_LABELS_STRUCTURAL <- file.path(DIR_LABELS, "labels_structural.rds")

## 06_Merge.R
PATH_PANEL_RAW <- file.path(DIR_PANEL, "panel_raw.rds")

## 06B_Feature_Eng.R
PATH_FEATURES_RAW  <- file.path(DIR_FEATURES, "features_raw.rds")
PATH_FEATURES_FUND <- file.path(DIR_FEATURES, "features_fund.rds")

## 08B_Autoencoder.py (outputs — parquets read by 09C)
PATH_FEATURES_LATENT_FUND <- file.path(DIR_FEATURES, "features_latent_fund.parquet")
PATH_FEATURES_LATENT_RAW  <- file.path(DIR_FEATURES, "features_latent_raw.parquet")
PATH_FEATURES_LATENT      <- PATH_FEATURES_LATENT_FUND   ## default alias

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
#
#   These constants are the single source of truth for 05B and 05C.
#   Both scripts should source config.R and use these constants directly
#   rather than hardcoding local copies.
#==============================================================================#

BUCKET_FWD_YEARS      <- 5L
BUCKET_MIN_MONTHS     <- 48L
BUCKET_LOSER_THRESH   <- -0.02   ## CAGR < -2%  → terminal loser (y=1)
BUCKET_PHOENIX_THRESH <-  0.00   ## CAGR >= 0%  → phoenix        (y=0)
BUCKET_LAST_YEAR      <- year(END_DATE) - BUCKET_FWD_YEARS   ## 2019

#==============================================================================#
# 13. Data Quality Thresholds
#==============================================================================#

MAX_CONSECUTIVE_NA <- 3L

#==============================================================================#
# 14. Feature Engineering Parameters
#==============================================================================#

WINDOW_SHORT         <- 36L    ## 3-year rolling window (months)
WINDOW_LONG          <- 60L    ## 5-year rolling window (months)
REPORTING_LAG_MONTHS <- 3L     ## Compustat availability lag (SEC 10-K deadline)
## Note: conservative — small-cap filers may use 4 months

ROLLING_STATS <- c(
  "mean", "min", "max", "median", "sd", "var",
  "mean_abs_diff", "median_abs_diff", "autocorr_lag1"
)

#==============================================================================#
# 15. Train / Test / OOS Split
#
#   Three-period design:
#     Train      : 1993–2015  model learning + HPO on holdout 2011–2014
#     Test       : 2016–2019  model selection + cross-model comparison
#     Live OOS   : 2020–2024  index construction ONLY (never model selection)
#
#   Boundary years (label shift artefact, CSI models only):
#     year = 2015  →  y_next = y(2016)  [test label after shift]
#     year = 2019  →  y_next = y(2020)  [OOS label after shift]
#     These rows are flagged "train_boundary"/"test_boundary" in eval_split
#     (08_Split.R). Excluded from AUC/AP metrics; predictions still generated
#     for index construction.
#
#   CV design:
#     CV_FOLDS = 4  →  4 year blocks; fold 1 always in training (no data
#                       precedes it in expanding window); folds 2–4 are
#                       the 3 usable validation folds.
#     09C_AutoGluon.py uses FOLD_BOUNDARIES (3 explicit time-anchored folds)
#     which matches the 3 usable folds from CV_FOLDS = 4.
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

## R-side CV (08_Split.R): 4 folds → fold 1 omitted → 3 usable validation folds
CV_FOLDS         <- 4L

## Python-side CV (09C): 3 explicit time-anchored expanding window folds
## Corresponds to the 3 usable folds from CV_FOLDS = 4 above
CV_FOLDS_PYTHON  <- 3L

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
#
#   EXCLUSION_RATE_CSI    : rank-based exclusion for M1–M4 (top 5% by p_csi)
#   EXCLUSION_RATE_BUCKET : rank-based exclusion for B1–B4 / S1–S4
#                           (higher rate needed — bucket positives are ~42%
#                           prevalence vs ~12% for CSI; 20% captures a
#                           meaningful tail without excessive dilution)
#   Both rates are applied annually, ranking all universe firms by p_csi
#   within each year and excluding the top X%.
#==============================================================================#

EXCLUSION_RATE_CSI    <- 0.05    ## M1–M4: top 5% flagged for exclusion
EXCLUSION_RATE_BUCKET <- 0.20    ## B1–B4 / S1–S4: top 20% flagged

PORT_CONC_SIZE_C1   <- 200L    ## C1 concentrated portfolio size
PORT_CONC_SIZE_C2   <- 100L    ## C2 concentrated portfolio size
PORT_M1_VETO_RATE   <- 0.10    ## C3 M1 veto threshold
PORT_RF_ANNUAL      <- 0.03    ## annualised risk-free rate for Sharpe

## Altman Z-score zombie threshold (from recovery classifier robustness Part B)
ZOMBIE_Z2_THRESH    <- -2.768814

#==============================================================================#
# 18. Plotting Parameters
#==============================================================================#

PLOT_WIDTH  <- 10
PLOT_HEIGHT <- 6
PLOT_DPI    <- 150

## Consistent strategy colours across all index plots
STRAT_COLOURS <- c(
  bench              = "#9E9E9E",
  ## CSI track
  s1_m1              = "#2196F3",
  s1_m3              = "#1565C0",
  ## Bucket track
  s1_b1              = "#4CAF50",
  s1_b3              = "#1B5E20",
  ## Structural track
  s1_s1              = "#9C27B0",
  s1_s3              = "#4A148C",
  ## Special strategies
  s4_zombie          = "#FF9800",
  c1_bucket          = "#E91E63",
  c1_structural      = "#880E4F",
  c2                 = "#CE93D8",
  c3                 = "#F44336",
  ## Naive benchmarks
  low_vol            = "#00BCD4",
  quality            = "#FF5722"
)

STRAT_LABELS <- c(
  bench              = "Benchmark (EW 3000)",
  s1_m1              = "M1 Excl. 5% (fund)",
  s1_m3              = "M3 Excl. 5% (raw)",
  s1_b1              = "B1 Excl. 20% (fund)",
  s1_b3              = "B3 Excl. 20% (raw)",
  s1_s1              = "S1 Excl. 20% (fund)",
  s1_s3              = "S3 Excl. 20% (raw)",
  s4_zombie          = "M1 + Zombie Filter",
  c1_bucket          = "C1: B1-Bucket Long 200",
  c1_structural      = "C1: B1-Structural Long 200",
  c2                 = "C2: B1 Long 100",
  c3                 = "C3: Structural + M1 Veto",
  low_vol            = "Low-Vol 200",
  quality            = "Quality 200 (Altman Z)"
)

#==============================================================================#
# 19. Model key → human label mapping
#
#   Used by 10_Evaluate.R and 11_Results.R when building comparison tables
#   and multi-model plots. Keys match MODEL values in 09C_AutoGluon.py.
#==============================================================================#

MODEL_LABELS <- c(
  fund                      = "M1 — Fundamentals",
  latent_fund               = "M2 — VAE (fund)",
  raw                       = "M3 — Full features",
  latent_raw                = "M4 — VAE (raw)",
  bucket                    = "B1 — Bucket (fund)",
  bucket_latent_fund        = "B2 — Bucket VAE (fund)",
  bucket_raw                = "B3 — Bucket (raw)",
  bucket_latent_raw         = "B4 — Bucket VAE (raw)",
  structural                = "S1 — Structural (fund)",
  structural_latent_fund    = "S2 — Structural VAE (fund)",
  structural_raw            = "S3 — Structural (raw)",
  structural_latent_raw     = "S4 — Structural VAE (raw)"
)

MODEL_TRACK <- c(
  fund                      = "CSI",
  latent_fund               = "CSI",
  raw                       = "CSI",
  latent_raw                = "CSI",
  bucket                    = "Bucket",
  bucket_latent_fund        = "Bucket",
  bucket_raw                = "Bucket",
  bucket_latent_raw         = "Bucket",
  structural                = "Structural",
  structural_latent_fund    = "Structural",
  structural_raw            = "Structural",
  structural_latent_raw     = "Structural"
)

#==============================================================================#
# 20. Confirm Load
#==============================================================================#

cat("[config.R] Loaded.\n")
cat(sprintf("  Root           : %s\n", DIR_ROOT))
cat(sprintf("  Period         : %s to %s\n",
            format(START_DATE), format(END_DATE)))
cat(sprintf("  Split          : Train ≤%d | Test %d–%d | OOS ≥%d\n",
            TRAIN_END_YR, TEST_START_YR, TEST_END_YR, OOS_START_YR))
cat(sprintf("  CV folds (R/Py): %d / %d  (fold 1 omitted in expanding window)\n",
            CV_FOLDS, CV_FOLDS_PYTHON))
cat(sprintf("  CSI base       : C=%.2f | M=%.2f | T=%d months\n",
            CSI_BASE$C, CSI_BASE$M, CSI_BASE$T))
cat(sprintf("  Bucket         : %d-yr fwd | loser < %.0f%% | phoenix ≥ %.0f%% | last year=%d\n",
            BUCKET_FWD_YEARS, BUCKET_LOSER_THRESH*100,
            BUCKET_PHOENIX_THRESH*100, BUCKET_LAST_YEAR))
cat(sprintf("  Exclusion rate : CSI=%.0f%% | Bucket/Structural=%.0f%%\n",
            EXCLUSION_RATE_CSI*100, EXCLUSION_RATE_BUCKET*100))
cat(sprintf("  Universe       : Top %d | min $%dM mktcap\n",
            UNIVERSE_SIZE, UNIVERSE_MIN_MKTCAP))
cat(sprintf("  Seed           : %d\n", SEED))
cat(sprintf("  Models         : %d  (%s)\n",
            length(MODEL_LABELS),
            paste(names(MODEL_LABELS), collapse=", ")))
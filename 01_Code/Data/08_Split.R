#==============================================================================#
#==== 08_Split.R ==============================================================#
#==== Train / Test / OOS Split Construction ===================================#
#==============================================================================#
#
# PURPOSE:
#   Construct two parallel split strategies on features_raw.rds:
#
#   STRATEGY 1 — Out-of-Time (OOT): Temporal split (PRIMARY)
#     Used for: model selection, HPO, index backtest in 11_Results.R
#     Logic: hard date cutoffs from config.R
#       Train : year <= TRAIN_END  (1993–2015)
#       Test  : TEST_START <= year <= TEST_END  (2016–2019)
#       OOS   : year >= OOS_START  (2020–2024)
#     Rationale: simulates real deployment — model trained on past,
#     evaluated on future. Avoids look-ahead bias at the regime level.
#
#   STRATEGY 2 — Out-of-Sample (OOS): Firm-level stratified split (SECONDARY)
#     Used for: cross-sectional robustness analysis in 12_Robustness.R
#     Logic: 70/30 stratified split at the FIRM level
#       Train : 70% of firms (all their years)
#       Test  : 30% of firms (all their years)
#     Stratification: by ever-CSI flag (y_firm = max(y, na.rm=TRUE) ∈ {0,1})
#       Preserves class balance across the split.
#     Rationale: tests cross-sectional generalisation — can the model
#     identify CSI firms it has never seen during training?
#
# INPUTS:
#   - config.R
#   - PATH_FEATURES_RAW  : (permno, year, y, censored, ~463 features)
#
# OUTPUTS:
#   - PATH_SPLITS        : list with $oot, $oos_firm, $diagnostics
#   - split_labels_oot.parquet   : for Python — includes split + eval_split
#   - split_labels_oos.parquet   : for Python — firm-level split
#
# OUTPUT SCHEMA (splits.rds):
#
#   $oot
#     $split_col      : character vector [nrow], values "train"/"test"/"oos"/NA
#     $eval_split_col : character vector [nrow], boundary-safe version of split
#                       Differs from split_col only at period boundaries:
#                         year == max(train years) → eval_split = "train_boundary"
#                         year == max(test  years) → eval_split = "test_boundary"
#                       09C uses eval_split to filter rows for AUC/AP metrics;
#                       split_col governs which rows receive predictions for
#                       index construction. See DESIGN DECISION [6].
#     $train_idx      : integer row indices where split_col == "train"
#     $test_idx       : integer row indices where split_col == "test"
#     $oos_idx        : integer row indices where split_col == "oos"
#     $train_eval_idx : row indices safe for metric evaluation (excl. boundary)
#     $test_eval_idx  : row indices safe for metric evaluation (excl. boundary)
#
#   $oos_firm
#     $split_col    : character vector [nrow], values "train"/"test"
#     $train_idx    : integer row indices for train firms
#     $test_idx     : integer row indices for test firms
#     $train_permno : permno vector in training set
#     $test_permno  : permno vector in test set
#
#   $diagnostics
#     $oot          : data.frame — rows/CSI events/prevalence per split
#     $oos_firm     : data.frame — rows/firms/CSI firms/prevalence per split
#
# DESIGN DECISIONS:
#
#   [1] FIRM-LEVEL OOS SPLIT — never split a firm's history across sets.
#       If permno 12345 goes to test, ALL its years go to test.
#       This prevents the model learning firm-specific patterns in train
#       and exploiting them in test — overly optimistic evaluation.
#
#   [2] STRATIFICATION ON EVER-CSI FLAG.
#       Firm-level label: y_firm = max(y, na.rm=TRUE), NA-only firms → 0.
#       Stratification ensures both train and test sets contain ~10% CSI firms
#       despite the imbalanced label distribution.
#
#   [3] OOT SPLIT USES CONFIG DATES — no stratification needed.
#       Temporal splits are deterministic given the date cutoffs.
#       Class balance is checked post-hoc in diagnostics.
#
#   [4] y=NA ROWS INCLUDED IN SPLITS.
#       panel_raw.rds includes zombie/censored rows (y=NA).
#       These rows carry firm-level information useful to the autoencoder.
#       09_Train.R filters to !is.na(y) before model training.
#       Split indices here cover ALL rows including y=NA.
#
#   [5] SEED FROM config.R — full reproducibility.
#
#   [6] SPLIT vs EVAL_SPLIT — prediction coverage vs metric validity.
#       After the label shift y(t) → y_next = y(t+1) applied in 09C:
#         - The last training year (2015) carries y_next = y(2016), a test label.
#         - The last test year   (2019) carries y_next = y(2020), an OOS label.
#       These boundary rows should be EXCLUDED from AUC/AP metric computation,
#       but their FEATURES at year t are still valid inputs for generating
#       predictions used in index construction.
#       Resolution: export TWO split signals per row:
#         split     : "train"/"test"/"oos" — governs prediction generation
#         eval_split: "train"/"test"/"oos"/"train_boundary"/"test_boundary"
#                     09C filters to eval_split ∈ {"train","test","oos"}
#                     for metric computation, but generates predictions for
#                     all rows where split ∈ {"train","test","oos"}.
#
#   [7] OOT CV — EXPANDING WINDOW, starting from fold 2.
#       Fold 1 would have zero training rows in an expanding window scheme
#       (nothing precedes it). The CV loop starts at fold 2.
#       Fold 1 years are always in the training portion of later folds.
#       Per-fold AUC/AP should be averaged across folds (not pooled) in 09C
#       to give equal weight to each time regime.
#
#==============================================================================#

source("config.R")

cat("\n[08_Split.R] START:", format(Sys.time()), "\n")

#==============================================================================#
# 0. Load features_raw
#==============================================================================#

cat("[08_Split.R] Loading features_raw...\n")

features <- as.data.table(readRDS(PATH_FEATURES_RAW))
setorder(features, permno, year)

n_rows   <- nrow(features)
n_permno <- n_distinct(features$permno)

TRAIN_END_YEAR <- year(TRAIN_END)   # 2015
TEST_START_YEAR <- year(TEST_START) # 2016
TEST_END_YEAR   <- year(TEST_END)   # 2019
OOS_START_YEAR  <- year(OOS_START)  # 2020

cat(sprintf("  Rows   : %d\n", n_rows))
cat(sprintf("  Permno : %d\n", n_permno))
cat(sprintf("  Years  : %d – %d\n", min(features$year), max(features$year)))
cat(sprintf("  y=1    : %d | y=0 : %d | y=NA : %d\n",
            sum(features$y == 1L, na.rm = TRUE),
            sum(features$y == 0L, na.rm = TRUE),
            sum(is.na(features$y))))
cat(sprintf("  Period boundaries: train≤%d | test %d–%d | oos≥%d\n",
            TRAIN_END_YEAR, TEST_START_YEAR, TEST_END_YEAR, OOS_START_YEAR))

#==============================================================================#
# 1. STRATEGY 1 — Out-of-Time (OOT) Temporal Split
#
#   Hard date cutoffs from config.R.
#
#   split_col     : standard split assignment for prediction generation
#   eval_split_col: boundary-safe variant for metric computation (design [6])
#
#   Boundary rows:
#     year == TRAIN_END_YEAR (2015): features at t=2015, y_next = y(2016) ∈ test
#     year == TEST_END_YEAR  (2019): features at t=2019, y_next = y(2020) ∈ OOS
#   These rows are flagged "train_boundary" / "test_boundary" in eval_split_col.
#   09C must filter to eval_split ∉ {"train_boundary","test_boundary"} for metrics,
#   but still generate predictions for ALL split-assigned rows for index use.
#==============================================================================#

cat("\n[08_Split.R] Strategy 1: Out-of-Time temporal split...\n")

## Primary split — governs prediction generation
oot_split_col <- fcase(
  features$year <= TRAIN_END_YEAR,                                             "train",
  features$year >= TEST_START_YEAR & features$year <= TEST_END_YEAR,           "test",
  features$year >= OOS_START_YEAR,                                             "oos",
  default = NA_character_
)

## Eval split — boundary-safe variant for AUC/AP metric computation
## Boundary years have their label contaminated by the next period after shift
oot_eval_split_col <- fcase(
  features$year == TRAIN_END_YEAR,                                             "train_boundary",
  features$year == TEST_END_YEAR,                                              "test_boundary",
  features$year <  TRAIN_END_YEAR,                                             "train",
  features$year >= TEST_START_YEAR & features$year < TEST_END_YEAR,            "test",
  features$year >= OOS_START_YEAR,                                             "oos",
  default = NA_character_
)

## Row indices per split (ALL rows for prediction generation)
oot_train_idx <- which(oot_split_col == "train")
oot_test_idx  <- which(oot_split_col == "test")
oot_oos_idx   <- which(oot_split_col == "oos")

## Eval-safe row indices (boundary years excluded — for metric computation only)
oot_train_eval_idx <- which(oot_eval_split_col == "train")
oot_test_eval_idx  <- which(oot_eval_split_col == "test")

## Boundary year counts for reporting
n_train_boundary <- sum(oot_eval_split_col == "train_boundary", na.rm = TRUE)
n_test_boundary  <- sum(oot_eval_split_col == "test_boundary",  na.rm = TRUE)

cat(sprintf("  Train rows     : %d  (%d permno, years %d–%d)\n",
            length(oot_train_idx),
            n_distinct(features$permno[oot_train_idx]),
            min(features$year[oot_train_idx]),
            max(features$year[oot_train_idx])))
cat(sprintf("  Test rows      : %d  (%d permno, years %d–%d)\n",
            length(oot_test_idx),
            n_distinct(features$permno[oot_test_idx]),
            min(features$year[oot_test_idx]),
            max(features$year[oot_test_idx])))
cat(sprintf("  OOS rows       : %d  (%d permno, years %d–%d)\n",
            length(oot_oos_idx),
            n_distinct(features$permno[oot_oos_idx]),
            min(features$year[oot_oos_idx]),
            max(features$year[oot_oos_idx])))
cat(sprintf("\n  Boundary rows (excluded from eval metrics only):\n"))
cat(sprintf("    train_boundary (year=%d): %d rows  → y_next = y(%d) [test label]\n",
            TRAIN_END_YEAR, n_train_boundary, TRAIN_END_YEAR + 1L))
cat(sprintf("    test_boundary  (year=%d): %d rows  → y_next = y(%d) [OOS label]\n",
            TEST_END_YEAR,  n_test_boundary,  TEST_END_YEAR  + 1L))
cat(sprintf("  Train eval rows: %d (excl. %d boundary)\n",
            length(oot_train_eval_idx), n_train_boundary))
cat(sprintf("  Test  eval rows: %d (excl. %d boundary)\n",
            length(oot_test_eval_idx),  n_test_boundary))

## Verify no gap between splits
n_unassigned <- sum(is.na(oot_split_col))
if (n_unassigned > 0)
  warning(sprintf("[08_Split.R] WARNING: %d rows unassigned to any split.",
                  n_unassigned))

## OOT diagnostics
fn_oot_diag <- function(idx, label) {
  y_vec <- features$y[idx]
  n_csi <- sum(y_vec == 1L, na.rm = TRUE)
  n_obs <- sum(!is.na(y_vec))
  data.frame(
    split      = label,
    n_rows     = length(idx),
    n_labelled = n_obs,
    n_csi      = n_csi,
    n_clean    = sum(y_vec == 0L, na.rm = TRUE),
    n_na       = sum(is.na(y_vec)),
    prevalence = round(100 * n_csi / max(n_obs, 1L), 2)
  )
}

oot_diag <- rbind(
  fn_oot_diag(oot_train_idx,      "train"),
  fn_oot_diag(oot_train_eval_idx, "train_eval"),
  fn_oot_diag(oot_test_idx,       "test"),
  fn_oot_diag(oot_test_eval_idx,  "test_eval"),
  fn_oot_diag(oot_oos_idx,        "oos")
)

cat("\n  OOT class balance (full split + eval-safe subset):\n")
print(oot_diag, row.names = FALSE)

#==============================================================================#
# 2. STRATEGY 2 — Out-of-Sample (OOS) Firm-Level Stratified Split
#
#   70/30 split at firm level, stratified by ever-CSI flag.
#   All years for a given permno go to the same set.
#==============================================================================#

cat("\n[08_Split.R] Strategy 2: Firm-level stratified OOS split...\n")

set.seed(SEED)

## 2A. Compute firm-level CSI flag for stratification
firm_profile <- features[, .(
  y_firm = as.integer(any(y == 1L, na.rm = TRUE) == 1L)
), by = permno]

firm_profile[is.nan(y_firm), y_firm := 0L]

cat(sprintf("  Firms total    : %d\n", nrow(firm_profile)))
cat(sprintf("  Ever-CSI firms : %d (%.1f%%)\n",
            sum(firm_profile$y_firm == 1L),
            100 * mean(firm_profile$y_firm == 1L)))
cat(sprintf("  Never-CSI firms: %d (%.1f%%)\n",
            sum(firm_profile$y_firm == 0L),
            100 * mean(firm_profile$y_firm == 0L)))

## 2B. Stratified 70/30 firm-level split
TRAIN_SIZE <- 0.70

fn_stratified_firm_split <- function(firm_dt, train_size, seed) {
  set.seed(seed)
  
  train_permno_list <- list()
  test_permno_list  <- list()
  
  for (stratum in c(0L, 1L)) {
    stratum_permnos <- firm_dt[y_firm == stratum, permno]
    n_stratum       <- length(stratum_permnos)
    shuffled        <- stratum_permnos[sample(n_stratum)]
    n_train         <- round(n_stratum * train_size)
    train_permno_list[[as.character(stratum)]] <- shuffled[seq_len(n_train)]
    test_permno_list[[as.character(stratum)]]  <- shuffled[(n_train + 1L):n_stratum]
  }
  
  list(
    train_permno = unlist(train_permno_list),
    test_permno  = unlist(test_permno_list)
  )
}

firm_split   <- fn_stratified_firm_split(firm_profile, TRAIN_SIZE, SEED)
train_permno <- firm_split$train_permno
test_permno  <- firm_split$test_permno

## 2C. Map firm assignments to row indices
oos_train_idx <- which(features$permno %in% train_permno)
oos_test_idx  <- which(features$permno %in% test_permno)

oos_split_col <- character(n_rows)
oos_split_col[oos_train_idx] <- "train"
oos_split_col[oos_test_idx]  <- "test"

n_unassigned_oos <- sum(oos_split_col == "")
if (n_unassigned_oos > 0)
  stop(sprintf("[08_Split.R] ASSERTION FAILED: %d rows unassigned in OOS split.",
               n_unassigned_oos))

cat(sprintf("  Train: %d rows | %d firms\n",
            length(oos_train_idx), length(train_permno)))
cat(sprintf("  Test : %d rows | %d firms\n",
            length(oos_test_idx),  length(test_permno)))

## 2D. OOS diagnostics
fn_oos_diag <- function(idx, label) {
  y_vec     <- features$y[idx]
  perm_vec  <- features$permno[idx]
  n_csi     <- sum(y_vec == 1L, na.rm = TRUE)
  n_obs     <- sum(!is.na(y_vec))
  n_firms   <- n_distinct(perm_vec)
  csi_firms <- n_distinct(perm_vec[!is.na(y_vec) & y_vec == 1L])
  data.frame(
    split         = label,
    n_rows        = length(idx),
    n_firms       = n_firms,
    n_labelled    = n_obs,
    n_csi_obs     = n_csi,
    n_csi_firms   = csi_firms,
    pct_csi_firms = round(100 * csi_firms / n_firms, 2),
    prevalence    = round(100 * n_csi / max(n_obs, 1L), 2)
  )
}

oos_diag <- rbind(
  fn_oos_diag(oos_train_idx, "train"),
  fn_oos_diag(oos_test_idx,  "test")
)

cat("\n  OOS firm-split class balance:\n")
print(oos_diag, row.names = FALSE)

prev_train <- oos_diag$prevalence[oos_diag$split == "train"]
prev_test  <- oos_diag$prevalence[oos_diag$split == "test"]
if (abs(prev_train - prev_test) > 3.0)
  warning(sprintf(
    "[08_Split.R] WARNING: OOS prevalence gap %.1f%% (train) vs %.1f%% (test)",
    prev_train, prev_test))

#==============================================================================#
# 3. OOT Expanding Window CV Folds
#
#   DESIGN (note [7]):
#   Expanding window: each fold trains on all years up to train_end_k,
#   validates on the next block of years.
#
#   FOLD 1 IS OMITTED. In an expanding window scheme, fold 1 would have zero
#   training rows (no years precede the first block). The CV loop starts at
#   fold 2 — fold 1 years serve as the initial training data for fold 2.
#
#   CV_K folds total in config → CV_K - 1 usable folds (2 through CV_K).
#
#   PER-FOLD METRICS (not pooled):
#   09C should compute AUC/AP for each fold independently and report the
#   average. Pooling predictions across folds and computing one metric gives
#   undue weight to larger folds and conflates time-regime effects.
#   The fold structure exported here facilitates per-fold evaluation.
#
#   BOUNDARY NOTE:
#   The last year of each fold's training window is also a boundary year
#   (its y_next label comes from the validation period). The eval_split
#   concept applies here too — 09C should exclude the last training year
#   of each fold from validation metric computation. The fold structure
#   below exports train_end_year per fold so 09C can apply this consistently.
#==============================================================================#

cat("\n[08_Split.R] Constructing OOT expanding window CV folds...\n")

CV_K <- CV_FOLDS   # From config.R

train_years    <- sort(unique(features$year[oot_train_idx]))
n_train_yrs    <- length(train_years)

## Assign each training year to a fold block (CV_K equal-sized blocks)
year_fold_assignment <- data.table(
  year = train_years,
  fold = as.integer(cut(seq_len(n_train_yrs),
                        breaks = CV_K,
                        labels = seq_len(CV_K)))
)

## Build folds starting from fold 2 — fold 1 has no valid training set
## For each fold k (k >= 2):
##   validation = rows in fold k
##   training   = rows in folds 1..(k-1)  [expanding window]
##
## Boundary exclusion for metrics:
##   train_end_year = max year in training window
##   Rows where year == train_end_year should be excluded from validation
##   metrics in 09C (their y_next label comes from the validation period).

oot_cv_folds <- lapply(seq(2L, CV_K), function(k) {
  
  val_years       <- year_fold_assignment[fold == k,  year]
  train_years_k   <- year_fold_assignment[fold < k,   year]
  train_end_year_k <- max(train_years_k)
  
  val_rows   <- which(features$year %in% val_years   & oot_split_col == "train")
  train_rows <- which(features$year %in% train_years_k & oot_split_col == "train")
  
  ## Eval-safe training rows: exclude last training year (boundary)
  train_eval_rows <- which(
    features$year %in% train_years_k &
      features$year < train_end_year_k &
      oot_split_col == "train"
  )
  
  list(
    fold             = k,
    train            = train_rows,
    train_eval       = train_eval_rows,   ## for metric computation only
    validation       = val_rows,
    val_years        = val_years,
    train_years      = train_years_k,
    train_end_year   = train_end_year_k,
    n_val_years      = length(val_years)
  )
})

cat(sprintf("  OOT CV: %d usable folds (fold 1 omitted — no training data precedes it)\n",
            length(oot_cv_folds)))
cat(sprintf("  Fold 1 years (%d–%d) always serve as initial training data.\n",
            min(year_fold_assignment[fold == 1L, year]),
            max(year_fold_assignment[fold == 1L, year])))

for (fd in oot_cv_folds) {
  cat(sprintf(
    "  Fold %d: train [%d–%d] n=%d (eval-safe: %d) | val [%d–%d] n=%d\n",
    fd$fold,
    min(fd$train_years), fd$train_end_year,
    length(fd$train),
    length(fd$train_eval),
    min(fd$val_years), max(fd$val_years),
    length(fd$validation)
  ))
}

#==============================================================================#
# 4. Firm-Level Stratified k-Fold CV (OOS strategy)
#==============================================================================#

cat("\n[08_Split.R] Constructing firm-level stratified k-fold CV...\n")

fn_firm_kfold <- function(firm_dt, k, seed) {
  ## firm_dt: already scoped to OOS train firms only
  set.seed(seed)
  folds <- vector("list", k)
  
  for (stratum in c(0L, 1L)) {
    stratum_permnos <- firm_dt[y_firm == stratum, permno]
    n_stratum       <- length(stratum_permnos)
    shuffled        <- stratum_permnos[sample(n_stratum)]
    fold_assignments <- ((seq_len(n_stratum) - 1L) %% k) + 1L
    for (f in seq_len(k)) {
      folds[[f]] <- c(folds[[f]], shuffled[fold_assignments == f])
    }
  }
  
  all_train_permnos <- firm_dt[, permno]
  
  lapply(seq_len(k), function(f) {
    val_permno_f   <- folds[[f]]
    train_permno_f <- setdiff(all_train_permnos, val_permno_f)
    
    val_rows   <- which(features$permno %in% val_permno_f   & oos_split_col == "train")
    train_rows <- which(features$permno %in% train_permno_f & oos_split_col == "train")
    
    list(train = train_rows, validation = val_rows, val_permno = val_permno_f)
  })
}

## Scope firm_dt to OOS train firms only (fixes previous over-broad scope)
oos_cv_folds <- fn_firm_kfold(
  firm_dt = firm_profile[permno %in% train_permno],
  k       = CV_K,
  seed    = SEED
)

cat(sprintf("  OOS firm CV: %d folds\n", CV_K))
for (k in seq_len(CV_K)) {
  cat(sprintf("    Fold %d: train %d rows | val %d rows (%d firms)\n",
              k,
              length(oos_cv_folds[[k]]$train),
              length(oos_cv_folds[[k]]$validation),
              length(oos_cv_folds[[k]]$val_permno)))
}

#==============================================================================#
# 5. Assertions
#==============================================================================#

cat("\n[08_Split.R] Running assertions...\n")

## A) OOT: every row assigned to split_col or NA
stopifnot(
  "OOT: row coverage incomplete" =
    length(oot_train_idx) + length(oot_test_idx) + length(oot_oos_idx) +
    sum(is.na(oot_split_col)) == n_rows
)

## B) OOT: no overlap between train/test/oos
stopifnot(
  "OOT: train/test overlap" = length(intersect(oot_train_idx, oot_test_idx)) == 0,
  "OOT: train/oos overlap"  = length(intersect(oot_train_idx, oot_oos_idx))  == 0,
  "OOT: test/oos overlap"   = length(intersect(oot_test_idx,  oot_oos_idx))  == 0
)

## C) eval_split boundary rows are subset of their respective splits
stopifnot(
  "train_eval must be subset of train" =
    all(oot_train_eval_idx %in% oot_train_idx),
  "test_eval must be subset of test" =
    all(oot_test_eval_idx %in% oot_test_idx)
)

## D) Boundary rows correctly excluded from eval sets
stopifnot(
  "Boundary year still in train_eval" =
    !any(features$year[oot_train_eval_idx] == TRAIN_END_YEAR),
  "Boundary year still in test_eval" =
    !any(features$year[oot_test_eval_idx] == TEST_END_YEAR)
)

## E) OOS firm split: every row assigned
stopifnot(
  "OOS: row coverage incomplete" =
    length(oos_train_idx) + length(oos_test_idx) == n_rows
)

## F) OOS firm split: no permno overlap between train and test
stopifnot(
  "OOS: firm overlap between train and test" =
    length(intersect(train_permno, test_permno)) == 0
)

## G) OOS firm split: all permno accounted for
stopifnot(
  "OOS: not all permno assigned" =
    length(union(train_permno, test_permno)) == n_permno
)

## H) CV folds: each validation fold is non-empty and within train set
for (fd in oot_cv_folds) {
  stopifnot(
    sprintf("OOT CV fold %d: empty validation set", fd$fold) =
      length(fd$validation) > 0,
    sprintf("OOT CV fold %d: empty training set", fd$fold) =
      length(fd$train) > 0,
    sprintf("OOT CV fold %d: val rows outside train split", fd$fold) =
      all(fd$validation %in% oot_train_idx)
  )
}

## I) CV folds cover train set: each training row appears in exactly one
##    validation fold across folds 2..K. Fold 1 rows never appear as
##    validation (they are always training data in the expanding window).
oot_cv_all_val      <- unlist(lapply(oot_cv_folds, `[[`, "validation"))
fold1_years         <- year_fold_assignment[fold == 1L, year]
fold1_train_idx     <- which(features$year %in% fold1_years & oot_split_col == "train")
expected_val_rows   <- setdiff(oot_train_idx, fold1_train_idx)

stopifnot(
  "OOT CV: validation rows don't cover folds 2..K of train set" =
    length(unique(oot_cv_all_val)) == length(expected_val_rows)
)

## J) Plausible split sizes
stopifnot(
  "OOT: train set too small (< 40% of rows)" =
    length(oot_train_idx) / n_rows >= 0.40,
  "OOS: train set not ~70% of rows" =
    abs(length(oos_train_idx) / n_rows - TRAIN_SIZE) < 0.05
)

cat("[08_Split.R] All assertions passed.\n")

#==============================================================================#
# 6. Save splits.rds
#==============================================================================#

splits <- list(
  
  ## Strategy 1: Out-of-Time
  oot = list(
    split_col        = oot_split_col,
    eval_split_col   = oot_eval_split_col,
    train_idx        = oot_train_idx,
    test_idx         = oot_test_idx,
    oos_idx          = oot_oos_idx,
    train_eval_idx   = oot_train_eval_idx,
    test_eval_idx    = oot_test_eval_idx,
    train_end_year   = TRAIN_END_YEAR,
    test_end_year    = TEST_END_YEAR,
    cv_folds         = oot_cv_folds
  ),
  
  ## Strategy 2: Out-of-Sample firm-level
  oos_firm = list(
    split_col    = oos_split_col,
    train_idx    = oos_train_idx,
    test_idx     = oos_test_idx,
    train_permno = train_permno,
    test_permno  = test_permno,
    cv_folds     = oos_cv_folds
  ),
  
  ## Metadata
  meta = list(
    n_rows           = n_rows,
    n_permno         = n_permno,
    train_size       = TRAIN_SIZE,
    cv_k             = CV_K,
    cv_k_usable      = CV_K - 1L,
    seed             = SEED,
    train_end        = TRAIN_END,
    test_start       = TEST_START,
    test_end         = TEST_END,
    oos_start        = OOS_START,
    train_end_year   = TRAIN_END_YEAR,
    test_end_year    = TEST_END_YEAR,
    boundary_note    = paste0(
      "Rows with year==", TRAIN_END_YEAR, " (train_boundary) and year==",
      TEST_END_YEAR, " (test_boundary) are included in split_col for ",
      "prediction generation but excluded from eval_split_col for ",
      "AUC/AP metric computation. See design note [6]."
    )
  ),
  
  ## Diagnostics
  diagnostics = list(
    oot      = oot_diag,
    oos_firm = oos_diag
  )
)

saveRDS(splits, PATH_SPLITS)
cat(sprintf("[08_Split.R] splits.rds saved.\n"))

#==============================================================================#
# 7. Export parquets for Python (09C_AutoGluon.py)
#
#   Two columns exported per row:
#     split      : "train"/"test"/"oos" — for prediction generation
#     eval_split : "train"/"test"/"oos"/"train_boundary"/"test_boundary"
#                  09C uses eval_split to filter rows for AUC/AP metrics
#
#   09C usage pattern:
#     df_all_preds = all rows where split != NA       → prediction output
#     df_metrics   = all rows where eval_split ∉ {*_boundary} → AUC/AP
#==============================================================================#

library(arrow)

arrow::write_parquet(
  data.frame(
    permno     = features$permno,
    year       = features$year,
    split      = splits$oot$split_col,
    eval_split = splits$oot$eval_split_col
  ),
  file.path(DIR_FEATURES, "split_labels_oot.parquet")
)

arrow::write_parquet(
  data.frame(
    permno = features$permno,
    year   = features$year,
    split  = splits$oos_firm$split_col
  ),
  file.path(DIR_FEATURES, "split_labels_oos.parquet")
)

cat("[08_Split.R] Parquets saved for Python.\n")
cat(sprintf("  split_labels_oot.parquet : split + eval_split columns\n"))
cat(sprintf("  split_labels_oos.parquet : split column only\n"))

#==============================================================================#
# 8. Summary diagnostics + plot
#==============================================================================#

cat("\n[08_Split.R] ══════════════════════════════════════\n")
cat("  Strategy 1 — Out-of-Time (OOT):\n")
print(oot_diag, row.names = FALSE)
cat("\n  Strategy 2 — Out-of-Sample firm-level (OOS):\n")
print(oos_diag, row.names = FALSE)

## Prevalence plot — full splits (not eval subsets, to keep chart clean)
diag_plot <- rbind(
  cbind(strategy = "OOT",      oot_diag[oot_diag$split %in% c("train","test","oos"),
                                        c("split","prevalence")]),
  cbind(strategy = "OOS_firm", oos_diag[, c("split","prevalence")])
)

p_prev <- ggplot(diag_plot, aes(x = split, y = prevalence, fill = strategy)) +
  geom_col(position = "dodge", width = 0.6) +
  geom_hline(yintercept = 100 * mean(features$y == 1L, na.rm = TRUE),
             linetype = "dashed", colour = "grey40") +
  labs(
    title    = "CSI Prevalence by Split Strategy",
    subtitle = "Dashed line = overall prevalence | Boundary rows shown in full split only",
    x = "Split", y = "Prevalence (%)", fill = "Strategy"
  ) +
  theme_minimal(base_size = 12)

ggsave(
  file.path(DIR_FIGURES, "split_prevalence.png"),
  plot = p_prev, width = PLOT_WIDTH, height = PLOT_HEIGHT, dpi = PLOT_DPI
)

cat(sprintf("\n  Prevalence (full splits):\n"))
cat(sprintf("    OOT  train : %.2f%% | test : %.2f%% | oos : %.2f%%\n",
            oot_diag$prevalence[oot_diag$split == "train"],
            oot_diag$prevalence[oot_diag$split == "test"],
            oot_diag$prevalence[oot_diag$split == "oos"]))
cat(sprintf("    OOS  train : %.2f%% | test : %.2f%%\n",
            oos_diag$prevalence[1], oos_diag$prevalence[2]))

cat(sprintf("\n  CV fold structure (%d usable folds, expanding window):\n",
            length(oot_cv_folds)))
cat(sprintf("    Fold 1 years (%d–%d) always in training — never validated.\n",
            min(fold1_years), max(fold1_years)))
for (fd in oot_cv_folds) {
  cat(sprintf("    Fold %d: val [%d–%d] n=%d | train boundary year=%d\n",
              fd$fold, min(fd$val_years), max(fd$val_years),
              length(fd$validation), fd$train_end_year))
}

cat("\n[08_Split.R] DONE:", format(Sys.time()), "\n")
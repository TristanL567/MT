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
#
# OUTPUT SCHEMA (splits.rds):
#
#   $oot
#     $split_col    : character vector [nrow], values "train"/"test"/"oos"/NA
#     $train_idx    : integer row indices where split_col == "train"
#     $test_idx     : integer row indices where split_col == "test"
#     $oos_idx      : integer row indices where split_col == "oos"
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
#     $label_rows_only : label distribution excluding y=NA rows
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

cat(sprintf("  Rows   : %d\n", n_rows))
cat(sprintf("  Permno : %d\n", n_permno))
cat(sprintf("  Years  : %d – %d\n", min(features$year), max(features$year)))
cat(sprintf("  y=1    : %d | y=0 : %d | y=NA : %d\n",
            sum(features$y == 1L, na.rm = TRUE),
            sum(features$y == 0L, na.rm = TRUE),
            sum(is.na(features$y))))

#==============================================================================#
# 1. STRATEGY 1 — Out-of-Time (OOT) Temporal Split
#
#   Hard date cutoffs from config.R.
#   TRAIN_END, TEST_START, TEST_END, OOS_START defined in config.R.
#==============================================================================#

cat("\n[08_Split.R] Strategy 1: Out-of-Time temporal split...\n")

## Assign split label per row based on year
oot_split_col <- fcase(
  features$year <= year(TRAIN_END),                                    "train",
  features$year >= year(TEST_START) & features$year <= year(TEST_END), "test",
  features$year >= year(OOS_START),                                    "oos",
  default = NA_character_   # Should not occur if date ranges are contiguous
)

## Row indices per split
oot_train_idx <- which(oot_split_col == "train")
oot_test_idx  <- which(oot_split_col == "test")
oot_oos_idx   <- which(oot_split_col == "oos")

cat(sprintf("  Train rows : %d  (%d permno, years %d–%d)\n",
            length(oot_train_idx),
            n_distinct(features$permno[oot_train_idx]),
            min(features$year[oot_train_idx]),
            max(features$year[oot_train_idx])))
cat(sprintf("  Test rows  : %d  (%d permno, years %d–%d)\n",
            length(oot_test_idx),
            n_distinct(features$permno[oot_test_idx]),
            min(features$year[oot_test_idx]),
            max(features$year[oot_test_idx])))
cat(sprintf("  OOS rows   : %d  (%d permno, years %d–%d)\n",
            length(oot_oos_idx),
            n_distinct(features$permno[oot_oos_idx]),
            min(features$year[oot_oos_idx]),
            max(features$year[oot_oos_idx])))

## Verify no gap between splits
n_unassigned <- sum(is.na(oot_split_col))
if (n_unassigned > 0)
  warning(sprintf("[08_Split.R] WARNING: %d rows unassigned to any split.",
                  n_unassigned))

## OOT diagnostics
fn_oot_diag <- function(idx, label) {
  y_vec  <- features$y[idx]
  n_csi  <- sum(y_vec == 1L, na.rm = TRUE)
  n_obs  <- sum(!is.na(y_vec))
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
  fn_oot_diag(oot_train_idx, "train"),
  fn_oot_diag(oot_test_idx,  "test"),
  fn_oot_diag(oot_oos_idx,   "oos")
)

cat("\n  OOT class balance:\n")
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
##     y_firm = 1 if firm ever had a confirmed CSI event
##     NA-only firms (zombie/censored only) → 0 (non-event, design note [2])
firm_profile <- features[, .(
  y_firm = as.integer(any(y == 1L, na.rm=TRUE) == 1L)
), by = permno]

## Coerce NaN (produced when all y values are NA) to 0
firm_profile[is.nan(y_firm), y_firm := 0L]

cat(sprintf("  Firms total    : %d\n", nrow(firm_profile)))
cat(sprintf("  Ever-CSI firms : %d (%.1f%%)\n",
            sum(firm_profile$y_firm == 1L),
            100 * mean(firm_profile$y_firm == 1L)))
cat(sprintf("  Never-CSI firms: %d (%.1f%%)\n",
            sum(firm_profile$y_firm == 0L),
            100 * mean(firm_profile$y_firm == 0L)))

## 2B. Stratified split within each stratum (y_firm ∈ {0, 1})
##     Split 70% train / 30% test within each stratum separately,
##     then combine — guarantees class balance is preserved.
TRAIN_SIZE <- 0.70

fn_stratified_firm_split <- function(firm_dt, train_size, seed) {
  set.seed(seed)
  
  train_permno_list <- list()
  test_permno_list  <- list()
  
  for (stratum in c(0L, 1L)) {
    stratum_permnos <- firm_dt[y_firm == stratum, permno]
    n_stratum       <- length(stratum_permnos)
    
    ## Shuffle within stratum for randomness
    shuffled <- stratum_permnos[sample(n_stratum)]
    
    n_train <- round(n_stratum * train_size)
    train_permno_list[[as.character(stratum)]] <- shuffled[seq_len(n_train)]
    test_permno_list[[as.character(stratum)]]  <- shuffled[(n_train + 1L):n_stratum]
  }
  
  train_permno <- unlist(train_permno_list)
  test_permno  <- unlist(test_permno_list)
  
  return(list(
    train_permno = train_permno,
    test_permno  = test_permno
  ))
}

firm_split   <- fn_stratified_firm_split(firm_profile, TRAIN_SIZE, SEED)
train_permno <- firm_split$train_permno
test_permno  <- firm_split$test_permno

## 2C. Map firm-level assignment back to row indices
oos_train_idx <- which(features$permno %in% train_permno)
oos_test_idx  <- which(features$permno %in% test_permno)

## Build split column (all rows)
oos_split_col <- character(n_rows)
oos_split_col[oos_train_idx] <- "train"
oos_split_col[oos_test_idx]  <- "test"

## Verify complete coverage
n_unassigned_oos <- sum(oos_split_col == "")
if (n_unassigned_oos > 0)
  stop(sprintf("[08_Split.R] ASSERTION FAILED: %d rows unassigned in OOS split.",
               n_unassigned_oos))

cat(sprintf("  Train: %d rows | %d firms\n",
            length(oos_train_idx), length(train_permno)))
cat(sprintf("  Test : %d rows | %d firms\n",
            length(oos_test_idx),  length(test_permno)))

## 2D. OOS firm-split diagnostics — verify class balance preserved
fn_oos_diag <- function(idx, label, permno_vec) {
  y_vec     <- features$y[idx]
  perm_vec  <- features$permno[idx]
  n_csi     <- sum(y_vec == 1L, na.rm = TRUE)
  n_obs     <- sum(!is.na(y_vec))
  n_firms   <- n_distinct(perm_vec)
  csi_firms <- n_distinct(perm_vec[!is.na(y_vec) & y_vec == 1L])
  data.frame(
    split      = label,
    n_rows     = length(idx),
    n_firms    = n_firms,
    n_labelled = n_obs,
    n_csi_obs  = n_csi,
    n_csi_firms = csi_firms,
    pct_csi_firms = round(100 * csi_firms / n_firms, 2),
    prevalence = round(100 * n_csi / max(n_obs, 1L), 2)
  )
}

oos_diag <- rbind(
  fn_oos_diag(oos_train_idx, "train", train_permno),
  fn_oos_diag(oos_test_idx,  "test",  test_permno)
)

cat("\n  OOS firm-split class balance:\n")
print(oos_diag, row.names = FALSE)

## Verify stratification worked — prevalence should be similar across splits
prev_train <- oos_diag$prevalence[oos_diag$split == "train"]
prev_test  <- oos_diag$prevalence[oos_diag$split == "test"]
if (abs(prev_train - prev_test) > 3.0)
  warning(sprintf(
    "[08_Split.R] WARNING: OOS prevalence gap %.1f%% (train) vs %.1f%% (test) — ",
    prev_train, prev_test),
    "stratification may not have preserved class balance adequately.")

#==============================================================================#
# 3. k-Fold CV indices for HPO (used inside 09_Train.R)
#
#   5-fold time-series CV on the OOT training set.
#   Folds respect temporal order — never use future data to predict past.
#   Within each fold: train on earlier years, validate on next year block.
#
#   Also provides firm-level stratified k-fold for the OOS split.
#==============================================================================#

cat("\n[08_Split.R] Constructing k-fold CV indices...\n")

CV_K <- CV_FOLDS   # From config.R (default 5)

## 3A. Time-series CV folds on OOT train set
##     Divide training years into CV_K consecutive blocks.
##     Fold k: train on years 1..(k-1), validate on year block k.
##     This mimics expanding window time-series CV.

train_years <- sort(unique(features$year[oot_train_idx]))
n_train_yrs <- length(train_years)

## Assign each training year to a fold (roughly equal-sized blocks)
year_fold_assignment <- data.table(
  year = train_years,
  fold = as.integer(cut(seq_len(n_train_yrs),
                        breaks = CV_K,
                        labels = seq_len(CV_K)))
)

## For each fold k, validation = rows where year in fold k
## Training = rows where year in folds 1..(k-1)
oot_cv_folds <- lapply(seq_len(CV_K), function(k) {
  val_years   <- year_fold_assignment[fold == k,  year]
  train_years_k <- year_fold_assignment[fold < k, year]
  
  val_rows   <- which(features$year %in% val_years   & oot_split_col == "train")
  train_rows <- which(features$year %in% train_years_k & oot_split_col == "train")
  
  list(train = train_rows, validation = val_rows,
       val_years = val_years, n_val_years = length(val_years))
})

cat(sprintf("  OOT CV: %d folds | validation fold sizes:\n", CV_K))
for (k in seq_len(CV_K)) {
  cat(sprintf("    Fold %d: train %d rows | val %d rows (years %d–%d)\n",
              k,
              length(oot_cv_folds[[k]]$train),
              length(oot_cv_folds[[k]]$validation),
              min(oot_cv_folds[[k]]$val_years),
              max(oot_cv_folds[[k]]$val_years)))
}

## 3B. Firm-level stratified k-fold CV on OOS train set
##     Each fold holds out a different 1/k of firms as validation.
##     Stratified to preserve CSI firm balance within each validation fold.

fn_firm_kfold <- function(firm_dt, k, seed) {
  set.seed(seed)
  folds <- vector("list", k)
  
  for (stratum in c(0L, 1L)) {
    stratum_permnos <- firm_dt[y_firm == stratum, permno]
    n_stratum       <- length(stratum_permnos)
    shuffled        <- stratum_permnos[sample(n_stratum)]
    
    ## Assign firms to folds cyclically
    fold_assignments <- ((seq_len(n_stratum) - 1L) %% k) + 1L
    for (f in seq_len(k)) {
      folds[[f]] <- c(folds[[f]], shuffled[fold_assignments == f])
    }
  }
  
  ## For each fold f, build train/val row indices
  ## Val = firms in fold f | Train = firms in all other folds
  train_permnos_all <- firm_dt[, permno]
  
  lapply(seq_len(k), function(f) {
    val_permno   <- folds[[f]]
    train_permno_f <- setdiff(train_permnos_all[
      train_permnos_all %in% unlist(
        firm_dt[y_firm %in% c(0L, 1L), permno])], val_permno)
    
    ## Only use firms that were assigned to OOS train set
    train_permno_f <- intersect(train_permno_f, firm_split$train_permno)
    val_permno_f   <- intersect(val_permno,     firm_split$train_permno)
    
    val_rows   <- which(features$permno %in% val_permno_f   & oos_split_col == "train")
    train_rows <- which(features$permno %in% train_permno_f & oos_split_col == "train")
    
    list(train = train_rows, validation = val_rows,
         val_permno = val_permno_f)
  })
}

oos_cv_folds <- fn_firm_kfold(
  firm_dt = firm_profile[permno %in% train_permno],
  k       = CV_K,
  seed    = SEED
)

cat(sprintf("\n  OOS firm CV: %d folds | validation fold sizes:\n", CV_K))
for (k in seq_len(CV_K)) {
  cat(sprintf("    Fold %d: train %d rows | val %d rows (%d firms)\n",
              k,
              length(oos_cv_folds[[k]]$train),
              length(oos_cv_folds[[k]]$validation),
              length(oos_cv_folds[[k]]$val_permno)))
}

#==============================================================================#
# 4. Assertions
#==============================================================================#

cat("\n[08_Split.R] Running assertions...\n")

## A) OOT: every row assigned
stopifnot(
  "[08_Split.R] OOT: row coverage incomplete" =
    length(oot_train_idx) + length(oot_test_idx) + length(oot_oos_idx) +
    sum(is.na(oot_split_col)) == n_rows
)

## B) OOT: no overlap between train/test/oos
stopifnot(
  "[08_Split.R] OOT: train/test overlap" =
    length(intersect(oot_train_idx, oot_test_idx)) == 0,
  "[08_Split.R] OOT: train/oos overlap" =
    length(intersect(oot_train_idx, oot_oos_idx)) == 0,
  "[08_Split.R] OOT: test/oos overlap" =
    length(intersect(oot_test_idx, oot_oos_idx)) == 0
)

## C) OOS firm split: every row assigned
stopifnot(
  "[08_Split.R] OOS: row coverage incomplete" =
    length(oos_train_idx) + length(oos_test_idx) == n_rows
)

## D) OOS firm split: no permno overlap
stopifnot(
  "[08_Split.R] OOS: firm overlap between train and test" =
    length(intersect(train_permno, test_permno)) == 0
)

## E) OOS firm split: all permno accounted for
stopifnot(
  "[08_Split.R] OOS: not all permno assigned" =
    length(union(train_permno, test_permno)) == n_permno
)

## F) CV folds cover OOT train set
oot_cv_all_val <- unlist(lapply(oot_cv_folds, `[[`, "validation"))
stopifnot(
  "[08_Split.R] OOT CV: validation rows don't cover train set" =
    length(unique(oot_cv_all_val)) == length(oot_train_idx)
)

## G) Plausible split sizes
stopifnot(
  "[08_Split.R] OOT: train set too small (< 40% of rows)" =
    length(oot_train_idx) / n_rows >= 0.40,
  "[08_Split.R] OOS: train set not ~70% of rows" =
    abs(length(oos_train_idx) / n_rows - TRAIN_SIZE) < 0.05
)

cat("[08_Split.R] All assertions passed.\n")

#==============================================================================#
# 5. Save splits.rds
#==============================================================================#

splits <- list(
  
  ## Strategy 1: Out-of-Time
  oot = list(
    split_col  = oot_split_col,
    train_idx  = oot_train_idx,
    test_idx   = oot_test_idx,
    oos_idx    = oot_oos_idx,
    train_years = year_fold_assignment[fold <= CV_K, year],
    cv_folds   = oot_cv_folds
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
    n_rows       = n_rows,
    n_permno     = n_permno,
    train_size   = TRAIN_SIZE,
    cv_k         = CV_K,
    seed         = SEED,
    train_end    = TRAIN_END,
    test_start   = TEST_START,
    test_end     = TEST_END,
    oos_start    = OOS_START
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
# 6. Summary diagnostics
#==============================================================================#

cat("\n[08_Split.R] ══════════════════════════════════════\n")
cat("  Strategy 1 — Out-of-Time (OOT):\n")
print(oot_diag, row.names = FALSE)

cat("\n  Strategy 2 — Out-of-Sample firm-level (OOS):\n")
print(oos_diag, row.names = FALSE)

## Prevalence comparison plot
diag_combined <- rbind(
  cbind(strategy = "OOT",      oot_diag[oot_diag$split != "oos", c("split", "prevalence")]),
  cbind(strategy = "OOS_firm", oos_diag[, c("split", "prevalence")])
)

p_prev <- ggplot(diag_combined, aes(x = split, y = prevalence,
                                    fill = strategy)) +
  geom_col(position = "dodge", width = 0.6) +
  geom_hline(yintercept = 100 * mean(features$y == 1L, na.rm = TRUE),
             linetype = "dashed", colour = "grey40") +
  labs(
    title    = "CSI Prevalence by Split Strategy",
    subtitle = "Dashed line = overall prevalence",
    x        = "Split",
    y        = "Prevalence (%)",
    fill     = "Strategy"
  ) +
  theme_minimal(base_size = 12)

ggsave(
  file.path(DIR_FIGURES, "split_prevalence.png"),
  plot   = p_prev,
  width  = PLOT_WIDTH,
  height = PLOT_HEIGHT,
  dpi    = PLOT_DPI
)

cat(sprintf("\n  Prevalence check:\n"))
cat(sprintf("    OOT  train : %.2f%% | test : %.2f%% | oos : %.2f%%\n",
            oot_diag$prevalence[1],
            oot_diag$prevalence[2],
            oot_diag$prevalence[3]))
cat(sprintf("    OOS  train : %.2f%% | test : %.2f%%\n",
            oos_diag$prevalence[1],
            oos_diag$prevalence[2]))

## Export split labels as parquet for Python (pyreadr cannot read nested lists)
library(arrow)

arrow::write_parquet(
  data.frame(permno = features$permno,
             year   = features$year,
             split  = splits$oot$split_col),
  file.path(DIR_FEATURES, "split_labels_oot.parquet")
)

arrow::write_parquet(
  data.frame(permno = features$permno,
             year   = features$year,
             split  = splits$oos_firm$split_col),
  file.path(DIR_FEATURES, "split_labels_oos.parquet")
)

cat("[08_Split.R] Split label parquets saved for Python.\n")

cat("\n[08_Split.R] DONE:", format(Sys.time()), "\n")

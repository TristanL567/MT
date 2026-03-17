#==============================================================================#
#==== 09_Train.R ==============================================================#
#==== XGBoost Training — Raw Features & Latent Features ======================#
#==============================================================================#
#
# PURPOSE:
#   Train XGBoost on two parallel feature sets:
#     (A) Raw features   : features_raw.rds  — all engineered ratios
#     (B) Latent features: features_latent.parquet — VAE z1..z24 + recon error
#
#   For each feature set:
#     1. Apply QuantileTransform (uniform [0,1]) — fitted on train only
#     2. Run Bayesian HPO with expanding window CV (OOT folds 2–5)
#     3. Train final model on full train set with optimal parameters
#     4. Predict on test set and OOS set
#     5. Save model and predictions
#
# INPUTS:
#   - config.R
#   - PATH_FEATURES_RAW       : (permno, year, y, censored, ~463 features)
#   - PATH_FEATURES_LATENT    : (permno, year, y, censored, z1..z24,
#                                 vae_recon_error, split)
#   - PATH_SPLITS             : splits.rds from 08_Split.R
#
# OUTPUTS:
#   - PATH_MODELS_DIR/xgb_raw.rds    : model + predictions (raw features)
#   - PATH_MODELS_DIR/xgb_latent.rds : model + predictions (latent features)
#   - PATH_EVAL_RESULTS              : combined evaluation table
#
# DESIGN DECISIONS:
#
#   [1] FOLD 1 SKIPPED in CV:
#       OOT expanding window fold 1 has zero training rows (1993–1997).
#       Cannot train on zero rows — fold 1 excluded from HPO CV.
#       HPO uses folds 2–5 (4 effective folds).
#
#   [2] QUANTILE TRANSFORM (uniform [0,1]) applied inside CV loop:
#       Fitted on each fold's training rows only.
#       Applied to that fold's validation rows using training ECDF.
#       Final model: fitted on full training set, applied to test/OOS.
#       Prevents distribution leakage between folds.
#
#   [3] AVERAGE PRECISION as HPO metric:
#       XGBoost native eval_metric = "aucpr" (area under PR curve).
#       Threshold-free, robust to class imbalance.
#       Recall at fixed FPR computed post-hoc for thesis primary metric.
#
#   [4] SCALE_POS_WEIGHT recomputed per fold:
#       Class balance varies slightly across temporal folds.
#       Recomputing n_neg/n_pos per fold is more correct than using
#       the overall training set ratio for every fold.
#
#   [5] y=NA ROWS EXCLUDED from training and evaluation:
#       Zombie/censored rows (y=NA) are excluded before building DMatrix.
#       They were retained through the pipeline for the autoencoder but
#       have no label for supervised training.
#
#   [6] BAYESIAN OPTIMISATION via rBayesianOptimization:
#       UCB acquisition function with kappa=2.576 (95% confidence bound).
#       10 initial random points + 20 Bayesian iterations = 30 total.
#       Each iteration runs xgb.cv with early stopping.
#
#   [7] TWO FEATURE SETS run in sequence:
#       Results stored in a named list and saved separately.
#       10_Evaluate.R loads both and compares performance.
#
#==============================================================================#

source("config.R")

suppressPackageStartupMessages({
  library(dplyr)
  library(data.table)
  library(xgboost)
  library(rBayesianOptimization)
  library(pROC)
  library(arrow)
  library(Matrix)
  library(tidyr)
})

cat("\n[09_Train.R] START:", format(Sys.time()), "\n")

#==============================================================================#
# 0. Check 08B output exists
#==============================================================================#

if (!file.exists(PATH_FEATURES_LATENT)) {
  stop(paste(
    "[09_Train.R] features_latent.parquet not found.",
    "Run 08B_Autoencoder.py before 09_Train.R.",
    sep = "\n"
  ))
}

#==============================================================================#
# 1. Load inputs
#==============================================================================#

cat("[09_Train.R] Loading inputs...\n")

features_raw    <- as.data.table(readRDS(PATH_FEATURES_RAW))
features_latent <- as.data.table(arrow::read_parquet(PATH_FEATURES_LATENT))
splits          <- readRDS(PATH_SPLITS)

cat(sprintf("  features_raw    : %d rows, %d cols\n",
            nrow(features_raw), ncol(features_raw)))
cat(sprintf("  features_latent : %d rows, %d cols\n",
            nrow(features_latent), ncol(features_latent)))

#==============================================================================#
# 2. Define feature column sets
#
#   Raw features: all numeric columns except identifiers and label
#   Latent features: z1..z24 + vae_recon_error
#   Both sets also include macro variables as features
#==============================================================================#

## Columns that are identifiers or labels — never features
ID_COLS <- c("permno", "year", "y", "censored", "param_id",
             "gvkey", "datadate", "lifetime_years",
             "fiscal_year_end_month", "split")

## Raw feature columns
RAW_FEATURE_COLS <- setdiff(
  names(features_raw)[sapply(features_raw, is.numeric)],
  ID_COLS
)

## Latent feature columns
LATENT_FEATURE_COLS <- c(
  paste0("z", seq_len(24)),   # z1..z24
  "vae_recon_error"
)
LATENT_FEATURE_COLS <- intersect(LATENT_FEATURE_COLS, names(features_latent))

cat(sprintf("  Raw features    : %d columns\n", length(RAW_FEATURE_COLS)))
cat(sprintf("  Latent features : %d columns\n", length(LATENT_FEATURE_COLS)))

#==============================================================================#
# 3. Quantile Transform utility
#
#   Fits uniform [0,1] QuantileTransform on train_mat column-wise.
#   Applies to train, val, test using train ECDF.
#   Handles NA — imputes with column median before transformation.
#==============================================================================#

fn_quantile_transform <- function(train_mat, apply_mats = list()) {
  ## Ensure matrix
  train_mat <- as.matrix(train_mat)
  n_cols    <- ncol(train_mat)
  n_train   <- nrow(train_mat)
  
  ## Storage
  train_out  <- matrix(NA_real_, nrow = n_train, ncol = n_cols)
  apply_outs <- lapply(apply_mats, function(m)
    matrix(NA_real_, nrow = nrow(m), ncol = n_cols))
  
  epsilon <- 0.5 / n_train   # Open-interval boundary
  
  for (j in seq_len(n_cols)) {
    
    x_tr <- train_mat[, j]
    
    ## Impute NAs with training median before transformation
    med_tr <- median(x_tr, na.rm = TRUE)
    if (is.na(med_tr)) med_tr <- 0
    x_tr[is.na(x_tr)] <- med_tr
    
    ## Rank-based transform for training set → (epsilon, 1-epsilon)
    n_valid <- sum(!is.na(x_tr))
    ranks   <- rank(x_tr, ties.method = "average")
    train_out[, j] <- (ranks - 0.5) / n_valid
    
    ## Build ECDF from training data
    ecdf_fn <- ecdf(x_tr)
    
    ## Apply to additional matrices using training ECDF
    for (k in seq_along(apply_mats)) {
      x_ap <- as.matrix(apply_mats[[k]])[, j]
      x_ap[is.na(x_ap)] <- med_tr   # Impute with training median
      probs <- ecdf_fn(x_ap)
      probs <- pmax(epsilon, pmin(probs, 1 - epsilon))
      apply_outs[[k]][, j] <- probs
    }
  }
  
  colnames(train_out) <- colnames(train_mat)
  for (k in seq_along(apply_outs)) {
    colnames(apply_outs[[k]]) <- colnames(train_mat)
  }
  
  list(train = train_out, applied = apply_outs)
}

#==============================================================================#
# 4. Recall at fixed FPR
#
#   Primary thesis evaluation metric: Recall at FPR <= threshold.
#   Computed post-hoc from predicted probabilities.
#==============================================================================#

fn_recall_at_fpr <- function(y_true, y_pred, fpr_target) {
  roc_obj     <- pROC::roc(y_true, y_pred, quiet = TRUE)
  fpr_vals    <- 1 - roc_obj$specificities
  recall_vals <- roc_obj$sensitivities
  eligible    <- which(fpr_vals <= fpr_target)
  if (length(eligible) == 0) return(0)
  max(recall_vals[eligible])
}

fn_avg_precision <- function(y_true, y_pred) {
  ## Trapezoid approximation of area under precision-recall curve
  roc_obj  <- pROC::roc(y_true, y_pred, quiet = TRUE)
  pr_auc   <- tryCatch(
    as.numeric(pROC::auc(pROC::roc(y_true, y_pred,
                                   quiet = TRUE))),
    error = function(e) NA_real_
  )
  pr_auc
}

#==============================================================================#
# 5. Core XGBoost training function
#
#   Adapted from the custom XGBoost_Training_revised function, extended with:
#     - Quantile transform per fold (design note [2])
#     - Average Precision metric (design note [3])
#     - OOT expanding window CV (design note [1])
#     - Two feature set support (design note [7])
#==============================================================================#

fn_train_xgboost <- function(
    feature_set_name,   # "raw" or "latent" — used for naming outputs
    panel_dt,           # data.table with permno, year, y + feature cols
    feature_cols,       # character vector of feature column names
    splits,             # splits.rds object from 08_Split.R
    n_init_points = 10L,
    n_iter_bayes  = 20L,
    nrounds_bo    = 500L,
    nrounds_final = 1000L,
    early_stop_bo = 20L,
    early_stop_final = 50L,
    nthread       = max(1L, parallel::detectCores() - 1L)
) {
  
  cat(sprintf("\n[09_Train.R] ── Feature set: %s ──────────────────────────\n",
              toupper(feature_set_name)))
  
  ##──────────────────────────────────────────────────────────────────────────##
  ## 5A. Build train / test / OOS subsets
  ##     Exclude y=NA rows (zombie/censored) — no label for training
  ##──────────────────────────────────────────────────────────────────────────##
  
  oot_split <- splits$oot$split_col
  
  ## Assign split membership to panel rows
  panel_dt[, split_oot := oot_split]
  
  ## Filter to labelled rows only
  train_full <- panel_dt[split_oot == "train" & !is.na(y)]
  test_set   <- panel_dt[split_oot == "test"  & !is.na(y)]
  oos_set    <- panel_dt[split_oot == "oos"   & !is.na(y)]
  
  cat(sprintf("  Train (labelled): %d | Test: %d | OOS: %d\n",
              nrow(train_full), nrow(test_set), nrow(oos_set)))
  
  y_train <- as.integer(train_full$y)
  y_test  <- as.integer(test_set$y)
  y_oos   <- as.integer(oos_set$y)
  
  ## Feature matrices
  X_train_raw_mat <- as.matrix(train_full[, ..feature_cols])
  X_test_raw_mat  <- as.matrix(test_set[,  ..feature_cols])
  X_oos_raw_mat   <- as.matrix(oos_set[,   ..feature_cols])
  
  ##──────────────────────────────────────────────────────────────────────────##
  ## 5B. Full-set quantile transform (for final model)
  ##     Fitted on full train set — applied to test and OOS
  ##──────────────────────────────────────────────────────────────────────────##
  
  cat("  Applying quantile transform (full train)...\n")
  
  qt_full <- fn_quantile_transform(
    train_mat  = X_train_raw_mat,
    apply_mats = list(test = X_test_raw_mat, oos = X_oos_raw_mat)
  )
  
  X_train_qt <- qt_full$train
  X_test_qt  <- qt_full$applied[["test"]]
  X_oos_qt   <- qt_full$applied[["oos"]]
  
  ## Overall class weight (for final model)
  n_neg_full       <- sum(y_train == 0L)
  n_pos_full       <- sum(y_train == 1L)
  scale_pos_weight <- n_neg_full / n_pos_full
  
  cat(sprintf("  Class balance — neg: %d | pos: %d | weight: %.2f\n",
              n_neg_full, n_pos_full, scale_pos_weight))
  
  ## Final DMatrix
  dtrain_full <- xgb.DMatrix(data  = X_train_qt, label = y_train)
  dtest       <- xgb.DMatrix(data  = X_test_qt,  label = y_test)
  doos        <- xgb.DMatrix(data  = X_oos_qt,   label = y_oos)
  
  ##──────────────────────────────────────────────────────────────────────────##
  ## 5C. Prepare CV fold indices for xgb.cv
  ##
  ##     OOT folds from splits$oot$cv_folds are train/val row indices
  ##     relative to the FULL panel (including y=NA rows).
  ##     Must remap to indices within train_full (y-labelled rows only).
  ##
  ##     Fold 1 skipped — zero training rows (design note [1]).
  ##──────────────────────────────────────────────────────────────────────────##
  
  cat("  Building CV fold indices...\n")
  
  ## Row numbers of train_full within the full panel
  train_full_rows <- which(oot_split == "train" & !is.na(panel_dt$y))
  
  ## Map panel row → position within train_full (0-indexed for xgb.cv)
  panel_to_train_pos <- integer(nrow(panel_dt))
  panel_to_train_pos[train_full_rows] <- seq_len(length(train_full_rows)) - 1L
  
  ## Build folds list for xgb.cv — validation indices only
  ## xgb.cv folds parameter: list of 0-indexed row indices for each val fold
  cv_folds_raw   <- splits$oot$cv_folds
  n_folds_total  <- length(cv_folds_raw)
  
  xgb_folds <- list()
  valid_fold_ids <- integer(0)
  
  for (k in seq_len(n_folds_total)) {
    
    fold_k       <- cv_folds_raw[[k]]
    train_rows_k <- fold_k$train
    val_rows_k   <- fold_k$validation
    
    ## Skip fold 1 — no training rows (design note [1])
    if (length(train_rows_k) == 0L) {
      cat(sprintf("    Fold %d: SKIPPED (0 training rows)\n", k))
      next
    }
    
    ## Intersect with labelled train rows
    val_rows_labelled   <- intersect(val_rows_k,   train_full_rows)
    train_rows_labelled <- intersect(train_rows_k, train_full_rows)
    
    if (length(val_rows_labelled) == 0L || length(train_rows_labelled) == 0L) {
      cat(sprintf("    Fold %d: SKIPPED (empty after label filter)\n", k))
      next
    }
    
    ## Convert to 0-indexed positions within train_full
    val_positions <- panel_to_train_pos[val_rows_labelled]
    
    xgb_folds[[length(xgb_folds) + 1L]] <- val_positions
    valid_fold_ids <- c(valid_fold_ids, k)
    
    cat(sprintf("    Fold %d: train %d | val %d rows\n",
                k,
                length(train_rows_labelled),
                length(val_rows_labelled)))
  }
  
  cat(sprintf("  Using %d CV folds for HPO\n", length(xgb_folds)))
  
  ##──────────────────────────────────────────────────────────────────────────##
  ## 5D. Bayesian HPO
  ##──────────────────────────────────────────────────────────────────────────##
  
  cat("\n[09_Train.R] Starting Bayesian HPO...\n")
  
  current_iter <- 0L
  total_iters  <- n_init_points + n_iter_bayes
  
  ## Objective function for Bayesian optimisation
  fn_xgb_cv_objective <- function(eta, max_depth, subsample,
                                  colsample_bytree, min_child_weight,
                                  gamma, lambda, alpha) {
    
    current_iter <<- current_iter + 1L
    
    depth_int <- as.integer(round(max_depth))
    mcw_int   <- as.integer(round(min_child_weight))
    
    cat(sprintf(
      "  [%02d/%02d] eta=%.3f | depth=%d | sub=%.2f | col=%.2f | "
      "mcw=%d | gamma=%.2f | L2=%.2f | L1=%.2f\n",
      current_iter, total_iters,
      eta, depth_int, subsample, colsample_bytree,
      mcw_int, gamma, lambda, alpha
    ))
    
    params <- list(
      booster          = "gbtree",
      objective        = "binary:logistic",
      eval_metric      = "aucpr",          # Average Precision (design note [3])
      eta              = eta,
      max_depth        = depth_int,
      subsample        = subsample,
      colsample_bytree = colsample_bytree,
      min_child_weight = mcw_int,
      gamma            = gamma,
      lambda           = lambda,
      alpha            = alpha,
      scale_pos_weight = scale_pos_weight,
      max_delta_step   = 1L,              # Recommended for imbalanced data
      nthread          = nthread
    )
    
    cv_result <- tryCatch(
      xgb.cv(
        params                = params,
        data                  = dtrain_full,
        nrounds               = nrounds_bo,
        folds                 = xgb_folds,
        early_stopping_rounds = early_stop_bo,
        verbose               = 0,
        maximize              = TRUE
      ),
      error = function(e) {
        cat(sprintf("    CV error: %s\n", e$message))
        NULL
      }
    )
    
    if (is.null(cv_result))
      return(list(Score = -Inf, Pred = 0))
    
    best_ap <- max(cv_result$evaluation_log$test_aucpr_mean)
    cat(sprintf("    → CV AUCPR: %.4f\n", best_ap))
    list(Score = best_ap, Pred = 0)
  }
  
  ## Hyperparameter search bounds
  bounds_list <- list(
    eta              = c(0.01, 0.30),
    max_depth        = c(3L,   8L),
    subsample        = c(0.50, 1.00),
    colsample_bytree = c(0.50, 1.00),
    min_child_weight = c(1L,   20L),
    gamma            = c(0.00, 5.00),
    lambda           = c(0.00, 5.00),
    alpha            = c(0.00, 5.00)
  )
  
  set.seed(SEED)
  time_bo <- system.time({
    bayes_result <- tryCatch(
      BayesianOptimization(
        FUN         = fn_xgb_cv_objective,
        bounds      = bounds_list,
        init_points = n_init_points,
        n_iter      = n_iter_bayes,
        acq         = "ucb",
        kappa       = 2.576,
        verbose     = FALSE
      ),
      error = function(e) {
        stop(sprintf("[09_Train.R] Bayesian optimisation failed: %s",
                     e$message))
      }
    )
  })
  
  ## Parse results
  bo_history <- bayes_result$History |>
    dplyr::rename(aucpr = Value) |>
    dplyr::mutate(
      max_depth        = as.integer(round(max_depth)),
      min_child_weight = as.integer(round(min_child_weight))
    ) |>
    dplyr::arrange(dplyr::desc(aucpr))
  
  cat("\n  Top 3 HPO configurations:\n")
  print(head(bo_history, 3))
  
  ##──────────────────────────────────────────────────────────────────────────##
  ## 5E. Final model training
  ##
  ##     1. Determine optimal nrounds via CV with best parameters
  ##     2. Train on full training set with optimal nrounds
  ##──────────────────────────────────────────────────────────────────────────##
  
  cat("\n[09_Train.R] Training final model...\n")
  
  best_row <- bo_history[1, ]
  
  final_params <- list(
    booster          = "gbtree",
    objective        = "binary:logistic",
    eval_metric      = "aucpr",
    eta              = best_row$eta,
    max_depth        = best_row$max_depth,
    subsample        = best_row$subsample,
    colsample_bytree = best_row$colsample_bytree,
    min_child_weight = best_row$min_child_weight,
    gamma            = best_row$gamma,
    lambda           = best_row$lambda,
    alpha            = best_row$alpha,
    scale_pos_weight = scale_pos_weight,
    max_delta_step   = 1L,
    nthread          = nthread
  )
  
  ## Find optimal nrounds
  cat("  Finding optimal nrounds...\n")
  cv_final <- xgb.cv(
    params                = final_params,
    data                  = dtrain_full,
    nrounds               = nrounds_final,
    folds                 = xgb_folds,
    early_stopping_rounds = early_stop_final,
    verbose               = 0,
    maximize              = TRUE
  )
  
  optimal_rounds <- cv_final$evaluation_log[
    which.max(cv_final$evaluation_log$test_aucpr_mean), iter
  ]
  cv_aucpr_mean  <- max(cv_final$evaluation_log$test_aucpr_mean)
  cv_aucpr_sd    <- cv_final$evaluation_log[
    which.max(cv_final$evaluation_log$test_aucpr_mean), test_aucpr_std
  ]
  
  cat(sprintf("  Optimal rounds : %d\n", optimal_rounds))
  cat(sprintf("  CV AUCPR       : %.4f (+/- %.4f)\n",
              cv_aucpr_mean, cv_aucpr_sd))
  
  ## Train final model on full training set
  model_final <- xgb.train(
    params  = final_params,
    data    = dtrain_full,
    nrounds = optimal_rounds,
    verbose = 0
  )
  
  ##──────────────────────────────────────────────────────────────────────────##
  ## 5F. Predictions on test and OOS
  ##──────────────────────────────────────────────────────────────────────────##
  
  cat("[09_Train.R] Generating predictions...\n")
  
  preds_train <- predict(model_final, dtrain_full)
  preds_test  <- predict(model_final, dtest)
  preds_oos   <- predict(model_final, doos)
  
  ##──────────────────────────────────────────────────────────────────────────##
  ## 5G. Evaluation metrics
  ##──────────────────────────────────────────────────────────────────────────##
  
  fn_eval_metrics <- function(y_true, y_pred, set_name) {
    auc_roc  <- as.numeric(pROC::auc(pROC::roc(y_true, y_pred, quiet=TRUE)))
    ap       <- fn_avg_precision(y_true, y_pred)
    rec_fpr3 <- fn_recall_at_fpr(y_true, y_pred, fpr_target = 0.03)
    rec_fpr5 <- fn_recall_at_fpr(y_true, y_pred, fpr_target = 0.05)
    brier    <- mean((y_pred - y_true)^2)
    
    data.frame(
      feature_set   = feature_set_name,
      set           = set_name,
      auc_roc       = round(auc_roc,  4),
      avg_precision = round(ap,       4),
      recall_fpr3   = round(rec_fpr3, 4),
      recall_fpr5   = round(rec_fpr5, 4),
      brier         = round(brier,    4),
      stringsAsFactors = FALSE
    )
  }
  
  metrics_train <- fn_eval_metrics(y_train, preds_train, "train_insample")
  metrics_test  <- fn_eval_metrics(y_test,  preds_test,  "test")
  metrics_oos   <- fn_eval_metrics(y_oos,   preds_oos,   "oos")
  
  eval_table <- rbind(metrics_train, metrics_test, metrics_oos)
  
  cat("\n  Evaluation results:\n")
  print(eval_table, row.names = FALSE)
  
  cat(sprintf(
    "\n  [%s] CV AUCPR: %.4f | Test AUCPR: %.4f | Test Recall@FPR5: %.4f\n",
    toupper(feature_set_name),
    cv_aucpr_mean,
    metrics_test$avg_precision,
    metrics_test$recall_fpr5
  ))
  
  ##──────────────────────────────────────────────────────────────────────────##
  ## 5H. Feature importance
  ##──────────────────────────────────────────────────────────────────────────##
  
  importance_mat <- xgb.importance(
    feature_names = feature_cols,
    model         = model_final
  )
  
  cat(sprintf("\n  Top 10 features by gain:\n"))
  print(head(importance_mat[order(-importance_mat$Gain), ], 10))
  
  ##──────────────────────────────────────────────────────────────────────────##
  ## 5I. Return
  ##──────────────────────────────────────────────────────────────────────────##
  
  list(
    ## Model
    model          = model_final,
    optimal_rounds = optimal_rounds,
    params         = final_params,
    
    ## HPO
    bo_history     = bo_history,
    cv_aucpr_mean  = cv_aucpr_mean,
    cv_aucpr_sd    = cv_aucpr_sd,
    
    ## Predictions
    preds = list(
      train = data.table(permno = train_full$permno,
                         year   = train_full$year,
                         y      = y_train,
                         p_csi  = preds_train),
      test  = data.table(permno = test_set$permno,
                         year   = test_set$year,
                         y      = y_test,
                         p_csi  = preds_test),
      oos   = data.table(permno = oos_set$permno,
                         year   = oos_set$year,
                         y      = y_oos,
                         p_csi  = preds_oos)
    ),
    
    ## Evaluation
    eval_table   = eval_table,
    importance   = importance_mat,
    
    ## Meta
    feature_set  = feature_set_name,
    feature_cols = feature_cols,
    n_features   = length(feature_cols),
    scale_pos_weight = scale_pos_weight,
    time_bo      = time_bo
  )
}

#==============================================================================#
# 6. Run training — Feature Set A: Raw features
#==============================================================================#

cat("\n[09_Train.R] ══════════════════════════════════════\n")
cat("  FEATURE SET A: Raw features\n")
cat("[09_Train.R] ══════════════════════════════════════\n")

result_raw <- fn_train_xgboost(
  feature_set_name = "raw",
  panel_dt         = copy(features_raw),
  feature_cols     = RAW_FEATURE_COLS,
  splits           = splits
)

## Save raw model
saveRDS(result_raw,
        file.path(DIR_MODELS, "xgb_raw.rds"))
cat("[09_Train.R] xgb_raw.rds saved.\n")

#==============================================================================#
# 7. Run training — Feature Set B: Latent features
#==============================================================================#

cat("\n[09_Train.R] ══════════════════════════════════════\n")
cat("  FEATURE SET B: Latent features\n")
cat("[09_Train.R] ══════════════════════════════════════\n")

## Join split column onto latent features for compatibility with fn_train_xgboost
## features_latent already has a split column from 08B_Autoencoder.py
## Rename to avoid conflict with the split_oot column added inside the function
setnames(features_latent, "split", "vae_split", skip_absent = TRUE)

result_latent <- fn_train_xgboost(
  feature_set_name = "latent",
  panel_dt         = copy(features_latent),
  feature_cols     = LATENT_FEATURE_COLS,
  splits           = splits
)

## Save latent model
saveRDS(result_latent,
        file.path(DIR_MODELS, "xgb_latent.rds"))
cat("[09_Train.R] xgb_latent.rds saved.\n")

#==============================================================================#
# 8. Combined evaluation table
#==============================================================================#

eval_combined <- rbind(result_raw$eval_table,
                       result_latent$eval_table)

cat("\n[09_Train.R] ══════════════════════════════════════\n")
cat("  Combined evaluation results:\n\n")
print(eval_combined, row.names = FALSE)

saveRDS(eval_combined, PATH_EVAL_RESULTS)
cat("\n[09_Train.R] evaluation_results.rds saved.\n")

#==============================================================================#
# 9. Assertions
#==============================================================================#

cat("[09_Train.R] Running assertions...\n")

## A) Predictions exist for all three sets
for (nm in c("raw", "latent")) {
  res <- if (nm == "raw") result_raw else result_latent
  stopifnot(
    nrow(res$preds$train) > 0,
    nrow(res$preds$test)  > 0,
    nrow(res$preds$oos)   > 0
  )
}

## B) No NA predictions
stopifnot(
  !anyNA(result_raw$preds$test$p_csi),
  !anyNA(result_latent$preds$test$p_csi)
)

## C) Predictions in [0, 1]
stopifnot(
  all(result_raw$preds$test$p_csi    >= 0 & result_raw$preds$test$p_csi    <= 1),
  all(result_latent$preds$test$p_csi >= 0 & result_latent$preds$test$p_csi <= 1)
)

## D) Test AUCPR > 0.1 (better than random given ~12% prevalence)
stopifnot(
  result_raw$eval_table[result_raw$eval_table$set == "test", "avg_precision"] > 0.10,
  result_latent$eval_table[result_latent$eval_table$set == "test", "avg_precision"] > 0.10
)

cat("[09_Train.R] All assertions passed.\n")

#==============================================================================#
# 10. Summary
#==============================================================================#

cat("\n[09_Train.R] ══════════════════════════════════════\n")
cat("  Final summary:\n\n")

cat(sprintf("  %-10s | CV AUCPR | Test AUCPR | Test R@FPR3 | Test R@FPR5\n",
            "Model"))
cat(sprintf("  %-10s | %-8s | %-10s | %-11s | %-11s\n",
            "----------", "--------", "----------",
            "-----------", "-----------"))

for (res in list(result_raw, result_latent)) {
  te <- res$eval_table[res$eval_table$set == "test", ]
  cat(sprintf(
    "  %-10s | %.4f   | %.4f     | %.4f      | %.4f\n",
    res$feature_set,
    res$cv_aucpr_mean,
    te$avg_precision,
    te$recall_fpr3,
    te$recall_fpr5
  ))
}

cat(sprintf("\n[09_Train.R] DONE: %s\n", format(Sys.time())))
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
#   - PATH_FEATURES_LATENT    : features_latent.parquet from 08B_Autoencoder.py
#   - PATH_SPLITS             : splits.rds from 08_Split.R
#
# OUTPUTS:
#   - DIR_MODELS/xgb_raw.rds    : model + predictions (raw features)
#   - DIR_MODELS/xgb_latent.rds : model + predictions (latent features)
#   - PATH_EVAL_RESULTS         : combined evaluation table
#
# CV DESIGN — EXPANDING WINDOW (not standard k-fold):
#
#   Expanding window CV trains on all years up to fold k, validates on the
#   next year block. Training set grows with each fold:
#
#     Fold 2: train [1998–2001] → validate [2002–2006]
#     Fold 3: train [1998–2006] → validate [2007–2010]
#     Fold 4: train [1998–2010] → validate [2011–2015]
#     Fold 5: train [1998–2015] → validate (to test boundary)
#
#   Fold 1 skipped — zero training rows.
#
# LABEL SHIFT:
#   features(t) predict y(t+1) — one year ahead.
#   Prevents contemporaneous leakage from price features (max_dd_12m)
#   into the CSI label which is triggered by the same price decline.
#
# DESIGN DECISIONS:
#   [1] Fold 1 skipped — zero training rows.
#   [2] QT uniform [0,1] fitted on full train, applied to test/OOS.
#   [3] Average Precision (AUCPR) as HPO metric.
#   [4] scale_pos_weight = n_neg/n_pos corrects class imbalance.
#   [5] y=NA rows excluded — no label for supervised training.
#   [6] Bayesian HPO: UCB kappa=2.576, 10 init + 20 Bayes = 30 total.
#   [7] Two feature sets run in sequence and compared.
#   [8] CV metrics from final model cv_final, not from HPO iterations.
#   [9] Split labels joined by (permno, year) for latent features —
#       safe regardless of row order after parquet round-trip from Python.
#
#==============================================================================#

source("config.R")

suppressPackageStartupMessages({
  library(dplyr)
  library(data.table)
  library(xgboost)
  library(rBayesianOptimization)
  library(pROC)
  library(PRROC)
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
#==============================================================================#

ID_COLS <- c("permno", "year", "y", "censored", "param_id",
             "gvkey", "datadate", "lifetime_years",
             "fiscal_year_end_month", "split", "vae_split", "split_oot")

RAW_FEATURE_COLS <- setdiff(
  names(features_raw)[sapply(features_raw, is.numeric)],
  ID_COLS
)

LATENT_FEATURE_COLS <- intersect(
  c(paste0("z", seq_len(24)), "vae_recon_error"),
  names(features_latent)
)

cat(sprintf("  Raw features    : %d columns\n", length(RAW_FEATURE_COLS)))
cat(sprintf("  Latent features : %d columns\n", length(LATENT_FEATURE_COLS)))

#==============================================================================#
# 3. Utility functions
#==============================================================================#

fn_quantile_transform <- function(train_mat, apply_mats = list()) {
  
  train_mat <- as.matrix(train_mat)
  n_cols    <- ncol(train_mat)
  n_train   <- nrow(train_mat)
  epsilon   <- 0.5 / n_train
  
  train_out  <- matrix(NA_real_, nrow = n_train, ncol = n_cols)
  apply_outs <- lapply(apply_mats, function(m)
    matrix(NA_real_, nrow = nrow(m), ncol = n_cols))
  
  for (j in seq_len(n_cols)) {
    x_tr   <- train_mat[, j]
    med_tr <- median(x_tr, na.rm = TRUE)
    if (is.na(med_tr)) med_tr <- 0
    x_tr[is.na(x_tr)] <- med_tr
    
    ranks          <- rank(x_tr, ties.method = "average")
    train_out[, j] <- (ranks - 0.5) / length(ranks)
    
    ecdf_fn <- ecdf(x_tr)
    for (k in seq_along(apply_mats)) {
      x_ap                 <- as.matrix(apply_mats[[k]])[, j]
      x_ap[is.na(x_ap)]   <- med_tr
      probs                <- ecdf_fn(x_ap)
      apply_outs[[k]][, j] <- pmax(epsilon, pmin(probs, 1 - epsilon))
    }
  }
  
  colnames(train_out) <- colnames(train_mat)
  for (k in seq_along(apply_outs))
    colnames(apply_outs[[k]]) <- colnames(train_mat)
  
  list(train = train_out, applied = apply_outs)
}

fn_avg_precision <- function(y_true, y_pred) {
  tryCatch({
    pr_obj <- PRROC::pr.curve(
      scores.class0 = y_pred[y_true == 1L],
      scores.class1 = y_pred[y_true == 0L],
      curve         = FALSE
    )
    pr_obj$auc.integral
  }, error = function(e) NA_real_)
}

fn_recall_at_fpr <- function(y_true, y_pred, fpr_target) {
  roc_obj     <- pROC::roc(y_true, y_pred, quiet = TRUE)
  fpr_vals    <- 1 - roc_obj$specificities
  recall_vals <- roc_obj$sensitivities
  eligible    <- which(fpr_vals <= fpr_target)
  if (length(eligible) == 0L) return(0)
  max(recall_vals[eligible])
}

fn_eval_metrics <- function(y_true, y_pred, feature_set_name, set_name) {
  data.frame(
    feature_set   = feature_set_name,
    set           = set_name,
    auc_roc       = round(as.numeric(pROC::auc(
      pROC::roc(y_true, y_pred, quiet = TRUE))), 4),
    avg_precision = round(fn_avg_precision(y_true, y_pred),       4),
    recall_fpr3   = round(fn_recall_at_fpr(y_true, y_pred, 0.03), 4),
    recall_fpr5   = round(fn_recall_at_fpr(y_true, y_pred, 0.05), 4),
    brier         = round(mean((y_pred - y_true)^2),              4),
    stringsAsFactors = FALSE
  )
}

#==============================================================================#
# 4. Core XGBoost training function
#==============================================================================#

fn_train_xgboost <- function(
    feature_set_name,
    panel_dt,
    feature_cols,
    splits,
    n_init_points    = 10L,
    n_iter_bayes     = 20L,
    nrounds_bo       = 500L,
    nrounds_final    = 1000L,
    early_stop_bo    = 20L,
    early_stop_final = 50L,
    nthread          = max(1L, parallel::detectCores() - 1L)
) {
  
  cat(sprintf(
    "\n[09_Train.R] ── Feature set: %s ──────────────────────────\n",
    toupper(feature_set_name)
  ))
  
  ##──────────────────────────────────────────────────────────────────────────##
  ## 4A. Assign split labels
  ##
  ##   For raw features: positional assignment from splits$oot$split_col
  ##   is safe because features_raw row order matches splits exactly.
  ##
  ##   For latent features: split_oot was pre-joined by (permno, year)
  ##   in Section 6 before this function is called. Use that column
  ##   directly — do NOT overwrite with positional assignment.
  ##
  ##   oot_split is derived from panel_dt$split_oot in both cases so
  ##   all downstream references to oot_split work identically.
  ##──────────────────────────────────────────────────────────────────────────##
  
  if (!"split_oot" %in% names(panel_dt)) {
    ## Raw features path — positional assignment is safe
    panel_dt[, split_oot := splits$oot$split_col]
    cat("  Split assigned positionally.\n")
  } else {
    cat("  Using pre-joined split_oot column.\n")
  }
  
  ## Derive oot_split vector from the column — used in 4C for row indexing
  oot_split <- panel_dt$split_oot
  
  ##──────────────────────────────────────────────────────────────────────────##
  ## 4B. Build train / test / OOS subsets
  ##     features(t) predict y(t+1) — one-year-ahead, no leakage
  ##──────────────────────────────────────────────────────────────────────────##
  
  panel_dt <- panel_dt[order(permno, year)]
  panel_dt[, y_next := shift(y, n = 1L, type = "lead"), by = permno]
  
  ## Re-derive oot_split after reorder — order may have changed
  oot_split <- panel_dt$split_oot
  
  train_full <- panel_dt[split_oot == "train" & !is.na(y_next)]
  test_set   <- panel_dt[split_oot == "test"  & !is.na(y_next)]
  oos_set    <- panel_dt[split_oot == "oos"   & !is.na(y_next)]
  
  y_train <- as.integer(train_full$y_next)
  y_test  <- as.integer(test_set$y_next)
  y_oos   <- as.integer(oos_set$y_next)
  
  ## Verify labels are clean — no NAs, only 0/1
  stopifnot(!anyNA(y_train), all(y_train %in% c(0L, 1L)))
  stopifnot(!anyNA(y_test),  all(y_test  %in% c(0L, 1L)))
  stopifnot(!anyNA(y_oos),   all(y_oos   %in% c(0L, 1L)))
  
  cat(sprintf("  Train (labelled): %d | Test: %d | OOS: %d\n",
              nrow(train_full), nrow(test_set), nrow(oos_set)))
  
  X_train_mat <- as.matrix(train_full[, ..feature_cols])
  X_test_mat  <- as.matrix(test_set[,  ..feature_cols])
  X_oos_mat   <- as.matrix(oos_set[,   ..feature_cols])
  
  ##──────────────────────────────────────────────────────────────────────────##
  ## 4C. Quantile transform — fitted on full train, applied to test/OOS
  ##──────────────────────────────────────────────────────────────────────────##
  
  cat("  Applying quantile transform (full train → test, OOS)...\n")
  
  qt_full    <- fn_quantile_transform(
    train_mat  = X_train_mat,
    apply_mats = list(test = X_test_mat, oos = X_oos_mat)
  )
  X_train_qt <- qt_full$train
  X_test_qt  <- qt_full$applied[["test"]]
  X_oos_qt   <- qt_full$applied[["oos"]]
  
  n_neg_full       <- sum(y_train == 0L)
  n_pos_full       <- sum(y_train == 1L)
  scale_pos_weight <- n_neg_full / n_pos_full
  
  cat(sprintf("  Class balance — neg: %d | pos: %d | weight: %.2f\n",
              n_neg_full, n_pos_full, scale_pos_weight))
  
  dtrain_full <- xgb.DMatrix(data = X_train_qt, label = y_train)
  dtest       <- xgb.DMatrix(data = X_test_qt,  label = y_test)
  doos        <- xgb.DMatrix(data = X_oos_qt,   label = y_oos)
  
  ##──────────────────────────────────────────────────────────────────────────##
  ## 4D. Build CV fold indices
  ##
  ##   OOT fold indices from splits$oot$cv_folds are relative to the full
  ##   panel. Must remap to 0-indexed positions within train_full.
  ##
  ##   CRITICAL: train_full_rows must use the same y_next filter as
  ##   train_full above — NOT the old !is.na(y) filter.
  ##   Uses oot_split which was re-derived after reorder in 4B.
  ##──────────────────────────────────────────────────────────────────────────##
  
  cat("  Building CV fold indices...\n")
  
  train_full_rows <- which(oot_split == "train" & !is.na(panel_dt$y_next))
  
  panel_to_pos <- integer(nrow(panel_dt))
  panel_to_pos[train_full_rows] <- seq_along(train_full_rows) - 1L
  
  cv_folds_raw   <- splits$oot$cv_folds
  xgb_folds      <- list()
  valid_fold_ids  <- integer(0)
  
  for (k in seq_along(cv_folds_raw)) {
    
    fold_k       <- cv_folds_raw[[k]]
    train_rows_k <- fold_k$train
    val_rows_k   <- fold_k$validation
    
    if (length(train_rows_k) == 0L) {
      cat(sprintf("    Fold %d: SKIPPED (0 training rows)\n", k))
      next
    }
    
    val_labelled   <- intersect(val_rows_k,   train_full_rows)
    train_labelled <- intersect(train_rows_k, train_full_rows)
    
    if (length(val_labelled) == 0L || length(train_labelled) == 0L) {
      cat(sprintf("    Fold %d: SKIPPED (empty after label filter)\n", k))
      next
    }
    
    xgb_folds[[length(xgb_folds) + 1L]] <- panel_to_pos[val_labelled]
    valid_fold_ids <- c(valid_fold_ids, k)
    
    cat(sprintf("    Fold %d: train %d | val %d rows\n",
                k, length(train_labelled), length(val_labelled)))
  }
  
  cat(sprintf("  Using %d CV folds for HPO\n", length(xgb_folds)))
  
  ##──────────────────────────────────────────────────────────────────────────##
  ## 4E. Bayesian HPO
  ##──────────────────────────────────────────────────────────────────────────##
  
  cat("\n[09_Train.R] Starting Bayesian HPO...\n")
  
  current_iter <- 0L
  total_iters  <- n_init_points + n_iter_bayes
  
  fn_xgb_cv_objective <- function(eta, max_depth, subsample,
                                  colsample_bytree, min_child_weight,
                                  gamma, lambda, alpha) {
    
    current_iter <<- current_iter + 1L
    depth_int     <- as.integer(round(max_depth))
    mcw_int       <- as.integer(round(min_child_weight))
    
    params <- list(
      booster          = "gbtree",
      objective        = "binary:logistic",
      eval_metric      = "aucpr",
      eta              = eta,
      max_depth        = depth_int,
      subsample        = subsample,
      colsample_bytree = colsample_bytree,
      min_child_weight = mcw_int,
      gamma            = gamma,
      lambda           = lambda,
      alpha            = alpha,
      scale_pos_weight = scale_pos_weight,
      max_delta_step   = 1L,
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
    cat(sprintf("  [%02d/%02d] → CV AUCPR: %.4f\n",
                current_iter, total_iters, best_ap))
    list(Score = best_ap, Pred = 0)
  }
  
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
      error = function(e) stop(sprintf(
        "[09_Train.R] Bayesian optimisation failed: %s", e$message))
    )
  })
  
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
  ## 4F. Final model
  ##──────────────────────────────────────────────────────────────────────────##
  
  cat("\n[09_Train.R] Training final model...\n")
  
  best_row <- bo_history[1L, ]
  
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
  
  cat("  Finding optimal nrounds (final CV)...\n")
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
  cv_aucpr_mean <- max(cv_final$evaluation_log$test_aucpr_mean)
  cv_aucpr_sd   <- cv_final$evaluation_log[
    which.max(cv_final$evaluation_log$test_aucpr_mean), test_aucpr_std
  ]
  
  cat(sprintf("  Optimal rounds : %d\n", optimal_rounds))
  cat(sprintf("  CV AUCPR (mean): %.4f\n", cv_aucpr_mean))
  cat(sprintf("  CV AUCPR (sd)  : %.4f\n", cv_aucpr_sd))
  
  cat("\n  Per-fold AUCPR breakdown at optimal rounds:\n")
  log_at_opt     <- cv_final$evaluation_log[iter == optimal_rounds]
  fold_col_names <- grep("^test_aucpr_(?!mean|std)",
                         names(cv_final$evaluation_log),
                         perl = TRUE, value = TRUE)
  
  if (length(fold_col_names) > 0) {
    for (i in seq_along(fold_col_names))
      cat(sprintf("    Fold %d AUCPR : %.4f\n",
                  valid_fold_ids[i], log_at_opt[[fold_col_names[i]]]))
  } else {
    cat("    (Per-fold breakdown not available in this xgboost version)\n")
  }
  cat(sprintf("    Mean AUCPR   : %.4f (+/- %.4f)\n",
              cv_aucpr_mean, cv_aucpr_sd))
  
  model_final <- xgb.train(
    params  = final_params,
    data    = dtrain_full,
    nrounds = optimal_rounds,
    verbose = 0
  )
  
  ##──────────────────────────────────────────────────────────────────────────##
  ## 4G. Predictions
  ##──────────────────────────────────────────────────────────────────────────##
  
  cat("[09_Train.R] Generating predictions...\n")
  
  preds_train <- predict(model_final, dtrain_full)
  preds_test  <- predict(model_final, dtest)
  preds_oos   <- predict(model_final, doos)
  
  ##──────────────────────────────────────────────────────────────────────────##
  ## 4H. Evaluation
  ##──────────────────────────────────────────────────────────────────────────##
  
  metrics_train <- fn_eval_metrics(y_train, preds_train,
                                   feature_set_name, "train_insample")
  metrics_test  <- fn_eval_metrics(y_test,  preds_test,
                                   feature_set_name, "test")
  metrics_oos   <- fn_eval_metrics(y_oos,   preds_oos,
                                   feature_set_name, "oos")
  
  metrics_cv <- data.frame(
    feature_set   = feature_set_name,
    set           = "cv_expanding_window",
    auc_roc       = NA_real_,
    avg_precision = round(cv_aucpr_mean, 4),
    recall_fpr3   = NA_real_,
    recall_fpr5   = NA_real_,
    brier         = NA_real_,
    stringsAsFactors = FALSE
  )
  
  eval_table <- rbind(metrics_cv, metrics_train, metrics_test, metrics_oos)
  
  cat("\n  Evaluation results:\n")
  print(eval_table, row.names = FALSE)
  
  cat(sprintf(
    paste0("\n  [%s] CV AUCPR: %.4f | Test AP: %.4f | ",
           "Test AUC-ROC: %.4f | Test R@FPR5: %.4f\n"),
    toupper(feature_set_name),
    cv_aucpr_mean,
    metrics_test$avg_precision,
    metrics_test$auc_roc,
    metrics_test$recall_fpr5
  ))
  
  ##──────────────────────────────────────────────────────────────────────────##
  ## 4I. Feature importance
  ##──────────────────────────────────────────────────────────────────────────##
  
  importance_mat <- xgb.importance(
    feature_names = feature_cols,
    model         = model_final
  )
  
  cat("\n  Top 10 features by gain:\n")
  print(head(importance_mat[order(-importance_mat$Gain), ], 10L))
  
  ##──────────────────────────────────────────────────────────────────────────##
  ## 4J. Return
  ##──────────────────────────────────────────────────────────────────────────##
  
  list(
    model          = model_final,
    optimal_rounds = optimal_rounds,
    params         = final_params,
    bo_history     = bo_history,
    cv_aucpr_mean  = cv_aucpr_mean,
    cv_aucpr_sd    = cv_aucpr_sd,
    cv_log         = cv_final$evaluation_log,
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
    eval_table       = eval_table,
    importance       = importance_mat,
    feature_set      = feature_set_name,
    feature_cols     = feature_cols,
    n_features       = length(feature_cols),
    scale_pos_weight = scale_pos_weight,
    time_bo          = time_bo
  )
}

#==============================================================================#
# 5. Feature Set A — Raw features
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

saveRDS(result_raw, file.path(DIR_MODELS, "xgb_raw.rds"))
cat("[09_Train.R] xgb_raw.rds saved.\n")

#==============================================================================#
# 6. Feature Set B — Latent features
#
#   split_oot joined by (permno, year) — safe regardless of row order
#   after parquet round-trip from Python. The function detects the
#   pre-joined column and uses it directly without positional override.
#==============================================================================#

cat("\n[09_Train.R] ══════════════════════════════════════\n")
cat("  FEATURE SET B: Latent features\n")
cat("[09_Train.R] ══════════════════════════════════════\n")

## Rename Python split column to avoid conflict with split_oot
setnames(features_latent, "split", "vae_split", skip_absent = TRUE)

## Build split reference from features_raw (guaranteed correct order)
split_ref <- data.table(
  permno    = features_raw$permno,
  year      = features_raw$year,
  split_oot = splits$oot$split_col
)

## Join split labels onto latent features by (permno, year)
features_latent <- merge(
  features_latent,
  split_ref,
  by    = c("permno", "year"),
  all.x = TRUE
)

n_missing <- sum(is.na(features_latent$split_oot))
cat(sprintf("  Latent rows        : %d\n", nrow(features_latent)))
cat(sprintf("  split_oot NA count : %d (should be 0)\n", n_missing))
if (n_missing > 0)
  stop("[09_Train.R] split_oot join failed — check permno/year alignment.")

result_latent <- fn_train_xgboost(
  feature_set_name = "latent",
  panel_dt         = copy(features_latent),
  feature_cols     = LATENT_FEATURE_COLS,
  splits           = splits
)

saveRDS(result_latent, file.path(DIR_MODELS, "xgb_latent.rds"))
cat("[09_Train.R] xgb_latent.rds saved.\n")

#==============================================================================#
# 7. Combined evaluation table
#==============================================================================#

eval_combined <- rbind(result_raw$eval_table,
                       result_latent$eval_table)

cat("\n[09_Train.R] ══════════════════════════════════════\n")
cat("  Combined evaluation results:\n\n")
print(eval_combined, row.names = FALSE)

saveRDS(eval_combined, PATH_EVAL_RESULTS)
cat("\n[09_Train.R] evaluation_results.rds saved.\n")

#==============================================================================#
# 8. Assertions
#==============================================================================#

cat("[09_Train.R] Running assertions...\n")

for (nm in c("raw", "latent")) {
  res <- if (nm == "raw") result_raw else result_latent
  stopifnot(
    nrow(res$preds$train) > 0L,
    nrow(res$preds$test)  > 0L,
    nrow(res$preds$oos)   > 0L
  )
}

stopifnot(
  !anyNA(result_raw$preds$test$p_csi),
  !anyNA(result_latent$preds$test$p_csi),
  all(between(result_raw$preds$test$p_csi,    0, 1)),
  all(between(result_latent$preds$test$p_csi, 0, 1)),
  result_raw$cv_aucpr_mean    > 0.12,
  result_latent$cv_aucpr_mean > 0.12
)

raw_test <- result_raw$eval_table[result_raw$eval_table$set == "test", ]
if (!is.na(raw_test$avg_precision) && !is.na(raw_test$auc_roc))
  if (abs(raw_test$avg_precision - raw_test$auc_roc) < 0.001)
    warning("[09_Train.R] avg_precision == auc_roc — check PRROC.")

cat("[09_Train.R] All assertions passed.\n")

#==============================================================================#
# 9. Final summary
#==============================================================================#

cat("\n[09_Train.R] ══════════════════════════════════════\n")
cat("  Final comparison summary:\n\n")
cat(sprintf("  %-10s | CV AUCPR | Test AP  | Test AUC | R@FPR3 | R@FPR5\n",
            "Model"))
cat(sprintf("  %-10s | %-8s | %-8s | %-8s | %-6s | %-6s\n",
            "----------", "--------", "--------",
            "--------", "------", "------"))

for (res in list(result_raw, result_latent)) {
  te <- res$eval_table[res$eval_table$set == "test", ]
  cat(sprintf(
    "  %-10s | %.4f   | %.4f   | %.4f   | %.4f | %.4f\n",
    res$feature_set,
    res$cv_aucpr_mean,
    te$avg_precision,
    te$auc_roc,
    te$recall_fpr3,
    te$recall_fpr5
  ))
}

cat(sprintf("\n[09_Train.R] DONE: %s\n", format(Sys.time())))

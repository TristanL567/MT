#==============================================================================#
#==== 09_Train.R ==============================================================#
#==== XGBoost Training — M1 / M2 / M3 / M4 ===================================#
#==============================================================================#
#
# PURPOSE:
#   Train XGBoost on four parallel feature sets:
#
#     M1 — Fundamentals only  : features_fund.rds
#          No price features. Ex-ante screening model.
#
#     M2 — VAE latent (fund)  : features_latent_fund.parquet
#          VAE trained on fund features → z1–z24 + recon error.
#          Tests whether VAE adds signal over raw fundamentals.
#
#     M3 — Full raw features  : features_raw.rds
#          All ~463 engineered features including price. Triage benchmark.
#
#     M4 — VAE latent (raw)   : features_latent_raw.parquet
#          VAE trained on full feature set → z1–z24 + recon error.
#          Tests whether VAE compresses full features usefully.
#
# MODEL SELECTION:
#   Set MODELS_TO_RUN below. Completed models are loaded from disk
#   automatically for the final comparison table.
#
# OUTPUTS:
#   DIR_MODELS/xgb_fund.rds          [M1]
#   DIR_MODELS/xgb_latent_fund.rds   [M2]
#   DIR_MODELS/xgb_raw.rds           [M3]
#   DIR_MODELS/xgb_latent_raw.rds    [M4]
#   PATH_EVAL_RESULTS                combined evaluation table
#
# CV DESIGN — EXPANDING WINDOW:
#   Fold 2: train [1998–2001] → validate [2002–2006]
#   Fold 3: train [1998–2006] → validate [2007–2010]
#   Fold 4: train [1998–2010] → validate [2011–2015]
#   Fold 1 skipped — zero training rows.
#
# LABEL SHIFT:
#   features(t) predict y(t+1) — one year ahead.
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
# CONFIG — Select which models to run
#
#   "raw"         = M3 — full feature set          (existing baseline)
#   "fund"        = M1 — fundamentals only          (NEW)
#   "latent_fund" = M2 — VAE latent on fund input   (NEW)
#   "latent_raw"  = M4 — VAE latent on raw input    (NEW)
#
#   Examples:
#     MODELS_TO_RUN <- c("fund", "latent_fund")
#     MODELS_TO_RUN <- c("latent_fund", "latent_raw")
#     MODELS_TO_RUN <- c("raw", "fund", "latent_fund", "latent_raw")
#==============================================================================#

MODELS_TO_RUN <- c("fund", "latent_fund")   # <- CHANGE THIS

cat(sprintf("[09_Train.R] Models selected: %s\n",
            paste(toupper(MODELS_TO_RUN), collapse = ", ")))

#==============================================================================#
# 0. File existence checks
#==============================================================================#

FILE_REQUIREMENTS <- list(
  raw         = PATH_FEATURES_RAW,
  fund        = PATH_FEATURES_FUND,
  latent_fund = PATH_FEATURES_LATENT_FUND,
  latent_raw  = PATH_FEATURES_LATENT_RAW
)

PREREQ_MESSAGES <- list(
  raw         = "Run 06B_Feature_Eng.R first.",
  fund        = "Run 06B_Feature_Eng.R first.",
  latent_fund = "Run 08B_Autoencoder.py with VAE_INPUT='fund' first.",
  latent_raw  = "Run 08B_Autoencoder.py with VAE_INPUT='raw' first."
)

for (nm in MODELS_TO_RUN) {
  p <- FILE_REQUIREMENTS[[nm]]
  if (!file.exists(p))
    stop(sprintf("[09_Train.R] Missing file for '%s':\n  %s\n  %s",
                 nm, p, PREREQ_MESSAGES[[nm]]))
}

#==============================================================================#
# 1. Load inputs
#==============================================================================#

cat("[09_Train.R] Loading inputs...\n")

# Always load features_raw — needed for split reference in latent join
features_raw <- as.data.table(readRDS(PATH_FEATURES_RAW))
splits       <- readRDS(PATH_SPLITS)
cat(sprintf("  features_raw         : %d rows, %d cols\n",
            nrow(features_raw), ncol(features_raw)))

if ("fund" %in% MODELS_TO_RUN) {
  features_fund <- as.data.table(readRDS(PATH_FEATURES_FUND))
  cat(sprintf("  features_fund        : %d rows, %d cols\n",
              nrow(features_fund), ncol(features_fund)))
}

if ("latent_fund" %in% MODELS_TO_RUN) {
  features_latent_fund <- as.data.table(
    arrow::read_parquet(PATH_FEATURES_LATENT_FUND))
  cat(sprintf("  features_latent_fund : %d rows, %d cols\n",
              nrow(features_latent_fund), ncol(features_latent_fund)))
}

if ("latent_raw" %in% MODELS_TO_RUN) {
  features_latent_raw <- as.data.table(
    arrow::read_parquet(PATH_FEATURES_LATENT_RAW))
  cat(sprintf("  features_latent_raw  : %d rows, %d cols\n",
              nrow(features_latent_raw), ncol(features_latent_raw)))
}

#==============================================================================#
# 2. Define feature column sets
#==============================================================================#

ID_COLS <- c("permno", "year", "y", "censored", "param_id",
             "gvkey", "datadate", "lifetime_years",
             "fiscal_year_end_month", "split", "vae_split", "split_oot")

LATENT_COLS <- c(paste0("z", seq_len(24)), "vae_recon_error")

RAW_FEATURE_COLS <- setdiff(
  names(features_raw)[sapply(features_raw, is.numeric)], ID_COLS)
cat(sprintf("  M3 raw features      : %d columns\n", length(RAW_FEATURE_COLS)))

if ("fund" %in% MODELS_TO_RUN) {
  FUND_FEATURE_COLS <- setdiff(
    names(features_fund)[sapply(features_fund, is.numeric)], ID_COLS)
  cat(sprintf("  M1 fund features     : %d columns\n", length(FUND_FEATURE_COLS)))
}

if ("latent_fund" %in% MODELS_TO_RUN) {
  LATENT_FUND_COLS <- intersect(LATENT_COLS, names(features_latent_fund))
  cat(sprintf("  M2 latent_fund cols  : %d columns\n", length(LATENT_FUND_COLS)))
}

if ("latent_raw" %in% MODELS_TO_RUN) {
  LATENT_RAW_COLS <- intersect(LATENT_COLS, names(features_latent_raw))
  cat(sprintf("  M4 latent_raw cols   : %d columns\n", length(LATENT_RAW_COLS)))
}

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
# 4. Helper — join split labels onto latent parquet data
#==============================================================================#

fn_join_split_latent <- function(features_latent_dt, features_raw_dt, splits) {
  setnames(features_latent_dt, "split", "vae_split", skip_absent = TRUE)
  split_ref <- data.table(
    permno    = features_raw_dt$permno,
    year      = features_raw_dt$year,
    split_oot = splits$oot$split_col
  )
  result    <- merge(features_latent_dt, split_ref,
                     by = c("permno", "year"), all.x = TRUE)
  n_missing <- sum(is.na(result$split_oot))
  if (n_missing > 0)
    stop(sprintf("[09_Train.R] split_oot join failed — %d NAs.", n_missing))
  result
}

#==============================================================================#
# 5. Core XGBoost training function
#==============================================================================#

fn_train_xgboost <- function(
    feature_set_name, panel_dt, feature_cols, splits,
    n_init_points    = 10L,
    n_iter_bayes     = 20L,
    nrounds_bo       = 500L,
    nrounds_final    = 1000L,
    early_stop_bo    = 20L,
    early_stop_final = 50L,
    nthread          = max(1L, parallel::detectCores() - 1L)
) {
  cat(sprintf("\n[09_Train.R] ── Feature set: %s ──\n",
              toupper(feature_set_name)))
  
  ## 5A. Split labels
  if (!"split_oot" %in% names(panel_dt)) {
    panel_dt[, split_oot := splits$oot$split_col]
    cat("  Split assigned positionally.\n")
  } else {
    cat("  Using pre-joined split_oot column.\n")
  }
  
  ## 5B. Label shift
  panel_dt <- panel_dt[order(permno, year)]
  panel_dt[, y_next := shift(y, n = 1L, type = "lead"), by = permno]
  oot_split  <- panel_dt$split_oot
  
  train_full <- panel_dt[split_oot == "train" & !is.na(y_next)]
  test_set   <- panel_dt[split_oot == "test"  & !is.na(y_next)]
  oos_set    <- panel_dt[split_oot == "oos"   & !is.na(y_next)]
  
  y_train <- as.integer(train_full$y_next)
  y_test  <- as.integer(test_set$y_next)
  y_oos   <- as.integer(oos_set$y_next)
  
  stopifnot(!anyNA(y_train), all(y_train %in% c(0L,1L)))
  stopifnot(!anyNA(y_test),  all(y_test  %in% c(0L,1L)))
  stopifnot(!anyNA(y_oos),   all(y_oos   %in% c(0L,1L)))
  
  cat(sprintf("  Train: %d | Test: %d | OOS: %d\n",
              nrow(train_full), nrow(test_set), nrow(oos_set)))
  
  X_train_mat <- as.matrix(train_full[, ..feature_cols])
  X_test_mat  <- as.matrix(test_set[,  ..feature_cols])
  X_oos_mat   <- as.matrix(oos_set[,   ..feature_cols])
  
  ## 5C. QT
  cat("  Applying quantile transform...\n")
  qt_full    <- fn_quantile_transform(X_train_mat,
                                      list(test=X_test_mat, oos=X_oos_mat))
  X_train_qt <- qt_full$train
  X_test_qt  <- qt_full$applied[["test"]]
  X_oos_qt   <- qt_full$applied[["oos"]]
  
  n_neg_full       <- sum(y_train == 0L)
  n_pos_full       <- sum(y_train == 1L)
  scale_pos_weight <- n_neg_full / n_pos_full
  cat(sprintf("  neg: %d | pos: %d | weight: %.2f\n",
              n_neg_full, n_pos_full, scale_pos_weight))
  
  dtrain_full <- xgb.DMatrix(data=X_train_qt, label=y_train)
  dtest       <- xgb.DMatrix(data=X_test_qt,  label=y_test)
  doos        <- xgb.DMatrix(data=X_oos_qt,   label=y_oos)
  
  ## 5D. CV folds
  cat("  Building CV folds...\n")
  train_full_rows <- which(oot_split == "train" & !is.na(panel_dt$y_next))
  panel_to_pos    <- integer(nrow(panel_dt))
  panel_to_pos[train_full_rows] <- seq_along(train_full_rows) - 1L
  
  xgb_folds     <- list()
  valid_fold_ids <- integer(0)
  
  for (k in seq_along(splits$oot$cv_folds)) {
    fold_k       <- splits$oot$cv_folds[[k]]
    train_rows_k <- fold_k$train
    val_rows_k   <- fold_k$validation
    if (length(train_rows_k) == 0L) {
      cat(sprintf("    Fold %d: SKIPPED\n", k)); next }
    val_labelled   <- intersect(val_rows_k,   train_full_rows)
    train_labelled <- intersect(train_rows_k, train_full_rows)
    if (length(val_labelled) == 0L || length(train_labelled) == 0L) {
      cat(sprintf("    Fold %d: SKIPPED (empty)\n", k)); next }
    xgb_folds[[length(xgb_folds)+1L]] <- panel_to_pos[val_labelled]
    valid_fold_ids <- c(valid_fold_ids, k)
    cat(sprintf("    Fold %d: train %d | val %d\n",
                k, length(train_labelled), length(val_labelled)))
  }
  cat(sprintf("  Using %d CV folds\n", length(xgb_folds)))
  
  ## 5E. Bayesian HPO
  cat("\n[09_Train.R] Bayesian HPO...\n")
  current_iter <- 0L
  total_iters  <- n_init_points + n_iter_bayes
  
  fn_xgb_cv_objective <- function(eta, max_depth, subsample,
                                  colsample_bytree, min_child_weight,
                                  gamma, lambda, alpha) {
    current_iter <<- current_iter + 1L
    params <- list(
      booster="gbtree", objective="binary:logistic", eval_metric="aucpr",
      eta=eta, max_depth=as.integer(round(max_depth)),
      subsample=subsample, colsample_bytree=colsample_bytree,
      min_child_weight=as.integer(round(min_child_weight)),
      gamma=gamma, lambda=lambda, alpha=alpha,
      scale_pos_weight=scale_pos_weight,
      max_delta_step=1L, nthread=nthread
    )
    cv_result <- tryCatch(
      xgb.cv(params=params, data=dtrain_full, nrounds=nrounds_bo,
             folds=xgb_folds, early_stopping_rounds=early_stop_bo,
             verbose=0, maximize=TRUE),
      error = function(e) { cat(sprintf("  CV error: %s\n", e$message)); NULL }
    )
    if (is.null(cv_result)) return(list(Score=-Inf, Pred=0))
    best_ap <- max(cv_result$evaluation_log$test_aucpr_mean)
    cat(sprintf("  [%02d/%02d] CV AUCPR: %.4f\n", current_iter, total_iters, best_ap))
    list(Score=best_ap, Pred=0)
  }
  
  bounds_list <- list(
    eta=c(0.01,0.30), max_depth=c(3L,8L),
    subsample=c(0.50,1.00), colsample_bytree=c(0.50,1.00),
    min_child_weight=c(1L,20L), gamma=c(0.00,5.00),
    lambda=c(0.00,5.00), alpha=c(0.00,5.00)
  )
  
  set.seed(SEED)
  time_bo <- system.time({
    bayes_result <- tryCatch(
      BayesianOptimization(FUN=fn_xgb_cv_objective, bounds=bounds_list,
                           init_points=n_init_points, n_iter=n_iter_bayes,
                           acq="ucb", kappa=2.576, verbose=FALSE),
      error=function(e) stop(sprintf("HPO failed: %s", e$message))
    )
  })
  
  bo_history <- bayes_result$History |>
    dplyr::rename(aucpr=Value) |>
    dplyr::mutate(max_depth=as.integer(round(max_depth)),
                  min_child_weight=as.integer(round(min_child_weight))) |>
    dplyr::arrange(dplyr::desc(aucpr))
  
  cat("\n  Top 3 HPO configs:\n")
  print(head(bo_history, 3))
  
  ## 5F. Final model
  cat("\n[09_Train.R] Training final model...\n")
  best_row     <- bo_history[1L,]
  final_params <- list(
    booster="gbtree", objective="binary:logistic", eval_metric="aucpr",
    eta=best_row$eta, max_depth=best_row$max_depth,
    subsample=best_row$subsample, colsample_bytree=best_row$colsample_bytree,
    min_child_weight=best_row$min_child_weight,
    gamma=best_row$gamma, lambda=best_row$lambda, alpha=best_row$alpha,
    scale_pos_weight=scale_pos_weight, max_delta_step=1L, nthread=nthread
  )
  
  cv_final <- xgb.cv(params=final_params, data=dtrain_full,
                     nrounds=nrounds_final, folds=xgb_folds,
                     early_stopping_rounds=early_stop_final,
                     verbose=0, maximize=TRUE)
  
  optimal_rounds <- cv_final$evaluation_log[
    which.max(cv_final$evaluation_log$test_aucpr_mean), iter]
  cv_aucpr_mean  <- max(cv_final$evaluation_log$test_aucpr_mean)
  cv_aucpr_sd    <- cv_final$evaluation_log[
    which.max(cv_final$evaluation_log$test_aucpr_mean), test_aucpr_std]
  
  cat(sprintf("  Optimal rounds : %d\n", optimal_rounds))
  cat(sprintf("  CV AUCPR (mean): %.4f\n", cv_aucpr_mean))
  
  model_final <- xgb.train(params=final_params, data=dtrain_full,
                           nrounds=optimal_rounds, verbose=0)
  
  ## 5G. Predictions
  preds_train <- predict(model_final, dtrain_full)
  preds_test  <- predict(model_final, dtest)
  preds_oos   <- predict(model_final, doos)
  
  ## 5H. Evaluation
  metrics_cv <- data.frame(
    feature_set=feature_set_name, set="cv_expanding_window",
    auc_roc=NA_real_, avg_precision=round(cv_aucpr_mean,4),
    recall_fpr3=NA_real_, recall_fpr5=NA_real_, brier=NA_real_,
    stringsAsFactors=FALSE
  )
  eval_table <- rbind(
    metrics_cv,
    fn_eval_metrics(y_train, preds_train, feature_set_name, "train_insample"),
    fn_eval_metrics(y_test,  preds_test,  feature_set_name, "test"),
    fn_eval_metrics(y_oos,   preds_oos,   feature_set_name, "oos")
  )
  
  cat("\n  Evaluation:\n")
  print(eval_table, row.names=FALSE)
  
  te <- eval_table[eval_table$set=="test",]
  cat(sprintf("\n  [%s] CV: %.4f | Test AP: %.4f | AUC: %.4f | R@FPR5: %.4f\n",
              toupper(feature_set_name), cv_aucpr_mean,
              te$avg_precision, te$auc_roc, te$recall_fpr5))
  
  ## 5I. Feature importance
  importance_mat <- xgb.importance(feature_names=feature_cols, model=model_final)
  cat("\n  Top 10 features:\n")
  print(head(importance_mat[order(-importance_mat$Gain),], 10L))
  
  ## 5J. Return
  list(
    model=model_final, optimal_rounds=optimal_rounds,
    params=final_params, bo_history=bo_history,
    cv_aucpr_mean=cv_aucpr_mean, cv_aucpr_sd=cv_aucpr_sd,
    cv_log=cv_final$evaluation_log,
    preds=list(
      train=data.table(permno=train_full$permno, year=train_full$year,
                       y=y_train, p_csi=preds_train),
      test =data.table(permno=test_set$permno,   year=test_set$year,
                       y=y_test,  p_csi=preds_test),
      oos  =data.table(permno=oos_set$permno,    year=oos_set$year,
                       y=y_oos,   p_csi=preds_oos)
    ),
    eval_table=eval_table, importance=importance_mat,
    feature_set=feature_set_name, feature_cols=feature_cols,
    n_features=length(feature_cols),
    scale_pos_weight=scale_pos_weight, time_bo=time_bo
  )
}

#==============================================================================#
# 6. Run selected models
#==============================================================================#

results_list <- list()

if ("raw" %in% MODELS_TO_RUN) {
  cat("\n[09_Train.R] ══ M3: Raw features ══\n")
  result_raw <- fn_train_xgboost("raw", copy(features_raw),
                                 RAW_FEATURE_COLS, splits)
  saveRDS(result_raw, file.path(DIR_MODELS, "xgb_raw.rds"))
  results_list[["raw"]] <- result_raw
  cat("[09_Train.R] xgb_raw.rds saved.\n")
}

if ("fund" %in% MODELS_TO_RUN) {
  cat("\n[09_Train.R] ══ M1: Fundamentals only ══\n")
  result_fund <- fn_train_xgboost("fund", copy(features_fund),
                                  FUND_FEATURE_COLS, splits)
  saveRDS(result_fund, file.path(DIR_MODELS, "xgb_fund.rds"))
  results_list[["fund"]] <- result_fund
  cat("[09_Train.R] xgb_fund.rds saved.\n")
}

if ("latent_fund" %in% MODELS_TO_RUN) {
  cat("\n[09_Train.R] ══ M2: Latent (fund VAE) ══\n")
  features_latent_fund <- fn_join_split_latent(
    features_latent_fund, features_raw, splits)
  result_latent_fund <- fn_train_xgboost("latent_fund",
                                         copy(features_latent_fund),
                                         LATENT_FUND_COLS, splits)
  saveRDS(result_latent_fund, file.path(DIR_MODELS, "xgb_latent_fund.rds"))
  results_list[["latent_fund"]] <- result_latent_fund
  cat("[09_Train.R] xgb_latent_fund.rds saved.\n")
}

if ("latent_raw" %in% MODELS_TO_RUN) {
  cat("\n[09_Train.R] ══ M4: Latent (raw VAE) ══\n")
  features_latent_raw <- fn_join_split_latent(
    features_latent_raw, features_raw, splits)
  result_latent_raw <- fn_train_xgboost("latent_raw",
                                        copy(features_latent_raw),
                                        LATENT_RAW_COLS, splits)
  saveRDS(result_latent_raw, file.path(DIR_MODELS, "xgb_latent_raw.rds"))
  results_list[["latent_raw"]] <- result_latent_raw
  cat("[09_Train.R] xgb_latent_raw.rds saved.\n")
}

#==============================================================================#
# 7. Load previously saved results not run this session
#==============================================================================#

SAVED_FILES <- list(
  raw         = file.path(DIR_MODELS, "xgb_raw.rds"),
  fund        = file.path(DIR_MODELS, "xgb_fund.rds"),
  latent_fund = file.path(DIR_MODELS, "xgb_latent_fund.rds"),
  latent_raw  = file.path(DIR_MODELS, "xgb_latent_raw.rds")
)

for (nm in names(SAVED_FILES)) {
  if (!nm %in% MODELS_TO_RUN && file.exists(SAVED_FILES[[nm]])) {
    results_list[[nm]] <- readRDS(SAVED_FILES[[nm]])
    cat(sprintf("[09_Train.R] Loaded existing %s for comparison.\n",
                basename(SAVED_FILES[[nm]])))
  }
}

#==============================================================================#
# 8. Combined evaluation table
#==============================================================================#

eval_combined <- do.call(rbind, lapply(results_list, function(r) r$eval_table))

cat("\n[09_Train.R] ══ Combined evaluation ══\n")
print(eval_combined, row.names=FALSE)
saveRDS(eval_combined, PATH_EVAL_RESULTS)
cat("[09_Train.R] evaluation_results.rds saved.\n")

#==============================================================================#
# 9. Assertions
#==============================================================================#

cat("[09_Train.R] Assertions...\n")
for (nm in MODELS_TO_RUN) {
  res <- results_list[[nm]]
  stopifnot(
    nrow(res$preds$train) > 0L, nrow(res$preds$test) > 0L,
    nrow(res$preds$oos)   > 0L, !anyNA(res$preds$test$p_csi),
    all(between(res$preds$test$p_csi, 0, 1)),
    res$cv_aucpr_mean > 0.05
  )
}
cat("[09_Train.R] All assertions passed.\n")

#==============================================================================#
# 10. Final summary table
#==============================================================================#

cat("\n[09_Train.R] ══ Final comparison: M1 / M2 / M3 / M4 ══\n\n")
cat(sprintf("  %-14s | CV AUCPR | Test AP  | AUC-ROC  | R@FPR3 | R@FPR5\n",
            "Model"))
cat(sprintf("  %-14s | %-8s | %-8s | %-8s | %-6s | %-6s\n",
            "--------------","--------","--------","--------","------","------"))

MODEL_LABELS <- c(raw="M3 raw", fund="M1 fund",
                  latent_fund="M2 lat-fund", latent_raw="M4 lat-raw")

for (nm in c("raw", "fund", "latent_fund", "latent_raw")) {
  if (!nm %in% names(results_list)) next
  res <- results_list[[nm]]
  te  <- res$eval_table[res$eval_table$set == "test",]
  cat(sprintf("  %-14s | %.4f   | %.4f   | %.4f   | %.4f | %.4f\n",
              MODEL_LABELS[nm], res$cv_aucpr_mean,
              te$avg_precision, te$auc_roc, te$recall_fpr3, te$recall_fpr5))
}

cat(sprintf("\n[09_Train.R] DONE: %s\n", format(Sys.time())))
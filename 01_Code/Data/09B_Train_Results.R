#==============================================================================#
#==== 09B_Train_Results.R =====================================================#
#==== Training Diagnostics — Feature Importance, CV Metrics, Calibration =====#
#==============================================================================#
#
# PURPOSE:
#   Inspect and summarise the trained XGBoost models WITHOUT touching the
#   test set. All outputs here are derived from:
#     - Training set predictions (in-sample)
#     - CV metrics from expanding window folds (honest estimate)
#     - Model internals (feature importance, tree structure)
#
#   The test set is reserved exclusively for 10_Evaluate.R.
#   Run this script after 09_Train.R to validate training quality before
#   committing to formal test-set evaluation.
#
# INPUTS:
#   - config.R
#   - DIR_MODELS/xgb_raw.rds    : trained raw feature model
#   - DIR_MODELS/xgb_latent.rds : trained latent feature model
#
# OUTPUTS:
#   - DIR_FIGURES/train_cv_curve_raw.png
#   - DIR_FIGURES/train_cv_curve_latent.png
#   - DIR_FIGURES/train_feature_importance_raw.png
#   - DIR_FIGURES/train_feature_importance_latent.png
#   - DIR_FIGURES/train_calibration_raw.png
#   - DIR_FIGURES/train_calibration_latent.png
#   - DIR_TABLES/train_summary.rds   : combined summary table
#
# SECTIONS:
#   1. Load models
#   2. CV metric summary (mean, SD, per-fold breakdown)
#   3. Feature importance plots (Gain, Cover, Frequency)
#   4. In-sample calibration plot (training set only)
#   5. In-sample precision-recall curve (training set only)
#   6. HPO history — did Bayesian search converge?
#   7. Combined summary table saved to disk
#
#==============================================================================#

source("config.R")

suppressPackageStartupMessages({
  library(dplyr)
  library(data.table)
  library(ggplot2)
  library(xgboost)
  library(PRROC)
  library(pROC)
  library(scales)
})

cat("\n[09B_Train_Results.R] START:", format(Sys.time()), "\n")

#==============================================================================#
# 1. Load trained models
#==============================================================================#

cat("[09B] Loading models...\n")

path_raw    <- file.path(DIR_MODELS, "xgb_raw.rds")
path_latent <- file.path(DIR_MODELS, "xgb_latent.rds")

if (!file.exists(path_raw))
  stop("[09B] xgb_raw.rds not found — run 09_Train.R first.")
if (!file.exists(path_latent))
  stop("[09B] xgb_latent.rds not found — run 09_Train.R first.")

result_raw    <- readRDS(path_raw)
result_latent <- readRDS(path_latent)

models <- list(raw = result_raw, latent = result_latent)

cat(sprintf("  xgb_raw    : %d features | %d rounds | CV AUCPR %.4f\n",
            result_raw$n_features,
            result_raw$optimal_rounds,
            result_raw$cv_aucpr_mean))
cat(sprintf("  xgb_latent : %d features | %d rounds | CV AUCPR %.4f\n",
            result_latent$n_features,
            result_latent$optimal_rounds,
            result_latent$cv_aucpr_mean))

#==============================================================================#
# 2. CV Metric Summary
#==============================================================================#

cat("\n[09B] ── Section 2: CV Metric Summary ────────────────────────────\n")

## CV AUCPR from cv_log — extract at optimal rounds for each model
for (nm in names(models)) {
  res <- models[[nm]]
  
  cat(sprintf("\n  Model: %s\n", toupper(nm)))
  cat(sprintf("    Optimal rounds : %d\n",   res$optimal_rounds))
  cat(sprintf("    CV AUCPR mean  : %.4f\n", res$cv_aucpr_mean))
  cat(sprintf("    CV AUCPR SD    : %.4f\n", res$cv_aucpr_sd))
  cat(sprintf("    n_features     : %d\n",   res$n_features))
  
  ## Full CV learning curve — AUCPR by round
  cv_log <- res$cv_log
  if (!is.null(cv_log) && "test_aucpr_mean" %in% names(cv_log)) {
    
    ## Plot CV learning curve
    p_cv <- ggplot(cv_log, aes(x = iter)) +
      geom_line(aes(y = test_aucpr_mean,  colour = "Validation"),
                linewidth = 0.8) +
      geom_ribbon(aes(ymin = test_aucpr_mean  - test_aucpr_std,
                      ymax = test_aucpr_mean  + test_aucpr_std),
                  fill = "steelblue", alpha = 0.15) +
      geom_line(aes(y = train_aucpr_mean, colour = "Training"),
                linewidth = 0.8, linetype = "dashed") +
      geom_vline(xintercept = res$optimal_rounds,
                 linetype = "dotted", colour = "grey40") +
      annotate("text",
               x     = res$optimal_rounds,
               y     = min(cv_log$test_aucpr_mean, na.rm = TRUE),
               label = paste0("opt=", res$optimal_rounds),
               hjust = -0.1, size = 3, colour = "grey40") +
      scale_colour_manual(values = c("Validation" = "steelblue",
                                     "Training"   = "coral")) +
      labs(
        title    = paste0("CV Learning Curve — ", toupper(nm), " features"),
        subtitle = paste0("Expanding window CV | AUCPR at optimal rounds: ",
                          round(res$cv_aucpr_mean, 4),
                          " (\u00b1", round(res$cv_aucpr_sd, 4), ")"),
        x        = "Boosting Round",
        y        = "AUCPR (Average Precision)",
        colour   = NULL
      ) +
      theme_minimal(base_size = 12) +
      theme(legend.position = "bottom")
    
    fname <- file.path(DIR_FIGURES,
                       paste0("train_cv_curve_", nm, ".png"))
    ggsave(fname, p_cv,
           width = PLOT_WIDTH, height = PLOT_HEIGHT, dpi = PLOT_DPI)
    cat(sprintf("    CV curve saved: %s\n", basename(fname)))
  }
}

#==============================================================================#
# 3. Feature Importance
#==============================================================================#

cat("\n[09B] ── Section 3: Feature Importance ──────────────────────────\n")

for (nm in names(models)) {
  res        <- models[[nm]]
  importance <- as.data.table(res$importance)
  
  if (nrow(importance) == 0L) {
    cat(sprintf("  [%s] No importance data available.\n", nm))
    next
  }
  
  ## Top N by Gain
  top_n <- min(25L, nrow(importance))
  imp_top <- importance[order(-Gain)][seq_len(top_n)]
  imp_top[, Feature := factor(Feature, levels = rev(Feature))]
  
  cat(sprintf("\n  [%s] Top 10 features by Gain:\n", toupper(nm)))
  print(head(importance[order(-Gain)], 10L))
  
  ## Gain plot
  p_imp <- ggplot(imp_top, aes(x = Feature, y = Gain)) +
    geom_col(fill = "steelblue", width = 0.7) +
    coord_flip() +
    scale_y_continuous(labels = scales::percent_format(accuracy = 0.1)) +
    labs(
      title    = paste0("Feature Importance (Gain) — ", toupper(nm)),
      subtitle = paste0("Top ", top_n, " of ", nrow(importance),
                        " features | Gain = fractional contribution to loss reduction"),
      x        = NULL,
      y        = "Gain"
    ) +
    theme_minimal(base_size = 11) +
    theme(axis.text.y = element_text(size = 8))
  
  fname <- file.path(DIR_FIGURES,
                     paste0("train_feature_importance_", nm, ".png"))
  ggsave(fname, p_imp,
         width = PLOT_WIDTH, height = max(PLOT_HEIGHT, top_n * 0.35),
         dpi = PLOT_DPI)
  cat(sprintf("  Importance plot saved: %s\n", basename(fname)))
  
  ## Gain concentration — how much of gain is in top 5 features?
  total_gain <- sum(importance$Gain)
  top5_gain  <- sum(head(importance[order(-Gain)], 5L)$Gain)
  cat(sprintf("  Gain concentration — top 5 features: %.1f%% of total\n",
              100 * top5_gain / total_gain))
}

#==============================================================================#
# 4. In-Sample Calibration (training set only)
#
#   A well-calibrated model outputs P(CSI=1) ≈ actual CSI rate for each
#   probability bin. Miscalibration indicates the model's probabilities
#   cannot be used directly as CSI risk scores without recalibration.
#
#   NOTE: Training set calibration is always optimistic (model saw this data).
#   Test set calibration is assessed in 10_Evaluate.R.
#   This plot checks whether the model learned a monotone probability mapping.
#==============================================================================#

cat("\n[09B] ── Section 4: In-Sample Calibration ────────────────────────\n")

for (nm in names(models)) {
  res         <- models[[nm]]
  preds_train <- res$preds$train
  
  if (is.null(preds_train) || nrow(preds_train) == 0L) next
  
  ## Bin predictions into deciles
  preds_train[, bin := cut(p_csi,
                           breaks = quantile(p_csi,
                                             probs = seq(0, 1, 0.1),
                                             na.rm = TRUE),
                           include.lowest = TRUE,
                           labels = FALSE)]
  
  calib_dt <- preds_train[, .(
    mean_pred   = mean(p_csi),
    mean_actual = mean(y),
    n           = .N
  ), by = bin][order(bin)]
  
  ## Overall calibration gap
  calib_gap <- mean(preds_train$p_csi) - mean(preds_train$y)
  cat(sprintf("  [%s] Calibration gap (mean pred - mean actual): %+.4f\n",
              toupper(nm), calib_gap))
  
  p_calib <- ggplot(calib_dt, aes(x = mean_pred, y = mean_actual)) +
    geom_abline(slope = 1, intercept = 0,
                linetype = "dashed", colour = "grey50") +
    geom_point(aes(size = n), colour = "steelblue", alpha = 0.8) +
    geom_line(colour = "steelblue", linewidth = 0.8) +
    scale_size_continuous(name = "n obs", range = c(2, 8)) +
    scale_x_continuous(labels = scales::percent_format(accuracy = 1),
                       limits = c(0, 1)) +
    scale_y_continuous(labels = scales::percent_format(accuracy = 1),
                       limits = c(0, 1)) +
    labs(
      title    = paste0("In-Sample Calibration — ", toupper(nm)),
      subtitle = paste0("Decile bins | Gap = ",
                        sprintf("%+.4f", calib_gap),
                        " (0 = perfect calibration) | TRAINING SET ONLY"),
      x        = "Mean Predicted P(CSI)",
      y        = "Mean Actual CSI Rate"
    ) +
    theme_minimal(base_size = 12)
  
  fname <- file.path(DIR_FIGURES,
                     paste0("train_calibration_", nm, ".png"))
  ggsave(fname, p_calib,
         width = PLOT_WIDTH * 0.8, height = PLOT_HEIGHT, dpi = PLOT_DPI)
  cat(sprintf("  Calibration plot saved: %s\n", basename(fname)))
}

#==============================================================================#
# 5. In-Sample Precision-Recall Curve (training set only)
#==============================================================================#

cat("\n[09B] ── Section 5: In-Sample PR Curve ───────────────────────────\n")

pr_summary <- list()

for (nm in names(models)) {
  res         <- models[[nm]]
  preds_train <- res$preds$train
  
  if (is.null(preds_train)) next
  
  pr_obj <- tryCatch(
    PRROC::pr.curve(
      scores.class0 = preds_train$p_csi[preds_train$y == 1L],
      scores.class1 = preds_train$p_csi[preds_train$y == 0L],
      curve         = TRUE
    ),
    error = function(e) NULL
  )
  
  if (is.null(pr_obj)) next
  
  pr_dt <- as.data.table(pr_obj$curve)
  setnames(pr_dt, c("recall", "precision", "threshold"))
  
  insample_ap <- pr_obj$auc.integral
  cat(sprintf("  [%s] In-sample Average Precision: %.4f\n",
              toupper(nm), insample_ap))
  
  ## Baseline = prevalence (random classifier)
  prevalence <- mean(preds_train$y)
  
  p_pr <- ggplot(pr_dt, aes(x = recall, y = precision)) +
    geom_hline(yintercept = prevalence,
               linetype = "dashed", colour = "grey60") +
    geom_line(colour = "steelblue", linewidth = 0.9) +
    annotate("text", x = 0.8, y = prevalence + 0.02,
             label = paste0("Baseline = ", round(100 * prevalence, 1), "%"),
             colour = "grey50", size = 3) +
    scale_x_continuous(labels = scales::percent_format(accuracy = 1),
                       limits = c(0, 1)) +
    scale_y_continuous(labels = scales::percent_format(accuracy = 1),
                       limits = c(0, 1)) +
    labs(
      title    = paste0("In-Sample Precision-Recall Curve — ", toupper(nm)),
      subtitle = paste0("Average Precision = ", round(insample_ap, 4),
                        " | TRAINING SET ONLY — optimistic"),
      x        = "Recall",
      y        = "Precision"
    ) +
    theme_minimal(base_size = 12)
  
  fname <- file.path(DIR_FIGURES,
                     paste0("train_pr_curve_", nm, ".png"))
  ggsave(fname, p_pr,
         width = PLOT_WIDTH * 0.8, height = PLOT_HEIGHT, dpi = PLOT_DPI)
  cat(sprintf("  PR curve saved: %s\n", basename(fname)))
  
  pr_summary[[nm]] <- list(insample_ap = insample_ap,
                           prevalence  = prevalence)
}

#==============================================================================#
# 6. HPO Convergence Check
#==============================================================================#

cat("\n[09B] ── Section 6: HPO Convergence ─────────────────────────────\n")

for (nm in names(models)) {
  res     <- models[[nm]]
  bo_hist <- res$bo_history
  
  if (is.null(bo_hist)) next
  
  ## Add iteration order (bo_history is sorted by aucpr desc — restore order)
  bo_plot <- bo_hist |>
    dplyr::arrange(Round) |>
    dplyr::mutate(
      cummax_aucpr = cummax(aucpr),
      phase        = ifelse(Round <= 10L, "Random init", "Bayesian")
    )
  
  p_bo <- ggplot(bo_plot, aes(x = Round)) +
    geom_point(aes(y = aucpr, colour = phase), alpha = 0.7, size = 2) +
    geom_line(aes(y = cummax_aucpr),
              colour = "steelblue", linewidth = 1.0, linetype = "solid") +
    geom_vline(xintercept = 10.5,
               linetype = "dotted", colour = "grey50") +
    annotate("text", x = 5.5, y = max(bo_plot$aucpr),
             label = "Random", colour = "grey50", size = 3) +
    annotate("text", x = 20, y = max(bo_plot$aucpr),
             label = "Bayesian", colour = "grey50", size = 3) +
    scale_colour_manual(values = c("Random init" = "coral",
                                   "Bayesian"    = "steelblue")) +
    labs(
      title    = paste0("HPO Convergence — ", toupper(nm)),
      subtitle = paste0("Line = running best | Best AUCPR = ",
                        round(max(bo_plot$aucpr), 4)),
      x        = "HPO Iteration",
      y        = "CV AUCPR",
      colour   = NULL
    ) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "bottom")
  
  fname <- file.path(DIR_FIGURES,
                     paste0("train_hpo_convergence_", nm, ".png"))
  ggsave(fname, p_bo,
         width = PLOT_WIDTH, height = PLOT_HEIGHT, dpi = PLOT_DPI)
  cat(sprintf("  [%s] HPO plot saved: %s\n", toupper(nm), basename(fname)))
  
  ## Convergence diagnostic
  last_10_range <- diff(range(tail(bo_plot$aucpr, 10L)))
  cat(sprintf("  [%s] AUCPR range in last 10 iterations: %.4f ",
              toupper(nm), last_10_range))
  if (last_10_range < 0.01)
    cat("✓ (converged)\n") else cat("⚠ (may not have converged — consider more iterations)\n")
}

#==============================================================================#
# 7. Combined Training Summary Table
#==============================================================================#

cat("\n[09B] ── Section 7: Combined Summary Table ───────────────────────\n")

## Assemble one row per model × set
summary_rows <- list()

for (nm in names(models)) {
  res <- models[[nm]]
  
  ## CV row
  summary_rows[[length(summary_rows) + 1L]] <- data.frame(
    model          = nm,
    set            = "cv_expanding_window",
    avg_precision  = round(res$cv_aucpr_mean, 4),
    ap_sd          = round(res$cv_aucpr_sd,   4),
    auc_roc        = NA_real_,
    recall_fpr3    = NA_real_,
    recall_fpr5    = NA_real_,
    brier          = NA_real_,
    n_features     = res$n_features,
    optimal_rounds = res$optimal_rounds,
    stringsAsFactors = FALSE
  )
  
  ## Train / test / OOS rows from eval_table
  for (s in c("train_insample", "test", "oos")) {
    row_s <- res$eval_table[res$eval_table$set == s, ]
    if (nrow(row_s) == 0L) next
    summary_rows[[length(summary_rows) + 1L]] <- data.frame(
      model          = nm,
      set            = s,
      avg_precision  = row_s$avg_precision,
      ap_sd          = NA_real_,
      auc_roc        = row_s$auc_roc,
      recall_fpr3    = row_s$recall_fpr3,
      recall_fpr5    = row_s$recall_fpr5,
      brier          = row_s$brier,
      n_features     = res$n_features,
      optimal_rounds = res$optimal_rounds,
      stringsAsFactors = FALSE
    )
  }
}

train_summary <- do.call(rbind, summary_rows)

cat("\n  Full training summary:\n\n")
print(train_summary, row.names = FALSE)

## Save
path_summary <- file.path(DIR_TABLES, "train_summary.rds")
saveRDS(train_summary, path_summary)
cat(sprintf("\n  train_summary.rds saved: %s\n", path_summary))

#==============================================================================#
# 8. Console summary
#==============================================================================#

cat("\n[09B] ══════════════════════════════════════════════════════\n")
cat("  Key training metrics — CV (honest) vs in-sample (optimistic):\n\n")
cat(sprintf("  %-10s | %-20s | %-10s | %-10s | %-10s\n",
            "Model", "Set", "Avg Prec", "AUC-ROC", "R@FPR5"))
cat(sprintf("  %-10s | %-20s | %-10s | %-10s | %-10s\n",
            "----------", "--------------------",
            "----------", "----------", "----------"))

for (nm in names(models)) {
  res <- models[[nm]]
  
  ## CV row
  cat(sprintf("  %-10s | %-20s | %-10.4f | %-10s | %-10s\n",
              nm, "cv_expanding_window",
              res$cv_aucpr_mean, "—", "—"))
  
  ## In-sample row
  tr <- res$eval_table[res$eval_table$set == "train_insample", ]
  if (nrow(tr) > 0L)
    cat(sprintf("  %-10s | %-20s | %-10.4f | %-10.4f | %-10.4f\n",
                nm, "train_insample",
                tr$avg_precision, tr$auc_roc, tr$recall_fpr5))
}

cat("\n  NOTE: Test-set metrics in 10_Evaluate.R — not shown here.\n")
cat(sprintf("\n[09B_Train_Results.R] DONE: %s\n", format(Sys.time())))
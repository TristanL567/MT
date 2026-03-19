#==============================================================================#
#==== 10_Evaluate.R ===========================================================#
#==== Formal Model Evaluation — Test Set & OOS ================================#
#==============================================================================#
#
# PURPOSE:
#   Formal evaluation of trained XGBoost models on held-out data.
#   The test set (2016-2019) is touched here for the first time.
#   Produces all performance tables and plots for thesis Chapter 4.
#
# INPUTS:
#   - config.R
#   - DIR_MODELS/xgb_raw.rds       : trained raw feature model
#   - DIR_MODELS/xgb_latent.rds    : trained latent feature model (if available)
#
# OUTPUTS:
#   - DIR_TABLES/eval_performance.rds     : full performance table
#   - DIR_TABLES/eval_threshold.rds       : threshold calibration table
#   - DIR_TABLES/eval_by_year.rds         : year-level performance breakdown
#   - DIR_FIGURES/eval_roc_curve.png      : ROC curves (raw vs latent)
#   - DIR_FIGURES/eval_pr_curve.png       : PR curves (raw vs latent)
#   - DIR_FIGURES/eval_calibration.png    : calibration plots (test set)
#   - DIR_FIGURES/eval_threshold.png      : recall/precision vs threshold
#   - DIR_FIGURES/eval_by_year.png        : year-level AP and prevalence
#   - DIR_FIGURES/eval_score_dist.png     : predicted score distributions
#
# EVALUATION DESIGN:
#
#   [1] TEST SET IS PRIMARY:
#       Test set (2016-2019) is the main evaluation window.
#       Used for model comparison and threshold selection.
#
#   [2] OOS EVALUATION IS SECONDARY:
#       OOS (2020-2024) provides out-of-time generalisation check.
#       2023 excluded — zero confirmed CSI events due to right-censoring
#       (18-month zombie window extends beyond END_DATE for most 2024 events).
#       Effective OOS window: 2020-2022.
#
#   [3] THRESHOLD CALIBRATION:
#       Rather than absolute probability threshold, use PERCENTILE RANK
#       of p_csi within each evaluation year. This corrects for the known
#       probability overestimation (Brier score finding in 09_Train.R).
#       Report performance at multiple FPR thresholds: 1%, 3%, 5%, 10%.
#
#   [4] PAPER BENCHMARK:
#       Tewari et al. report 61% recall at FPR <= 3%.
#       Direct comparison at this operating point.
#
#   [5] MODEL COMPARISON:
#       Raw features vs latent features across all metrics.
#       Raw model is primary — latent is ablation study.
#
#   [6] YEAR-LEVEL BREAKDOWN:
#       Performance reported separately per year within test and OOS.
#       Identifies whether performance is stable or driven by specific years.
#
#==============================================================================#

source("config.R")

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(ggplot2)
  library(pROC)
  library(PRROC)
  library(scales)
  library(tidyr)
})

cat("\n[10_Evaluate.R] START:", format(Sys.time()), "\n")

## Create output directories if needed
dir.create(DIR_TABLES,  showWarnings = FALSE, recursive = TRUE)
dir.create(DIR_FIGURES, showWarnings = FALSE, recursive = TRUE)

#==============================================================================#
# 0. Load models
#==============================================================================#

cat("[10_Evaluate.R] Loading models...\n")

path_raw    <- file.path(DIR_MODELS, "xgb_raw.rds")
path_latent <- file.path(DIR_MODELS, "xgb_latent.rds")

if (!file.exists(path_raw))
  stop("[10_Evaluate.R] xgb_raw.rds not found — run 09_Train.R first.")

result_raw    <- readRDS(path_raw)
result_latent <- if (file.exists(path_latent)) readRDS(path_latent) else NULL

if (is.null(result_latent))
  cat("  Note: xgb_latent.rds not found — latent model skipped.\n")

models <- list(raw = result_raw)
if (!is.null(result_latent)) models$latent <- result_latent

cat(sprintf("  Models loaded: %s\n", paste(names(models), collapse = ", ")))

#==============================================================================#
# 1. Utility functions
#==============================================================================#

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
  if (length(eligible) == 0L) return(NA_real_)
  max(recall_vals[eligible])
}

fn_precision_at_fpr <- function(y_true, y_pred, fpr_target) {
  roc_obj   <- pROC::roc(y_true, y_pred, quiet = TRUE)
  fpr_vals  <- 1 - roc_obj$specificities
  eligible  <- which(fpr_vals <= fpr_target)
  if (length(eligible) == 0L) return(NA_real_)
  ## Threshold at FPR <= target
  thresh    <- roc_obj$thresholds[eligible[which.max(
    roc_obj$sensitivities[eligible]
  )]]
  pred_pos  <- y_pred >= thresh
  if (sum(pred_pos) == 0L) return(NA_real_)
  mean(y_true[pred_pos] == 1L)
}

fn_eval_full <- function(y_true, y_pred, model_name, set_name) {
  
  ## Guard — need at least one positive and one negative
  if (sum(y_true == 1L) == 0L || sum(y_true == 0L) == 0L) {
    cat(sprintf("  Skipping %s / %s — no valid labels\n", model_name, set_name))
    return(NULL)
  }
  
  auc_roc  <- as.numeric(pROC::auc(pROC::roc(y_true, y_pred, quiet=TRUE)))
  ap       <- fn_avg_precision(y_true, y_pred)
  brier    <- mean((y_pred - y_true)^2)
  prev     <- mean(y_true)
  
  data.frame(
    model         = model_name,
    set           = set_name,
    n_obs         = length(y_true),
    n_pos         = sum(y_true == 1L),
    prevalence    = round(prev,    4),
    auc_roc       = round(auc_roc, 4),
    avg_precision = round(ap,      4),
    recall_fpr1   = round(fn_recall_at_fpr(y_true, y_pred, 0.01), 4),
    recall_fpr3   = round(fn_recall_at_fpr(y_true, y_pred, 0.03), 4),
    recall_fpr5   = round(fn_recall_at_fpr(y_true, y_pred, 0.05), 4),
    recall_fpr10  = round(fn_recall_at_fpr(y_true, y_pred, 0.10), 4),
    brier         = round(brier,   4),
    brier_naive   = round(prev * (1 - prev), 4),  ## Naive baseline
    brier_skill   = round(1 - brier / (prev * (1-prev)), 4),  ## Brier skill score
    stringsAsFactors = FALSE
  )
}

#==============================================================================#
# 2. Core performance evaluation
#
#   Test set: full 2016-2019
#   OOS clean: 2020-2022 only (2023 excluded — right-censoring)
#==============================================================================#

cat("\n[10_Evaluate.R] ── Section 2: Core Performance ──────────────────\n")

eval_rows <- list()

for (nm in names(models)) {
  res <- models[[nm]]
  
  ## Test set — full
  eval_rows[[length(eval_rows)+1]] <- fn_eval_full(
    res$preds$test$y, res$preds$test$p_csi, nm, "test_2016_2019"
  )
  
  ## OOS — 2020-2022 only (exclude 2023 right-censoring)
  oos_clean <- res$preds$oos[res$preds$oos$year <= 2022, ]
  if (nrow(oos_clean) > 0 && sum(oos_clean$y == 1L) > 0) {
    eval_rows[[length(eval_rows)+1]] <- fn_eval_full(
      oos_clean$y, oos_clean$p_csi, nm, "oos_2020_2022"
    )
  }
  
  ## OOS — full (for reference)
  oos_full <- res$preds$oos
  if (sum(oos_full$y == 1L) > 0) {
    eval_rows[[length(eval_rows)+1]] <- fn_eval_full(
      oos_full$y, oos_full$p_csi, nm, "oos_2020_2024_full"
    )
  }
}

eval_performance <- do.call(rbind, Filter(Negate(is.null), eval_rows))

cat("\n  Core performance table:\n\n")
print(eval_performance[, c("model", "set", "n_obs", "prevalence",
                           "auc_roc", "avg_precision",
                           "recall_fpr3", "recall_fpr5", "brier_skill")],
      row.names = FALSE)

## Paper benchmark row for comparison
paper_benchmark <- data.frame(
  model = "paper_tewari", set = "test", n_obs = NA, n_pos = NA,
  prevalence = NA, auc_roc = NA, avg_precision = NA,
  recall_fpr1 = NA, recall_fpr3 = 0.61, recall_fpr5 = NA,
  recall_fpr10 = NA, brier = NA, brier_naive = NA, brier_skill = NA,
  stringsAsFactors = FALSE
)

eval_with_paper <- rbind(eval_performance, paper_benchmark)

cat("\n  vs Paper benchmark (Tewari et al. — R@FPR3 = 61%):\n")
cat(sprintf("  %-15s | %-20s | %8s | %8s | %8s\n",
            "Model", "Set", "AP", "R@FPR3", "R@FPR5"))
cat(sprintf("  %-15s | %-20s | %8s | %8s | %8s\n",
            "---------------", "--------------------",
            "--------", "--------", "--------"))
for (i in seq_len(nrow(eval_with_paper))) {
  r <- eval_with_paper[i, ]
  cat(sprintf("  %-15s | %-20s | %8s | %8s | %8s\n",
              r$model, r$set,
              ifelse(is.na(r$avg_precision), "—",
                     sprintf("%.4f", r$avg_precision)),
              ifelse(is.na(r$recall_fpr3),  "—",
                     sprintf("%.4f", r$recall_fpr3)),
              ifelse(is.na(r$recall_fpr5),  "—",
                     sprintf("%.4f", r$recall_fpr5))))
}

saveRDS(eval_performance, file.path(DIR_TABLES, "eval_performance.rds"))
cat("\n  eval_performance.rds saved.\n")

#==============================================================================#
# 3. Threshold calibration
#
#   Report performance at FPR thresholds: 1%, 3%, 5%, 10%
#   For each threshold: recall, precision, F1, n_flagged
#   Use test set only for threshold selection.
#==============================================================================#

cat("\n[10_Evaluate.R] ── Section 3: Threshold Calibration ─────────────\n")

fpr_targets <- c(0.01, 0.03, 0.05, 0.10)

thresh_rows <- list()

for (nm in names(models)) {
  res      <- models[[nm]]
  y_true   <- res$preds$test$y
  y_pred   <- res$preds$test$p_csi
  roc_obj  <- pROC::roc(y_true, y_pred, quiet = TRUE)
  n_neg    <- sum(y_true == 0L)
  n_pos    <- sum(y_true == 1L)
  
  for (fpr_t in fpr_targets) {
    
    fpr_vals    <- 1 - roc_obj$specificities
    recall_vals <- roc_obj$sensitivities
    thresh_vals <- roc_obj$thresholds
    
    eligible <- which(fpr_vals <= fpr_t)
    if (length(eligible) == 0L) next
    
    best_idx   <- eligible[which.max(recall_vals[eligible])]
    best_thresh <- thresh_vals[best_idx]
    recall      <- recall_vals[best_idx]
    fpr_actual  <- fpr_vals[best_idx]
    
    ## At this threshold
    pred_pos  <- as.integer(y_pred >= best_thresh)
    tp        <- sum(pred_pos == 1L & y_true == 1L)
    fp        <- sum(pred_pos == 1L & y_true == 0L)
    tn        <- sum(pred_pos == 0L & y_true == 0L)
    fn        <- sum(pred_pos == 0L & y_true == 1L)
    precision <- if ((tp + fp) > 0) tp / (tp + fp) else NA_real_
    f1        <- if (!is.na(precision) && (precision + recall) > 0)
      2 * precision * recall / (precision + recall) else NA_real_
    n_flagged <- tp + fp
    
    thresh_rows[[length(thresh_rows)+1]] <- data.frame(
      model        = nm,
      fpr_target   = fpr_t,
      fpr_actual   = round(fpr_actual,  4),
      threshold    = round(best_thresh, 4),
      recall       = round(recall,      4),
      precision    = round(precision,   4),
      f1           = round(f1,          4),
      n_flagged    = n_flagged,
      n_correct    = tp,
      n_false_alarm = fp,
      stringsAsFactors = FALSE
    )
  }
}

eval_threshold <- do.call(rbind, thresh_rows)

cat("\n  Threshold calibration (test set 2016-2019):\n\n")
print(eval_threshold, row.names = FALSE)

saveRDS(eval_threshold, file.path(DIR_TABLES, "eval_threshold.rds"))
cat("\n  eval_threshold.rds saved.\n")

#==============================================================================#
# 4. Year-level performance breakdown
#
#   Performance reported per year within test and OOS.
#   Identifies COVID-era vs benign macro performance.
#==============================================================================#

cat("\n[10_Evaluate.R] ── Section 4: Year-Level Breakdown ──────────────\n")

year_rows <- list()

for (nm in names(models)) {
  res <- models[[nm]]
  
  for (split_nm in c("test", "oos")) {
    preds_dt <- as.data.table(res$preds[[split_nm]])
    
    for (yr in sort(unique(preds_dt$year))) {
      yr_dt  <- preds_dt[year == yr]
      n_pos  <- sum(yr_dt$y == 1L, na.rm = TRUE)
      n_obs  <- nrow(yr_dt)
      
      ## Skip years with no positive labels
      if (n_pos == 0L || n_obs - n_pos == 0L) {
        year_rows[[length(year_rows)+1]] <- data.frame(
          model      = nm, split = split_nm, year = yr,
          n_obs      = n_obs, n_pos = n_pos,
          prevalence = round(n_pos / n_obs, 4),
          auc_roc    = NA_real_, avg_precision = NA_real_,
          recall_fpr3 = NA_real_, recall_fpr5 = NA_real_,
          note       = "insufficient_labels",
          stringsAsFactors = FALSE
        )
        next
      }
      
      auc_roc <- tryCatch(
        as.numeric(pROC::auc(pROC::roc(yr_dt$y, yr_dt$p_csi, quiet=TRUE))),
        error = function(e) NA_real_
      )
      ap <- fn_avg_precision(yr_dt$y, yr_dt$p_csi)
      r3 <- fn_recall_at_fpr(yr_dt$y, yr_dt$p_csi, 0.03)
      r5 <- fn_recall_at_fpr(yr_dt$y, yr_dt$p_csi, 0.05)
      
      year_rows[[length(year_rows)+1]] <- data.frame(
        model         = nm,
        split         = split_nm,
        year          = yr,
        n_obs         = n_obs,
        n_pos         = n_pos,
        prevalence    = round(n_pos / n_obs, 4),
        auc_roc       = round(auc_roc, 4),
        avg_precision = round(ap,      4),
        recall_fpr3   = round(r3,      4),
        recall_fpr5   = round(r5,      4),
        note          = "ok",
        stringsAsFactors = FALSE
      )
    }
  }
}

eval_by_year <- do.call(rbind, year_rows)

cat("\n  Year-level breakdown (raw model):\n\n")
raw_by_year <- eval_by_year[eval_by_year$model == "raw", ]
print(raw_by_year[, c("split", "year", "n_obs", "n_pos", "prevalence",
                      "avg_precision", "recall_fpr3", "recall_fpr5")],
      row.names = FALSE)

saveRDS(eval_by_year, file.path(DIR_TABLES, "eval_by_year.rds"))
cat("\n  eval_by_year.rds saved.\n")

#==============================================================================#
# 5. Plots
#==============================================================================#

cat("\n[10_Evaluate.R] ── Section 5: Plots ─────────────────────────────\n")

## Colour palette
model_colours <- c(
  "raw"    = "steelblue",
  "latent" = "coral",
  "paper"  = "grey50"
)

##──────────────────────────────────────────────────────────────────────────────
## 5A. ROC Curves — test set
##──────────────────────────────────────────────────────────────────────────────

roc_data <- list()
for (nm in names(models)) {
  res     <- models[[nm]]
  roc_obj <- pROC::roc(res$preds$test$y, res$preds$test$p_csi, quiet=TRUE)
  roc_data[[nm]] <- data.frame(
    model       = nm,
    fpr         = 1 - roc_obj$specificities,
    tpr         = roc_obj$sensitivities,
    auc         = as.numeric(pROC::auc(roc_obj))
  )
}

roc_df <- do.call(rbind, roc_data)
roc_df$model_label <- sprintf("%s (AUC=%.3f)",
                              roc_df$model,
                              roc_df$auc)

p_roc <- ggplot(roc_df, aes(x = fpr, y = tpr,
                            colour = model,
                            group  = model)) +
  geom_line(linewidth = 1.0) +
  geom_abline(slope = 1, intercept = 0,
              linetype = "dashed", colour = "grey70") +
  geom_vline(xintercept = c(0.03, 0.05),
             linetype = "dotted", colour = "grey50", linewidth = 0.5) +
  annotate("text", x = 0.03, y = 0.05,
           label = "FPR=3%", hjust = -0.1, size = 3, colour = "grey50") +
  annotate("text", x = 0.05, y = 0.05,
           label = "FPR=5%", hjust = -0.1, size = 3, colour = "grey50") +
  scale_colour_manual(
    values = model_colours,
    labels = setNames(
      sprintf("%s (AUC=%.3f)",
              names(models),
              sapply(names(models), function(nm)
                unique(roc_df$auc[roc_df$model == nm]))),
      names(models)
    )
  ) +
  scale_x_continuous(labels = scales::percent_format(accuracy=1),
                     limits = c(0, 1)) +
  scale_y_continuous(labels = scales::percent_format(accuracy=1),
                     limits = c(0, 1)) +
  labs(
    title    = "ROC Curves — Test Set (2016–2019)",
    subtitle = "Dotted lines mark FPR = 3% and 5% operating thresholds",
    x        = "False Positive Rate (1 - Specificity)",
    y        = "True Positive Rate (Recall)",
    colour   = "Model"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

ggsave(file.path(DIR_FIGURES, "eval_roc_curve.png"),
       p_roc, width = PLOT_WIDTH, height = PLOT_HEIGHT, dpi = PLOT_DPI)
cat("  eval_roc_curve.png saved.\n")

##──────────────────────────────────────────────────────────────────────────────
## 5B. Precision-Recall Curves — test set
##──────────────────────────────────────────────────────────────────────────────

pr_data <- list()
for (nm in names(models)) {
  res    <- models[[nm]]
  pr_obj <- PRROC::pr.curve(
    scores.class0 = res$preds$test$p_csi[res$preds$test$y == 1L],
    scores.class1 = res$preds$test$p_csi[res$preds$test$y == 0L],
    curve         = TRUE
  )
  pr_curve_dt <- as.data.frame(pr_obj$curve)
  names(pr_curve_dt) <- c("recall", "precision", "threshold")
  pr_curve_dt$model <- nm
  pr_curve_dt$ap    <- pr_obj$auc.integral
  pr_data[[nm]] <- pr_curve_dt
}

pr_df       <- do.call(rbind, pr_data)
prevalence  <- mean(result_raw$preds$test$y)

p_pr <- ggplot(pr_df, aes(x = recall, y = precision,
                          colour = model, group = model)) +
  geom_line(linewidth = 1.0) +
  geom_hline(yintercept = prevalence,
             linetype = "dashed", colour = "grey60", linewidth = 0.8) +
  annotate("text", x = 0.85, y = prevalence + 0.02,
           label = sprintf("Baseline = %.1f%%", 100*prevalence),
           colour = "grey50", size = 3) +
  scale_colour_manual(
    values = model_colours,
    labels = setNames(
      sprintf("%s (AP=%.3f)",
              names(models),
              sapply(names(models), function(nm)
                unique(pr_df$ap[pr_df$model == nm]))),
      names(models)
    )
  ) +
  scale_x_continuous(labels = scales::percent_format(accuracy=1),
                     limits = c(0, 1)) +
  scale_y_continuous(labels = scales::percent_format(accuracy=1),
                     limits = c(0, 1)) +
  labs(
    title    = "Precision-Recall Curves — Test Set (2016–2019)",
    subtitle = sprintf("Dashed line = baseline prevalence (%.1f%%)",
                       100 * prevalence),
    x        = "Recall",
    y        = "Precision",
    colour   = "Model"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

ggsave(file.path(DIR_FIGURES, "eval_pr_curve.png"),
       p_pr, width = PLOT_WIDTH, height = PLOT_HEIGHT, dpi = PLOT_DPI)
cat("  eval_pr_curve.png saved.\n")

##──────────────────────────────────────────────────────────────────────────────
## 5C. Calibration plot — test set
##     Predicted probability deciles vs actual CSI rate
##──────────────────────────────────────────────────────────────────────────────

calib_data <- list()
for (nm in names(models)) {
  res    <- models[[nm]]
  dt     <- as.data.table(res$preds$test)
  dt[, bin := cut(p_csi,
                  breaks = quantile(p_csi,
                                    probs = seq(0, 1, 0.1),
                                    na.rm = TRUE),
                  include.lowest = TRUE,
                  labels = FALSE)]
  calib_dt <- dt[, .(
    mean_pred   = mean(p_csi),
    mean_actual = mean(y),
    n           = .N
  ), by = bin][order(bin)]
  calib_dt[, model := nm]
  calib_data[[nm]] <- calib_dt
}

calib_df <- do.call(rbind, calib_data)

p_calib <- ggplot(calib_df, aes(x = mean_pred, y = mean_actual,
                                colour = model, size = n)) +
  geom_abline(slope = 1, intercept = 0,
              linetype = "dashed", colour = "grey50") +
  geom_point(alpha = 0.8) +
  geom_line(aes(group = model), linewidth = 0.7, alpha = 0.6) +
  scale_colour_manual(values = model_colours) +
  scale_size_continuous(name = "n obs", range = c(2, 8)) +
  scale_x_continuous(labels = scales::percent_format(accuracy=1),
                     limits = c(0, 1)) +
  scale_y_continuous(labels = scales::percent_format(accuracy=1),
                     limits = c(0, 1)) +
  labs(
    title    = "Calibration Plot — Test Set (2016–2019)",
    subtitle = "Decile bins | Diagonal = perfect calibration",
    x        = "Mean Predicted P(CSI)",
    y        = "Actual CSI Rate",
    colour   = "Model"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

ggsave(file.path(DIR_FIGURES, "eval_calibration.png"),
       p_calib, width = PLOT_WIDTH * 0.9, height = PLOT_HEIGHT, dpi = PLOT_DPI)
cat("  eval_calibration.png saved.\n")

##──────────────────────────────────────────────────────────────────────────────
## 5D. Recall and Precision vs FPR threshold — raw model, test set
##──────────────────────────────────────────────────────────────────────────────

roc_raw  <- pROC::roc(result_raw$preds$test$y,
                      result_raw$preds$test$p_csi, quiet=TRUE)
thresh_df <- data.frame(
  fpr       = 1 - roc_raw$specificities,
  recall    = roc_raw$sensitivities,
  threshold = roc_raw$thresholds
)

## Compute precision at each threshold
thresh_df$precision <- sapply(thresh_df$threshold, function(t) {
  pred_pos <- as.integer(result_raw$preds$test$p_csi >= t)
  tp <- sum(pred_pos == 1L & result_raw$preds$test$y == 1L)
  fp <- sum(pred_pos == 1L & result_raw$preds$test$y == 0L)
  if ((tp + fp) == 0L) return(NA_real_)
  tp / (tp + fp)
})

thresh_long <- thresh_df |>
  filter(fpr <= 0.15) |>
  select(fpr, recall, precision) |>
  pivot_longer(cols = c(recall, precision),
               names_to = "metric", values_to = "value")

p_thresh <- ggplot(thresh_long,
                   aes(x = fpr, y = value,
                       colour = metric, linetype = metric)) +
  geom_line(linewidth = 1.0) +
  geom_vline(xintercept = c(0.03, 0.05),
             linetype = "dotted", colour = "grey50") +
  annotate("text", x = 0.03, y = 0.95,
           label = "3%", hjust = -0.2, size = 3.5, colour = "grey40") +
  annotate("text", x = 0.05, y = 0.95,
           label = "5%", hjust = -0.2, size = 3.5, colour = "grey40") +
  scale_colour_manual(values = c("recall" = "steelblue",
                                 "precision" = "coral")) +
  scale_x_continuous(labels = scales::percent_format(accuracy=1),
                     limits = c(0, 0.15)) +
  scale_y_continuous(labels = scales::percent_format(accuracy=1),
                     limits = c(0, 1)) +
  labs(
    title    = "Recall & Precision vs FPR Threshold — Raw Model, Test Set",
    subtitle = "Operating range 0–15% FPR | Dotted lines at 3% and 5%",
    x        = "False Positive Rate",
    y        = "Metric Value",
    colour   = NULL, linetype = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

ggsave(file.path(DIR_FIGURES, "eval_threshold.png"),
       p_thresh, width = PLOT_WIDTH, height = PLOT_HEIGHT, dpi = PLOT_DPI)
cat("  eval_threshold.png saved.\n")

##──────────────────────────────────────────────────────────────────────────────
## 5E. Year-level Average Precision — raw model
##──────────────────────────────────────────────────────────────────────────────

raw_by_year_plot <- eval_by_year[
  eval_by_year$model == "raw" &
    eval_by_year$note  == "ok", ]

raw_by_year_plot$period <- ifelse(
  raw_by_year_plot$split == "test", "Test (2016–2019)",
  ifelse(raw_by_year_plot$year <= 2022, "OOS (2020–2022)", "OOS — censored")
)

p_by_year <- ggplot(raw_by_year_plot,
                    aes(x = year, y = avg_precision, fill = period)) +
  geom_col(width = 0.7) +
  geom_line(aes(y = prevalence * 3, group = 1),
            colour = "grey30", linewidth = 0.8, linetype = "dashed") +
  scale_fill_manual(values = c(
    "Test (2016–2019)" = "steelblue",
    "OOS (2020–2022)"  = "coral",
    "OOS — censored"   = "grey80"
  )) +
  scale_y_continuous(
    name     = "Average Precision",
    sec.axis = sec_axis(~ . / 3,
                        name   = "CSI Prevalence",
                        labels = scales::percent_format(accuracy=1))
  ) +
  labs(
    title    = "Year-Level Average Precision — Raw Model",
    subtitle = "Dashed line = CSI prevalence (right axis, scaled)",
    x        = "Year",
    fill     = "Period"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

ggsave(file.path(DIR_FIGURES, "eval_by_year.png"),
       p_by_year, width = PLOT_WIDTH, height = PLOT_HEIGHT, dpi = PLOT_DPI)
cat("  eval_by_year.png saved.\n")

##──────────────────────────────────────────────────────────────────────────────
## 5F. Score distribution — CSI vs non-CSI firms, test set
##──────────────────────────────────────────────────────────────────────────────

score_dt <- as.data.table(result_raw$preds$test)
score_dt[, label := fifelse(y == 1L, "CSI (y=1)", "Clean (y=0)")]

p_score <- ggplot(score_dt, aes(x = p_csi, fill = label)) +
  geom_density(alpha = 0.5) +
  geom_vline(xintercept = c(
    eval_threshold[eval_threshold$model == "raw" &
                     eval_threshold$fpr_target == 0.03, "threshold"],
    eval_threshold[eval_threshold$model == "raw" &
                     eval_threshold$fpr_target == 0.05, "threshold"]
  ), linetype = "dashed", colour = c("steelblue", "coral"),
  linewidth = 0.8) +
  scale_fill_manual(values = c("CSI (y=1)" = "coral",
                               "Clean (y=0)" = "steelblue")) +
  scale_x_continuous(labels = scales::percent_format(accuracy=1)) +
  labs(
    title    = "Predicted P(CSI) Distribution — Test Set (2016–2019)",
    subtitle = "Dashed lines = thresholds at FPR=3% (blue) and FPR=5% (red)",
    x        = "Predicted P(CSI)",
    y        = "Density",
    fill     = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

ggsave(file.path(DIR_FIGURES, "eval_score_dist.png"),
       p_score, width = PLOT_WIDTH, height = PLOT_HEIGHT, dpi = PLOT_DPI)
cat("  eval_score_dist.png saved.\n")

#==============================================================================#
# 6. Brier skill score summary
#
#   Brier skill score = 1 - Brier / Brier_naive
#   Positive = better than naive baseline (always predict prevalence)
#   1.0 = perfect, 0.0 = no skill, negative = worse than naive
#==============================================================================#

cat("\n[10_Evaluate.R] ── Section 6: Brier Skill Scores ─────────────────\n")

cat("\n  Brier skill score (1 = perfect, 0 = no skill vs naive):\n\n")
brier_cols <- c("model", "set", "prevalence", "brier",
                "brier_naive", "brier_skill")
print(eval_performance[, brier_cols], row.names = FALSE)

#==============================================================================#
# 7. Final summary — thesis-ready table
#==============================================================================#

cat("\n[10_Evaluate.R] ── Section 7: Thesis Summary Table ──────────────\n")

cat("\n  ══════════════════════════════════════════════════════\n")
cat("  THESIS RESULTS TABLE\n")
cat("  ══════════════════════════════════════════════════════\n\n")

cat(sprintf("  %-18s | %-20s | %7s | %7s | %7s | %7s | %7s\n",
            "Model", "Set", "AP", "AUC", "R@FPR3", "R@FPR5", "Brier"))
cat(sprintf("  %-18s | %-20s | %7s | %7s | %7s | %7s | %7s\n",
            "------------------", "--------------------",
            "-------", "-------", "-------", "-------", "-------"))

for (i in seq_len(nrow(eval_performance))) {
  r <- eval_performance[i, ]
  cat(sprintf("  %-18s | %-20s | %7.4f | %7.4f | %7.4f | %7.4f | %7.4f\n",
              r$model, r$set,
              r$avg_precision, r$auc_roc,
              r$recall_fpr3, r$recall_fpr5,
              r$brier))
}

## Paper benchmark
cat(sprintf("  %-18s | %-20s | %7s | %7s | %7s | %7s | %7s\n",
            "paper_tewari", "test (unspecified)",
            "—", "—", "0.6100", "—", "—"))

cat("\n  ══════════════════════════════════════════════════════\n")

#==============================================================================#
# 8. Assertions
#==============================================================================#

cat("\n[10_Evaluate.R] Running assertions...\n")

## A) All expected sets present for raw model
expected_sets <- c("test_2016_2019", "oos_2020_2022")
raw_sets      <- eval_performance[eval_performance$model == "raw", "set"]
missing_sets  <- setdiff(expected_sets, raw_sets)
if (length(missing_sets) > 0)
  warning(sprintf("Missing evaluation sets: %s",
                  paste(missing_sets, collapse = ", ")))

## B) AUC-ROC plausible
stopifnot(all(eval_performance$auc_roc > 0.5, na.rm = TRUE))

## C) Recall monotone in FPR — R@FPR5 >= R@FPR3
raw_test <- eval_performance[
  eval_performance$model == "raw" &
    eval_performance$set   == "test_2016_2019", ]
if (nrow(raw_test) > 0)
  stopifnot(raw_test$recall_fpr5 >= raw_test$recall_fpr3)

## D) Threshold table has all FPR targets for raw model
raw_thresh_fprs <- eval_threshold[eval_threshold$model == "raw",
                                  "fpr_target"]
stopifnot(all(fpr_targets %in% raw_thresh_fprs))

cat("[10_Evaluate.R] All assertions passed.\n")

cat(sprintf("\n[10_Evaluate.R] DONE: %s\n", format(Sys.time())))

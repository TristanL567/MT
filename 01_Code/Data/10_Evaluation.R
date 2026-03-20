#==============================================================================#
#==== 10_Evaluate.R ===========================================================#
#==== Formal Model Evaluation — AutoGluon M1 / M3 / M4 + XGBoost Reference ===#
#==============================================================================#
#
# PURPOSE:
#   Formal evaluation of AutoGluon models on held-out data.
#   Primary models: M1 (fund), M3 (raw), M4 (latent_raw)
#   Reference:      M3 XGBoost (hand-tuned Bayesian HPO)
#
#   Predictions were generated in 09C_AutoGluon.py and saved as parquet files.
#   This script loads those predictions and produces all evaluation tables
#   and plots for thesis Chapter 4.
#
# INPUTS:
#   - config.R
#   - DIR_TABLES/ag_fund/ag_preds_test.parquet        [M1 test predictions]
#   - DIR_TABLES/ag_fund/ag_preds_oos.parquet         [M1 OOS predictions]
#   - DIR_TABLES/ag_raw/ag_preds_test.parquet         [M3 AG test]
#   - DIR_TABLES/ag_raw/ag_preds_oos.parquet          [M3 AG OOS]
#   - DIR_TABLES/ag_latent_raw/ag_preds_test.parquet  [M4 test]
#   - DIR_TABLES/ag_latent_raw/ag_preds_oos.parquet   [M4 OOS]
#   - DIR_MODELS/xgb_raw.rds                          [M3 XGB reference]
#
# OUTPUTS:
#   - DIR_TABLES/eval_performance.rds      full performance table
#   - DIR_TABLES/eval_threshold.rds        threshold calibration (M1 + M3)
#   - DIR_TABLES/eval_by_year.rds          year-level breakdown
#   - DIR_FIGURES/eval_roc_curve.png       M1 vs M3 ROC curves
#   - DIR_FIGURES/eval_pr_curve.png        M1 vs M3 PR curves
#   - DIR_FIGURES/eval_calibration.png     M1 vs M3 calibration
#   - DIR_FIGURES/eval_score_dist.png      M1 vs M3 score distributions
#   - DIR_FIGURES/eval_threshold.png       Recall/Precision vs FPR (M1 + M3)
#   - DIR_FIGURES/eval_by_year.png         Year-level AP (M1 vs M3)
#   - DIR_FIGURES/eval_m1_vs_m3_vs_m4.png M1 / M3 / M4 bar comparison
#
# EVALUATION DESIGN:
#   [1] PRIMARY METRIC: Average Precision (AP/AUCPR) — threshold-free
#   [2] ECONOMIC METRIC: Recall at FPR <= 3% — paper benchmark comparison
#   [3] TEST SET (2016–2019): model selection and threshold calibration
#   [4] OOS (2020–2022): out-of-time generalisation (2023 right-censored)
#   [5] PAPER BENCHMARK: Tewari et al. R@FPR3 = 0.61
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
  library(arrow)
})

cat("\n[10_Evaluate.R] START:", format(Sys.time()), "\n")

dir.create(DIR_TABLES,  showWarnings = FALSE, recursive = TRUE)
dir.create(DIR_FIGURES, showWarnings = FALSE, recursive = TRUE)

#==============================================================================#
# 0. Model registry — paths and labels
#==============================================================================#

## Primary AutoGluon models
AG_MODELS <- list(
  M1 = list(
    label     = "M1 AG Fund",
    colour    = "#2196F3",
    test_path = file.path(DIR_TABLES, "ag_fund", "ag_preds_test.parquet"),
    oos_path  = file.path(DIR_TABLES, "ag_fund", "ag_preds_oos.parquet")
  ),
  M3 = list(
    label     = "M3 AG Raw",
    colour    = "#F44336",
    test_path = file.path(DIR_TABLES, "ag_preds_test.parquet"),  # ← root
    oos_path  = file.path(DIR_TABLES, "ag_preds_oos.parquet")    # ← root
  ),
  M4 = list(
    label     = "M4 AG Latent Raw",
    colour    = "#FF9800",
    test_path = file.path(DIR_TABLES, "ag_latent_raw", "ag_preds_test.parquet"),
    oos_path  = file.path(DIR_TABLES, "ag_latent_raw", "ag_preds_oos.parquet")
  )
)

## XGBoost reference
XGB_RAW_PATH  <- file.path(DIR_MODELS, "xgb_raw.rds")
XGB_FUND_PATH <- file.path(DIR_MODELS, "xgb_fund.rds")

## Colour palette (shared across all plots)
ALL_COLOURS <- c(
  "M1 AG Fund"       = "#2196F3",
  "M3 AG Raw"        = "#F44336",
  "M4 AG Latent Raw" = "#FF9800",
  "M3 XGB Raw"       = "#4CAF50"
)

#==============================================================================#
# 1. Load predictions
#==============================================================================#

cat("[10_Evaluate.R] Loading predictions...\n")

preds <- list()

for (nm in names(AG_MODELS)) {
  cfg <- AG_MODELS[[nm]]
  
  if (!file.exists(cfg$test_path)) {
    cat(sprintf("  [%s] Test predictions not found: %s\n", nm, cfg$test_path))
    next
  }
  
  preds[[nm]] <- list(
    label = cfg$label,
    test  = as.data.table(arrow::read_parquet(cfg$test_path)),
    oos   = if (file.exists(cfg$oos_path))
      as.data.table(arrow::read_parquet(cfg$oos_path))
    else NULL
  )
  cat(sprintf("  [%s] %s: test=%d | oos=%d\n",
              nm, cfg$label,
              nrow(preds[[nm]]$test),
              if (!is.null(preds[[nm]]$oos)) nrow(preds[[nm]]$oos) else 0))
}

## XGBoost reference
if (file.exists(XGB_RAW_PATH)) {
  xgb_raw <- readRDS(XGB_RAW_PATH)
  preds[["XGB_M3"]] <- list(
    label = "M3 XGB Raw",
    test  = as.data.table(xgb_raw$preds$test),
    oos   = as.data.table(xgb_raw$preds$oos)
  )
  cat(sprintf("  [XGB_M3] XGBoost raw: test=%d | oos=%d\n",
              nrow(preds$XGB_M3$test), nrow(preds$XGB_M3$oos)))
}

cat(sprintf("  Models loaded: %s\n",
            paste(names(preds), collapse = ", ")))

#==============================================================================#
# 2. Utility functions
#==============================================================================#

fn_avg_precision <- function(y_true, y_pred) {
  tryCatch({
    PRROC::pr.curve(
      scores.class0 = y_pred[y_true == 1L],
      scores.class1 = y_pred[y_true == 0L],
      curve         = FALSE
    )$auc.integral
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

fn_eval_full <- function(y_true, y_pred, model_label, set_name) {
  if (sum(y_true == 1L) == 0L || sum(y_true == 0L) == 0L) return(NULL)
  data.frame(
    model         = model_label,
    set           = set_name,
    n_obs         = length(y_true),
    n_pos         = sum(y_true == 1L),
    prevalence    = round(mean(y_true), 4),
    auc_roc       = round(as.numeric(pROC::auc(
      pROC::roc(y_true, y_pred, quiet=TRUE))), 4),
    avg_precision = round(fn_avg_precision(y_true, y_pred), 4),
    recall_fpr1   = round(fn_recall_at_fpr(y_true, y_pred, 0.01), 4),
    recall_fpr3   = round(fn_recall_at_fpr(y_true, y_pred, 0.03), 4),
    recall_fpr5   = round(fn_recall_at_fpr(y_true, y_pred, 0.05), 4),
    recall_fpr10  = round(fn_recall_at_fpr(y_true, y_pred, 0.10), 4),
    brier         = round(mean((y_pred - y_true)^2), 4),
    stringsAsFactors = FALSE
  )
}

#==============================================================================#
# 3. Core performance evaluation
#==============================================================================#

cat("\n[10_Evaluate.R] ── Section 3: Core Performance ──────────────────\n")

eval_rows <- list()

for (nm in names(preds)) {
  p   <- preds[[nm]]
  lbl <- p$label
  
  ## Test set — full 2016–2019
  eval_rows[[length(eval_rows)+1]] <- fn_eval_full(
    p$test$y, p$test$p_csi, lbl, "test_2016_2019"
  )
  
  if (!is.null(p$oos)) {
    ## OOS 2020–2022 (exclude 2023 — right censoring)
    oos_clean <- p$oos[p$oos$year <= 2022, ]
    if (nrow(oos_clean) > 0 && sum(oos_clean$y == 1L) > 0)
      eval_rows[[length(eval_rows)+1]] <- fn_eval_full(
        oos_clean$y, oos_clean$p_csi, lbl, "oos_2020_2022"
      )
  }
}

eval_performance <- do.call(rbind, Filter(Negate(is.null), eval_rows))

cat("\n  Core performance table:\n\n")
print(eval_performance[, c("model", "set", "n_obs", "prevalence",
                           "auc_roc", "avg_precision",
                           "recall_fpr3", "recall_fpr5", "brier")],
      row.names = FALSE)

## Paper benchmark
paper_bm <- data.frame(
  model="Paper (Tewari)", set="test", n_obs=NA, n_pos=NA,
  prevalence=NA, auc_roc=NA, avg_precision=NA,
  recall_fpr1=NA, recall_fpr3=0.61, recall_fpr5=NA,
  recall_fpr10=NA, brier=NA, stringsAsFactors=FALSE
)

cat("\n  vs Paper benchmark (R@FPR3 = 0.61):\n")
cat(sprintf("  %-22s | %-20s | %7s | %7s | %7s\n",
            "Model", "Set", "AP", "R@FPR3", "R@FPR5"))
cat(sprintf("  %-22s | %-20s | %7s | %7s | %7s\n",
            "----------------------", "--------------------",
            "-------", "-------", "-------"))
for (i in seq_len(nrow(eval_performance))) {
  r <- eval_performance[i,]
  cat(sprintf("  %-22s | %-20s | %7.4f | %7.4f | %7.4f\n",
              r$model, r$set, r$avg_precision, r$recall_fpr3, r$recall_fpr5))
}
cat(sprintf("  %-22s | %-20s | %7s | %7s | %7s\n",
            "Paper (Tewari)", "test", "—", "0.6100", "—"))

saveRDS(eval_performance, file.path(DIR_TABLES, "eval_performance.rds"))
cat("\n  eval_performance.rds saved.\n")

#==============================================================================#
# 4. Threshold calibration — M1 and M3 AG
#==============================================================================#

cat("\n[10_Evaluate.R] ── Section 4: Threshold Calibration ─────────────\n")

fpr_targets <- c(0.01, 0.03, 0.05, 0.10)
thresh_rows <- list()

for (nm in c("M1", "M3")) {
  if (is.null(preds[[nm]])) next
  y_true  <- preds[[nm]]$test$y
  y_pred  <- preds[[nm]]$test$p_csi
  roc_obj <- pROC::roc(y_true, y_pred, quiet = TRUE)
  lbl     <- preds[[nm]]$label
  
  for (fpr_t in fpr_targets) {
    fpr_vals    <- 1 - roc_obj$specificities
    recall_vals <- roc_obj$sensitivities
    thresh_vals <- roc_obj$thresholds
    eligible    <- which(fpr_vals <= fpr_t)
    if (length(eligible) == 0L) next
    
    best_idx    <- eligible[which.max(recall_vals[eligible])]
    best_thresh <- thresh_vals[best_idx]
    recall      <- recall_vals[best_idx]
    fpr_actual  <- fpr_vals[best_idx]
    
    pred_pos  <- as.integer(y_pred >= best_thresh)
    tp        <- sum(pred_pos == 1L & y_true == 1L)
    fp        <- sum(pred_pos == 1L & y_true == 0L)
    precision <- if ((tp+fp) > 0) tp/(tp+fp) else NA_real_
    f1        <- if (!is.na(precision) && (precision+recall) > 0)
      2*precision*recall/(precision+recall) else NA_real_
    
    thresh_rows[[length(thresh_rows)+1]] <- data.frame(
      model        = lbl,
      fpr_target   = fpr_t,
      fpr_actual   = round(fpr_actual,  4),
      threshold    = round(best_thresh, 4),
      recall       = round(recall,      4),
      precision    = round(precision,   4),
      f1           = round(f1,          4),
      n_flagged    = tp + fp,
      n_correct    = tp,
      n_false_alarm = fp,
      stringsAsFactors = FALSE
    )
  }
}

eval_threshold <- do.call(rbind, thresh_rows)
cat("\n  Threshold calibration (test set 2016–2019):\n\n")
print(eval_threshold, row.names = FALSE)
saveRDS(eval_threshold, file.path(DIR_TABLES, "eval_threshold.rds"))
cat("\n  eval_threshold.rds saved.\n")

#==============================================================================#
# 5. Year-level breakdown — M1 and M3 AG
#==============================================================================#

cat("\n[10_Evaluate.R] ── Section 5: Year-Level Breakdown ──────────────\n")

year_rows <- list()

for (nm in c("M1", "M3")) {
  if (is.null(preds[[nm]])) next
  p   <- preds[[nm]]
  lbl <- p$label
  
  for (split_nm in c("test", "oos")) {
    preds_dt <- if (split_nm == "test") p$test else p$oos
    if (is.null(preds_dt)) next
    preds_dt <- as.data.table(preds_dt)
    
    for (yr in sort(unique(preds_dt$year))) {
      yr_dt <- preds_dt[year == yr]
      n_pos <- sum(yr_dt$y == 1L, na.rm=TRUE)
      n_obs <- nrow(yr_dt)
      
      if (n_pos == 0L || n_obs - n_pos == 0L) {
        year_rows[[length(year_rows)+1]] <- data.frame(
          model=lbl, split=split_nm, year=yr, n_obs=n_obs, n_pos=n_pos,
          prevalence=round(n_pos/n_obs,4),
          auc_roc=NA_real_, avg_precision=NA_real_,
          recall_fpr3=NA_real_, recall_fpr5=NA_real_,
          note="insufficient_labels", stringsAsFactors=FALSE)
        next
      }
      
      year_rows[[length(year_rows)+1]] <- data.frame(
        model         = lbl,
        split         = split_nm,
        year          = yr,
        n_obs         = n_obs,
        n_pos         = n_pos,
        prevalence    = round(n_pos/n_obs, 4),
        auc_roc       = round(tryCatch(
          as.numeric(pROC::auc(pROC::roc(yr_dt$y, yr_dt$p_csi, quiet=TRUE))),
          error=function(e) NA_real_), 4),
        avg_precision = round(fn_avg_precision(yr_dt$y, yr_dt$p_csi), 4),
        recall_fpr3   = round(fn_recall_at_fpr(yr_dt$y, yr_dt$p_csi, 0.03), 4),
        recall_fpr5   = round(fn_recall_at_fpr(yr_dt$y, yr_dt$p_csi, 0.05), 4),
        note          = "ok",
        stringsAsFactors = FALSE
      )
    }
  }
}

eval_by_year <- do.call(rbind, year_rows)
cat("\n  Year-level breakdown (M1 and M3):\n\n")
print(eval_by_year[eval_by_year$note == "ok",
                   c("model","split","year","n_pos","prevalence",
                     "avg_precision","recall_fpr3","recall_fpr5")],
      row.names = FALSE)
saveRDS(eval_by_year, file.path(DIR_TABLES, "eval_by_year.rds"))
cat("\n  eval_by_year.rds saved.\n")

#==============================================================================#
# 6. Plots — M1 vs M3 (primary comparison)
#==============================================================================#

cat("\n[10_Evaluate.R] ── Section 6: Plots ─────────────────────────────\n")

## Helper: get test predictions for a named model
get_test <- function(nm) {
  if (is.null(preds[[nm]])) return(NULL)
  preds[[nm]]$test
}

##──────────────────────────────────────────────────────────────────────────────
## 6A. ROC Curves — M1 vs M3 (test set)
##──────────────────────────────────────────────────────────────────────────────

roc_list <- list()
for (nm in c("M1", "M3")) {
  dt <- get_test(nm)
  if (is.null(dt)) next
  roc_obj <- pROC::roc(dt$y, dt$p_csi, quiet=TRUE)
  auc_val <- as.numeric(pROC::auc(roc_obj))
  lbl     <- preds[[nm]]$label
  roc_list[[nm]] <- data.frame(
    model = sprintf("%s (AUC=%.3f)", lbl, auc_val),
    fpr   = 1 - roc_obj$specificities,
    tpr   = roc_obj$sensitivities,
    colour_key = lbl
  )
}
roc_df <- do.call(rbind, roc_list)

p_roc <- ggplot(roc_df, aes(x=fpr, y=tpr, colour=colour_key, group=colour_key)) +
  geom_line(linewidth=1.0) +
  geom_abline(slope=1, intercept=0, linetype="dashed", colour="grey70") +
  geom_vline(xintercept=c(0.03,0.05), linetype="dotted",
             colour="grey50", linewidth=0.5) +
  annotate("text", x=0.03, y=0.06, label="FPR=3%",
           hjust=-0.1, size=3, colour="grey45") +
  annotate("text", x=0.05, y=0.06, label="FPR=5%",
           hjust=-0.1, size=3, colour="grey45") +
  scale_colour_manual(values=ALL_COLOURS) +
  scale_x_continuous(labels=percent_format(accuracy=1), limits=c(0,1)) +
  scale_y_continuous(labels=percent_format(accuracy=1), limits=c(0,1)) +
  labs(title="ROC Curves — Test Set 2016–2019: M1 vs M3",
       subtitle="M1 = fundamentals only (ex-ante) | M3 = full features (triage)",
       x="False Positive Rate", y="True Positive Rate (Recall)",
       colour=NULL) +
  theme_minimal(base_size=12) + theme(legend.position="bottom")

ggsave(file.path(DIR_FIGURES,"eval_roc_curve.png"), p_roc,
       width=PLOT_WIDTH, height=PLOT_HEIGHT, dpi=PLOT_DPI)
cat("  eval_roc_curve.png saved.\n")

##──────────────────────────────────────────────────────────────────────────────
## 6B. Precision-Recall Curves — M1 vs M3
##──────────────────────────────────────────────────────────────────────────────

pr_list <- list()
for (nm in c("M1", "M3")) {
  dt <- get_test(nm); if (is.null(dt)) next
  pr_obj <- PRROC::pr.curve(
    scores.class0 = dt$p_csi[dt$y==1L],
    scores.class1 = dt$p_csi[dt$y==0L],
    curve=TRUE)
  lbl <- preds[[nm]]$label
  pr_df_nm <- as.data.frame(pr_obj$curve)
  names(pr_df_nm) <- c("recall","precision","threshold")
  pr_df_nm$colour_key <- sprintf("%s (AP=%.3f)", lbl, pr_obj$auc.integral)
  pr_df_nm$colour_map <- lbl
  pr_list[[nm]] <- pr_df_nm
}
pr_df      <- do.call(rbind, pr_list)
prevalence <- mean(as.integer(get_test("M3")$y))

p_pr <- ggplot(pr_df, aes(x=recall, y=precision,
                          colour=colour_map, group=colour_map)) +
  geom_line(linewidth=1.0) +
  geom_hline(yintercept=prevalence, linetype="dashed",
             colour="grey60", linewidth=0.8) +
  annotate("text", x=0.85, y=prevalence+0.015,
           label=sprintf("Baseline=%.1f%%",100*prevalence),
           colour="grey50", size=3) +
  scale_colour_manual(
    values=ALL_COLOURS,
    labels=setNames(unique(pr_df$colour_key), unique(pr_df$colour_map))
  ) +
  scale_x_continuous(labels=percent_format(accuracy=1), limits=c(0,1)) +
  scale_y_continuous(labels=percent_format(accuracy=1), limits=c(0,1)) +
  labs(title="Precision-Recall Curves — Test Set 2016–2019: M1 vs M3",
       subtitle=sprintf("Dashed = baseline prevalence (%.1f%%)",100*prevalence),
       x="Recall", y="Precision", colour=NULL) +
  theme_minimal(base_size=12) + theme(legend.position="bottom")

ggsave(file.path(DIR_FIGURES,"eval_pr_curve.png"), p_pr,
       width=PLOT_WIDTH, height=PLOT_HEIGHT, dpi=PLOT_DPI)
cat("  eval_pr_curve.png saved.\n")

##──────────────────────────────────────────────────────────────────────────────
## 6C. Calibration — M1 vs M3 (test set)
##──────────────────────────────────────────────────────────────────────────────

calib_list <- list()
for (nm in c("M1", "M3")) {
  dt <- as.data.table(get_test(nm)); if (is.null(dt)) next
  dt[, bin := cut(p_csi,
                  breaks=quantile(p_csi, probs=seq(0,1,0.1), na.rm=TRUE),
                  include.lowest=TRUE, labels=FALSE)]
  cb <- dt[, .(mean_pred=mean(p_csi), mean_actual=mean(y), n=.N), by=bin
  ][order(bin)]
  cb[, colour_key := preds[[nm]]$label]
  calib_list[[nm]] <- cb
}
calib_df <- rbindlist(calib_list)

p_calib <- ggplot(calib_df,
                  aes(x=mean_pred, y=mean_actual,
                      colour=colour_key, size=n)) +
  geom_abline(slope=1, intercept=0, linetype="dashed", colour="grey50") +
  geom_point(alpha=0.8) +
  geom_line(aes(group=colour_key), linewidth=0.7, alpha=0.6) +
  scale_colour_manual(values=ALL_COLOURS) +
  scale_size_continuous(name="n obs", range=c(2,8)) +
  scale_x_continuous(labels=percent_format(accuracy=1), limits=c(0,1)) +
  scale_y_continuous(labels=percent_format(accuracy=1), limits=c(0,1)) +
  labs(title="Calibration — Test Set 2016–2019: M1 vs M3",
       subtitle="Decile bins | Diagonal = perfect calibration",
       x="Mean Predicted P(CSI)", y="Actual CSI Rate", colour=NULL) +
  theme_minimal(base_size=12) + theme(legend.position="bottom")

ggsave(file.path(DIR_FIGURES,"eval_calibration.png"), p_calib,
       width=PLOT_WIDTH*0.9, height=PLOT_HEIGHT, dpi=PLOT_DPI)
cat("  eval_calibration.png saved.\n")

##──────────────────────────────────────────────────────────────────────────────
## 6D. Score distributions — M1 vs M3
##──────────────────────────────────────────────────────────────────────────────

score_list <- list()
for (nm in c("M1","M3")) {
  dt <- as.data.table(get_test(nm)); if (is.null(dt)) next
  dt[, label     := fifelse(y==1L, "CSI (y=1)", "Clean (y=0)")]
  dt[, model_lbl := preds[[nm]]$label]
  score_list[[nm]] <- dt
}
score_df <- rbindlist(score_list)

p_score <- ggplot(score_df,
                  aes(x=p_csi, fill=label, colour=label)) +
  geom_density(alpha=0.35, linewidth=0.6) +
  facet_wrap(~model_lbl, ncol=1) +
  scale_fill_manual(values=c("CSI (y=1)"="coral","Clean (y=0)"="steelblue")) +
  scale_colour_manual(values=c("CSI (y=1)"="coral","Clean (y=0)"="steelblue")) +
  scale_x_continuous(labels=percent_format(accuracy=1)) +
  labs(title="Predicted P(CSI) Distribution — Test Set 2016–2019",
       subtitle="Better separation = higher AUCPR",
       x="Predicted P(CSI)", y="Density", fill=NULL, colour=NULL) +
  theme_minimal(base_size=12) + theme(legend.position="bottom")

ggsave(file.path(DIR_FIGURES,"eval_score_dist.png"), p_score,
       width=PLOT_WIDTH, height=PLOT_HEIGHT*1.4, dpi=PLOT_DPI)
cat("  eval_score_dist.png saved.\n")

##──────────────────────────────────────────────────────────────────────────────
## 6E. Recall & Precision vs FPR — M1 and M3
##──────────────────────────────────────────────────────────────────────────────

thresh_plot_list <- list()
for (nm in c("M1","M3")) {
  dt <- get_test(nm); if (is.null(dt)) next
  roc_obj   <- pROC::roc(dt$y, dt$p_csi, quiet=TRUE)
  thresh_df <- data.frame(
    fpr       = 1 - roc_obj$specificities,
    recall    = roc_obj$sensitivities,
    threshold = roc_obj$thresholds
  )
  thresh_df$precision <- sapply(thresh_df$threshold, function(t) {
    pp <- as.integer(dt$p_csi >= t)
    tp <- sum(pp==1L & dt$y==1L)
    fp <- sum(pp==1L & dt$y==0L)
    if ((tp+fp)==0L) return(NA_real_)
    tp/(tp+fp)
  })
  thresh_df$model_lbl <- preds[[nm]]$label
  thresh_plot_list[[nm]] <- thresh_df
}

thresh_long <- rbindlist(thresh_plot_list) |>
  filter(fpr <= 0.15) |>
  select(fpr, recall, precision, model_lbl) |>
  pivot_longer(cols=c(recall, precision),
               names_to="metric", values_to="value")

p_thresh <- ggplot(thresh_long,
                   aes(x=fpr, y=value, colour=model_lbl,
                       linetype=metric)) +
  geom_line(linewidth=0.9) +
  geom_vline(xintercept=c(0.03,0.05), linetype="dotted",
             colour="grey50", linewidth=0.5) +
  annotate("text",x=0.03,y=0.96,label="3%",
           hjust=-0.2, size=3.5, colour="grey40") +
  annotate("text",x=0.05,y=0.96,label="5%",
           hjust=-0.2, size=3.5, colour="grey40") +
  scale_colour_manual(values=ALL_COLOURS) +
  scale_linetype_manual(values=c("recall"="solid","precision"="dashed")) +
  scale_x_continuous(labels=percent_format(accuracy=1), limits=c(0,0.15)) +
  scale_y_continuous(labels=percent_format(accuracy=1), limits=c(0,1)) +
  labs(title="Recall & Precision vs FPR — Test Set 2016–2019: M1 vs M3",
       x="False Positive Rate", y="Metric Value",
       colour="Model", linetype="Metric") +
  theme_minimal(base_size=12) + theme(legend.position="bottom")

ggsave(file.path(DIR_FIGURES,"eval_threshold.png"), p_thresh,
       width=PLOT_WIDTH, height=PLOT_HEIGHT, dpi=PLOT_DPI)
cat("  eval_threshold.png saved.\n")

##──────────────────────────────────────────────────────────────────────────────
## 6F. Year-level AP — M1 vs M3
##──────────────────────────────────────────────────────────────────────────────

yby_plot <- eval_by_year[eval_by_year$note=="ok", ]
yby_plot$period <- ifelse(yby_plot$split=="test",
                          "Test (2016–2019)", "OOS (2020–2022)")
yby_plot <- yby_plot[yby_plot$year <= 2022, ]

p_yby <- ggplot(yby_plot,
                aes(x=year, y=avg_precision,
                    fill=model, group=model)) +
  geom_col(position=position_dodge(width=0.75), width=0.7, alpha=0.85) +
  geom_hline(yintercept=0.61, linetype="dashed",
             colour="grey40", linewidth=0.7) +
  annotate("text", x=2015.7, y=0.625,
           label="Paper R@FPR3=0.61", hjust=0, size=3, colour="grey40") +
  facet_wrap(~period, scales="free_x", nrow=1) +
  scale_fill_manual(values=ALL_COLOURS) +
  scale_y_continuous(labels=number_format(accuracy=0.01),
                     limits=c(0,1)) +
  labs(title="Year-Level Average Precision — M1 vs M3",
       x="Year", y="Average Precision", fill="Model") +
  theme_minimal(base_size=11) + theme(legend.position="bottom")

ggsave(file.path(DIR_FIGURES,"eval_by_year.png"), p_yby,
       width=PLOT_WIDTH*1.2, height=PLOT_HEIGHT, dpi=PLOT_DPI)
cat("  eval_by_year.png saved.\n")

##──────────────────────────────────────────────────────────────────────────────
## 6G. M1 / M3 / M4 bar comparison — test and OOS
##──────────────────────────────────────────────────────────────────────────────

bar_df <- eval_performance |>
  filter(set %in% c("test_2016_2019","oos_2020_2022")) |>
  filter(model %in% c("M1 AG Fund","M3 AG Raw","M4 AG Latent Raw")) |>
  mutate(set_label = ifelse(set=="test_2016_2019",
                            "Test 2016–2019", "OOS 2020–2022")) |>
  select(model, set_label, avg_precision, recall_fpr3)

bar_long <- bar_df |>
  pivot_longer(cols=c(avg_precision, recall_fpr3),
               names_to="metric",
               values_to="value") |>
  mutate(metric=recode(metric,
                       avg_precision="Average Precision",
                       recall_fpr3="Recall@FPR3"))

p_bar <- ggplot(bar_long,
                aes(x=model, y=value, fill=model)) +
  geom_col(width=0.65, alpha=0.85) +
  geom_hline(data=data.frame(metric="Recall@FPR3", yint=0.61),
             aes(yintercept=yint), linetype="dashed",
             colour="grey40", linewidth=0.7, inherit.aes=FALSE) +
  facet_grid(metric ~ set_label, scales="free_y") +
  scale_fill_manual(values=ALL_COLOURS) +
  scale_y_continuous(labels=number_format(accuracy=0.01)) +
  labs(title="M1 / M3 / M4 Performance Comparison",
       subtitle="Dashed line = paper benchmark R@FPR3=0.61 (Recall@FPR3 panel only)",
       x=NULL, y=NULL, fill=NULL) +
  theme_minimal(base_size=11) +
  theme(legend.position="none",
        axis.text.x=element_text(angle=15, hjust=1, size=9))

ggsave(file.path(DIR_FIGURES,"eval_m1_vs_m3_vs_m4.png"), p_bar,
       width=PLOT_WIDTH*1.1, height=PLOT_HEIGHT*1.2, dpi=PLOT_DPI)
cat("  eval_m1_vs_m3_vs_m4.png saved.\n")

#==============================================================================#
# 7. Assertions
#==============================================================================#

cat("\n[10_Evaluate.R] Running assertions...\n")

stopifnot(all(eval_performance$auc_roc > 0.5, na.rm=TRUE))

## R@FPR5 >= R@FPR3 for all rows
stopifnot(all(eval_performance$recall_fpr5 >= eval_performance$recall_fpr3,
              na.rm=TRUE))

## M3 test AP > 0.70 (known result)
m3_test <- eval_performance[eval_performance$model=="M3 AG Raw" &
                              eval_performance$set=="test_2016_2019", ]
if (nrow(m3_test) > 0)
  stopifnot(m3_test$avg_precision > 0.70)

cat("[10_Evaluate.R] All assertions passed.\n")

#==============================================================================#
# 8. Final thesis summary table
#==============================================================================#

cat("\n[10_Evaluate.R] ══════════════════════════════════════\n")
cat("  THESIS RESULTS TABLE — AutoGluon M1 / M3 / M4\n")
cat("  ══════════════════════════════════════\n\n")

cat(sprintf("  %-22s | %-20s | %6s | %6s | %6s | %6s\n",
            "Model", "Set", "AP", "AUC", "R@FPR3", "R@FPR5"))
cat(sprintf("  %-22s | %-20s | %6s | %6s | %6s | %6s\n",
            "----------------------", "--------------------",
            "------", "------", "------", "------"))

for (i in seq_len(nrow(eval_performance))) {
  r <- eval_performance[i,]
  cat(sprintf("  %-22s | %-20s | %6.4f | %6.4f | %6.4f | %6.4f\n",
              r$model, r$set,
              r$avg_precision, r$auc_roc,
              r$recall_fpr3,   r$recall_fpr5))
}
cat(sprintf("  %-22s | %-20s | %6s | %6s | %6s | %6s\n",
            "Paper (Tewari)","test","—","—","0.6100","—"))

cat(sprintf("\n[10_Evaluate.R] DONE: %s\n", format(Sys.time())))

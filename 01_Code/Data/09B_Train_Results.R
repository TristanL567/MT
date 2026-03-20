#==============================================================================#
#==== 09B_Train_Results.R =====================================================#
#==== AutoGluon Training Diagnostics — Leaderboard, CV, Feature Importance ===#
#==============================================================================#
#
# PURPOSE:
#   Inspect and summarise the trained AutoGluon models WITHOUT touching the
#   test set beyond what was already evaluated in 09C_AutoGluon.py.
#   All outputs here are derived from:
#     - AutoGluon leaderboard (holdout 2011–2015 validation scores)
#     - CV metrics from Stage 2 expanding window folds (honest estimate)
#     - Feature importance (permutation importance on holdout)
#     - XGBoost HPO convergence as supplementary comparison
#
#   Primary models: M1 (fund), M3 (raw), M4 (latent_raw)
#   Reference:      M3 XGBoost (hand-tuned Bayesian HPO)
#
# INPUTS:
#   - config.R
#   - DIR_TABLES/ag_fund/ag_eval_summary.json
#   - DIR_TABLES/ag_raw/ag_eval_summary.json        (saved as ag_predictor/)
#   - DIR_TABLES/ag_latent_raw/ag_eval_summary.json
#   - DIR_TABLES/ag_latent_fund/ag_eval_summary.json
#   - DIR_TABLES/ag_fund/ag_leaderboard.csv
#   - DIR_TABLES/ag_raw/ag_leaderboard.csv
#   - DIR_TABLES/ag_latent_raw/ag_leaderboard.csv
#   - DIR_TABLES/ag_fund/ag_feature_importance.csv
#   - DIR_TABLES/ag_raw/ag_feature_importance.csv
#   - DIR_MODELS/xgb_raw.rds   (XGBoost HPO reference)
#   - DIR_MODELS/xgb_fund.rds
#
# OUTPUTS:
#   - DIR_FIGURES/ag_leaderboard_comparison.png
#   - DIR_FIGURES/ag_cv_comparison.png
#   - DIR_FIGURES/ag_feature_importance_M1.png
#   - DIR_FIGURES/ag_feature_importance_M3.png
#   - DIR_FIGURES/xgb_hpo_convergence_raw.png
#   - DIR_FIGURES/xgb_hpo_convergence_fund.png
#   - DIR_TABLES/ag_train_summary.rds
#
#==============================================================================#

source("config.R")

suppressPackageStartupMessages({
  library(dplyr)
  library(data.table)
  library(ggplot2)
  library(jsonlite)
  library(scales)
  library(tidyr)
  library(xgboost)
})

cat("\n[09B_Train_Results.R] START:", format(Sys.time()), "\n")

#==============================================================================#
# 0. Path helpers
#==============================================================================#

## AutoGluon output directories
## Note: M3 was saved to ag_predictor (old script) not ag_raw
AG_DIRS <- list(
  M1 = file.path(DIR_TABLES, "ag_fund"),
  M2 = file.path(DIR_TABLES, "ag_latent_fund"),
  M3 = file.path(DIR_TABLES, "ag_raw"),
  M4 = file.path(DIR_TABLES, "ag_latent_raw")
)

MODEL_LABELS <- c(
  M1 = "M1 — Fund (AG)",
  M2 = "M2 — Latent Fund (AG)",
  M3 = "M3 — Raw (AG)",
  M4 = "M4 — Latent Raw (AG)"
)

MODEL_COLOURS <- c(
  "M1 — Fund (AG)"         = "#2196F3",   # blue
  "M2 — Latent Fund (AG)"  = "#9C27B0",   # purple
  "M3 — Raw (AG)"          = "#F44336",   # red
  "M4 — Latent Raw (AG)"   = "#FF9800",   # orange
  "M3 — Raw (XGB)"         = "#4CAF50"    # green
)

#==============================================================================#
# 1. Load AutoGluon eval summaries
#==============================================================================#

cat("[09B] Loading AutoGluon eval summaries...\n")

ag_summaries <- list()

for (nm in names(AG_DIRS)) {
  path <- file.path(AG_DIRS[[nm]], "ag_eval_summary.json")
  if (!file.exists(path)) {
    cat(sprintf("  [%s] Summary not found: %s\n", nm, path))
    next
  }
  ag_summaries[[nm]] <- jsonlite::fromJSON(path)
  cat(sprintf("  [%s] Loaded: CV AP=%.4f | Test AP=%.4f\n",
              nm,
              ag_summaries[[nm]]$cv_avg_precision %||% NA,
              ag_summaries[[nm]]$test$avg_precision %||% NA))
}

## Helper for null-coalescing
`%||%` <- function(a, b) if (!is.null(a) && !is.na(a)) a else b

#==============================================================================#
# 2. Load AutoGluon leaderboards
#==============================================================================#

cat("\n[09B] Loading leaderboards...\n")

leaderboards <- list()

for (nm in names(AG_DIRS)) {
  path <- file.path(AG_DIRS[[nm]], "ag_leaderboard.csv")
  if (!file.exists(path)) next
  lb <- fread(path)
  lb[, model_family := nm]
  leaderboards[[nm]] <- lb
  cat(sprintf("  [%s] %d models in leaderboard\n", nm, nrow(lb)))
}

#==============================================================================#
# 3. Load feature importance
#==============================================================================#

cat("\n[09B] Loading feature importance...\n")

feat_imp <- list()

for (nm in c("M1", "M3")) {
  path <- file.path(AG_DIRS[[nm]], "ag_feature_importance.csv")
  if (!file.exists(path)) next
  fi <- fread(path)
  setnames(fi, old = names(fi)[1], new = "feature")
  fi[, model := nm]
  feat_imp[[nm]] <- fi
  cat(sprintf("  [%s] %d features\n", nm, nrow(fi)))
}

#==============================================================================#
# 4. Load XGBoost models for HPO convergence comparison
#==============================================================================#

cat("\n[09B] Loading XGBoost models...\n")

xgb_models <- list()

for (nm in c("raw", "fund")) {
  path <- file.path(DIR_MODELS, sprintf("xgb_%s.rds", nm))
  if (!file.exists(path)) {
    cat(sprintf("  xgb_%s.rds not found — skipping\n", nm))
    next
  }
  xgb_models[[nm]] <- readRDS(path)
  cat(sprintf("  xgb_%s: CV AUCPR=%.4f | rounds=%d\n",
              nm,
              xgb_models[[nm]]$cv_aucpr_mean,
              xgb_models[[nm]]$optimal_rounds))
}

#==============================================================================#
# 5. CV Comparison Table — AG models + XGBoost reference
#==============================================================================#

cat("\n[09B] ── Section 5: CV Comparison ───────────────────────────────\n")

cv_rows <- list()

## AutoGluon CV results
for (nm in names(ag_summaries)) {
  s <- ag_summaries[[nm]]
  cv_rows[[length(cv_rows)+1]] <- data.frame(
    model         = MODEL_LABELS[[nm]],
    cv_ap         = round(s$cv_avg_precision %||% NA_real_, 4),
    test_ap       = round(s$test$avg_precision %||% NA_real_, 4),
    test_auc      = round(s$test$auc_roc %||% NA_real_, 4),
    test_r_fpr3   = round(s$test$recall_fpr3 %||% NA_real_, 4),
    test_r_fpr5   = round(s$test$recall_fpr5 %||% NA_real_, 4),
    oos_ap        = round(s$oos_2020_2022$avg_precision %||% NA_real_, 4),
    oos_r_fpr3    = round(s$oos_2020_2022$recall_fpr3 %||% NA_real_, 4),
    stringsAsFactors = FALSE
  )
}

## XGBoost reference row
for (nm in names(xgb_models)) {
  res  <- xgb_models[[nm]]
  te   <- res$eval_table[res$eval_table$set == "test", ]
  oos  <- res$eval_table[res$eval_table$set == "oos",  ]
  lbl  <- ifelse(nm == "raw", "M3 — Raw (XGB)", "M1 — Fund (XGB)")
  cv_rows[[length(cv_rows)+1]] <- data.frame(
    model       = lbl,
    cv_ap       = round(res$cv_aucpr_mean, 4),
    test_ap     = round(te$avg_precision,  4),
    test_auc    = round(te$auc_roc,        4),
    test_r_fpr3 = round(te$recall_fpr3,    4),
    test_r_fpr5 = round(te$recall_fpr5,    4),
    oos_ap      = round(oos$avg_precision,  4),
    oos_r_fpr3  = round(oos$recall_fpr3,    4),
    stringsAsFactors = FALSE
  )
}

cv_table <- do.call(rbind, cv_rows)

cat("\n  Full CV and test comparison:\n\n")
print(cv_table, row.names = FALSE)

#==============================================================================#
# 6. Plot: CV AP comparison — all models
#==============================================================================#

cat("\n[09B] ── Section 6: CV AP Comparison Plot ───────────────────────\n")

cv_long <- cv_table |>
  select(model, cv_ap, test_ap, oos_ap) |>
  pivot_longer(cols = c(cv_ap, test_ap, oos_ap),
               names_to  = "metric",
               values_to = "ap") |>
  mutate(
    metric = factor(metric,
                    levels = c("cv_ap", "test_ap", "oos_ap"),
                    labels = c("CV (expanding window)",
                               "Test 2016–2019",
                               "OOS 2020–2022")),
    model = factor(model, levels = rev(unique(cv_table$model)))
  ) |>
  filter(!is.na(ap))

p_cv <- ggplot(cv_long, aes(x = model, y = ap, fill = metric)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.7) +
  geom_hline(yintercept = 0.61, linetype = "dashed",
             colour = "grey40", linewidth = 0.8) +
  annotate("text", x = 0.6, y = 0.62,
           label = "Paper benchmark R@FPR3=0.61",
           hjust = 0, size = 3, colour = "grey40") +
  coord_flip() +
  scale_fill_manual(values = c(
    "CV (expanding window)" = "#90CAF9",
    "Test 2016–2019"        = "#1565C0",
    "OOS 2020–2022"         = "#F44336"
  )) +
  scale_y_continuous(labels = scales::number_format(accuracy = 0.01),
                     limits = c(0, 1)) +
  labs(
    title    = "Average Precision: CV vs Test vs OOS — All Models",
    subtitle = "Dashed line = paper benchmark (R@FPR3=0.61 mapped to AP scale)",
    x        = NULL,
    y        = "Average Precision",
    fill     = "Evaluation Set"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

ggsave(file.path(DIR_FIGURES, "ag_cv_comparison.png"),
       p_cv, width = PLOT_WIDTH * 1.2, height = PLOT_HEIGHT * 1.1,
       dpi = PLOT_DPI)
cat("  ag_cv_comparison.png saved.\n")

#==============================================================================#
# 7. Plot: Leaderboard comparison — M1 vs M3 individual model scores
#==============================================================================#

cat("\n[09B] ── Section 7: Leaderboard Plot ───────────────────────────\n")

## Combine M1 and M3 leaderboards (non-FULL models only)
lb_list <- list()
for (nm in c("M1", "M3")) {
  if (is.null(leaderboards[[nm]])) next
  lb <- leaderboards[[nm]][!grepl("_FULL", model)]
  lb[, ag_model := MODEL_LABELS[[nm]]]
  lb_list[[nm]] <- lb[!is.na(score_val), .(model, score_val, ag_model)]
}

if (length(lb_list) > 0) {
  lb_df <- rbindlist(lb_list)
  
  ## Clean model names for display
  lb_df[, model_clean := gsub("_", " ", model)]
  lb_df[, model_clean := factor(model_clean,
                                levels = lb_df[order(score_val),
                                               unique(model_clean)])]
  
  p_lb <- ggplot(lb_df, aes(x = model_clean, y = score_val,
                            fill = ag_model)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.75) +
    coord_flip() +
    scale_fill_manual(values = c(
      "M1 — Fund (AG)" = "#2196F3",
      "M3 — Raw (AG)"  = "#F44336"
    )) +
    scale_y_continuous(labels = scales::number_format(accuracy = 0.01),
                       limits = c(0.4, 0.75)) +
    labs(
      title    = "AutoGluon Holdout AP — Individual Models: M1 vs M3",
      subtitle = "Holdout = years 2011–2015 | good_quality preset",
      x        = "Algorithm",
      y        = "Average Precision (holdout)",
      fill     = "Feature Set"
    ) +
    theme_minimal(base_size = 11) +
    theme(legend.position = "bottom",
          axis.text.y = element_text(size = 9))
  
  ggsave(file.path(DIR_FIGURES, "ag_leaderboard_comparison.png"),
         p_lb, width = PLOT_WIDTH, height = PLOT_HEIGHT * 1.2,
         dpi = PLOT_DPI)
  cat("  ag_leaderboard_comparison.png saved.\n")
}

#==============================================================================#
# 8. Feature importance plots — M1 and M3
#==============================================================================#

cat("\n[09B] ── Section 8: Feature Importance Plots ───────────────────\n")

for (nm in names(feat_imp)) {
  fi    <- feat_imp[[nm]]
  top_n <- min(20L, nrow(fi))
  
  ## Sort by importance descending, take top N
  fi_top <- fi[order(-importance)][seq_len(top_n)]
  fi_top[, feature := factor(feature, levels = rev(feature))]
  
  label <- MODEL_LABELS[[nm]]
  colour <- MODEL_COLOURS[[label]]
  
  p_fi <- ggplot(fi_top, aes(x = feature, y = importance)) +
    geom_col(fill = colour, width = 0.7, alpha = 0.85) +
    geom_errorbar(aes(ymin = importance - stddev,
                      ymax = importance + stddev),
                  width = 0.3, colour = "grey40", linewidth = 0.5) +
    coord_flip() +
    scale_y_continuous(labels = scales::number_format(accuracy = 0.0001)) +
    labs(
      title    = paste0("Feature Importance — ", label),
      subtitle = paste0("Permutation importance on holdout (2011–2015) | ",
                        "Error bars = ±1 SD across 5 permutations"),
      x        = NULL,
      y        = "Permutation Importance"
    ) +
    theme_minimal(base_size = 11) +
    theme(axis.text.y = element_text(size = 9))
  
  fname <- file.path(DIR_FIGURES,
                     paste0("ag_feature_importance_", nm, ".png"))
  ggsave(fname, p_fi,
         width  = PLOT_WIDTH,
         height = max(PLOT_HEIGHT, top_n * 0.32),
         dpi    = PLOT_DPI)
  cat(sprintf("  Feature importance plot saved: %s\n", basename(fname)))
  
  cat(sprintf("\n  [%s] Top 10 features:\n", nm))
  print(head(fi[order(-importance), .(feature, importance, stddev, p_value)],
             10L))
}

#==============================================================================#
# 9. XGBoost HPO convergence — supplementary
#==============================================================================#

cat("\n[09B] ── Section 9: XGBoost HPO Convergence ─────────────────────\n")

for (nm in names(xgb_models)) {
  res     <- xgb_models[[nm]]
  bo_hist <- res$bo_history
  
  if (is.null(bo_hist)) next
  
  bo_plot <- bo_hist |>
    dplyr::arrange(Round) |>
    dplyr::mutate(
      cummax_aucpr = cummax(aucpr),
      phase        = ifelse(Round <= 10L, "Random init", "Bayesian")
    )
  
  p_bo <- ggplot(bo_plot, aes(x = Round)) +
    geom_point(aes(y = aucpr, colour = phase), alpha = 0.7, size = 2.5) +
    geom_line(aes(y = cummax_aucpr),
              colour = "steelblue", linewidth = 1.0) +
    geom_vline(xintercept = 10.5,
               linetype = "dotted", colour = "grey50") +
    annotate("text", x = 5.5, y = max(bo_plot$aucpr) * 1.01,
             label = "Random", colour = "grey50", size = 3) +
    annotate("text", x = 20, y = max(bo_plot$aucpr) * 1.01,
             label = "Bayesian", colour = "grey50", size = 3) +
    scale_colour_manual(values = c("Random init" = "coral",
                                   "Bayesian"    = "steelblue")) +
    labs(
      title    = paste0("XGBoost HPO Convergence — ", toupper(nm),
                        " features"),
      subtitle = paste0("Line = running best | Best CV AUCPR = ",
                        round(max(bo_plot$aucpr), 4),
                        " | Optimal rounds = ", res$optimal_rounds),
      x        = "HPO Iteration",
      y        = "CV AUCPR",
      colour   = NULL
    ) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "bottom")
  
  fname <- file.path(DIR_FIGURES,
                     paste0("xgb_hpo_convergence_", nm, ".png"))
  ggsave(fname, p_bo,
         width = PLOT_WIDTH, height = PLOT_HEIGHT, dpi = PLOT_DPI)
  cat(sprintf("  [%s] HPO convergence plot saved: %s\n",
              toupper(nm), basename(fname)))
  
  ## Convergence diagnostic
  last_10_range <- diff(range(tail(bo_plot$aucpr, 10L)))
  cat(sprintf("  [%s] AUCPR range in last 10 iterations: %.4f %s\n",
              toupper(nm), last_10_range,
              ifelse(last_10_range < 0.01, "(converged)", "(check convergence)")))
}

#==============================================================================#
# 10. Save combined training summary
#==============================================================================#

cat("\n[09B] ── Section 10: Save Summary ───────────────────────────────\n")

saveRDS(cv_table, file.path(DIR_TABLES, "ag_train_summary.rds"))
cat("  ag_train_summary.rds saved.\n")

cat("\n[09B] ══════════════════════════════════════════════════════\n")
cat("  Summary — AutoGluon vs XGBoost:\n\n")

cat(sprintf("  %-22s | %7s | %7s | %7s | %7s\n",
            "Model", "CV AP", "Test AP", "R@FPR3", "OOS AP"))
cat(sprintf("  %-22s | %7s | %7s | %7s | %7s\n",
            "----------------------",
            "-------", "-------", "-------", "-------"))

for (i in seq_len(nrow(cv_table))) {
  r <- cv_table[i, ]
  cat(sprintf("  %-22s | %7.4f | %7.4f | %7.4f | %7s\n",
              r$model, r$cv_ap, r$test_ap, r$test_r_fpr3,
              ifelse(is.na(r$oos_ap), "—", sprintf("%.4f", r$oos_ap))))
}

cat(sprintf("\n[09B_Train_Results.R] DONE: %s\n", format(Sys.time())))
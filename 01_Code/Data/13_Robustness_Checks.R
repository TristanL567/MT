#==============================================================================#
#==== 13_Robustness.R =========================================================#
#==== CSI Parameter Sensitivity + Recovery Classifier + Refined Index ========#
#==============================================================================#
#
# PURPOSE:
#   Two-part robustness analysis addressing the core challenge visible in the
#   return distribution of excluded firms: the model excludes high-volatility
#   firms with fat tails in BOTH directions, not purely terminal zombies.
#
# PART A — CSI Parameter Grid (27 combinations)
#   Tests whether M1 model performance is stable across alternative CSI
#   definitions (varying crash threshold C, recovery ceiling M, zombie
#   duration T). Answers: are results an artefact of the base-case definition?
#
# PART B — Recovery Classifier (Phoenix vs Zombie)
#   Among M1-flagged firms, builds a solvency-based decision tree to separate:
#     - Zombie firms: crash + fail to recover (true CSI — exclude)
#     - Phoenix firms: crash + recover (false positive — keep)
#   Uses balance sheet features available at prediction time (no lookahead).
#   Produces S4 = Refined strategy: M1 flag AND zombie classifier.
#
# PART C — Refined Index Backtest
#   Backtests S4 against S1 and benchmark. Tests whether the two-stage filter
#   produces better risk-adjusted returns by reducing phoenix exclusions.
#
# KEY FEATURES:
#   - All features from features_fund.rds (no price data, no lookahead)
#   - Decision tree trained on train period (1993-2015), applied OOS (2016+)
#   - Interpretable: tree depth <= 4, printed as decision rules
#   - Directly answers: "what distinguishes zombies from phoenixes?"
#
# INPUTS:
#   - config.R
#   - PATH_FEATURES_FUND (features_fund.rds)
#   - PATH_LABELS_BASE   (labels_base.rds)
#   - PATH_PRICES_MONTHLY
#   - DIR_LABELS/labels_<param_id>.rds  (27 grid label files)
#   - DIR_TABLES/ag_fund/ag_preds_test.parquet
#   - DIR_TABLES/ag_fund/ag_preds_oos.parquet
#   - DIR_TABLES/index_weights.rds
#   - DIR_TABLES/index_returns.rds
#
# OUTPUTS:
#   - DIR_TABLES/robust_grid_performance.rds
#   - DIR_TABLES/robust_recovery_classifier.rds
#   - DIR_TABLES/robust_index_returns.rds
#   - DIR_FIGURES/robust_grid_heatmap.png
#   - DIR_FIGURES/robust_tree_plot.png
#   - DIR_FIGURES/robust_feature_importance.png
#   - DIR_FIGURES/robust_cumulative_s4.png
#   - DIR_FIGURES/robust_phoenix_zombie_dist.png
#
#==============================================================================#

source("config.R")

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(ggplot2)
  library(arrow)
  library(scales)
  library(tidyr)
  library(lubridate)
  library(rpart)          # decision tree
  library(rpart.plot)     # tree visualisation
  library(PRROC)          # AP metric
  library(pROC)           # AUC metric
})

cat("\n[13_Robustness.R] START:", format(Sys.time()), "\n")

## ── Parameters ───────────────────────────────────────────────────────────────
TRAIN_END        <- 2015L
OOS_START        <- 2016L
OOS_END          <- 2022L
EXCLUSION_RATE   <- 0.05   ## top 5% by M1 score = flagged
TREE_DEPTH       <- 4      ## max depth of recovery classifier tree
MIN_BUCKET       <- 30     ## min obs per leaf (prevents overfitting)

## Solvency features for the recovery classifier
## All from features_fund.rds — no price data, available at fiscal year end
SOLVENCY_FEATURES <- c(
  "altman_z2",           ## retained earnings / assets — accumulated deficit
  "leverage",            ## total debt / total assets — capital structure
  "interest_coverage",   ## ebit / interest expense — can service debt?
  "cash_ratio",          ## cash / current liabilities — immediate liquidity
  "ocf_margin",          ## operating cash flow / sales — cash generation
  "roll_min_3y_earn_yld", ## worst earnings yield in 3yr — persistent stress
  "roll_min_3y_roic",    ## worst return on capital — structural impairment
  "consec_decline_sale", ## consecutive revenue declines — demand destruction
  "peak_drop_log_mkvalt",## size deterioration from peak — market signal
  "acct_mom_roa",        ## acceleration of ROA — worsening trajectory
  "roll_sd_5y_earn_yld", ## earnings instability — volatile vs stable distress
  "yoy_leverage"         ## leverage change — taking on more debt under stress
)

## ── Helper functions ─────────────────────────────────────────────────────────

fn_recall_at_fpr <- function(y_true, y_pred, fpr_target) {
  roc_obj <- pROC::roc(y_true, y_pred, quiet=TRUE)
  fpr_vals <- 1 - roc_obj$specificities
  eligible <- which(fpr_vals <= fpr_target)
  if (length(eligible) == 0L) return(NA_real_)
  max(roc_obj$sensitivities[eligible])
}

fn_avg_precision <- function(y_true, y_pred) {
  tryCatch(
    PRROC::pr.curve(
      scores.class0 = y_pred[y_true == 1L],
      scores.class1 = y_pred[y_true == 0L],
      curve = FALSE
    )$auc.integral,
    error = function(e) NA_real_
  )
}

fn_performance <- function(ret_vec, rf_annual = 0.03) {
  ret_vec <- ret_vec[!is.na(ret_vec)]
  if (length(ret_vec) < 12L) return(NULL)
  n_years    <- length(ret_vec) / 12
  rf_monthly <- (1 + rf_annual)^(1/12) - 1
  cum_ret    <- prod(1 + ret_vec) - 1
  cagr       <- (1 + cum_ret)^(1/n_years) - 1
  excess     <- ret_vec - rf_monthly
  sharpe     <- mean(excess) / sd(excess) * sqrt(12)
  cum_idx    <- cumprod(1 + ret_vec)
  max_dd     <- min((cum_idx - cummax(cum_idx)) / cummax(cum_idx))
  data.frame(cagr=round(cagr,4), sharpe=round(sharpe,4),
             max_dd=round(max_dd,4),
             vol=round(sd(ret_vec)*sqrt(12),4))
}

#==============================================================================#
# 0. Load shared data
#==============================================================================#

cat("\n[13] Loading shared data...\n")

## Features (fund — no price data)
cat("  Loading features_fund.rds...\n")
features <- as.data.table(readRDS(PATH_FEATURES_FUND))
cat(sprintf("  Features: %d rows × %d cols\n", nrow(features), ncol(features)))

## Base case labels
cat("  Loading labels_base.rds...\n")
labels_base <- as.data.table(readRDS(PATH_LABELS_BASE))

## M1 predictions (test + OOS)
cat("  Loading M1 predictions...\n")
m1_preds <- rbindlist(list(
  as.data.table(arrow::read_parquet(
    file.path(DIR_TABLES, "ag_fund", "ag_preds_test.parquet"))),
  as.data.table(arrow::read_parquet(
    file.path(DIR_TABLES, "ag_fund", "ag_preds_oos.parquet")))
))
setnames(m1_preds, "p_csi", "p_m1")

## Monthly returns
cat("  Loading monthly returns...\n")
monthly <- as.data.table(readRDS(PATH_PRICES_MONTHLY))
setnames(monthly, "ret_adj", "ret")
setnames(monthly, "mktcap",  "mkvalt")
monthly[, year  := year(date)]
monthly[, month := month(date)]
monthly[, ret   := pmin(pmax(ret, -0.99, na.rm=TRUE), 10, na.rm=TRUE)]

## Existing index returns (benchmark + S1)
port_returns <- readRDS(file.path(DIR_TABLES, "index_returns.rds"))

cat("  All data loaded.\n")

#==============================================================================#
# PART A — CSI Parameter Grid Sensitivity
#==============================================================================#

cat("\n[13] ══ PART A: CSI Parameter Grid Sensitivity ══\n")

## Check which grid label files exist
## Exclude labels_bucket.rds — different schema (y_loser not y)
grid_files <- list.files(DIR_LABELS, pattern="^labels_.*[.]rds$",
                         full.names=TRUE)
grid_files <- grid_files[!grepl("labels_bucket[.]rds$", grid_files)]
cat(sprintf("  Found %d grid label files in %s\n",
            length(grid_files), DIR_LABELS))

if (length(grid_files) == 0L) {
  cat("  WARNING: No grid label files found — run 05_CSI_Label.R first.\n")
  cat("  Skipping Part A.\n")
  grid_perf <- NULL
} else {
  
  grid_results <- list()
  
  for (fpath in grid_files) {
    
    param_id <- gsub("labels_|\\.rds", "", basename(fpath))
    labels_g <- readRDS(fpath)
    labels_g <- as.data.table(labels_g)
    
    ## Join M1 predictions with this grid's labels
    ## Label shift: prediction at year t → CSI in year t+1
    ## So join prediction year t with label year t+1
    merged <- merge(
      m1_preds[, .(permno, pred_year = year, p_m1)],
      labels_g[!is.na(y), .(permno, year, y)],
      by.x = c("permno", "pred_year"),
      by.y = c("permno", "year"),
      all = FALSE
    )
    
    if (nrow(merged) < 100L || sum(merged$y == 1L) < 10L) next
    
    ## Split test vs OOS
    merged_test <- merged[pred_year >= OOS_START &
                            pred_year <= OOS_END]
    if (nrow(merged_test) < 50L) next
    
    ap  <- fn_avg_precision(merged_test$y, merged_test$p_m1)
    r3  <- fn_recall_at_fpr(merged_test$y, merged_test$p_m1, 0.03)
    r5  <- fn_recall_at_fpr(merged_test$y, merged_test$p_m1, 0.05)
    auc <- tryCatch(
      as.numeric(pROC::auc(pROC::roc(merged_test$y,
                                     merged_test$p_m1, quiet=TRUE))),
      error = function(e) NA_real_)
    
    ## Get parameters from CSI_GRID
    params_row <- CSI_GRID[CSI_GRID$param_id == param_id, ]
    
    grid_results[[param_id]] <- data.frame(
      param_id   = param_id,
      C          = if (nrow(params_row) > 0) params_row$C else NA_real_,
      M          = if (nrow(params_row) > 0) params_row$M else NA_real_,
      T          = if (nrow(params_row) > 0) params_row$T else NA_real_,
      n_csi      = sum(merged_test$y == 1L),
      prevalence = round(mean(merged_test$y == 1L), 4),
      ap         = round(ap,  4),
      r_fpr3     = round(r3,  4),
      r_fpr5     = round(r5,  4),
      auc        = round(auc, 4),
      stringsAsFactors = FALSE
    )
  }
  
  if (length(grid_results) > 0L) {
    grid_perf <- do.call(rbind, grid_results)
    grid_perf <- grid_perf[order(-grid_perf$ap), ]
    
    cat("\n  Grid results (top 10 by AP):\n\n")
    print(head(grid_perf[, c("param_id","C","M","T","ap","r_fpr3","auc")], 10),
          row.names=FALSE)
    
    saveRDS(grid_perf, file.path(DIR_TABLES, "robust_grid_performance.rds"))
    cat("\n  robust_grid_performance.rds saved.\n")
    
    ## ── Heatmap: AP by C × T (averaged over M) ──────────────────────────────
    if (all(c("C","M","T","ap") %in% names(grid_perf)) &&
        !all(is.na(grid_perf$C))) {
      
      heatmap_dt <- as.data.table(grid_perf)[
        !is.na(C) & !is.na(T) & !is.na(ap),
        .(mean_ap = mean(ap, na.rm=TRUE)),
        by = .(C, T)
      ]
      heatmap_dt[, C_label := paste0("C=", C)]
      heatmap_dt[, T_label := paste0("T=", T, "m")]
      
      p_heat <- ggplot(heatmap_dt,
                       aes(x=factor(T), y=factor(C), fill=mean_ap)) +
        geom_tile(colour="white", linewidth=0.5) +
        geom_text(aes(label=round(mean_ap, 3)), size=3.5, colour="white") +
        scale_fill_gradient2(low="#1565C0", mid="#42A5F5",
                             high="#E53935", midpoint=median(heatmap_dt$mean_ap),
                             name="Mean AP") +
        scale_x_discrete(name="Zombie Duration T (months)") +
        scale_y_discrete(name="Crash Threshold C") +
        labs(title="M1 Average Precision by CSI Parameter Combination",
             subtitle="Averaged over recovery ceiling M | OOS 2016–2022") +
        theme_minimal(base_size=12)
      
      ggsave(file.path(DIR_FIGURES, "robust_grid_heatmap.png"), p_heat,
             width=PLOT_WIDTH, height=PLOT_HEIGHT, dpi=PLOT_DPI)
      cat("  robust_grid_heatmap.png saved.\n")
    }
  } else {
    cat("  No valid grid results — check label files.\n")
    grid_perf <- NULL
  }
}

#==============================================================================#
# PART B — Recovery Classifier (Phoenix vs Zombie)
#==============================================================================#

cat("\n[13] ══ PART B: Recovery Classifier ══\n")

## ── B1. Build the training dataset ──────────────────────────────────────────
##
## We want to predict: among firms that CRASH (p_m1 is high),
## which ones become confirmed zombies vs phoenixes?
##
## Label:
##   zombie  (y=1) = firm crashed AND confirmed CSI (y_base = 1)
##   phoenix (y=0) = firm crashed but recovered    (y_base = 0, high drawdown)
##
## Features: solvency features at year t (prediction year, before crash)

cat("  Building recovery classifier dataset...\n")

## Get the set of M1-flagged firms across all years (train + test)
## Using rank-based flagging: top EXCLUSION_RATE by p_m1 per year
all_m1 <- rbindlist(list(
  as.data.table(arrow::read_parquet(
    file.path(DIR_TABLES, "ag_fund", "ag_preds_test.parquet"))),
  as.data.table(arrow::read_parquet(
    file.path(DIR_TABLES, "ag_fund", "ag_preds_oos.parquet")))
))

## Also try to load CV predictions for in-sample training
m1_cv_path <- file.path(DIR_TABLES, "ag_fund", "ag_cv_results.parquet")
if (file.exists(m1_cv_path)) {
  m1_cv <- as.data.table(arrow::read_parquet(m1_cv_path))
  all_m1 <- rbindlist(list(m1_cv[, .(permno, year, p_csi)],
                           all_m1[, .(permno, year, p_csi)]),
                      use.names=TRUE)
  cat("  CV predictions included for in-sample training.\n")
}

## Rank-based flag per year
all_m1[, flag_m1 := {
  n_pred  <- sum(!is.na(p_csi))
  if (n_pred == 0L) rep(FALSE, .N)
  else {
    cutoff <- ceiling(n_pred * EXCLUSION_RATE)
    r <- frank(-p_csi, ties.method="first", na.last="keep")
    !is.na(r) & r <= cutoff
  }
}, by = year]

flagged_firms <- all_m1[flag_m1 == TRUE, .(permno, year, p_csi)]
cat(sprintf("  Total M1-flagged firm-years: %d\n", nrow(flagged_firms)))

## Join with base case labels to get actual outcome
## y_base at year t = did firm enter CSI in year t+1
flagged_with_label <- merge(
  flagged_firms,
  labels_base[, .(permno, year, y_base = y)],
  by = c("permno", "year"),
  all.x = TRUE
)

## We need to further identify "phoenixes" among non-CSI flagged firms
## A phoenix = flagged but did NOT become CSI (y_base = 0)
## A zombie  = flagged AND became CSI (y_base = 1)
## Exclude NA labels (censored / zombie window)
flagged_labelled <- flagged_with_label[!is.na(y_base)]

cat(sprintf("  Flagged firms with valid labels: %d\n", nrow(flagged_labelled)))
cat(sprintf("  Zombies (y=1): %d | Phoenixes (y=0): %d\n",
            sum(flagged_labelled$y_base == 1L),
            sum(flagged_labelled$y_base == 0L)))

## ── B2. Join solvency features ───────────────────────────────────────────────

cat("  Joining solvency features...\n")

## Check which solvency features actually exist in features_fund
available_features <- intersect(SOLVENCY_FEATURES, names(features))
missing_features   <- setdiff(SOLVENCY_FEATURES, names(features))

if (length(missing_features) > 0L)
  cat(sprintf("  Missing features (will skip): %s\n",
              paste(missing_features, collapse=", ")))

cat(sprintf("  Using %d of %d solvency features.\n",
            length(available_features), length(SOLVENCY_FEATURES)))

classifier_data <- merge(
  flagged_labelled[, .(permno, year, p_csi, y_zombie = y_base)],
  features[, c("permno","year", available_features), with=FALSE],
  by = c("permno","year"),
  all.x = TRUE
)

## Remove rows with all-NA features
n_before <- nrow(classifier_data)
feature_cols_avail <- intersect(available_features, names(classifier_data))
classifier_data <- classifier_data[
  rowSums(!is.na(classifier_data[, ..feature_cols_avail])) >= 3L
]
cat(sprintf("  Classifier data: %d rows (dropped %d all-NA)\n",
            nrow(classifier_data), n_before - nrow(classifier_data)))

## ── B3. Train/test split ─────────────────────────────────────────────────────

train_cls <- classifier_data[year <= TRAIN_END]
test_cls  <- classifier_data[year >= OOS_START & year <= OOS_END]

cat(sprintf("  Train: %d rows | %d zombies | %d phoenixes\n",
            nrow(train_cls),
            sum(train_cls$y_zombie == 1L),
            sum(train_cls$y_zombie == 0L)))
cat(sprintf("  Test:  %d rows | %d zombies | %d phoenixes\n",
            nrow(test_cls),
            sum(test_cls$y_zombie == 1L),
            sum(test_cls$y_zombie == 0L)))

if (nrow(train_cls) < 50L || sum(train_cls$y_zombie == 1L) < 10L) {
  cat("\n  WARNING: Insufficient training data for classifier.\n")
  cat("  This is expected if CV predictions are unavailable (train period has no M1 flags).\n")
  cat("  Recommendation: run 09C_AutoGluon.py Stage 2 to generate CV fold predictions,\n")
  cat("  or lower EXCLUSION_RATE to flag more firms in the available test/OOS period.\n")
  classifier_trained <- FALSE
} else {
  classifier_trained <- TRUE
}

if (classifier_trained) {
  
  ## ── B4. Fit decision tree ─────────────────────────────────────────────────
  
  cat("\n  Fitting recovery classifier (decision tree)...\n")
  
  ## Build formula from available features
  tree_formula <- as.formula(
    paste("factor(y_zombie) ~",
          paste(feature_cols_avail, collapse=" + "))
  )
  
  ## Class weights — zombies are minority class
  n_phx <- sum(train_cls$y_zombie == 0L)
  n_zmb <- sum(train_cls$y_zombie == 1L)
  class_wts <- c("0" = 1.0, "1" = n_phx / n_zmb)
  
  tree_model <- rpart(
    formula   = tree_formula,
    data      = train_cls,
    method    = "class",
    weights   = ifelse(train_cls$y_zombie == 1L,
                       class_wts["1"], class_wts["0"]),
    control   = rpart.control(
      maxdepth  = TREE_DEPTH,
      minbucket = MIN_BUCKET,
      cp        = 0.001      ## complexity parameter — prune later
    )
  )
  
  ## Prune to optimal CP (1-SE rule)
  cp_table  <- as.data.table(tree_model$cptable)
  opt_cp    <- cp_table[xerror == min(xerror), CP][1]
  tree_pruned <- prune(tree_model, cp = opt_cp)
  
  cat("\n  Tree structure (pruned):\n")
  print(tree_pruned)
  
  ## ── B5. Evaluate on test set ─────────────────────────────────────────────
  
  cat("\n  Evaluating on test set...\n")
  
  if (nrow(test_cls) > 0L && sum(test_cls$y_zombie == 1L) >= 5L) {
    
    test_proba <- predict(tree_pruned, newdata=test_cls, type="prob")[, "1"]
    test_cls[, p_zombie := test_proba]
    test_cls[, pred_zombie := as.integer(test_proba >= 0.5)]
    
    ## Confusion matrix
    tp <- sum(test_cls$pred_zombie == 1L & test_cls$y_zombie == 1L)
    fp <- sum(test_cls$pred_zombie == 1L & test_cls$y_zombie == 0L)
    fn <- sum(test_cls$pred_zombie == 0L & test_cls$y_zombie == 1L)
    tn <- sum(test_cls$pred_zombie == 0L & test_cls$y_zombie == 0L)
    
    recall    <- tp / (tp + fn)
    precision <- tp / (tp + fp)
    f1        <- 2 * recall * precision / (recall + precision)
    
    tree_ap  <- fn_avg_precision(test_cls$y_zombie, test_cls$p_zombie)
    tree_auc <- tryCatch(
      as.numeric(pROC::auc(pROC::roc(test_cls$y_zombie,
                                     test_cls$p_zombie, quiet=TRUE))),
      error = function(e) NA_real_)
    
    cat(sprintf("\n  Recovery Classifier — Test Set Performance:\n"))
    cat(sprintf("  AP=%.4f | AUC=%.4f | Recall=%.3f | Precision=%.3f | F1=%.3f\n",
                tree_ap, tree_auc, recall, precision, f1))
    cat(sprintf("  TP=%d FP=%d FN=%d TN=%d\n", tp, fp, fn, tn))
    cat(sprintf("  Interpretation: Of M1-flagged firms, %.0f%% correctly identified as zombie\n",
                recall*100))
    cat(sprintf("  False alarm rate among flagged: %.0f%% of phoenixes wrongly called zombie\n",
                fp/(fp+tn)*100))
    
  } else {
    cat("  Insufficient test observations for classifier evaluation.\n")
    test_cls[, p_zombie := NA_real_]
    test_cls[, pred_zombie := NA_integer_]
  }
  
  ## ── B6. Feature importance ────────────────────────────────────────────────
  
  cat("\n  Feature importance (Gini impurity):\n")
  
  feat_imp_tree <- as.data.table(tree_pruned$variable.importance,
                                 keep.rownames=TRUE)
  setnames(feat_imp_tree, c("feature","importance"))
  feat_imp_tree[, importance := importance / sum(importance)]
  feat_imp_tree <- feat_imp_tree[order(-importance)]
  print(feat_imp_tree, row.names=FALSE)
  
  p_feat_imp <- ggplot(feat_imp_tree,
                       aes(x=reorder(feature, importance),
                           y=importance)) +
    geom_col(fill="#2196F3", alpha=0.85, width=0.7) +
    coord_flip() +
    scale_y_continuous(labels=percent_format(accuracy=1),
                       name="Relative Importance (Gini)") +
    labs(title="Recovery Classifier — Feature Importance",
         subtitle="Features distinguishing Zombie from Phoenix firms",
         x=NULL) +
    theme_minimal(base_size=11) +
    theme(axis.text.y=element_text(size=9))
  
  ggsave(file.path(DIR_FIGURES, "robust_feature_importance.png"), p_feat_imp,
         width=PLOT_WIDTH, height=PLOT_HEIGHT, dpi=PLOT_DPI)
  cat("  robust_feature_importance.png saved.\n")
  
  ## ── B7. Tree visualisation ────────────────────────────────────────────────
  
  png(file.path(DIR_FIGURES, "robust_tree_plot.png"),
      width=PLOT_WIDTH*100, height=PLOT_HEIGHT*100, res=PLOT_DPI)
  rpart.plot(tree_pruned,
             type=4, extra=104,
             box.palette=list("tomato","steelblue"),
             main="Recovery Classifier: Zombie vs Phoenix Decision Tree",
             sub=paste0("Trained on M1-flagged firms | Train ≤", TRAIN_END,
                        " | Pruned depth=", tree_pruned$control$maxdepth))
  dev.off()
  cat("  robust_tree_plot.png saved.\n")
  
  ## ── B8. Phoenix vs Zombie feature distributions ───────────────────────────
  
  ## Show the top 2 splitting features distribution by outcome
  top2_features <- head(feat_imp_tree$feature, 2L)
  
  if (length(top2_features) >= 1L) {
    dist_dt <- classifier_data[!is.na(get(top2_features[1])),
                               .(feature_val = get(top2_features[1]),
                                 outcome = ifelse(y_zombie==1L,
                                                  "Zombie (CSI)", "Phoenix (Recovery)"),
                                 split = ifelse(year <= TRAIN_END,
                                                "Train", "OOS"))]
    
    p_dist <- ggplot(dist_dt,
                     aes(x=feature_val, fill=outcome, colour=outcome)) +
      geom_density(alpha=0.35, linewidth=0.7) +
      facet_wrap(~split) +
      scale_fill_manual(values=c("Zombie (CSI)"="tomato",
                                 "Phoenix (Recovery)"="steelblue")) +
      scale_colour_manual(values=c("Zombie (CSI)"="tomato",
                                   "Phoenix (Recovery)"="steelblue")) +
      labs(title=paste0("Top Splitting Feature: ", top2_features[1]),
           subtitle="Distribution among M1-flagged firms only",
           x=top2_features[1], y="Density",
           fill=NULL, colour=NULL) +
      theme_minimal(base_size=12) +
      theme(legend.position="bottom")
    
    ggsave(file.path(DIR_FIGURES, "robust_phoenix_zombie_dist.png"), p_dist,
           width=PLOT_WIDTH, height=PLOT_HEIGHT, dpi=PLOT_DPI)
    cat("  robust_phoenix_zombie_dist.png saved.\n")
  }
  
  ## Save classifier results
  saveRDS(list(
    model       = tree_pruned,
    feat_imp    = feat_imp_tree,
    train_data  = train_cls,
    test_data   = test_cls
  ), file.path(DIR_TABLES, "robust_recovery_classifier.rds"))
  cat("  robust_recovery_classifier.rds saved.\n")
  
} ## end if classifier_trained

#==============================================================================#
# PART C — Refined Index: S4 = M1 flag AND zombie classifier
#==============================================================================#

cat("\n[13] ══ PART C: Refined Index Backtest (S4) ══\n")

if (!classifier_trained) {
  cat("  Skipping Part C — classifier not trained.\n")
} else {
  
  ## ── C1. Apply classifier to all M1-flagged firms ─────────────────────────
  
  cat("  Applying recovery classifier to M1-flagged firms...\n")
  
  ## Get all flagged firms across all years
  all_flagged <- all_m1[flag_m1 == TRUE, .(permno, year, p_csi)]
  
  ## Join features for classifier prediction
  all_flagged_feat <- merge(
    all_flagged,
    features[, c("permno","year", feature_cols_avail), with=FALSE],
    by = c("permno","year"),
    all.x = TRUE
  )
  
  ## Predict zombie probability for all flagged firms
  all_flagged_feat[, p_zombie := tryCatch(
    predict(tree_pruned,
            newdata = .SD[, feature_cols_avail, with=FALSE],
            type="prob")[, "1"],
    error = function(e) rep(NA_real_, .N)
  )]
  
  ## S4: only exclude firms flagged by M1 AND classified as zombie
  ## Use 0.5 probability threshold on the recovery classifier
  S4_ZOMBIE_THRESH <- 0.5
  
  zombie_flags <- all_flagged_feat[p_zombie >= S4_ZOMBIE_THRESH,
                                   .(permno, year)]
  zombie_flags[, flag_s4 := TRUE]
  
  cat(sprintf("  M1 flagged: %d firm-years\n", nrow(all_flagged)))
  cat(sprintf("  S4 (M1 + zombie): %d firm-years (%.0f%% of M1 flags retained)\n",
              nrow(zombie_flags),
              100 * nrow(zombie_flags) / nrow(all_flagged)))
  
  ## ── C2. Build universe and apply S4 exclusion ────────────────────────────
  
  cat("  Building S4 portfolio weights...\n")
  
  ## Load universe from saved weights
  weights_bench <- readRDS(file.path(DIR_TABLES, "index_weights.rds"))
  bench_ew <- weights_bench[strategy=="bench" & weighting=="ew",
                            .(permno, port_year, mkvalt_dec)]
  
  ## S4 exclusions: prediction at year t → port_year t+1
  zombie_flags[, port_year := year + 1L]
  
  s4_incl <- bench_ew[!zombie_flags[, .(permno, port_year)],
                      on=c("permno","port_year")]
  s4_incl[, w := 1.0 / .N, by=port_year]
  
  cat(sprintf("  S4 avg firms per year: %.0f\n",
              s4_incl[, .N, by=port_year][, mean(N)]))
  
  ## ── C3. Compute S4 monthly returns ───────────────────────────────────────
  
  cat("  Computing S4 monthly returns...\n")
  
  s4_monthly <- merge(
    monthly[, .(permno, date, year, month, ret)],
    s4_incl[, .(permno, port_year, w)],
    by.x = c("permno","year"),
    by.y = c("permno","port_year"),
    all.y = FALSE
  )
  s4_monthly <- s4_monthly[!is.na(ret) & !is.na(w)]
  
  s4_ret <- s4_monthly[, .(
    port_ret   = sum(w * ret, na.rm=TRUE),
    n_holdings = .N,
    strategy   = "s4",
    weighting  = "ew"
  ), by=.(date, year, month)]
  
  ## Combine with existing benchmark and S1
  bench_ret <- port_returns[strategy=="bench" & weighting=="ew"]
  s1_ret    <- port_returns[strategy=="s1"    & weighting=="ew"]
  
  all_ret <- rbindlist(list(bench_ret, s1_ret,
                            s4_ret[, names(bench_ret), with=FALSE]),
                       use.names=TRUE)
  
  saveRDS(all_ret, file.path(DIR_TABLES, "robust_index_returns.rds"))
  
  ## ── C4. Performance comparison ────────────────────────────────────────────
  
  cat("\n  S4 performance vs Benchmark and S1:\n\n")
  
  periods <- list(
    full = c(1998L, OOS_END),
    oos  = c(OOS_START, OOS_END)
  )
  
  perf_rows <- list()
  for (strat in c("bench","s1","s4")) {
    ret_dt <- all_ret[strategy==strat]
    for (per_nm in names(periods)) {
      per <- periods[[per_nm]]
      ret_sub <- ret_dt[year >= per[1] & year <= per[2]]
      pf <- fn_performance(ret_sub$port_ret)
      if (is.null(pf)) next
      pf$strategy <- strat
      pf$period   <- per_nm
      perf_rows[[length(perf_rows)+1]] <- pf
    }
  }
  
  perf_s4 <- do.call(rbind, perf_rows)
  
  STRAT_LABELS_C <- c(
    bench = "Benchmark",
    s1    = "S1: M1 Only",
    s4    = "S4: M1 + Zombie Filter"
  )
  
  cat(sprintf("  %-24s | %6s | %6s | %6s | %7s\n",
              "Strategy", "CAGR", "Vol", "Sharpe", "MaxDD"))
  cat(sprintf("  %-24s | %6s | %6s | %6s | %7s\n",
              "------------------------","------","------","------","-------"))
  for (per_nm in c("oos","full")) {
    cat(sprintf("  -- Period: %s --\n", per_nm))
    for (s in c("bench","s1","s4")) {
      r <- perf_s4[perf_s4$strategy==s & perf_s4$period==per_nm, ]
      if (nrow(r)==0L) next
      cat(sprintf("  %-24s | %6.2f%% | %6.2f%% | %6.3f | %7.2f%%\n",
                  STRAT_LABELS_C[s],
                  r$cagr*100, r$vol*100, r$sharpe, r$max_dd*100))
    }
  }
  
  ## ── C5. Cumulative return plot: Benchmark vs S1 vs S4 ────────────────────
  
  plot_ret <- all_ret[strategy %in% c("bench","s1","s4")
  ][order(strategy, date)]
  plot_ret[, cum_idx := cumprod(1 + port_ret), by=strategy]
  plot_ret[, strat_label := STRAT_LABELS_C[strategy]]
  
  STRAT_COLOURS_C <- c(
    bench = "#9E9E9E",
    s1    = "#2196F3",
    s4    = "#4CAF50"
  )
  
  oos_date <- as.Date(sprintf("%d-01-01", OOS_START))
  
  p_s4 <- ggplot(plot_ret,
                 aes(x=date, y=cum_idx,
                     colour=strategy, group=strategy)) +
    geom_line(linewidth=0.9) +
    geom_vline(xintercept=as.numeric(oos_date),
               linetype="dashed", colour="grey40", linewidth=0.7) +
    annotate("text", x=oos_date,
             y=max(plot_ret$cum_idx, na.rm=TRUE)*0.95,
             label="OOS start\n(2016)", hjust=-0.1, size=3, colour="grey40") +
    scale_colour_manual(values=STRAT_COLOURS_C,
                        labels=STRAT_LABELS_C) +
    scale_y_continuous(labels=dollar_format(prefix="$"),
                       name="Portfolio Value ($1 invested)") +
    scale_x_date(date_breaks="2 years", date_labels="%Y") +
    labs(title="Benchmark vs S1 (M1 Only) vs S4 (M1 + Zombie Filter)",
         subtitle="S4 removes phoenix false positives from S1 exclusion list",
         x=NULL, colour="Strategy") +
    theme_minimal(base_size=12) +
    theme(legend.position="bottom",
          axis.text.x=element_text(angle=30, hjust=1))
  
  ggsave(file.path(DIR_FIGURES, "robust_cumulative_s4.png"), p_s4,
         width=PLOT_WIDTH*1.2, height=PLOT_HEIGHT, dpi=PLOT_DPI)
  cat("  robust_cumulative_s4.png saved.\n")
  
} ## end Part C

#==============================================================================#
# Final summary
#==============================================================================#

cat("\n[13] ══════════════════════════════════════════════════════\n")
cat("  ROBUSTNESS CHECK SUMMARY\n")
cat("  ══════════════════════════════════════════════════════\n\n")

cat("  PART A — CSI Grid Sensitivity:\n")
if (!is.null(grid_perf) && nrow(grid_perf) > 0L) {
  cat(sprintf("  %d grid combinations evaluated\n", nrow(grid_perf)))
  cat(sprintf("  AP range: [%.3f, %.3f] — base case: %.3f\n",
              min(grid_perf$ap, na.rm=TRUE),
              max(grid_perf$ap, na.rm=TRUE),
              grid_perf[grid_perf$param_id=="BASE", "ap"]))
  cat(sprintf("  Base case rank by AP: %d of %d\n",
              which(grid_perf$param_id=="BASE"),
              nrow(grid_perf)))
} else {
  cat("  No grid results available.\n")
}

cat("\n  PART B — Recovery Classifier:\n")
if (classifier_trained) {
  cat(sprintf("  Features used: %s\n",
              paste(feature_cols_avail, collapse=", ")))
  cat(sprintf("  Tree nodes: %d\n", nrow(tree_pruned$frame)))
  if (exists("recall"))
    cat(sprintf("  Test recall: %.1f%% | Precision: %.1f%% | F1: %.3f\n",
                recall*100, precision*100, f1))
} else {
  cat("  Classifier not trained — insufficient flagged firms in train period.\n")
  cat("  Rerun after generating CV fold predictions in 09C_AutoGluon.py.\n")
}

cat("\n  PART C — S4 Refined Index:\n")
if (classifier_trained && exists("perf_s4")) {
  s4_oos <- perf_s4[perf_s4$strategy=="s4" & perf_s4$period=="oos", ]
  s1_oos <- perf_s4[perf_s4$strategy=="s1" & perf_s4$period=="oos", ]
  bm_oos <- perf_s4[perf_s4$strategy=="bench" & perf_s4$period=="oos", ]
  if (nrow(s4_oos) > 0L && nrow(s1_oos) > 0L) {
    cat(sprintf("  S4 vs S1 OOS: CAGR %+.2f%% | Sharpe %+.3f\n",
                (s4_oos$cagr - s1_oos$cagr)*100,
                s4_oos$sharpe - s1_oos$sharpe))
    cat(sprintf("  S4 vs Bench OOS: CAGR %+.2f%% | Sharpe %+.3f\n",
                (s4_oos$cagr - bm_oos$cagr)*100,
                s4_oos$sharpe - bm_oos$sharpe))
  }
} else {
  cat("  S4 not computed.\n")
}


#==============================================================================#
# PART D — Tiered Threshold by Long-Run Return Bucket
#==============================================================================#
#
# MOTIVATION (from Table 1 analysis):
#   64.8% of imploded firms = "Value Destruction (temp. Recovery)" — negative
#   long-run CAGR but with interim bounce. These are the primary exclusion target.
#   14% of imploded firms are High/Moderate Growth (phoenixes) — excluding them
#   is pure cost. The binary CSI label treats both identically.
#
# APPROACH:
#   1. Compute post-CSI long-run CAGR for each firm in the universe
#   2. Assign to 3 return buckets:
#        Permanent loser : CAGR < -2%  → always exclude (low threshold)
#        Temporary loser : CAGR -2–0%  → exclude if confident (medium threshold)
#        Phoenix         : CAGR > 0%   → never exclude (infinite threshold)
#   3. Evaluate whether M1 scores are monotonically ordered across buckets
#   4. Apply tiered thresholds and backtest S5
#
#==============================================================================#

cat("\n[13] ══ PART D: Tiered Threshold by Return Bucket ══\n")

## ── D1. Compute long-run CAGR per firm ──────────────────────────────────────
##
## For each permno, compute geometric mean annual return over full available
## history in prices_monthly. This measures the firm's ultimate outcome.

cat("  Computing long-run CAGR per firm...\n")

## Require at least 3 years of returns for a reliable CAGR estimate
## Use full price history but require at least 3 years and cap at realistic bounds
## ret is already capped at [-0.99, 10] from the loading step
firm_cagr <- monthly[!is.na(ret), {
  n_months <- .N
  if (n_months < 36L) {
    .(cagr = NA_real_, n_months = n_months)
  } else {
    ## Geometric mean: (prod(1+r))^(12/n_months) - 1
    ## annualised from monthly returns
    cum_factor <- prod(1 + ret, na.rm=TRUE)
    ## Guard against extreme values from data errors
    if (!is.finite(cum_factor) || cum_factor <= 0) {
      .(cagr = NA_real_, n_months = n_months)
    } else {
      .(cagr    = cum_factor^(12/n_months) - 1,
        n_months = n_months)
    }
  }
}, by = permno]

firm_cagr <- firm_cagr[!is.na(cagr)]

cat(sprintf("  CAGR computed for %d firms\n", nrow(firm_cagr)))
cat(sprintf("  CAGR distribution: p10=%.1f%% | p25=%.1f%% | median=%.1f%% | p75=%.1f%% | p90=%.1f%%\n",
            quantile(firm_cagr$cagr, 0.10)*100,
            quantile(firm_cagr$cagr, 0.25)*100,
            quantile(firm_cagr$cagr, 0.50)*100,
            quantile(firm_cagr$cagr, 0.75)*100,
            quantile(firm_cagr$cagr, 0.90)*100))

## ── D2. Assign return buckets ────────────────────────────────────────────────

## Thresholds matching Table 1 categories
BUCKET_PERM_LOSER  <- -0.02   ## CAGR < -2%  → permanent loser
BUCKET_PHOENIX     <-  0.00   ## CAGR >  0%  → phoenix

firm_cagr[, return_bucket := fcase(
  cagr <  BUCKET_PERM_LOSER, "permanent_loser",   ## CAGR < -2%
  cagr >= BUCKET_PERM_LOSER &
    cagr < BUCKET_PHOENIX,   "temporary_loser",   ## -2% to 0%
  cagr >= BUCKET_PHOENIX,    "phoenix",            ## CAGR > 0%
  default                  = NA_character_
)]

bucket_counts <- firm_cagr[, .N, by=return_bucket][order(return_bucket)]
cat("\n  Return bucket distribution:\n")
print(bucket_counts, row.names=FALSE)

## ── D3. Join buckets to M1 predictions ──────────────────────────────────────

cat("\n  Joining return buckets to M1 predictions...\n")

## All M1 honest predictions (test + OOS)
m1_buckets <- merge(
  m1_preds[, .(permno, year, p_m1 = as.numeric(p_m1))],
  firm_cagr[, .(permno, cagr, return_bucket)],
  by = "permno",
  all.x = TRUE
)

## Also join actual CSI labels to confirm bucket alignment
m1_buckets <- merge(
  m1_buckets,
  labels_base[, .(permno, year, y_csi = y)],
  by = c("permno","year"),
  all.x = TRUE
)

cat(sprintf("  M1 predictions with bucket info: %d rows\n",
            sum(!is.na(m1_buckets$return_bucket))))

## ── D4. Check monotonicity: are M1 scores ordered across buckets? ────────────
##
## Key diagnostic: if M1 correctly identifies outcome severity,
## permanent losers should have highest p_m1, phoenixes lowest.

cat("\n  M1 score distribution by return bucket (OOS 2016–2022):\n\n")

oos_buckets <- m1_buckets[year >= OOS_START & year <= OOS_END & !is.na(return_bucket)]

bucket_scores <- oos_buckets[, .(
  n           = .N,
  mean_p_m1   = round(mean(p_m1, na.rm=TRUE), 4),
  median_p_m1 = round(median(p_m1, na.rm=TRUE), 4),
  p75_p_m1    = round(quantile(p_m1, 0.75, na.rm=TRUE), 4),
  p90_p_m1    = round(quantile(p_m1, 0.90, na.rm=TRUE), 4),
  pct_csi     = round(mean(y_csi == 1L, na.rm=TRUE)*100, 1)
), by = return_bucket][order(return_bucket)]

print(bucket_scores, row.names=FALSE)

## Monotonicity test: is mean_p_m1(permanent) > mean_p_m1(temporary) > mean_p_m1(phoenix)?
if (all(c("permanent_loser","temporary_loser","phoenix") %in% bucket_scores$return_bucket)) {
  p_perm <- bucket_scores[return_bucket=="permanent_loser", mean_p_m1]
  p_temp <- bucket_scores[return_bucket=="temporary_loser", mean_p_m1]
  p_phx  <- bucket_scores[return_bucket=="phoenix",         mean_p_m1]
  monotone <- p_perm > p_temp && p_temp > p_phx
  cat(sprintf("\n  Monotonicity check: perm=%.4f > temp=%.4f > phoenix=%.4f → %s\n",
              p_perm, p_temp, p_phx,
              if (monotone) "✓ MONOTONE" else "✗ NOT MONOTONE"))
}

## ── D5. Plot: M1 score density by return bucket ──────────────────────────────

bucket_colours <- c(
  permanent_loser = "#E53935",
  temporary_loser = "#FF9800",
  phoenix         = "#2196F3"
)
bucket_labels <- c(
  permanent_loser = "Permanent Loser (CAGR < −2%)",
  temporary_loser = "Temporary Loser (−2% to 0%)",
  phoenix         = "Phoenix (CAGR > 0%)"
)

p_bucket_scores <- ggplot(
  oos_buckets[!is.na(return_bucket)],
  aes(x=p_m1, fill=return_bucket, colour=return_bucket)
) +
  geom_density(alpha=0.30, linewidth=0.7) +
  scale_fill_manual(values=bucket_colours, labels=bucket_labels) +
  scale_colour_manual(values=bucket_colours, labels=bucket_labels) +
  scale_x_continuous(labels=percent_format(accuracy=1),
                     name="M1 Predicted CSI Probability") +
  labs(
    title    = "M1 Score Distribution by Long-Run Return Bucket",
    subtitle = "OOS 2016–2022 | Right shift = model correctly scores permanent losers higher",
    y        = "Density", fill=NULL, colour=NULL
  ) +
  theme_minimal(base_size=12) +
  theme(legend.position="bottom")

ggsave(file.path(DIR_FIGURES, "robust_bucket_score_dist.png"), p_bucket_scores,
       width=PLOT_WIDTH, height=PLOT_HEIGHT, dpi=PLOT_DPI)
cat("  robust_bucket_score_dist.png saved.\n")

## ── D6. Precision by bucket at different thresholds ──────────────────────────
##
## For each threshold τ, what fraction of flagged firms fall in each bucket?
## This directly answers: "does a higher p_m1 threshold select more permanent losers?"

cat("\n  Precision-by-bucket at varying thresholds:\n\n")

thresholds <- c(0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)

`%||%` <- function(a, b) if (length(a) == 0 || is.null(a)) b else a

thresh_rows <- lapply(thresholds, function(tau) {
  flagged <- oos_buckets[p_m1 >= tau & !is.na(return_bucket)]
  if (nrow(flagged) < 5L) return(NULL)
  tab <- flagged[, .N, by=return_bucket]
  tab[, pct := N / sum(N)]
  data.frame(
    threshold       = tau,
    n_flagged       = nrow(flagged),
    pct_perm_loser  = tab[return_bucket=="permanent_loser", pct] %||% 0,
    pct_temp_loser  = tab[return_bucket=="temporary_loser", pct] %||% 0,
    pct_phoenix     = tab[return_bucket=="phoenix",         pct] %||% 0
  )
})
thresh_table <- do.call(rbind, Filter(Negate(is.null), thresh_rows))
thresh_table[, c("pct_perm_loser","pct_temp_loser","pct_phoenix")] <-
  round(thresh_table[, c("pct_perm_loser","pct_temp_loser","pct_phoenix")] * 100, 1)

cat("  threshold | n_flagged | %perm_loser | %temp_loser | %phoenix\n")
cat("  ----------+-----------+-------------+-------------+---------\n")
for (i in seq_len(nrow(thresh_table))) {
  r <- thresh_table[i,]
  cat(sprintf("  %9.2f | %9d | %11.1f%% | %11.1f%% | %7.1f%%\n",
              r$threshold, r$n_flagged,
              r$pct_perm_loser, r$pct_temp_loser, r$pct_phoenix))
}

## Plot: stacked bar of bucket composition by threshold
thresh_long <- as.data.table(thresh_table) |>
  pivot_longer(cols=c(pct_perm_loser, pct_temp_loser, pct_phoenix),
               names_to="bucket", values_to="pct") |>
  mutate(bucket = recode(bucket,
                         pct_perm_loser = "permanent_loser",
                         pct_temp_loser = "temporary_loser",
                         pct_phoenix    = "phoenix"
  ))

p_thresh_bucket <- ggplot(thresh_long,
                          aes(x=factor(threshold), y=pct,
                              fill=bucket)) +
  geom_col(width=0.7, alpha=0.85) +
  scale_fill_manual(values=bucket_colours, labels=bucket_labels) +
  scale_y_continuous(labels=function(x) paste0(x, "%"),
                     name="% of flagged firms") +
  labs(
    title    = "Composition of Flagged Firms by M1 Threshold",
    subtitle = "Higher threshold → should enrich permanent losers if model is well-ordered",
    x        = "M1 Threshold (τ)",
    fill     = NULL
  ) +
  theme_minimal(base_size=12) +
  theme(legend.position="bottom")

ggsave(file.path(DIR_FIGURES, "robust_bucket_threshold_comp.png"), p_thresh_bucket,
       width=PLOT_WIDTH, height=PLOT_HEIGHT, dpi=PLOT_DPI)
cat("  robust_bucket_threshold_comp.png saved.\n")

## ── D7. Tiered threshold strategy S5 ────────────────────────────────────────
##
## Apply different exclusion rates by predicted bucket:
##   - Firms with very high M1 score (top 3%) → always exclude (likely permanent losers)
##   - Firms with medium M1 score (top 3–8%) → exclude only if altman_z2 < zombie threshold
##   - Firms with low M1 score → keep
##
## This implements the "higher confidence for ambiguous middle bucket" idea.

cat("\n  Building S5: tiered threshold strategy...\n")

TIER_HIGH_RATE <- 0.03   ## top 3% always excluded (permanent loser candidates)
TIER_MED_RATE  <- 0.08   ## top 3–8% excluded only if zombie solvency signal present
ZOMBIE_Z2_THRESH <- -2.768814  ## from decision tree split

## Load December mktcap for universe construction
dec_mktcap_s5 <- monthly[month == 12L & !is.na(mkvalt),
                         .(mkvalt_dec = {x <- mkvalt[!is.na(mkvalt)]; if(length(x)==0) NA_real_ else x[length(x)]}),
                         by = .(permno, year)]

universe_s5 <- dec_mktcap_s5[mkvalt_dec >= 100]
universe_s5[, rank_mkvalt := frank(-mkvalt_dec, ties.method="first"), by=year]
universe_s5 <- universe_s5[rank_mkvalt <= 3000L]

## Join M1 predictions and solvency feature
ann_s5 <- merge(
  universe_s5[, .(permno, year, mkvalt_dec)],
  m1_preds[, .(permno, year, p_m1 = as.numeric(p_m1))],
  by = c("permno","year"),
  all.x = TRUE
)

ann_s5 <- merge(
  ann_s5,
  features[, .(permno, year, altman_z2)],
  by = c("permno","year"),
  all.x = TRUE
)

## Tiered flags per year
ann_s5[, c("flag_high","flag_med") := {
  
  ## Tier 1: top TIER_HIGH_RATE by M1 score — always exclude
  n_pred   <- sum(!is.na(p_m1))
  cut_high <- ceiling(n_pred * TIER_HIGH_RATE)
  cut_med  <- ceiling(n_pred * TIER_MED_RATE)
  
  r <- frank(-p_m1, ties.method="first", na.last="keep")
  
  flag_h <- !is.na(r) & r <= cut_high
  
  ## Tier 2: top TIER_HIGH_RATE to TIER_MED_RATE AND zombie solvency signal
  flag_m <- !is.na(r) & r > cut_high & r <= cut_med &
    !is.na(altman_z2) & altman_z2 < ZOMBIE_Z2_THRESH
  
  list(flag_h, flag_m)
}, by = year]

ann_s5[, flag_s5  := flag_high | flag_med]
ann_s5[, incl_s5  := !flag_s5]
ann_s5[, port_year := year + 1L]

## Equal-weight portfolio
s5_incl <- ann_s5[incl_s5 == TRUE]
s5_incl[, w := 1.0 / .N, by=port_year]

cat(sprintf("  S5 avg firms/year: %.0f (vs 2992 for S4, 3000 for bench)\n",
            s5_incl[, .N, by=port_year][, mean(N)]))

## Monthly returns
s5_monthly <- merge(
  monthly[, .(permno, date, year, month, ret)],
  s5_incl[, .(permno, port_year, w)],
  by.x = c("permno","year"),
  by.y = c("permno","port_year"),
  all.y = FALSE
)
s5_monthly <- s5_monthly[!is.na(ret) & !is.na(w)]

s5_ret <- s5_monthly[, .(
  port_ret   = sum(w * ret, na.rm=TRUE),
  n_holdings = .N,
  strategy   = "s5",
  weighting  = "ew"
), by=.(date, year, month)]

## Combine all strategies for comparison
all_ret_d <- rbindlist(list(
  port_returns[strategy == "bench" & weighting == "ew"],
  port_returns[strategy == "s1"    & weighting == "ew"],
  s4_ret[, names(port_returns[strategy=="bench"&weighting=="ew"]), with=FALSE],
  s5_ret[, names(port_returns[strategy=="bench"&weighting=="ew"]), with=FALSE]
), use.names=TRUE)

## ── D8. Performance comparison ───────────────────────────────────────────────

cat("\n  S5 performance (OOS and Full):\n\n")

perf_d_rows <- list()
for (strat in c("bench","s1","s4","s5")) {
  ret_dt <- all_ret_d[strategy == strat]
  for (per_nm in c("oos","full")) {
    yr_range <- if (per_nm=="oos") c(OOS_START, OOS_END) else c(1998L, OOS_END)
    ret_sub  <- ret_dt[year >= yr_range[1] & year <= yr_range[2]]
    pf <- fn_performance(ret_sub$port_ret)
    if (is.null(pf)) next
    pf$strategy <- strat; pf$period <- per_nm
    perf_d_rows[[length(perf_d_rows)+1]] <- pf
  }
}
perf_d <- do.call(rbind, perf_d_rows)

STRAT_LABELS_D <- c(
  bench = "Benchmark",
  s1    = "S1: M1 Only (5%)",
  s4    = "S4: M1 + Zombie Filter",
  s5    = "S5: Tiered (3%/8% + solvency)"
)
STRAT_COLOURS_D <- c(
  bench = "#9E9E9E",
  s1    = "#2196F3",
  s4    = "#4CAF50",
  s5    = "#9C27B0"
)

cat(sprintf("  %-30s | %7s | %6s | %6s | %8s\n",
            "Strategy","CAGR","Vol","Sharpe","MaxDD"))
cat(sprintf("  %-30s | %7s | %6s | %6s | %8s\n",
            "------------------------------","-------","------","------","--------"))
for (per_nm in c("oos","full")) {
  cat(sprintf("  ── Period: %-4s ──\n", per_nm))
  for (s in c("bench","s1","s4","s5")) {
    r <- perf_d[perf_d$strategy==s & perf_d$period==per_nm, ]
    if (nrow(r)==0L) next
    cat(sprintf("  %-30s | %6.2f%% | %5.2f%% | %6.3f | %7.2f%%\n",
                STRAT_LABELS_D[s],
                r$cagr*100, r$vol*100, r$sharpe, r$max_dd*100))
  }
}

## ── D9. Cumulative return — all four strategies ───────────────────────────────

plot_d <- all_ret_d[strategy %in% c("bench","s1","s4","s5")][order(strategy,date)]
plot_d[, cum_idx := cumprod(1 + port_ret), by=strategy]

oos_date_d <- as.Date(sprintf("%d-01-01", OOS_START))

p_tiered <- ggplot(plot_d,
                   aes(x=date, y=cum_idx,
                       colour=strategy, group=strategy)) +
  geom_line(linewidth=0.9) +
  geom_vline(xintercept=as.numeric(oos_date_d),
             linetype="dashed", colour="grey40", linewidth=0.7) +
  annotate("text", x=oos_date_d,
           y=max(plot_d$cum_idx, na.rm=TRUE)*0.95,
           label="OOS\n(2016)", hjust=-0.1, size=3, colour="grey40") +
  scale_colour_manual(values=STRAT_COLOURS_D, labels=STRAT_LABELS_D) +
  scale_y_continuous(labels=dollar_format(prefix="$"),
                     name="Portfolio Value ($1 invested)") +
  scale_x_date(date_breaks="2 years", date_labels="%Y") +
  labs(
    title    = "Strategy Comparison: Benchmark / S1 / S4 / S5 (Tiered)",
    subtitle = paste0("S5 = top 3% always excluded + top 3–8% excluded if altman_z2 < ",
                      round(ZOMBIE_Z2_THRESH, 2)),
    x        = NULL, colour = "Strategy"
  ) +
  theme_minimal(base_size=12) +
  theme(legend.position="bottom",
        axis.text.x=element_text(angle=30, hjust=1))

ggsave(file.path(DIR_FIGURES, "robust_cumulative_tiered.png"), p_tiered,
       width=PLOT_WIDTH*1.2, height=PLOT_HEIGHT, dpi=PLOT_DPI)
cat("  robust_cumulative_tiered.png saved.\n")

## Save results
saveRDS(list(
  firm_cagr     = firm_cagr,
  bucket_scores = bucket_scores,
  thresh_table  = thresh_table,
  perf_d        = perf_d
), file.path(DIR_TABLES, "robust_tiered_results.rds"))
cat("  robust_tiered_results.rds saved.\n")

## ── D10. Console summary ─────────────────────────────────────────────────────

cat("\n[13] ══ PART D SUMMARY ══\n\n")

cat("  Return bucket distribution (all universe firms):\n")
print(bucket_counts, row.names=FALSE)

cat("\n  Key diagnostic — monotonicity of M1 scores across buckets:\n")
print(bucket_scores[, .(return_bucket, mean_p_m1, pct_csi)], row.names=FALSE)

if (exists("monotone"))
  cat(sprintf("  → Monotonicity: %s\n",
              if (monotone) "CONFIRMED — higher p_m1 predicts worse long-run outcome"
              else "NOT CONFIRMED — M1 scores do not reliably separate return buckets"))

cat("\n  S5 vs S1 improvement (OOS):\n")
s5_r <- perf_d[perf_d$strategy=="s5" & perf_d$period=="oos",]
s1_r <- perf_d[perf_d$strategy=="s1" & perf_d$period=="oos",]
bm_r <- perf_d[perf_d$strategy=="bench" & perf_d$period=="oos",]
if (nrow(s5_r)>0 && nrow(s1_r)>0)
  cat(sprintf("  CAGR: S5 %+.2f%% vs S1 | Sharpe: S5 %+.3f vs S1\n",
              (s5_r$cagr-s1_r$cagr)*100, s5_r$sharpe-s1_r$sharpe))
if (nrow(s5_r)>0 && nrow(bm_r)>0)
  cat(sprintf("  CAGR: S5 %+.2f%% vs Bench | Sharpe: S5 %+.3f vs Bench\n",
              (s5_r$cagr-bm_r$cagr)*100, s5_r$sharpe-bm_r$sharpe))


#==============================================================================#
# PART E — Concentrated Long Portfolio (C1 / C2 / C3)
#==============================================================================#
#
# MOTIVATION:
#   S1-S5 exclude 110-600 firms from 3,000 → each position ~0.03% weight.
#   Maximum achievable alpha ~3.7%/yr. Signal is diluted by construction.
#
#   Solution: build a CONCENTRATED portfolio selecting the lowest-risk firms
#   by B1 score (within-year decile 1 = 9.3% terminal loser rate vs 31.5%
#   base rate). Each position gets 0.5-1.0% weight → signal actually matters.
#
# STRATEGIES:
#   C1 : Bottom 200 firms by B1 score (within-year rank) — EW, ~0.5% each
#   C2 : Bottom 100 firms by B1 score (tightest screen)  — EW, ~1.0% each
#   C3 : C1 + M1 veto — remove any C1 firm in top 10% by M1 score
#        (two-layer: B1 selects structural quality, M1 removes crash risk)
#
# ALSO REPORTS:
#   S6 : Broad exclusion — exclude top 20% by B1 score from 3,000-firm universe
#        Tests whether B1 adds value in the standard index exclusion context
#
# INPUTS:
#   DIR_TABLES/ag_bucket/ag_preds_test.parquet  — B1 test predictions
#   DIR_TABLES/ag_fund/ag_preds_test.parquet    — M1 test predictions (for C3 veto)
#   DIR_TABLES/index_returns.rds                — benchmark + S1 returns
#   prices_monthly.rds                          — monthly returns
#
# OUTPUTS:
#   DIR_TABLES/robust_conc_returns.rds          — monthly returns C1/C2/C3/S6
#   DIR_TABLES/robust_conc_performance.rds      — performance table
#   DIR_FIGURES/13_robustness/partE/            — all plots
#
#==============================================================================#

cat("\n[13] ══ PART E: Concentrated Long Portfolio ══\n")

## ── E0. Parameters ───────────────────────────────────────────────────────────

C1_SIZE          <- 200L   ## bottom 200 by B1 score
C2_SIZE          <- 100L   ## bottom 100 by B1 score
C3_M1_VETO_RATE  <- 0.10   ## veto C1 firms in top 10% by M1 score
S6_EXCL_RATE     <- 0.20   ## exclude top 20% by B1 score (broad index)

## Figure output directory
FIGS_E <- file.path(DIR_FIGURES, "13_robustness", "partE")
dir.create(FIGS_E, recursive=TRUE, showWarnings=FALSE)

## ── E1. Load B1 and M1 predictions ──────────────────────────────────────────

cat("  Loading B1 predictions...\n")

b1_test <- as.data.table(arrow::read_parquet(
  file.path(DIR_TABLES, "ag_bucket", "ag_preds_test.parquet")))
setnames(b1_test, "p_csi", "p_b1")
setnames(b1_test, "y",     "y_loser")

## Also load M1 for the C3 veto — test + OOS predictions
m1_all <- rbindlist(list(
  as.data.table(arrow::read_parquet(
    file.path(DIR_TABLES, "ag_fund", "ag_preds_test.parquet"))),
  as.data.table(arrow::read_parquet(
    file.path(DIR_TABLES, "ag_fund", "ag_preds_oos.parquet")))
))
setnames(m1_all, "p_csi", "p_m1")

cat(sprintf("  B1 predictions: %d rows | years %d-%d\n",
            nrow(b1_test), min(b1_test$year), max(b1_test$year)))
cat(sprintf("  M1 predictions: %d rows | years %d-%d\n",
            nrow(m1_all), min(m1_all$year), max(m1_all$year)))

## ── E2. Build universe (same as 11_Results.R) ────────────────────────────────

cat("  Building universe...\n")

monthly_e <- as.data.table(readRDS(PATH_PRICES_MONTHLY))
setnames(monthly_e, "ret_adj", "ret")
setnames(monthly_e, "mktcap",  "mkvalt")
monthly_e[, year  := year(date)]
monthly_e[, month := month(date)]
monthly_e[, ret   := pmin(pmax(ret, -0.99, na.rm=TRUE), 10, na.rm=TRUE)]

dec_mktcap_e <- monthly_e[month == 12L & !is.na(mkvalt),
                          .(mkvalt_dec = mkvalt[.N]),
                          by=.(permno, year)]
universe_e <- dec_mktcap_e[mkvalt_dec >= 100]
universe_e[, rank_mkvalt := frank(-mkvalt_dec, ties.method="first"), by=year]
universe_e <- universe_e[rank_mkvalt <= 3000L]

## ── E3. Rank firms within each year by B1 and M1 score ──────────────────────

cat("  Computing within-year ranks...\n")

## Join B1 to universe
ann_e <- merge(
  universe_e[, .(permno, year, mkvalt_dec)],
  b1_test[, .(permno, year, p_b1, y_loser)],
  by=c("permno","year"),
  all.x=TRUE
)

## Join M1
ann_e <- merge(
  ann_e,
  m1_all[, .(permno, year, p_m1)],
  by=c("permno","year"),
  all.x=TRUE
)

## Within-year ranks — only computed for years with B1 predictions
## Years without predictions: all flags default to FALSE (no inclusion in C1/C2/C3)
ann_e[, n_pred_b1 := sum(!is.na(p_b1)), by=year]

## B1 ascending rank (rank 1 = safest) — NA-safe: rank only within rows with predictions
ann_e[, rank_b1_asc := NA_integer_]
ann_e[!is.na(p_b1),
      rank_b1_asc := frank(p_b1, ties.method="first"),
      by=year]
ann_e[, rank_b1_asc_pct := fifelse(!is.na(rank_b1_asc),
                                   rank_b1_asc / n_pred_b1,
                                   NA_real_)]

## B1 descending rank (rank 1 = riskiest) — for S6 exclusion
ann_e[, rank_b1_desc := NA_integer_]
ann_e[!is.na(p_b1),
      rank_b1_desc := frank(-p_b1, ties.method="first"),
      by=year]
ann_e[, rank_b1_desc_pct := fifelse(!is.na(rank_b1_desc),
                                    rank_b1_desc / n_pred_b1,
                                    NA_real_)]

## M1 descending rank (rank 1 = highest crash risk) — NA-safe
ann_e[, n_pred_m1 := sum(!is.na(p_m1)), by=year]
ann_e[, rank_m1_desc := NA_integer_]
ann_e[!is.na(p_m1),
      rank_m1_desc := frank(-p_m1, ties.method="first"),
      by=year]
ann_e[, rank_m1_desc_pct := fifelse(!is.na(rank_m1_desc),
                                    rank_m1_desc / n_pred_m1,
                                    NA_real_)]

## ── E4. Assign inclusion flags ───────────────────────────────────────────────

## C1: bottom 200 by B1 within each prediction year
ann_e[, incl_c1 := !is.na(rank_b1_asc) & rank_b1_asc <= C1_SIZE]

## C2: bottom 100 by B1
ann_e[, incl_c2 := !is.na(rank_b1_asc) & rank_b1_asc <= C2_SIZE]

## C3: C1 minus M1 top-10% veto — only veto if M1 rank exists
ann_e[, m1_vetoed := !is.na(rank_m1_desc_pct) & rank_m1_desc_pct <= C3_M1_VETO_RATE]
ann_e[, incl_c3 := incl_c1 == TRUE & m1_vetoed == FALSE]

## S6: exclude top 20% by B1 — firms with no B1 prediction are kept (conservative)
ann_e[, incl_s6 := is.na(rank_b1_desc_pct) | rank_b1_desc_pct > S6_EXCL_RATE]

## Portfolio year = year + 1
ann_e[, port_year := year + 1L]

## Summary
cat("\n  Strategy composition per year (mean):\n")
cat(sprintf("    C1: %.0f firms | C2: %.0f firms | C3: %.0f firms | S6: %.0f firms\n",
            ann_e[incl_c1==TRUE, .N, by=year][, mean(N)],
            ann_e[incl_c2==TRUE, .N, by=year][, mean(N)],
            ann_e[incl_c3==TRUE, .N, by=year][, mean(N)],
            ann_e[incl_s6==TRUE, .N, by=year][, mean(N)]))

## Show within-year decile quality of selected firms
cat("\n  B1 within-year decile composition of C1 selected firms:\n")
ann_e[incl_c1==TRUE & !is.na(p_b1), .(
  mean_rank_pct = round(mean(rank_b1_asc_pct)*100, 1),
  pct_loser     = round(mean(y_loser==1L, na.rm=TRUE)*100, 1)
)] |> print()

## ── E5. Build weights and compute returns ────────────────────────────────────

cat("\n  Computing monthly returns for all strategies...\n")

fn_strat_returns <- function(ann_dt, incl_col, strat_name) {
  
  incl <- ann_dt[get(incl_col) == TRUE]
  incl[, w := 1.0 / .N, by=port_year]
  
  port_monthly <- merge(
    monthly_e[, .(permno, date, year, month, ret)],
    incl[, .(permno, port_year, w)],
    by.x=c("permno","year"),
    by.y=c("permno","port_year"),
    all.y=FALSE
  )
  port_monthly <- port_monthly[!is.na(ret) & !is.na(w)]
  
  port_ret <- port_monthly[, .(
    port_ret   = sum(w * ret, na.rm=TRUE),
    n_holdings = .N,
    strategy   = strat_name,
    weighting  = "ew"
  ), by=.(date, year, month)]
  
  port_ret
}

ret_c1 <- fn_strat_returns(ann_e, "incl_c1", "c1")
ret_c2 <- fn_strat_returns(ann_e, "incl_c2", "c2")
ret_c3 <- fn_strat_returns(ann_e, "incl_c3", "c3")
ret_s6 <- fn_strat_returns(ann_e, "incl_s6", "s6")

## Load existing benchmark and S1/S4 returns
bench_e <- port_returns[strategy=="bench" & weighting=="ew"]
s1_e    <- port_returns[strategy=="s1"    & weighting=="ew"]
s4_ret_e <- readRDS(file.path(DIR_TABLES, "robust_index_returns.rds"))[
  strategy=="s4" & weighting=="ew"]

all_ret_e <- rbindlist(list(
  bench_e, s1_e, s4_ret_e,
  ret_c1[, names(bench_e), with=FALSE],
  ret_c2[, names(bench_e), with=FALSE],
  ret_c3[, names(bench_e), with=FALSE],
  ret_s6[, names(bench_e), with=FALSE]
), use.names=TRUE)

saveRDS(all_ret_e, file.path(DIR_TABLES, "robust_conc_returns.rds"))

## ── E6. Performance metrics ──────────────────────────────────────────────────

cat("  Computing performance...\n")

STRAT_LABELS_E <- c(
  bench = "Benchmark (EW 3000)",
  s1    = "S1: M1 Exclusion (5%)",
  s4    = "S4: M1 + Zombie Filter",
  s6    = "S6: B1 Exclusion (20%)",
  c1    = "C1: B1 Long 200",
  c2    = "C2: B1 Long 100",
  c3    = "C3: B1 Long 200 + M1 Veto"
)

STRAT_COLOURS_E <- c(
  bench = "#9E9E9E",
  s1    = "#2196F3",
  s4    = "#4CAF50",
  s6    = "#FF9800",
  c1    = "#9C27B0",
  c2    = "#E91E63",
  c3    = "#F44336"
)

## B1 predictions only available 2016-2019 → test period only
## But we can still compute full-period returns for C1/C2/C3
## NOTE: pre-2016 years have no B1 predictions → C1/C2/C3 will use
## whatever firms happened to rank lowest in the feature space
## Report both periods and note the honest OOS period is 2016-2019

periods_e <- list(
  oos  = c(2016L, 2019L),   ## honest OOS — B1 predictions exist
  full = c(1998L, 2022L)    ## full backtest — pre-2016 uses model extrapolation
)

perf_e_rows <- list()
for (strat in c("bench","s1","s4","s6","c1","c2","c3")) {
  ret_dt <- all_ret_e[strategy==strat]
  for (per_nm in names(periods_e)) {
    per     <- periods_e[[per_nm]]
    ret_sub <- ret_dt[year >= per[1] & year <= per[2]]
    pf      <- fn_performance(ret_sub$port_ret)
    if (is.null(pf)) next
    pf$strategy <- strat
    pf$period   <- per_nm
    perf_e_rows[[length(perf_e_rows)+1]] <- pf
  }
}
perf_e <- do.call(rbind, perf_e_rows)
saveRDS(perf_e, file.path(DIR_TABLES, "robust_conc_performance.rds"))

## Print performance table
for (per_nm in c("oos","full")) {
  cat(sprintf("\n  ── Period: %s ──\n", per_nm))
  if (per_nm == "oos")
    cat("  (Honest OOS: B1 predictions exist for all firms)\n")
  else
    cat("  (Full period: pre-2016 uses model extrapolation — interpret cautiously)\n")
  cat(sprintf("  %-28s | %6s | %6s | %7s | %7s | %6s\n",
              "Strategy","CAGR","Vol","Sharpe","MaxDD","Calmar"))
  cat(sprintf("  %-28s | %6s | %6s | %7s | %7s | %6s\n",
              paste(rep("-",28),collapse=""),
              "------","------","-------","-------","------"))
  for (s in c("bench","s1","s4","s6","c1","c2","c3")) {
    r <- perf_e[perf_e$strategy==s & perf_e$period==per_nm,]
    if (nrow(r)==0L) next
    marker <- if (s %in% c("c1","c2","c3")) " ◄" else ""
    cat(sprintf("  %-28s | %5.2f%% | %5.2f%% | %7.3f | %7.2f%% | %6.3f%s\n",
                STRAT_LABELS_E[s],
                r$cagr*100, r$vol*100, r$sharpe,
                r$max_dd*100, r$calmar, marker))
  }
}

## ── E7. Sector concentration check ──────────────────────────────────────────

cat("\n  Checking sector concentration of C1 portfolio...\n")

## Use SIC codes from features if available
if ("sic" %in% names(features) || "sic_code" %in% names(features)) {
  sic_col <- if ("sic" %in% names(features)) "sic" else "sic_code"
  sic_dt  <- features[, .(permno, year, sic = get(sic_col))]
  sic_dt[, sector := fcase(
    sic >= 100  & sic <= 999,   "Agriculture",
    sic >= 1000 & sic <= 1499,  "Mining",
    sic >= 1500 & sic <= 1999,  "Construction",
    sic >= 2000 & sic <= 3999,  "Manufacturing",
    sic >= 4000 & sic <= 4999,  "Transport/Utilities",
    sic >= 5000 & sic <= 5199,  "Wholesale",
    sic >= 5200 & sic <= 5999,  "Retail",
    sic >= 6000 & sic <= 6799,  "Finance",
    sic >= 7000 & sic <= 8999,  "Services",
    sic >= 9000 & sic <= 9999,  "Public",
    default = "Other"
  )]
  
  c1_sector <- merge(
    ann_e[incl_c1==TRUE, .(permno, year, port_year)],
    sic_dt,
    by=c("permno","year"),
    all.x=TRUE
  )
  
  cat("\n  C1 sector composition (% of selected firms per year, mean):\n")
  sector_comp <- c1_sector[!is.na(sector), .(n=.N), by=.(year, sector)]
  sector_comp[, pct := n/sum(n)*100, by=year]
  sector_mean <- sector_comp[, .(mean_pct=round(mean(pct),1)), by=sector
  ][order(-mean_pct)]
  print(sector_mean, row.names=FALSE)
  
  if (sector_mean[1, mean_pct] > 50)
    cat("  WARNING: Top sector > 50% — C1 may be a sector tilt\n")
  else
    cat("  Sector concentration appears reasonable\n")
} else {
  cat("  SIC codes not available in features — skipping sector check\n")
  cat("  Consider joining from Compustat fundamentals\n")
}

## ── E8. Plots ────────────────────────────────────────────────────────────────

cat("\n  Generating Part E plots...\n")

## Plot E1: Cumulative returns — OOS only (honest)
oos_ret <- all_ret_e[year >= 2016 & year <= 2019 &
                       strategy %in% c("bench","s1","c1","c2","c3")]
oos_ret <- oos_ret[order(strategy, date)]
oos_ret[, cum_idx := cumprod(1 + port_ret), by=strategy]
oos_ret[, label := STRAT_LABELS_E[strategy]]

p_conc_oos <- ggplot(oos_ret,
                     aes(x=date, y=cum_idx,
                         colour=strategy, group=strategy)) +
  geom_line(linewidth=0.9) +
  scale_colour_manual(values=STRAT_COLOURS_E, labels=STRAT_LABELS_E) +
  scale_y_continuous(labels=dollar_format(prefix="$"),
                     name="Portfolio Value ($1 invested)") +
  scale_x_date(date_breaks="6 months", date_labels="%Y-%m") +
  labs(
    title    = "Concentrated Portfolio vs Benchmark — Honest OOS (2016-2019)",
    subtitle = "C1/C2/C3 use B1 predictions available for all firms in this period",
    x=NULL, colour="Strategy"
  ) +
  theme_minimal(base_size=12) +
  theme(legend.position="bottom",
        axis.text.x=element_text(angle=30, hjust=1))

ggsave(file.path(FIGS_E, "conc_cumulative_oos.png"), p_conc_oos,
       width=PLOT_WIDTH*1.2, height=PLOT_HEIGHT, dpi=PLOT_DPI)
cat("  conc_cumulative_oos.png saved.\n")

## Plot E2: Cumulative returns — full period (all strategies)
full_ret <- all_ret_e[year >= 1998 & year <= 2022 &
                        strategy %in% c("bench","s1","s4","s6","c1","c3")]
full_ret <- full_ret[order(strategy, date)]
full_ret[, cum_idx := cumprod(1 + port_ret), by=strategy]

oos_date_e <- as.Date("2016-01-01")

p_conc_full <- ggplot(full_ret,
                      aes(x=date, y=cum_idx,
                          colour=strategy, group=strategy)) +
  geom_line(linewidth=0.85) +
  geom_vline(xintercept=as.numeric(oos_date_e),
             linetype="dashed", colour="grey40", linewidth=0.7) +
  annotate("text", x=oos_date_e,
           y=max(full_ret$cum_idx, na.rm=TRUE)*0.92,
           label="B1 OOS→", hjust=-0.05, size=3, colour="grey40") +
  scale_colour_manual(values=STRAT_COLOURS_E, labels=STRAT_LABELS_E) +
  scale_y_continuous(labels=dollar_format(prefix="$"),
                     name="Portfolio Value ($1 invested)") +
  scale_x_date(date_breaks="2 years", date_labels="%Y") +
  labs(
    title    = "All Strategies — Full Period (1998-2022)",
    subtitle = "Dashed = start of B1 honest OOS period | Pre-2016 C1/C2/C3 use model extrapolation",
    x=NULL, colour="Strategy"
  ) +
  theme_minimal(base_size=12) +
  theme(legend.position="bottom",
        axis.text.x=element_text(angle=30, hjust=1))

ggsave(file.path(FIGS_E, "conc_cumulative_full.png"), p_conc_full,
       width=PLOT_WIDTH*1.2, height=PLOT_HEIGHT, dpi=PLOT_DPI)
cat("  conc_cumulative_full.png saved.\n")

## Plot E3: Annual returns bar — OOS
ann_ret_e <- all_ret_e[year >= 2016 & year <= 2019 &
                         strategy %in% c("bench","c1","c3","s6"),
                       .(ann_ret=prod(1+port_ret)-1),
                       by=.(strategy, year)]

p_ann_e <- ggplot(ann_ret_e,
                  aes(x=year, y=ann_ret, fill=strategy)) +
  geom_col(position=position_dodge(0.8), width=0.7, alpha=0.85) +
  geom_hline(yintercept=0, colour="black", linewidth=0.3) +
  scale_fill_manual(values=STRAT_COLOURS_E, labels=STRAT_LABELS_E) +
  scale_y_continuous(labels=percent_format(accuracy=1)) +
  labs(title="Annual Returns — OOS Period (2016-2019)",
       x=NULL, y="Annual Return", fill="Strategy") +
  theme_minimal(base_size=11) +
  theme(legend.position="bottom")

ggsave(file.path(FIGS_E, "conc_annual_returns.png"), p_ann_e,
       width=PLOT_WIDTH, height=PLOT_HEIGHT, dpi=PLOT_DPI)
cat("  conc_annual_returns.png saved.\n")

## Plot E4: Drawdown — full period
dd_e <- full_ret[strategy %in% c("bench","c1","c3")][order(strategy,date)]
dd_e[, cum_idx  := cumprod(1+port_ret), by=strategy]
dd_e[, peak     := cummax(cum_idx),      by=strategy]
dd_e[, drawdown := (cum_idx-peak)/peak]

p_dd_e <- ggplot(dd_e, aes(x=date, y=drawdown,
                           colour=strategy, group=strategy)) +
  geom_line(linewidth=0.8) +
  geom_hline(yintercept=0, colour="black", linewidth=0.3) +
  geom_vline(xintercept=as.numeric(oos_date_e),
             linetype="dashed", colour="grey40", linewidth=0.7) +
  scale_colour_manual(values=STRAT_COLOURS_E, labels=STRAT_LABELS_E) +
  scale_y_continuous(labels=percent_format(accuracy=1), name="Drawdown") +
  scale_x_date(date_breaks="2 years", date_labels="%Y") +
  labs(title="Drawdown — Concentrated Portfolios vs Benchmark",
       subtitle="Dashed = B1 OOS start",
       x=NULL, colour="Strategy") +
  theme_minimal(base_size=12) +
  theme(legend.position="bottom",
        axis.text.x=element_text(angle=30, hjust=1))

ggsave(file.path(FIGS_E, "conc_drawdown.png"), p_dd_e,
       width=PLOT_WIDTH*1.2, height=PLOT_HEIGHT, dpi=PLOT_DPI)
cat("  conc_drawdown.png saved.\n")

## ── E9. Console summary ──────────────────────────────────────────────────────

cat("\n[13] ══ PART E SUMMARY ══\n\n")

cat("  Decile validation (within-year B1 ranking):\n")
cat("    Decile 1 (C1 pool):  9.3% terminal losers\n")
cat("    Decile 10 (excluded): 71.4% terminal losers\n")
cat("    Ratio: 7.7x — year_cat does NOT distort within-year rankings\n")

cat("\n  OOS performance highlights:\n")
for (s in c("bench","s1","c1","c2","c3","s6")) {
  r <- perf_e[perf_e$strategy==s & perf_e$period=="oos",]
  if (nrow(r)==0L) next
  bm <- perf_e[perf_e$strategy=="bench" & perf_e$period=="oos",]
  cat(sprintf("    %-28s : CAGR %+.2f%% vs bench | Sharpe %+.3f vs bench\n",
              STRAT_LABELS_E[s],
              (r$cagr - bm$cagr)*100,
              r$sharpe - bm$sharpe))
}

cat("\n  Key question: does concentration solve the dilution problem?\n")
c1_oos <- perf_e[perf_e$strategy=="c1" & perf_e$period=="oos",]
bm_oos <- perf_e[perf_e$strategy=="bench" & perf_e$period=="oos",]
if (nrow(c1_oos) > 0 && nrow(bm_oos) > 0) {
  if (c1_oos$cagr > bm_oos$cagr) {
    cat(sprintf("  ✓ YES — C1 outperforms benchmark by %+.2f%% CAGR in OOS\n",
                (c1_oos$cagr - bm_oos$cagr)*100))
  } else {
    cat(sprintf("  ✗ NO  — C1 underperforms benchmark by %.2f%% CAGR in OOS\n",
                abs(c1_oos$cagr - bm_oos$cagr)*100))
    cat("    Signal exists (7.7x decile ratio) but alpha does not materialise\n")
    cat("    Possible causes: sector concentration, tracking error, bull market regime\n")
  }
}

cat(sprintf("\n[13_Robustness.R] DONE: %s\n", format(Sys.time())))

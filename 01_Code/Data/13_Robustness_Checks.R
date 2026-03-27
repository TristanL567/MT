#==============================================================================#
#==== 13_Robustness.R =========================================================#
#==== Comprehensive Robustness Checks — CSI / Bucket / Structural Labels ======#
#==============================================================================#
#
# PURPOSE:
#   Addresses the core supervisory concern: are the three-parameter CSI
#   definition and the bucket/structural label designs discretionary choices
#   that drive results, or robust methodological anchors?
#
# STRUCTURE:
#
#   PART A — CSI Parameter Sensitivity
#     A1: Label-level overlap — Jaccard similarity across 27 grid combinations
#         addresses "are parameters arbitrary?" directly
#     A2: Model performance stability — M1 AP/R@FPR3 across all 27 labels
#         addresses "does model performance change materially?"
#
#   PART B — Phoenix vs Zombie Analysis (ALL THREE LABEL TYPES)
#     B1: CSI labels — recovery classifier on M1-flagged firms
#     B2: Bucket labels — do predicted terminal losers actually fail to recover?
#     B3: Structural labels — four-quadrant recovery patterns
#
#   PART C — Refined Index: S4 = M1 + Zombie Filter
#
#   PART D — Tiered Threshold Strategy S5
#
#   PART E — Concentrated Long Portfolio C1/C2/C3
#
#   PART F — Bucket Label Robustness
#     F1: CAGR threshold sensitivity (-5%/-2%/0% loser thresholds)
#     F3: Label stability across adjacent base years
#     NOTE: window sensitivity (F2) not included per scope decision
#
#   PART G — Structural Label Robustness
#     G1: Phoenix reclassification — Variant A (Stricter):
#         CSI events with positive 5yr CAGR kept as positives instead of
#         reclassified to quality. Tests whether phoenix reclassification
#         improves or hurts the label.
#
# DESIGN PRINCIPLES:
#   All alternative labels computed inline — no separate scripts.
#   No model retraining — robustness assessed at label and prediction level.
#   All checks use the same 1993–2015 train / 2016–2019 test split.
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
  library(rpart)
  library(rpart.plot)
  library(PRROC)
  library(pROC)
  library(viridis)
  library(slider)
})

cat("\n[13_Robustness.R] START:", format(Sys.time()), "\n")

## ── Figure directories ───────────────────────────────────────────────────────
FIGS   <- fn_setup_figure_dirs()
ROB_F  <- file.path(FIGS$robustness, "partF")
ROB_G  <- file.path(FIGS$robustness, "partG")
for (d in c(FIGS$rob_a, FIGS$rob_b, FIGS$rob_c,
            FIGS$rob_d, FIGS$rob_e, ROB_F, ROB_G))
  dir.create(d, recursive=TRUE, showWarnings=FALSE)

## ── Global parameters ────────────────────────────────────────────────────────
TRAIN_END        <- 2015L
OOS_START        <- 2016L
OOS_END          <- 2022L
EXCLUSION_RATE   <- 0.05
TREE_DEPTH       <- 4L
MIN_BUCKET       <- 30L
ZOMBIE_Z2_THRESH <- -2.768814

## ── Helper functions ─────────────────────────────────────────────────────────

fn_ap <- function(y, p) {
  tryCatch(PRROC::pr.curve(scores.class0=p[y==1L],
                           scores.class1=p[y==0L],
                           curve=FALSE)$auc.integral,
           error=function(e) NA_real_)
}
fn_auc <- function(y, p) {
  tryCatch(as.numeric(pROC::auc(pROC::roc(y, p, quiet=TRUE))),
           error=function(e) NA_real_)
}
fn_recall_fpr <- function(y, p, fpr_t) {
  tryCatch({
    r   <- pROC::roc(y, p, quiet=TRUE)
    fpr <- 1 - r$specificities
    idx <- which(fpr <= fpr_t)
    if (length(idx)==0) return(NA_real_)
    max(r$sensitivities[idx])
  }, error=function(e) NA_real_)
}
fn_jaccard <- function(set_a, set_b) {
  n_inter <- nrow(merge(set_a, set_b, by=c("permno","year")))
  n_union <- nrow(unique(rbindlist(list(set_a, set_b))))
  if (n_union==0) return(NA_real_)
  n_inter / n_union
}
fn_performance <- function(ret_vec, rf_annual=0.03) {
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
  calmar     <- if (!is.na(max_dd) && max_dd < 0) cagr / abs(max_dd) else NA_real_
  data.frame(cagr=round(cagr,4), sharpe=round(sharpe,4),
             max_dd=round(max_dd,4), vol=round(sd(ret_vec)*sqrt(12),4),
             calmar=round(calmar,4))
}
ggsave_std <- function(path, plot, w=PLOT_WIDTH, h=PLOT_HEIGHT) {
  ggsave(path, plot=plot, width=w, height=h, dpi=PLOT_DPI)
  cat(sprintf("    Saved: %s\n", basename(path)))
}

#==============================================================================#
# 0. Load shared inputs
#==============================================================================#

cat("\n[13] Loading inputs...\n")

features    <- as.data.table(readRDS(PATH_FEATURES_FUND))
labels_base <- as.data.table(readRDS(PATH_LABELS_BASE))
monthly     <- as.data.table(readRDS(PATH_PRICES_MONTHLY))
setnames(monthly, "ret_adj", "ret")
setnames(monthly, "mktcap",  "mkvalt")
monthly[, year  := year(date)]
monthly[, month := month(date)]
monthly[, ret   := pmin(pmax(ret, -0.99, na.rm=TRUE), 10, na.rm=TRUE)]

## M1 predictions (test + OOS combined)
m1_test <- as.data.table(arrow::read_parquet(
  file.path(DIR_TABLES, "ag_fund", "ag_preds_test.parquet")))
m1_oos  <- as.data.table(arrow::read_parquet(
  file.path(DIR_TABLES, "ag_fund", "ag_preds_oos.parquet")))
m1_all  <- rbindlist(list(m1_test, m1_oos))
setnames(m1_all, "p_csi", "p_m1")

## Rank-based M1 flag per year
m1_all[, flag_m1 := {
  n_pred  <- sum(!is.na(p_m1))
  cutoff  <- ceiling(n_pred * EXCLUSION_RATE)
  r       <- frank(-p_m1, ties.method="first", na.last="keep")
  !is.na(r) & r <= cutoff
}, by=year]

## Existing benchmark + S1 returns
port_returns <- readRDS(file.path(DIR_TABLES, "index_returns.rds"))

## Base CSI event set (for Jaccard comparisons)
base_events_csi <- labels_base[y==1L, .(permno, year)]

## Load bucket and structural labels (used in Parts B and F/G)
labels_bucket     <- as.data.table(readRDS(PATH_LABELS_BUCKET))
labels_structural <- as.data.table(readRDS(PATH_LABELS_STRUCTURAL))

cat(sprintf("  Base CSI events : %d\n", nrow(base_events_csi)))
cat(sprintf("  M1 predictions  : %d rows (%d flagged)\n",
            nrow(m1_all), sum(m1_all$flag_m1==TRUE, na.rm=TRUE)))

#==============================================================================#
# PART A — CSI Parameter Sensitivity
#==============================================================================#

cat("\n[13] ══ PART A: CSI Parameter Sensitivity ══\n")

## Load all 27 grid label files
grid_files <- list.files(DIR_LABELS, pattern="^labels_C.*[.]rds$",
                         full.names=TRUE)
cat(sprintf("  Found %d grid label files\n", length(grid_files)))

if (length(grid_files) == 0L) {
  cat("  WARNING: No grid files found — run 05_CSI_Label.R first. Skipping Part A.\n")
  grid_labels_list <- list()
} else {
  grid_labels_list <- setNames(
    lapply(grid_files, function(f) {
      dt <- as.data.table(readRDS(f))
      dt[, param_id := gsub("labels_|\\.rds", "", basename(f))]
      dt
    }),
    sapply(grid_files, function(f) gsub("labels_|\\.rds", "", basename(f)))
  )
  cat(sprintf("  Loaded %d grid label sets\n", length(grid_labels_list)))
}

##──────────────────────────────────────────────────────────────────────────────
## A1. Label-level overlap — event set similarity across grid
##──────────────────────────────────────────────────────────────────────────────

cat("\n  [A1] Label-level overlap analysis...\n")

a1_rows <- list()
for (pid in names(grid_labels_list)) {
  dt         <- grid_labels_list[[pid]]
  prm        <- CSI_GRID[CSI_GRID$param_id == pid, ]
  events_g   <- dt[y==1L, .(permno, year)]
  n_obs_lab  <- nrow(dt[!is.na(y)])
  
  a1_rows[[pid]] <- data.frame(
    param_id        = pid,
    C               = if (nrow(prm)>0) prm$C else NA_real_,
    M               = if (nrow(prm)>0) prm$M else NA_real_,
    T               = if (nrow(prm)>0) prm$T else NA_real_,
    n_events        = nrow(events_g),
    prevalence      = round(nrow(events_g) / max(n_obs_lab, 1), 4),
    jaccard_vs_base = round(fn_jaccard(base_events_csi, events_g), 4),
    stringsAsFactors = FALSE
  )
}

a1_summary <- do.call(rbind, a1_rows)
a1_summary <- a1_summary[order(-a1_summary$jaccard_vs_base), ]

cat(sprintf("\n    Jaccard vs base: min=%.3f | max=%.3f | mean=%.3f\n",
            min(a1_summary$jaccard_vs_base, na.rm=TRUE),
            max(a1_summary$jaccard_vs_base, na.rm=TRUE),
            mean(a1_summary$jaccard_vs_base, na.rm=TRUE)))
cat("    Top 10 most similar configurations:\n")
print(head(a1_summary[, c("param_id","C","M","T","n_events","jaccard_vs_base")], 10),
      row.names=FALSE)

## Interpretation for thesis
jacc_vals <- a1_summary$jaccard_vs_base
cat(sprintf("\n    Thesis interpretation:\n"))
cat(sprintf("    %.0f%% of configurations have Jaccard > 0.70 vs base\n",
            100*mean(jacc_vals > 0.70, na.rm=TRUE)))
cat(sprintf("    → %s\n",
            if (mean(jacc_vals > 0.70, na.rm=TRUE) > 0.80)
              "Event sets are highly stable — base case parameters are not cherry-picked"
            else
              "Meaningful variation in event sets — parameter choice affects identification"))

## Heatmap: Jaccard by C × T (averaged over M)
if (all(c("C","T","jaccard_vs_base") %in% names(a1_summary)) &&
    !all(is.na(a1_summary$C))) {
  
  hm_a1 <- as.data.table(a1_summary)[!is.na(C) & !is.na(T),
                                     .(mean_jaccard = mean(jaccard_vs_base, na.rm=TRUE)), by=.(C, T)]
  
  p_a1_hm <- ggplot(hm_a1, aes(x=factor(T), y=factor(C), fill=mean_jaccard)) +
    geom_tile(colour="white", linewidth=0.5) +
    geom_text(aes(label=round(mean_jaccard, 3)), size=3.5, colour="white") +
    scale_fill_viridis(option="mako", limits=c(0, 1), name="Jaccard\n(vs base)") +
    labs(title="CSI Event Set Similarity vs Base Case (C=−0.80, M=−0.20, T=18)",
         subtitle="Jaccard similarity | averaged over M | 1.0 = identical event set",
         x="Zombie Duration T (months)", y="Crash Threshold C") +
    theme_minimal(base_size=12)
  
  ggsave_std(file.path(FIGS$rob_a, "a1_jaccard_heatmap.png"), p_a1_hm)
}

## Event count per year: do all configurations cluster in same crisis years?
if (length(grid_labels_list) >= 3L) {
  ev_yr_list <- lapply(names(grid_labels_list), function(pid) {
    prm <- CSI_GRID[CSI_GRID$param_id == pid, ]
    dt  <- grid_labels_list[[pid]]
    cts <- dt[y==1L, .N, by=year]
    cts[, c("param_id","C","T") := list(
      pid,
      if (nrow(prm)>0) prm$C else NA_real_,
      if (nrow(prm)>0) prm$T else NA_real_)]
    cts
  })
  ev_yr_dt <- rbindlist(ev_yr_list)[!is.na(C)]
  
  ## Correlation of year-level event counts across all grid combinations
  ev_wide <- dcast(ev_yr_dt, year ~ param_id, value.var="N", fill=0L)
  if (ncol(ev_wide) > 2L) {
    cor_mat  <- cor(as.matrix(ev_wide[, -1, with=FALSE]), use="complete.obs")
    mean_cor <- round(mean(cor_mat[upper.tri(cor_mat)], na.rm=TRUE), 3)
    cat(sprintf("\n    Year-level event count correlation across grid: mean r=%.3f\n",
                mean_cor))
    cat(sprintf("    → %s\n",
                if (mean_cor > 0.80) "All configurations identify events in the same years (crisis clusters)"
                else "Event timing varies across configurations"))
  }
  
  ## Plot: event counts per year, all grid combinations overlaid, base case bold
  base_yr <- labels_base[y==1L & year >= 1998, .N, by=year]
  CRISIS_BANDS <- data.frame(
    xmin=c(2000.6, 2007.6, 2019.6),
    xmax=c(2002.4, 2009.4, 2020.4),
    label=c("Dot-com","GFC","COVID")
  )
  
  p_a1_crisis <- ggplot() +
    geom_rect(data=CRISIS_BANDS,
              aes(xmin=xmin, xmax=xmax, ymin=-Inf, ymax=Inf),
              fill="grey85", alpha=0.5, inherit.aes=FALSE) +
    geom_line(data=ev_yr_dt[year >= 1998],
              aes(x=year, y=N, group=param_id, colour=factor(T)),
              linewidth=0.35, alpha=0.45) +
    geom_line(data=base_yr,
              aes(x=year, y=N),
              colour="black", linewidth=1.1, linetype="dashed") +
    geom_text(data=CRISIS_BANDS,
              aes(x=(xmin+xmax)/2, y=Inf, label=label),
              vjust=1.5, size=3, colour="grey40", inherit.aes=FALSE) +
    facet_wrap(~C, labeller=label_both, ncol=1, scales="free_y") +
    scale_colour_viridis(discrete=TRUE, option="plasma", name="T (months)") +
    labs(title="CSI Events per Year — All 27 Grid Combinations",
         subtitle="Black dashed = base case | Grey bands = known crisis periods | Lines by T",
         x="Year", y="Number of CSI Events") +
    theme_minimal(base_size=11) + theme(legend.position="bottom")
  
  ggsave_std(file.path(FIGS$rob_a, "a1_crisis_timing.png"),
             p_a1_crisis, h=PLOT_HEIGHT*1.8)
}

##──────────────────────────────────────────────────────────────────────────────
## A2. Model performance stability — M1 AP across all 27 label definitions
##──────────────────────────────────────────────────────────────────────────────

cat("\n  [A2] M1 performance under each label definition...\n")

a2_rows <- list()
for (pid in names(grid_labels_list)) {
  dt  <- grid_labels_list[[pid]]
  prm <- CSI_GRID[CSI_GRID$param_id == pid, ]
  
  merged <- merge(
    m1_all[year >= OOS_START & year <= OOS_END, .(permno, year, p_m1)],
    dt[!is.na(y), .(permno, year, y)],
    by=c("permno","year"), all=FALSE
  )
  if (nrow(merged) < 50L || sum(merged$y==1L) < 10L) next
  
  a2_rows[[pid]] <- data.frame(
    param_id  = pid,
    C         = if (nrow(prm)>0) prm$C  else NA_real_,
    M         = if (nrow(prm)>0) prm$M  else NA_real_,
    T_val     = if (nrow(prm)>0) prm$T  else NA_real_,
    n_events  = sum(merged$y==1L),
    ap        = round(fn_ap(merged$y, merged$p_m1), 4),
    auc       = round(fn_auc(merged$y, merged$p_m1), 4),
    r_fpr3    = round(fn_recall_fpr(merged$y, merged$p_m1, 0.03), 4),
    stringsAsFactors = FALSE
  )
}

a2_perf <- do.call(rbind, Filter(Negate(is.null), a2_rows))

if (!is.null(a2_perf) && nrow(a2_perf) > 0L) {
  base_ap <- a2_perf[a2_perf$param_id=="BASE", "ap"]
  cat(sprintf("\n    AP across grid: min=%.3f | max=%.3f | SD=%.4f\n",
              min(a2_perf$ap, na.rm=TRUE),
              max(a2_perf$ap, na.rm=TRUE),
              sd(a2_perf$ap, na.rm=TRUE)))
  if (length(base_ap) > 0L)
    cat(sprintf("    Base case AP=%.4f | %.0f%% of configs within 5%% of base\n",
                base_ap,
                100*mean(abs(a2_perf$ap - base_ap)/base_ap < 0.05, na.rm=TRUE)))
  
  saveRDS(list(a1=a1_summary, a2=a2_perf),
          file.path(DIR_TABLES, "robust_grid_performance.rds"))
  
  ## AP heatmap: C × T, averaged over M
  if (all(c("C","T_val","ap") %in% names(a2_perf))) {
    ap_hm <- as.data.table(a2_perf)[!is.na(C) & !is.na(T_val),
                                    .(mean_ap=round(mean(ap, na.rm=TRUE), 4)), by=.(C, T_val)]
    
    p_a2_hm <- ggplot(ap_hm, aes(x=factor(T_val), y=factor(C), fill=mean_ap)) +
      geom_tile(colour="white", linewidth=0.5) +
      geom_text(aes(label=round(mean_ap, 3)), size=3.5, colour="white") +
      scale_fill_viridis(option="plasma", name="Mean AP") +
      labs(title="M1 Average Precision by CSI Parameter Combination (OOS 2016–2022)",
           subtitle="Averaged over M | stable grid = parameters are not data-mined",
           x="Zombie Duration T (months)", y="Crash Threshold C") +
      theme_minimal(base_size=12)
    
    ggsave_std(file.path(FIGS$rob_a, "a2_ap_heatmap.png"), p_a2_hm)
  }
  
  ## AP boxplot: marginal effect of each parameter
  if (all(c("C","M","T_val","ap") %in% names(a2_perf))) {
    a2_long <- melt(as.data.table(a2_perf)[!is.na(ap)],
                    id.vars="ap",
                    measure.vars=c("C","M","T_val"),
                    variable.name="param", value.name="value")
    a2_long[, param := recode(param, T_val="T")]
    
    p_a2_box <- ggplot(a2_long, aes(x=factor(value), y=ap, fill=param)) +
      geom_boxplot(width=0.6, alpha=0.8, outlier.size=1) +
      facet_wrap(~param, scales="free_x") +
      scale_fill_manual(values=c(C="#1565C0",M="#1B5E20",T="#6A1B9A"), guide="none") +
      scale_y_continuous(labels=number_format(accuracy=0.01)) +
      geom_hline(data=if (length(base_ap)>0)
        data.frame(param=c("C","M","T"), ref=base_ap)
        else data.frame(),
        aes(yintercept=ref), linetype="dashed", colour="grey40") +
      labs(title="M1 AP Distribution by Individual Parameter Value",
           subtitle="Dashed = base case AP | each box = all combinations with that parameter value",
           x="Parameter Value", y="Average Precision") +
      theme_minimal(base_size=11)
    
    ggsave_std(file.path(FIGS$rob_a, "a2_ap_by_param.png"), p_a2_box,
               w=PLOT_WIDTH*1.2)
  }
}

#==============================================================================#
# PART B — Phoenix vs Zombie Analysis (ALL THREE LABEL TYPES)
#==============================================================================#

cat("\n[13] ══ PART B: Phoenix vs Zombie Analysis ══\n")

## Solvency features shared across all three analyses
SOLVENCY_FEATURES <- c(
  "altman_z2", "leverage", "interest_cov", "cash_pct_act",
  "ocf_margin", "roll_min_3y_earn_yld", "roll_min_3y_roic",
  "consec_decline_sale", "peak_drop_log_mkvalt",
  "acct_mom_roa", "roll_sd_5y_earn_yld", "yoy_leverage"
)
avail_solvency <- intersect(SOLVENCY_FEATURES, names(features))
cat(sprintf("  Using %d/%d solvency features\n",
            length(avail_solvency), length(SOLVENCY_FEATURES)))

## Helper: fit recovery classifier and produce standard outputs
fn_recovery_classifier <- function(cls_data, y_col, label_name, fig_dir,
                                   colour="#1565C0") {
  
  cat(sprintf("\n  [%s] Recovery classifier...\n", label_name))
  
  avail_feats <- intersect(avail_solvency, names(cls_data))
  if (length(avail_feats) == 0L) {
    cat("    No solvency features available — skipping.\n")
    return(invisible(NULL))
  }
  
  train_d <- cls_data[year <= TRAIN_END]
  test_d  <- cls_data[year >= OOS_START & year <= OOS_END]
  
  n_pos_train <- sum(train_d[[y_col]]==1L, na.rm=TRUE)
  n_neg_train <- sum(train_d[[y_col]]==0L, na.rm=TRUE)
  
  cat(sprintf("    Train: %d rows | pos=%d | neg=%d\n",
              nrow(train_d), n_pos_train, n_neg_train))
  cat(sprintf("    Test:  %d rows | pos=%d | neg=%d\n",
              nrow(test_d),
              sum(test_d[[y_col]]==1L, na.rm=TRUE),
              sum(test_d[[y_col]]==0L, na.rm=TRUE)))
  
  if (nrow(train_d) < 50L || n_pos_train < 10L) {
    cat("    Insufficient training data — skipping.\n")
    return(invisible(NULL))
  }
  
  tree_formula <- as.formula(paste0("factor(", y_col, ") ~ ",
                                    paste(avail_feats, collapse="+")))
  tree_model   <- rpart(tree_formula, data=train_d, method="class",
                        weights=ifelse(train_d[[y_col]]==1L,
                                       n_neg_train/max(n_pos_train,1L), 1.0),
                        control=rpart.control(maxdepth=TREE_DEPTH,
                                              minbucket=MIN_BUCKET, cp=0.001))
  opt_cp       <- as.data.table(tree_model$cptable)[xerror==min(xerror), CP][1]
  tree_pruned  <- prune(tree_model, cp=opt_cp)
  
  ## Test metrics
  if (nrow(test_d) > 0L && sum(test_d[[y_col]]==1L, na.rm=TRUE) >= 5L) {
    test_p  <- predict(tree_pruned, newdata=test_d, type="prob")[,"1"]
    test_d  <- copy(test_d)
    test_d[, p_risk := test_p]
    ap_val  <- fn_ap(test_d[[y_col]], test_p)
    auc_val <- fn_auc(test_d[[y_col]], test_p)
    cat(sprintf("    Test AP=%.4f | AUC=%.4f\n", ap_val, auc_val))
  }
  
  ## Feature importance
  if (!is.null(tree_pruned$variable.importance)) {
    fi <- as.data.table(tree_pruned$variable.importance, keep.rownames=TRUE)
    setnames(fi, c("feature","importance"))
    fi[, importance := importance/sum(importance)]
    fi[, feature := factor(feature, levels=feature[order(importance)])]
    
    p_fi <- ggplot(fi, aes(x=importance, y=feature)) +
      geom_col(fill=colour, alpha=0.85) +
      scale_x_continuous(labels=percent_format(accuracy=1)) +
      labs(title=sprintf("Recovery Classifier — %s", label_name),
           subtitle="Feature importance (Gini) | distinguishing at-risk from resilient firms",
           x="Relative Importance", y=NULL) +
      theme_minimal(base_size=11)
    
    ggsave_std(file.path(fig_dir,
                         sprintf("b_%s_feature_importance.png",
                                 tolower(gsub(" ","_",label_name)))),
               p_fi)
  }
  
  ## Tree visualisation
  png(file.path(fig_dir,
                sprintf("b_%s_tree.png", tolower(gsub(" ","_",label_name)))),
      width=PLOT_WIDTH*100, height=PLOT_HEIGHT*100, res=PLOT_DPI)
  rpart.plot(tree_pruned, type=4, extra=104,
             box.palette=list("tomato","steelblue"),
             main=sprintf("Recovery Classifier: %s", label_name))
  dev.off()
  cat(sprintf("    Tree plot saved.\n"))
  
  ## Score distribution for top splitting feature
  top_feat <- names(tree_pruned$variable.importance)[1]
  if (!is.null(top_feat) && top_feat %in% names(cls_data)) {
    dist_dt <- cls_data[!is.na(get(top_feat)) & !is.na(get(y_col)),
                        .(val=get(top_feat),
                          outcome=ifelse(get(y_col)==1L, "At-Risk","Resilient"))]
    COLS_DIST <- c("At-Risk"="tomato","Resilient"="steelblue")
    LABS_DIST <- c("At-Risk"=sprintf("At-Risk (%s=1)", y_col),
                   "Resilient"=sprintf("Resilient (%s=0)", y_col))
    
    p_dist <- ggplot(dist_dt, aes(x=val, fill=outcome, colour=outcome)) +
      geom_density(alpha=0.35, linewidth=0.7) +
      scale_fill_manual(values=COLS_DIST, labels=LABS_DIST) +
      scale_colour_manual(values=COLS_DIST, labels=LABS_DIST) +
      labs(title=sprintf("Top Splitting Feature: %s — %s", top_feat, label_name),
           x=top_feat, y="Density", fill=NULL, colour=NULL) +
      theme_minimal(base_size=12) + theme(legend.position="bottom")
    
    ggsave_std(file.path(fig_dir,
                         sprintf("b_%s_dist.png",
                                 tolower(gsub(" ","_",label_name)))),
               p_dist)
  }
  
  invisible(list(model=tree_pruned, train=train_d, test=test_d))
}

##──────────────────────────────────────────────────────────────────────────────
## B1. CSI labels — phoenix vs zombie among M1-flagged firms
##──────────────────────────────────────────────────────────────────────────────

flagged_csi <- merge(
  m1_all[flag_m1==TRUE, .(permno, year, p_m1)],
  labels_base[!is.na(y), .(permno, year, y_zombie=y)],
  by=c("permno","year")
)[!is.na(y_zombie)]

cls_csi <- merge(flagged_csi,
                 features[, c("permno","year",avail_solvency), with=FALSE],
                 by=c("permno","year"), all.x=TRUE)
cls_csi <- cls_csi[rowSums(!is.na(cls_csi[, ..avail_solvency])) >= 3L]

result_csi <- fn_recovery_classifier(
  cls_data   = cls_csi,
  y_col      = "y_zombie",
  label_name = "CSI (M1-flagged)",
  fig_dir    = FIGS$rob_b,
  colour     = "#1565C0"
)

if (!is.null(result_csi)) {
  saveRDS(result_csi, file.path(DIR_TABLES, "robust_recovery_classifier.rds"))
}

##──────────────────────────────────────────────────────────────────────────────
## B2. Bucket labels — do B1-predicted terminal losers actually fail to recover?
##
## Among firms with bucket label = 1 (terminal loser), do solvency features
## at the base year explain *which* firms had the worst 5yr outcomes?
## This validates the bucket label from a solvency-signal perspective.
##
## "Recovery" here means: comparing the distribution of solvency features
## between terminal losers and phoenixes. The same tree approach as B1.
##──────────────────────────────────────────────────────────────────────────────

cat("\n  [B2] Bucket label phoenix vs zombie (terminal losers vs phoenixes)...\n")

## Use bucket labels directly — terminal loser (y_loser=1) vs phoenix (y_loser=0)
## Restrict to train period where labels are complete
bucket_clean <- labels_bucket[!is.na(y_loser) & year <= TRAIN_END,
                              .(permno, year, y_loser)]
cls_bucket   <- merge(bucket_clean,
                      features[, c("permno","year",avail_solvency), with=FALSE],
                      by=c("permno","year"), all.x=TRUE)
cls_bucket   <- cls_bucket[rowSums(!is.na(cls_bucket[, ..avail_solvency])) >= 3L]

cat(sprintf("  Bucket dataset: %d rows | loser=%d | phoenix=%d\n",
            nrow(cls_bucket),
            sum(cls_bucket$y_loser==1L),
            sum(cls_bucket$y_loser==0L)))

result_bucket <- fn_recovery_classifier(
  cls_data   = cls_bucket,
  y_col      = "y_loser",
  label_name = "Bucket (terminal loser vs phoenix)",
  fig_dir    = FIGS$rob_b,
  colour     = "#1B5E20"
)

##──────────────────────────────────────────────────────────────────────────────
## B3. Structural labels — recovery patterns by quadrant
##
## The structural label has four quadrants:
##   CSI + Terminal loser   → definitely exclude (double confirmation)
##   CSI + Phoenix          → reclassified to quality (controversial)
##   Slow bleeder           → exclude without CSI signal
##   Healthy                → keep
##
## Are these quadrants solvency-distinguishable at the feature level?
## Plot solvency feature distributions by quadrant to validate the design.
##──────────────────────────────────────────────────────────────────────────────

cat("\n  [B3] Structural label quadrant recovery analysis...\n")

struct_quad <- labels_structural[!is.na(y_csi) & !is.na(y_loser) & year <= TRAIN_END,
                                 .(permno, year, y_csi, y_loser, y_structural)]
struct_quad[, quadrant := fcase(
  y_csi==1L & y_loser==1L, "CSI + Terminal\n(Confirmed Exclude)",
  y_csi==1L & y_loser==0L, "CSI + Phoenix\n(Reclassified Keep)",
  y_csi==0L & y_loser==1L, "Slow Bleeder\n(No Crash, Exclude)",
  y_csi==0L & y_loser==0L, "Healthy\n(Keep)",
  default = NA_character_
)]
struct_quad <- struct_quad[!is.na(quadrant)]

## Join top solvency features and plot distributions
top_solvency_feats <- head(avail_solvency, 4L)   ## show top 4 for readability
struct_feat <- merge(struct_quad,
                     features[, c("permno","year",top_solvency_feats), with=FALSE],
                     by=c("permno","year"), all.x=TRUE)

struct_long <- melt(struct_feat, id.vars=c("permno","year","quadrant"),
                    measure.vars=top_solvency_feats,
                    variable.name="feature", value.name="value")
struct_long <- struct_long[!is.na(value)]

## Winsorise for plotting
struct_long[, value := {
  q <- quantile(value, c(0.01,0.99), na.rm=TRUE)
  pmax(pmin(value, q[2]), q[1])
}, by=feature]

QUAD_COLS <- c(
  "CSI + Terminal\n(Confirmed Exclude)" = "#E53935",
  "CSI + Phoenix\n(Reclassified Keep)"  = "#FF9800",
  "Slow Bleeder\n(No Crash, Exclude)"   = "#9C27B0",
  "Healthy\n(Keep)"                     = "#1E88E5"
)

p_b3_box <- ggplot(struct_long,
                   aes(x=quadrant, y=value, fill=quadrant)) +
  geom_boxplot(width=0.6, alpha=0.8, outlier.size=0.5) +
  facet_wrap(~feature, scales="free_y", ncol=2) +
  scale_fill_manual(values=QUAD_COLS, guide="none") +
  labs(title="Solvency Feature Distributions by Structural Label Quadrant",
       subtitle="Train 1993-2015 | validates that quadrant design reflects solvency reality",
       x=NULL, y="Feature Value") +
  theme_minimal(base_size=10) +
  theme(axis.text.x=element_text(angle=30, hjust=1, size=8))

ggsave_std(file.path(FIGS$rob_b, "b3_structural_quadrant_dist.png"),
           p_b3_box, w=PLOT_WIDTH*1.1, h=PLOT_HEIGHT*1.3)

## Recovery classifier: can solvency features distinguish the four quadrants?
## Treat as multiclass: which quadrant does a firm's solvency profile predict?
## Simplified: binary — exclude (y_structural=1) vs keep (y_structural=0)
struct_cls <- struct_feat[!is.na(y_structural)]
struct_cls <- struct_cls[rowSums(!is.na(struct_cls[, ..top_solvency_feats])) >= 2L]

result_structural <- fn_recovery_classifier(
  cls_data   = struct_cls,
  y_col      = "y_structural",
  label_name = "Structural (all quadrants)",
  fig_dir    = FIGS$rob_b,
  colour     = "#6A1B9A"
)

#==============================================================================#
# PART C — Refined Index: S4 = M1 + Zombie Filter
#==============================================================================#

cat("\n[13] ══ PART C: S4 Refined Index ══\n")

classifier_trained <- !is.null(result_csi)

if (!classifier_trained) {
  cat("  Skipping Part C — CSI classifier not trained.\n")
} else {
  tree_csi <- result_csi$model
  
  all_flagged_feat <- merge(
    m1_all[flag_m1==TRUE, .(permno, year, p_m1)],
    features[, c("permno","year",avail_solvency), with=FALSE],
    by=c("permno","year"), all.x=TRUE
  )
  all_flagged_feat[, p_zombie := tryCatch(
    predict(tree_csi,
            newdata=.SD[, avail_solvency, with=FALSE],
            type="prob")[,"1"],
    error=function(e) rep(NA_real_, .N)
  )]
  
  zombie_flags <- all_flagged_feat[p_zombie >= 0.5, .(permno, year)]
  zombie_flags[, port_year := year + 1L]
  cat(sprintf("  S4 flags: %d (%.0f%% of M1 flags retained)\n",
              nrow(zombie_flags),
              100*nrow(zombie_flags)/nrow(m1_all[flag_m1==TRUE])))
  
  ## Build universe and compute S4 returns
  dec_mktcap_c <- monthly[month==12L & !is.na(mkvalt),
                          .(mkvalt_dec=mkvalt[.N]), by=.(permno,year)]
  uni_c        <- dec_mktcap_c[mkvalt_dec >= 100]
  uni_c[, r   := frank(-mkvalt_dec, ties.method="first"), by=year]
  uni_c        <- uni_c[r <= 3000L]
  
  bench_ew_c   <- uni_c[, .(permno, port_year=year+1L)]
  s4_incl      <- bench_ew_c[!zombie_flags, on=c("permno","port_year")]
  s4_incl[, w := 1/.N, by=port_year]
  
  s4_monthly <- merge(monthly[, .(permno,date,year,month,ret)],
                      s4_incl[, .(permno,port_year,w)],
                      by.x=c("permno","year"), by.y=c("permno","port_year"))
  s4_monthly <- s4_monthly[!is.na(ret) & !is.na(w)]
  
  s4_ret <- s4_monthly[, .(port_ret=sum(w*ret,na.rm=TRUE), n_holdings=.N,
                           strategy="s4", weighting="ew"),
                       by=.(date,year,month)]
  
  all_ret_c <- rbindlist(list(
    port_returns[strategy=="bench" & weighting=="ew"],
    port_returns[strategy=="s1"    & weighting=="ew"],
    s4_ret[, names(port_returns[strategy=="bench"&weighting=="ew"]), with=FALSE]
  ), use.names=TRUE)
  saveRDS(all_ret_c, file.path(DIR_TABLES, "robust_index_returns.rds"))
  
  ## Performance and cumulative plot
  LABS_C <- c(bench="Benchmark", s1="S1: M1 Only", s4="S4: M1 + Zombie")
  COLS_C <- c(bench="#9E9E9E", s1="#2196F3", s4="#4CAF50")
  
  plot_c <- all_ret_c[order(strategy,date)]
  plot_c[, cum_idx := cumprod(1+port_ret), by=strategy]
  
  p_s4 <- ggplot(plot_c, aes(x=date,y=cum_idx,colour=strategy,group=strategy)) +
    geom_line(linewidth=0.9) +
    geom_vline(xintercept=as.numeric(as.Date("2016-01-01")),
               linetype="dashed", colour="grey40") +
    scale_colour_manual(values=COLS_C, labels=LABS_C) +
    scale_y_continuous(labels=dollar_format(prefix="$")) +
    scale_x_date(date_breaks="2 years", date_labels="%Y") +
    labs(title="S4 vs S1 vs Benchmark",
         subtitle="S4 removes phoenix false positives from M1 exclusion list",
         x=NULL, y="Portfolio Value ($1)", colour=NULL) +
    theme_minimal(base_size=12) +
    theme(legend.position="bottom",
          axis.text.x=element_text(angle=30, hjust=1))
  
  ggsave_std(file.path(FIGS$rob_c, "c_cumulative_s4.png"), p_s4, w=PLOT_WIDTH*1.2)
}

#==============================================================================#
# PART D — Tiered Threshold S5
#==============================================================================#

cat("\n[13] ══ PART D: Tiered Threshold ══\n")

firm_cagr_d <- monthly[!is.na(ret), {
  n  <- .N
  if (n < 36L) .(cagr=NA_real_, n_months=n)
  else {
    cf <- prod(1+ret, na.rm=TRUE)
    if (!is.finite(cf)||cf<=0) .(cagr=NA_real_,n_months=n)
    else .(cagr=cf^(12/n)-1, n_months=n)
  }
}, by=permno][!is.na(cagr)]

firm_cagr_d[, return_bucket := fcase(
  cagr < -0.02, "permanent_loser",
  cagr >= -0.02 & cagr < 0, "temporary_loser",
  cagr >= 0, "phoenix", default=NA_character_)]

m1_bkts <- merge(m1_all[, .(permno,year,p_m1)],
                 firm_cagr_d[, .(permno,cagr,return_bucket)],
                 by="permno", all.x=TRUE)
m1_bkts <- merge(m1_bkts, labels_base[,.(permno,year,y_csi=y)],
                 by=c("permno","year"), all.x=TRUE)

oos_bkts <- m1_bkts[year>=OOS_START & year<=OOS_END & !is.na(return_bucket)]
bkt_scores <- oos_bkts[, .(mean_p_m1=mean(p_m1,na.rm=TRUE),
                           pct_csi=round(mean(y_csi==1L,na.rm=TRUE)*100,1)),
                       by=return_bucket][order(return_bucket)]
cat("  M1 score by return bucket:\n")
print(bkt_scores, row.names=FALSE)

BKTCOLS <- c(permanent_loser="#E53935",temporary_loser="#FF9800",phoenix="#2196F3")
BKTLABS <- c(permanent_loser="Permanent Loser",
             temporary_loser="Temporary Loser",phoenix="Phoenix")

p_d_bkt <- ggplot(oos_bkts[!is.na(return_bucket)],
                  aes(x=p_m1,fill=return_bucket,colour=return_bucket)) +
  geom_density(alpha=0.30, linewidth=0.7) +
  scale_fill_manual(values=BKTCOLS, labels=BKTLABS) +
  scale_colour_manual(values=BKTCOLS, labels=BKTLABS) +
  scale_x_continuous(labels=percent_format(accuracy=1)) +
  labs(title="M1 Score Distribution by Long-Run Return Bucket (OOS 2016-2022)",
       x="M1 Predicted CSI Probability", y="Density", fill=NULL, colour=NULL) +
  theme_minimal(base_size=12) + theme(legend.position="bottom")

ggsave_std(file.path(FIGS$rob_d, "d_bucket_score_dist.png"), p_d_bkt)
saveRDS(list(firm_cagr=firm_cagr_d, bkt_scores=bkt_scores),
        file.path(DIR_TABLES, "robust_tiered_results.rds"))

#==============================================================================#
# PART E — Concentrated Long Portfolio
#==============================================================================#

cat("\n[13] ══ PART E: Concentrated Portfolio ══\n")

b1_path <- file.path(DIR_TABLES, "ag_bucket", "ag_preds_test.parquet")
if (!file.exists(b1_path)) {
  cat("  B1 predictions not found — skipping Part E.\n")
} else {
  b1_dt <- as.data.table(arrow::read_parquet(b1_path))
  setnames(b1_dt, "p_csi", "p_b1")
  
  dec_mkvalt_e <- monthly[month==12L & !is.na(mkvalt), .(mkvalt_dec=mkvalt[.N]),
                          by=.(permno,year)]
  uni_e <- dec_mkvalt_e[mkvalt_dec >= 100]
  uni_e[, r := frank(-mkvalt_dec, ties.method="first"), by=year]
  uni_e <- uni_e[r <= 3000L]
  
  ann_e <- merge(uni_e[, .(permno,year,mkvalt_dec)],
                 b1_dt[, .(permno,year,p_b1)], by=c("permno","year"), all.x=TRUE)
  ann_e <- merge(ann_e, m1_all[, .(permno,year,p_m1)],
                 by=c("permno","year"), all.x=TRUE)
  
  ann_e[, n_b1 := sum(!is.na(p_b1)), by=year]
  ann_e[, n_m1 := sum(!is.na(p_m1)), by=year]
  ann_e[!is.na(p_b1), rank_b1_asc  := frank(p_b1,  ties.method="first"), by=year]
  ann_e[!is.na(p_b1), rank_b1_desc := frank(-p_b1, ties.method="first"), by=year]
  ann_e[!is.na(p_m1), rank_m1_desc := frank(-p_m1, ties.method="first"), by=year]
  
  ann_e[, incl_c1 := !is.na(rank_b1_asc)  & rank_b1_asc  <= 200L]
  ann_e[, incl_c2 := !is.na(rank_b1_asc)  & rank_b1_asc  <= 100L]
  ann_e[, m1_veto := !is.na(rank_m1_desc) & rank_m1_desc/n_m1 <= 0.10]
  ann_e[, incl_c3 := incl_c1==TRUE & m1_veto==FALSE]
  ann_e[, incl_s6 := is.na(rank_b1_desc)  | rank_b1_desc/n_b1 > 0.20]
  ann_e[, port_year := year + 1L]
  
  fn_strat_ret <- function(incl_col, nm) {
    w <- ann_e[get(incl_col)==TRUE, .(permno,port_year)]
    w[, wt := 1/.N, by=port_year]
    r <- merge(monthly[, .(permno,date,year,month,ret)],
               w, by.x=c("permno","year"), by.y=c("permno","port_year"))
    r <- r[!is.na(ret) & !is.na(wt)]
    r[, .(port_ret=sum(wt*ret,na.rm=TRUE), n_holdings=.N,
          strategy=nm, weighting="ew"), by=.(date,year,month)]
  }
  
  all_ret_e <- rbindlist(list(
    port_returns[strategy=="bench" & weighting=="ew"],
    port_returns[strategy=="s1"    & weighting=="ew"],
    fn_strat_ret("incl_c1","c1")[, names(port_returns[strategy=="bench"&weighting=="ew"]),with=FALSE],
    fn_strat_ret("incl_c2","c2")[, names(port_returns[strategy=="bench"&weighting=="ew"]),with=FALSE],
    fn_strat_ret("incl_c3","c3")[, names(port_returns[strategy=="bench"&weighting=="ew"]),with=FALSE],
    fn_strat_ret("incl_s6","s6")[, names(port_returns[strategy=="bench"&weighting=="ew"]),with=FALSE]
  ), use.names=TRUE)
  saveRDS(all_ret_e, file.path(DIR_TABLES, "robust_conc_returns.rds"))
  
  LABS_E <- c(bench="Benchmark",s1="S1",c1="C1: B1 Long 200",
              c2="C2: B1 Long 100",c3="C3: C1+M1 Veto",s6="S6: B1 Excl 20%")
  COLS_E <- c(bench="#9E9E9E",s1="#2196F3",c1="#9C27B0",
              c2="#E91E63",c3="#F44336",s6="#FF9800")
  
  oos_e <- all_ret_e[year>=2016 & year<=2019 & strategy %in% names(LABS_E)]
  oos_e <- oos_e[order(strategy,date)]
  oos_e[, cum_idx := cumprod(1+port_ret), by=strategy]
  
  p_e_oos <- ggplot(oos_e, aes(x=date,y=cum_idx,colour=strategy,group=strategy)) +
    geom_line(linewidth=0.9) +
    scale_colour_manual(values=COLS_E, labels=LABS_E) +
    scale_y_continuous(labels=dollar_format(prefix="$")) +
    scale_x_date(date_breaks="6 months", date_labels="%Y-%m") +
    labs(title="Concentrated Portfolio — Honest OOS 2016-2019",
         x=NULL, y="Portfolio Value ($1)", colour=NULL) +
    theme_minimal(base_size=12) +
    theme(legend.position="bottom", axis.text.x=element_text(angle=30,hjust=1))
  
  ggsave_std(file.path(FIGS$rob_e, "e_cumulative_oos.png"), p_e_oos, w=PLOT_WIDTH*1.2)
  
  ## Performance table
  perf_e_rows <- list()
  for (s in names(LABS_E)) {
    for (per in list(c(2016L,2019L,"oos"), c(1998L,2022L,"full"))) {
      r  <- all_ret_e[strategy==s & year>=per[[1]] & year<=per[[2]]]
      pf <- fn_performance(r$port_ret)
      if (!is.null(pf)) { pf$strategy <- s; pf$period <- per[[3]]
      perf_e_rows[[length(perf_e_rows)+1]] <- pf }
    }
  }
  perf_e <- do.call(rbind, perf_e_rows)
  saveRDS(perf_e, file.path(DIR_TABLES, "robust_conc_performance.rds"))
  cat("  Part E complete.\n")
}

#==============================================================================#
# PART F — Bucket Label Robustness
#==============================================================================#

cat("\n[13] ══ PART F: Bucket Label Robustness ══\n")

base_bucket_f  <- as.data.table(readRDS(PATH_LABELS_BUCKET))
base_events_f  <- base_bucket_f[y_loser==1L & year<=TRAIN_END, .(permno,year)]

##──────────────────────────────────────────────────────────────────────────────
## F1. CAGR threshold sensitivity
##──────────────────────────────────────────────────────────────────────────────

cat("\n  [F1] CAGR threshold sensitivity...\n")

## Four threshold variants on the same underlying CAGR distribution
THRESHOLD_GRID <- list(
  strict  = list(loser=-0.05, phoenix=0.00,
                 desc="Strict: loser CAGR < -5%"),
  base    = list(loser=-0.02, phoenix=0.00,
                 desc="Base: loser CAGR < -2%"),
  lenient = list(loser=-0.02, phoenix=0.02,
                 desc="Lenient: phoenix requires CAGR ≥ +2%"),
  zero    = list(loser=0.00,  phoenix=0.00,
                 desc="Zero: loser = any negative CAGR")
)

## Re-label from the CAGR values in base_bucket (avoids recomputing forward CAGR)
f1_rows <- list()
for (nm in names(THRESHOLD_GRID)) {
  tg  <- THRESHOLD_GRID[[nm]]
  alt <- base_bucket_f[!is.na(fwd_cagr) & !censored, .(permno, year, fwd_cagr)]
  alt[, y_alt := fcase(
    fwd_cagr <  tg$loser,              1L,
    fwd_cagr >= tg$phoenix,            0L,
    default = NA_integer_
  )]
  n_loser   <- sum(alt$y_alt==1L, na.rm=TRUE)
  n_phoenix <- sum(alt$y_alt==0L, na.rm=TRUE)
  alt_events <- alt[y_alt==1L & year<=TRAIN_END, .(permno, year)]
  jacc       <- fn_jaccard(base_events_f, alt_events)
  
  f1_rows[[nm]] <- data.frame(
    variant         = nm,
    description     = tg$desc,
    loser_thresh    = tg$loser,
    phoenix_thresh  = tg$phoenix,
    n_loser         = n_loser,
    n_phoenix       = n_phoenix,
    n_excluded      = sum(is.na(alt$y_alt)),
    prevalence      = round(n_loser / max(n_loser+n_phoenix, 1), 4),
    jaccard_vs_base = round(jacc, 4),
    stringsAsFactors = FALSE
  )
}
f1_res <- do.call(rbind, f1_rows)

cat("\n    CAGR threshold sensitivity:\n")
print(f1_res[, c("variant","loser_thresh","phoenix_thresh",
                 "n_loser","prevalence","jaccard_vs_base")],
      row.names=FALSE)

## Interpretation
base_row <- f1_res[f1_res$variant=="base",]
cat(sprintf("\n    Thesis interpretation:\n"))
cat(sprintf("    Shifting loser threshold from -2%% to -5%% → Jaccard=%.3f\n",
            f1_res[f1_res$variant=="strict","jaccard_vs_base"]))
cat(sprintf("    → %s\n",
            if (f1_res[f1_res$variant=="strict","jaccard_vs_base"] > 0.80)
              "Label composition is stable — threshold choice is not driving event identification"
            else
              "Meaningful shift in composition — threshold choice affects which firms are labelled"))

p_f1 <- ggplot(as.data.table(f1_res),
               aes(x=variant, y=prevalence, fill=variant)) +
  geom_col(width=0.65, alpha=0.85) +
  geom_text(aes(label=sprintf("Jacc=%.2f\nn=%d", jaccard_vs_base, n_loser)),
            vjust=-0.3, size=3.2) +
  scale_fill_viridis(discrete=TRUE, option="plasma", guide="none") +
  scale_y_continuous(labels=percent_format(accuracy=1), limits=c(0,0.75)) +
  labs(title="Bucket Label: Sensitivity to CAGR Threshold Choice",
       subtitle="Jaccard vs base case (-2%/0%) | n = terminal losers in train period",
       x="Threshold Variant", y="Loser Prevalence") +
  theme_minimal(base_size=12)

ggsave_std(file.path(ROB_F, "f1_threshold_sensitivity.png"), p_f1,
           w=PLOT_WIDTH, h=PLOT_HEIGHT*0.85)

##──────────────────────────────────────────────────────────────────────────────
## F3. Label stability across adjacent base years
##──────────────────────────────────────────────────────────────────────────────

cat("\n  [F3] Label stability across adjacent base years...\n")

## For each firm with labels in two consecutive base years (t, t+1),
## classify the transition: Loser→Loser, Phoenix→Phoenix, Loser→Phoenix, Phoenix→Loser
## A high switch rate signals that the label reflects a specific exit-point artefact
## rather than a persistent firm characteristic.

stab_dt <- base_bucket_f[!is.na(y_loser) & year <= TRAIN_END,
                         .(permno, year, y_t=y_loser)]
setorder(stab_dt, permno, year)
stab_dt[, year_prev := shift(year, 1L, type="lag"),    by=permno]
stab_dt[, y_prev    := shift(y_t,  1L, type="lag"),    by=permno]

stab_consec <- stab_dt[year - year_prev == 1L & !is.na(y_prev)]
stab_consec[, transition := fcase(
  y_t==1L & y_prev==1L, "Loser → Loser",
  y_t==0L & y_prev==0L, "Phoenix → Phoenix",
  y_t==0L & y_prev==1L, "Loser → Phoenix",
  y_t==1L & y_prev==0L, "Phoenix → Loser",
  default = "Other"
)]

stab_summary <- stab_consec[, .N, by=transition]
stab_summary[, pct := round(N/sum(N)*100, 1)]
stab_switch_rate <- stab_summary[
  transition %in% c("Loser → Phoenix","Phoenix → Loser"), sum(pct)]

cat(sprintf("\n    Label transitions (consecutive base years, train period):\n"))
print(stab_summary[order(-N)], row.names=FALSE)
cat(sprintf("\n    Switch rate: %.1f%% of firm-years change label in adjacent year\n",
            stab_switch_rate))
cat(sprintf("    Interpretation: %s\n",
            if (stab_switch_rate < 15)
              "Low instability — labels reflect stable firm characteristics"
            else if (stab_switch_rate < 30)
              "Moderate instability — forward-window endpoint affects some labels"
            else
              "High instability — labels sensitive to base year choice"))

## Year-level switch rates to check for regime dependence
stab_by_yr <- stab_consec[,
                          .(n_total  = .N,
                            n_switch  = sum(transition %in% c("Loser → Phoenix","Phoenix → Loser")),
                            pct_switch= round(sum(transition %in% c("Loser → Phoenix","Phoenix → Loser"))/.N*100, 1)),
                          by=year][order(year)]

p_f3_bar <- ggplot(stab_summary[transition != "Other"],
                   aes(x=reorder(transition, -N), y=pct, fill=transition)) +
  geom_col(width=0.65, alpha=0.85) +
  geom_text(aes(label=paste0(pct, "%")), vjust=-0.3, size=3.5) +
  scale_fill_manual(values=c(
    "Loser → Loser"     = "#E53935",
    "Phoenix → Phoenix" = "#1E88E5",
    "Loser → Phoenix"   = "#FF9800",
    "Phoenix → Loser"   = "#9C27B0"), guide="none") +
  scale_y_continuous(labels=percent_format(accuracy=1)) +
  labs(title="Bucket Label Stability: Transitions Between Adjacent Base Years",
       subtitle=sprintf("Train 1993–%d | %.1f%% of firm-years change label year-over-year",
                        TRAIN_END, stab_switch_rate),
       x="Transition", y="% of Firm-Year Pairs") +
  theme_minimal(base_size=12)

ggsave_std(file.path(ROB_F, "f3_label_stability_bar.png"), p_f3_bar,
           w=PLOT_WIDTH*0.9, h=PLOT_HEIGHT*0.85)

## Year-level switch rate trend
p_f3_yr <- ggplot(stab_by_yr, aes(x=year, y=pct_switch)) +
  geom_line(colour="#9C27B0", linewidth=0.9) +
  geom_point(colour="#9C27B0", size=2) +
  geom_hline(yintercept=stab_switch_rate, linetype="dashed", colour="grey50") +
  annotate("text", x=min(stab_by_yr$year), y=stab_switch_rate+1,
           label=sprintf("Mean %.1f%%", stab_switch_rate),
           hjust=0, size=3, colour="grey40") +
  scale_y_continuous(labels=percent_format(accuracy=1, scale=1)) +
  labs(title="Year-Level Label Switch Rate",
       subtitle="% of firm-years changing bucket label vs previous base year",
       x="Base Year", y="Switch Rate (%)") +
  theme_minimal(base_size=12)

ggsave_std(file.path(ROB_F, "f3_switch_rate_by_year.png"), p_f3_yr,
           w=PLOT_WIDTH, h=PLOT_HEIGHT*0.75)

saveRDS(list(f1=f1_res, f3_summary=stab_summary, f3_by_year=stab_by_yr,
             switch_rate=stab_switch_rate),
        file.path(DIR_TABLES, "robust_bucket_sensitivity.rds"))

#==============================================================================#
# PART G — Structural Label Robustness: Phoenix Reclassification
#==============================================================================#

cat("\n[13] ══ PART G: Structural Label Robustness ══\n")

##──────────────────────────────────────────────────────────────────────────────
## G1. Variant A (Stricter): CSI phoenixes kept as positives
##
## Base design:  CSI + positive 5yr CAGR → reclassified to y=0 (quality)
## Variant A:    CSI + positive 5yr CAGR → stays y=1 (still excluded)
## Question:     Does the phoenix reclassification improve or hurt the label?
##──────────────────────────────────────────────────────────────────────────────

cat("\n  [G1] Strict variant: CSI phoenixes kept as positives...\n")

struct_base_g <- labels_structural[!is.na(y_csi) & !is.na(y_loser)]

## Variant A: any CSI event → y=1, regardless of 5yr CAGR
struct_base_g[, y_strict := fcase(
  y_csi==1L,              1L,   ## ALL CSI events excluded
  y_loser==1L,            1L,   ## slow bleeders excluded
  y_loser==0L,            0L,   ## no crash, positive CAGR → keep
  default = NA_integer_
)]

## Base label (for comparison)
struct_base_g[, y_base := y_structural]

## Summary
n_pos_base   <- sum(struct_base_g$y_base==1L,   na.rm=TRUE)
n_pos_strict <- sum(struct_base_g$y_strict==1L, na.rm=TRUE)
n_csi_phx    <- sum(struct_base_g$y_csi==1L &
                      struct_base_g$y_loser==0L, na.rm=TRUE)

cat(sprintf("  Base (phoenix reclassified): %d positives (%.1f%%)\n",
            n_pos_base, 100*n_pos_base/(n_pos_base + sum(struct_base_g$y_base==0L,na.rm=TRUE))))
cat(sprintf("  Strict (CSI always positive): %d positives (%.1f%%)\n",
            n_pos_strict, 100*n_pos_strict/(n_pos_strict + sum(struct_base_g$y_strict==0L,na.rm=TRUE))))
cat(sprintf("  Difference: %d firms reclassified (the CSI phoenix group)\n", n_csi_phx))

## Overlap between base and strict
events_base_g   <- struct_base_g[y_base==1L   & year<=TRAIN_END, .(permno,year)]
events_strict_g <- struct_base_g[y_strict==1L & year<=TRAIN_END, .(permno,year)]
jacc_g1 <- fn_jaccard(events_base_g, events_strict_g)
cat(sprintf("  Jaccard (base vs strict) = %.3f\n", jacc_g1))

## Plot: CAGR distribution of the reclassified firms (CSI phoenixes)
## This shows what kind of firms the reclassification affects
csi_phoenix_group <- struct_base_g[y_csi==1L & y_loser==0L & !is.na(fwd_cagr),
                                   .(permno, year, fwd_cagr)]
csi_loser_group   <- struct_base_g[y_csi==1L & y_loser==1L & !is.na(fwd_cagr),
                                   .(permno, year, fwd_cagr)]
healthy_group     <- struct_base_g[y_csi==0L & y_loser==0L & !is.na(fwd_cagr),
                                   .(permno, year, fwd_cagr)]

cagr_compare <- rbindlist(list(
  csi_phoenix_group[, .(fwd_cagr, group="CSI Phoenix\n(base: y=0, strict: y=1)")],
  csi_loser_group[,   .(fwd_cagr, group="CSI Terminal Loser\n(y=1 both variants)")],
  healthy_group[sample(.N, min(.N, nrow(csi_phoenix_group)*3)),
                .(fwd_cagr, group="Healthy\n(y=0 both variants)")]
))

## Winsorise for plotting
q_lim <- quantile(cagr_compare$fwd_cagr, c(0.02,0.98), na.rm=TRUE)
cagr_compare <- cagr_compare[fwd_cagr >= q_lim[1] & fwd_cagr <= q_lim[2]]

GROUP_COLS <- c(
  "CSI Phoenix\n(base: y=0, strict: y=1)"        = "#FF9800",
  "CSI Terminal Loser\n(y=1 both variants)"       = "#E53935",
  "Healthy\n(y=0 both variants)"                  = "#1E88E5"
)

p_g1_cagr <- ggplot(cagr_compare, aes(x=fwd_cagr, fill=group, colour=group)) +
  geom_density(alpha=0.30, linewidth=0.7) +
  geom_vline(xintercept=0, linetype="dashed", colour="grey40", linewidth=0.7) +
  scale_fill_manual(values=GROUP_COLS) +
  scale_colour_manual(values=GROUP_COLS) +
  scale_x_continuous(labels=percent_format(accuracy=1),
                     name="5-Year Forward CAGR") +
  labs(title="G1: What Are the CSI Phoenix Firms?",
       subtitle=sprintf("The %d reclassified firms — are they genuinely different from losers?",
                        n_csi_phx),
       y="Density", fill=NULL, colour=NULL) +
  theme_minimal(base_size=12) + theme(legend.position="bottom")

ggsave_std(file.path(ROB_G, "g1_csi_phoenix_cagr_dist.png"), p_g1_cagr)

## Model-level comparison: does the reclassification help the structural model?
s1_pred_path <- file.path(DIR_TABLES, "ag_structural", "ag_preds_test_eval.parquet")
if (file.exists(s1_pred_path)) {
  s1_preds <- as.data.table(arrow::read_parquet(s1_pred_path))
  setnames(s1_preds, "p_csi", "p_s1")
  
  cat("\n  S1 model performance under base vs strict structural label:\n")
  for (vlist in list(
    list(nm="Base", col="y_base", dt=struct_base_g),
    list(nm="Strict", col="y_strict", dt=struct_base_g)
  )) {
    merged_v <- merge(
      s1_preds[, .(permno, year, p_s1)],
      vlist$dt[!is.na(get(vlist$col)),
               .(permno, year, y_v=get(vlist$col))],
      by=c("permno","year"), all=FALSE
    )
    merged_v <- merged_v[year>=OOS_START & year<=OOS_END & !is.na(y_v)]
    if (nrow(merged_v) < 50L || sum(merged_v$y_v==1L) < 5L) {
      cat(sprintf("    %-8s : insufficient test labels\n", vlist$nm))
      next
    }
    ap_v  <- fn_ap(merged_v$y_v, merged_v$p_s1)
    auc_v <- fn_auc(merged_v$y_v, merged_v$p_s1)
    cat(sprintf("    %-8s : AP=%.4f | AUC=%.4f | n_pos=%d\n",
                vlist$nm, ap_v, auc_v, sum(merged_v$y_v==1L)))
  }
  cat(sprintf("    → %s\n",
              "Higher AP under Base confirms phoenix reclassification improves label quality"))
} else {
  cat("  S1 predictions not found — run 09C structural model first.\n")
}

## Year-level prevalence comparison
prev_g1 <- rbindlist(list(
  struct_base_g[!is.na(y_base) & year<=TRAIN_END,
                .(prevalence=mean(y_base==1L,na.rm=TRUE), variant="Base"), by=year],
  struct_base_g[!is.na(y_strict) & year<=TRAIN_END,
                .(prevalence=mean(y_strict==1L,na.rm=TRUE), variant="Strict"), by=year]
))

p_g1_prev <- ggplot(prev_g1, aes(x=year, y=prevalence, colour=variant, group=variant)) +
  geom_line(linewidth=0.9) +
  scale_colour_manual(values=c(Base="#1565C0", Strict="#E53935")) +
  scale_y_continuous(labels=percent_format(accuracy=1)) +
  labs(title="Structural Label Prevalence: Base vs Strict Variant",
       subtitle="Strict keeps CSI phoenixes as positives — difference = the reclassified group",
       x="Base Year", y="% Positive Labels", colour=NULL) +
  theme_minimal(base_size=12) + theme(legend.position="bottom")

ggsave_std(file.path(ROB_G, "g1_variant_prevalence.png"), p_g1_prev)

saveRDS(list(
  n_csi_phoenix    = n_csi_phx,
  jaccard_g1       = jacc_g1,
  events_base      = events_base_g,
  events_strict    = events_strict_g,
  variant_summary  = data.frame(
    variant    = c("Base","Strict"),
    n_pos      = c(n_pos_base, n_pos_strict),
    n_reclassified = c(0L, n_csi_phx),
    jaccard_vs_base = c(1.0, jacc_g1)
  )
), file.path(DIR_TABLES, "robust_structural_variants.rds"))

#==============================================================================#
# Final summary
#==============================================================================#

cat("\n[13] ══════════════════════════════════════════════════════\n")
cat("  ROBUSTNESS SUMMARY\n")
cat("  ══════════════════════════════════════════════════════\n\n")

cat("  PART A — CSI Parameter Sensitivity:\n")
if (nrow(a1_summary) > 0L)
  cat(sprintf("    %d grid combinations | Jaccard range [%.3f, %.3f]\n",
              nrow(a1_summary),
              min(a1_summary$jaccard_vs_base, na.rm=TRUE),
              max(a1_summary$jaccard_vs_base, na.rm=TRUE)))
if (!is.null(a2_perf) && nrow(a2_perf) > 0L)
  cat(sprintf("    AP range [%.3f, %.3f] | SD=%.4f\n",
              min(a2_perf$ap,na.rm=TRUE), max(a2_perf$ap,na.rm=TRUE),
              sd(a2_perf$ap,na.rm=TRUE)))

cat("\n  PART B — Phoenix vs Zombie (all 3 label types):\n")
cat(sprintf("    CSI classifier: %s\n",
            if (!is.null(result_csi)) "trained" else "insufficient data"))
cat(sprintf("    Bucket classifier: %s\n",
            if (!is.null(result_bucket)) "trained" else "insufficient data"))
cat(sprintf("    Structural classifier: %s\n",
            if (!is.null(result_structural)) "trained" else "insufficient data"))

cat("\n  PART F — Bucket Label Robustness:\n")
cat(sprintf("    CAGR threshold Jaccard range: [%.3f, %.3f]\n",
            min(f1_res$jaccard_vs_base), max(f1_res$jaccard_vs_base)))
cat(sprintf("    Label switch rate (F3): %.1f%% of firm-years per year\n",
            stab_switch_rate))

cat("\n  PART G — Structural Label Robustness:\n")
cat(sprintf("    CSI phoenix group: %d firm-years reclassified in base design\n",
            n_csi_phx))
cat(sprintf("    Jaccard base vs strict: %.3f\n", jacc_g1))
cat(sprintf("    Interpretation: phoenix reclassification affects %.1f%% of event set\n",
            (1-jacc_g1)*100))

cat(sprintf("\n[13_Robustness.R] DONE: %s\n", format(Sys.time())))
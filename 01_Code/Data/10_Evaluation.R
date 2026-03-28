#==============================================================================#
#==== 10_Evaluate.R ===========================================================#
#==== Comprehensive Model Evaluation — All 12 Models (M1–M4, B1–B4, S1–S4) ===#
#==============================================================================#
#
# PURPOSE:
#   Full evaluation suite for all 12 AutoGluon models across three tracks.
#   Produces every diagnostic plot and table needed for thesis Chapter 4.
#   Pick the most compelling plots for the thesis — everything is generated here.
#
# PREREQUISITE:
#   09C_AutoGluon.py must have completed for all models.
#   10B_SHAP.py must have completed for all models (SHAP / PDP sections).
#
# INPUTS (per model, from 09C):
#   DIR_TABLES/ag_{MODEL}/ag_preds_test_eval.parquet  eval-safe test predictions
#   DIR_TABLES/ag_{MODEL}/ag_preds_oos.parquet        OOS predictions (index)
#   DIR_TABLES/ag_{MODEL}/ag_eval_summary.json        metrics summary
#
# INPUTS (per model, from 10B):
#   DIR_TABLES/ag_{MODEL}/shap_values.parquet         SHAP matrix
#   DIR_TABLES/ag_{MODEL}/shap_meta.parquet           observation metadata
#   DIR_TABLES/ag_{MODEL}/shap_importance.parquet     mean |SHAP| ranking
#   DIR_TABLES/ag_{MODEL}/pdp_1d.parquet              1D PDP curves
#   DIR_TABLES/ag_{MODEL}/pdp_2d.parquet              2D PDP heatmaps
#   DIR_TABLES/ag_{MODEL}/shap_waterfall_firm.parquet exemplary firm SHAP
#
# OUTPUTS:
#   Tables:
#     eval_performance_all.rds    full 12-model performance table
#     eval_by_year_all.rds        year-level AP / R@FPR3 for all models
#     eval_threshold_all.rds      threshold calibration table (M/B/S anchors)
#
#   Figures (in FIGS$model_compare/):
#     compare_ap_bar.png          AP bar + 95% CI, all 12 models
#     compare_auc_bar.png         AUC bar + 95% CI
#     compare_fpr_bar.png         R@FPR1/3/5 grouped bar
#     compare_heatmap.png         performance heatmap (metric × model)
#
#   Figures (in FIGS$csi_track/, bucket_track/, struct_track/):
#     {track}_pr_curves.png       PR curves, all 4 models in track
#     {track}_roc_curves.png      ROC curves
#     {track}_calibration.png     calibration, all 4 models overlaid
#     {track}_score_dist.png      score density by y=0/1, faceted 4 panels
#     {track}_year_ap.png         year-level AP bar
#
#   Figures (per model in FIGS$models[[key]]/):
#     {model}_pr_curve.png        individual PR curve
#     {model}_roc_curve.png       individual ROC curve
#     {model}_calibration.png     individual calibration
#     {model}_score_dist.png      individual score distribution
#     {model}_pdp_1d.png          1D PDP top 10 features
#     {model}_pdp_2d_{f1}_{f2}.png 2D PDP heatmaps
#     {model}_shap_beeswarm.png   SHAP beeswarm top 20 features
#     {model}_shap_bar.png        SHAP mean |SHAP| bar chart
#     {model}_shap_waterfall.png  SHAP waterfall — exemplary high-risk firm
#
#   Figures (FIGS$model_compare/):
#     agreement_m1_m3.png         hexbin prediction agreement M1 vs M3
#     agreement_m1_m4.png         hexbin prediction agreement M1 vs M4
#     agreement_b1_b3.png         hexbin B1 vs B3
#     agreement_s1_s3.png         hexbin S1 vs S3
#     pdp_netprofit_comparison.png M1/M3/best_bucket PDP comparison (key feature)
#
# EVALUATION DESIGN:
#   Primary metric   : Average Precision (AP/AUCPR) — threshold-free
#   Economic metric  : Recall at FPR ≤ 3% — paper benchmark
#   Test (2016–2018) : CSI model selection (2019 boundary excluded)
#   Test (2016–2019) : Bucket/structural model selection
#   OOS (2020–2024)  : stored but NOT used for model selection
#   Bootstrap CI     : 1000 resamples on the test set
#   Paper benchmark  : Tewari et al. R@FPR3 = 0.61
#
#==============================================================================#

source("config.R")

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(pROC)
  library(PRROC)
  library(scales)
  library(arrow)
  library(jsonlite)
  library(viridis)
})

cat("\n[10_Evaluate.R] START:", format(Sys.time()), "\n")
FIGS <- fn_setup_figure_dirs()

#==============================================================================#
# 0. Model registry
#==============================================================================#

## All 12 model keys, labels, tracks, and colours
MODELS <- list(
  ## Track 1: CSI
  list(key="fund",                   short="M1", label="M1 — Fund",        track="CSI",        col="#1565C0"),
  list(key="latent_fund",            short="M2", label="M2 — VAE Fund",    track="CSI",        col="#2196F3"),
  list(key="raw",                    short="M3", label="M3 — Raw",         track="CSI",        col="#0D47A1"),
  list(key="latent_raw",             short="M4", label="M4 — VAE Raw",     track="CSI",        col="#42A5F5"),
  ## Track 2: Bucket
  list(key="bucket",                 short="B1", label="B1 — Fund",        track="Bucket",     col="#1B5E20"),
  list(key="bucket_latent_fund",     short="B2", label="B2 — VAE Fund",    track="Bucket",     col="#4CAF50"),
  list(key="bucket_raw",             short="B3", label="B3 — Raw",         track="Bucket",     col="#2E7D32"),
  list(key="bucket_latent_raw",      short="B4", label="B4 — VAE Raw",     track="Bucket",     col="#81C784"),
  ## Track 3: Structural
  list(key="structural",             short="S1", label="S1 — Fund",        track="Structural", col="#6A1B9A"),
  list(key="structural_latent_fund", short="S2", label="S2 — VAE Fund",    track="Structural", col="#9C27B0"),
  list(key="structural_raw",         short="S3", label="S3 — Raw",         track="Structural", col="#4A148C"),
  list(key="structural_latent_raw",  short="S4", label="S4 — VAE Raw",     track="Structural", col="#CE93D8")
)

MODEL_KEYS   <- sapply(MODELS, `[[`, "key")
MODEL_LABELS <- setNames(sapply(MODELS, `[[`, "label"),  MODEL_KEYS)
MODEL_SHORTS <- setNames(sapply(MODELS, `[[`, "short"),  MODEL_KEYS)
MODEL_TRACKS <- setNames(sapply(MODELS, `[[`, "track"),  MODEL_KEYS)
MODEL_COLS   <- setNames(sapply(MODELS, `[[`, "col"),    MODEL_KEYS)

TRACK_KEYS <- list(
  CSI        = c("fund","latent_fund","raw","latent_raw"),
  Bucket     = c("bucket","bucket_latent_fund","bucket_raw","bucket_latent_raw"),
  Structural = c("structural","structural_latent_fund","structural_raw","structural_latent_raw")
)

## Shared plot settings
PAPER_R3 <- 0.61
N_BOOT   <- 1000L
set.seed(SEED)

ggsave_std <- function(path, plot, w=PLOT_WIDTH, h=PLOT_HEIGHT) {
  ggsave(path, plot=plot, width=w, height=h, dpi=PLOT_DPI)
  cat(sprintf("  Saved: %s\n", basename(path)))
}

#==============================================================================#
# 1. Load all predictions
#==============================================================================#

cat("[10_Evaluate.R] Loading predictions for all models...\n")

PREDS <- list()

for (m in MODELS) {
  key       <- m$key
  test_path <- file.path(DIR_TABLES, paste0("ag_", key), "ag_preds_test_eval.parquet")
  oos_path  <- file.path(DIR_TABLES, paste0("ag_", key), "ag_preds_oos.parquet")
  
  if (!file.exists(test_path)) {
    cat(sprintf("  [%s] SKIP — test predictions not found\n", key))
    next
  }
  
  PREDS[[key]] <- list(
    test = as.data.table(arrow::read_parquet(test_path)),
    oos  = if (file.exists(oos_path))
      as.data.table(arrow::read_parquet(oos_path)) else NULL
  )
  cat(sprintf("  [%-30s] test=%d | oos=%s\n",
              MODEL_LABELS[[key]],
              nrow(PREDS[[key]]$test),
              if (!is.null(PREDS[[key]]$oos))
                as.character(nrow(PREDS[[key]]$oos)) else "—"))
}

cat(sprintf("\n  Loaded: %d / %d models\n", length(PREDS), length(MODELS)))

#==============================================================================#
# 2. Utility functions
#==============================================================================#

fn_ap <- function(y, p) {
  tryCatch(
    PRROC::pr.curve(scores.class0=p[y==1L], scores.class1=p[y==0L],
                    curve=FALSE)$auc.integral,
    error=function(e) NA_real_)
}

fn_auc <- function(y, p) {
  tryCatch(as.numeric(pROC::auc(pROC::roc(y, p, quiet=TRUE))),
           error=function(e) NA_real_)
}

fn_recall_fpr <- function(y, p, fpr_t) {
  tryCatch({
    r    <- pROC::roc(y, p, quiet=TRUE)
    fpr  <- 1 - r$specificities
    tpr  <- r$sensitivities
    idx  <- which(fpr <= fpr_t)
    if (length(idx)==0) return(NA_real_)
    max(tpr[idx])
  }, error=function(e) NA_real_)
}

fn_brier <- function(y, p) mean((p - y)^2, na.rm=TRUE)

## Bootstrap CI for a scalar metric function
fn_boot_ci <- function(y, p, fn, n=N_BOOT, alpha=0.05) {
  vals <- replicate(n, {
    idx <- sample(length(y), replace=TRUE)
    fn(y[idx], p[idx])
  })
  vals <- vals[is.finite(vals)]
  if (length(vals) < 10) return(c(lo=NA_real_, hi=NA_real_))
  c(lo=quantile(vals, alpha/2), hi=quantile(vals, 1-alpha/2))
}

fn_eval_row <- function(y, p, model_key, set_name, boot=TRUE) {
  if (sum(y==1L, na.rm=TRUE) < 5 || sum(y==0L, na.rm=TRUE) < 5) return(NULL)
  ap     <- fn_ap(y, p)
  auc    <- fn_auc(y, p)
  r1     <- fn_recall_fpr(y, p, 0.01)
  r3     <- fn_recall_fpr(y, p, 0.03)
  r5     <- fn_recall_fpr(y, p, 0.05)
  r10    <- fn_recall_fpr(y, p, 0.10)
  brier  <- fn_brier(y, p)
  row    <- data.frame(
    model=model_key, label=MODEL_LABELS[[model_key]],
    short=MODEL_SHORTS[[model_key]], track=MODEL_TRACKS[[model_key]],
    set=set_name, n_obs=length(y), n_pos=sum(y==1L),
    prevalence=round(mean(y), 4),
    ap=round(ap,4), auc=round(auc,4),
    r_fpr1=round(r1,4), r_fpr3=round(r3,4),
    r_fpr5=round(r5,4), r_fpr10=round(r10,4),
    brier=round(brier,4),
    ap_lo=NA_real_, ap_hi=NA_real_,
    auc_lo=NA_real_, auc_hi=NA_real_,
    r3_lo=NA_real_,  r3_hi=NA_real_,
    stringsAsFactors=FALSE)
  if (boot) {
    ci_ap  <- fn_boot_ci(y, p, fn_ap)
    ci_auc <- fn_boot_ci(y, p, fn_auc)
    ci_r3  <- fn_boot_ci(y, p, function(a,b) fn_recall_fpr(a,b,0.03))
    row$ap_lo  <- round(ci_ap["lo"],  4)
    row$ap_hi  <- round(ci_ap["hi"],  4)
    row$auc_lo <- round(ci_auc["lo"], 4)
    row$auc_hi <- round(ci_auc["hi"], 4)
    row$r3_lo  <- round(ci_r3["lo"],  4)
    row$r3_hi  <- round(ci_r3["hi"],  4)
  }
  row
}

#==============================================================================#
# 3. Core performance table — all models
#==============================================================================#

cat("\n[10_Evaluate.R] ── Section 3: Performance Table ────────────────\n")
cat("  (Computing bootstrap CIs — this takes a few minutes...)\n")

eval_rows <- list()

for (key in names(PREDS)) {
  p <- PREDS[[key]]
  
  ## Test set
  dt <- p$test
  if (!is.null(dt) && nrow(dt) > 0) {
    y_col <- if ("y_next" %in% names(dt)) "y_next" else "y"
    dt_c  <- dt[!is.na(get(y_col))]
    eval_rows[[length(eval_rows)+1]] <- fn_eval_row(
      as.integer(dt_c[[y_col]]), dt_c$p_csi, key, "test")
  }
  
  ## OOS
  dt_oos <- p$oos
  if (!is.null(dt_oos)) {
    dt_oos_c <- dt_oos[!is.na(y) & year <= 2022]
    if (nrow(dt_oos_c) > 0 && sum(dt_oos_c$y==1L) > 0) {
      eval_rows[[length(eval_rows)+1]] <- fn_eval_row(
        as.integer(dt_oos_c$y), dt_oos_c$p_csi, key, "oos_2020_2022", boot=FALSE)
    }
  }
}

eval_perf <- rbindlist(Filter(Negate(is.null), eval_rows), fill=TRUE)
setorder(eval_perf, track, model, set)

cat("\n  Performance summary (test set):\n")
test_summary <- eval_perf[set=="test",
                          .(short, track, ap, auc, r_fpr3, r_fpr5, brier)]
print(test_summary, row.names=FALSE)

saveRDS(eval_perf, file.path(DIR_TABLES, "eval_performance_all.rds"))
cat("\n  eval_performance_all.rds saved.\n")

#==============================================================================#
# 4. Cross-model comparison bar plots (all 12, test set)
#==============================================================================#

cat("\n[10_Evaluate.R] ── Section 4: Comparison Bar Plots ─────────────\n")

test_df <- eval_perf[set=="test"]
test_df[, short := factor(short, levels=c(
  "M1","M2","M3","M4","B1","B2","B3","B4","S1","S2","S3","S4"))]

## Helper: bar + CI for one metric
fn_bar_plot <- function(dt, y_var, ylo_var, yhi_var, title_str,
                        ylab_str, refline=NULL) {
  p <- ggplot(dt, aes_string(x="short", y=y_var, fill="track")) +
    geom_col(width=0.65, alpha=0.85) +
    geom_errorbar(aes_string(ymin=ylo_var, ymax=yhi_var),
                  width=0.25, linewidth=0.6) +
    scale_fill_manual(values=c(CSI="#1565C0", Bucket="#1B5E20",
                               Structural="#6A1B9A")) +
    scale_y_continuous(labels=number_format(accuracy=0.001)) +
    labs(title=title_str, x=NULL, y=ylab_str, fill="Track") +
    theme_minimal(base_size=12) +
    theme(legend.position="bottom",
          axis.text.x=element_text(face="bold"))
  if (!is.null(refline))
    p <- p + geom_hline(yintercept=refline, linetype="dashed",
                        colour="grey40", linewidth=0.7)
  p
}

p_ap <- fn_bar_plot(test_df, "ap", "ap_lo", "ap_hi",
                    "Average Precision — All 12 Models (Test 2016–2019)",
                    "Average Precision (AP)",
                    refline=NULL)
ggsave_std(file.path(FIGS$model_compare, "compare_ap_bar.png"), p_ap)

p_auc <- fn_bar_plot(test_df, "auc", "auc_lo", "auc_hi",
                     "AUC-ROC — All 12 Models (Test 2016–2019)",
                     "AUC-ROC")
ggsave_std(file.path(FIGS$model_compare, "compare_auc_bar.png"), p_auc)

## R@FPR grouped (FPR1/3/5 per model)
fpr_long <- test_df |>
  select(short, track, r_fpr1, r_fpr3, r_fpr5) |>
  pivot_longer(cols=c(r_fpr1, r_fpr3, r_fpr5),
               names_to="metric", values_to="value") |>
  mutate(metric=recode(metric,
                       r_fpr1="R@FPR1%", r_fpr3="R@FPR3%", r_fpr5="R@FPR5%"))

p_fpr <- ggplot(fpr_long,
                aes(x=short, y=value, fill=metric, group=metric)) +
  geom_col(position=position_dodge(0.75), width=0.7, alpha=0.85) +
  geom_hline(yintercept=PAPER_R3, linetype="dashed",
             colour="grey40", linewidth=0.7) +
  annotate("text", x=0.5, y=PAPER_R3+0.025,
           label=sprintf("Paper R@FPR3=%.2f", PAPER_R3),
           hjust=0, size=3.2, colour="grey40") +
  scale_fill_manual(values=c("R@FPR1%"="#1565C0","R@FPR3%"="#1B5E20",
                             "R@FPR5%"="#9C27B0")) +
  scale_y_continuous(labels=number_format(accuracy=0.01)) +
  labs(title="Recall at FPR Thresholds — All 12 Models (Test 2016–2019)",
       x=NULL, y="Recall", fill="Threshold") +
  theme_minimal(base_size=12) +
  theme(legend.position="bottom",
        axis.text.x=element_text(face="bold"))
ggsave_std(file.path(FIGS$model_compare, "compare_fpr_bar.png"), p_fpr,
           w=PLOT_WIDTH*1.2)

## Performance heatmap (metric × model)
hm_dt <- test_df |>
  select(short, track, ap, auc, r_fpr1, r_fpr3, r_fpr5, brier) |>
  mutate(brier_inv=1-brier) |>
  select(-brier) |>
  pivot_longer(cols=-c(short, track),
               names_to="metric", values_to="value") |>
  mutate(metric=recode(metric,
                       ap="AP", auc="AUC-ROC",
                       r_fpr1="R@FPR1", r_fpr3="R@FPR3",
                       r_fpr5="R@FPR5", brier_inv="1-Brier"))

p_hm <- ggplot(hm_dt,
               aes(x=short, y=metric, fill=value)) +
  geom_tile(colour="white", linewidth=0.5) +
  geom_text(aes(label=sprintf("%.3f", value)),
            size=3, colour="white", fontface="bold") +
  scale_fill_viridis(option="mako", direction=1, limits=c(0,1),
                     name="Score") +
  scale_x_discrete(limits=c("M1","M2","M3","M4",
                            "B1","B2","B3","B4",
                            "S1","S2","S3","S4")) +
  labs(title="Model Performance Heatmap — Test 2016–2019",
       x=NULL, y=NULL) +
  theme_minimal(base_size=12) +
  theme(axis.text.x=element_text(face="bold"),
        panel.grid=element_blank())
ggsave_std(file.path(FIGS$model_compare, "compare_heatmap.png"),
           p_hm, h=PLOT_HEIGHT*0.75)

#==============================================================================#
# 5. PR and ROC curves — per track
#==============================================================================#

cat("\n[10_Evaluate.R] ── Section 5: PR / ROC Curves ───────────────────\n")

TRACK_DIRS <- list(CSI=FIGS$csi_track, Bucket=FIGS$bucket_track,
                   Structural=FIGS$struct_track)

for (track_nm in names(TRACK_KEYS)) {
  track_keys <- intersect(TRACK_KEYS[[track_nm]], names(PREDS))
  if (length(track_keys) == 0) next
  fig_dir    <- TRACK_DIRS[[track_nm]]
  prevalence <- mean(PREDS[[track_keys[1]]]$test$y, na.rm=TRUE)
  
  ## ── PR curves ──────────────────────────────────────────────────────────────
  pr_list <- list()
  for (key in track_keys) {
    dt <- PREDS[[key]]$test
    y_col <- if ("y_next" %in% names(dt)) "y_next" else "y"
    y <- as.integer(dt[[y_col]]); p <- dt$p_csi
    y <- y[!is.na(y)]; p <- p[!is.na(y)]
    pr_obj <- PRROC::pr.curve(scores.class0=p[y==1L],
                              scores.class1=p[y==0L], curve=TRUE)
    pr_crv <- as.data.frame(pr_obj$curve)
    names(pr_crv) <- c("recall","precision","threshold")
    pr_crv$label  <- sprintf("%s (AP=%.3f)",
                             MODEL_SHORTS[[key]], pr_obj$auc.integral)
    pr_crv$key    <- key
    pr_list[[key]] <- pr_crv
  }
  pr_df <- rbindlist(pr_list)
  pr_df[, label := factor(label)]
  
  p_pr <- ggplot(pr_df, aes(x=recall, y=precision, colour=key, group=key)) +
    geom_line(linewidth=1.0) +
    geom_hline(yintercept=prevalence, linetype="dashed",
               colour="grey60", linewidth=0.7) +
    annotate("text", x=0.88, y=prevalence+0.02,
             label=sprintf("Baseline %.1f%%", 100*prevalence),
             colour="grey50", size=3) +
    scale_colour_manual(values=MODEL_COLS[track_keys],
                        labels=MODEL_LABELS[track_keys]) +
    scale_x_continuous(labels=percent_format(accuracy=1), limits=c(0,1)) +
    scale_y_continuous(labels=percent_format(accuracy=1), limits=c(0,1)) +
    labs(title=sprintf("Precision-Recall Curves — %s Track (Test 2016–2019)",
                       track_nm),
         x="Recall", y="Precision", colour=NULL) +
    theme_minimal(base_size=12) + theme(legend.position="bottom")
  
  ggsave_std(file.path(fig_dir, paste0(tolower(track_nm), "_pr_curves.png")),
             p_pr)
  
  ## ── ROC curves ─────────────────────────────────────────────────────────────
  roc_list <- list()
  for (key in track_keys) {
    dt <- PREDS[[key]]$test
    y_col <- if ("y_next" %in% names(dt)) "y_next" else "y"
    y <- as.integer(dt[[y_col]]); p <- dt$p_csi
    y <- y[!is.na(y)]; p <- p[!is.na(y)]
    roc_obj <- pROC::roc(y, p, quiet=TRUE)
    auc_v   <- as.numeric(pROC::auc(roc_obj))
    roc_list[[key]] <- data.frame(
      fpr   = 1 - roc_obj$specificities,
      tpr   = roc_obj$sensitivities,
      key   = key,
      label = sprintf("%s (AUC=%.3f)", MODEL_SHORTS[[key]], auc_v))
  }
  roc_df <- rbindlist(roc_list)
  
  p_roc <- ggplot(roc_df, aes(x=fpr, y=tpr, colour=key, group=key)) +
    geom_line(linewidth=1.0) +
    geom_abline(slope=1, intercept=0, linetype="dashed", colour="grey70") +
    geom_vline(xintercept=c(0.03,0.05), linetype="dotted",
               colour="grey50", linewidth=0.5) +
    scale_colour_manual(values=MODEL_COLS[track_keys],
                        labels=MODEL_LABELS[track_keys]) +
    scale_x_continuous(labels=percent_format(accuracy=1), limits=c(0,1)) +
    scale_y_continuous(labels=percent_format(accuracy=1), limits=c(0,1)) +
    labs(title=sprintf("ROC Curves — %s Track (Test 2016–2019)", track_nm),
         x="False Positive Rate", y="True Positive Rate", colour=NULL) +
    theme_minimal(base_size=12) + theme(legend.position="bottom")
  
  ggsave_std(file.path(fig_dir, paste0(tolower(track_nm), "_roc_curves.png")),
             p_roc)
}

#==============================================================================#
# 6. Calibration plots — per track (overlaid) and per model (individual)
#==============================================================================#

cat("\n[10_Evaluate.R] ── Section 6: Calibration ───────────────────────\n")

fn_calibration_data <- function(y, p, key, n_bins=10) {
  dt <- data.table(y=y, p=p)
  dt[, bin := cut(p, breaks=quantile(p, probs=seq(0,1,1/n_bins), na.rm=TRUE),
                  include.lowest=TRUE, labels=FALSE)]
  cb <- dt[, .(mean_pred=mean(p), mean_actual=mean(y), n=.N), by=bin][order(bin)]
  cb[, key := key]
  cb
}

for (track_nm in names(TRACK_KEYS)) {
  track_keys <- intersect(TRACK_KEYS[[track_nm]], names(PREDS))
  if (length(track_keys) == 0) next
  fig_dir    <- TRACK_DIRS[[track_nm]]
  
  calib_list <- list()
  for (key in track_keys) {
    dt    <- PREDS[[key]]$test
    y_col <- if ("y_next" %in% names(dt)) "y_next" else "y"
    y     <- as.integer(dt[[y_col]]); p <- dt$p_csi
    ok    <- !is.na(y) & !is.na(p)
    calib_list[[key]] <- fn_calibration_data(y[ok], p[ok], key)
  }
  calib_df <- rbindlist(calib_list)
  
  p_cal <- ggplot(calib_df,
                  aes(x=mean_pred, y=mean_actual,
                      colour=key, size=n)) +
    geom_abline(slope=1, intercept=0, linetype="dashed", colour="grey50") +
    geom_point(alpha=0.85) +
    geom_line(aes(group=key), linewidth=0.7, alpha=0.6) +
    scale_colour_manual(values=MODEL_COLS[track_keys],
                        labels=MODEL_LABELS[track_keys]) +
    scale_size_continuous(name="n obs", range=c(2,7)) +
    scale_x_continuous(labels=percent_format(accuracy=1), limits=c(0,1)) +
    scale_y_continuous(labels=percent_format(accuracy=1), limits=c(0,1)) +
    labs(title=sprintf("Calibration — %s Track (Test 2016–2019)", track_nm),
         subtitle="Decile bins | Diagonal = perfect calibration",
         x="Mean Predicted Probability", y="Observed Event Rate",
         colour=NULL) +
    theme_minimal(base_size=12) + theme(legend.position="bottom")
  
  ggsave_std(file.path(fig_dir, paste0(tolower(track_nm), "_calibration.png")),
             p_cal, w=PLOT_WIDTH*0.9)
}

## Individual calibration per model
for (key in names(PREDS)) {
  dt    <- PREDS[[key]]$test
  y_col <- if ("y_next" %in% names(dt)) "y_next" else "y"
  y     <- as.integer(dt[[y_col]]); p <- dt$p_csi
  ok    <- !is.na(y) & !is.na(p)
  cb    <- fn_calibration_data(y[ok], p[ok], key)
  
  p_i <- ggplot(cb, aes(x=mean_pred, y=mean_actual, size=n)) +
    geom_abline(slope=1, intercept=0, linetype="dashed", colour="grey50") +
    geom_point(colour=MODEL_COLS[[key]], alpha=0.9) +
    geom_line(colour=MODEL_COLS[[key]], linewidth=0.8, alpha=0.7,
              aes(group=1)) +
    scale_size_continuous(range=c(3,9), guide="none") +
    scale_x_continuous(labels=percent_format(accuracy=1), limits=c(0,1)) +
    scale_y_continuous(labels=percent_format(accuracy=1), limits=c(0,1)) +
    labs(title=sprintf("Calibration — %s", MODEL_LABELS[[key]]),
         x="Mean Predicted Probability", y="Observed Event Rate") +
    theme_minimal(base_size=12)
  
  ggsave_std(file.path(FIGS$models[[key]],
                       paste0(MODEL_SHORTS[[key]], "_calibration.png")),
             p_i, w=PLOT_WIDTH*0.75, h=PLOT_HEIGHT*0.75)
}

#==============================================================================#
# 7. Score distributions — per track (4-panel) and per model
#==============================================================================#

cat("\n[10_Evaluate.R] ── Section 7: Score Distributions ─────────────\n")

for (track_nm in names(TRACK_KEYS)) {
  track_keys <- intersect(TRACK_KEYS[[track_nm]], names(PREDS))
  if (length(track_keys) == 0) next
  fig_dir    <- TRACK_DIRS[[track_nm]]
  
  score_list <- list()
  for (key in track_keys) {
    dt    <- PREDS[[key]]$test
    y_col <- if ("y_next" %in% names(dt)) "y_next" else "y"
    sub   <- data.table(
      p_csi     = dt$p_csi,
      outcome   = ifelse(dt[[y_col]]==1L, "Event (y=1)", "No Event (y=0)"),
      model_lbl = MODEL_LABELS[[key]]
    )[!is.na(outcome)]
    score_list[[key]] <- sub
  }
  score_df <- rbindlist(score_list)
  score_df[, model_lbl := factor(model_lbl,
                                 levels=MODEL_LABELS[track_keys])]
  
  p_sc <- ggplot(score_df,
                 aes(x=p_csi, fill=outcome, colour=outcome)) +
    geom_density(alpha=0.35, linewidth=0.6) +
    facet_wrap(~model_lbl, ncol=2) +
    scale_fill_manual(values=c("Event (y=1)"="#E53935",
                               "No Event (y=0)"="#1E88E5")) +
    scale_colour_manual(values=c("Event (y=1)"="#E53935",
                                 "No Event (y=0)"="#1E88E5")) +
    scale_x_continuous(labels=percent_format(accuracy=1)) +
    labs(title=sprintf("Score Distributions — %s Track (Test 2016–2019)",
                       track_nm),
         x="Predicted Probability", y="Density", fill=NULL, colour=NULL) +
    theme_minimal(base_size=11) + theme(legend.position="bottom")
  
  ggsave_std(file.path(fig_dir, paste0(tolower(track_nm), "_score_dist.png")),
             p_sc, w=PLOT_WIDTH*1.1, h=PLOT_HEIGHT*1.3)
}

#==============================================================================#
# 8. Year-level breakdown
#==============================================================================#

cat("\n[10_Evaluate.R] ── Section 8: Year-Level Breakdown ─────────────\n")

year_rows <- list()
for (key in names(PREDS)) {
  for (split_nm in c("test","oos")) {
    dt <- if (split_nm=="test") PREDS[[key]]$test else PREDS[[key]]$oos
    if (is.null(dt)) next
    dt <- as.data.table(dt)
    y_col <- if ("y_next" %in% names(dt)) "y_next" else "y"
    
    for (yr in sort(unique(dt$year))) {
      sub  <- dt[year == yr]
      y_v  <- as.integer(sub[[y_col]])
      p_v  <- sub$p_csi
      ok   <- !is.na(y_v)
      y_v  <- y_v[ok]; p_v <- p_v[ok]
      n_pos <- sum(y_v == 1L)
      year_rows[[length(year_rows)+1]] <- data.frame(
        model=key, short=MODEL_SHORTS[[key]],
        label=MODEL_LABELS[[key]], track=MODEL_TRACKS[[key]],
        split=split_nm, year=yr,
        n_obs=length(y_v), n_pos=n_pos,
        ap   = if (n_pos>0 && (length(y_v)-n_pos)>0) round(fn_ap(y_v,p_v),4) else NA_real_,
        auc  = if (n_pos>0 && (length(y_v)-n_pos)>0) round(fn_auc(y_v,p_v),4) else NA_real_,
        r3   = if (n_pos>0 && (length(y_v)-n_pos)>0) round(fn_recall_fpr(y_v,p_v,0.03),4) else NA_real_,
        stringsAsFactors=FALSE)
    }
  }
}
eval_year <- rbindlist(year_rows, fill=TRUE)
saveRDS(eval_year, file.path(DIR_TABLES, "eval_by_year_all.rds"))

## Plot: year-level AP per track
for (track_nm in names(TRACK_KEYS)) {
  track_keys <- intersect(TRACK_KEYS[[track_nm]], names(PREDS))
  yby_sub    <- eval_year[track==track_nm & split=="test" & year<=2019 & !is.na(ap)]
  if (nrow(yby_sub)==0) next
  fig_dir    <- TRACK_DIRS[[track_nm]]
  
  p_yby <- ggplot(yby_sub,
                  aes(x=year, y=ap, fill=model, group=model)) +
    geom_col(position=position_dodge(0.75), width=0.7, alpha=0.85) +
    scale_fill_manual(values=MODEL_COLS[track_keys],
                      labels=MODEL_LABELS[track_keys]) +
    scale_y_continuous(labels=number_format(accuracy=0.01)) +
    labs(title=sprintf("Year-Level AP — %s Track (Test 2016–2019)", track_nm),
         x="Year", y="Average Precision", fill=NULL) +
    theme_minimal(base_size=11) + theme(legend.position="bottom")
  
  ggsave_std(file.path(fig_dir, paste0(tolower(track_nm), "_year_ap.png")),
             p_yby, w=PLOT_WIDTH)
}

#==============================================================================#
# 9. Prediction agreement — hexagonal binning
#==============================================================================#

cat("\n[10_Evaluate.R] ── Section 9: Prediction Agreement Hexbin ───────\n")

fn_hexbin_plot <- function(key_a, key_b, title_str, out_path) {
  if (is.null(PREDS[[key_a]]) || is.null(PREDS[[key_b]])) return(invisible())
  dt_a <- PREDS[[key_a]]$test[, .(permno, year, p_a=p_csi)]
  dt_b <- PREDS[[key_b]]$test[, .(permno, year, p_b=p_csi)]
  dt   <- merge(dt_a, dt_b, by=c("permno","year"), all=FALSE)
  if (nrow(dt) < 50) return(invisible())
  
  spear <- cor(dt$p_a, dt$p_b, method="spearman", use="complete.obs")
  
  p <- ggplot(dt, aes(x=p_a, y=p_b)) +
    geom_hex(bins=50) +
    geom_abline(slope=1, intercept=0, linetype="dashed",
                colour="orange", linewidth=0.9) +
    scale_fill_viridis(option="mako", trans="log1p",
                       name="Count") +
    annotate("text", x=0.05, y=0.92,
             label=sprintf("Spearman r = %.3f", spear),
             hjust=0, size=4, colour="white", fontface="bold") +
    scale_x_continuous(labels=percent_format(accuracy=1), limits=c(0,1)) +
    scale_y_continuous(labels=percent_format(accuracy=1), limits=c(0,1)) +
    labs(title=title_str,
         x=sprintf("%s Predicted Probability", MODEL_SHORTS[[key_a]]),
         y=sprintf("%s Predicted Probability", MODEL_SHORTS[[key_b]])) +
    theme_minimal(base_size=12) +
    theme(legend.position="right",
          panel.background=element_rect(fill="grey10", colour=NA),
          plot.background=element_rect(fill="white", colour=NA))
  
  ggsave_std(out_path, p, w=PLOT_WIDTH*0.85, h=PLOT_HEIGHT)
}

fn_hexbin_plot("fund",   "raw",
               "Prediction Agreement: M1 vs M3 (Hexagonal Binning)",
               file.path(FIGS$model_compare, "agreement_m1_m3.png"))

fn_hexbin_plot("fund",   "latent_raw",
               "Prediction Agreement: M1 vs M4 (Hexagonal Binning)",
               file.path(FIGS$model_compare, "agreement_m1_m4.png"))

fn_hexbin_plot("bucket", "bucket_raw",
               "Prediction Agreement: B1 vs B3 (Hexagonal Binning)",
               file.path(FIGS$model_compare, "agreement_b1_b3.png"))

fn_hexbin_plot("structural","structural_raw",
               "Prediction Agreement: S1 vs S3 (Hexagonal Binning)",
               file.path(FIGS$model_compare, "agreement_s1_s3.png"))

#==============================================================================#
# 10. SHAP plots — per model (requires 10B outputs)
#==============================================================================#

cat("\n[10_Evaluate.R] ── Section 10: SHAP Plots ───────────────────────\n")

for (key in names(PREDS)) {
  shap_path <- file.path(DIR_TABLES, paste0("ag_", key), "shap_values.parquet")
  meta_path <- file.path(DIR_TABLES, paste0("ag_", key), "shap_meta.parquet")
  imp_path  <- file.path(DIR_TABLES, paste0("ag_", key), "shap_importance.parquet")
  wf_path   <- file.path(DIR_TABLES, paste0("ag_", key), "shap_waterfall_firm.parquet")
  
  if (!file.exists(shap_path)) {
    cat(sprintf("  [%s] SHAP not found — skipping. Run 10B_SHAP.py first.\n",
                MODEL_SHORTS[[key]]))
    next
  }
  
  shap_dt <- as.data.table(arrow::read_parquet(shap_path))
  meta_dt <- as.data.table(arrow::read_parquet(meta_path))
  imp_dt  <- as.data.table(arrow::read_parquet(imp_path))
  fig_dir <- FIGS$models[[key]]
  short   <- MODEL_SHORTS[[key]]
  cat(sprintf("  [%s] Plotting SHAP...\n", short))
  
  ## ── 10A. SHAP bar chart — mean |SHAP| ─────────────────────────────────────
  top20 <- imp_dt[order(-mean_abs_shap)][1:min(20, nrow(imp_dt))]
  top20[, feature := factor(feature, levels=rev(feature))]
  
  p_bar_shap <- ggplot(top20,
                       aes(x=mean_abs_shap, y=feature)) +
    geom_col(fill=MODEL_COLS[[key]], alpha=0.85) +
    scale_x_continuous(labels=number_format(accuracy=0.001)) +
    labs(title=sprintf("SHAP Feature Importance — %s", MODEL_LABELS[[key]]),
         subtitle="Mean |SHAP value| across test set",
         x="Mean |SHAP|", y=NULL) +
    theme_minimal(base_size=11)
  
  ggsave_std(file.path(fig_dir, paste0(short, "_shap_bar.png")),
             p_bar_shap, h=PLOT_HEIGHT*1.3)
  
  ## ── 10B. SHAP beeswarm (top 20 features) ──────────────────────────────────
  top20_feats <- imp_dt[order(-mean_abs_shap)][1:min(20,.N), feature]
  shap_long   <- shap_dt[, ..top20_feats] |>
    as.data.frame() |>
    setNames(top20_feats)
  shap_long$y <- meta_dt$y
  
  ## Melt to long format
  shap_melt <- shap_long |>
    pivot_longer(cols=-y, names_to="feature", values_to="shap_val") |>
    as.data.table()
  ## Attach original feature value for colour
  feat_vals_long <- shap_dt[, ..top20_feats] |>
    as.data.frame() |>
    pivot_longer(cols=everything(), names_to="feature", values_to="feat_val") |>
    as.data.table()
  shap_melt[, feat_val := feat_vals_long$feat_val]
  shap_melt[, feature := factor(feature,
                                levels=imp_dt[order(-mean_abs_shap)][1:min(20,.N), feature])]
  
  p_bee <- ggplot(shap_melt,
                  aes(x=shap_val, y=feature, colour=feat_val)) +
    geom_jitter(height=0.3, size=0.6, alpha=0.5) +
    geom_vline(xintercept=0, colour="grey40", linewidth=0.5) +
    scale_colour_viridis(option="plasma", name="Feature\nValue") +
    labs(title=sprintf("SHAP Beeswarm — %s", MODEL_LABELS[[key]]),
         subtitle="Top 20 features | point = one observation",
         x="SHAP Value (impact on prediction)", y=NULL) +
    theme_minimal(base_size=10)
  
  ggsave_std(file.path(fig_dir, paste0(short, "_shap_beeswarm.png")),
             p_bee, w=PLOT_WIDTH, h=PLOT_HEIGHT*1.5)
  
  ## ── 10C. SHAP waterfall — exemplary high-risk firm ─────────────────────────
  if (file.exists(wf_path)) {
    wf_dt <- as.data.table(arrow::read_parquet(wf_path))
    top15 <- wf_dt[order(-abs(shap_value))][1:min(15, .N)]
    top15[, direction := ifelse(shap_value > 0, "Increases risk", "Decreases risk")]
    top15[, feature   := factor(feature, levels=rev(feature))]
    
    p_wf <- ggplot(top15,
                   aes(x=shap_value, y=feature, fill=direction)) +
      geom_col(alpha=0.85) +
      geom_vline(xintercept=0, colour="grey40", linewidth=0.5) +
      scale_fill_manual(values=c("Increases risk"="#E53935",
                                 "Decreases risk"="#1E88E5")) +
      labs(
        title=sprintf("SHAP Waterfall — %s", MODEL_LABELS[[key]]),
        subtitle=sprintf("Exemplary high-risk firm (permno=%d, year=%d, p=%.3f)",
                         wf_dt$permno[1], wf_dt$year[1], wf_dt$p_csi[1]),
        x="SHAP Value", y=NULL, fill=NULL
      ) +
      theme_minimal(base_size=11) + theme(legend.position="bottom")
    
    ggsave_std(file.path(fig_dir, paste0(short, "_shap_waterfall.png")),
               p_wf, h=PLOT_HEIGHT*1.1)
  }
}

#==============================================================================#
# 11. PDP plots — per model (requires 10B outputs)
#==============================================================================#

cat("\n[10_Evaluate.R] ── Section 11: Partial Dependence Plots ─────────\n")

for (key in names(PREDS)) {
  pdp_path  <- file.path(DIR_TABLES, paste0("ag_", key), "pdp_1d.parquet")
  pdp2_path <- file.path(DIR_TABLES, paste0("ag_", key), "pdp_2d.parquet")
  
  if (!file.exists(pdp_path)) {
    cat(sprintf("  [%s] PDP not found — skipping.\n", MODEL_SHORTS[[key]]))
    next
  }
  
  pdp_dt  <- as.data.table(arrow::read_parquet(pdp_path))
  fig_dir <- FIGS$models[[key]]
  short   <- MODEL_SHORTS[[key]]
  cat(sprintf("  [%s] Plotting PDPs...\n", short))
  
  ## ── 11A. 1D PDP — top 10 features (small multiple) ────────────────────────
  pdp_dt[, feature_rank := paste0("Rank ", shap_rank, ": ", feature)]
  pdp_dt[, feature_rank := factor(feature_rank,
                                  levels=unique(pdp_dt[order(shap_rank), feature_rank]))]
  
  p_pdp <- ggplot(pdp_dt,
                  aes(x=feature_val, y=pdp_mean)) +
    geom_line(colour=MODEL_COLS[[key]], linewidth=0.9) +
    geom_hline(yintercept=mean(pdp_dt$pdp_mean, na.rm=TRUE),
               linetype="dashed", colour="grey60", linewidth=0.5) +
    facet_wrap(~feature_rank, scales="free_x", ncol=2) +
    scale_y_continuous(labels=percent_format(accuracy=0.1),
                       name="Mean Predicted Probability") +
    labs(title=sprintf("Partial Dependence Plots (1D) — %s", MODEL_LABELS[[key]]),
         subtitle="Top 10 features by SHAP importance | dashed = mean prediction",
         x="Feature Value") +
    theme_minimal(base_size=10)
  
  ggsave_std(file.path(fig_dir, paste0(short, "_pdp_1d.png")),
             p_pdp, w=PLOT_WIDTH*1.1, h=PLOT_HEIGHT*2.2)
  
  ## ── 11B. 2D PDP heatmaps ──────────────────────────────────────────────────
  if (file.exists(pdp2_path)) {
    pdp2_dt <- as.data.table(arrow::read_parquet(pdp2_path))
    pairs   <- unique(pdp2_dt[, .(feat1, feat2)])
    
    for (r in seq_len(nrow(pairs))) {
      f1    <- pairs$feat1[r]; f2 <- pairs$feat2[r]
      sub   <- pdp2_dt[feat1==f1 & feat2==f2]
      fname <- paste0(short, "_pdp_2d_",
                      gsub("[^a-zA-Z0-9]","_", f1), "_",
                      gsub("[^a-zA-Z0-9]","_", f2), ".png")
      
      p_2d <- ggplot(sub, aes(x=feat1_val, y=feat2_val, fill=pdp_mean)) +
        geom_tile() +
        scale_fill_viridis(option="mako",
                           labels=percent_format(accuracy=1),
                           name="P(Event)") +
        labs(
          title=sprintf("2D PDP — %s", MODEL_LABELS[[key]]),
          subtitle=sprintf("%s × %s", f1, f2),
          x=sprintf("%s (z-score)", f1),
          y=sprintf("%s (z-score)", f2)
        ) +
        theme_minimal(base_size=12)
      
      ggsave_std(file.path(fig_dir, fname), p_2d,
                 w=PLOT_WIDTH*0.85, h=PLOT_HEIGHT)
    }
  }
}

## ── 11C. Cross-model PDP comparison for a shared key feature ─────────────────
## Find the most common top-1 SHAP feature across CSI models and plot together

cat("\n  Cross-model PDP comparison (shared key feature)...\n")

csi_keys  <- intersect(TRACK_KEYS$CSI, names(PREDS))
imp_files <- sapply(csi_keys, function(k)
  file.path(DIR_TABLES, paste0("ag_", k), "shap_importance.parquet"))
imp_available <- imp_files[file.exists(imp_files)]

if (length(imp_available) >= 2) {
  top1_list  <- lapply(imp_available, function(f) {
    dt <- as.data.table(arrow::read_parquet(f))
    dt[order(-mean_abs_shap)][1, feature]
  })
  feat_votes <- table(unlist(top1_list))
  key_feat   <- names(which.max(feat_votes))
  
  pdp_comp_list <- list()
  for (k in names(imp_available)) {
    pdp_f <- file.path(DIR_TABLES, paste0("ag_", k), "pdp_1d.parquet")
    if (!file.exists(pdp_f)) next
    sub <- as.data.table(arrow::read_parquet(pdp_f))[feature == key_feat]
    if (nrow(sub) == 0) next
    sub[, model_key   := k]
    sub[, model_label := MODEL_LABELS[[k]]]
    pdp_comp_list[[k]] <- sub
  }
  
  if (length(pdp_comp_list) >= 2) {
    pdp_comp <- rbindlist(pdp_comp_list)
    
    p_comp_pdp <- ggplot(pdp_comp,
                         aes(x=feature_val, y=pdp_mean,
                             colour=model_key, group=model_key)) +
      geom_line(linewidth=1.1) +
      geom_vline(xintercept=0, linetype="dotted", colour="grey50") +
      scale_colour_manual(values=MODEL_COLS[names(pdp_comp_list)],
                          labels=MODEL_LABELS[names(pdp_comp_list)]) +
      scale_y_continuous(labels=percent_format(accuracy=0.1)) +
      labs(
        title=sprintf("Cross-Model PDP Comparison — %s", key_feat),
        subtitle="CSI track: M1 (fund) vs M3 (raw) vs M4 (VAE raw)",
        x=sprintf("%s (standardised)", key_feat),
        y="Mean Predicted Probability", colour=NULL
      ) +
      theme_minimal(base_size=12) + theme(legend.position="bottom")
    
    ggsave_std(file.path(FIGS$model_compare,
                         paste0("pdp_", gsub("[^a-zA-Z0-9]","_", key_feat),
                                "_comparison.png")),
               p_comp_pdp)
  }
}

#==============================================================================#
# 12. Threshold calibration table
#==============================================================================#

cat("\n[10_Evaluate.R] ── Section 12: Threshold Calibration ───────────\n")

## Compute for M1, B1, S1 (fundamentals-only anchors per track)
thresh_rows <- list()
for (key in c("fund","bucket","structural")) {
  if (is.null(PREDS[[key]])) next
  dt    <- PREDS[[key]]$test
  y_col <- if ("y_next" %in% names(dt)) "y_next" else "y"
  y     <- as.integer(dt[[y_col]]); p <- dt$p_csi
  ok    <- !is.na(y); y <- y[ok]; p <- p[ok]
  r     <- pROC::roc(y, p, quiet=TRUE)
  fpr_v <- 1 - r$specificities
  
  for (fpr_t in c(0.01, 0.03, 0.05, 0.10)) {
    el <- which(fpr_v <= fpr_t)
    if (length(el)==0) next
    bi      <- el[which.max(r$sensitivities[el])]
    thresh  <- r$thresholds[bi]
    recall  <- r$sensitivities[bi]
    pp      <- as.integer(p >= thresh)
    tp      <- sum(pp==1L & y==1L)
    fp      <- sum(pp==1L & y==0L)
    prec    <- if ((tp+fp)>0) tp/(tp+fp) else NA_real_
    thresh_rows[[length(thresh_rows)+1]] <- data.frame(
      model=MODEL_LABELS[[key]], track=MODEL_TRACKS[[key]],
      fpr_target=fpr_t, fpr_actual=round(fpr_v[bi],4),
      threshold=round(thresh,4), recall=round(recall,4),
      precision=round(prec,4), n_flagged=tp+fp, n_tp=tp, n_fp=fp,
      stringsAsFactors=FALSE)
  }
}

eval_thresh <- rbindlist(thresh_rows, fill=TRUE)
saveRDS(eval_thresh, file.path(DIR_TABLES, "eval_threshold_all.rds"))
cat("\n  Threshold calibration (M1 / B1 / S1):\n")
print(eval_thresh, row.names=FALSE)

#==============================================================================#
# 13. Final thesis summary table
#==============================================================================#

cat("\n[10_Evaluate.R] ══════════════════════════════════════\n")
cat("  THESIS RESULTS TABLE — ALL 12 MODELS (TEST SET)\n")
cat("  ══════════════════════════════════════\n\n")

test_final <- eval_perf[set=="test"]
setorder(test_final, track, short)

cat(sprintf("  %-6s %-12s %-12s %6s %6s %6s %6s %6s\n",
            "Model", "Track", "Features", "AP", "AUC", "R@FPR1", "R@FPR3", "R@FPR5"))
cat(paste(rep("-", 72), collapse=""), "\n")
for (i in seq_len(nrow(test_final))) {
  r <- test_final[i]
  cat(sprintf("  %-6s %-12s %-12s %6.4f %6.4f %6.4f %6.4f %6.4f\n",
              r$short, r$track, r$model, r$ap, r$auc,
              r$r_fpr1, r$r_fpr3, r$r_fpr5))
}
cat(sprintf("  %-6s %-12s %-12s %6s %6s %6s %6s %6s\n",
            "Paper","CSI","—","—","—","—","0.6100","—"))

cat(sprintf("\n[10_Evaluate.R] DONE: %s\n", format(Sys.time())))
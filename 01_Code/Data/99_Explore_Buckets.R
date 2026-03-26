#==============================================================================#
#==== explore_buckets.R =======================================================#
#==== Quick validation: can fundamentals predict terminal losers? =============#
#==============================================================================#
#
# PURPOSE:
#   Before integrating bucket prediction into the main pipeline, validate
#   whether the idea is feasible. Three quick tests:
#
#   TEST 1 — Feature separation
#     Do terminal losers and phoenixes look different in key fundamentals
#     BEFORE their outcome is known? If distributions don't separate,
#     prediction is hopeless regardless of model complexity.
#
#   TEST 2 — Simple logistic regression
#     Can a 5-feature logistic regression separate terminal losers from
#     phoenixes? Reports AUC, AP, and coefficient signs.
#     If AUC < 0.60, the signal is too weak to be useful.
#
#   TEST 3 — Temporal stability
#     Does the model trained on 1993–2005 predict 2006–2015 outcomes?
#     If signal degrades severely OOS, the concept is not viable.
#
# KEY DESIGN NOTE:
#   Long-run CAGR is inherently a LOOKAHEAD variable — only known ex-post.
#   This is fine: we're predicting "will this firm be a terminal loser?"
#   using features at year t. Outcome uses ALL future returns after year t.
#   This is the intended design, not a leak.
#
# INPUTS:  config.R | features_fund.rds | prices_monthly.rds
#
#==============================================================================#

source("config.R")

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(scales)
  library(pROC)
  library(PRROC)
})

cat("\n[explore_buckets.R] START:", format(Sys.time()), "\n")

## Parameters
CAGR_LOSER_THRESH   <- -0.02
CAGR_PHOENIX_THRESH <-  0.00
MIN_MONTHS_CAGR     <- 36L
SPLIT_YEAR          <- 2005L
TEST_FEATURES       <- c(
  "altman_z2", "leverage", "ocf_margin", "roll_min_3y_roic",
  "interest_coverage", "cash_ratio", "consec_decline_sale",
  "acct_mom_roa", "roll_sd_5y_earn_yld"
)

#==============================================================================#
# 1. Compute long-run CAGR per firm
#==============================================================================#

cat("\n[1] Computing long-run CAGR per firm...\n")

monthly <- as.data.table(readRDS(PATH_PRICES_MONTHLY))
setnames(monthly, "ret_adj", "ret")
monthly[, ret := pmin(pmax(ret, -0.99, na.rm=TRUE), 10, na.rm=TRUE)]

firm_cagr <- monthly[!is.na(ret), {
  n   <- .N
  cf  <- prod(1 + ret, na.rm=TRUE)
  if (n < MIN_MONTHS_CAGR || !is.finite(cf) || cf <= 0)
    .(cagr=NA_real_, n_months=n,
      last_year=max(year(date)), first_year=min(year(date)))
  else
    .(cagr=cf^(12/n)-1, n_months=n,
      last_year=max(year(date)), first_year=min(year(date)))
}, by=permno][!is.na(cagr)]

firm_cagr[, bucket := fcase(
  cagr <  CAGR_LOSER_THRESH,   "terminal_loser",
  cagr >= CAGR_PHOENIX_THRESH, "phoenix",
  default = "temporary_loser"
)]

firm_cagr_binary <- firm_cagr[bucket != "temporary_loser"]
firm_cagr_binary[, y_loser := as.integer(bucket == "terminal_loser")]

cat(sprintf("  Terminal losers: %d | Phoenixes: %d | Base rate: %.1f%%\n",
            sum(firm_cagr_binary$y_loser),
            sum(firm_cagr_binary$y_loser==0L),
            mean(firm_cagr_binary$y_loser)*100))

#==============================================================================#
# 2. Join features to outcomes
#==============================================================================#

cat("\n[2] Joining features...\n")

features <- as.data.table(readRDS(PATH_FEATURES_FUND))
avail    <- intersect(TEST_FEATURES, names(features))
missing  <- setdiff(TEST_FEATURES, names(features))
if (length(missing) > 0)
  cat(sprintf("  Missing: %s\n", paste(missing, collapse=", ")))
cat(sprintf("  Using: %s\n", paste(avail, collapse=", ")))

analysis_dt <- merge(
  features[, c("permno","year", avail), with=FALSE],
  firm_cagr_binary[, .(permno, y_loser, bucket, cagr, last_year)],
  by = "permno", all.x=FALSE
)[year < last_year]   ## only pre-outcome observations

cat(sprintf("  Dataset: %d obs | %d firms | years %d-%d\n",
            nrow(analysis_dt), uniqueN(analysis_dt$permno),
            min(analysis_dt$year), max(analysis_dt$year)))

#==============================================================================#
# TEST 1 — Feature separation (Mann-Whitney U)
#==============================================================================#

cat("\n[TEST 1] Feature separation\n\n")

sep_results <- lapply(avail, function(feat) {
  vl <- analysis_dt[y_loser==1L & !is.na(get(feat)), get(feat)]
  vp <- analysis_dt[y_loser==0L & !is.na(get(feat)), get(feat)]
  if (length(vl) < 10 || length(vp) < 10) return(NULL)
  wt <- wilcox.test(vl, vp, exact=FALSE)
  ## Winsorise means at 1–99%
  w  <- function(x) { q <- quantile(x, c(.01,.99)); pmax(pmin(x,q[2]),q[1]) }
  data.frame(feature=feat,
             mean_loser=round(mean(w(vl)),3),
             mean_phoenix=round(mean(w(vp)),3),
             direction=ifelse(mean(w(vl))<mean(w(vp)),"loser<phoenix","loser>phoenix"),
             p_value=signif(wt$p.value,3),
             sig=wt$p.value<0.01,
             stringsAsFactors=FALSE)
})
sep_table <- do.call(rbind, Filter(Negate(is.null), sep_results))

cat(sprintf("  %-28s | %11s | %13s | %-14s | %9s | %s\n",
            "Feature","Mean(loser)","Mean(phoenix)","Direction","p-value","Sig"))
cat(sprintf("  %s\n", paste(rep("-",90), collapse="")))
for (i in seq_len(nrow(sep_table))) {
  r <- sep_table[i,]
  cat(sprintf("  %-28s | %11.3f | %13.3f | %-14s | %9s | %s\n",
              r$feature, r$mean_loser, r$mean_phoenix, r$direction,
              r$p_value, if(r$sig) "***" else ""))
}

## Density plot of top 3 significant features
sig_feats <- sep_table[sep_table$sig==TRUE, "feature"]
cat(sprintf("\n  %d of %d features significant (p<0.01)\n",
            length(sig_feats), nrow(sep_table)))

if (length(sig_feats) >= 1) {
  top3 <- head(sig_feats, 3)
  plot_dt <- rbindlist(lapply(top3, function(f) {
    dt <- analysis_dt[!is.na(get(f)), .(val=get(f), bucket)]
    q  <- quantile(dt$val, c(.02,.98), na.rm=TRUE)
    dt[val >= q[1] & val <= q[2]][, feature := f]
  }))
  
  p_sep <- ggplot(plot_dt, aes(x=val, fill=bucket, colour=bucket)) +
    geom_density(alpha=0.35, linewidth=0.7) +
    facet_wrap(~feature, scales="free", ncol=1) +
    scale_fill_manual(values=c(terminal_loser="#E53935", phoenix="#2196F3")) +
    scale_colour_manual(values=c(terminal_loser="#E53935", phoenix="#2196F3")) +
    labs(title="Feature Separation: Terminal Loser vs Phoenix",
         subtitle="Separated distributions → feature is predictive",
         x=NULL, y="Density", fill=NULL, colour=NULL) +
    theme_minimal(base_size=11) +
    theme(legend.position="bottom")
  
  ggsave(file.path(DIR_FIGURES, "explore_feature_separation.png"), p_sep,
         width=PLOT_WIDTH, height=PLOT_HEIGHT*1.5, dpi=PLOT_DPI)
  cat("  explore_feature_separation.png saved.\n")
}

#==============================================================================#
# TEST 2 — Logistic regression (full period)
#==============================================================================#

cat("\n[TEST 2] Logistic regression (full period)\n\n")

if (length(sig_feats) == 0L) {
  cat("  No significant features — skipping.\n")
  auc_lr <- NA; ap_lr <- NA
} else {
  lr_data <- analysis_dt[, c("permno","year","y_loser", sig_feats),
                         with=FALSE]
  lr_data <- lr_data[complete.cases(lr_data[, sig_feats, with=FALSE])]
  
  ## Standardise
  for (f in sig_feats) {
    m <- mean(lr_data[[f]], na.rm=TRUE)
    s <- sd(lr_data[[f]], na.rm=TRUE)
    if (s > 0) lr_data[, (f) := (get(f)-m)/s]
  }
  
  formula_lr <- as.formula(paste("y_loser ~",
                                 paste(sig_feats, collapse="+")))
  lr_model   <- glm(formula_lr, data=lr_data,
                    family=binomial(link="logit"))
  
  coef_dt <- as.data.table(summary(lr_model)$coefficients,
                           keep.rownames=TRUE)
  setnames(coef_dt, c("term","est","se","z","pval"))
  coef_dt[, OR  := round(exp(est), 3)]
  coef_dt[, est := round(est, 4)]
  coef_dt[, sig := fcase(pval<.001,"***", pval<.01,"**",
                         pval<.05,"*", default="")]
  print(coef_dt[, .(term, est, OR, pval=signif(pval,3), sig)],
        row.names=FALSE)
  
  lr_data[, p_loser := predict(lr_model, type="response")]
  
  auc_lr <- as.numeric(pROC::auc(
    pROC::roc(lr_data$y_loser, lr_data$p_loser, quiet=TRUE)))
  ap_lr  <- PRROC::pr.curve(
    scores.class0=lr_data[y_loser==1L, p_loser],
    scores.class1=lr_data[y_loser==0L, p_loser],
    curve=FALSE)$auc.integral
  
  cat(sprintf("\n  Full-period: AUC=%.4f | AP=%.4f | Baseline AP=%.4f | Lift=%.2fx\n",
              auc_lr, ap_lr, mean(lr_data$y_loser),
              ap_lr/mean(lr_data$y_loser)))
}

#==============================================================================#
# TEST 3 — Temporal stability
#==============================================================================#

cat("\n[TEST 3] Temporal stability (train <=", SPLIT_YEAR,
    "| test >", SPLIT_YEAR, ")\n\n")

if (length(sig_feats) == 0L) {
  cat("  Skipping — no significant features.\n")
} else {
  lr_ts <- analysis_dt[, c("permno","year","y_loser", sig_feats),
                       with=FALSE]
  lr_ts <- lr_ts[complete.cases(lr_ts[, sig_feats, with=FALSE])]
  
  train_ts <- lr_ts[year <= SPLIT_YEAR]
  test_ts  <- lr_ts[year >  SPLIT_YEAR]
  
  ## Standardise on train only
  scales_list <- list()
  for (f in sig_feats) {
    m <- mean(train_ts[[f]], na.rm=TRUE)
    s <- sd(train_ts[[f]], na.rm=TRUE)
    scales_list[[f]] <- c(m=m, s=s)
    if (s > 0) {
      train_ts[, (f) := (get(f)-m)/s]
      test_ts[,  (f) := (get(f)-m)/s]
    }
  }
  
  cat(sprintf("  Train: %d obs | %d losers | %d phoenixes\n",
              nrow(train_ts), sum(train_ts$y_loser),
              sum(train_ts$y_loser==0L)))
  cat(sprintf("  Test:  %d obs | %d losers | %d phoenixes\n",
              nrow(test_ts), sum(test_ts$y_loser),
              sum(test_ts$y_loser==0L)))
  
  lr_fit <- glm(formula_lr, data=train_ts,
                family=binomial(link="logit"))
  
  auc_train <- as.numeric(pROC::auc(pROC::roc(
    train_ts$y_loser,
    predict(lr_fit, type="response"), quiet=TRUE)))
  
  test_ts[, p_loser := predict(lr_fit, newdata=test_ts, type="response")]
  
  auc_ts <- as.numeric(pROC::auc(
    pROC::roc(test_ts$y_loser, test_ts$p_loser, quiet=TRUE)))
  ap_ts  <- PRROC::pr.curve(
    scores.class0=test_ts[y_loser==1L, p_loser],
    scores.class1=test_ts[y_loser==0L, p_loser],
    curve=FALSE)$auc.integral
  
  cat(sprintf("\n  Train AUC=%.4f | Test AUC=%.4f | Degradation=%.4f\n",
              auc_train, auc_ts, auc_train-auc_ts))
  cat(sprintf("  Test AP=%.4f | Baseline=%.4f | Lift=%.2fx\n",
              ap_ts, mean(test_ts$y_loser),
              ap_ts/mean(test_ts$y_loser)))
  
  ## Decile table
  test_ts[, decile := cut(p_loser,
                          breaks=quantile(p_loser, probs=seq(0,1,.1)),
                          labels=1:10, include.lowest=TRUE)]
  decile_dt <- test_ts[!is.na(decile), .(
    n=.N, pct_loser=round(mean(y_loser)*100,1)
  ), by=decile][order(as.integer(as.character(decile)))]
  
  cat("\n  Decile analysis (test set — higher decile = higher predicted risk):\n\n")
  cat(sprintf("  %6s | %6s | %s\n", "Decile","N","% Terminal Loser"))
  cat(sprintf("  %s\n", paste(rep("-",35),collapse="")))
  for (i in seq_len(nrow(decile_dt))) {
    r <- decile_dt[i,]
    bar <- paste(rep("█", round(r$pct_loser/2)), collapse="")
    cat(sprintf("  %6s | %6d | %5.1f%%  %s\n",
                r$decile, r$n, r$pct_loser, bar))
  }
  cat(sprintf("\n  Top vs bottom decile loser rate: %.1f%% vs %.1f%% (%.1fx)\n",
              decile_dt[decile==10, pct_loser],
              decile_dt[decile==1,  pct_loser],
              decile_dt[decile==10, pct_loser] /
                max(decile_dt[decile==1, pct_loser], 0.1)))
  
  ## OOS prediction density plot
  p_oos <- ggplot(test_ts, aes(x=p_loser, fill=factor(y_loser),
                               colour=factor(y_loser))) +
    geom_density(alpha=0.35, linewidth=0.7) +
    scale_fill_manual(values=c("0"="#2196F3","1"="#E53935"),
                      labels=c("0"="Phoenix","1"="Terminal Loser")) +
    scale_colour_manual(values=c("0"="#2196F3","1"="#E53935"),
                        labels=c("0"="Phoenix","1"="Terminal Loser")) +
    scale_x_continuous(labels=percent_format(accuracy=1),
                       name="Predicted P(Terminal Loser)") +
    labs(title="OOS Predicted Probability by Actual Outcome",
         subtitle=sprintf("Test period: year > %d | AUC=%.3f | AP=%.3f",
                          SPLIT_YEAR, auc_ts, ap_ts),
         y="Density", fill=NULL, colour=NULL) +
    theme_minimal(base_size=12) +
    theme(legend.position="bottom")
  
  ggsave(file.path(DIR_FIGURES, "explore_oos_prediction.png"), p_oos,
         width=PLOT_WIDTH, height=PLOT_HEIGHT, dpi=PLOT_DPI)
  cat("  explore_oos_prediction.png saved.\n")
}

#==============================================================================#
# Final assessment
#==============================================================================#

cat("\n[explore_buckets.R] ══════════════════════════════════\n")
cat("  FEASIBILITY VERDICT\n")
cat("  ══════════════════════════════════════════════════\n\n")

n_sig <- sum(sep_table$sig)
cat(sprintf("  Feature separation : %d/%d significant (p<0.01) %s\n",
            n_sig, nrow(sep_table),
            if(n_sig>=3) "✓" else "✗"))

if (exists("auc_lr") && !is.na(auc_lr))
  cat(sprintf("  Full-period AUC    : %.3f %s\n", auc_lr,
              if(auc_lr>0.70) "✓ STRONG" else if(auc_lr>0.60) "~ MODERATE"
              else "✗ WEAK"))

if (exists("auc_ts") && !is.na(auc_ts))
  cat(sprintf("  OOS AUC            : %.3f %s\n", auc_ts,
              if(auc_ts>0.65) "✓ STABLE" else if(auc_ts>0.55) "~ MARGINAL"
              else "✗ UNSTABLE"))

cat("\n  RECOMMENDATION:\n")
proceed <- exists("auc_ts") && !is.na(auc_ts) && auc_ts > 0.60 && n_sig >= 3
if (proceed) {
  cat("  ✓ Signal is viable. Proceed with bucket prediction.\n")
  cat("    Next: train AutoGluon on y_loser target, build\n")
  cat("    concentrated anti-loser portfolio from bottom decile.\n")
} else if (exists("auc_ts") && !is.na(auc_ts) && auc_ts > 0.55) {
  cat("  ~ Marginal signal. Consider using as second-stage filter\n")
  cat("    on top of CSI prediction rather than standalone model.\n")
} else {
  cat("  ✗ Signal too weak. Stick with S4 (M1 + zombie filter).\n")
}

cat(sprintf("\n[explore_buckets.R] DONE: %s\n", format(Sys.time())))
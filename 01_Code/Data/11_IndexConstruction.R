#==============================================================================#
#==== 11_Results.R ============================================================#
#==== Crash-Filtered Index Construction and Backtest =========================#
#==============================================================================#
#
# PURPOSE:
#   Construct and backtest four equity index strategies using M1 and M3
#   AutoGluon predictions. Compare risk-adjusted returns against an
#   unfiltered pseudo-Russell 3000 benchmark.
#
# STRATEGIES:
#   Benchmark : Unfiltered pseudo-Russell 3000 (top N by mktcap, annual rebal)
#   S1        : M1-filtered  — exclude firms with p_csi(M1) > THRESH_FPR5
#   S2        : M3-filtered  — exclude firms with p_csi(M3) > THRESH_FPR5
#   S3        : Combined     — exclude if M1 OR M3 flags (union)
#
# WEIGHTING:
#   Both equal-weight (EW) and market-cap weight (CW) are computed.
#
# REBALANCING:
#   Annual, at end of December each year.
#   Predictions at year t drive the portfolio held Jan–Dec of year t+1.
#
# PERIODS:
#   In-sample  : 1998–2015 (uses train-period predictions — less honest)
#   Honest OOS : 2016–2024 (uses test + OOS predictions only)
#   Results reported separately and combined.
#
# TRANSACTION COSTS:
#   TRANSACTION_COST_BPS = 0  (one-way, per trade)
#   Set > 0 to model realistic costs (e.g. 5 bps one-way = 10 bps round-trip)
#
# INPUTS:
#   - config.R
#   - panel_raw.rds                              monthly price data (if ret exists)
#   - DIR_TABLES/ag_fund/ag_preds_test.parquet   M1 test predictions
#   - DIR_TABLES/ag_fund/ag_preds_oos.parquet    M1 OOS predictions
#   - DIR_TABLES/ag_preds_test.parquet           M3 test predictions (root)
#   - DIR_TABLES/ag_preds_oos.parquet            M3 OOS predictions (root)
#   - DIR_TABLES/ag_fund/ag_cv_results.parquet   M1 train-period CV predictions
#   - DIR_TABLES/eval_threshold.rds              calibrated thresholds
#   - Optional: monthly_returns.rds              pre-joined CRSP monthly returns
#
# OUTPUTS:
#   - DIR_TABLES/index_weights.rds               annual weights per strategy
#   - DIR_TABLES/index_returns.rds               monthly portfolio returns
#   - DIR_TABLES/index_performance.rds           summary performance table
#   - DIR_FIGURES/index_cumulative_ew.png        cumulative return (EW)
#   - DIR_FIGURES/index_cumulative_cw.png        cumulative return (CW)
#   - DIR_FIGURES/index_annual_returns.png       annual return bars
#   - DIR_FIGURES/index_drawdown.png             drawdown chart
#   - DIR_FIGURES/index_exclusion_count.png      firms excluded per year
#   - DIR_FIGURES/index_csi_avoided.png          CSI events avoided vs cost
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
})

cat("\n[11_Results.R] START:", format(Sys.time()), "\n")

#==============================================================================#
# 0. Parameters — adjust here
#==============================================================================#

## Universe
UNIVERSE_SIZE        <- 3000L      # pseudo-Russell 3000 (top N by mktcap)
MIN_MKTCAP_MM        <- 100        # minimum market cap $100M to enter universe

## Threshold — FPR=5% calibrated on test set
## These are read from eval_threshold.rds below (auto-populated)
## Override manually here if needed:
THRESH_M1_OVERRIDE   <- NULL       # e.g. 0.2967 — set NULL to use calibrated
THRESH_M3_OVERRIDE   <- NULL       # e.g. 0.0145
EXCLUSION_RATE_M1 <- 0.05   # exclude top 5% by M1 score
EXCLUSION_RATE_M3 <- 0.05   # exclude top 5% by M3 score

## Transaction costs (one-way, basis points)
TRANSACTION_COST_BPS <- 0          # 0 = frictionless | 5 = realistic

## Rebalancing
REBAL_MONTH          <- 12L        # December rebalancing

## Backtest periods
INSAMPLE_START       <- 1998L
INSAMPLE_END         <- 2015L
OOS_START            <- 2016L
OOS_END              <- 2022L      # 2023 excluded — right-censoring

## Combined strategy
COMBINED_LOGIC       <- "union"    # "union" = M1 OR M3 | "intersection" = M1 AND M3

#==============================================================================#
# 1. Load calibrated thresholds
#==============================================================================#

cat("[11] Loading calibrated thresholds...\n")

thresh_path <- file.path(DIR_TABLES, "eval_threshold.rds")
if (!file.exists(thresh_path))
  stop("[11] eval_threshold.rds not found — run 10_Evaluate.R first.")

eval_threshold <- readRDS(thresh_path)

## Extract FPR=5% thresholds
get_thresh <- function(model_label, override) {
  if (!is.null(override)) return(override)
  row <- eval_threshold[eval_threshold$model == model_label &
                          eval_threshold$fpr_target == 0.05, ]
  if (nrow(row) == 0L)
    stop(sprintf("[11] No FPR=5%% threshold found for model: %s", model_label))
  row$threshold
}

THRESH_M1 <- get_thresh("M1 AG Fund", THRESH_M1_OVERRIDE)
THRESH_M3 <- get_thresh("M3 AG Raw",  THRESH_M3_OVERRIDE)

cat(sprintf("  M1 threshold (FPR=5%%): %.4f\n", THRESH_M1))
cat(sprintf("  M3 threshold (FPR=5%%): %.4f\n", THRESH_M3))

#==============================================================================#
# 2. Load predictions — M1 and M3
#==============================================================================#

cat("\n[11] Loading model predictions...\n")

load_preds <- function(paths, model_name) {
  parts <- lapply(paths, function(p) {
    if (!file.exists(p)) {
      cat(sprintf("  [%s] Not found: %s\n", model_name, p))
      return(NULL)
    }
    as.data.table(arrow::read_parquet(p))
  })
  parts <- Filter(Negate(is.null), parts)
  if (length(parts) == 0L) stop(sprintf("[11] No predictions found for %s", model_name))
  dt <- rbindlist(parts)
  setnames(dt, "p_csi", paste0("p_csi_", tolower(model_name)))
  dt[, .(permno, year, y, p_csi = get(paste0("p_csi_", tolower(model_name))))]
}

## M1 — test + OOS (honest) and CV (in-sample)
m1_honest <- load_preds(
  c(file.path(DIR_TABLES, "ag_fund", "ag_preds_test.parquet"),
    file.path(DIR_TABLES, "ag_fund", "ag_preds_oos.parquet")),
  "M1"
)
m1_honest[, period := "honest"]

## M1 in-sample CV predictions (if available)
m1_cv_path <- file.path(DIR_TABLES, "ag_fund", "ag_cv_results.parquet")
if (file.exists(m1_cv_path)) {
  m1_cv <- as.data.table(arrow::read_parquet(m1_cv_path))
  m1_cv <- m1_cv[, .(permno, year, y, p_csi = p_csi)]
  m1_cv[, period := "insample"]
} else {
  cat("  [M1] CV predictions not found — in-sample period will use honest preds only\n")
  m1_cv <- NULL
}

## M3 — test + OOS
m3_honest <- load_preds(
  c(file.path(DIR_TABLES, "ag_preds_test.parquet"),
    file.path(DIR_TABLES, "ag_preds_oos.parquet")),
  "M3"
)
m3_honest[, period := "honest"]

## M3 in-sample CV predictions
m3_cv_path <- file.path(DIR_TABLES, "ag_cv_results.parquet")
if (file.exists(m3_cv_path)) {
  m3_cv <- as.data.table(arrow::read_parquet(m3_cv_path))
  m3_cv <- m3_cv[, .(permno, year, y, p_csi = p_csi)]
  m3_cv[, period := "insample"]
} else {
  cat("  [M3] CV predictions not found — in-sample period will use honest preds only\n")
  m3_cv <- NULL
}

## Combine all predictions
m1_all <- rbindlist(list(m1_cv, m1_honest), use.names = TRUE)
m3_all <- rbindlist(list(m3_cv, m3_honest), use.names = TRUE)

setnames(m1_all, "p_csi", "p_m1")
setnames(m3_all, "p_csi", "p_m3")

## Join M1 and M3 predictions
## y (true CSI label) carried from m1_all — identical in m3_all
preds_all <- merge(
  m1_all[, .(permno, year, y, p_m1, period_m1 = period)],
  m3_all[, .(permno, year, p_m3, period_m3 = period)],
  by = c("permno", "year"),
  all = TRUE
)

cat(sprintf("  Predictions joined: %d rows | years %d–%d\n",
            nrow(preds_all),
            min(preds_all$year), max(preds_all$year)))

#==============================================================================#
# 3. Load monthly returns
#==============================================================================#

cat("\n[11] Loading monthly returns...\n")

## Load directly from PATH_PRICES_MONTHLY (prices_monthly.rds)
## Columns: permno, date, price, ret_adj, ret_excl_div,
##          div_amount, mktcap, vol, shrout, dlstcd, dlret_applied
monthly <- as.data.table(readRDS(PATH_PRICES_MONTHLY))

## Standardise column names used throughout this script
## ret_adj → ret  (total return including dividends)
## mktcap  → mkvalt (market cap $000s: price * shrout)
setnames(monthly, "ret_adj", "ret")
setnames(monthly, "mktcap",  "mkvalt")

## Add year and month
monthly[, year  := year(date)]
monthly[, month := month(date)]

## Parse dates if needed
if (!inherits(monthly$date, "Date"))
  monthly[, date := as.Date(date)]

## Cap returns at extreme values (data errors / delisting artifacts)
monthly[, ret := pmin(pmax(ret, -0.99, na.rm=TRUE), 10, na.rm=TRUE)]


#==============================================================================#
# 4. Build annual universe — pseudo-Russell 3000
#==============================================================================#

cat("\n[11] Building annual pseudo-Russell 3000 universe...\n")

## Use December market cap for each year to determine universe membership
## Market cap in panel is at annual (fiscal year end) grain
## Use monthly mkvalt if available, else fall back to annual panel

if ("mkvalt" %in% names(monthly) && !all(is.na(monthly$mkvalt))) {
  
  ## Use December mktcap from monthly data
  dec_mktcap <- monthly[month == REBAL_MONTH & !is.na(mkvalt),
                        .(mkvalt_dec = mean(mkvalt, na.rm=TRUE)),
                        by = .(permno, year)]
  
} else {
  
  ## Fall back: load mkvalt from features or panel
  cat("  mkvalt not in monthly — loading from panel_raw.rds...\n")
  panel <- as.data.table(readRDS(PATH_PANEL_RAW))
  
  if ("mkvalt" %in% names(panel)) {
    dec_mktcap <- panel[, .(mkvalt_dec = mean(mkvalt, na.rm=TRUE)),
                        by = .(permno, year)]
  } else if ("log_mkvalt" %in% names(panel)) {
    dec_mktcap <- panel[, .(mkvalt_dec = mean(exp(log_mkvalt), na.rm=TRUE)),
                        by = .(permno, year)]
    cat("  Using exp(log_mkvalt) as mkvalt proxy\n")
  } else {
    stop("[11] Cannot find market cap — need mkvalt or log_mkvalt in panel_raw.rds")
  }
}

## Build universe per year: top UNIVERSE_SIZE by mkvalt, min cap filter
universe <- dec_mktcap[mkvalt_dec >= MIN_MKTCAP_MM]
universe[, rank_mkvalt := frank(-mkvalt_dec, ties.method = "first"),
         by = year]
universe <- universe[rank_mkvalt <= UNIVERSE_SIZE]
universe[, in_universe := TRUE]

cat(sprintf("  Universe built: %d firm-years | avg %.0f firms/year\n",
            nrow(universe),
            nrow(universe) / uniqueN(universe$year)))

#==============================================================================#
# 5. Apply exclusion rules — build annual weights
#==============================================================================#

cat("\n[11] Applying exclusion rules and building weights...\n")

## Join universe with predictions
## Prediction at year t → portfolio held in year t+1
## So prediction year maps to PORTFOLIO year = pred_year + 1

ann <- merge(universe[, .(permno, year, mkvalt_dec, rank_mkvalt)],
             preds_all[, .(permno, year, p_m1, p_m3, period_m1, period_m3)],
             by = c("permno", "year"),
             all.x = TRUE)

## Exclusion flags at threshold

ann[, flag_m1 := rank(-p_m1)/n > (1 - EXCLUSION_RATE_M1)]
ann[, flag_m3 := rank(-p_m3)/n > (1 - EXCLUSION_RATE_M3)]

ann[, flag_s1 := flag_m1]
ann[, flag_s2 := flag_m3]
ann[, flag_s3 := if (COMBINED_LOGIC == "union")
  flag_m1 | flag_m3
  else
    flag_m1 & flag_m3]

## Portfolio year = year + 1 (prediction drives next year's holdings)
ann[, port_year := year + 1L]

## For each strategy, a firm is INCLUDED if not flagged AND prediction exists
## Firms with no prediction are included (conservative: don't exclude unknowns)
ann[, incl_bench := TRUE]
ann[, incl_s1    := !flag_s1]
ann[, incl_s2    := !flag_s2]
ann[, incl_s3    := !flag_s3]

## Compute weights per strategy per port_year
fn_weights <- function(dt, incl_col, weight_type) {
  dt_incl <- dt[get(incl_col) == TRUE]
  if (weight_type == "ew") {
    dt_incl[, w := 1.0 / .N, by = port_year]
  } else {
    dt_incl[, w := mkvalt_dec / sum(mkvalt_dec, na.rm=TRUE), by = port_year]
  }
  dt_incl[, .(permno, port_year, mkvalt_dec, w,
              flag_m1, flag_m3, p_m1, p_m3)]
}

strategies <- c("bench", "s1", "s2", "s3")
weight_types <- c("ew", "cw")

weights_list <- list()
for (strat in strategies) {
  for (wt in weight_types) {
    key <- paste0(strat, "_", wt)
    weights_list[[key]] <- fn_weights(ann, paste0("incl_", strat), wt)
    weights_list[[key]][, strategy := strat]
    weights_list[[key]][, weighting := wt]
  }
}

weights_all <- rbindlist(weights_list)

## Summary: firms per year per strategy
excl_summary <- ann[, .(
  n_universe = .N,
  n_pred_m1  = sum(!is.na(p_m1)),
  n_pred_m3  = sum(!is.na(p_m3)),
  n_flag_m1  = sum(flag_m1, na.rm=TRUE),
  n_flag_m3  = sum(flag_m3, na.rm=TRUE),
  n_flag_s3  = sum(flag_s3, na.rm=TRUE),
  n_incl_bench = sum(incl_bench),
  n_incl_s1    = sum(incl_s1),
  n_incl_s2    = sum(incl_s2),
  n_incl_s3    = sum(incl_s3)
), by = year][order(year)]

cat("\n  Exclusion summary (first 10 years shown):\n")
print(head(excl_summary, 10L), row.names = FALSE)

saveRDS(weights_all,   file.path(DIR_TABLES, "index_weights.rds"))
saveRDS(excl_summary,  file.path(DIR_TABLES, "index_exclusion_summary.rds"))

#==============================================================================#
# 6. Compute monthly portfolio returns
#==============================================================================#

cat("\n[11] Computing monthly portfolio returns...\n")

## Join weights to monthly returns
## weights port_year = year the portfolio is HELD
## monthly year = calendar year of the return observation
## → join on permno + (monthly$year == weights$port_year)

returns_list <- list()

for (strat in strategies) {
  for (wt in weight_types) {
    key <- paste0(strat, "_", wt)
    w   <- weights_list[[key]]
    
    ## Join monthly returns to weights
    port_monthly <- merge(
      monthly[, .(permno, date, year, month, ret)],
      w[, .(permno, port_year, w)],
      by.x = c("permno", "year"),
      by.y = c("permno", "port_year"),
      all.y = FALSE
    )
    port_monthly <- port_monthly[!is.na(ret) & !is.na(w)]
    
    ## Apply transaction costs at rebalancing month
    ## Cost = TRANSACTION_COST_BPS / 10000 applied to each position change
    ## Simplified: deduct flat cost from December return for non-zero cost
    if (TRANSACTION_COST_BPS > 0) {
      port_monthly[month == REBAL_MONTH,
                   ret := ret - TRANSACTION_COST_BPS / 10000]
    }
    
    ## Weighted portfolio return per month
    port_ret <- port_monthly[, .(
      port_ret   = sum(w * ret, na.rm=TRUE),
      n_holdings = .N,
      strategy   = strat,
      weighting  = wt
    ), by = .(date, year, month)]
    
    returns_list[[key]] <- port_ret
  }
}

port_returns <- rbindlist(returns_list)
port_returns <- port_returns[order(strategy, weighting, date)]

saveRDS(port_returns, file.path(DIR_TABLES, "index_returns.rds"))
cat(sprintf("  Monthly returns computed: %d strategy-months\n", nrow(port_returns)))

#==============================================================================#
# 7. Performance metrics
#==============================================================================#

cat("\n[11] Computing performance metrics...\n")

fn_performance <- function(ret_vec, dates, rf_annual = 0.03) {
  
  ret_vec <- ret_vec[!is.na(ret_vec)]
  if (length(ret_vec) < 12L) return(NULL)
  
  n_months   <- length(ret_vec)
  n_years    <- n_months / 12
  rf_monthly <- (1 + rf_annual)^(1/12) - 1
  
  cum_ret    <- prod(1 + ret_vec) - 1
  cagr       <- (1 + cum_ret)^(1/n_years) - 1
  vol_ann    <- sd(ret_vec) * sqrt(12)
  excess_ret <- ret_vec - rf_monthly
  sharpe     <- mean(excess_ret) / sd(excess_ret) * sqrt(12)
  
  ## Sortino — downside deviation only
  dd_ret     <- ret_vec[ret_vec < rf_monthly]
  sortino    <- if (length(dd_ret) > 1)
    mean(excess_ret) / (sd(dd_ret) * sqrt(12)) else NA_real_
  
  ## Maximum drawdown
  cum_index  <- cumprod(1 + ret_vec)
  peak       <- cummax(cum_index)
  drawdowns  <- (cum_index - peak) / peak
  max_dd     <- min(drawdowns)
  
  ## Calmar
  calmar     <- if (max_dd != 0) cagr / abs(max_dd) else NA_real_
  
  ## Win rate
  win_rate   <- mean(ret_vec > 0)
  
  data.frame(
    n_months   = n_months,
    cum_ret    = round(cum_ret,  4),
    cagr       = round(cagr,     4),
    vol_ann    = round(vol_ann,  4),
    sharpe     = round(sharpe,   4),
    sortino    = round(sortino,  4),
    max_dd     = round(max_dd,   4),
    calmar     = round(calmar,   4),
    win_rate   = round(win_rate, 4)
  )
}

## Compute for each strategy × weighting × period
perf_rows <- list()

periods <- list(
  full     = c(INSAMPLE_START, OOS_END),
  insample = c(INSAMPLE_START, INSAMPLE_END),
  oos      = c(OOS_START, OOS_END)
)

for (strat in strategies) {
  for (wt in weight_types) {
    key <- paste0(strat, "_", wt)
    ret_dt <- port_returns[strategy == strat & weighting == wt]
    
    for (per_name in names(periods)) {
      per <- periods[[per_name]]
      ret_sub <- ret_dt[year >= per[1] & year <= per[2]]
      if (nrow(ret_sub) < 12L) next
      
      pf <- fn_performance(ret_sub$port_ret, ret_sub$date)
      if (is.null(pf)) next
      
      pf$strategy  <- strat
      pf$weighting <- wt
      pf$period    <- per_name
      perf_rows[[length(perf_rows)+1]] <- pf
    }
  }
}

perf_table <- do.call(rbind, perf_rows)

cat("\n  Performance table (OOS, equal-weight):\n\n")
oos_ew <- perf_table[perf_table$weighting == "ew" &
                       perf_table$period    == "oos", ]
print(oos_ew[, c("strategy","weighting","period",
                 "cagr","vol_ann","sharpe","max_dd","calmar")],
      row.names = FALSE)

saveRDS(perf_table, file.path(DIR_TABLES, "index_performance.rds"))

#==============================================================================#
# 8. CSI avoided — how many true CSI events did each strategy prevent?
#==============================================================================#

cat("\n[11] Computing CSI avoidance statistics...\n")

## Join labels to exclusion flags
## y = 1 means the firm actually became a CSI in the NEXT year
## flag = 1 means the model flagged it for exclusion

csi_audit <- merge(
  ann[, .(permno, year, port_year,
          flag_s1, flag_s2, flag_s3, incl_s1, incl_s2, incl_s3,
          p_m1, p_m3, mkvalt_dec)],
  ## y comes entirely from predictions (label shift: y at t = CSI in t+1)
  preds_all[, .(permno, year, y)],
  by = c("permno", "year"),
  all.x = TRUE
)

## y at year t = did firm enter CSI in year t+1 (label shift already applied)
csi_audit <- csi_audit[!is.na(y)]

csi_stats <- list()
for (strat in c("s1", "s2", "s3")) {
  flag_col <- paste0("flag_", strat)
  incl_col <- paste0("incl_", strat)
  
  dt <- csi_audit[, .(
    n_total      = .N,
    n_csi        = sum(y == 1L, na.rm=TRUE),
    n_flagged    = sum(get(flag_col), na.rm=TRUE),
    tp           = sum(get(flag_col) == TRUE  & y == 1L, na.rm=TRUE),  ## correctly excluded
    fp           = sum(get(flag_col) == TRUE  & y == 0L, na.rm=TRUE),  ## wrongly excluded
    fn           = sum(get(flag_col) == FALSE & y == 1L, na.rm=TRUE),  ## missed CSI
    tn           = sum(get(flag_col) == FALSE & y == 0L, na.rm=TRUE)   ## correctly included
  )]
  
  dt[, strategy   := strat]
  dt[, recall     := round(tp / (tp + fn), 4)]    ## CSI events caught
  dt[, precision  := round(tp / (tp + fp), 4)]    ## of exclusions, how many were real CSI
  dt[, fpr        := round(fp / (fp + tn), 4)]    ## false exclusion rate
  dt[, f1         := round(2*recall*precision / (recall+precision), 4)]
  csi_stats[[strat]] <- dt
}

csi_table <- rbindlist(csi_stats)
cat("\n  CSI avoidance table (full period):\n\n")
print(csi_table, row.names = FALSE)
saveRDS(csi_table, file.path(DIR_TABLES, "index_csi_avoidance.rds"))

#==============================================================================#
# 9. Plots
#==============================================================================#

cat("\n[11] Generating plots...\n")

## Strategy labels for display
STRAT_LABELS <- c(
  bench = "Benchmark (Unfiltered)",
  s1    = "S1: M1 Ex-Ante Screen",
  s2    = "S2: M3 Triage Filter",
  s3    = "S3: M1 + M3 Combined"
)

STRAT_COLOURS <- c(
  bench = "#9E9E9E",
  s1    = "#2196F3",
  s2    = "#F44336",
  s3    = "#4CAF50"
)

##──────────────────────────────────────────────────────────────────────────────
## 9A. Cumulative return — equal-weight
##──────────────────────────────────────────────────────────────────────────────

fn_cum_ret_plot <- function(wt_type, title_suffix) {
  
  ret_dt <- port_returns[weighting == wt_type]
  ret_dt <- ret_dt[order(strategy, date)]
  
  ## Cumulative index (base = 1 at start)
  ret_dt[, cum_idx := cumprod(1 + port_ret), by = .(strategy, weighting)]
  
  ## Honest OOS start date for annotation
  oos_start_date <- as.Date(sprintf("%d-01-01", OOS_START))
  
  ret_dt[, strat_label := STRAT_LABELS[strategy]]
  ret_dt[, strat_label := factor(strat_label, levels = STRAT_LABELS)]
  
  p <- ggplot(ret_dt, aes(x=date, y=cum_idx,
                          colour=strategy, group=strategy)) +
    geom_line(linewidth=0.9) +
    geom_vline(xintercept=as.numeric(oos_start_date),
               linetype="dashed", colour="grey40", linewidth=0.7) +
    annotate("text", x=oos_start_date, y=max(ret_dt$cum_idx, na.rm=TRUE)*0.95,
             label="OOS start\n(2016)", hjust=-0.1, size=3, colour="grey40") +
    geom_vline(xintercept=as.numeric(as.Date(sprintf("%d-01-01", INSAMPLE_END+1))),
               linetype="dotted", colour="grey60", linewidth=0.5) +
    scale_colour_manual(values=STRAT_COLOURS, labels=STRAT_LABELS) +
    scale_y_continuous(labels=scales::dollar_format(prefix="$"),
                       name="Portfolio Value ($1 invested)") +
    scale_x_date(date_breaks="2 years", date_labels="%Y") +
    labs(
      title    = paste0("Crash-Filtered Index — Cumulative Return (", title_suffix, ")"),
      subtitle = paste0("Pseudo-Russell 3000 universe | Annual rebalancing | ",
                        "Dashed = OOS start (2016)"),
      x        = NULL,
      colour   = "Strategy"
    ) +
    theme_minimal(base_size=12) +
    theme(legend.position="bottom",
          axis.text.x=element_text(angle=30, hjust=1))
  
  p
}

p_cum_ew <- fn_cum_ret_plot("ew", "Equal-Weight")
ggsave(file.path(DIR_FIGURES, "index_cumulative_ew.png"), p_cum_ew,
       width=PLOT_WIDTH*1.2, height=PLOT_HEIGHT, dpi=PLOT_DPI)
cat("  index_cumulative_ew.png saved.\n")

p_cum_cw <- fn_cum_ret_plot("cw", "Cap-Weight")
ggsave(file.path(DIR_FIGURES, "index_cumulative_cw.png"), p_cum_cw,
       width=PLOT_WIDTH*1.2, height=PLOT_HEIGHT, dpi=PLOT_DPI)
cat("  index_cumulative_cw.png saved.\n")

##──────────────────────────────────────────────────────────────────────────────
## 9B. Annual returns bar chart — equal-weight
##──────────────────────────────────────────────────────────────────────────────

ann_ret <- port_returns[weighting == "ew",
                        .(ann_ret = prod(1 + port_ret) - 1),
                        by = .(strategy, year)]
ann_ret[, strat_label := STRAT_LABELS[strategy]]

p_ann <- ggplot(ann_ret,
                aes(x=year, y=ann_ret, fill=strategy)) +
  geom_col(position=position_dodge(width=0.8), width=0.7, alpha=0.85) +
  geom_hline(yintercept=0, colour="black", linewidth=0.4) +
  geom_vline(xintercept=OOS_START - 0.5,
             linetype="dashed", colour="grey40", linewidth=0.7) +
  annotate("text", x=OOS_START - 0.4, y=max(ann_ret$ann_ret, na.rm=TRUE)*0.9,
           label="OOS→", hjust=0, size=3, colour="grey40") +
  scale_fill_manual(values=STRAT_COLOURS, labels=STRAT_LABELS) +
  scale_y_continuous(labels=scales::percent_format(accuracy=1)) +
  labs(title="Annual Returns — Equal-Weight",
       x=NULL, y="Annual Return", fill="Strategy") +
  theme_minimal(base_size=11) +
  theme(legend.position="bottom",
        axis.text.x=element_text(angle=30, hjust=1, size=8))

ggsave(file.path(DIR_FIGURES, "index_annual_returns.png"), p_ann,
       width=PLOT_WIDTH*1.4, height=PLOT_HEIGHT, dpi=PLOT_DPI)
cat("  index_annual_returns.png saved.\n")

##──────────────────────────────────────────────────────────────────────────────
## 9C. Drawdown chart — equal-weight
##──────────────────────────────────────────────────────────────────────────────

dd_dt <- port_returns[weighting == "ew"][order(strategy, date)]
dd_dt[, cum_idx  := cumprod(1 + port_ret), by=strategy]
dd_dt[, peak     := cummax(cum_idx),        by=strategy]
dd_dt[, drawdown := (cum_idx - peak) / peak]
dd_dt[, strat_label := STRAT_LABELS[strategy]]

p_dd <- ggplot(dd_dt, aes(x=date, y=drawdown,
                          colour=strategy, group=strategy)) +
  geom_line(linewidth=0.8) +
  geom_hline(yintercept=0, colour="black", linewidth=0.3) +
  geom_vline(xintercept=as.numeric(as.Date(sprintf("%d-01-01", OOS_START))),
             linetype="dashed", colour="grey40", linewidth=0.7) +
  scale_colour_manual(values=STRAT_COLOURS, labels=STRAT_LABELS) +
  scale_y_continuous(labels=scales::percent_format(accuracy=1),
                     name="Drawdown") +
  scale_x_date(date_breaks="2 years", date_labels="%Y") +
  labs(title="Drawdown — Equal-Weight",
       subtitle="Dashed = OOS start (2016)",
       x=NULL, colour="Strategy") +
  theme_minimal(base_size=12) +
  theme(legend.position="bottom",
        axis.text.x=element_text(angle=30, hjust=1))

ggsave(file.path(DIR_FIGURES, "index_drawdown.png"), p_dd,
       width=PLOT_WIDTH*1.2, height=PLOT_HEIGHT, dpi=PLOT_DPI)
cat("  index_drawdown.png saved.\n")

##──────────────────────────────────────────────────────────────────────────────
## 9D. Exclusion count per year
##──────────────────────────────────────────────────────────────────────────────

excl_long <- excl_summary |>
  select(year, n_flag_m1, n_flag_m3, n_flag_s3) |>
  pivot_longer(cols=c(n_flag_m1, n_flag_m3, n_flag_s3),
               names_to="strategy", values_to="n_excluded") |>
  mutate(strategy = recode(strategy,
                           n_flag_m1 = "s1",
                           n_flag_m3 = "s2",
                           n_flag_s3 = "s3"),
         strat_label = STRAT_LABELS[strategy])

p_excl <- ggplot(excl_long,
                 aes(x=year, y=n_excluded, colour=strategy, group=strategy)) +
  geom_line(linewidth=0.9) +
  geom_point(size=2) +
  geom_vline(xintercept=OOS_START - 0.5,
             linetype="dashed", colour="grey40", linewidth=0.7) +
  scale_colour_manual(values=STRAT_COLOURS, labels=STRAT_LABELS) +
  scale_y_continuous(name="Firms excluded") +
  labs(title="Annual Exclusion Count — Firms Flagged per Strategy",
       subtitle=paste0("FPR=5% threshold | M1 θ=", round(THRESH_M1,3),
                       " | M3 θ=", round(THRESH_M3,3)),
       x=NULL, colour="Strategy") +
  theme_minimal(base_size=12) +
  theme(legend.position="bottom")

ggsave(file.path(DIR_FIGURES, "index_exclusion_count.png"), p_excl,
       width=PLOT_WIDTH, height=PLOT_HEIGHT, dpi=PLOT_DPI)
cat("  index_exclusion_count.png saved.\n")

##──────────────────────────────────────────────────────────────────────────────
## 9E. CSI avoided vs false positives — tradeoff chart
##──────────────────────────────────────────────────────────────────────────────

csi_plot_dt <- csi_table[, .(
  strategy,
  tp, fp,
  recall    = recall,
  precision = precision,
  label     = STRAT_LABELS[strategy]
)]

p_csi <- ggplot(csi_plot_dt,
                aes(x=fp, y=tp, colour=strategy, label=label)) +
  geom_point(size=5, alpha=0.8) +
  ggrepel::geom_label_repel(size=3, fill="white", show.legend=FALSE) +
  scale_colour_manual(values=STRAT_COLOURS, labels=STRAT_LABELS) +
  labs(title="CSI Events Avoided vs Winners Excluded",
       subtitle="Each point = one strategy | x=false exclusions | y=CSI events avoided",
       x="False Positives (Winners Excluded)",
       y="True Positives (CSI Events Avoided)",
       colour="Strategy") +
  theme_minimal(base_size=12) +
  theme(legend.position="bottom")

## ggrepel optional — fall back if not installed
p_csi_path <- file.path(DIR_FIGURES, "index_csi_avoided.png")
tryCatch({
  library(ggrepel)
  ggsave(p_csi_path, p_csi,
         width=PLOT_WIDTH, height=PLOT_HEIGHT, dpi=PLOT_DPI)
  cat("  index_csi_avoided.png saved.\n")
}, error = function(e) {
  ## Without ggrepel — plain labels
  p_csi2 <- p_csi +
    geom_text(aes(label=strategy), hjust=-0.3, size=3)
  ggsave(p_csi_path, p_csi2,
         width=PLOT_WIDTH, height=PLOT_HEIGHT, dpi=PLOT_DPI)
  cat("  index_csi_avoided.png saved (no ggrepel).\n")
})

#==============================================================================#
# 10. Final summary console table
#==============================================================================#

cat("\n[11] ══════════════════════════════════════════════════════\n")
cat("  FINAL PERFORMANCE TABLE — Equal-Weight\n")
cat("  ══════════════════════════════════════════════════════\n\n")

ew_full <- perf_table[perf_table$weighting == "ew" &
                        perf_table$period == "full", ]

cat(sprintf("  %-28s | %6s | %6s | %6s | %7s | %6s\n",
            "Strategy", "CAGR", "Vol", "Sharpe", "MaxDD", "Calmar"))
cat(sprintf("  %-28s | %6s | %6s | %6s | %7s | %6s\n",
            "----------------------------",
            "------","------","------","-------","------"))

strat_order <- c("bench","s1","s2","s3")
for (s in strat_order) {
  r <- ew_full[ew_full$strategy == s, ]
  if (nrow(r) == 0L) next
  cat(sprintf("  %-28s | %6.2f%% | %6.2f%% | %6.3f | %7.2f%% | %6.3f\n",
              STRAT_LABELS[s],
              r$cagr*100, r$vol_ann*100, r$sharpe,
              r$max_dd*100, r$calmar))
}

cat("\n[11] ══════════════════════════════════════════════════════\n")
cat("  FINAL PERFORMANCE TABLE — Cap-Weight\n")
cat("  ══════════════════════════════════════════════════════\n\n")

cw_full <- perf_table[perf_table$weighting == "cw" &
                        perf_table$period == "full", ]

for (s in strat_order) {
  r <- cw_full[cw_full$strategy == s, ]
  if (nrow(r) == 0L) next
  cat(sprintf("  %-28s | %6.2f%% | %6.2f%% | %6.3f | %7.2f%% | %6.3f\n",
              STRAT_LABELS[s],
              r$cagr*100, r$vol_ann*100, r$sharpe,
              r$max_dd*100, r$calmar))
}

cat(sprintf("\n[11_Results.R] DONE: %s\n", format(Sys.time())))
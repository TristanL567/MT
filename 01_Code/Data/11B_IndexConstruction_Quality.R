#==============================================================================#
#==== 11B_IndexConstruction_Quality.R ==========================================#
#==== Concentrated Quality & Factor Proxy Strategies ==========================#
#==============================================================================#
#
# PURPOSE:
#   Construct and backtest concentrated quality portfolios (C1/C3) using
#   an ML crash-filter as a negative pre-screen, plus factor proxy benchmarks.
#   Implements Doc2_Concentrated_Quality_Strategies.xlsx methodology.
#
# STRATEGY OVERVIEW:
#   Benchmarks    : BENCH-MW, BENCH-EW  (full 3000-stock universe)
#   Factor proxies:
#     QUAL-Proxy  : Top quintile by quality score, ≥$100M, MW + EW
#     USMV-Proxy  : Bottom quintile by 12m realised vol, ≥$100M, MW + EW
#   Concentrated quality (C1 = MW, C3 = EW):
#     C1/C3 with "none" exclusion rate = baseline concentrated-quality portfolio
#     C1/C3 with ML screen (1pct/5pct/10pct/opt):
#       c1_csi    : Best CSI model (M3 = raw, AP=0.794 in test)
#       c1_bucket : Best Bucket model (B1 = fund)
#       c1_comb   : Best Combined (B1 ∪ structural_raw — S4-equivalent)
#
# QUALITY SCORE (QUAL / MSCI methodology):
#   Q = z(ROE) + z(−D/E) + z(−EPS_std5)
#   ROE      = ni / ceq
#   D/E      = (dltt + dlc) / ceq  [inverted — lower is better]
#   EPS_std5 = trailing 5yr SD of epspx [inverted — lower is better]
#   All three components are winsorised at 1%/99% cross-sectionally, then
#   z-scored within the annual universe before summing. At least 2-of-3
#   components must be non-missing for a score to be assigned.
#
# USMV PROXY:
#   12m annualised realised volatility = sqrt(12) × SD(monthly log returns)
#   Computed for each permno for the full calendar year ending December Y.
#   Bottom quintile (lowest vol, ≥$100M mktcap) forms the proxy.
#
# KEY DIFFERENCES FROM 11_IndexConstruction.R:
#   - Only 3 ML prediction keys loaded (best model per track)
#   - Two selection stages: ML exclusion → quality ranking → top-200
#   - Two additional proxies (QUAL, USMV) replace S1–S4 combined strategies
#   - Separate output files: quality_*.rds (no overwrite of Doc1 outputs)
#
# INPUTS:
#   config.R
#   PATH_PRICES_MONTHLY
#   DIR_COMP_PROC/fundamentals.rds    (ni, ceq, dltt, dlc, epspx)
#   DIR_TABLES/ag_{key}/*.parquet     (predictions for raw, bucket, structural_raw)
#
# OUTPUTS:
#   DIR_TABLES/quality_weights.rds
#   DIR_TABLES/quality_returns.rds
#   DIR_TABLES/quality_performance.rds
#   DIR_TABLES/quality_exclusion_summary.rds
#
#==============================================================================#

source("config.R")

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(arrow)
  library(scales)
  library(lubridate)
})

cat("\n[11B_IndexConstruction_Quality.R] START:", format(Sys.time()), "\n")
FIGS <- fn_setup_figure_dirs()

`%||%` <- function(a, b) if (is.null(a) || length(a) == 0) b else a

## ── Parameters ───────────────────────────────────────────────────────────────
UNIVERSE_SIZE    <- 3000L     # base universe (top N by Dec mktcap)
MIN_MKTCAP_MM    <- 100       # minimum market cap filter ($M)
QUALITY_TOP_N    <- 200L      # concentrated quality portfolio size
QUAL_QUINTILE    <- 0.20      # top fraction for QUAL proxy
LOWVOL_QUINTILE  <- 0.20      # bottom fraction for USMV proxy
RF_ANNUAL        <- 0.03
REBAL_MONTHS     <- c(3L, 6L, 9L, 12L)
UNIVERSE_MONTH   <- 12L
TC_BPS           <- 0L

EXCL_RATES <- c("1pct" = 0.01, "5pct" = 0.05, "10pct" = 0.10)
ALL_RATES  <- c(names(EXCL_RATES), "opt")
C1_RATES   <- c("none", ALL_RATES)   # "none" = no ML screen (baseline C1)

INSAMPLE_START <- 1998L
OOS_END        <- 2024L

## ── Best models per track (selected by test-period AP / index Sharpe) ────────
## M3 (raw CSI): AP=0.794, R@FPR3=0.672 — best CSI model in test period
## B1 (fund bucket): competitive test-period index Sharpe
## S4-equiv (B1 ∪ structural_raw): best test-period MW Sharpe in Doc1
C1_SCREENS <- list(
  csi    = list(type = "simple", key  = "raw"),
  bucket = list(type = "simple", key  = "bucket"),
  comb   = list(type = "union",  key1 = "bucket", key2 = "structural_raw")
)
PRED_KEYS_Q <- c("raw", "bucket", "structural_raw")

## ── Labels ───────────────────────────────────────────────────────────────────
STRATEGY_LABELS <- c(
  bench       = "Benchmark (Full Universe)",
  qual_proxy  = "QUAL Proxy — Top Quintile Quality",
  usmv_proxy  = "USMV Proxy — Bottom Quintile Vol",
  c1_csi      = "C1 — Best CSI Screen (M3-Raw)",
  c1_bucket   = "C1 — Best Bucket Screen (B1-Fund)",
  c1_comb     = "C1 — Best Combined Screen (B1\u222aStruct-Raw)"
)
STRATEGY_TRACK <- c(
  bench       = "Benchmark",
  qual_proxy  = "Factor Proxy",
  usmv_proxy  = "Factor Proxy",
  c1_csi      = "Concentrated Quality",
  c1_bucket   = "Concentrated Quality",
  c1_comb     = "Concentrated Quality"
)

#==============================================================================#
# 1. Monthly prices
#==============================================================================#

cat("[11B] Loading monthly prices...\n")
monthly <- as.data.table(readRDS(PATH_PRICES_MONTHLY))
setnames(monthly, "ret_adj", "ret")
setnames(monthly, "mktcap",  "mkvalt")
monthly[, year  := year(date)]
monthly[, month := month(date)]
if (!inherits(monthly$date, "Date")) monthly[, date := as.Date(date)]
monthly[, ret := pmin(pmax(ret, -0.99, na.rm=TRUE), 10, na.rm=TRUE)]
cat(sprintf("  %d rows | %d permnos | %d-%d\n",
            nrow(monthly), uniqueN(monthly$permno),
            min(monthly$year), max(monthly$year)))

#==============================================================================#
# 2. Annual universe (Top 3000 by December mktcap, ≥$100M)
#==============================================================================#

cat("[11B] Building annual universe...\n")
dec_mv <- monthly[month == UNIVERSE_MONTH & !is.na(mkvalt),
                  .(mkvalt_dec = mkvalt[.N]), by = .(permno, year)]
universe_ann <- dec_mv[mkvalt_dec >= MIN_MKTCAP_MM]
universe_ann[, rank_mv := frank(-mkvalt_dec, ties.method="first"), by = year]
universe_ann <- universe_ann[rank_mv <= UNIVERSE_SIZE]
cat(sprintf("  %d firm-years | avg %.0f/yr | %d-%d\n",
            nrow(universe_ann),
            nrow(universe_ann) / uniqueN(universe_ann$year),
            min(universe_ann$year), max(universe_ann$year)))

#==============================================================================#
# 3. Quality scores from Compustat fundamentals
#==============================================================================#

cat("[11B] Computing quality scores...\n")
fund_raw <- as.data.table(readRDS(file.path(DIR_COMP_PROC, "fundamentals.rds")))
fund <- fund_raw[, .(permno, fyear, ni, ceq, dltt, dlc, epspx)]
fund <- fund[!is.na(permno) & !is.na(fyear)]

## --- EPS variability: trailing 5-year rolling SD (≥2 obs required) ----------
setorder(fund, permno, fyear)
fund[, eps_std5 := {
  n   <- .N
  out <- rep(NA_real_, n)
  for (j in seq_len(n)) {
    lo   <- max(1L, j - 4L)
    vals <- epspx[lo:j]
    if (sum(!is.na(vals)) >= 2L) out[j] <- sd(vals, na.rm = TRUE)
  }
  out
}, by = permno]

## --- Raw quality components --------------------------------------------------
fund[, roe := fifelse(!is.na(ceq) & ceq > 0 & !is.na(ni), ni / ceq, NA_real_)]
fund[, de  := fifelse(!is.na(ceq) & ceq > 0,
                      (fifelse(is.na(dltt), 0, dltt) +
                       fifelse(is.na(dlc),  0, dlc)) / ceq,
                      NA_real_)]

## --- Cross-sectional z-scores within annual universe members -----------------
winsorize <- function(x, lo = 0.01, hi = 0.99) {
  q <- quantile(x, probs = c(lo, hi), na.rm = TRUE)
  pmin(pmax(x, q[1L]), q[2L])
}

q_list <- vector("list", length(unique(universe_ann$year)))
qi     <- 0L

for (yr in sort(unique(universe_ann$year))) {
  uni_yr <- universe_ann[year == yr, .(permno)]
  q_yr   <- fund[fyear == yr & permno %in% uni_yr$permno,
                 .(permno, roe, de, eps_std5)]
  if (nrow(q_yr) < 20L) next

  q_yr[, roe_w := if (sum(!is.na(roe)) > 5L)  winsorize(roe)      else roe]
  q_yr[, de_w  := if (sum(!is.na(de))  > 5L)  winsorize(de)       else de]
  q_yr[, eps_w := if (sum(!is.na(eps_std5)) > 5L) winsorize(eps_std5) else eps_std5]

  safe_z <- function(x) {
    s <- sd(x, na.rm = TRUE)
    if (is.na(s) || s == 0) return(rep(NA_real_, length(x)))
    (x - mean(x, na.rm = TRUE)) / s
  }
  q_yr[, z_roe := safe_z(roe_w)]
  q_yr[, z_de  := safe_z(de_w)]
  q_yr[, z_eps := safe_z(eps_w)]

  ## Q = z(ROE) + z(-D/E) + z(-EPS_std); require ≥2 of 3 components
  q_yr[, n_comp := (!is.na(z_roe)) + (!is.na(z_de)) + (!is.na(z_eps))]
  q_yr[n_comp >= 2L, q_score := rowSums(cbind(
    fifelse(is.na(z_roe), 0, z_roe),
    fifelse(is.na(z_de),  0, -z_de),
    fifelse(is.na(z_eps), 0, -z_eps)
  ))]

  qi <- qi + 1L
  q_list[[qi]] <- q_yr[!is.na(q_score), .(permno, year = yr, q_score)]
}
quality_scores <- rbindlist(q_list[seq_len(qi)])
cat(sprintf("  Quality scores: %d firm-years | %d-%d | avg coverage %.0f%%\n",
            nrow(quality_scores),
            min(quality_scores$year), max(quality_scores$year),
            100 * nrow(quality_scores) / nrow(universe_ann)))

#==============================================================================#
# 4. Realised volatility (12m) for USMV proxy
#==============================================================================#

cat("[11B] Computing 12m realised volatility...\n")
monthly[, log_ret := log(1 + ret)]
vol_ann <- monthly[!is.na(ret), {
  vals <- log_ret[!is.na(log_ret)]
  if (length(vals) >= 6L) .(vol12m = sd(vals) * sqrt(12))
  else                     .(vol12m = NA_real_)
}, by = .(permno, year)]
cat(sprintf("  Vol computed: %d firm-years\n", nrow(vol_ann[!is.na(vol12m)])))

#==============================================================================#
# 5. Load ML predictions (best model per track)
#==============================================================================#

cat("[11B] Loading ML predictions...\n")
SRC_PRI <- c(oos = 1L, test = 2L, boundary = 3L, cv = 4L)
PREDS   <- list()

for (key in PRED_KEYS_Q) {
  tdir <- file.path(DIR_TABLES, paste0("ag_", key))
  fmap <- c(
    oos      = file.path(tdir, "ag_preds_oos.parquet"),
    test     = file.path(tdir, "ag_preds_test.parquet"),
    boundary = file.path(tdir, "ag_preds_train_boundary.parquet"),
    cv       = file.path(tdir, "ag_cv_results.parquet")
  )
  parts <- Filter(Negate(is.null), lapply(names(fmap), function(nm) {
    if (!file.exists(fmap[[nm]])) return(NULL)
    dt <- as.data.table(arrow::read_parquet(fmap[[nm]]))
    dt[, src := nm][, .(permno, year, p_csi, src)]
  }))
  if (length(parts) == 0L) { cat(sprintf("  [%-20s] SKIP — no files\n", key)); next }
  comb <- rbindlist(parts)
  comb[, src_rank := SRC_PRI[src]]
  setorder(comb, permno, year, src_rank)
  comb <- comb[!duplicated(comb[, .(permno, year)])]
  PREDS[[key]] <- comb[, .(permno, year, p_csi)]
  cat(sprintf("  [%-20s] %d rows | %d-%d\n",
              key, nrow(PREDS[[key]]),
              min(PREDS[[key]]$year), max(PREDS[[key]]$year)))
}

#==============================================================================#
# 6. Optimal thresholds (F1-maximising, from CV — no test contamination)
#==============================================================================#

cat("[11B] Computing optimal thresholds...\n")
fn_opt_threshold <- function(key) {
  cv_path <- file.path(DIR_TABLES, paste0("ag_", key), "ag_cv_results.parquet")
  if (!file.exists(cv_path)) return(NA_real_)
  cv <- as.data.table(arrow::read_parquet(cv_path))
  cv <- cv[!is.na(y) & !is.na(p_csi)]
  if (nrow(cv) < 100L) return(NA_real_)
  thresholds <- quantile(cv$p_csi, probs = seq(0.50, 0.99, by = 0.005))
  f1s <- vapply(thresholds, function(t) {
    pred_pos <- cv$p_csi >= t
    tp <- sum(pred_pos & cv$y == 1L)
    fp <- sum(pred_pos & cv$y == 0L)
    fn <- sum(!pred_pos & cv$y == 1L)
    if (tp == 0L) return(0)
    pr <- tp / (tp + fp); rc <- tp / (tp + fn)
    2 * pr * rc / (pr + rc)
  }, numeric(1L))
  best_t <- thresholds[[which.max(f1s)]]
  cat(sprintf("  [%-20s] opt = %.4f | CV-F1 = %.4f\n", key, best_t, max(f1s)))
  best_t
}

OPT_THRESH <- setNames(vapply(PRED_KEYS_Q, fn_opt_threshold, numeric(1L)),
                       PRED_KEYS_Q)

#==============================================================================#
# 7. Helper functions
#==============================================================================#

## Exclusion flag for a given model key and rate
fn_excl_flag <- function(uni, p_key, pred_yr, rate) {
  if (rate == "none") return(rep(FALSE, nrow(uni)))
  p <- PREDS[[p_key]]
  if (is.null(p)) return(rep(FALSE, nrow(uni)))
  p_yr <- p[year == pred_yr, .(permno, p_csi)]
  u    <- merge(uni[, .(permno)], p_yr, by = "permno", all.x = TRUE)
  setorder(u, permno); setorder(uni, permno)
  if (rate == "opt") {
    thresh <- OPT_THRESH[[p_key]]
    if (is.na(thresh)) return(rep(FALSE, nrow(uni)))
    return(!is.na(u$p_csi) & u$p_csi >= thresh)
  } else {
    n_p <- sum(!is.na(u$p_csi))
    if (n_p == 0L) return(rep(FALSE, nrow(uni)))
    co  <- ceiling(n_p * EXCL_RATES[[rate]])
    rk  <- frank(-u$p_csi, ties.method = "first", na.last = "keep")
    return(!is.na(rk) & rk <= co)
  }
}

## Factor-proxy weight rows (no top-N cap)
fn_proxy_weights <- function(dt, qdate, q_yr, q_mo, model_key) {
  if (nrow(dt) == 0L) return(NULL)
  sm <- sum(dt$mkvalt_dec, na.rm = TRUE)
  rbind(
    data.table(permno = dt$permno, mkvalt_dec = dt$mkvalt_dec,
               qdate = qdate, q_year = q_yr, q_month = q_mo,
               model_key = model_key, excl_rate = "none", weighting = "mw",
               w = dt$mkvalt_dec / sm),
    data.table(permno = dt$permno, mkvalt_dec = dt$mkvalt_dec,
               qdate = qdate, q_year = q_yr, q_month = q_mo,
               model_key = model_key, excl_rate = "none", weighting = "ew",
               w = 1 / nrow(dt))
  )
}

## Concentrated quality (top-N) weight rows — EW (C3) and MW (C1)
fn_conc_weights <- function(top_n_dt, qdate, q_yr, q_mo, model_key, excl_rate) {
  if (nrow(top_n_dt) == 0L) return(NULL)
  sm <- sum(top_n_dt$mkvalt_dec, na.rm = TRUE)
  rbind(
    data.table(permno = top_n_dt$permno, mkvalt_dec = top_n_dt$mkvalt_dec,
               qdate = qdate, q_year = q_yr, q_month = q_mo,
               model_key = model_key, excl_rate = excl_rate, weighting = "mw",
               w = top_n_dt$mkvalt_dec / sm),
    data.table(permno = top_n_dt$permno, mkvalt_dec = top_n_dt$mkvalt_dec,
               qdate = qdate, q_year = q_yr, q_month = q_mo,
               model_key = model_key, excl_rate = excl_rate, weighting = "ew",
               w = 1 / nrow(top_n_dt))
  )
}

#==============================================================================#
# 8. Build quarterly weights
#==============================================================================#

cat("\n[11B] Building quarterly weights...\n")

q_dates <- monthly[month %in% REBAL_MONTHS,
                   .(qdate = max(date)), by = .(year, month)]
setorder(q_dates, qdate)
q_dates <- q_dates[year >= INSAMPLE_START & year <= OOS_END]
N_Q     <- nrow(q_dates)

w_list <- list()
entry  <- 0L

for (i in seq_len(N_Q)) {
  q_yr    <- q_dates$year[i]
  q_mo    <- q_dates$month[i]
  qdate   <- q_dates$qdate[i]
  pred_yr <- q_yr - 1L

  uni_q <- universe_ann[year == q_yr, .(permno, mkvalt_dec)]
  if (nrow(uni_q) == 0L) next
  setorder(uni_q, permno)

  ## Quality scores and vol for this rebalance (lagged by 1yr)
  qs_q  <- quality_scores[year == pred_yr, .(permno, q_score)]
  vol_q <- vol_ann[year == pred_yr, .(permno, vol12m)]

  ## ── Benchmark (full universe, EW + MW) ─────────────────────────────────────
  sm_all <- sum(uni_q$mkvalt_dec, na.rm = TRUE)
  entry  <- entry + 1L
  w_list[[entry]] <- rbind(
    data.table(permno = uni_q$permno, mkvalt_dec = uni_q$mkvalt_dec,
               qdate = qdate, q_year = q_yr, q_month = q_mo,
               model_key = "bench", excl_rate = "none", weighting = "mw",
               w = uni_q$mkvalt_dec / sm_all),
    data.table(permno = uni_q$permno, mkvalt_dec = uni_q$mkvalt_dec,
               qdate = qdate, q_year = q_yr, q_month = q_mo,
               model_key = "bench", excl_rate = "none", weighting = "ew",
               w = 1 / nrow(uni_q))
  )

  ## ── QUAL Proxy — top quintile by quality score ──────────────────────────────
  uni_wq <- merge(uni_q, qs_q, by = "permno", all.x = TRUE)
  n_valid_q <- sum(!is.na(uni_wq$q_score))
  if (n_valid_q >= 10L) {
    n_top_q <- ceiling(n_valid_q * QUAL_QUINTILE)
    setorder(uni_wq, -q_score)
    top_qual <- uni_wq[!is.na(q_score)][seq_len(min(.N, n_top_q))]
    entry    <- entry + 1L
    w_list[[entry]] <- fn_proxy_weights(top_qual, qdate, q_yr, q_mo, "qual_proxy")
  }

  ## ── USMV Proxy — bottom quintile by realised vol ────────────────────────────
  uni_wv <- merge(uni_q, vol_q, by = "permno", all.x = TRUE)
  n_valid_v <- sum(!is.na(uni_wv$vol12m))
  if (n_valid_v >= 10L) {
    n_bot_v <- ceiling(n_valid_v * LOWVOL_QUINTILE)
    setorder(uni_wv, vol12m)
    bot_vol <- uni_wv[!is.na(vol12m)][seq_len(min(.N, n_bot_v))]
    entry   <- entry + 1L
    w_list[[entry]] <- fn_proxy_weights(bot_vol, qdate, q_yr, q_mo, "usmv_proxy")
  }

  ## ── C1 / C3 Concentrated Quality Strategies ─────────────────────────────────
  ## For each ML screen and exclusion rate:
  ##   1. Apply ML exclusion flag to full universe
  ##   2. Join quality scores on remaining stocks
  ##   3. Select top QUALITY_TOP_N by quality score
  ##   4. MW = C1 weighting; EW = C3 weighting
  for (screen_name in names(C1_SCREENS)) {
    scr    <- C1_SCREENS[[screen_name]]
    mk     <- paste0("c1_", screen_name)

    for (rate in C1_RATES) {
      ## Exclusion flag
      if (scr$type == "simple") {
        flag <- fn_excl_flag(uni_q, scr$key, pred_yr, rate)
      } else if (scr$type == "union") {
        f1   <- fn_excl_flag(uni_q, scr$key1, pred_yr, rate)
        f2   <- fn_excl_flag(uni_q, scr$key2, pred_yr, rate)
        flag <- f1 | f2
      } else {
        flag <- rep(FALSE, nrow(uni_q))
      }

      ## Remaining stocks after exclusion
      incl <- uni_q[!flag]
      if (nrow(incl) == 0L) next

      ## Join quality scores
      incl_q <- merge(incl, qs_q, by = "permno", all.x = TRUE)
      incl_q <- incl_q[!is.na(q_score)]
      if (nrow(incl_q) == 0L) next

      ## Top N by quality
      setorder(incl_q, -q_score)
      top_n <- head(incl_q, QUALITY_TOP_N)

      entry <- entry + 1L
      w_list[[entry]] <- fn_conc_weights(top_n, qdate, q_yr, q_mo, mk, rate)
    }
  }

  if (i %% 20L == 0L || i == N_Q)
    cat(sprintf("  %d/%d quarters done\n", i, N_Q))
}

weights_all <- rbindlist(w_list, use.names = TRUE, fill = TRUE)
setorder(weights_all, model_key, excl_rate, weighting, qdate, permno)
saveRDS(weights_all, file.path(DIR_TABLES, "quality_weights.rds"))
cat(sprintf("  Weights: %d rows | %d strategies saved\n",
            nrow(weights_all),
            uniqueN(weights_all[, .(model_key, excl_rate, weighting)])))

#==============================================================================#
# 9. Monthly portfolio returns
#==============================================================================#

cat("\n[11B] Computing monthly returns...\n")
strats   <- unique(weights_all[, .(model_key, excl_rate, weighting)])
N_S      <- nrow(strats)
ret_list <- vector("list", N_S)

for (i in seq_len(N_S)) {
  sk  <- strats[i]
  w_s <- weights_all[model_key == sk$model_key &
                       excl_rate == sk$excl_rate &
                       weighting == sk$weighting,
                     .(permno, qdate, w)]
  rdates <- sort(unique(w_s$qdate))
  rel_p  <- unique(w_s$permno)
  m_sub  <- monthly[permno %in% rel_p, .(permno, date, year, month, ret)]
  m_sub[, aqd := {
    idx        <- findInterval(date, rdates, left.open = FALSE)
    idx[idx == 0L] <- NA_integer_
    rdates[idx]
  }]
  m_sub  <- m_sub[!is.na(aqd)]
  setnames(w_s, "qdate", "aqd")
  m_s    <- merge(m_sub, w_s, by = c("permno", "aqd"), all.x = FALSE)
  m_s    <- m_s[!is.na(ret) & !is.na(w)]
  if (TC_BPS > 0L) m_s[month %in% REBAL_MONTHS, ret := ret - TC_BPS / 10000]

  ret_list[[i]] <- m_s[, .(
    port_ret   = sum(w * ret, na.rm = TRUE),
    n_holdings = uniqueN(permno),
    model_key  = sk$model_key,
    excl_rate  = sk$excl_rate,
    weighting  = sk$weighting
  ), by = .(date, year, month)]

  if (i %% 20L == 0L || i == N_S)
    cat(sprintf("  %d/%d strategies\n", i, N_S))
}

port_returns <- rbindlist(ret_list)
setorder(port_returns, model_key, excl_rate, weighting, date)
saveRDS(port_returns, file.path(DIR_TABLES, "quality_returns.rds"))
cat(sprintf("  Returns: %d rows saved\n", nrow(port_returns)))

#==============================================================================#
# 10. Performance metrics
#==============================================================================#

cat("\n[11B] Computing performance metrics...\n")

fn_perf <- function(rv, rf = RF_ANNUAL) {
  rv <- rv[is.finite(rv)]
  if (length(rv) < 12L) return(NULL)
  ny   <- length(rv) / 12
  rfm  <- (1 + rf)^(1/12) - 1
  cum  <- prod(1 + rv) - 1
  cagr <- (1 + cum)^(1/ny) - 1
  vol  <- sd(rv) * sqrt(12)
  exc  <- rv - rfm
  sh   <- mean(exc) / sd(exc) * sqrt(12)
  ddr  <- exc[rv < rfm]
  srt  <- if (length(ddr) > 1L) mean(exc) / (sd(ddr) * sqrt(12)) else NA_real_
  ci   <- cumprod(1 + rv); pk <- cummax(ci)
  mdd  <- min((ci - pk) / pk)
  cal  <- if (mdd < 0) cagr / abs(mdd) else NA_real_
  es975 <- mean(rv[rv <= quantile(rv, 0.025)])
  es99  <- mean(rv[rv <= quantile(rv, 0.010)])
  data.frame(
    n_months = length(rv), cum_ret = round(cum, 4L),  cagr    = round(cagr, 4L),
    vol      = round(vol, 4L),       sharpe  = round(sh, 4L),   sortino = round(srt, 4L),
    max_dd   = round(mdd, 4L),       calmar  = round(cal, 4L),
    es_975   = round(es975, 4L),     es_99   = round(es99, 4L),
    win_rate = round(mean(rv > 0), 4L)
  )
}

PERIODS_P <- list(
  insample = c(INSAMPLE_START, TRAIN_END_YR),
  test     = c(TEST_START_YR,  TEST_END_YR),
  oos      = c(OOS_START_YR,   OOS_END),
  full     = c(INSAMPLE_START, OOS_END)
)

perf_rows <- list()
for (i in seq_len(N_S)) {
  sk  <- strats[i]
  rdt <- port_returns[model_key == sk$model_key &
                        excl_rate == sk$excl_rate &
                        weighting == sk$weighting]
  for (pnm in names(PERIODS_P)) {
    yr  <- PERIODS_P[[pnm]]
    sub <- rdt[year >= yr[1L] & year <= yr[2L]]
    pf  <- fn_perf(sub$port_ret)
    if (is.null(pf)) next
    pf$model_key <- sk$model_key
    pf$excl_rate <- sk$excl_rate
    pf$weighting <- sk$weighting
    pf$period    <- pnm
    pf$track     <- STRATEGY_TRACK[sk$model_key] %||% "—"
    pf$label     <- STRATEGY_LABELS[sk$model_key] %||% sk$model_key
    perf_rows[[length(perf_rows) + 1L]] <- pf
  }
}

perf_all <- rbindlist(perf_rows, fill = TRUE)
setDT(perf_all)
saveRDS(perf_all, file.path(DIR_TABLES, "quality_performance.rds"))
cat("  quality_performance.rds saved.\n")

## Console summary — test + OOS, MW
for (per in c("test", "oos")) {
  cat(sprintf("\n  ── %s period (MW) ──\n", toupper(per)))
  sub <- perf_all[period == per & weighting == "mw"]
  cat(sprintf("  %-30s %-6s %+7s %7s %8s\n",
              "Strategy", "Excl", "CAGR%", "Sharpe", "MaxDD%"))
  cat(strrep("-", 65), "\n")
  for (j in seq_len(nrow(sub))) {
    r <- sub[j]
    cat(sprintf("  %-30s %-6s %+6.2f%% %7.3f %+7.2f%%\n",
                substr(r$label, 1L, 30L), r$excl_rate,
                r$cagr * 100, r$sharpe, r$max_dd * 100))
  }
}

#==============================================================================#
# 11. Exclusion & composition diagnostics
#==============================================================================#

excl_d <- weights_all[q_month == UNIVERSE_MONTH,
                      .(n_included = uniqueN(permno)),
                      by = .(model_key, excl_rate, weighting, q_year)]
uni_sz <- universe_ann[, .(n_universe = .N), by = year]
excl_d <- merge(excl_d, uni_sz, by.x = "q_year", by.y = "year", all.x = TRUE)
excl_d[, excl_pct := round((n_universe - n_included) / n_universe * 100, 2L)]
saveRDS(excl_d, file.path(DIR_TABLES, "quality_exclusion_summary.rds"))
cat("  quality_exclusion_summary.rds saved.\n")

#==============================================================================#
# 12. Core plots
#==============================================================================#

cat("\n[11B] Generating plots...\n")
FIG_CONC <- FIGS$index_conc  # 11_index/concentrated/

## ── 12.1  Proxy comparison: QUAL vs USMV vs Benchmark ────────────────────────
proxy_r <- port_returns[model_key %in% c("bench", "qual_proxy", "usmv_proxy") &
                          weighting == "mw"]
if (nrow(proxy_r) > 0L) {
  proxy_r[, cum_idx := cumprod(1 + port_ret), by = model_key]
  PROXY_COLS <- c(bench = "#9E9E9E", qual_proxy = "#1565C0", usmv_proxy = "#2E7D32")
  PROXY_LABS <- c(bench = "Benchmark MW", qual_proxy = "QUAL Proxy MW",
                  usmv_proxy = "USMV Proxy MW")
  p_proxy <- ggplot(proxy_r, aes(x = date, y = cum_idx,
                                  colour = model_key, group = model_key)) +
    geom_line(linewidth = 0.9) +
    geom_vline(xintercept = as.numeric(as.Date("2016-01-01")),
               linetype = "dashed", colour = "grey40") +
    geom_vline(xintercept = as.numeric(as.Date("2020-01-01")),
               linetype = "dotted", colour = "grey40") +
    scale_colour_manual(values = PROXY_COLS, labels = PROXY_LABS) +
    scale_y_continuous(labels = dollar_format(prefix = "$")) +
    scale_x_date(date_breaks = "2 years", date_labels = "%Y") +
    labs(title = "Factor Proxies vs Benchmark (Market-Cap Weighted)",
         subtitle = "QUAL = top quintile quality score | USMV = bottom quintile vol",
         x = NULL, y = "Portfolio Value ($1)", colour = NULL) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "bottom")
  ggsave(file.path(FIG_CONC, "factor_proxies_mw.png"),
         p_proxy, width = PLOT_WIDTH * 1.2, height = PLOT_HEIGHT, dpi = PLOT_DPI)
  cat("  Saved: factor_proxies_mw.png\n")
}

## ── 12.2  C1 strategies vs QUAL proxy vs Benchmark (opt exclusion, MW) ────────
c1_keys <- paste0("c1_", names(C1_SCREENS))
c1_r    <- port_returns[model_key %in% c("bench", "qual_proxy", c1_keys) &
                          ((model_key %in% c("bench", "qual_proxy") & excl_rate == "none") |
                           (model_key %in% c1_keys & excl_rate == "opt")) &
                          weighting == "mw"]
if (nrow(c1_r) > 0L) {
  c1_r[, cum_idx := cumprod(1 + port_ret), by = .(model_key, excl_rate)]
  C1_COLS <- c(bench       = "#9E9E9E",
               qual_proxy  = "#1565C0",
               c1_csi      = "#E53935",
               c1_bucket   = "#FF6F00",
               c1_comb     = "#6A1B9A")
  C1_LABS <- c(bench       = "Benchmark",
               qual_proxy  = "QUAL Proxy",
               c1_csi      = "C1-CSI (M3-Opt)",
               c1_bucket   = "C1-Bucket (B1-Opt)",
               c1_comb     = "C1-Comb (Opt)")
  p_c1 <- ggplot(c1_r, aes(x = date, y = cum_idx,
                             colour = model_key, group = model_key)) +
    geom_line(linewidth = 0.9) +
    geom_vline(xintercept = as.numeric(as.Date("2016-01-01")),
               linetype = "dashed", colour = "grey40") +
    geom_vline(xintercept = as.numeric(as.Date("2020-01-01")),
               linetype = "dotted", colour = "grey40") +
    scale_colour_manual(values = C1_COLS, labels = C1_LABS) +
    scale_y_continuous(labels = dollar_format(prefix = "$")) +
    scale_x_date(date_breaks = "2 years", date_labels = "%Y") +
    labs(title = "C1 Concentrated Quality — ML Screen vs QUAL Proxy (MW, Opt Excl.)",
         subtitle = sprintf("Top %d stocks by quality score after ML crash-filter exclusion",
                            QUALITY_TOP_N),
         x = NULL, y = "Portfolio Value ($1)", colour = NULL) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "bottom")
  ggsave(file.path(FIG_CONC, "c1_vs_qual_proxy_opt_mw.png"),
         p_c1, width = PLOT_WIDTH * 1.2, height = PLOT_HEIGHT, dpi = PLOT_DPI)
  cat("  Saved: c1_vs_qual_proxy_opt_mw.png\n")
}

## ── 12.3  Delta Sharpe vs QUAL proxy across exclusion rates (test + OOS) ─────
bench_qual <- perf_all[model_key == "qual_proxy" & weighting == "mw",
                        .(period, sharpe_qual = sharpe, maxdd_qual = max_dd)]

c1_perf <- perf_all[model_key %in% c1_keys & weighting == "mw" &
                      period %in% c("test", "oos") &
                      excl_rate %in% c("none", "5pct", "opt")]
c1_delta <- merge(c1_perf, bench_qual, by = "period")
c1_delta[, delta_sharpe := sharpe - sharpe_qual]
c1_delta[, delta_maxdd  := max_dd - maxdd_qual]   # negative = better (smaller drawdown)

if (nrow(c1_delta) > 0L) {
  c1_delta[, screen := factor(model_key,
                               levels = c1_keys,
                               labels = c("CSI (M3)", "Bucket (B1)", "Comb (B1\u222aS-Raw)"))]
  c1_delta[, Period := factor(period, levels = c("test", "oos"),
                               labels = c("Test 2016-19", "OOS 2020-24"))]
  c1_delta[, rate_lab := factor(excl_rate,
                                 levels = c("none", "5pct", "opt"),
                                 labels = c("No screen", "Excl 5%", "Excl Opt"))]

  p_delta <- ggplot(c1_delta, aes(x = rate_lab, y = delta_sharpe,
                                   fill = screen, group = screen)) +
    geom_col(position = position_dodge(width = 0.75), width = 0.65) +
    geom_hline(yintercept = 0, linetype = "dashed", colour = "grey40") +
    facet_wrap(~Period) +
    scale_fill_manual(values = c("#E53935", "#FF6F00", "#6A1B9A")) +
    labs(title = "\u0394 Sharpe Ratio: C1 Strategies vs QUAL Proxy (MW)",
         x = "Exclusion rate", y = "\u0394 Sharpe (C1 \u2212 QUAL proxy)", fill = "ML Screen") +
    theme_minimal(base_size = 12) +
    theme(legend.position = "bottom")
  ggsave(file.path(FIG_CONC, "c1_delta_sharpe_vs_qual.png"),
         p_delta, width = PLOT_WIDTH * 1.4, height = PLOT_HEIGHT, dpi = PLOT_DPI)
  cat("  Saved: c1_delta_sharpe_vs_qual.png\n")

  p_deltamdd <- ggplot(c1_delta, aes(x = rate_lab, y = delta_maxdd * 100,
                                      fill = screen, group = screen)) +
    geom_col(position = position_dodge(width = 0.75), width = 0.65) +
    geom_hline(yintercept = 0, linetype = "dashed", colour = "grey40") +
    facet_wrap(~Period) +
    scale_fill_manual(values = c("#E53935", "#FF6F00", "#6A1B9A")) +
    labs(title = "\u0394 Max Drawdown: C1 Strategies vs QUAL Proxy (MW)",
         subtitle = "Negative = lower drawdown than QUAL proxy (better)",
         x = "Exclusion rate", y = "\u0394 Max DD (pp)", fill = "ML Screen") +
    theme_minimal(base_size = 12) +
    theme(legend.position = "bottom")
  ggsave(file.path(FIG_CONC, "c1_delta_maxdd_vs_qual.png"),
         p_deltamdd, width = PLOT_WIDTH * 1.4, height = PLOT_HEIGHT, dpi = PLOT_DPI)
  cat("  Saved: c1_delta_maxdd_vs_qual.png\n")
}

## ── 12.4  Exclusion rate sweep for C1-CSI (EW vs MW) ─────────────────────────
for (mk in c1_keys) {
  mk_r <- port_returns[model_key == mk]
  if (nrow(mk_r) == 0L) next
  bench_mw <- port_returns[model_key == "bench" & weighting == "mw"][, .(date, port_ret)]
  bench_mw[, cum_idx := cumprod(1 + port_ret)]
  qual_mw  <- port_returns[model_key == "qual_proxy" & weighting == "mw"][, .(date, port_ret)]
  qual_mw[, cum_idx := cumprod(1 + port_ret)]

  all_r <- rbindlist(list(
    bench_mw[, .(date, cum_idx, series = "Benchmark")],
    qual_mw[,  .(date, cum_idx, series = "QUAL Proxy")],
    mk_r[weighting == "mw", {
      .(date, cum_idx = cumprod(1 + port_ret),
        series = paste0("C1 Excl-", excl_rate))
    }, by = excl_rate][, .(date, cum_idx, series)]
  ))

  SWEEP_COLS <- c("Benchmark" = "#9E9E9E", "QUAL Proxy" = "#1565C0",
                  "C1 Excl-none"  = "#B0BEC5",
                  "C1 Excl-1pct"  = "#BBDEFB",
                  "C1 Excl-5pct"  = "#1E88E5",
                  "C1 Excl-10pct" = "#0D47A1",
                  "C1 Excl-opt"   = "#E53935")

  screen_label <- STRATEGY_LABELS[mk] %||% mk
  p_sweep <- ggplot(all_r, aes(x = date, y = cum_idx,
                                colour = series, group = series)) +
    geom_line(linewidth = 0.8) +
    geom_vline(xintercept = as.numeric(as.Date("2016-01-01")),
               linetype = "dashed", colour = "grey40") +
    geom_vline(xintercept = as.numeric(as.Date("2020-01-01")),
               linetype = "dotted", colour = "grey40") +
    scale_colour_manual(values = SWEEP_COLS[names(SWEEP_COLS) %in% unique(all_r$series)],
                        drop = FALSE) +
    scale_y_continuous(labels = dollar_format(prefix = "$")) +
    scale_x_date(date_breaks = "2 years", date_labels = "%Y") +
    labs(title = paste0(screen_label, " — Exclusion Rate Sweep (MW)"),
         x = NULL, y = "Portfolio Value ($1)", colour = NULL) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "bottom", axis.text.x = element_text(angle = 30, hjust = 1))
  fname <- paste0(gsub("c1_", "excl_sweep_", mk), "_mw.png")
  ggsave(file.path(FIG_CONC, fname),
         p_sweep, width = PLOT_WIDTH * 1.2, height = PLOT_HEIGHT, dpi = PLOT_DPI)
  cat(sprintf("  Saved: %s\n", fname))
}

cat(sprintf("\n[11B_IndexConstruction_Quality.R] DONE: %s\n", format(Sys.time())))

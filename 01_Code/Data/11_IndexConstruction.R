#==============================================================================#
#==== 11_IndexConstruction.R ==================================================#
#==== Crash-Filtered Index Construction — All Strategies ======================#
#==============================================================================#
#
# PURPOSE:
#   Construct and backtest crash-filtered equity indices for all strategies
#   defined in Doc1_Universe_Index_Strategies.xlsx.
#
# STRATEGY MATRIX:
#   Benchmarks  : BENCH-MW, BENCH-EW (2)
#   Simple      : M1-M4, B1-B4 x {1%, 5%, 10%, opt} x {ew, mw} = 64
#   Combined    : S1-S4 x {1%, 5%, 10%, opt} x {ew, mw} = 32
#   Total       : ~98 strategies
#
# COMBINED STRATEGIES (construction-level, not AutoGluon model labels):
#   S1: M1 UNION  B1        (exclude if flagged by either CSI or Bucket)
#   S2: M1 INTERSECT B1     (exclude only if flagged by both)
#   S3: M1 UNION  ag_structural       (CSI + AutoGluon structural signal)
#   S4: B1 UNION  ag_structural_raw   (Bucket + AutoGluon structural-raw signal)
#
# EXCLUSION RATES:
#   1pct, 5pct, 10pct : rank-based (top X% by predicted score excluded)
#   opt               : threshold from F1-maximising CV threshold per model
#
# INDEX DESIGN:
#   Universe   : Top 3000 by December market cap, min $100M, refreshed annually.
#   Rebalancing: Quarterly (Mar/Jun/Sep/Dec).
#   Signal     : Annual (Compustat-based — stable within year).
#   Weighting  : EW (equal-weight) and MW (market-cap weight).
#
# INPUTS:
#   config.R, PATH_PRICES_MONTHLY
#   DIR_TABLES/ag_{key}/ag_cv_results.parquet      (optimal threshold)
#   DIR_TABLES/ag_{key}/ag_preds_test.parquet
#   DIR_TABLES/ag_{key}/ag_preds_oos.parquet
#   DIR_TABLES/ag_{key}/ag_preds_train_boundary.parquet  (optional)
#
# OUTPUTS:
#   DIR_TABLES/index_weights.rds
#   DIR_TABLES/index_returns.rds
#   DIR_TABLES/index_performance.rds
#   DIR_TABLES/index_exclusion_summary.rds
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

cat("\n[11_IndexConstruction.R] START:", format(Sys.time()), "\n")
FIGS <- fn_setup_figure_dirs()

`%||%` <- function(a, b) if (is.null(a) || length(a) == 0) b else a
## Safe lookup for named vectors (returns NULL, not error, on missing key)
vlookup <- function(vec, key) {
  if (is.null(key) || !nzchar(key) || !(key %in% names(vec))) return(NULL)
  vec[[key]]
}

## ── Parameters ───────────────────────────────────────────────────────────────
UNIVERSE_SIZE  <- 3000L
MIN_MKTCAP_MM  <- 100
RF_ANNUAL      <- 0.03
REBAL_MONTHS   <- c(3L, 6L, 9L, 12L)
UNIVERSE_MONTH <- 12L
TC_BPS         <- 0L

EXCL_RATES <- c("1pct" = 0.01, "5pct" = 0.05, "10pct" = 0.10)  # opt added separately
ALL_RATES  <- c(names(EXCL_RATES), "opt")

INSAMPLE_START <- 1998L
OOS_END        <- 2024L

## ── Short names for display (not in config.R) ────────────────────────────────
MODEL_SHORTS <- c(
  fund="M1", latent_fund="M2", raw="M3", latent_raw="M4",
  bucket="B1", bucket_latent_fund="B2", bucket_raw="B3", bucket_latent_raw="B4",
  structural="AG-S1", structural_latent_fund="AG-S2",
  structural_raw="AG-S3", structural_latent_raw="AG-S4"
)

## ── All AutoGluon model keys (used for simple strategies + combined inputs) ──
ALL_MODEL_KEYS <- names(MODEL_LABELS)

## ── Simple strategies: M1-M4 and B1-B4 only (not structural as standalone) ──
SIMPLE_KEYS <- c("fund", "latent_fund", "raw", "latent_raw",
                 "bucket", "bucket_latent_fund", "bucket_raw", "bucket_latent_raw")

## ── Combined strategy definitions ────────────────────────────────────────────
COMBO_STRATS <- list(
  S1 = list(type  = "union",
            keys  = c("fund", "bucket"),
            label = "S1 — M1 Union B1"),
  S2 = list(type  = "intersect",
            keys  = c("fund", "bucket"),
            label = "S2 — M1 Intersect B1"),
  S3 = list(type  = "union",
            keys  = c("fund", "structural"),
            label = "S3 — M1 Union Structural"),
  S4 = list(type  = "union",
            keys  = c("bucket", "structural_raw"),
            label = "S4 — B1 Union Structural-Raw")
)

## Keys needed for predictions (simple + all combo component keys)
PRED_KEYS <- unique(c(SIMPLE_KEYS,
                      unlist(lapply(COMBO_STRATS, `[[`, "keys"))))

#==============================================================================#
# 1. Monthly prices
#==============================================================================#

cat("[11] Loading monthly prices...\n")
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
# 2. Annual universe
#==============================================================================#

cat("[11] Building annual universe...\n")
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
# 3. Load predictions for all required model keys
#==============================================================================#

cat("[11] Loading predictions...\n")
SRC_PRI <- c(oos=1L, test=2L, boundary=3L, cv=4L)
PREDS   <- list()

for (key in PRED_KEYS) {
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
  if (length(parts) == 0) { cat(sprintf("  [%-35s] SKIP\n", key)); next }

  comb <- rbindlist(parts)
  comb[, src_rank := SRC_PRI[src]]
  setorder(comb, permno, year, src_rank)
  comb <- comb[!duplicated(comb[, .(permno, year)])]
  PREDS[[key]] <- comb[, .(permno, year, p_csi)]
  cat(sprintf("  [%-35s] %d rows | %d-%d\n",
              MODEL_LABELS[[key]] %||% key,
              nrow(PREDS[[key]]),
              min(PREDS[[key]]$year), max(PREDS[[key]]$year)))
}
cat(sprintf("  Loaded: %d/%d required models\n", length(PREDS), length(PRED_KEYS)))

#==============================================================================#
# 4. Optimal thresholds from CV results (F1-maximising, no test contamination)
#==============================================================================#

cat("[11] Computing optimal thresholds from CV results...\n")

fn_opt_threshold <- function(key) {
  cv_path <- file.path(DIR_TABLES, paste0("ag_", key), "ag_cv_results.parquet")
  if (!file.exists(cv_path)) {
    cat(sprintf("  [%s] CV not found — opt threshold set to NA\n", key))
    return(NA_real_)
  }
  cv <- as.data.table(arrow::read_parquet(cv_path))
  cv <- cv[!is.na(y) & !is.na(p_csi)]
  if (nrow(cv) < 100) return(NA_real_)

  ## Grid search over candidate thresholds
  thresholds <- quantile(cv$p_csi, probs = seq(0.50, 0.99, by = 0.005))
  f1s <- vapply(thresholds, function(t) {
    pred_pos <- cv$p_csi >= t
    tp <- sum(pred_pos & cv$y == 1L)
    fp <- sum(pred_pos & cv$y == 0L)
    fn <- sum(!pred_pos & cv$y == 1L)
    if (tp == 0) return(0)
    prec <- tp / (tp + fp); rec <- tp / (tp + fn)
    2 * prec * rec / (prec + rec)
  }, numeric(1))

  best_t <- thresholds[[which.max(f1s)]]
  best_f1 <- max(f1s)
  cat(sprintf("  [%-15s] opt threshold = %.4f | CV-F1 = %.4f\n",
              key, best_t, best_f1))
  best_t
}

OPT_THRESH <- setNames(
  vapply(PRED_KEYS, fn_opt_threshold, numeric(1)),
  PRED_KEYS
)

#==============================================================================#
# 5. Build quarterly weights
#==============================================================================#

cat("\n[11] Building quarterly weights...\n")

q_dates <- monthly[month %in% REBAL_MONTHS,
                   .(qdate = max(date)), by = .(year, month)]
setorder(q_dates, qdate)
q_dates <- q_dates[year >= INSAMPLE_START & year <= OOS_END]
N_Q     <- nrow(q_dates)

## Helper: given universe dt (permno, mkvalt_dec) and an exclusion flag vector,
## return weight rows for both ew and mw.
fn_weights <- function(uni, excl_flag, qdate, q_yr, q_mo, model_key, excl_rate) {
  incl <- uni[!excl_flag]
  if (nrow(incl) == 0L) return(NULL)
  sm <- sum(incl$mkvalt_dec, na.rm=TRUE)
  rbind(
    data.table(permno=incl$permno, mkvalt_dec=incl$mkvalt_dec,
               qdate=qdate, q_year=q_yr, q_month=q_mo,
               model_key=model_key, excl_rate=excl_rate, weighting="ew",
               w=1/nrow(incl)),
    data.table(permno=incl$permno, mkvalt_dec=incl$mkvalt_dec,
               qdate=qdate, q_year=q_yr, q_month=q_mo,
               model_key=model_key, excl_rate=excl_rate, weighting="mw",
               w=incl$mkvalt_dec / sm)
  )
}

## Helper: compute exclusion flag for a single model at a given rate.
## Returns logical vector aligned to uni (permno order).
fn_excl_flag <- function(uni, p_key, pred_yr, rate) {
  p <- PREDS[[p_key]]
  if (is.null(p)) return(rep(FALSE, nrow(uni)))
  p_yr <- p[year == pred_yr, .(permno, p_csi)]
  u    <- merge(uni[, .(permno)], p_yr, by="permno", all.x=TRUE)
  setorder(u, permno); setorder(uni, permno)  # ensure alignment

  if (rate == "opt") {
    thresh <- OPT_THRESH[[p_key]]
    if (is.na(thresh)) return(rep(FALSE, nrow(uni)))
    return(!is.na(u$p_csi) & u$p_csi >= thresh)
  } else {
    n_p <- sum(!is.na(u$p_csi))
    if (n_p == 0) return(rep(FALSE, nrow(uni)))
    co  <- ceiling(n_p * EXCL_RATES[[rate]])
    rk  <- frank(-u$p_csi, ties.method="first", na.last="keep")
    return(!is.na(rk) & rk <= co)
  }
}

w_list <- list()
entry  <- 0L

for (i in seq_len(N_Q)) {
  q_yr  <- q_dates$year[i]
  q_mo  <- q_dates$month[i]
  qdate <- q_dates$qdate[i]
  pred_yr <- q_yr - 1L

  uni_q <- universe_ann[year == q_yr, .(permno, mkvalt_dec)]
  if (nrow(uni_q) == 0L) next
  setorder(uni_q, permno)

  ## ── Benchmarks ─────────────────────────────────────────────────────────────
  entry <- entry + 1L
  w_list[[entry]] <- fn_weights(uni_q, rep(FALSE, nrow(uni_q)),
                                qdate, q_yr, q_mo, "bench", "none")

  ## ── Simple strategies (M1-M4, B1-B4) ───────────────────────────────────────
  for (key in SIMPLE_KEYS) {
    if (is.null(PREDS[[key]])) next
    for (rate in ALL_RATES) {
      flag <- fn_excl_flag(uni_q, key, pred_yr, rate)
      entry <- entry + 1L
      w_list[[entry]] <- fn_weights(uni_q, flag, qdate, q_yr, q_mo, key, rate)
    }
  }

  ## ── Combined strategies (S1-S4) ─────────────────────────────────────────────
  for (sname in names(COMBO_STRATS)) {
    cs    <- COMBO_STRATS[[sname]]
    k1    <- cs$keys[1]; k2 <- cs$keys[2]
    if (is.null(PREDS[[k1]]) || is.null(PREDS[[k2]])) next

    for (rate in ALL_RATES) {
      f1   <- fn_excl_flag(uni_q, k1, pred_yr, rate)
      f2   <- fn_excl_flag(uni_q, k2, pred_yr, rate)
      flag <- if (cs$type == "union") f1 | f2 else f1 & f2
      entry <- entry + 1L
      w_list[[entry]] <- fn_weights(uni_q, flag, qdate, q_yr, q_mo,
                                    sname, rate)
    }
  }

  if (i %% 20 == 0 || i == N_Q) cat(sprintf("  %d/%d quarters\n", i, N_Q))
}

weights_all <- rbindlist(w_list, use.names=TRUE, fill=TRUE)
setorder(weights_all, model_key, excl_rate, weighting, qdate, permno)
saveRDS(weights_all, file.path(DIR_TABLES, "index_weights.rds"))
cat(sprintf("  Weights: %d rows saved\n", nrow(weights_all)))

#==============================================================================#
# 6. Monthly portfolio returns
#==============================================================================#

cat("\n[11] Computing monthly returns...\n")
strats  <- unique(weights_all[, .(model_key, excl_rate, weighting)])
N_S     <- nrow(strats)
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
    idx <- findInterval(date, rdates, left.open=FALSE)
    idx[idx == 0L] <- NA_integer_
    rdates[idx]
  }]
  m_sub  <- m_sub[!is.na(aqd)]
  setnames(w_s, "qdate", "aqd")
  m_s    <- merge(m_sub, w_s, by=c("permno","aqd"), all.x=FALSE)
  m_s    <- m_s[!is.na(ret) & !is.na(w)]
  if (TC_BPS > 0) m_s[month %in% REBAL_MONTHS, ret := ret - TC_BPS/10000]

  ret_list[[i]] <- m_s[, .(
    port_ret   = sum(w * ret, na.rm=TRUE),
    n_holdings = uniqueN(permno),
    model_key  = sk$model_key,
    excl_rate  = sk$excl_rate,
    weighting  = sk$weighting
  ), by = .(date, year, month)]

  if (i %% 50 == 0 || i == N_S) cat(sprintf("  %d/%d strategies\n", i, N_S))
}

port_returns <- rbindlist(ret_list)
setorder(port_returns, model_key, excl_rate, weighting, date)
saveRDS(port_returns, file.path(DIR_TABLES, "index_returns.rds"))
cat(sprintf("  Returns: %d rows\n", nrow(port_returns)))

#==============================================================================#
# 7. Performance metrics
#==============================================================================#

cat("\n[11] Computing performance metrics...\n")

fn_perf <- function(rv, rf = RF_ANNUAL) {
  rv <- rv[is.finite(rv)]
  if (length(rv) < 12) return(NULL)
  ny   <- length(rv) / 12
  rfm  <- (1 + rf)^(1/12) - 1
  cum  <- prod(1 + rv) - 1
  cagr <- (1 + cum)^(1/ny) - 1
  vol  <- sd(rv) * sqrt(12)
  exc  <- rv - rfm
  sh   <- mean(exc) / sd(exc) * sqrt(12)
  ddr  <- exc[rv < rfm]
  srt  <- if (length(ddr) > 1) mean(exc) / (sd(ddr) * sqrt(12)) else NA_real_
  ci   <- cumprod(1 + rv); pk <- cummax(ci)
  mdd  <- min((ci - pk) / pk)
  cal  <- if (mdd < 0) cagr / abs(mdd) else NA_real_
  ## Expected Shortfall
  es975 <- mean(rv[rv <= quantile(rv, 0.025)])
  es99  <- mean(rv[rv <= quantile(rv, 0.010)])
  turn  <- NA_real_  # computed separately if needed
  data.frame(
    n_months=length(rv), cum_ret=round(cum,4), cagr=round(cagr,4),
    vol=round(vol,4), sharpe=round(sh,4), sortino=round(srt,4),
    max_dd=round(mdd,4), calmar=round(cal,4),
    es_975=round(es975,4), es_99=round(es99,4),
    win_rate=round(mean(rv > 0), 4)
  )
}

PERIODS_P <- list(
  insample = c(INSAMPLE_START,  TRAIN_END_YR),
  test     = c(TEST_START_YR,   TEST_END_YR),
  oos      = c(OOS_START_YR,    OOS_END),
  full     = c(INSAMPLE_START,  OOS_END)
)

## Strategy labels for combined strategies
COMBO_LABELS <- setNames(
  sapply(COMBO_STRATS, `[[`, "label"), names(COMBO_STRATS))
COMBO_TRACK  <- setNames(rep("Combined", length(COMBO_STRATS)), names(COMBO_STRATS))

perf_rows <- list()
for (i in seq_len(N_S)) {
  sk  <- strats[i]
  rdt <- port_returns[model_key == sk$model_key &
                        excl_rate == sk$excl_rate &
                        weighting == sk$weighting]
  for (pnm in names(PERIODS_P)) {
    yr  <- PERIODS_P[[pnm]]
    sub <- rdt[year >= yr[1] & year <= yr[2]]
    pf  <- fn_perf(sub$port_ret); if (is.null(pf)) next
    pf$model_key  <- sk$model_key
    pf$excl_rate  <- sk$excl_rate
    pf$weighting  <- sk$weighting
    pf$period     <- pnm
    pf$track      <- vlookup(MODEL_TRACK,  sk$model_key) %||%
                     vlookup(COMBO_TRACK, sk$model_key) %||% "—"
    pf$short      <- vlookup(MODEL_SHORTS, sk$model_key) %||% sk$model_key
    pf$label      <- vlookup(MODEL_LABELS, sk$model_key) %||%
                     vlookup(COMBO_LABELS, sk$model_key) %||% sk$model_key
    perf_rows[[length(perf_rows) + 1]] <- pf
  }
}

perf_all <- rbindlist(perf_rows, fill=TRUE)
setDT(perf_all)
saveRDS(perf_all, file.path(DIR_TABLES, "index_performance.rds"))
cat("  index_performance.rds saved.\n")

## Console summary — OOS, EW, 5% exclusion
cat("\n  ── OOS performance (EW, 5pct exclusion) ──\n")
oos5 <- perf_all[period == "oos" &
                   excl_rate %in% c("none","5pct") &
                   weighting == "ew"]
setorder(oos5, track, short)
bench_r <- oos5[model_key == "bench"]
if (nrow(bench_r) > 0)
  cat(sprintf("  BENCH-EW : CAGR=%+.2f%% | Sharpe=%.3f | MaxDD=%.2f%%\n",
              bench_r$cagr*100, bench_r$sharpe, bench_r$max_dd*100))
for (j in seq_len(nrow(oos5[model_key != "bench"]))) {
  r <- oos5[model_key != "bench"][j]
  cat(sprintf("  %-10s: CAGR=%+.2f%% | Sharpe=%.3f | MaxDD=%.2f%%\n",
              r$short, r$cagr*100, r$sharpe, r$max_dd*100))
}

#==============================================================================#
# 8. Exclusion diagnostics
#==============================================================================#

excl_d <- weights_all[q_month == UNIVERSE_MONTH,
                      .(n_included = .N),
                      by = .(model_key, excl_rate, weighting, q_year)]
uni_sz <- universe_ann[, .(n_universe = .N), by = year]
excl_d <- merge(excl_d, uni_sz, by.x="q_year", by.y="year", all.x=TRUE)
excl_d[, n_excluded := n_universe - n_included]
excl_d[, excl_pct   := round(n_excluded / n_universe * 100, 2)]
saveRDS(excl_d, file.path(DIR_TABLES, "index_exclusion_summary.rds"))
cat("  index_exclusion_summary.rds saved.\n")

#==============================================================================#
# 9. Core plots
#==============================================================================#

cat("\n[11] Generating core plots...\n")

RATE_COLS <- c(none="#9E9E9E", "1pct"="#BBDEFB", "5pct"="#1E88E5",
               "10pct"="#0D47A1", opt="#E53935")
RATE_LABS <- c(none="Benchmark", "1pct"="Excl 1%", "5pct"="Excl 5%",
               "10pct"="Excl 10%", opt="Excl Opt")

## Benchmark cumulative
bench_ew <- port_returns[model_key == "bench" & weighting == "ew"][order(date)]
bench_ew[, cum_idx := cumprod(1 + port_ret)]
p_b <- ggplot(bench_ew, aes(x=date, y=cum_idx)) +
  geom_line(colour="#9E9E9E", linewidth=1) +
  geom_vline(xintercept=as.numeric(as.Date("2016-01-01")),
             linetype="dashed", colour="#1565C0") +
  geom_vline(xintercept=as.numeric(as.Date("2020-01-01")),
             linetype="dotted", colour="#E53935") +
  scale_y_continuous(labels=dollar_format(prefix="$")) +
  scale_x_date(date_breaks="2 years", date_labels="%Y") +
  labs(title=sprintf("Benchmark EW — Pseudo-Russell %d", UNIVERSE_SIZE),
       subtitle="Annual universe | Quarterly rebalancing",
       x=NULL, y="Portfolio Value ($1)") +
  theme_minimal(base_size=12)
ggsave(file.path(FIGS$index_general, "benchmark_ew_cumulative.png"),
       p_b, width=PLOT_WIDTH*1.2, height=PLOT_HEIGHT, dpi=PLOT_DPI)

## M1 exclusion rate comparison
fn_plot_exclusion_rates <- function(key, key_label, fig_path) {
  if (is.null(PREDS[[key]])) return(invisible(NULL))
  ret_key <- port_returns[model_key == key & weighting == "ew"]
  bench_s <- bench_ew[, .(date, port_ret, excl_rate="none")]
  key_s   <- ret_key[, .(date, port_ret, excl_rate)]
  comb    <- rbind(bench_s, key_s)
  comb[, cum_idx := cumprod(1 + port_ret), by = excl_rate]
  comb[, excl_rate := factor(excl_rate, levels=names(RATE_COLS))]
  p <- ggplot(comb, aes(x=date, y=cum_idx, colour=excl_rate, group=excl_rate)) +
    geom_line(linewidth=0.85) +
    geom_vline(xintercept=as.numeric(as.Date("2016-01-01")),
               linetype="dashed", colour="grey40") +
    geom_vline(xintercept=as.numeric(as.Date("2020-01-01")),
               linetype="dotted", colour="grey40") +
    scale_colour_manual(values=RATE_COLS, labels=RATE_LABS, drop=FALSE) +
    scale_y_continuous(labels=dollar_format(prefix="$")) +
    scale_x_date(date_breaks="2 years", date_labels="%Y") +
    labs(title=sprintf("%s — Exclusion Rate Comparison (EW)", key_label),
         x=NULL, y="Portfolio Value ($1)", colour="Strategy") +
    theme_minimal(base_size=12) +
    theme(legend.position="bottom", axis.text.x=element_text(angle=30, hjust=1))
  ggsave(fig_path, p, width=PLOT_WIDTH*1.2, height=PLOT_HEIGHT, dpi=PLOT_DPI)
  invisible(p)
}

fn_plot_exclusion_rates("fund",   "M1 (CSI — Fund)",    file.path(FIGS$index_csi,    "m1_excl_rates.png"))
fn_plot_exclusion_rates("raw",    "M3 (CSI — Raw)",     file.path(FIGS$index_csi,    "m3_excl_rates.png"))
fn_plot_exclusion_rates("bucket", "B1 (Bucket — Fund)", file.path(FIGS$index_bucket, "b1_excl_rates.png"))

## Combined strategies comparison (5pct, EW)
combo_keys_present <- names(COMBO_STRATS)[names(COMBO_STRATS) %in%
                                            unique(port_returns$model_key)]
if (length(combo_keys_present) > 0) {
  combo_r <- port_returns[model_key %in% combo_keys_present &
                            excl_rate == "5pct" & weighting == "ew"]
  bench_s <- bench_ew[, .(date, port_ret, model_key="bench")]
  all_r   <- rbind(combo_r[, .(date, port_ret, model_key)], bench_s)
  all_r[, cum_idx := cumprod(1 + port_ret), by = model_key]
  COMBO_COLS <- c(bench="#9E9E9E", S1="#E53935", S2="#FF9800",
                  S3="#4CAF50", S4="#9C27B0")
  COMBO_LBLS <- c(bench="Benchmark", S1="S1 M1∪B1", S2="S2 M1∩B1",
                  S3="S3 M1∪Struct", S4="S4 B1∪Struct-Raw")
  p_combo <- ggplot(all_r, aes(x=date, y=cum_idx,
                                colour=model_key, group=model_key)) +
    geom_line(linewidth=0.85) +
    geom_vline(xintercept=as.numeric(as.Date("2016-01-01")),
               linetype="dashed", colour="grey40") +
    geom_vline(xintercept=as.numeric(as.Date("2020-01-01")),
               linetype="dotted", colour="grey40") +
    scale_colour_manual(values=COMBO_COLS, labels=COMBO_LBLS) +
    scale_y_continuous(labels=dollar_format(prefix="$")) +
    scale_x_date(date_breaks="2 years", date_labels="%Y") +
    labs(title="Combined Strategies — S1–S4 vs Benchmark (EW, 5% excl.)",
         x=NULL, y="Portfolio Value ($1)", colour="Strategy") +
    theme_minimal(base_size=12) +
    theme(legend.position="bottom", axis.text.x=element_text(angle=30, hjust=1))
  ggsave(file.path(FIGS$index_struct, "combined_strategies_5pct.png"),
         p_combo, width=PLOT_WIDTH*1.2, height=PLOT_HEIGHT, dpi=PLOT_DPI)
  cat("  Saved: combined_strategies_5pct.png\n")
}

cat(sprintf("\n[11_IndexConstruction.R] DONE: %s\n", format(Sys.time())))

#==============================================================================#
#==== 06B_Feature_Eng.R =======================================================#
#==== Feature Engineering — Ratios, Dynamics, Rolling Statistics ==============#
#==============================================================================#
#
# PURPOSE:
#   Construct the full feature matrix from panel_raw.rds.
#   Produces features_raw.rds — one row per (permno, year), all engineered
#   features attached, ready for autoencoder (08B) and training (09).
#
# INPUTS:
#   - config.R
#   - PATH_PANEL_RAW       : (permno, year, y, ~60 raw vars, macro, prices)
#   - PATH_PRICES_MONTHLY  : monthly prices for rolling price momentum features
#
# OUTPUT:
#   - PATH_FEATURES_RAW    : (permno, year, y, censored, ~460 features)
#
# FEATURE FAMILIES:
#   1. Point-in-time ratios          (~40)  : Foundation ratios, Altman Z
#   2. YoY changes                   (~41)  : First derivatives
#   3. Acceleration                  (~41)  : Second derivatives
#   4. Expanding mean & volatility   (~82)  : Long-run trend baseline (lagged)
#   5. Peak deterioration            (~18)  : Drawdown from 5yr rolling peak
#   6. Consecutive decline counters  (~10)  : Zombie dynamics
#   7. Accounting momentum           (~15)  : 2Y vs expanding mean
#   8. Rolling statistics 3yr        (~90)  : Short-window dynamics
#   9. Rolling statistics 5yr        (~90)  : Long-window dynamics
#  10. Price momentum & volatility   (~8)   : From prices_monthly
#  11. Macro interaction terms       (~6)   : Firm × macro regime
#
# DESIGN DECISIONS:
#
#   [1] ALL TRANSFORMATIONS ARE STRICTLY BACKWARD-LOOKING:
#       Expanding stats use shift(cummean(x), lag=1) to exclude current year.
#       Rolling stats use align="right" in slider::slide_dbl().
#       No feature at year t uses any information from year t onward.
#
#   [2] RATIO CONSTRUCTION USES SAFE DIVISION:
#       All denominators guarded against zero and NA via safe_div().
#       Raw (unwinsorised) ratios saved — winsorisation in 09_Train.R.
#
#   [3] PROGRAMMATIC FEATURE DETECTION:
#       meta_cols and id_cols defined explicitly. All other numeric columns
#       treated as base variables. No hardcoded feature lists except for
#       semantically meaningful ratio groups.
#
#   [4] mkvalt IS ALREADY FILLED:
#       coalesce(mkvalt, prcc_f * csho) was applied in 06_Merge.R.
#
#   [5] ROLLING PEAK FOR DRAWDOWN (Family 10):
#       max_dd_12m / max_dd_60m use a 3-YEAR (36-month) ROLLING PEAK,
#       not the all-time cumulative peak. This ensures the feature measures
#       CURRENT deterioration from a recent reference point.
#       Rationale: a firm stable at a depressed level after a 2002 crash
#       should show max_dd ≈ 0%, not -80%, in 2007.
#       Fallback: firms with < 36 months of history use expanding cummax.
#
#   [6] ROLLING PEAK FOR FUNDAMENTALS (Family 5):
#       peak_drop_* uses a 5-YEAR ROLLING PEAK of the ratio, not cummax
#       over the firm's full history. Same rationale as [5] — measures
#       recent deterioration, not cumulative loss from a decade-old peak.
#       Implemented via slider::slide_dbl() per firm per column.
#       Fallback: expanding cummax for firms with < 5 years of history.
#
#   [7] TROUGH RISE (Family 5):
#       trough_rise_* retains cummin (all-time trough) — rising leverage
#       from an all-time low is meaningful regardless of age.
#
#==============================================================================#

source("config.R")

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(slider)
  library(lubridate)
})

cat("\n[06B_Feature_Eng.R] START:", format(Sys.time()), "\n")

#==============================================================================#
# 0. Load inputs
#==============================================================================#

cat("[06B_Feature_Eng.R] Loading inputs...\n")

panel    <- as.data.table(readRDS(PATH_PANEL_RAW))
prices_m <- as.data.table(readRDS(PATH_PRICES_MONTHLY))

setorder(panel,    permno, year)
setorder(prices_m, permno, date)

cat(sprintf("  panel_raw      : %d rows, %d cols, %d permno\n",
            nrow(panel), ncol(panel), n_distinct(panel$permno)))
cat(sprintf("  prices_monthly : %d rows, %d permno\n",
            nrow(prices_m), n_distinct(prices_m$permno)))

#==============================================================================#
# 0B. Utility functions
#==============================================================================#

## Safe division — returns NA when denominator is zero or NA
safe_div <- function(x, y, na_val = NA_real_) {
  ifelse(is.na(y) | y == 0, na_val, x / y)
}

## Safe log — returns NA for non-positive values
safe_log <- function(x) {
  ifelse(is.na(x) | x <= 0, NA_real_, log(x))
}

#==============================================================================#
# 1. FEATURE FAMILY 1 — Point-in-Time Ratios
#==============================================================================#

cat("[06B_Feature_Eng.R] Family 1: Point-in-time ratios...\n")

panel[, `:=`(
  
  ##── Profitability ──────────────────────────────────────────────────────────
  earn_yld         = safe_div(epspx,   prcc_f),
  ocf_per_share    = safe_div(oancf,   csho),
  roa              = safe_div(ni,      at),
  roe              = safe_div(ni,      seq),
  roic             = safe_div(oiadp,   icapt),
  ebit_roa         = safe_div(ebit,    at),
  gross_margin     = safe_div(gp,      sale),
  ebitda_margin    = safe_div(ebitda,  sale),
  ocf_margin       = safe_div(oancf,   sale),
  
  ##── Leverage ───────────────────────────────────────────────────────────────
  leverage         = safe_div(dltt + dlc, at),
  net_debt_ebitda  = safe_div(dltt + dlc - che, ebitda),
  std_debt_pct     = safe_div(dlc,  dltt + dlc),
  eff_int_rate     = safe_div(xint, dltt + dlc),
  interest_cov     = safe_div(oiadp, xint),
  dd1_ratio        = safe_div(dd1,  at),
  
  ##── Liquidity ──────────────────────────────────────────────────────────────
  current_ratio    = safe_div(act,  lct),
  quick_ratio      = safe_div(act - invt, lct),
  cash_pct_act     = safe_div(che,  act),
  wcap_ratio       = safe_div(wcap, at),
  
  ##── Valuation & Market ─────────────────────────────────────────────────────
  bp_ratio         = safe_div(seq,  mkvalt),
  ev_to_sales      = safe_div(mkvalt + dltt + dlc - che, sale),
  div_yield        = safe_div(dvc,  mkvalt),
  cash_div_cf      = safe_div(dv,   oancf),
  mkt_to_book      = safe_div(mkvalt, seq),
  
  ##── Quality & Efficiency ───────────────────────────────────────────────────
  accruals_ratio   = safe_div(ni - oancf, at),
  asset_turnover   = safe_div(sale, at),
  capex_intensity  = safe_div(capx, at),
  rd_intensity     = safe_div(xrd,  at),
  reinvest_rate    = safe_div(capx, oancf),
  
  ##── Size ───────────────────────────────────────────────────────────────────
  log_at           = safe_log(at),
  log_mkvalt       = safe_log(mkvalt),
  log_emp          = safe_log(emp * 1000),
  
  ##── Zombie Precursors ──────────────────────────────────────────────────────
  rental_ratio     = safe_div(xrent, at),
  assets_per_emp   = safe_div(at,    emp),
  ni_per_emp       = safe_div(ni,    emp),
  min_int_tcap     = safe_div(mib,   seq + mib + dltt),
  
  ##── Comprehensive income ───────────────────────────────────────────────────
  compr_inc_ratio  = safe_div(citotal, at),
  
  ##── Altman Z-Score components ──────────────────────────────────────────────
  altman_z1        = safe_div(wcap,   at),
  altman_z2        = safe_div(re,     at),
  altman_z3        = safe_div(ebit,   at),
  altman_z4        = safe_div(mkvalt, lt),
  altman_z5        = safe_div(sale,   at)
)]

panel[, altman_z       := 1.2*altman_z1 + 1.4*altman_z2 +
        3.3*altman_z3 + 0.6*altman_z4 + 1.0*altman_z5]
panel[, invest_st_ratio := safe_div(ivst, at)]

cat("  Point-in-time ratios complete.\n")

#==============================================================================#
# 1B. Define ratio groups for dynamic feature construction
#==============================================================================#

PEAK_RATIOS <- c(
  "earn_yld", "ocf_per_share", "roa", "roe", "roic", "ebit_roa",
  "gross_margin", "ebitda_margin", "ocf_margin",
  "current_ratio", "quick_ratio", "cash_pct_act", "wcap_ratio",
  "interest_cov", "asset_turnover", "bp_ratio",
  "log_mkvalt", "log_emp"
)

TROUGH_RATIOS <- c(
  "leverage", "net_debt_ebitda", "std_debt_pct",
  "eff_int_rate", "dd1_ratio", "rental_ratio",
  "accruals_ratio"
)

CONSEC_RATIOS <- c(
  "earn_yld", "ocf_per_share", "roa", "roic",
  "gross_margin", "interest_cov", "log_mkvalt", "log_emp",
  "current_ratio", "wcap_ratio"
)

ROLLING_CORE <- c(
  "earn_yld", "ocf_per_share", "roa", "roic",
  "leverage", "net_debt_ebitda", "interest_cov",
  "current_ratio", "cash_pct_act",
  "accruals_ratio", "asset_turnover",
  "gross_margin", "ebitda_margin",
  "log_mkvalt", "log_at"
)

ratio_cols <- c(
  "earn_yld", "ocf_per_share", "roa", "roe", "roic", "ebit_roa",
  "gross_margin", "ebitda_margin", "ocf_margin",
  "leverage", "net_debt_ebitda", "std_debt_pct", "eff_int_rate",
  "interest_cov", "dd1_ratio",
  "current_ratio", "quick_ratio", "cash_pct_act", "wcap_ratio",
  "bp_ratio", "ev_to_sales", "div_yield", "mkt_to_book",
  "accruals_ratio", "asset_turnover", "capex_intensity",
  "rd_intensity", "reinvest_rate",
  "log_at", "log_mkvalt", "log_emp",
  "rental_ratio", "assets_per_emp", "ni_per_emp",
  "altman_z1", "altman_z2", "altman_z3", "altman_z4", "altman_z5",
  "altman_z", "invest_st_ratio"
)

ratio_cols <- intersect(ratio_cols, names(panel))
cat(sprintf("  ratio_cols for dynamics: %d\n", length(ratio_cols)))

#==============================================================================#
# 2. FEATURE FAMILY 2 — Year-over-Year Changes (First Derivatives)
#==============================================================================#

cat("[06B_Feature_Eng.R] Family 2: YoY changes...\n")

panel[, paste0("yoy_", ratio_cols) :=
        lapply(.SD, function(x) x - shift(x, n = 1L, type = "lag")),
      by = permno, .SDcols = ratio_cols]

#==============================================================================#
# 3. FEATURE FAMILY 3 — Acceleration (Second Derivatives)
#==============================================================================#

cat("[06B_Feature_Eng.R] Family 3: Acceleration (2nd differences)...\n")

yoy_cols <- paste0("yoy_", ratio_cols)

panel[, paste0("accel_", ratio_cols) :=
        lapply(.SD, function(x) x - shift(x, n = 1L, type = "lag")),
      by = permno, .SDcols = yoy_cols]

#==============================================================================#
# 4. FEATURE FAMILY 4 — Expanding Mean & Volatility
#
#   Strictly lagged: year t feature uses only years 1..t-1.
#   cummean shifted by 1 excludes current year.
#==============================================================================#

cat("[06B_Feature_Eng.R] Family 4: Expanding mean & volatility...\n")

panel[, paste0("expmean_", ratio_cols) :=
        lapply(.SD, function(x) {
          shift(cummean(x), n = 1L, type = "lag")
        }),
      by = permno, .SDcols = ratio_cols]

panel[, paste0("expvol_", ratio_cols) :=
        lapply(.SD, function(x) {
          n      <- seq_along(x)
          mu     <- cummean(x)
          expvar <- (cumsum(x^2) - n * mu^2) / pmax(n - 1L, 1L)
          lagged <- shift(sqrt(pmax(0, expvar)), n = 1L, type = "lag")
          fifelse(n < 3L, NA_real_, lagged)
        }),
      by = permno, .SDcols = ratio_cols]

#==============================================================================#
# 5. FEATURE FAMILY 5 — Peak Deterioration & Trough Rise
#
#   PEAK DETERIORATION — CORRECTED:
#     Uses 5-year rolling peak (not cummax all-time peak).
#     Measures how far a ratio has fallen from its RECENT best.
#     A firm whose earnings yield peaked in 1995 and has been stable at
#     a lower level for a decade should show peak_drop ≈ 0, not -10pp.
#     Implemented per-column via explicit loop to avoid data.table
#     group-context length mismatches with frollapply.
#
#   TROUGH RISE — UNCHANGED:
#     Retains cummin (all-time trough). Rising leverage from an all-time
#     low is meaningful regardless of when the trough occurred.
#==============================================================================#

cat("[06B_Feature_Eng.R] Family 5: Peak deterioration & trough rise...\n")

valid_peak   <- intersect(PEAK_RATIOS,   ratio_cols)
valid_trough <- intersect(TROUGH_RATIOS, ratio_cols)

## Peak deterioration — 5-year rolling peak per firm per column
if (length(valid_peak) > 0) {
  for (col in valid_peak) {
    out_col <- paste0("peak_drop_", col)
    
    panel[, (out_col) := {
      x <- get(col)
      n <- length(x)
      
      ## 5-year rolling peak via slider (handles group context correctly)
      ## .before = 4L means window covers current + 4 prior = 5 years
      ## .complete = FALSE returns partial window for early rows
      rolling_peak_5y <- slider::slide_dbl(
        x,
        .f        = function(v) max(v, na.rm = TRUE),
        .before   = 4L,
        .complete = FALSE,
        .after    = 0L
      )
      
      ## Expanding cummax fallback — for firms with < 5 years of history
      ## Replace NA with -Inf before cummax so NAs do not propagate
      x_safe       <- replace(x, is.na(x), -Inf)
      expanding_pk <- cummax(x_safe)
      expanding_pk[is.infinite(expanding_pk)] <- NA_real_
      
      ## Use rolling peak once 5 years of data are available
      ref_peak_raw <- ifelse(
        seq_len(n) >= 5L,
        rolling_peak_5y,
        expanding_pk
      )
      
      ## Lag by 1 — year t uses peak known through year t-1 only
      ref_peak <- shift(ref_peak_raw, n = 1L, type = "lag")
      
      x - ref_peak
    }, by = permno]
  }
  cat(sprintf("  peak_drop_* : %d columns\n", length(valid_peak)))
}

## Trough rise — all-time cummin (unchanged, see design note [7])
if (length(valid_trough) > 0) {
  for (col in valid_trough) {
    out_col <- paste0("trough_rise_", col)
    
    panel[, (out_col) := {
      x <- get(col)
      
      ## Replace NA with +Inf before cummin so NAs do not propagate
      x_safe        <- replace(x, is.na(x), Inf)
      expanding_tr  <- cummin(x_safe)
      expanding_tr[is.infinite(expanding_tr)] <- NA_real_
      
      lagged_trough <- shift(expanding_tr, n = 1L, type = "lag")
      x - lagged_trough
    }, by = permno]
  }
  cat(sprintf("  trough_rise_* : %d columns\n", length(valid_trough)))
}

#==============================================================================#
# 6. FEATURE FAMILY 6 — Consecutive Decline Counters
#==============================================================================#

cat("[06B_Feature_Eng.R] Family 6: Consecutive decline counters...\n")

valid_consec <- intersect(CONSEC_RATIOS, ratio_cols)

for (col in valid_consec) {
  yoy_col <- paste0("yoy_", col)
  out_col <- paste0("consec_decline_", col)
  
  panel[, (out_col) := {
    yoy     <- get(yoy_col)
    counter <- integer(.N)
    for (i in seq_len(.N)) {
      if (i == 1L || is.na(yoy[i])) {
        counter[i] <- 0L
      } else if (yoy[i] < 0) {
        counter[i] <- counter[i - 1L] + 1L
      } else {
        counter[i] <- 0L
      }
    }
    counter
  }, by = permno]
}

cat(sprintf("  consec_decline_* : %d columns\n", length(valid_consec)))

#==============================================================================#
# 7. FEATURE FAMILY 7 — Accounting Momentum
#
#   Short-run vs long-run baseline: 2Y rolling mean minus expanding mean.
#   Positive = recently improved above historical average.
#   Negative = recently deteriorated below historical average.
#==============================================================================#

cat("[06B_Feature_Eng.R] Family 7: Accounting momentum...\n")

valid_rolling_core <- intersect(ROLLING_CORE, ratio_cols)

panel[, paste0("acct_mom_", valid_rolling_core) :=
        lapply(.SD, function(x) {
          roll2 <- frollmean(x, n = 2L, align = "right", fill = NA)
          expmn <- shift(cummean(x), n = 1L, type = "lag")
          roll2 - expmn
        }),
      by = permno, .SDcols = valid_rolling_core]

#==============================================================================#
# 8 & 9. FEATURE FAMILIES 8 & 9 — Rolling Statistics (3yr and 5yr)
#
#   Since panel is annual:
#     WINDOW_SHORT = 12 months → 1 year → 3-year window
#     WINDOW_LONG  = 60 months → 5 years
#   (WINDOW_SHORT_YRS corrected to 3 to match thesis configuration)
#
#   Statistics: mean, min, max, sd, OLS trend, autocorrelation lag-1.
#   Applied to ROLLING_CORE ratios only.
#==============================================================================#

cat("[06B_Feature_Eng.R] Families 8 & 9: Rolling statistics...\n")

WINDOW_SHORT_YRS <- 3L   # 3-year rolling window
WINDOW_LONG_YRS  <- 5L   # 5-year rolling window

fn_trend <- function(x) {
  valid <- !is.na(x)
  n     <- sum(valid)
  if (n < 3L) return(NA_real_)
  t  <- seq_along(x)[valid]
  xv <- x[valid]
  cov(t, xv) / var(t)
}

fn_autocorr <- function(x) {
  valid <- x[!is.na(x)]
  if (length(valid) < 3L) return(NA_real_)
  cor(valid[-length(valid)], valid[-1], use = "complete.obs")
}

fn_roll <- function(x, window, fn) {
  slider::slide_dbl(
    x,
    .f        = fn,
    .before   = window - 1L,
    .complete = FALSE
  )
}

for (w in c(WINDOW_SHORT_YRS, WINDOW_LONG_YRS)) {
  
  w_suffix <- sprintf("_%dy", w)
  cat(sprintf("  Rolling window: %d year(s)...\n", w))
  
  panel[, paste0("roll_mean",     w_suffix, "_", valid_rolling_core) :=
          lapply(.SD, fn_roll, window = w, fn = mean),
        by = permno, .SDcols = valid_rolling_core]
  
  panel[, paste0("roll_min",      w_suffix, "_", valid_rolling_core) :=
          lapply(.SD, fn_roll, window = w, fn = min),
        by = permno, .SDcols = valid_rolling_core]
  
  panel[, paste0("roll_max",      w_suffix, "_", valid_rolling_core) :=
          lapply(.SD, fn_roll, window = w, fn = max),
        by = permno, .SDcols = valid_rolling_core]
  
  panel[, paste0("roll_sd",       w_suffix, "_", valid_rolling_core) :=
          lapply(.SD, fn_roll, window = w,
                 fn = function(x) if (sum(!is.na(x)) < 2L) NA_real_
                 else sd(x, na.rm = TRUE)),
        by = permno, .SDcols = valid_rolling_core]
  
  panel[, paste0("roll_trend",    w_suffix, "_", valid_rolling_core) :=
          lapply(.SD, fn_roll, window = w, fn = fn_trend),
        by = permno, .SDcols = valid_rolling_core]
  
  panel[, paste0("roll_autocorr", w_suffix, "_", valid_rolling_core) :=
          lapply(.SD, fn_roll, window = w, fn = fn_autocorr),
        by = permno, .SDcols = valid_rolling_core]
}

#==============================================================================#
# 10. FEATURE FAMILY 10 — Price Momentum & Volatility
#
#   CORRECTED: max_dd_12m / max_dd_60m now use a 3-YEAR (36-month)
#   rolling peak instead of the all-time cumulative peak (cummax).
#
#   Rationale (design note [5]):
#     The model answers "will this ongoing drawdown become a permanent CSI
#     event?" A firm stable at a depressed level after a 2002 crash should
#     show max_dd ≈ 0% in 2007 — not -80% from the 1999 all-time high.
#     Using a 3-year rolling peak ensures the feature measures current
#     price trajectory, not cumulative historical loss.
#
#   Fallback: firms with < 36 months of history use expanding cummax.
#   This is handled via ifelse(is.na(rolling_peak_3y), cummax, rolling).
#==============================================================================#

cat("[06B_Feature_Eng.R] Family 10: Price momentum & volatility...\n")

## Step 1: Wealth index from inception
prices_m[, wealth_index := cumprod(1 + fifelse(is.na(ret_adj), 0, ret_adj)),
         by = permno]

## Step 2: 3-year (36-month) rolling peak via slider — handles group context
## Step 3: Reference peak = rolling where available, expanding otherwise
prices_m[, drawdown := {
  ## Rolling 36-month peak — resets after recovery
  rolling_peak_3y <- slider::slide_dbl(
    wealth_index,
    .f        = max,
    .before   = 35L,   ## 35 prior + current month = 36 months
    .complete = FALSE,
    .after    = 0L
  )
  
  ## Expanding cummax fallback for first < 36 months of history
  expanding_peak <- cummax(wealth_index)
  
  ## Use rolling once >= 36 months of data available, expanding before that
  ref_peak <- ifelse(
    seq_along(wealth_index) >= 36L,
    rolling_peak_3y,
    expanding_peak
  )
  
  wealth_index / ref_peak - 1
}, by = permno]

prices_m[, cal_year  := year(date)]
prices_m[, cal_month := month(date)]

## Compound returns helper
fn_compound_ret <- function(ret_vec) {
  clean <- ret_vec[!is.na(ret_vec)]
  if (length(clean) == 0L) return(NA_real_)
  prod(1 + clean) - 1
}

## Compute annual price features per (permno, year)
## Inside the by group, all columns refer only to rows for that (permno, year)
price_feats <- prices_m[, {
  
  all_hist <- ret_adj    ## Monthly returns for this permno × year group
  n_hist   <- length(all_hist)
  
  ## Momentum: compound returns over trailing windows ending Dec of year t
  mom_1m  <- if (n_hist >= 1L)  all_hist[n_hist]                    else NA_real_
  mom_3m  <- if (n_hist >= 3L)  fn_compound_ret(tail(all_hist,  3L)) else NA_real_
  mom_6m  <- if (n_hist >= 6L)  fn_compound_ret(tail(all_hist,  6L)) else NA_real_
  mom_24m <- if (n_hist >= 24L) fn_compound_ret(tail(all_hist, 24L)) else NA_real_
  
  ## Volatility: sd of monthly returns
  vol_12m <- if (sum(!is.na(tail(all_hist, 12L))) >= 3L)
    sd(tail(all_hist, 12L), na.rm = TRUE) else NA_real_
  vol_60m <- if (sum(!is.na(tail(all_hist, 60L))) >= 12L)
    sd(tail(all_hist, 60L), na.rm = TRUE) else NA_real_
  
  ## Max drawdown: worst value in trailing window of the CORRECTED drawdown
  ## drawdown is already computed from the 3yr rolling peak above
  dd_hist    <- drawdown
  max_dd_12m <- if (n_hist >= 12L) min(tail(dd_hist, 12L), na.rm = TRUE) else NA_real_
  max_dd_60m <- if (n_hist >= 60L) min(tail(dd_hist, 60L), na.rm = TRUE) else NA_real_
  
  list(
    mom_1m     = mom_1m,
    mom_3m     = mom_3m,
    mom_6m     = mom_6m,
    mom_24m    = mom_24m,
    vol_12m    = vol_12m,
    vol_60m    = vol_60m,
    max_dd_12m = max_dd_12m,
    max_dd_60m = max_dd_60m
  )
  
}, by = .(permno, year = cal_year)]

## Join price momentum features onto annual panel
panel <- merge(
  panel,
  price_feats,
  by    = c("permno", "year"),
  all.x = TRUE
)

setorder(panel, permno, year)

cat(sprintf("  Price momentum features added: %d columns\n",
            ncol(price_feats) - 2L))

#==============================================================================#
# 11. FEATURE FAMILY 11 — Macro Interaction Terms
#==============================================================================#

cat("[06B_Feature_Eng.R] Family 11: Macro interactions...\n")

panel[, `:=`(
  interact_lev_rate  = leverage       * fedfunds,
  interact_cov_rate  = interest_cov   * fedfunds,
  interact_nde_hyspr = net_debt_ebitda * hy_spread,
  interact_roa_gdp   = roa            * gdp_growth,
  interact_ret_vix   = log_return     * vix,
  interact_acc_hyspr = accruals_ratio * hy_spread
)]

#==============================================================================#
# 12. Identify all feature columns
#==============================================================================#

cat("[06B_Feature_Eng.R] Identifying feature columns...\n")

meta_cols <- c(
  ## Identifiers
  "permno", "year", "y", "censored", "param_id", "gvkey",
  "datadate", "fyr", "sich", "fiscal_year_end_month",
  ## Raw Compustat balance sheet
  "at", "act", "che", "ivst", "rect", "invt", "wcap", "ppent",
  "intan", "intano", "gdwl", "txdba", "aco",
  ## Raw Compustat liabilities
  "lt", "lct", "dltt", "dlc", "dd1", "ap", "txp", "txditc",
  "seq", "ceq", "re", "pstk", "mib", "icapt",
  ## Raw Compustat income statement
  "sale", "revt", "cogs", "gp", "xsga", "xrd", "dp",
  "ebit", "ebitda", "oiadp", "oibdp", "xopr", "xint",
  "pretax_income", "ni", "ib", "epspx", "dvc", "dvt", "citotal",
  ## Raw Compustat cash flow
  "oancf", "capx", "ivncf", "fincf", "dv", "sstk",
  "prstkc", "dltis", "dltr", "sppe",
  ## Raw Compustat market
  "csho", "prcc_f", "mkvalt",
  ## Raw Compustat other
  "emp", "xrent",
  ## Panel price features (from 06_Merge.R — kept as explicit features below)
  "n_months_ret", "avg_mktcap",
  ## Raw macro levels (some kept as explicit features below)
  "gdp", "gdp_growth", "unrate", "fedfunds", "gs10",
  "term_spread", "hy_spread", "vix", "cpi", "cpi_inflation",
  "indpro", "indpro_growth", "recession",
  ## Rolling peak intermediate — not a feature
  "rolling_peak_3y",
  ## Lifetime
  "lifetime_years"
)

## All numeric columns not in meta_cols
feature_cols <- setdiff(
  names(panel)[sapply(panel, is.numeric)],
  meta_cols
)

## Explicitly include macro and price signals as features
feature_cols <- unique(c(
  feature_cols,
  "log_return", "ann_return",
  "term_spread", "hy_spread", "unrate",
  "fedfunds", "vix", "cpi_inflation",
  "gdp_growth", "indpro_growth", "recession"
))

feature_cols <- intersect(feature_cols, names(panel))

cat(sprintf("  Total feature columns: %d\n", length(feature_cols)))

#==============================================================================#
# 13. Construct features_raw
#==============================================================================#

id_cols   <- c("permno", "year", "y", "censored", "param_id",
               "gvkey", "datadate", "lifetime_years",
               "fiscal_year_end_month")
keep_cols <- unique(c(id_cols, feature_cols))
keep_cols <- intersect(keep_cols, names(panel))

features_raw <- panel[, ..keep_cols]

saveRDS(features_raw, PATH_FEATURES_RAW)
cat(sprintf("[06B_Feature_Eng.R] features_raw.rds saved: %d rows, %d cols\n",
            nrow(features_raw), ncol(features_raw)))

#==============================================================================#
# 14. Assertions
#==============================================================================#

cat("[06B_Feature_Eng.R] Running assertions...\n")

## A) No duplicate (permno, year)
n_dup <- sum(duplicated(features_raw[, .(permno, year)]))
if (n_dup > 0)
  stop(sprintf("ASSERTION FAILED: %d duplicate (permno, year).", n_dup))

## B) y values valid
invalid_y <- features_raw[!is.na(y) & !y %in% c(0L, 1L)]
if (nrow(invalid_y) > 0)
  stop("ASSERTION FAILED: Invalid y values.")

## C) Core features exist
core_required <- c("earn_yld", "ocf_per_share", "roa", "leverage",
                   "altman_z", "log_return", "vol_12m", "max_dd_12m")
missing_core <- setdiff(core_required, names(features_raw))
if (length(missing_core) > 0)
  stop(sprintf("ASSERTION FAILED: Missing core features: %s",
               paste(missing_core, collapse = ", ")))

## D) Feature count plausible
if (length(feature_cols) < 100L)
  warning(sprintf("WARNING: Only %d features — expected >= 100.",
                  length(feature_cols)))

## E) max_dd_12m distribution check — verify rolling peak fix worked
## CSI firms (y=1) should no longer have median near -90%
dd_by_label <- features_raw[!is.na(y), .(
  median_dd = median(max_dd_12m, na.rm = TRUE),
  mean_dd   = mean(max_dd_12m,   na.rm = TRUE)
), by = y]
cat("\n  max_dd_12m sanity check (rolling peak fix):\n")
print(dd_by_label)
if (dd_by_label[y == 1L, median_dd] < -0.85)
  warning(paste(
    "max_dd_12m median for CSI firms is still < -85%.",
    "Rolling peak fix may not have taken effect.",
    "Check Family 10 drawdown computation."
  ))

cat("[06B_Feature_Eng.R] All assertions passed.\n")

#==============================================================================#
# 15. Validation diagnostics
#==============================================================================#

cat("\n[06B_Feature_Eng.R] ══════════════════════════════════════\n")
cat(sprintf("  Rows           : %d\n", nrow(features_raw)))
cat(sprintf("  Feature columns: %d\n", length(feature_cols)))
cat(sprintf("  Permno         : %d\n", n_distinct(features_raw$permno)))

cat("\n  NA audit by feature family:\n")

family_prefixes <- c(
  "Point-in-time ratios" = "^(earn_yld|ocf_per_share|roa|roe|roic|leverage|
                              current_ratio|bp_ratio|altman_z|accruals_ratio)",
  "YoY changes"          = "^yoy_",
  "Acceleration"         = "^accel_",
  "Expanding mean"       = "^expmean_",
  "Expanding vol"        = "^expvol_",
  "Peak deterioration"   = "^peak_drop_",
  "Trough rise"          = "^trough_rise_",
  "Consec. declines"     = "^consec_decline_",
  "Acct. momentum"       = "^acct_mom_",
  "Rolling 3yr"          = "^roll_.*_3y_",
  "Rolling 5yr"          = "^roll_.*_5y_",
  "Price momentum"       = "^(mom_|vol_|max_dd_)",
  "Macro interactions"   = "^interact_"
)

for (fname in names(family_prefixes)) {
  pat  <- family_prefixes[[fname]]
  cols <- grep(pat, names(features_raw), value = TRUE, perl = TRUE)
  if (length(cols) == 0L) next
  n_na  <- sum(is.na(features_raw[, .SD, .SDcols = cols]))
  n_tot <- length(cols) * nrow(features_raw)
  cat(sprintf("    %-25s : %4d cols | %5.1f%% NA\n",
              fname, length(cols), 100 * n_na / n_tot))
}

cat("\n  Key paper feature coverage (% non-missing):\n")
paper_features <- c("earn_yld", "ocf_per_share", "roa", "roic",
                    "leverage", "altman_z", "log_return",
                    "vol_12m", "max_dd_12m", "fedfunds", "gdp_growth")
for (v in intersect(paper_features, names(features_raw))) {
  pct <- 100 * mean(!is.na(features_raw[[v]]))
  cat(sprintf("    %-20s : %5.1f%%\n", v, pct))
}

cat("\n[06B_Feature_Eng.R] DONE:", format(Sys.time()), "\n")
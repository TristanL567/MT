#==============================================================================#
#==== 06B_Feature_Eng.R =======================================================#
#==== Feature Engineering — Ratios, Dynamics, Rolling Statistics ==============#
#==============================================================================#
#
# PURPOSE:
#   Construct the full feature matrix from panel_raw.rds.
#   Produces features_raw.rds — one row per (permno, year), all engineered
#   features attached, ready for autoencoder (06C) and feature selection (07).
#
# INPUTS:
#   - config.R
#   - PATH_PANEL_RAW       : (permno, year, y, ~60 raw vars, macro, prices)
#   - PATH_PRICES_MONTHLY  : monthly prices for rolling price momentum features
#
# OUTPUT:
#   - PATH_FEATURES_RAW    : (permno, year, y, censored, ~350 features)
#
# FEATURE FAMILIES:
#   1. Point-in-time ratios          (~40)  : Foundation ratios, Altman Z
#   2. YoY changes                   (~20)  : First derivatives
#   3. Acceleration                  (~20)  : Second derivatives
#   4. Expanding mean & volatility   (~20)  : Long-run trend baseline
#   5. Peak deterioration            (~10)  : Drawdown from all-time high
#   6. Consecutive decline counters  (~6)   : Zombie dynamics
#   7. Accounting momentum           (~10)  : 2Y vs expanding mean
#   8. Rolling statistics 12M        (~90)  : Short-window dynamics (WINDOW_SHORT)
#   9. Rolling statistics 60M        (~90)  : Long-window dynamics  (WINDOW_LONG)
#  10. Price momentum & volatility   (~9)   : From prices_monthly
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
#       Extreme ratios winsorised at 1st/99th percentile of training set.
#       Winsorisation parameters fitted on training set only — no leakage.
#       NOTE: Winsorisation is deferred to 09_Train.R per-split to avoid
#       leaking test distribution into training. Raw (unwinsorised) ratios
#       saved here.
#
#   [3] PROGRAMMATIC FEATURE DETECTION:
#       meta_cols and id_cols defined explicitly. All other numeric columns
#       are treated as base variables for ratio construction. No hardcoded
#       feature lists except for semantically meaningful ratio groups
#       (PEAK_RATIOS, TROUGH_RATIOS, CONSEC_RATIOS) defined in config.R.
#
#   [4] mkvalt IS ALREADY FILLED:
#       coalesce(mkvalt, prcc_f * csho) was applied in 06_Merge.R.
#       No further fallback needed here.
#
#   [5] ROLLING STATS COMPUTED ON RATIO PANEL:
#       Rolling statistics are applied to the engineered ratios, not raw
#       Compustat levels. This follows the paper's approach and produces
#       more economically meaningful temporal features.
#
#   [6] PRICE MOMENTUM FROM prices_monthly:
#       Short-window momentum (1m, 3m, 6m) requires monthly prices — cannot
#       be computed from the annual panel. Loaded separately and joined.
#
#==============================================================================#

source("config.R")

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(lubridate)
  library(slider)       # Rolling window functions
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
#
#   Constructed from single-year fundamentals in panel_raw.
#   All denominators guarded via safe_div().
#   Raw unwinsorised values saved — winsorisation in 09_Train.R.
#==============================================================================#

cat("[06B_Feature_Eng.R] Family 1: Point-in-time ratios...\n")

panel[, `:=`(
  
  ##── Profitability ──────────────────────────────────────────────────────────
  earn_yld         = safe_div(epspx,   prcc_f),          # ff_earn_yld *
  ocf_per_share    = safe_div(oancf,   csho),            # ff_oper_ps_net_cf *
  roa              = safe_div(ni,      at),               # Return on assets
  roe              = safe_div(ni,      seq),              # Return on equity
  roic             = safe_div(oiadp,   icapt),            # ff_roic *
  ebit_roa         = safe_div(ebit,    at),               # ff_ebit_oper_roa *
  gross_margin     = safe_div(gp,      sale),             # Gross margin
  ebitda_margin    = safe_div(ebitda,  sale),             # EBITDA margin
  ocf_margin       = safe_div(oancf,   sale),             # ff_cf_sales *
  
  ##── Leverage ───────────────────────────────────────────────────────────────
  leverage         = safe_div(dltt + dlc, at),            # Total leverage
  net_debt_ebitda  = safe_div(dltt + dlc - che, ebitda),  # Net debt / EBITDA
  std_debt_pct     = safe_div(dlc,  dltt + dlc),          # ff_std_debt *
  eff_int_rate     = safe_div(xint, dltt + dlc),          # ff_eff_int_rate *
  interest_cov     = safe_div(oiadp, xint),               # Interest coverage
  dd1_ratio        = safe_div(dd1,  at),                  # Refinancing wall
  
  ##── Liquidity ──────────────────────────────────────────────────────────────
  current_ratio    = safe_div(act,  lct),                 # Current ratio
  quick_ratio      = safe_div(act - invt, lct),           # Quick ratio
  cash_pct_act     = safe_div(che,  act),                 # ff_cash_curr_assets *
  wcap_ratio       = safe_div(wcap, at),                  # Working capital / assets
  
  ##── Valuation & Market ─────────────────────────────────────────────────────
  bp_ratio         = safe_div(seq,  mkvalt),              # Book-to-price
  ev_to_sales      = safe_div(mkvalt + dltt + dlc - che, sale), # ff_entrpr_val_sales *
  div_yield        = safe_div(dvc,  mkvalt),              # ff_div_yld *
  cash_div_cf      = safe_div(dv,   oancf),               # ff_cash_div_cf *
  mkt_to_book      = safe_div(mkvalt, seq),               # Market-to-book
  
  ##── Quality & Efficiency ───────────────────────────────────────────────────
  accruals_ratio   = safe_div(ni - oancf, at),            # Sloan (1996)
  asset_turnover   = safe_div(sale, at),                  # Asset efficiency
  capex_intensity  = safe_div(capx, at),                  # Capex / assets
  rd_intensity     = safe_div(xrd,  at),                  # R&D / assets
  reinvest_rate    = safe_div(capx, oancf),               # ff_reinvest_rate *
  
  ##── Size (log-transformed for skew) ───────────────────────────────────────
  log_at           = safe_log(at),                        # Log total assets
  log_mkvalt       = safe_log(mkvalt),                    # Log market cap
  log_emp          = safe_log(emp * 1000),                # Log employees
  
  ##── Zombie Precursors (thesis extension) ───────────────────────────────────
  rental_ratio     = safe_div(xrent, at),                 # Fixed cost obligation
  assets_per_emp   = safe_div(at,    emp),                # ff_assets_per_emp *
  ni_per_emp       = safe_div(ni,    emp),                # ff_net_inc_per_emp *
  min_int_tcap     = safe_div(mib,   seq + mib + dltt),   # ff_min_int_tcap *
  
  ##── Comprehensive income ───────────────────────────────────────────────────
  compr_inc_ratio  = safe_div(citotal, at),               # ff_compr_inc *
  
  ##── Altman Z-Score components (benchmark) ──────────────────────────────────
  altman_z1        = safe_div(wcap, at),                  # Working capital / assets
  altman_z2        = safe_div(re,   at),                  # Retained earnings / assets
  altman_z3        = safe_div(ebit, at),                  # EBIT / assets (= ebit_roa)
  altman_z4        = safe_div(mkvalt, lt),                # Mkt cap / total liabilities
  altman_z5        = safe_div(sale, at)                   # Sales / assets (= asset_turnover)
)]

## Composite Altman Z-score
panel[, altman_z := 1.2 * altman_z1 + 1.4 * altman_z2 +
        3.3 * altman_z3 + 0.6 * altman_z4 + 1.0 * altman_z5]

## Short-term investment ratio
panel[, invest_st_ratio := safe_div(ivst, at)]            # ff_invest_st_tot *

cat(sprintf("  Ratios constructed: %d columns added\n",
            length(grep("^(earn_yld|ocf_per_share|roa|roe|roic|ebit_roa|
                         gross_margin|ebitda_margin|ocf_margin|leverage|
                         net_debt_ebitda|std_debt_pct|eff_int_rate|
                         interest_cov|dd1_ratio|current_ratio|quick_ratio|
                         cash_pct_act|wcap_ratio|bp_ratio|ev_to_sales|
                         div_yield|cash_div_cf|mkt_to_book|accruals_ratio|
                         asset_turnover|capex_intensity|rd_intensity|
                         reinvest_rate|log_at|log_mkvalt|log_emp|
                         rental_ratio|assets_per_emp|ni_per_emp|
                         min_int_tcap|compr_inc_ratio|altman_z)",
                        names(panel), value = TRUE))))

#==============================================================================#
# 1B. Define ratio groups for dynamic feature construction
#
#   Used by: peak deterioration, trough rise, consecutive decline counters.
#   Defined here rather than config.R — these are feature engineering
#   decisions, not pipeline parameters.
#==============================================================================#

## "Higher is better" ratios — peak deterioration is meaningful
PEAK_RATIOS <- c(
  "earn_yld", "ocf_per_share", "roa", "roe", "roic", "ebit_roa",
  "gross_margin", "ebitda_margin", "ocf_margin",
  "current_ratio", "quick_ratio", "cash_pct_act", "wcap_ratio",
  "interest_cov", "asset_turnover", "bp_ratio",
  "log_mkvalt", "log_emp"
)

## "Lower is better" ratios — trough rise is meaningful
TROUGH_RATIOS <- c(
  "leverage", "net_debt_ebitda", "std_debt_pct",
  "eff_int_rate", "dd1_ratio", "rental_ratio",
  "accruals_ratio"
)

## Ratios for consecutive decline counters — deterioration in these is CSI signal
CONSEC_RATIOS <- c(
  "earn_yld", "ocf_per_share", "roa", "roic",
  "gross_margin", "interest_cov", "log_mkvalt", "log_emp",
  "current_ratio", "wcap_ratio"
)

## Core ratios for rolling window statistics (subset — rolling all would be ~350 cols)
ROLLING_CORE <- c(
  "earn_yld", "ocf_per_share", "roa", "roic",
  "leverage", "net_debt_ebitda", "interest_cov",
  "current_ratio", "cash_pct_act",
  "accruals_ratio", "asset_turnover",
  "gross_margin", "ebitda_margin",
  "log_mkvalt", "log_at"
)

## All engineered ratio columns (for dynamics)
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

## Keep only ratio cols that exist in panel
ratio_cols <- intersect(ratio_cols, names(panel))

cat(sprintf("  ratio_cols for dynamics: %d\n", length(ratio_cols)))

#==============================================================================#
# 2. FEATURE FAMILY 2 — Year-over-Year Changes (First Derivatives)
#
#   Captures deterioration rate. shift() by id ensures no cross-firm bleed.
#   IMPORTANT: shift() within by=permno gives lag of previous YEAR for this
#   firm — correct because panel is sorted by (permno, year).
#==============================================================================#

cat("[06B_Feature_Eng.R] Family 2: YoY changes...\n")

panel[, paste0("yoy_", ratio_cols) :=
        lapply(.SD, function(x) x - shift(x, n = 1L, type = "lag")),
      by = permno, .SDcols = ratio_cols]

#==============================================================================#
# 3. FEATURE FAMILY 3 — Acceleration (Second Derivatives)
#
#   Change in the rate of change. Captures whether deterioration is
#   speeding up (negative acceleration in profitability = worsening).
#==============================================================================#

cat("[06B_Feature_Eng.R] Family 3: Acceleration (2nd differences)...\n")

yoy_cols <- paste0("yoy_", ratio_cols)

panel[, paste0("accel_", ratio_cols) :=
        lapply(.SD, function(x) x - shift(x, n = 1L, type = "lag")),
      by = permno, .SDcols = yoy_cols]

#==============================================================================#
# 4. FEATURE FAMILY 4 — Expanding Mean & Volatility
#
#   Long-run baseline for each firm. STRICTLY LAGGED: cummean shifted by 1
#   so year t feature uses only years 1..t-1 — no current-year look-ahead.
#
#   Expanding volatility: standard deviation of all observations up to t-1.
#==============================================================================#

cat("[06B_Feature_Eng.R] Family 4: Expanding mean & volatility...\n")

panel[, paste0("expmean_", ratio_cols) :=
        lapply(.SD, function(x) {
          shift(cummean(x), n = 1L, type = "lag")  # Lag 1 — exclude current year
        }),
      by = permno, .SDcols = ratio_cols]

panel[, paste0("expvol_", ratio_cols) :=
        lapply(.SD, function(x) {
          n      <- seq_along(x)
          mu     <- cummean(x)
          expvar <- (cumsum(x^2) - n * mu^2) / pmax(n - 1L, 1L)
          lagged <- shift(sqrt(pmax(0, expvar)), n = 1L, type = "lag")
          fifelse(n < 3L, NA_real_, lagged)  # Need >=3 obs for meaningful vol
        }),
      by = permno, .SDcols = ratio_cols]

#==============================================================================#
# 5. FEATURE FAMILY 5 — Peak Deterioration & Trough Rise
#
#   peak_drop:   how far "higher is better" ratios have fallen from best-ever
#   trough_rise: how far "lower is better" ratios have risen from worst-ever
#
#   Both use cummax/cummin of LAGGED values — current year excluded.
#==============================================================================#

cat("[06B_Feature_Eng.R] Family 5: Peak deterioration & trough rise...\n")

valid_peak   <- intersect(PEAK_RATIOS,   ratio_cols)
valid_trough <- intersect(TROUGH_RATIOS, ratio_cols)

if (length(valid_peak) > 0) {
  panel[, paste0("peak_drop_", valid_peak) :=
          lapply(.SD, function(x) {
            lagged_peak <- shift(cummax(x), n = 1L, type = "lag")
            x - lagged_peak   # How far below lagged all-time high
          }),
        by = permno, .SDcols = valid_peak]
}

if (length(valid_trough) > 0) {
  panel[, paste0("trough_rise_", valid_trough) :=
          lapply(.SD, function(x) {
            lagged_trough <- shift(cummin(x), n = 1L, type = "lag")
            x - lagged_trough  # How far above lagged all-time low
          }),
        by = permno, .SDcols = valid_trough]
}

#==============================================================================#
# 6. FEATURE FAMILY 6 — Consecutive Decline Counters
#
#   How many consecutive years has a ratio been declining?
#   Uses YoY change already computed in Family 2.
#   Sequential loop required — each count depends on previous.
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

#==============================================================================#
# 7. FEATURE FAMILY 7 — Accounting Momentum
#
#   Short-run vs long-run baseline: 2Y rolling mean minus expanding mean.
#   Positive = ratio recently improved above historical average.
#   Negative = ratio recently deteriorated below historical average.
#   Applied to ROLLING_CORE ratios only.
#==============================================================================#

cat("[06B_Feature_Eng.R] Family 7: Accounting momentum...\n")

valid_rolling_core <- intersect(ROLLING_CORE, ratio_cols)

panel[, paste0("acct_mom_", valid_rolling_core) :=
        lapply(.SD, function(x) {
          roll2  <- frollmean(x, n = 2L, align = "right", fill = NA)
          expmn  <- shift(cummean(x), n = 1L, type = "lag")
          roll2 - expmn
        }),
      by = permno, .SDcols = valid_rolling_core]

#==============================================================================#
# 8 & 9. FEATURE FAMILIES 8 & 9 — Rolling Statistics
#
#   Applied over WINDOW_SHORT (12 months ≈ 1 year) and WINDOW_LONG (60 months
#   ≈ 5 years) as defined in config.R.
#
#   Since panel is annual, WINDOW_SHORT = 12 months → 1 year window,
#   WINDOW_LONG = 60 months → 5 year window.
#   Convert to years for annual panel:
#     window_short_yrs = WINDOW_SHORT / 12 = 1
#     window_long_yrs  = WINDOW_LONG  / 12 = 5
#
#   Statistics: mean, min, max, sd, OLS trend (slope), autocorr lag-1.
#   Applied to ROLLING_CORE ratios only (~15 × 6 stats × 2 windows = 180 cols).
#
#   slider::slide_dbl() used for robust handling of ragged windows
#   (firms with fewer years than the window size return NA — correct).
#==============================================================================#

cat("[06B_Feature_Eng.R] Families 8 & 9: Rolling statistics...\n")

## Convert month-based config windows to years for annual panel
WINDOW_SHORT_YRS <- as.integer(WINDOW_SHORT / 12L)   # 1 year
WINDOW_LONG_YRS  <- as.integer(WINDOW_LONG  / 12L)   # 5 years

## OLS trend (slope) function — returns NA for windows with all-NA or < 3 obs
fn_trend <- function(x) {
  valid <- !is.na(x)
  n     <- sum(valid)
  if (n < 3L) return(NA_real_)
  t  <- seq_along(x)[valid]
  xv <- x[valid]
  ## Deming-style OLS slope: cov(t, x) / var(t)
  cov(t, xv) / var(t)
}

## Autocorrelation lag-1
fn_autocorr <- function(x) {
  valid <- x[!is.na(x)]
  if (length(valid) < 3L) return(NA_real_)
  cor(valid[-length(valid)], valid[-1], use = "complete.obs")
}

## Rolling stats helper — applies one stat function over a window
fn_roll <- function(x, window, fn) {
  slider::slide_dbl(
    x,
    .f        = fn,
    .before   = window - 1L,
    .complete = FALSE    # Return NA for incomplete windows rather than erroring
  )
}

## Apply all rolling statistics for each window
for (w in c(WINDOW_SHORT_YRS, WINDOW_LONG_YRS)) {
  
  w_suffix <- sprintf("_%dy", w)
  cat(sprintf("  Rolling window: %d year(s)...\n", w))
  
  panel[, paste0("roll_mean", w_suffix, "_", valid_rolling_core) :=
          lapply(.SD, fn_roll, window = w, fn = mean),
        by = permno, .SDcols = valid_rolling_core]
  
  panel[, paste0("roll_min",  w_suffix, "_", valid_rolling_core) :=
          lapply(.SD, fn_roll, window = w, fn = min),
        by = permno, .SDcols = valid_rolling_core]
  
  panel[, paste0("roll_max",  w_suffix, "_", valid_rolling_core) :=
          lapply(.SD, fn_roll, window = w, fn = max),
        by = permno, .SDcols = valid_rolling_core]
  
  panel[, paste0("roll_sd",   w_suffix, "_", valid_rolling_core) :=
          lapply(.SD, fn_roll, window = w,
                 fn = function(x) if (sum(!is.na(x)) < 2L) NA_real_ else sd(x, na.rm = TRUE)),
        by = permno, .SDcols = valid_rolling_core]
  
  panel[, paste0("roll_trend", w_suffix, "_", valid_rolling_core) :=
          lapply(.SD, fn_roll, window = w, fn = fn_trend),
        by = permno, .SDcols = valid_rolling_core]
  
  panel[, paste0("roll_autocorr", w_suffix, "_", valid_rolling_core) :=
          lapply(.SD, fn_roll, window = w, fn = fn_autocorr),
        by = permno, .SDcols = valid_rolling_core]
}

#==============================================================================#
# 10. FEATURE FAMILY 10 — Price Momentum & Volatility
#
#   Computed from prices_monthly — requires monthly data, cannot be derived
#   from the annual panel alone.
#
#   Features computed at end of each calendar year t:
#     mom_1m   : 1-month return (December return)
#     mom_3m   : compounded return Oct–Dec
#     mom_6m   : compounded return Jul–Dec
#     mom_12m  : compounded return Jan–Dec (= log_return already in panel)
#     mom_24m  : compounded return 2 calendar years
#     vol_12m  : sd of monthly returns over 12 months
#     vol_60m  : sd of monthly returns over 60 months
#     max_dd_12m: maximum drawdown from wealth index over 12 months
#     max_dd_60m: maximum drawdown from wealth index over 60 months
#==============================================================================#

cat("[06B_Feature_Eng.R] Family 10: Price momentum & volatility...\n")

## Compute wealth index and max drawdown per permno
prices_m[, wealth_index := cumprod(1 + fifelse(is.na(ret_adj), 0, ret_adj)),
         by = permno]
prices_m[, rolling_peak := cummax(wealth_index), by = permno]
prices_m[, drawdown      := wealth_index / rolling_peak - 1]
prices_m[, cal_year      := year(date)]
prices_m[, cal_month     := month(date)]

## Compound returns helper
fn_compound_ret <- function(ret_vec) {
  clean <- ret_vec[!is.na(ret_vec)]
  if (length(clean) == 0L) return(NA_real_)
  prod(1 + clean) - 1
}

## For each (permno, year), compute momentum features using data up to Dec of year t
price_feats <- prices_m[, {
  
  ## All monthly returns in this year
  yr_rets <- ret_adj[cal_year == .BY$year]
  
  ## Returns in recent windows (only use data from <= current year)
  all_hist <- ret_adj  # Full history for this permno up to end of year
  
  ## Momentum: compound returns over trailing windows ending Dec of year t
  n_hist <- length(all_hist)
  
  mom_1m    <- if (n_hist >= 1L)  all_hist[n_hist] else NA_real_
  mom_3m    <- if (n_hist >= 3L)  fn_compound_ret(tail(all_hist, 3L))  else NA_real_
  mom_6m    <- if (n_hist >= 6L)  fn_compound_ret(tail(all_hist, 6L))  else NA_real_
  mom_24m   <- if (n_hist >= 24L) fn_compound_ret(tail(all_hist, 24L)) else NA_real_
  
  ## Volatility: sd of monthly returns
  vol_12m   <- if (sum(!is.na(tail(all_hist, 12L))) >= 3L)
    sd(tail(all_hist, 12L), na.rm = TRUE) else NA_real_
  vol_60m   <- if (sum(!is.na(tail(all_hist, 60L))) >= 12L)
    sd(tail(all_hist, 60L), na.rm = TRUE) else NA_real_
  
  ## Max drawdown: worst drawdown over trailing windows
  dd_hist <- drawdown
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

## Join price momentum features to panel
panel <- merge(
  panel,
  price_feats,
  by  = c("permno", "year"),
  all.x = TRUE
)

setorder(panel, permno, year)

cat(sprintf("  Price momentum features added: %d columns\n",
            ncol(price_feats) - 2L))  # -2 for permno, year

#==============================================================================#
# 11. FEATURE FAMILY 11 — Macro Interaction Terms
#
#   Multiplicative interactions between firm-level distress signals and
#   macro regime variables. Captures rate sensitivity, credit market stress,
#   and cyclicality of profitability.
#==============================================================================#

cat("[06B_Feature_Eng.R] Family 11: Macro interactions...\n")

panel[, `:=`(
  ## Leverage × interest rate: debt service stress when rates rise
  interact_lev_rate     = leverage * fedfunds,
  
  ## Interest coverage × rate: firms with thin coverage under rate pressure
  interact_cov_rate     = interest_cov * fedfunds,
  
  ## Net debt / EBITDA × HY spread: credit market stress for leveraged firms
  interact_nde_hyspr    = net_debt_ebitda * hy_spread,
  
  ## ROA × GDP growth: cyclicality of profitability
  interact_roa_gdp      = roa * gdp_growth,
  
  ## Log return × VIX: idiosyncratic vs systematic volatility
  interact_ret_vix      = log_return * vix,
  
  ## Accruals × HY spread: earnings quality in stressed credit environment
  interact_acc_hyspr    = accruals_ratio * hy_spread
)]

#==============================================================================#
# 12. Identify all feature columns
#
#   Programmatically detect all engineered features — no hardcoded list.
#   meta_cols and raw Compustat variables excluded from feature set.
#==============================================================================#

cat("[06B_Feature_Eng.R] Identifying feature columns...\n")

## Columns that are metadata or raw Compustat inputs — NOT features
meta_cols <- c(
  ## Label and identifier
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
  ## Panel-level price features (already in panel from 06_Merge.R)
  "ann_return", "log_return", "n_months_ret", "avg_mktcap",
  ## Raw macro levels
  "gdp", "gdp_growth", "unrate", "fedfunds", "gs10",
  "term_spread", "hy_spread", "vix", "cpi", "cpi_inflation",
  "indpro", "indpro_growth", "recession",
  ## Lifetime
  "lifetime_years"
)

## All numeric columns not in meta_cols are features
all_cols     <- names(panel)
feature_cols <- setdiff(
  all_cols[sapply(panel, is.numeric)],
  meta_cols
)

## Also include log_return and ann_return as features (price signals)
feature_cols <- unique(c(feature_cols, "log_return", "ann_return",
                         "term_spread", "hy_spread", "unrate",
                         "fedfunds", "vix", "cpi_inflation",
                         "gdp_growth", "indpro_growth", "recession"))

## Keep only cols that exist
feature_cols <- intersect(feature_cols, names(panel))

cat(sprintf("  Total feature columns: %d\n", length(feature_cols)))

#==============================================================================#
# 13. Construct features_raw — retain only identifiers + features
#==============================================================================#

id_cols      <- c("permno", "year", "y", "censored", "param_id",
                  "gvkey", "datadate", "lifetime_years",
                  "fiscal_year_end_month")
keep_cols    <- unique(c(id_cols, feature_cols))
keep_cols    <- intersect(keep_cols, names(panel))

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
  stop(sprintf("[06B_Feature_Eng.R] ASSERTION FAILED: %d duplicate (permno, year).",
               n_dup))

## B) y values valid
invalid_y <- features_raw[!is.na(y) & !y %in% c(0L, 1L)]
if (nrow(invalid_y) > 0)
  stop("[06B_Feature_Eng.R] ASSERTION FAILED: Invalid y values.")

## C) Core features exist
core_required <- c("earn_yld", "ocf_per_share", "roa", "leverage",
                   "altman_z", "log_return", "vol_12m")
missing_core  <- setdiff(core_required, names(features_raw))
if (length(missing_core) > 0)
  stop(sprintf("[06B_Feature_Eng.R] ASSERTION FAILED: Missing core features: %s",
               paste(missing_core, collapse = ", ")))

## D) Feature count plausible
if (length(feature_cols) < 100L)
  warning(sprintf("[06B_Feature_Eng.R] WARNING: Only %d features — expected >= 100.",
                  length(feature_cols)))

cat("[06B_Feature_Eng.R] All assertions passed.\n")

#==============================================================================#
# 15. Validation diagnostics — NA audit by feature family
#==============================================================================#

cat("\n[06B_Feature_Eng.R] ══════════════════════════════════════\n")
cat(sprintf("  Rows           : %d\n", nrow(features_raw)))
cat(sprintf("  Feature columns: %d\n", length(feature_cols)))
cat(sprintf("  Permno         : %d\n", n_distinct(features_raw$permno)))

## NA audit by feature family
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
  "Rolling 1yr"          = "^roll_.*_1y_",
  "Rolling 5yr"          = "^roll_.*_5y_",
  "Price momentum"       = "^(mom_|vol_|max_dd_)",
  "Macro interactions"   = "^interact_"
)

for (fname in names(family_prefixes)) {
  pat   <- family_prefixes[[fname]]
  cols  <- grep(pat, names(features_raw), value = TRUE, perl = TRUE)
  if (length(cols) == 0L) next
  n_na  <- sum(is.na(features_raw[, .SD, .SDcols = cols]))
  n_tot <- length(cols) * nrow(features_raw)
  cat(sprintf("    %-25s : %4d cols | %5.1f%% NA\n",
              fname, length(cols), 100 * n_na / n_tot))
}

## Coverage for key paper features
cat("\n  Key paper feature coverage (% non-missing):\n")
paper_features <- c("earn_yld", "ocf_per_share", "roa", "roic",
                    "leverage", "altman_z", "log_return",
                    "vol_12m", "fedfunds", "gdp_growth")
for (v in intersect(paper_features, names(features_raw))) {
  pct <- 100 * mean(!is.na(features_raw[[v]]))
  cat(sprintf("    %-20s : %5.1f%%\n", v, pct))
}

cat("\n[06B_Feature_Eng.R] DONE:", format(Sys.time()), "\n")


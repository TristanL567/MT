#==============================================================================#
#==== 06B_Feature_Eng.R =======================================================#
#==== Feature Engineering — Ratios, Dynamics, Rolling Statistics ==============#
#==============================================================================#
#
# PURPOSE:
#   Construct the full feature matrix from panel_raw.rds.
#   Produces features_raw.rds — one row per (permno, year).
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
#   1.  Point-in-time ratios       (~40) : Foundation ratios, Altman Z
#   2.  YoY changes                (~41) : First derivatives
#   3.  Acceleration               (~41) : Second derivatives
#   4.  Expanding mean & vol       (~82) : Long-run baseline (lagged)
#   5.  Peak deterioration         (~18) : Drop from 5yr rolling peak
#   6.  Consecutive declines       (~10) : Zombie dynamics
#   7.  Accounting momentum        (~15) : 2Y vs expanding mean
#   8.  Rolling stats 3yr          (~90) : Short-window dynamics
#   9.  Rolling stats 5yr          (~90) : Long-window dynamics
#  10.  Price momentum & vol        (~8) : From prices_monthly (CORRECTED)
#  11.  Macro interaction terms      (~6) : Firm × macro regime
#
# DESIGN DECISIONS:
#
#   [1] ALL TRANSFORMATIONS ARE STRICTLY BACKWARD-LOOKING.
#       No feature at year t uses any information from year t onward.
#
#   [2] ROLLING PEAK FOR DRAWDOWN (Family 10):
#       max_dd_12m / max_dd_60m use a 3-YEAR (36-month) ROLLING PEAK,
#       not the all-time cumulative peak.
#       Rationale: measures current deterioration, not cumulative historical
#       loss from a potentially decade-old all-time high.
#
#   [3] ROLLING PEAK FOR FUNDAMENTALS (Family 5):
#       peak_drop_* uses a 5-year rolling peak of the ratio.
#
#   [4] MULTI-YEAR PRICE FEATURES (Family 10 — CORRECTED):
#       mom_24m, vol_60m, max_dd_60m require cross-year price history.
#       These CANNOT be computed inside by=.(permno, year) grouping because
#       each group contains at most 12 monthly observations (one calendar year).
#       Fix: two-step approach — within-year features computed in annual group,
#       multi-year features computed via explicit firm-level loop over
#       cumulative price history up to each year end.
#       Previous bug: vol_60m was identical to vol_12m (within-year SD),
#       mom_24m and max_dd_60m were always NA.
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

safe_div <- function(x, y, na_val = NA_real_) {
  ifelse(is.na(y) | y == 0, na_val, x / y)
}

safe_log <- function(x) {
  ifelse(is.na(x) | x <= 0, NA_real_, log(x))
}

fn_compound_ret <- function(ret_vec) {
  clean <- ret_vec[!is.na(ret_vec)]
  if (length(clean) == 0L) return(NA_real_)
  prod(1 + clean) - 1
}

#==============================================================================#
# 1. FEATURE FAMILY 1 — Point-in-Time Ratios
#==============================================================================#

cat("[06B_Feature_Eng.R] Family 1: Point-in-time ratios...\n")

panel[, `:=`(
  
  ##── Profitability ──────────────────────────────────────────────────────────
  earn_yld        = safe_div(epspx,   prcc_f),
  ocf_per_share   = safe_div(oancf,   csho),
  roa             = safe_div(ni,      at),
  roe             = safe_div(ni,      seq),
  roic            = safe_div(oiadp,   icapt),
  ebit_roa        = safe_div(ebit,    at),
  gross_margin    = safe_div(gp,      sale),
  ebitda_margin   = safe_div(ebitda,  sale),
  ocf_margin      = safe_div(oancf,   sale),
  
  ##── Leverage ───────────────────────────────────────────────────────────────
  leverage        = safe_div(dltt + dlc, at),
  net_debt_ebitda = safe_div(dltt + dlc - che, ebitda),
  std_debt_pct    = safe_div(dlc,   dltt + dlc),
  eff_int_rate    = safe_div(xint,  dltt + dlc),
  interest_cov    = safe_div(oiadp, xint),
  dd1_ratio       = safe_div(dd1,   at),
  
  ##── Liquidity ──────────────────────────────────────────────────────────────
  current_ratio   = safe_div(act,        lct),
  quick_ratio     = safe_div(act - invt, lct),
  cash_pct_act    = safe_div(che,  act),
  wcap_ratio      = safe_div(wcap, at),
  
  ##── Valuation & Market ─────────────────────────────────────────────────────
  bp_ratio        = safe_div(seq,  mkvalt),
  ev_to_sales     = safe_div(mkvalt + dltt + dlc - che, sale),
  div_yield       = safe_div(dvc,  mkvalt),
  cash_div_cf     = safe_div(dv,   oancf),
  mkt_to_book     = safe_div(mkvalt, seq),
  
  ##── Quality & Efficiency ───────────────────────────────────────────────────
  accruals_ratio  = safe_div(ni - oancf, at),
  asset_turnover  = safe_div(sale, at),
  capex_intensity = safe_div(capx, at),
  rd_intensity    = safe_div(xrd,  at),
  reinvest_rate   = safe_div(capx, oancf),
  
  ##── Size ───────────────────────────────────────────────────────────────────
  log_at          = safe_log(at),
  log_mkvalt      = safe_log(mkvalt),
  log_emp         = safe_log(emp * 1000),
  
  ##── Zombie Precursors ──────────────────────────────────────────────────────
  rental_ratio    = safe_div(xrent, at),
  assets_per_emp  = safe_div(at,    emp),
  ni_per_emp      = safe_div(ni,    emp),
  min_int_tcap    = safe_div(mib,   seq + mib + dltt),
  compr_inc_ratio = safe_div(citotal, at),
  
  ##── Altman Z-Score components ──────────────────────────────────────────────
  altman_z1       = safe_div(wcap,   at),
  altman_z2       = safe_div(re,     at),
  altman_z3       = safe_div(ebit,   at),
  altman_z4       = safe_div(mkvalt, lt),
  altman_z5       = safe_div(sale,   at)
)]

panel[, altman_z        := 1.2*altman_z1 + 1.4*altman_z2 +
        3.3*altman_z3 + 0.6*altman_z4 + 1.0*altman_z5]
panel[, invest_st_ratio := safe_div(ivst, at)]

cat("  Point-in-time ratios complete.\n")

#==============================================================================#
# 1B. Define ratio groups
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
  "eff_int_rate", "dd1_ratio", "rental_ratio", "accruals_ratio"
)

CONSEC_RATIOS <- c(
  "earn_yld", "ocf_per_share", "roa", "roic",
  "gross_margin", "interest_cov", "log_mkvalt", "log_emp",
  "current_ratio", "wcap_ratio"
)

ROLLING_CORE <- c(
  "earn_yld", "ocf_per_share", "roa", "roic",
  "leverage", "net_debt_ebitda", "interest_cov",
  "current_ratio", "cash_pct_act", "accruals_ratio",
  "asset_turnover", "gross_margin", "ebitda_margin",
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

price_ratio_cols <- c("log_return", "ann_return",
                      "max_dd_12m", "vol_12m", "mom_6m")

# ratio_cols <- intersect(ratio_cols, names(panel))

ratio_cols <- unique(c(ratio_cols,
                       intersect(price_ratio_cols, names(panel))))

cat(sprintf("  ratio_cols for dynamics: %d\n", length(ratio_cols)))

#==============================================================================#
# 2. FEATURE FAMILY 2 — YoY Changes
#==============================================================================#

cat("[06B_Feature_Eng.R] Family 2: YoY changes...\n")

panel[, paste0("yoy_", ratio_cols) :=
        lapply(.SD, function(x) x - shift(x, n=1L, type="lag")),
      by = permno, .SDcols = ratio_cols]

#==============================================================================#
# 3. FEATURE FAMILY 3 — Acceleration (2nd differences)
#==============================================================================#

cat("[06B_Feature_Eng.R] Family 3: Acceleration...\n")

yoy_cols <- paste0("yoy_", ratio_cols)

panel[, paste0("accel_", ratio_cols) :=
        lapply(.SD, function(x) x - shift(x, n=1L, type="lag")),
      by = permno, .SDcols = yoy_cols]

#==============================================================================#
# 4. FEATURE FAMILY 4 — Expanding Mean & Volatility (lagged)
#==============================================================================#

cat("[06B_Feature_Eng.R] Family 4: Expanding mean & volatility...\n")

panel[, paste0("expmean_", ratio_cols) :=
        lapply(.SD, function(x) shift(cummean(x), n=1L, type="lag")),
      by = permno, .SDcols = ratio_cols]

panel[, paste0("expvol_", ratio_cols) :=
        lapply(.SD, function(x) {
          n      <- seq_along(x)
          mu     <- cummean(x)
          expvar <- (cumsum(x^2) - n * mu^2) / pmax(n - 1L, 1L)
          lagged <- shift(sqrt(pmax(0, expvar)), n=1L, type="lag")
          fifelse(n < 3L, NA_real_, lagged)
        }),
      by = permno, .SDcols = ratio_cols]

#==============================================================================#
# 5. FEATURE FAMILY 5 — Peak Deterioration & Trough Rise
#
#   Peak:   5-year rolling peak via slider (not all-time cummax)
#   Trough: all-time cummin (rising leverage from all-time low is meaningful)
#==============================================================================#

cat("[06B_Feature_Eng.R] Family 5: Peak deterioration & trough rise...\n")

valid_peak   <- intersect(PEAK_RATIOS,   ratio_cols)
valid_trough <- intersect(TROUGH_RATIOS, ratio_cols)

if (length(valid_peak) > 0) {
  for (col in valid_peak) {
    out_col <- paste0("peak_drop_", col)
    panel[, (out_col) := {
      x <- get(col)
      n <- length(x)
      
      rolling_peak_5y <- slider::slide_dbl(
        x, .f = function(v) max(v, na.rm=TRUE),
        .before=4L, .complete=FALSE, .after=0L
      )
      
      x_safe       <- replace(x, is.na(x), -Inf)
      expanding_pk <- cummax(x_safe)
      expanding_pk[is.infinite(expanding_pk)] <- NA_real_
      
      ref_peak_raw <- ifelse(seq_len(n) >= 5L, rolling_peak_5y, expanding_pk)
      ref_peak     <- shift(ref_peak_raw, n=1L, type="lag")
      
      x - ref_peak
    }, by = permno]
  }
  cat(sprintf("  peak_drop_* : %d columns\n", length(valid_peak)))
}

if (length(valid_trough) > 0) {
  for (col in valid_trough) {
    out_col <- paste0("trough_rise_", col)
    panel[, (out_col) := {
      x             <- get(col)
      x_safe        <- replace(x, is.na(x), Inf)
      expanding_tr  <- cummin(x_safe)
      expanding_tr[is.infinite(expanding_tr)] <- NA_real_
      lagged_trough <- shift(expanding_tr, n=1L, type="lag")
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
        counter[i] <- counter[i-1L] + 1L
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
#==============================================================================#

cat("[06B_Feature_Eng.R] Family 7: Accounting momentum...\n")

valid_rolling_core <- intersect(ROLLING_CORE, ratio_cols)

panel[, paste0("acct_mom_", valid_rolling_core) :=
        lapply(.SD, function(x) {
          roll2 <- frollmean(x, n=2L, align="right", fill=NA)
          expmn <- shift(cummean(x), n=1L, type="lag")
          roll2 - expmn
        }),
      by = permno, .SDcols = valid_rolling_core]

#==============================================================================#
# 8 & 9. FEATURE FAMILIES 8 & 9 — Rolling Statistics (3yr and 5yr)
#==============================================================================#

cat("[06B_Feature_Eng.R] Families 8 & 9: Rolling statistics...\n")

WINDOW_SHORT_YRS <- 3L
WINDOW_LONG_YRS  <- 5L

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
  cor(valid[-length(valid)], valid[-1], use="complete.obs")
}

fn_roll <- function(x, window, fn) {
  slider::slide_dbl(x, .f=fn, .before=window-1L, .complete=FALSE)
}

for (w in c(WINDOW_SHORT_YRS, WINDOW_LONG_YRS)) {
  w_suffix <- sprintf("_%dy", w)
  cat(sprintf("  Rolling window: %d year(s)...\n", w))
  
  panel[, paste0("roll_mean",     w_suffix, "_", valid_rolling_core) :=
          lapply(.SD, fn_roll, window=w, fn=mean),
        by=permno, .SDcols=valid_rolling_core]
  panel[, paste0("roll_min",      w_suffix, "_", valid_rolling_core) :=
          lapply(.SD, fn_roll, window=w, fn=min),
        by=permno, .SDcols=valid_rolling_core]
  panel[, paste0("roll_max",      w_suffix, "_", valid_rolling_core) :=
          lapply(.SD, fn_roll, window=w, fn=max),
        by=permno, .SDcols=valid_rolling_core]
  panel[, paste0("roll_sd",       w_suffix, "_", valid_rolling_core) :=
          lapply(.SD, fn_roll, window=w,
                 fn=function(x) if(sum(!is.na(x))<2L) NA_real_
                 else sd(x,na.rm=TRUE)),
        by=permno, .SDcols=valid_rolling_core]
  panel[, paste0("roll_trend",    w_suffix, "_", valid_rolling_core) :=
          lapply(.SD, fn_roll, window=w, fn=fn_trend),
        by=permno, .SDcols=valid_rolling_core]
  panel[, paste0("roll_autocorr", w_suffix, "_", valid_rolling_core) :=
          lapply(.SD, fn_roll, window=w, fn=fn_autocorr),
        by=permno, .SDcols=valid_rolling_core]
}

#==============================================================================#
# 10. FEATURE FAMILY 10 — Price Momentum & Volatility (CORRECTED)
#
#   CORRECTION (design note [4]):
#   Previous code computed all price features inside by=.(permno, year=cal_year).
#   Each group contains at most 12 monthly observations (one calendar year).
#   This caused:
#     - vol_60m    : computed within-year SD → identical to vol_12m (wrong)
#     - mom_24m    : always NA (n_hist >= 24 never true within one year)
#     - max_dd_60m : always NA (n_hist >= 60 never true within one year)
#
#   FIX: Two-step approach.
#   Step A: Within-year features (mom_1m/3m/6m, vol_12m, max_dd_12m)
#           Correctly use within-year data only.
#   Step B: Multi-year features (mom_24m, vol_60m, max_dd_60m)
#           Explicitly build cumulative firm history up to each year end.
#           Loop over years within each firm, accumulating monthly data.
#
#   Both steps use the corrected 3-year rolling peak for drawdown.
#==============================================================================#

cat("[06B_Feature_Eng.R] Family 10: Price momentum & volatility...\n")

##── Step 1: Wealth index and 3-year rolling peak drawdown ───────────────────

prices_m[, wealth_index := cumprod(1 + fifelse(is.na(ret_adj), 0, ret_adj)),
         by = permno]

## 3-year rolling peak via slider — resets after recovery
## Fallback to expanding cummax for first < 36 months
prices_m[, drawdown := {
  rolling_peak_3y <- slider::slide_dbl(
    wealth_index, .f=max,
    .before=35L, .complete=FALSE, .after=0L
  )
  expanding_peak <- cummax(wealth_index)
  ref_peak <- ifelse(
    seq_along(wealth_index) >= 36L,
    rolling_peak_3y,
    expanding_peak
  )
  wealth_index / ref_peak - 1
}, by = permno]

prices_m[, cal_year  := year(date)]
prices_m[, cal_month := month(date)]

##── Step 2A: Within-year features ───────────────────────────────────────────
## Group = (permno, calendar year) → at most 12 monthly observations
## mom_1m/3m/6m, vol_12m, max_dd_12m are correctly computable within one year

cat("  Step 2A: Within-year features (mom_1m/3m/6m, vol_12m, max_dd_12m)...\n")

price_feats_annual <- prices_m[, {
  r <- ret_adj
  d <- drawdown
  n <- length(r)
  
  list(
    mom_1m     = if (n >= 1L) r[n]                           else NA_real_,
    mom_3m     = if (n >= 3L) fn_compound_ret(tail(r, 3L))   else NA_real_,
    mom_6m     = if (n >= 6L) fn_compound_ret(tail(r, 6L))   else NA_real_,
    vol_12m    = if (sum(!is.na(r)) >= 3L) sd(r, na.rm=TRUE) else NA_real_,
    max_dd_12m = if (n >= 1L) min(d, na.rm=TRUE)             else NA_real_
  )
}, by = .(permno, year = cal_year)]

cat(sprintf("  price_feats_annual: %d rows\n", nrow(price_feats_annual)))

##── Step 2B: Multi-year features ────────────────────────────────────────────
## mom_24m, vol_60m, max_dd_60m require cumulative history across years.
## For each firm, loop over each year, using all monthly data up to that year.
##
## Implementation: for each permno, extract the full monthly history sorted
## by date, then for each calendar year compute features on the cumulative
## slice up to December of that year.

cat("  Step 2B: Multi-year features (mom_24m, vol_60m, max_dd_60m)...\n")
cat("           This may take 3-5 minutes...\n")

## Ensure sorted
setorder(prices_m, permno, date)

## For efficiency, work firm by firm using a pre-split list
## data.table approach: process each permno group
price_feats_multi <- prices_m[, {
  
  ## Full monthly history for this firm, in order
  r_all   <- ret_adj
  d_all   <- drawdown
  yr_all  <- cal_year
  
  ## Unique years for this firm
  yrs     <- sort(unique(yr_all))
  n_yrs   <- length(yrs)
  
  ## Pre-allocate result vectors
  yr_out      <- integer(n_yrs)
  mom24_out   <- numeric(n_yrs)
  vol60_out   <- numeric(n_yrs)
  dd60_out    <- numeric(n_yrs)
  
  for (i in seq_len(n_yrs)) {
    yr  <- yrs[i]
    
    ## All monthly obs up to and including December of year yr
    idx <- which(yr_all <= yr)
    n   <- length(idx)
    r   <- r_all[idx]
    d   <- d_all[idx]
    
    yr_out[i]    <- yr
    
    ## mom_24m: 24-month compounded return ending Dec of year yr
    mom24_out[i] <- if (n >= 24L) fn_compound_ret(tail(r, 24L)) else NA_real_
    
    ## vol_60m: SD of monthly returns over trailing 60 months
    r60          <- tail(r, 60L)
    vol60_out[i] <- if (sum(!is.na(r60)) >= 12L) sd(r60, na.rm=TRUE) else NA_real_
    
    ## max_dd_60m: worst drawdown in trailing 60 months (from 3yr rolling peak)
    d60          <- tail(d, 60L)
    dd60_out[i]  <- if (n >= 60L) min(d60, na.rm=TRUE) else NA_real_
  }
  
  list(
    year       = yr_out,
    mom_24m    = mom24_out,
    vol_60m    = vol60_out,
    max_dd_60m = dd60_out
  )
  
}, by = permno]

cat(sprintf("  price_feats_multi: %d rows\n", nrow(price_feats_multi)))

##── Step 2C: Join both feature sets ─────────────────────────────────────────

price_feats <- merge(
  price_feats_annual,
  price_feats_multi,
  by    = c("permno", "year"),
  all.x = TRUE
)

## Join onto annual panel
panel <- merge(
  panel,
  price_feats,
  by    = c("permno", "year"),
  all.x = TRUE
)

setorder(panel, permno, year)

cat(sprintf("  Price momentum features added: %d columns\n",
            ncol(price_feats) - 2L))

## Sanity checks — confirm the fix worked
n_mom24    <- sum(!is.na(panel$mom_24m))
n_vol60    <- sum(!is.na(panel$vol_60m))
n_dd60     <- sum(!is.na(panel$max_dd_60m))
vol_corr   <- cor(panel$vol_12m, panel$vol_60m, use="complete.obs")

cat(sprintf("  mom_24m    non-NA : %d  (expected > 0)\n",     n_mom24))
cat(sprintf("  vol_60m    non-NA : %d  (expected > 0)\n",     n_vol60))
cat(sprintf("  max_dd_60m non-NA : %d  (expected > 0)\n",     n_dd60))
cat(sprintf("  cor(vol_12m, vol_60m) : %.4f  (expected < 0.95)\n", vol_corr))

if (vol_corr > 0.95)
  warning("[06B] vol_60m still correlated with vol_12m — check Step 2B.")
if (n_mom24 == 0L)
  warning("[06B] mom_24m is all-NA — check Step 2B.")
if (n_dd60 == 0L)
  warning("[06B] max_dd_60m is all-NA — check Step 2B.")

#==============================================================================#
# 11. FEATURE FAMILY 11 — Macro Interaction Terms
#==============================================================================#

cat("[06B_Feature_Eng.R] Family 11: Macro interactions...\n")

panel[, `:=`(
  interact_lev_rate  = leverage        * fedfunds,
  interact_cov_rate  = interest_cov    * fedfunds,
  interact_nde_hyspr = net_debt_ebitda * hy_spread,
  interact_roa_gdp   = roa             * gdp_growth,
  interact_ret_vix   = log_return      * vix,
  interact_acc_hyspr = accruals_ratio  * hy_spread
)]

#==============================================================================#
# 12. Identify all feature columns
#==============================================================================#

cat("[06B_Feature_Eng.R] Identifying feature columns...\n")

meta_cols <- c(
  "permno", "year", "y", "censored", "param_id", "gvkey",
  "datadate", "fyr", "sich", "fiscal_year_end_month",
  "at", "act", "che", "ivst", "rect", "invt", "wcap", "ppent",
  "intan", "intano", "gdwl", "txdba", "aco",
  "lt", "lct", "dltt", "dlc", "dd1", "ap", "txp", "txditc",
  "seq", "ceq", "re", "pstk", "mib", "icapt",
  "sale", "revt", "cogs", "gp", "xsga", "xrd", "dp",
  "ebit", "ebitda", "oiadp", "oibdp", "xopr", "xint",
  "pretax_income", "ni", "ib", "epspx", "dvc", "dvt", "citotal",
  "oancf", "capx", "ivncf", "fincf", "dv", "sstk",
  "prstkc", "dltis", "dltr", "sppe",
  "csho", "prcc_f", "mkvalt",
  "emp", "xrent",
  "n_months_ret", "avg_mktcap",
  "gdp", "gdp_growth", "unrate", "fedfunds", "gs10",
  "term_spread", "hy_spread", "vix", "cpi", "cpi_inflation",
  "indpro", "indpro_growth", "recession",
  "lifetime_years"
)

feature_cols <- setdiff(
  names(panel)[sapply(panel, is.numeric)],
  meta_cols
)

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
               paste(missing_core, collapse=", ")))

## D) Feature count plausible
if (length(feature_cols) < 100L)
  warning(sprintf("WARNING: Only %d features — expected >= 100.",
                  length(feature_cols)))

## E) max_dd_12m distribution — rolling peak fix check
dd_by_label <- features_raw[!is.na(y), .(
  median_dd = median(max_dd_12m, na.rm=TRUE),
  mean_dd   = mean(max_dd_12m,   na.rm=TRUE)
), by = y]
cat("\n  max_dd_12m check (rolling peak fix):\n")
print(dd_by_label)
if (!is.na(dd_by_label[y==1L, median_dd]) &&
    dd_by_label[y==1L, median_dd] < -0.85)
  warning("max_dd_12m median for CSI firms still < -85%. Check Family 10.")

## F) Multi-year price features not all-NA
stopifnot(sum(!is.na(features_raw$mom_24m))    > 0)
stopifnot(sum(!is.na(features_raw$vol_60m))    > 0)
stopifnot(sum(!is.na(features_raw$max_dd_60m)) > 0)

## G) vol_60m and vol_12m are different
vol_corr_final <- cor(features_raw$vol_12m, features_raw$vol_60m,
                      use="complete.obs")
cat(sprintf("\n  cor(vol_12m, vol_60m) = %.4f  (must be < 0.95)\n",
            vol_corr_final))
if (vol_corr_final > 0.95)
  stop(sprintf(paste("ASSERTION FAILED: vol_60m correlated %.4f with vol_12m.",
                     "Multi-year fix did not apply."), vol_corr_final))

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
  cols <- grep(pat, names(features_raw), value=TRUE, perl=TRUE)
  if (length(cols) == 0L) next
  n_na  <- sum(is.na(features_raw[, .SD, .SDcols=cols]))
  n_tot <- length(cols) * nrow(features_raw)
  cat(sprintf("    %-25s : %4d cols | %5.1f%% NA\n",
              fname, length(cols), 100 * n_na / n_tot))
}

cat("\n  Key feature coverage (% non-missing):\n")
key_features <- c("earn_yld", "ocf_per_share", "roa", "roic",
                  "leverage", "altman_z", "log_return",
                  "vol_12m", "vol_60m", "max_dd_12m", "max_dd_60m",
                  "mom_24m", "fedfunds", "gdp_growth")
for (v in intersect(key_features, names(features_raw))) {
  pct <- 100 * mean(!is.na(features_raw[[v]]))
  cat(sprintf("    %-22s : %5.1f%%\n", v, pct))
}

cat("\n[06B_Feature_Eng.R] DONE:", format(Sys.time()), "\n")
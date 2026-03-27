#==============================================================================#
#==== 04_Macro.R ==============================================================#
#==== FRED Macroeconomic Variable Download & Monthly Alignment ================#
#==============================================================================#
#
# PURPOSE:
#   Download macroeconomic variables from FRED via the fredr package.
#   Aligns all series to a common monthly frequency via forward-filling,
#   producing a clean (date, variable) panel ready for 06_Merge.R.
#
# INPUTS:
#   - config.R                      : START_DATE, END_DATE, PATH_* constants
#   - FRED API key                  : set via fredr_set_key() in 00_Master.R
#
# OUTPUTS:
#   - Raw/macro_raw.rds             : list of raw FRED series, as-downloaded
#   - Processed/macro_monthly.rds   : all series aligned to monthly,
#                                     wide format, one row per month-end date
#
#   Output schema (macro_monthly.rds):
#
#   LEVELS (kept as-is — directional content already encoded or level is the
#   meaningful feature):
#     date (month-end)
#     gdp             Gross Domestic Product, $bn (quarterly → fwd-filled)
#     unrate          Unemployment rate, % (monthly level)
#     fedfunds        Federal funds effective rate, % (monthly level)
#     gs10            10-yr Treasury rate, % (monthly level)
#     term_spread     GS10 − FEDFUNDS, % (monthly level)
#     hy_spread       ICE BofA HY OAS, % (monthly level)
#     vix             CBOE VIX, index level (monthly average of daily)
#     cpi             CPI all-urban index (monthly level — used for inflation)
#     indpro          Industrial production index (monthly level)
#     recession       NBER recession indicator, 0/1 (monthly)
#
#   YoY CHANGES (12-month change — computed here before any aggregation):
#     gdp_growth      GDP:    (gdp_t / gdp_t-12) - 1, %
#     cpi_inflation   CPI:    (cpi_t / cpi_t-12) - 1, %
#     indpro_growth   INDPRO: (indpro_t / indpro_t-12) - 1, %
#     d_unrate        UNRATE: unrate_t - unrate_t-12 (pp change)
#     d_hy_spread     HY OAS: hy_spread_t - hy_spread_t-12 (pp change)
#     d_vix           VIX:    (vix_t / vix_t-12) - 1, %
#
#   RATIONALE FOR VARIABLE TREATMENT:
#
#   GDP / CPI / INDPRO → YoY ratio change
#     These are trending indices. The level is non-stationary and carries
#     no cross-sectional information; the growth rate is the economically
#     meaningful quantity for firm distress prediction.
#
#   UNRATE → level + YoY pp change
#     Level: captures the absolute slack in the labour market (higher
#     unemployment = weaker demand environment for firms).
#     Change: captures the direction — a rising rate signals deteriorating
#     conditions even if the level is still moderate. Both matter.
#
#   HY_SPREAD → level + YoY pp change
#     Level: captures prevailing credit stress (high spread = tight credit).
#     Change: a rapidly widening spread is a leading indicator of distress
#     clusters, often precedes CSI events by 6–12 months.
#
#   VIX → level + YoY % change
#     Level: contemporaneous fear gauge.
#     Change: a sustained rise in VIX amplifies distress risk beyond the
#     level effect. % change preferred over pp because VIX is bounded below
#     and the absolute magnitude varies across regimes.
#
#   FEDFUNDS / GS10 / TERM_SPREAD → level only
#     Rate levels interact directly with leverage (debt service burden).
#     Rate *changes* are less theoretically motivated for annual prediction:
#     the firm's balance sheet responds to the prevailing rate, not the
#     change. Term spread already encodes the directional regime signal.
#
# DESIGN DECISIONS:
#   [1] Forward-fill quarterly/monthly → monthly spine (no look-ahead).
#   [2] Growth rates computed here on monthly data, before annual aggregation
#       in 06_Merge.R. This avoids aggregating raw levels and then computing
#       growth from the aggregate (which would conflate within-year and
#       year-over-year changes).
#   [3] Raw series saved before any transformation (immutable reference).
#   [4] All lags use 12 months on the already-monthly series to ensure
#       consistent YoY comparisons regardless of quarterly forward-filling.
#
#==============================================================================#

source("config.R")

suppressPackageStartupMessages({
  library(fredr)
  library(dplyr)
  library(tidyr)
  library(lubridate)
  library(data.table)
})

cat("\n[04_Macro.R] START:", format(Sys.time()), "\n")

#==============================================================================#
# 0. FRED API authentication
#==============================================================================#

FRED_API_KEY <- "ea754f3d4a236f5d4b2c8957fae36c4a"
fredr_set_key(FRED_API_KEY)

tryCatch({
  fredr_series("GDP")
  cat("[04_Macro.R] FRED API connection verified.\n")
}, error = function(e) {
  stop("[04_Macro.R] FRED API connection failed. Check API key and internet access.")
})

#==============================================================================#
# 1. Define series to pull
#==============================================================================#

## Pull extra history for 12-month lags — 2 years before START_DATE
PULL_START <- START_DATE - years(2)

FRED_SERIES <- list(
  list(id = "GDP",          name = "gdp",       note = "Quarterly — forward-filled to monthly"),
  list(id = "UNRATE",       name = "unrate",    note = "Monthly level"),
  list(id = "FEDFUNDS",     name = "fedfunds",  note = "Monthly level"),
  list(id = "GS10",         name = "gs10",      note = "Monthly level"),
  list(id = "BAMLH0A0HYM2", name = "hy_spread", note = "Daily — averaged to monthly"),
  list(id = "VIXCLS",       name = "vix",       note = "Daily — averaged to monthly"),
  list(id = "CPIAUCSL",     name = "cpi",       note = "Monthly index level"),
  list(id = "INDPRO",       name = "indpro",    note = "Monthly index level"),
  list(id = "USREC",        name = "recession", note = "Monthly — NBER indicator 0/1")
)

#==============================================================================#
# 2. Download all series from FRED
#==============================================================================#

cat("[04_Macro.R] Pulling", length(FRED_SERIES), "series from FRED...\n")

macro_raw <- list()

for (s in FRED_SERIES) {
  cat(sprintf("  Pulling %-15s (%s)...", s$id, s$note))
  
  result <- tryCatch({
    fredr(
      series_id         = s$id,
      observation_start = PULL_START,
      observation_end   = END_DATE
    ) |>
      select(date, value) |>
      rename(!!s$name := value)
  }, error = function(e) {
    warning(sprintf("\n[04_Macro.R] WARNING: Failed to pull %s: %s",
                    s$id, e$message))
    NULL
  })
  
  if (!is.null(result)) {
    macro_raw[[s$name]] <- result
    cat(sprintf(" %d observations\n", nrow(result)))
  } else {
    cat(" FAILED — skipped\n")
  }
}

saveRDS(macro_raw, PATH_MACRO_RAW)
cat("[04_Macro.R] Raw series saved to PATH_MACRO_RAW\n")

#==============================================================================#
# 3. Build monthly date spine (month-end dates)
#==============================================================================#

date_spine <- tibble(
  date = seq(
    from = floor_date(PULL_START, "month"),
    to   = floor_date(END_DATE,   "month"),
    by   = "month"
  )
) |>
  mutate(date = ceiling_date(date, "month") - days(1))

cat("[04_Macro.R] Monthly date spine:", nrow(date_spine), "months from",
    format(min(date_spine$date)), "to", format(max(date_spine$date)), "\n")

#==============================================================================#
# 4. Align each series to monthly frequency
#
#   QUARTERLY (GDP): assign to quarter end, forward-fill to monthly.
#   DAILY (VIX, HY spread): average within each calendar month.
#   MONTHLY (all others): align to month-end, forward-fill any gaps.
#==============================================================================#

fn_align_to_monthly <- function(series_df, series_name, date_spine) {
  
  col_name   <- setdiff(names(series_df), "date")
  date_gaps  <- as.integer(diff(sort(series_df$date)))
  median_gap <- median(date_gaps, na.rm = TRUE)
  
  if (median_gap <= 7) {
    ## DAILY → monthly average
    aligned <- series_df |>
      mutate(date = ceiling_date(floor_date(date, "month"), "month") - days(1)) |>
      group_by(date) |>
      summarise(!!col_name := mean(.data[[col_name]], na.rm = TRUE),
                .groups = "drop")
    
  } else if (median_gap >= 80) {
    ## QUARTERLY → assign to last month of quarter, then forward-fill
    aligned <- series_df |>
      mutate(date = ceiling_date(date + months(2), "month") - days(1)) |>
      select(date, !!col_name)
    
  } else {
    ## MONTHLY → align to month-end
    aligned <- series_df |>
      mutate(date = ceiling_date(floor_date(date, "month"), "month") - days(1)) |>
      group_by(date) |>
      summarise(!!col_name := mean(.data[[col_name]], na.rm = TRUE),
                .groups = "drop")
  }
  
  ## Join to spine and forward-fill (no look-ahead — uses last known value)
  result <- date_spine |>
    left_join(aligned, by = "date") |>
    arrange(date) |>
    fill(all_of(col_name), .direction = "down")
  
  return(result)
}

#==============================================================================#
# 5. Align all series and join to single wide table
#==============================================================================#

cat("[04_Macro.R] Aligning all series to monthly frequency...\n")

macro_aligned <- date_spine

for (s in FRED_SERIES) {
  if (!is.null(macro_raw[[s$name]])) {
    aligned <- fn_align_to_monthly(macro_raw[[s$name]], s$name, date_spine)
    macro_aligned <- left_join(macro_aligned, aligned, by = "date")
    cat(sprintf("  Aligned: %-12s — missing months after fill: %d\n",
                s$name,
                sum(is.na(macro_aligned[[s$name]]))))
  }
}

#==============================================================================#
# 6. Compute derived macro features
#
#   All computations operate on the monthly-aligned series.
#   12-month lags are used throughout for consistent YoY comparisons.
#
#   RATIO changes (gdp, cpi, indpro):
#     (x_t / x_{t-12}) - 1  expressed as a decimal (not multiplied by 100)
#     Reason: consistent with how these appear in the ML feature matrix
#     after quantile transform. Decimal form avoids scale discontinuity
#     when combined with level features.
#
#   PP changes (unrate, hy_spread):
#     x_t - x_{t-12}
#     Reason: both variables are already expressed in % or percentage points,
#     so a pp change is natural and directly interpretable.
#
#   VIX change: (vix_t / vix_{t-12}) - 1
#     Reason: VIX is strictly positive and the distribution is log-approximately
#     normal. A 10-point move from 12 to 22 is qualitatively different from
#     a 10-point move from 35 to 45. % change normalises for the level.
#
#   Term spread: gs10 - fedfunds (level, computed here)
#     Negative term spread = yield curve inversion = recession precursor.
#==============================================================================#

cat("[04_Macro.R] Computing derived macro features...\n")

macro_monthly <- macro_aligned |>
  arrange(date) |>
  mutate(
    
    ##── YoY ratio changes ────────────────────────────────────────────────────
    gdp_growth    = gdp    / lag(gdp,    12) - 1,   ## decimal, e.g. 0.025 = 2.5%
    cpi_inflation = cpi    / lag(cpi,    12) - 1,   ## decimal
    indpro_growth = indpro / lag(indpro, 12) - 1,   ## decimal
    
    ##── YoY percentage-point changes ─────────────────────────────────────────
    ## Change in unemployment rate (pp): captures direction of labour market
    d_unrate      = unrate    - lag(unrate,    12),  ## pp, e.g. +1.5 = rate rose 1.5pp
    
    ## Change in HY spread (pp): rising spread = tightening credit = distress precursor
    d_hy_spread   = hy_spread - lag(hy_spread, 12),  ## pp
    
    ##── VIX: YoY % change ────────────────────────────────────────────────────
    ## % change preferred over pp — see rationale above
    d_vix         = vix / lag(vix, 12) - 1,          ## decimal
    
    ##── Term spread (level) ──────────────────────────────────────────────────
    term_spread   = gs10 - fedfunds,                  ## pp, level
    
    ##── Recession flag ───────────────────────────────────────────────────────
    recession     = as.integer(recession)
  ) |>
  
  ## Restrict to analysis window after lag computation
  filter(date >= ceiling_date(floor_date(START_DATE, "month"), "month") - days(1))

cat("[04_Macro.R] Derived features computed.\n")

## Report NA counts for derived variables (expected for first 12 months)
derived_vars <- c("gdp_growth", "cpi_inflation", "indpro_growth",
                  "d_unrate", "d_hy_spread", "d_vix", "term_spread")
cat("  NA counts for derived features (expected ~12 for YoY vars):\n")
for (v in derived_vars) {
  cat(sprintf("    %-18s : %d NAs\n", v, sum(is.na(macro_monthly[[v]]))))
}

#==============================================================================#
# 7. Assertions
#==============================================================================#

cat("[04_Macro.R] Running assertions...\n")

## A) No duplicate dates
n_dup <- sum(duplicated(macro_monthly$date))
if (n_dup > 0)
  stop(sprintf("[04_Macro.R] ASSERTION FAILED: %d duplicate dates.", n_dup))

## B) Date range covers full analysis window
if (min(macro_monthly$date) > as.Date("1993-01-31"))
  stop("[04_Macro.R] ASSERTION FAILED: macro_monthly starts after START_DATE.")
if (max(macro_monthly$date) < as.Date("2024-11-30"))
  stop("[04_Macro.R] ASSERTION FAILED: macro_monthly ends before END_DATE.")

## C) Core series not mostly missing
core_vars <- c("gdp", "unrate", "fedfunds", "cpi", "hy_spread",
               "gdp_growth", "cpi_inflation", "d_unrate", "d_hy_spread")
for (v in core_vars) {
  if (v %in% names(macro_monthly)) {
    pct_na <- mean(is.na(macro_monthly[[v]]))
    if (pct_na > 0.10)
      warning(sprintf("[04_Macro.R] WARNING: %s is %.1f%% missing after fill.",
                      v, 100 * pct_na))
  }
}

## D) YoY direction checks — sanity on sign
## During 2008-2009 recession, gdp_growth should go negative
gdp_2009 <- macro_monthly |>
  filter(format(date, "%Y") == "2009") |>
  pull(gdp_growth) |>
  mean(na.rm = TRUE)
if (!is.na(gdp_2009) && gdp_2009 > 0)
  warning("[04_Macro.R] WARNING: gdp_growth positive in 2009 — check computation.")

## E) Plausible row count
expected_months <- interval(START_DATE, END_DATE) %/% months(1) + 1L
if (nrow(macro_monthly) < expected_months - 2L)
  stop(sprintf("[04_Macro.R] ASSERTION FAILED: Only %d rows, expected ~%d.",
               nrow(macro_monthly), expected_months))

cat("[04_Macro.R] All assertions passed.\n")

#==============================================================================#
# 8. Save
#==============================================================================#

saveRDS(macro_monthly, PATH_MACRO_MONTHLY)
cat("[04_Macro.R] Monthly macro panel saved:", nrow(macro_monthly), "rows\n")

#==============================================================================#
# 9. Summary diagnostics
#==============================================================================#

cat("\n[04_Macro.R] ══════════════════════════════════════\n")
cat("  Rows         :", nrow(macro_monthly), "\n")
cat("  Date range   :", format(min(macro_monthly$date)),
    "to", format(max(macro_monthly$date)), "\n")
cat("  Columns      :", ncol(macro_monthly), "\n")

cat("\n  Variable treatment summary:\n")
cat("    LEVELS    : gdp, unrate, fedfunds, gs10, hy_spread, vix, cpi,",
    "indpro, term_spread, recession\n")
cat("    YoY ratio : gdp_growth, cpi_inflation, indpro_growth, d_vix\n")
cat("    YoY pp    : d_unrate, d_hy_spread\n")

cat("\n  Variable coverage (% non-missing):\n")
all_vars <- setdiff(names(macro_monthly), "date")
for (v in all_vars) {
  pct <- 100 * mean(!is.na(macro_monthly[[v]]))
  cat(sprintf("    %-20s : %5.1f%%\n", v, pct))
}

cat("\n  Macro snapshot — most recent month:\n")
print(tail(macro_monthly, 1))

cat("\n[04_Macro.R] DONE:", format(Sys.time()), "\n")
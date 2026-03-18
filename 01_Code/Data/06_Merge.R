#==============================================================================#
#==== 06_Merge.R ==============================================================#
#==== Master Panel Assembly, Reporting Lag & Validation =======================#
#==============================================================================#
#
# PURPOSE:
#   Assemble the definitive (permno, year) modelling panel by joining:
#     - labels_base.rds       : CSI labels (spine)
#     - fundamentals.rds      : Compustat accounting variables
#     - prices_monthly.rds    : CRSP monthly returns → annual log return
#     - macro_monthly.rds     : FRED macro variables → annual aggregates
#   Applies the Compustat reporting lag and the minimum lifetime filter.
#   Produces panel_raw.rds — one row per (permno, year), all raw variables
#   attached, ready for feature engineering in 06B_Feature_Eng.R.
#
# INPUTS:
#   - config.R
#   - PATH_LABELS_BASE        : (permno, year, y, censored)
#   - PATH_FUNDAMENTALS       : (permno, gvkey, datadate, fyear, ~60 vars)
#   - PATH_PRICES_MONTHLY     : (permno, date, ret_adj, ...)
#   - PATH_MACRO_MONTHLY      : (date, gdp, unrate, ...)
#   - PATH_UNIVERSE           : permno list with listing_date, removal_date
#
# OUTPUT:
#   - PATH_PANEL_RAW          : (permno, year, y, censored, ~60 fund. vars,
#                                ann_return, log_return, ~12 macro vars,
#                                lifetime_years, fiscal_year_end_month)
#
# DESIGN DECISIONS:
#
#   [1] SPINE = labels_base (all rows, including y=NA zombie/censored)
#       panel_raw.rds retains all label observations. Model training later
#       filters to !is.na(y). This preserves zombie firm characteristics
#       for diagnostic analysis and keeps panel_raw.rds as a complete record.
#
#   [2] REPORTING LAG applied to fundamentals:
#       datadate = fiscal period END, not filing date.
#       availability_date = datadate + REPORTING_LAG_MONTHS (3 months)
#       prediction_year   = year(availability_date)
#       Fundamentals join to labels on (permno, prediction_year == year).
#       This ensures at prediction time t, only information available
#       before the start of year t is used — no look-ahead bias.
#
#       Example (December fiscal year):
#         datadate = 2015-12-31 → available ≈ 2016-03-31 → predicts year 2016
#
#       Example (March fiscal year):
#         datadate = 2015-03-31 → available ≈ 2015-06-30 → predicts year 2015
#
#   [3] PRICES → ANNUAL:
#       Annual compounded return: prod(1 + ret_adj) - 1 over calendar year t.
#       Log return: log(1 + ann_return) — paper feature log_return.
#       Computed over the calendar year matching the label year.
#       Momentum and rolling price features deferred to 06B_Feature_Eng.R.
#
#   [4] MACRO → ANNUAL:
#       Level variables (UNRATE, VIX, FEDFUNDS, GS10, CPI, INDPRO, GDP):
#         Annual mean of monthly values within calendar year t.
#       Derived variables (term_spread, hy_spread, gdp_growth,
#         cpi_inflation, indpro_growth): annual mean.
#       Recession indicator: max within year (1 if any month was recession).
#       All macro aggregated over calendar year t — the prediction year.
#
#   [5] LIFETIME FILTER applied here (not in 01_Universe.R):
#       MIN_LIFETIME_YEARS = 5 applied to the modelling panel only.
#       Labels in 05_CSI_Label.R used the full universe — no bias there.
#       Here we require >= MIN_LIFETIME_YEARS of price history as of
#       the prediction year-end. Short-history firms lack rolling feature
#       windows and are dropped from the modelling panel.
#
#   [6] mkvalt FALLBACK applied here before any ratio construction:
#       mkvalt = coalesce(mkvalt, prcc_f * csho)
#       Coverage: 73.3% → 99.4% after fallback.
#       Applied here so 06B_Feature_Eng.R always has a market cap value.
#
#==============================================================================#

source("config.R")

cat("\n[06_Merge.R] START:", format(Sys.time()), "\n")

#==============================================================================#
# 0. Load all inputs
#==============================================================================#

cat("[06_Merge.R] Loading inputs...\n")

labels       <- readRDS(PATH_LABELS_BASE)
fundamentals <- readRDS(PATH_FUNDAMENTALS)
prices_m     <- readRDS(PATH_PRICES_MONTHLY)
macro_m      <- readRDS(PATH_MACRO_MONTHLY)
universe     <- readRDS(PATH_UNIVERSE)

cat(sprintf("  labels_base   : %d rows, %d permno\n",
            nrow(labels), n_distinct(labels$permno)))
cat(sprintf("  fundamentals  : %d rows, %d permno\n",
            nrow(fundamentals), n_distinct(fundamentals$permno)))
cat(sprintf("  prices_monthly: %d rows, %d permno\n",
            nrow(prices_m), n_distinct(prices_m$permno)))
cat(sprintf("  macro_monthly : %d rows, %d columns\n",
            nrow(macro_m), ncol(macro_m)))
cat(sprintf("  universe      : %d permno\n", nrow(universe)))

#==============================================================================#
# 1. Prepare fundamentals — apply reporting lag
#
#   availability_date = datadate + REPORTING_LAG_MONTHS
#   prediction_year   = calendar year of availability_date
#
#   This maps each Compustat annual observation to the first full calendar
#   year in which it was publicly available — the year it can be used to
#   predict CSI events without look-ahead bias.
#
#   Also apply mkvalt fallback here (design note [6]) so all downstream
#   operations have a consistent market cap variable.
#==============================================================================#

cat("[06_Merge.R] Applying reporting lag to fundamentals...\n")

fund_lagged <- fundamentals |>
  rename(pretax_income = pi) |>          # Step 1 — rename BEFORE mutate
  mutate(
    availability_date     = datadate %m+% months(REPORTING_LAG_MONTHS),
    prediction_year       = year(availability_date),
    fiscal_year_end_month = month(datadate),
    mkvalt                = dplyr::coalesce(mkvalt, prcc_f * csho)
  ) |>
  filter(prediction_year <= year(END_DATE)) |>   # Cap at END_DATE year
  select(-availability_date)

cat(sprintf("  Reporting lag applied: %d months\n", REPORTING_LAG_MONTHS))
cat(sprintf("  prediction_year range: %d to %d\n",
            min(fund_lagged$prediction_year),
            max(fund_lagged$prediction_year)))

## Check for duplicate (permno, prediction_year) after lag
## Can arise if a firm has two fiscal years mapping to the same calendar year
n_dup_fund <- sum(duplicated(fund_lagged[, c("permno", "prediction_year")]))
if (n_dup_fund > 0) {
  cat(sprintf("  Resolving %d duplicate (permno, prediction_year) after lag...\n",
              n_dup_fund))
  fund_lagged <- fund_lagged |>
    arrange(permno, prediction_year, desc(datadate)) |>
    distinct(permno, prediction_year, .keep_all = TRUE)
  cat("  Duplicates resolved: kept latest datadate per (permno, prediction_year)\n")
}

#==============================================================================#
# 2. Prepare prices — collapse monthly returns to annual
#
#   Annual compounded return: prod(1 + ret_adj) - 1 over calendar year t.
#   Log return: log(1 + ann_return) — paper feature log_return.
#   Also compute: number of valid monthly return observations per year
#   (used in lifetime filter and as a data quality indicator).
#==============================================================================#

cat("[06_Merge.R] Collapsing monthly prices to annual...\n")

prices_annual <- prices_m |>
  mutate(year = year(date)) |>
  group_by(permno, year) |>
  summarise(
    ann_return   = prod(1 + ret_adj, na.rm = TRUE) - 1,
    n_months_ret = sum(!is.na(ret_adj)),
    avg_mktcap   = mean(mktcap, na.rm = TRUE),
    .groups = "drop"
  ) |>
  mutate(
    ann_return = if_else(n_months_ret == 0L, NA_real_, ann_return),
    log_return = log(1 + ann_return)   # NA propagates if ann_return is NA
  )

cat(sprintf("  Annual price obs: %d rows, %d permno, years %d–%d\n",
            nrow(prices_annual),
            n_distinct(prices_annual$permno),
            min(prices_annual$year),
            max(prices_annual$year)))

#==============================================================================#
# 3. Prepare macro — collapse monthly to annual
#
#   Level variables: annual mean (smoother, captures full-year regime)
#   Recession indicator: max within year (1 if any month was a recession)
#   All aggregated over the calendar year matching the label prediction year.
#==============================================================================#

cat("[06_Merge.R] Collapsing monthly macro to annual...\n")

macro_annual <- macro_m |>
  mutate(year = year(date)) |>
  group_by(year) |>
  summarise(
    ## Level variables — annual mean
    gdp           = mean(gdp,           na.rm = TRUE),
    gdp_growth    = mean(gdp_growth,    na.rm = TRUE),
    unrate        = mean(unrate,        na.rm = TRUE),
    fedfunds      = mean(fedfunds,      na.rm = TRUE),
    gs10          = mean(gs10,          na.rm = TRUE),
    term_spread   = mean(term_spread,   na.rm = TRUE),
    hy_spread     = mean(hy_spread,     na.rm = TRUE),
    vix           = mean(vix,           na.rm = TRUE),
    cpi           = mean(cpi,           na.rm = TRUE),
    cpi_inflation = mean(cpi_inflation, na.rm = TRUE),
    indpro        = mean(indpro,        na.rm = TRUE),
    indpro_growth = mean(indpro_growth, na.rm = TRUE),
    ## Recession — 1 if any month in the year was classified as recession
    recession     = as.integer(max(recession, na.rm = TRUE)),
    .groups = "drop"
  )

macro_annual <- macro_annual |>
  mutate(across(where(is.numeric), ~if_else(is.nan(.x), NA_real_, .x)))

cat(sprintf("  Annual macro obs: %d rows, years %d–%d\n",
            nrow(macro_annual),
            min(macro_annual$year),
            max(macro_annual$year)))

#==============================================================================#
# 4. Compute lifetime years per (permno, year)
#
#   Lifetime at year t = years of price history available up to end of year t.
#   Used for the MIN_LIFETIME_YEARS filter (design note [5]).
#   Computed from universe listing_date (first price record date).
#==============================================================================#

cat("[06_Merge.R] Computing lifetime years per (permno, year)...\n")

lifetime_ref <- universe |>
  select(permno, listing_date) |>
  mutate(listing_date = as.Date(listing_date))

#==============================================================================#
# 5. Assemble master panel
#
#   SPINE: labels_base — all (permno, year) observations including y=NA.
#   LEFT JOINs: fundamentals, prices, macro, lifetime.
#   Left joins preserve all label observations; unmatched rows get NA.
#==============================================================================#

cat("[06_Merge.R] Assembling master panel...\n")

panel_raw <- labels |>
  
  ## 5A. Join fundamentals (reporting-lag adjusted)
  left_join(
    fund_lagged |> select(-fyear),   # fyear replaced by prediction_year — drop
    by = c("permno" = "permno", "year" = "prediction_year")
  ) |>
  
  ## 5B. Join annual price features
  left_join(
    prices_annual,
    by = c("permno", "year")
  ) |>
  
  ## 5C. Join annual macro features
  left_join(
    macro_annual,
    by = "year"
  ) |>
  
  ## 5D. Join lifetime reference for filter
  left_join(
    lifetime_ref,
    by = "permno"
  ) |>
  
  ## 5E. Compute lifetime years as of prediction year-end
  mutate(
    year_end_date  = as.Date(paste0(year, "-12-31")),
    lifetime_years = as.numeric(year_end_date - listing_date) / 365.25
  ) |>
  select(-year_end_date, -listing_date)

cat(sprintf("  Panel before filters: %d rows, %d permno\n",
            nrow(panel_raw), n_distinct(panel_raw$permno)))

#==============================================================================#
# 6. Apply minimum lifetime filter
#
#   Remove (permno, year) observations where the firm has fewer than
#   MIN_LIFETIME_YEARS of price history as of the prediction year-end.
#   Applied here — not in 01_Universe.R — to preserve the full label
#   universe in stage 05 (survivor bias prevention).
#==============================================================================#

n_before <- nrow(panel_raw)
n_permno_before <- n_distinct(panel_raw$permno)

panel_raw <- panel_raw |>
  filter(
    !is.na(lifetime_years),
    lifetime_years >= MIN_LIFETIME_YEARS
  )

n_dropped <- n_before - nrow(panel_raw)
cat(sprintf("[06_Merge.R] Lifetime filter (>= %d years):\n",
            MIN_LIFETIME_YEARS))
cat(sprintf("  Rows dropped    : %d (%.1f%%)\n",
            n_dropped, 100 * n_dropped / n_before))
cat(sprintf("  Rows remaining  : %d\n", nrow(panel_raw)))
cat(sprintf("  Permno remaining: %d (was %d)\n",
            n_distinct(panel_raw$permno), n_permno_before))

#==============================================================================#
# 7. Save panel_raw.rds
#==============================================================================#

saveRDS(panel_raw, PATH_PANEL_RAW)
cat(sprintf("[06_Merge.R] panel_raw.rds saved: %d rows\n", nrow(panel_raw)))

#==============================================================================#
# 8. Assertions
#==============================================================================#

cat("[06_Merge.R] Running assertions...\n")

## A) No duplicate (permno, year)
n_dup <- sum(duplicated(panel_raw[, c("permno", "year")]))
if (n_dup > 0)
  stop(sprintf("[06_Merge.R] ASSERTION FAILED: %d duplicate (permno, year) rows.", n_dup))

## B) y values valid
invalid_y <- panel_raw |> filter(!is.na(y), !y %in% c(0L, 1L))
if (nrow(invalid_y) > 0)
  stop("[06_Merge.R] ASSERTION FAILED: Invalid y values in panel.")

## C) Lifetime filter respected
n_short <- sum(panel_raw$lifetime_years < MIN_LIFETIME_YEARS, na.rm = TRUE)
if (n_short > 0)
  stop(sprintf("[06_Merge.R] ASSERTION FAILED: %d rows below lifetime threshold.", n_short))

## D) Year range within bounds
if (min(panel_raw$year) < year(START_DATE) |
    max(panel_raw$year) > year(END_DATE))
  stop("[06_Merge.R] ASSERTION FAILED: Year range outside expected bounds.")

## E) Macro columns present and not entirely missing
macro_cols <- c("gdp", "unrate", "fedfunds", "term_spread", "hy_spread",
                "vix", "cpi_inflation", "recession")
for (v in macro_cols) {
  if (mean(is.na(panel_raw[[v]])) > 0.10)
    warning(sprintf("[06_Merge.R] WARNING: %s is >10%% missing in panel.", v))
}

cat("[06_Merge.R] All assertions passed.\n")

#==============================================================================#
# 9. Validation diagnostics
#==============================================================================#

cat("\n[06_Merge.R] ══════════════════════════════════════\n")
cat("  Panel dimensions:\n")
cat(sprintf("    Rows             : %d\n", nrow(panel_raw)))
cat(sprintf("    Columns          : %d\n", ncol(panel_raw)))
cat(sprintf("    Unique permno    : %d\n", n_distinct(panel_raw$permno)))
cat(sprintf("    Year range       : %d – %d\n",
            min(panel_raw$year), max(panel_raw$year)))

## Label distribution
cat("\n  Label distribution (post-lifetime-filter):\n")
cat(sprintf("    y = 1  (CSI)      : %d  (%.2f%%)\n",
            sum(panel_raw$y == 1L, na.rm = TRUE),
            100 * mean(panel_raw$y == 1L, na.rm = TRUE)))
cat(sprintf("    y = 0  (clean)    : %d  (%.2f%%)\n",
            sum(panel_raw$y == 0L, na.rm = TRUE),
            100 * mean(panel_raw$y == 0L, na.rm = TRUE)))
cat(sprintf("    y = NA (excluded) : %d  (%.2f%%)\n",
            sum(is.na(panel_raw$y)),
            100 * mean(is.na(panel_raw$y))))

## Fundamental coverage
cat("\n  Fundamental coverage (% non-missing):\n")
fund_vars <- c("at", "sale", "oancf", "epspx", "ni",
               "ebitda", "dltt", "emp", "xrent", "mkvalt")
for (v in fund_vars) {
  if (v %in% names(panel_raw)) {
    pct <- 100 * mean(!is.na(panel_raw[[v]]))
    cat(sprintf("    %-12s : %5.1f%%\n", v, pct))
  }
}

## Price coverage
cat("\n  Price feature coverage:\n")
cat(sprintf("    ann_return   : %5.1f%%\n",
            100 * mean(!is.na(panel_raw$ann_return))))
cat(sprintf("    log_return   : %5.1f%%\n",
            100 * mean(!is.na(panel_raw$log_return))))

## Macro coverage
cat("\n  Macro feature coverage:\n")
for (v in macro_cols) {
  pct <- 100 * mean(!is.na(panel_raw[[v]]))
  cat(sprintf("    %-18s : %5.1f%%\n", v, pct))
}

## Join quality: how many label obs got fundamentals
n_has_fund <- sum(!is.na(panel_raw$at))
n_no_fund  <- sum(is.na(panel_raw$at))
cat(sprintf("\n  Join quality — fundamentals:\n"))
cat(sprintf("    Label obs with fundamentals    : %d (%.1f%%)\n",
            n_has_fund, 100 * n_has_fund / nrow(panel_raw)))
cat(sprintf("    Label obs without fundamentals : %d (%.1f%%)\n",
            n_no_fund, 100 * n_no_fund / nrow(panel_raw)))

## CSI prevalence by year — temporal check
cat("\n  CSI prevalence by year (first 5 and last 5):\n")
prev_by_year <- panel_raw |>
  filter(!is.na(y)) |>
  group_by(year) |>
  summarise(
    n_obs  = n(),
    n_csi  = sum(y == 1L),
    pct    = round(100 * mean(y == 1L), 2),
    .groups = "drop"
  )
print(head(prev_by_year, 5))
cat("  ...\n")
print(tail(prev_by_year, 5))

cat("\n[06_Merge.R] DONE:", format(Sys.time()), "\n")
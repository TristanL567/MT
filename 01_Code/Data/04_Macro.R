#==============================================================================#
#==== 04_Macro.R ==============================================================#
#==== FRED Macroeconomic Variable Download & Monthly Alignment ================#
#==============================================================================#
#
# PURPOSE:
#   Download macroeconomic variables from FRED via the fredr package.
#   Aligns all series to a common monthly frequency via forward-filling,
#   producing a clean (date, variable) panel ready for 06_Feature_Eng.R.
#
# INPUTS:
#   - config.R                      : START_DATE, END_DATE, PATH_* constants
#   - FRED API key                  : set via fredr_set_key() in 00_Master.R
#                                     or hardcoded here for standalone runs
#
# OUTPUTS:
#   - Raw/macro_raw.rds             : list of raw FRED series, as-downloaded
#   - Processed/macro_monthly.rds   : all series forward-filled to monthly,
#                                     wide format, one row per month-end date
#
#   Output schema (macro_monthly.rds):
#     date (month-end), gdp, gdp_growth, unrate, fedfunds,
#     term_spread, hy_spread, vix, cpi, cpi_inflation,
#     indpro, indpro_growth, recession
#
# SERIES PULLED AND RATIONALE:
#
#   GDP          Gross Domestic Product (quarterly)
#                Paper feature: GDP. Levels + YoY growth rate computed here.
#
#   UNRATE       Civilian Unemployment Rate (monthly)
#                Paper feature: Unemployment_Rate. Level used directly.
#
#   FEDFUNDS     Federal Funds Effective Rate (monthly)
#                Interest rate level — interacts with leverage features.
#                High rates increase debt service burden for leveraged firms.
#
#   GS10         10-Year Treasury Constant Maturity Rate (monthly)
#                Used with FEDFUNDS to compute term spread (GS10 − FEDFUNDS).
#                Term spread is a leading indicator of recession and credit stress.
#
#   BAMLH0A0HYM2 ICE BofA US High Yield OAS (daily → monthly)
#                Credit spread proxy — widens before CSI clusters.
#                One of the strongest macro predictors of distress.
#
#   VIXCLS       CBOE Volatility Index (daily → monthly)
#                Market fear gauge — elevated VIX precedes implosion clusters.
#                Not in paper but standard in distress prediction literature.
#
#   CPIAUCSL     Consumer Price Index, All Urban Consumers (monthly)
#                Used to compute YoY inflation rate. High inflation erodes
#                real earnings and increases input costs for zombie firms.
#
#   INDPRO       Industrial Production Index (monthly)
#                Business cycle proxy — complements GDP at higher frequency.
#                YoY growth rate computed here.
#
#   USREC        NBER Recession Indicator (monthly, 0/1)
#                Binary recession flag. Useful as a regime variable in
#                feature engineering and for robustness analysis.
#
# DESIGN DECISIONS:
#
#   [1] Forward-fill to monthly:
#       All series are aligned to the last calendar day of each month.
#       Lower-frequency series (quarterly GDP) and daily series (VIX, HY spread)
#       are forward-filled using the most recent available observation.
#       This is the standard no-look-ahead approach for macro alignment:
#       at any month-end, we use only information available at that date.
#
#   [2] Growth rates computed here, not in 06_Feature_Eng.R:
#       GDP growth (YoY), CPI inflation (YoY), and INDPRO growth (YoY) are
#       computed in this script from the raw levels. This keeps 06_Feature_Eng.R
#       clean — it receives ready-to-use macro features, not raw levels that
#       require transformation.
#
#   [3] Term spread computed here:
#       term_spread = GS10 − FEDFUNDS. Negative term spread (yield curve
#       inversion) historically precedes recessions and credit distress.
#       Computed here so 06_Feature_Eng.R treats it as a single feature.
#
#   [4] Raw series saved as list before any transformation:
#       macro_raw.rds contains the unmodified fredr() output for each series.
#       Immutable reference — if transformation logic changes, re-run from
#       this file without re-hitting the FRED API.
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
#
#    Key is set here for standalone runs of this script.
#    In full pipeline runs via 00_Master.R, fredr_set_key() is called once
#    before this script is sourced.
#==============================================================================#

FRED_API_KEY <- "ea754f3d4a236f5d4b2c8957fae36c4a"
fredr_set_key(FRED_API_KEY)

## Verify connection
tryCatch({
  fredr_series("GDP")
  cat("[04_Macro.R] FRED API connection verified.\n")
}, error = function(e) {
  stop("[04_Macro.R] FRED API connection failed. Check API key and internet access.")
})

#==============================================================================#
# 1. Define series to pull
#
#    Each entry: list(id, short_name, frequency_note)
#    Pulling from START_DATE - 2 years to ensure YoY growth rates are
#    available from the first analysis month.
#==============================================================================#

PULL_START <- START_DATE - years(2)   # Extra history for YoY computation

FRED_SERIES <- list(
  list(id = "GDP",          name = "gdp",       note = "Quarterly — forward-filled to monthly"),
  list(id = "UNRATE",       name = "unrate",    note = "Monthly"),
  list(id = "FEDFUNDS",     name = "fedfunds",  note = "Monthly"),
  list(id = "GS10",         name = "gs10",      note = "Monthly"),
  list(id = "BAMLH0A0HYM2", name = "hy_spread", note = "Daily — averaged to monthly"),
  list(id = "VIXCLS",       name = "vix",       note = "Daily — averaged to monthly"),
  list(id = "CPIAUCSL",     name = "cpi",       note = "Monthly"),
  list(id = "INDPRO",       name = "indpro",    note = "Monthly"),
  list(id = "USREC",        name = "recession", note = "Monthly — NBER recession indicator")
)

#==============================================================================#
# 2. Download all series from FRED
#
#    Each series pulled individually with error handling.
#    If a single series fails, pipeline warns but continues — macro data
#    is supplementary; a missing series should not halt the pipeline.
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
    warning(sprintf("\n[04_Macro.R] WARNING: Failed to pull %s: %s", s$id, e$message))
    NULL
  })
  
  if (!is.null(result)) {
    macro_raw[[s$name]] <- result
    cat(sprintf(" %d observations\n", nrow(result)))
  } else {
    cat(" FAILED — skipped\n")
  }
}

## Save raw — immutable reference before any transformation
saveRDS(macro_raw, PATH_MACRO_RAW)
cat("[04_Macro.R] Raw series saved to PATH_MACRO_RAW\n")

#==============================================================================#
# 3. Build monthly date spine
#
#    All series will be aligned to this spine via forward-fill.
#    Using last day of each month as the reference date — consistent with
#    prices_monthly which uses month-end dates from crsp.msf.
#==============================================================================#

date_spine <- tibble(
  date = seq(
    from = floor_date(PULL_START,  "month"),
    to   = floor_date(END_DATE,    "month"),
    by   = "month"
  )
) |>
  mutate(date = ceiling_date(date, "month") - days(1))   # Last day of month

cat("[04_Macro.R] Monthly date spine:", nrow(date_spine),
    "months from", format(min(date_spine$date)),
    "to", format(max(date_spine$date)), "\n")

#==============================================================================#
# 4. Align each series to monthly frequency
#
#    Strategy by native frequency:
#
#    QUARTERLY (GDP):
#      Each quarter's value is assigned to the last day of the quarter.
#      Forward-filled to monthly — Q1 value persists through Jan/Feb/Mar.
#      This is the standard approach: at any month-end you know the most
#      recently released quarterly GDP figure.
#
#    DAILY (VIX, HY spread):
#      Averaged within each month to a single monthly value.
#      Monthly average is more stable than a single end-of-month observation
#      and better captures the prevailing macro regime during the month.
#
#    MONTHLY (all others):
#      Joined directly to the date spine on exact date match,
#      then forward-filled for any gaps (e.g. holidays, missing releases).
#==============================================================================#

fn_align_to_monthly <- function(series_df, series_name, date_spine) {
  
  col_name <- setdiff(names(series_df), "date")
  
  ## Determine native frequency from observation spacing
  date_gaps <- as.integer(diff(sort(series_df$date)))
  median_gap <- median(date_gaps, na.rm = TRUE)
  
  if (median_gap <= 7) {
    ## DAILY: average within each month
    aligned <- series_df |>
      mutate(date = ceiling_date(floor_date(date, "month"), "month") - days(1)) |>
      group_by(date) |>
      summarise(!!col_name := mean(.data[[col_name]], na.rm = TRUE), .groups = "drop")
    
  } else if (median_gap >= 80) {
    ## QUARTERLY: assign to month-end, then forward-fill
    aligned <- series_df |>
      mutate(date = ceiling_date(date + months(2), "month") - days(1)) |>
      ## Assign to last month of quarter (e.g. Q1 Jan → Mar 31)
      select(date, !!col_name)
    
  } else {
    ## MONTHLY: align to month-end directly
    aligned <- series_df |>
      mutate(date = ceiling_date(floor_date(date, "month"), "month") - days(1)) |>
      group_by(date) |>
      summarise(!!col_name := mean(.data[[col_name]], na.rm = TRUE), .groups = "drop")
  }
  
  ## Join to spine and forward-fill
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
#    All growth rates are YoY (12-month lag) to avoid seasonality.
#    Lag is applied to the already-monthly-aligned series.
#==============================================================================#

cat("[04_Macro.R] Computing derived features...\n")

macro_monthly <- macro_aligned |>
  arrange(date) |>
  mutate(
    ## GDP: YoY growth rate (quarterly data forward-filled to monthly)
    gdp_growth    = (gdp    / lag(gdp,    12) - 1) * 100,
    
    ## CPI: YoY inflation rate
    cpi_inflation = (cpi    / lag(cpi,    12) - 1) * 100,
    
    ## Industrial production: YoY growth rate
    indpro_growth = (indpro / lag(indpro, 12) - 1) * 100,
    
    ## Term spread: 10Y Treasury minus Fed Funds Rate
    ## Negative = yield curve inversion = recession leading indicator
    term_spread   = gs10 - fedfunds,
    
    ## Recession flag: ensure integer (FRED returns numeric 0/1)
    recession     = as.integer(recession)
  ) |>
  ## Restrict to analysis window after growth rate computation
  ## (keep PULL_START buffer rows for lag, then trim)
  filter(date >= ceiling_date(floor_date(START_DATE, "month"), "month") - days(1))

cat("[04_Macro.R] Derived features computed.\n")

#==============================================================================#
# 7. Assertions
#==============================================================================#

cat("[04_Macro.R] Running assertions...\n")

## A) No duplicate dates
n_dup <- sum(duplicated(macro_monthly$date))
if (n_dup > 0)
  stop(sprintf("[04_Macro.R] ASSERTION FAILED: %d duplicate dates.", n_dup))

## B) Date range covers full analysis window
if (min(macro_monthly$date) > as.Date("1998-01-31"))
  stop("[04_Macro.R] ASSERTION FAILED: macro_monthly starts after START_DATE.")
if (max(macro_monthly$date) < as.Date("2024-11-30"))
  stop("[04_Macro.R] ASSERTION FAILED: macro_monthly ends before END_DATE.")

## C) Core series not entirely missing
core_vars <- c("gdp", "unrate", "fedfunds", "cpi", "hy_spread")
for (v in core_vars) {
  if (v %in% names(macro_monthly)) {
    pct_na <- mean(is.na(macro_monthly[[v]]))
    if (pct_na > 0.05)
      warning(sprintf("[04_Macro.R] WARNING: %s is %.1f%% missing after fill.",
                      v, 100 * pct_na))
  }
}

## D) Plausible row count
expected_months <- interval(START_DATE, END_DATE) %/% months(1) + 1L
if (nrow(macro_monthly) < expected_months - 2L)
  stop(sprintf("[04_Macro.R] ASSERTION FAILED: Only %d rows, expected ~%d.",
               nrow(macro_monthly), expected_months))

cat("[04_Macro.R] All assertions passed.\n")

#==============================================================================#
# 8. Save processed macro panel
#==============================================================================#

saveRDS(macro_monthly, PATH_MACRO_MONTHLY)
cat("[04_Macro.R] Monthly macro panel saved:", nrow(macro_monthly), "rows\n")

#==============================================================================#
# 9. Summary diagnostics
#==============================================================================#

cat("\n[04_Macro.R] ══════════════════════════════════════\n")
cat("  Rows              :", nrow(macro_monthly), "\n")
cat("  Date range        :", format(min(macro_monthly$date)),
    "to", format(max(macro_monthly$date)), "\n")
cat("  Columns           :", ncol(macro_monthly), "\n")

cat("\n  Variable coverage (% non-missing):\n")
all_vars <- setdiff(names(macro_monthly), "date")
for (v in all_vars) {
  pct <- 100 * mean(!is.na(macro_monthly[[v]]))
  cat(sprintf("    %-18s : %5.1f%%\n", v, pct))
}

cat("\n  Macro snapshot — most recent month:\n")
print(tail(macro_monthly, 1))

cat("\n[04_Macro.R] DONE:", format(Sys.time()), "\n")


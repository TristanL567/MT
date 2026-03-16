#==============================================================================#
#==== 00 - Description ========================================================#
#==============================================================================#

#==============================================================================#
#==== 02_Prices.R =============================================================# 
#==== CRSP Price & Return Download ============================================#
#==============================================================================#
#
# PURPOSE:
#   Download and clean CRSP price/return data for all permno in universe.rds.
#   Produces two outputs at different frequencies serving different pipeline
#   stages downstream:
#
#   WEEKLY  (crsp.dsf → aggregated)  → used by 05_CSI_Label.R
#     Drawdown detection requires weekly granularity. A monthly close can
#     straddle the -80% threshold entirely, missing the crash onset.
#
#   MONTHLY (crsp.msf)               → used by 06_Feature_Engineering.R
#     Sufficient for feature construction since Compustat fundamentals
#     are quarterly. Lighter in memory than daily/weekly.
#
# INPUTS:
#   - wrds                           : WRDS connection (from 00_Master.R)
#   - config.R                       : START_DATE, END_DATE, DIR_CRSP_*
#   - Processed/universe.rds         : target permno list
#
# OUTPUTS:
#   - Raw/prices_daily_raw.rds       : as-downloaded daily, before any cleaning
#   - Raw/prices_monthly_raw.rds     : as-downloaded monthly, before any cleaning
#   - Processed/prices_weekly.rds    : cleaned, delisting-adjusted, weekly
#   - Processed/prices_monthly.rds   : cleaned, delisting-adjusted, monthly
#
#   Weekly schema:
#     permno, week_end, price, ret_adj, ret_excl_div,
#     mktcap, vol, shrout, dlret_applied
#
#   Monthly schema:
#     permno, date, price, ret_adj, ret_excl_div,
#     div_amount, mktcap, vol, shrout, dlret_applied
#
# CRITICAL DESIGN DECISIONS:
#   1. Delisting return (dlret) adjustment:
#      CRSP msf/dsf ret is NA or stale at delisting. We substitute dlret
#      from msedelist for the terminal observation. Without this, imploded
#      stocks have understated terminal returns, corrupting CSI labels.
#
#   2. Sentinel value filter:
#      CRSP uses -66, -77, -88, -99 as sentinel codes in ret fields.
#      These are NOT valid returns and must be removed before any
#      cumulative return or drawdown computation.
#
#   3. Price sign:
#      Negative prc in CRSP = price is a bid-ask midpoint, not a trade price.
#      abs(prc) is always the correct close price.
#
#   4. permno filter pushed to DB:
#      We filter by target_permnos before collect() to avoid pulling the
#      full CRSP universe (~100M+ rows for daily) into R memory.
#
#==============================================================================#

source("config.R")

cat("\n[02_Prices.R] START:", format(Sys.time()), "\n")

#==============================================================================#
# 0. Load universe — defines target permno set
#==============================================================================#

universe <- readRDS(PATH_UNIVERSE)
target_permnos <- universe$permno
cat("[02_Prices.R] Target universe:", length(target_permnos), "permno\n")

#==============================================================================#
# 1. Connect to CRSP tables (lazy)
#==============================================================================#

dsf_db       <- tbl(wrds, I("crsp_a_stock.dsf"))
msf_db       <- tbl(wrds, I("crsp_a_stock.msf"))
msedelist_db <- tbl(wrds, I("crsp_a_stock.msedelist"))

#==============================================================================#
# 2. Download delisting returns
#
#    RATIONALE:
#    msedelist contains the actual return on the delisting date (dlret).
#    This is the authoritative terminal return — CRSP's msf/dsf ret field
#    is unreliable or missing for the final observation of delisted stocks.
#
#    dlstcd codes for reference:
#      100       = still active (no delisting)
#      200-299   = mergers / acquisitions (often positive dlret)
#      400-499   = liquidations (typically large negative dlret)
#      500-599   = dropped by exchange (performance-related — critical for CSI)
#      volunt.   = voluntary delistings
#
#    We pull dlret and dlstcd for all target permno regardless of dlstcd,
#    then apply at the terminal date of each stock's price series.
#==============================================================================#

delisting_full <- msedelist_db |>
  select(
    permno,
    dlstdt,   # Delisting date
    dlstcd,   # Delisting code
    dlret     # Delisting return (the actual return on final trading day)
  ) |>
  collect() |>
  mutate(dlstdt = as.Date(dlstdt))

delisting <- delisting_full |>
  filter(permno %in% target_permnos)

cat("[02_Prices.R] Delisting records pulled:", nrow(delisting), "\n")
cat("[02_Prices.R] Non-active records (dlstcd != 100):",
    sum(delisting$dlstcd != 100, na.rm = TRUE), "\n")

#==============================================================================#
# 3. Daily prices — download, save raw, aggregate to weekly
#==============================================================================#

##---------------------------------------------------------------------------##
## 3A. Download daily (filtered to target permno before collect)
##---------------------------------------------------------------------------##

cat("[02_Prices.R] Pulling daily prices from crsp.dsf...\n")

prices_daily_raw <- dsf_db |>
  filter(
    permno %in% target_permnos,
    date   >= START_DATE,
    date   <= END_DATE
  ) |>
  select(
    permno,
    date,
    prc,     # Price (negative = bid-ask midpoint; always use abs)
    ret,     # Daily total return including dividends
    retx,    # Daily return excluding dividends
    shrout,  # Shares outstanding (thousands)
    vol      # Daily volume (shares)
  ) |>
  collect() |>
  mutate(date = as.Date(date))

## Save raw — immutable reference
saveRDS(prices_daily_raw,   PATH_PRICES_DAILY_RAW)
cat("[02_Prices.R] Daily raw saved:", nrow(prices_daily_raw), "rows\n")

##---------------------------------------------------------------------------##
## 3B. Clean daily
##
##   Steps:
##   (i)  abs(prc)           — remove sign convention artifact
##   (ii) Sentinel removal   — CRSP uses -66/-77/-88/-99 as missing codes
##                             in ret; these are NOT valid returns
##   (iii) Delisting join    — substitute dlret at terminal date
##---------------------------------------------------------------------------##

CRSP_SENTINELS <- c(-66, -77, -88, -99)

prices_daily_clean <- prices_daily_raw |>
  mutate(
    price = abs(prc),
    ym    = floor_date(date, "month"),    # ADD THIS — needed for join
    ret   = if_else(ret  %in% CRSP_SENTINELS, NA_real_, ret),
    retx  = if_else(retx %in% CRSP_SENTINELS, NA_real_, retx)
  ) |>
  left_join(
    delisting |>
      filter(dlstcd != 100) |>
      mutate(ym = floor_date(dlstdt, "month")) |>
      select(permno, ym, dlret, dlstcd),
    by = c("permno", "ym")
  ) |>
  mutate(
    dlret_applied = !is.na(dlret),
    ret_adj       = case_when(
      !is.na(dlret) ~ dlret,
      TRUE          ~ ret
    ),
    ret_excl_div  = case_when(
      !is.na(dlret) ~ dlret,
      TRUE          ~ retx
    ),
    mktcap = price * shrout
  ) |>
  filter(!is.na(price), price > 0) |>
  select(
    permno, date,
    price, ret_adj, ret_excl_div,
    mktcap, vol, shrout,
    dlstcd, dlret_applied
  ) |>
  arrange(permno, date)

##---------------------------------------------------------------------------##
## 3C. Aggregate daily → weekly
##
##   Convention: week ending Friday (label = last trading day of the week).
##   Using floor_date(..., "week", week_start = 1) gives Monday of each week;
##   we label by the last observed trading date within the week instead —
##   this avoids forward-filling into non-trading days.
##
##   Return aggregation: compound daily returns within each week.
##     Weekly ret = prod(1 + daily_ret) - 1
##     This is exact; simple summation would introduce approximation error
##     that accumulates meaningfully over a 5-year rolling window.
##
##   Price: last observed price in the week (closing price for label purposes).
##   Volume: sum of daily volume within the week.
##   Shrout: last observed (most recent shares outstanding).
##   dlret_applied: TRUE if any day in the week had a delisting return applied.
##---------------------------------------------------------------------------##

prices_weekly <- prices_daily_clean |>
  mutate(
    week_end = floor_date(date, "week", week_start = 1) + days(6)
    ## floor_date to Monday + 6 days = Sunday label for the week.
    ## Actual last trading day may be Thursday or Friday — we use week_end
    ## as a consistent weekly time index, not an actual trading date.
  ) |>
  group_by(permno, week_end) |>
  summarise(
    price        = last(price),
    ret_adj      = if_else(
      all(is.na(ret_adj)),
      NA_real_,
      prod(1 + ret_adj, na.rm = TRUE) - 1
    ),
    ret_excl_div = if_else(
      all(is.na(ret_excl_div)),
      NA_real_,
      prod(1 + ret_excl_div, na.rm = TRUE) - 1
    ),
    mktcap        = last(mktcap),
    vol           = sum(vol, na.rm = TRUE),
    shrout        = last(shrout),
    dlret_applied = any(dlret_applied, na.rm = TRUE),
    .groups       = "drop"
  ) |>
  filter(week_end <= END_DATE + days(7)) |>    # Guard against post-END_DATE weeks
  arrange(permno, week_end)

saveRDS(prices_weekly, PATH_PRICES_WEEKLY)
cat("[02_Prices.R] Weekly prices saved:", nrow(prices_weekly), "rows,",
    n_distinct(prices_weekly$permno), "permno\n")

#==============================================================================#
# 4. Monthly prices — download, clean, save
#
#    Separate pull from crsp.msf (not aggregated from daily).
#    Monthly CRSP ret already reflects corporate actions and dividends.
#    We still apply the delisting adjustment for terminal months.
#==============================================================================#

cat("[02_Prices.R] Pulling monthly prices from crsp.msf...\n")

## 4A. Download — pull filtered to target permno before collect()
##     Save raw immediately as immutable snapshot — do NOT modify this object
prices_monthly_raw <- msf_db |>
  filter(
    permno %in% target_permnos,
    date   >= START_DATE,
    date   <= END_DATE
  ) |>
  select(permno, date, prc, ret, retx, shrout, vol) |>
  collect() |>
  mutate(date = as.Date(date))

saveRDS(prices_monthly_raw, PATH_PRICES_MONTHLY_RAW)
cat("[02_Prices.R] Monthly raw saved:", nrow(prices_monthly_raw), "rows\n")

## 4B. Clean monthly — sentinel removal, delisting join, derived variables
##     prices_monthly_raw is never modified — all cleaning on a new object
prices_monthly <- prices_monthly_raw |>
  mutate(
    price = abs(prc),
    ym    = floor_date(date, "month"),    # Required for delisting join
    ret   = if_else(ret  %in% CRSP_SENTINELS, NA_real_, ret),
    retx  = if_else(retx %in% CRSP_SENTINELS, NA_real_, retx)
  ) |>
  left_join(
    delisting |>
      filter(dlstcd != 100) |>           # Exclude active stocks — no dlret
      mutate(ym = floor_date(dlstdt, "month")) |>
      select(permno, ym, dlret, dlstcd),
    by = c("permno", "ym")               # Year-month join — dlstdt is mid-month
  ) |>
  mutate(
    dlret_applied = !is.na(dlret),
    ret_adj       = case_when(
      !is.na(dlret) ~ dlret,
      TRUE          ~ ret
    ),
    ret_excl_div  = case_when(
      !is.na(dlret) ~ dlret,             # retx not separately available at delisting
      TRUE          ~ retx
    )
  ) |>
  arrange(permno, date) |>
  group_by(permno) |>
  mutate(
    prev_price = lag(price),
    div_amount = (ret_adj - ret_excl_div) * prev_price,
    mktcap     = price * shrout
  ) |>
  ungroup() |>
  filter(!is.na(price), price > 0) |>
  select(
    permno, date,
    price, ret_adj, ret_excl_div,
    div_amount, mktcap, vol, shrout,
    dlstcd, dlret_applied
  ) |>
  arrange(permno, date)

saveRDS(prices_monthly,     PATH_PRICES_MONTHLY)
cat("[02_Prices.R] Monthly prices saved:", nrow(prices_monthly), "rows,",
    n_distinct(prices_monthly$permno), "permno\n")

#==============================================================================#
# 5. Assertions
#==============================================================================#

## A) No permno in output that wasn't in universe
orphan_weekly <- setdiff(prices_weekly$permno,  target_permnos)
orphan_monthly <- setdiff(prices_monthly$permno, target_permnos)

if (length(orphan_weekly) > 0)
  stop(sprintf("[02_Prices.R] ASSERTION FAILED: %d orphan permno in weekly output.",
               length(orphan_weekly)))

if (length(orphan_monthly) > 0)
  stop(sprintf("[02_Prices.R] ASSERTION FAILED: %d orphan permno in monthly output.",
               length(orphan_monthly)))

## B) No zero or negative prices
n_bad_price_w <- sum(prices_weekly$price  <= 0, na.rm = TRUE)
n_bad_price_m <- sum(prices_monthly$price <= 0, na.rm = TRUE)

if (n_bad_price_w > 0)
  stop(sprintf("[02_Prices.R] ASSERTION FAILED: %d non-positive prices in weekly output.",
               n_bad_price_w))
if (n_bad_price_m > 0)
  stop(sprintf("[02_Prices.R] ASSERTION FAILED: %d non-positive prices in monthly output.",
               n_bad_price_m))

## C) No sentinel values remaining in ret_adj
n_sentinel_w <- sum(prices_weekly$ret_adj  %in% CRSP_SENTINELS, na.rm = TRUE)
n_sentinel_m <- sum(prices_monthly$ret_adj %in% CRSP_SENTINELS, na.rm = TRUE)

if (n_sentinel_w > 0 | n_sentinel_m > 0)
  stop("[02_Prices.R] ASSERTION FAILED: Sentinel values remain in ret_adj.")

## D) Date range respected
if (min(prices_weekly$week_end)  < START_DATE |
    min(prices_monthly$date)     < START_DATE)
  warning("[02_Prices.R] WARNING: Observations before START_DATE detected.")

## E) Delisting coverage — informational, not a hard stop
n_dlret <- sum(!is.na(delisting$dlret))
cat("[02_Prices.R] Delisting records with valid dlret:", n_dlret,
    "of", nrow(delisting), "\n")

cat("[02_Prices.R] All assertions passed.\n")

##
saveRDS(delisting_full,     PATH_DELISTING)   # Also save delisting as raw reference

#==============================================================================#
# 6. Summary diagnostics
#==============================================================================#

cat("\n[02_Prices.R] Weekly output summary:\n")
cat("  Rows         :", nrow(prices_weekly), "\n")
cat("  Permno       :", n_distinct(prices_weekly$permno), "\n")
cat("  Date range   :", format(min(prices_weekly$week_end)),
    "to", format(max(prices_weekly$week_end)), "\n")
cat("  dlret applied:", sum(prices_weekly$dlret_applied), "weeks\n")

cat("\n[02_Prices.R] Monthly output summary:\n")
cat("  Rows         :", nrow(prices_monthly), "\n")
cat("  Permno       :", n_distinct(prices_monthly$permno), "\n")
cat("  Date range   :", format(min(prices_monthly$date)),
    "to", format(max(prices_monthly$date)), "\n")
cat("  dlret applied:", sum(prices_monthly$dlret_applied), "months\n")

cat("\n[02_Prices.R] DONE:", format(Sys.time()), "\n")

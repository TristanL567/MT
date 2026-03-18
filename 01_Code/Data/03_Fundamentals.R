#==============================================================================#
#==== 03_Fundamentals.R =======================================================#
#==== Compustat Fundamentals Download & CCM Merge =============================#
#==============================================================================#
#
# PURPOSE:
#   Download annual accounting fundamentals from Compustat (comp.funda) and
#   link them to CRSP permno via the CCM link table (crsp_a_ccm.ccmxpf_linktable).
#   Produces a permno-keyed fundamentals panel ready for feature engineering.
#
# INPUTS:
#   - wrds                          : WRDS connection (from 00_Master.R)
#   - config.R                      : START_DATE, END_DATE, PATH_* constants
#   - Processed/universe.rds        : target permno list
#
# OUTPUTS:
#   - Raw/fundamentals_raw.rds      : as-downloaded from comp.funda, unmodified
#   - Raw/ccm_link_raw.rds          : as-downloaded CCM link table, unmodified
#   - Processed/fundamentals.rds    : CCM-merged, deduplicated, permno-keyed
#
#   Output schema (fundamentals.rds):
#     permno, gvkey, datadate, fyear,
#     [~60 raw Compustat variables — see Variable Reference xlsx]
#
# CRITICAL DESIGN DECISIONS:
#
#   [1] Standard Compustat deduplication filter:
#       indfmt="INDL", datafmt="STD", popsrc="D", consol="C"
#       Without this, comp.funda returns multiple rows per (gvkey, fyear)
#       from different reporting formats (as-reported, restated, international).
#       This is a hard requirement — never omit these four filters.
#
#   [2] Point-in-time CCM merge:
#       Each (gvkey, datadate) row is matched to a permno only if
#       datadate falls within [linkdt, linkenddt] of a valid link.
#       A simple join on gvkey alone introduces look-ahead bias for firms
#       that changed their CRSP entity over time.
#
#   [3] Link quality hierarchy:
#       linktype filtered to LC > LU > LS (LC = best researched link).
#       linkprim filtered to P/C (primary links only).
#       Remaining duplicates resolved by link priority then link duration.
#
#   [4] fyear pull starts at START_DATE - 1 year (1997):
#       Feature engineering requires 1-year lagged values for growth rates
#       (e.g. ff_emp_gr, ff_mkt_val_gr, ff_cf_ps_gr). Pulling one extra year
#       ensures the first analysis year (1998) has a valid lag available.
#
#   [5] Reporting lag deferred to 06_Feature_Eng.R:
#       datadate is the fiscal period END, not the filing date. A ~3-month
#       reporting lag must be applied before joining to the label panel to
#       prevent look-ahead bias. This is handled in 06_Feature_Eng.R where
#       the feature matrix is assembled. The raw pull preserves datadate as-is.
#
#   [6] Foreign sales (ff_for_sales_pct) NOT included:
#       Requires comp.segments — a separate geographic segment table.
#       Omitted from this pull as a known deviation from Tewari et al.
#       flagged in the thesis.
#
#   [7] Float shares (ff_shs_float) NOT available in comp.funda:
#       Paper sourced from FactSet. csho (shares outstanding) used as proxy.
#       Flagged as known deviation.
#
#==============================================================================#

source("config.R")

cat("\n[03_Fundamentals.R] START:", format(Sys.time()), "\n")

#==============================================================================#
# 0. Load universe — defines target permno set
#==============================================================================#

universe       <- readRDS(PATH_UNIVERSE)
target_permnos <- universe$permno
cat("[03_Fundamentals.R] Target universe:", length(target_permnos), "permno\n")

#==============================================================================#
# 1. Connect to Compustat tables (lazy)
#==============================================================================#

funda_db  <- tbl(wrds, I("comp.funda"))
ccm_db    <- tbl(wrds, I("crsp_a_ccm.ccmxpf_linktable"))

#==============================================================================#
# 2. Download Compustat annual fundamentals
#
#    DEDUPLICATION FILTER (mandatory — see design note [1]):
#      indfmt = "INDL" : Industrial format only
#                        Excludes banks/insurance (different balance sheet)
#      datafmt = "STD" : Standardised format only
#                        Excludes as-reported and restated duplicates
#      popsrc  = "D"   : Domestic population
#                        Excludes international cross-listings
#      consol  = "C"   : Consolidated statements only
#                        Excludes parent-only and subsidiary filings
#
#    VARIABLE LIST: ~60 raw variables covering all paper features plus
#    thesis extension variables. See compustat_variable_reference.xlsx
#    for full mapping to paper ff_* features and short_name labels.
#==============================================================================#

cat("[03_Fundamentals.R] Pulling comp.funda...\n")

## Pull start year: one year before START_DATE for lag computation (note [4])
PULL_START_YEAR <- year(START_DATE) - 1L   # 1997

fundamentals_raw <- funda_db |>
  filter(
    ## Standard deduplication — mandatory (design note [1])
    indfmt  == "INDL",
    datafmt == "STD",
    popsrc  == "D",
    consol  == "C",
    ## Date range — one extra year for lag headroom
    fyear   >= PULL_START_YEAR,
    fyear   <= year(END_DATE)
  ) |>
  select(
    ##------------------------------------------------------------------##
    ## Identifiers & metadata
    ##------------------------------------------------------------------##
    gvkey,       # Compustat company key — primary join key to CCM
    datadate,    # Fiscal period end date — used for point-in-time CCM merge
    fyear,       # Fiscal year label
    fyr,         # Fiscal year-end month (for reporting lag adjustment)
    sich,        # Historical SIC code (time-varying, preferred over sic)
    
    ##------------------------------------------------------------------##
    ## Balance sheet — assets
    ## Reference: compustat_variable_reference.xlsx, section "BS Assets"
    ##------------------------------------------------------------------##
    at,          # Total Assets                    [size normaliser]
    act,         # Current Assets — Total          [liquidity]
    che,         # Cash & Short-Term Investments   [ff_cash_curr_assets]
    ivst,        # Short-Term Investments          [ff_invest_st_tot *]
    rect,        # Receivables — Total             [channel stuffing]
    invt,        # Inventories — Total             [working capital]
    wcap,        # Working Capital                 [liquidity buffer]
    ppent,       # Net PP&E                        [asset intensity]
    intan,       # Intangibles — Total             [soft assets]
    intano,      # Other Intangibles               [ff_intang_oth *]
    gdwl,        # Goodwill                        [impairment risk]
    txdba,       # Deferred Tax Asset LT           [ff_dfd_tax_assets_lt * / NOL proxy]
    aco,         # Other Current Assets            [asset quality]
    
    ##------------------------------------------------------------------##
    ## Balance sheet — liabilities & equity
    ##------------------------------------------------------------------##
    lt,          # Total Liabilities               [leverage]
    lct,         # Current Liabilities             [short-term obligations]
    dltt,        # Long-Term Debt                  [leverage]
    dlc,         # Debt in Current Liabilities     [short-term debt]
    dd1,         # LT Debt Due in 1 Year           [refinancing wall]
    ap,          # Accounts Payable                [trade credit]
    txp,         # Income Taxes Payable            [tax obligations]
    txditc,      # Deferred Taxes & ITC            [tax shield]
    seq,         # Stockholders Equity             [book value]
    ceq,         # Common Equity — Total           [book equity]
    re,          # Retained Earnings               [accumulated deficit signal]
    pstk,        # Preferred Stock                 [capital structure]
    mib,         # Minority Interest (BS)          [ff_min_int_tcap]
    icapt,       # Invested Capital — Total        [ROIC denominator]
    
    ##------------------------------------------------------------------##
    ## Income statement
    ##------------------------------------------------------------------##
    sale,        # Sales/Turnover (Net)            [ff_cf_sales denominator]
    revt,        # Revenue — Total                 [broader revenue]
    cogs,        # Cost of Goods Sold              [gross margin]
    gp,          # Gross Profit                    [pricing power]
    xsga,        # SG&A Expense                   [operating leverage]
    xrd,         # R&D Expense                     [innovation / distress]
    dp,          # Depreciation & Amortisation     [non-cash charge]
    ebit,        # EBIT                            [ff_ebit_oper_roa]
    ebitda,      # EBITDA                          [operating profitability]
    oiadp,       # Operating Inc. After Dep.       [ff_roic numerator]
    oibdp,       # Operating Inc. Before Dep.      [alternative EBITDA]
    xopr,        # Total Operating Expenses        [cost structure]
    xint,        # Interest Expense                [ff_eff_int_rate / debt burden]
    pi,          # Pretax Income                   [pre-tax profitability]
    ni,          # Net Income                      [ff_net_inc_per_emp]
    ib,          # Income Before Extraordinary     [clean earnings]
    epspx,       # EPS Basic (excl. extraord.)     [ff_earn_yld * — paper top feature]
    dvc,         # Common Dividends                [ff_div_yld]
    dvt,         # Total Dividends                 [payout]
    citotal,     # Comprehensive Income            [ff_compr_inc *]
    
    ##------------------------------------------------------------------##
    ## Cash flow statement
    ##------------------------------------------------------------------##
    oancf,       # Operating CF                    [ff_oper_ps_net_cf * — paper top feature]
    capx,        # Capital Expenditures            [ff_reinvest_rate]
    ivncf,       # Investing CF                    [investment cycle]
    fincf,       # Financing CF                    [capital structure changes]
    dv,          # Cash Dividends Paid (CF)        [ff_cash_div_cf *]
    sstk,        # Stock Issuance                  [distress financing signal]
    prstkc,      # Stock Buyback                   [capital return capacity]
    dltis,       # LT Debt Issued                  [refinancing activity]
    dltr,        # LT Debt Repaid                  [deleveraging]
    sppe,        # Sale of PP&E                    [asset liquidation signal]
    
    ##------------------------------------------------------------------##
    ## Market & size
    ##------------------------------------------------------------------##
    csho,        # Common Shares Outstanding       [ff_shs_float proxy]
    prcc_f,      # Fiscal Year-End Price           [ff_earn_yld denominator]
    mkvalt,      # Market Value — Total Fiscal     [ff_mkt_val_gr / ff_entrpr_val_sales]
    
    ##------------------------------------------------------------------##
    ## Zombie precursors (thesis extension — beyond Tewari et al.)
    ##------------------------------------------------------------------##
    emp,         # Employees (thousands)           [ff_emp_gr * / headcount decline]
    xrent        # Rental Expense                  [fixed cost obligation]
  ) |>
  collect() |>
  mutate(datadate = as.Date(datadate))

## Save raw — immutable reference, never overwrite
saveRDS(fundamentals_raw, PATH_FUNDAMENTALS_RAW)
cat("[03_Fundamentals.R] Raw fundamentals saved:",
    nrow(fundamentals_raw), "rows,",
    n_distinct(fundamentals_raw$gvkey), "unique gvkey\n")

#==============================================================================#
# 3. Download CCM link table
#
#    linktype hierarchy (design note [3]):
#      LC = Link researched by CRSP (best quality)
#      LU = Link unresearched but used (acceptable)
#      LS = Link to S&P security (acceptable)
#      Excluded: LN, NU, NR, NC (non-primary or flagged as problematic)
#
#    linkprim:
#      P = Primary CRSP permno for this Compustat gvkey
#      C = Primary link (used when P unavailable)
#      Excluded: N, J (non-primary)
#==============================================================================#

cat("[03_Fundamentals.R] Pulling CCM link table...\n")

VALID_LINKTYPES <- c("LC", "LU", "LS")
VALID_LINKPRIMS <- c("P", "C")

## Pull full link table then filter in R
## (avoids PostgreSQL %in% performance issues with character vectors)
ccm_link_raw <- ccm_db |>
  select(
    gvkey,
    lpermno,    # CRSP permno
    lpermco,    # CRSP permco
    linktype,   # Link quality type
    linkprim,   # Primary link indicator
    liid,       # Issue identifier
    linkdt,     # Link start date
    linkenddt   # Link end date (NA = still active)
  ) |>
  collect() |>
  mutate(
    linkdt    = as.Date(linkdt),
    linkenddt = as.Date(linkenddt)
  )

## Save raw CCM — immutable reference
saveRDS(ccm_link_raw, PATH_CCM_LINK)
cat("[03_Fundamentals.R] Raw CCM link saved:",
    nrow(ccm_link_raw), "rows\n")

## Filter to valid links
ccm_link <- ccm_link_raw |>
  filter(
    linktype %in% VALID_LINKTYPES,
    linkprim %in% VALID_LINKPRIMS,
    !is.na(lpermno)
  ) |>
  ## Add numeric priority for deduplication: LC=1 (best), LU=2, LS=3
  mutate(
    link_priority = case_when(
      linktype == "LC" ~ 1L,
      linktype == "LU" ~ 2L,
      linktype == "LS" ~ 3L
    ),
    ## Duration proxy for tiebreaking: longer links are more reliable
    link_duration = as.integer(
      coalesce(linkenddt, as.Date(END_DATE)) - linkdt
    )
  )

cat("[03_Fundamentals.R] Valid CCM links after filtering:",
    nrow(ccm_link), "rows,",
    n_distinct(ccm_link$lpermno), "unique permno\n")

#==============================================================================#
# 4. Point-in-time CCM merge
#
#    For each Compustat (gvkey, datadate) row, find the permno where:
#      - gvkey matches
#      - datadate falls within [linkdt, linkenddt]
#      - linktype is LC/LU/LS and linkprim is P/C
#
#    This is a non-equi join — cannot be done with standard dplyr join.
#    We use data.table foverlaps() for efficiency on large datasets.
#
#    RESULT: Each Compustat observation gets the permno that was valid
#    at the time of the fiscal period end. Look-ahead bias prevented.
#==============================================================================#

cat("[03_Fundamentals.R] Performing point-in-time CCM merge...\n")

## Convert to data.table for non-equi join
dt_fund <- as.data.table(fundamentals_raw)
dt_link <- as.data.table(ccm_link)

## For the non-equi join, treat datadate as a single-point interval
## foverlaps requires two date columns on both tables
dt_fund[, datadate_end := datadate]   # Point-in-time: start == end

## linkenddt NA means still active — replace with far-future date for join
dt_link[, linkenddt_join := fifelse(
  is.na(linkenddt),
  as.Date("2099-12-31"),
  linkenddt
)]

## Set keys for overlap join
setkey(dt_link, gvkey, linkdt, linkenddt_join)
setkey(dt_fund, gvkey, datadate, datadate_end)

## Non-equi join: find all link records where datadate falls in [linkdt, linkenddt]
merged <- foverlaps(
  dt_fund,
  dt_link,
  by.x    = c("gvkey", "datadate",     "datadate_end"),
  by.y    = c("gvkey", "linkdt",        "linkenddt_join"),
  type    = "within",
  nomatch = NA   # Keep Compustat rows with no CCM match (will be NA permno)
)

## Clean up join helper columns
merged[, c("datadate_end", "linkenddt_join") := NULL]

cat("[03_Fundamentals.R] After merge:",
    nrow(merged), "rows (including unmatched)\n")
cat("[03_Fundamentals.R] Matched rows (permno not NA):",
    sum(!is.na(merged$lpermno)), "\n")
cat("[03_Fundamentals.R] Unmatched Compustat rows (no CCM link):",
    sum(is.na(merged$lpermno)), "\n")

#==============================================================================#
# 5. Resolve duplicates and filter to universe permno
#
#    After the point-in-time join, duplicates can still arise when:
#      (a) A gvkey has multiple valid links on the same datadate
#          (e.g. firm changed CRSP entity mid-year)
#      (b) A permno maps to multiple gvkey at the same time
#          (rare but possible with LS link type)
#
#    Resolution: within each (gvkey, datadate), keep the row with:
#      1. Lowest link_priority (LC preferred over LU over LS)
#      2. Longest link_duration as tiebreaker
#
#    Then filter to target universe permno.
#==============================================================================#

fundamentals <- merged |>
  ## Drop unmatched rows — no permno means no label to train on
  filter(!is.na(lpermno)) |>
  
  ## Resolve duplicates: best link per (gvkey, datadate)
  as.data.table() |>
  setorder(gvkey, datadate, link_priority, -link_duration) |>
  unique(by = c("gvkey", "datadate")) |>
  
  ## Rename lpermno → permno for consistency with rest of pipeline
  setnames("lpermno", "permno") |>
  
  ## Filter to universe permno — keep only stocks in our CRSP universe
  _[permno %in% target_permnos] |>
  
  ## Final deduplication: if a permno still has two gvkey for same fyear,
  ## keep the latest datadate (most complete fiscal year)
  setorder(permno, fyear, -datadate) |>
  unique(by = c("permno", "fyear")) |>
  
  ## Drop link metadata — not needed downstream
  _[, c("lpermco", "liid", "linktype", "linkprim",
        "linkdt", "linkenddt", "link_priority", "link_duration") := NULL] |>
  
  ## Final sort
  setorder(permno, datadate) |>
  as_tibble()

cat("[03_Fundamentals.R] After deduplication and universe filter:",
    nrow(fundamentals), "rows,",
    n_distinct(fundamentals$permno), "permno,",
    n_distinct(fundamentals$gvkey), "gvkey\n")

#==============================================================================#
# 6. Assertions
#==============================================================================#

cat("[03_Fundamentals.R] Running assertions...\n")

## A) No duplicate (permno, fyear) rows
n_dup <- sum(duplicated(fundamentals[, c("permno", "fyear")]))
if (n_dup > 0)
  stop(sprintf(
    "[03_Fundamentals.R] ASSERTION FAILED: %d duplicate (permno, fyear) rows.",
    n_dup
  ))

## B) All permno in universe
orphan_permno <- setdiff(fundamentals$permno, target_permnos)
if (length(orphan_permno) > 0)
  stop(sprintf(
    "[03_Fundamentals.R] ASSERTION FAILED: %d permno not in universe.",
    length(orphan_permno)
  ))

## C) Fiscal year range within expected bounds
if (min(fundamentals$fyear) > year(START_DATE) |
    max(fundamentals$fyear) < year(END_DATE) - 2L)
  warning("[03_Fundamentals.R] WARNING: Unexpected fyear range — check pull.")

## D) Plausible row count — expect >> 30,000 firm-year observations
if (nrow(fundamentals) < 20000)
  stop(sprintf(
    "[03_Fundamentals.R] ASSERTION FAILED: Only %d rows — suspiciously low.",
    nrow(fundamentals)
  ))

## E) Key variables not entirely missing
key_check_vars <- c("at", "sale", "oancf", "epspx", "ni", "emp")
for (v in key_check_vars) {
  pct_na <- mean(is.na(fundamentals[[v]]))
  if (pct_na > 0.80)
    warning(sprintf(
      "[03_Fundamentals.R] WARNING: %s is %.0f%% missing — check Compustat availability.",
      v, 100 * pct_na
    ))
}

cat("[03_Fundamentals.R] All assertions passed.\n")

#==============================================================================#
# 7. Save processed fundamentals
#==============================================================================#

saveRDS(fundamentals, PATH_FUNDAMENTALS)
cat("[03_Fundamentals.R] Processed fundamentals saved:",
    nrow(fundamentals), "rows\n")

#==============================================================================#
# 8. Summary diagnostics
#==============================================================================#

cat("\n[03_Fundamentals.R] ══════════════════════════════════════\n")
cat("  Rows              :", nrow(fundamentals), "\n")
cat("  Unique permno     :", n_distinct(fundamentals$permno), "\n")
cat("  Unique gvkey      :", n_distinct(fundamentals$gvkey), "\n")
cat("  Fiscal year range :", min(fundamentals$fyear),
    "to", max(fundamentals$fyear), "\n")

cat("\n  Coverage by key variable (% non-missing):\n")
key_vars_diag <- c("at", "sale", "oancf", "epspx", "ni",
                   "ebitda", "dltt", "emp", "xrent", "mkvalt")
for (v in key_vars_diag) {
  if (v %in% names(fundamentals)) {
    pct_avail <- 100 * mean(!is.na(fundamentals[[v]]))
    cat(sprintf("    %-12s : %5.1f%%\n", v, pct_avail))
  }
}

cat("\n  Unmatched Compustat rows (no CCM link, dropped):",
    sum(is.na(merged$lpermno)), "\n")
cat("  Universe permno with no Compustat data:",
    sum(!target_permnos %in% fundamentals$permno), "\n")

cat("\n[03_Fundamentals.R] DONE:", format(Sys.time()), "\n")

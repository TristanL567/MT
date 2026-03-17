#==============================================================================#
#==== 00 - Description ========================================================#
#==============================================================================#


#==============================================================================#
#==== 1 - Working Directory & Libraries =======================================#
#==============================================================================#

silent=F
.libPaths()

Directory <- this.path::this.dir() ## You need to install the package first incase you do not have it.
setwd(Directory)

#==== 1A - Libraries ==========================================================#

## Needs to enable checking for install & if not then autoinstall.

packages <- c("here", "xts", "dplyr", "tidyr",
              "RPostgres", "tidyverse", "tidyfinance", "scales",
              "RSQLite", "dbplyr", "lubridate", "data.table",
              "slider", "arrow"
)

for(i in 1:length(packages)){
  package_name <- packages[i]
  if (!requireNamespace(package_name, quietly = TRUE)) {
    install.packages(package_name, character.only = TRUE)
    cat(paste("Package '", package_name, "' was not installed. It has now been installed and loaded.\n", sep = ""))
  } else {
    cat(paste("Package '", package_name, "' is already installed and has been loaded.\n", sep = ""))
  }
  library(package_name, character.only = TRUE)
}

#==== 1B - Functions ==========================================================#

sourceFunctions <- function (functionDirectory)  {
  functionFiles <- list.files(path = functionDirectory, pattern = "*.R", 
                              full.names = T)
  ssource <- function(path) {
    try(source(path, echo = F, verbose = F, print.eval = T, 
               local = F))
  }
  sapply(functionFiles, ssource)
}


#==== 1C - Parameters =========================================================#




#==== 1D - Setup WRDS =========================================================#

# Path_Environment <- file.path(Directory, ".Renviron")
# readRenviron(Path_Environment)
# Sys.getenv("WRDS_USER")
# set_wrds_credentials()
wrds <- get_wrds_connection()

print(wrds)

#==============================================================================#
#==== 01_Universe.R ===========================================================#
#==== Stock Universe Construction & Filtering =================================#
#==============================================================================#
#
# PURPOSE:
#   Build the investable stock universe from CRSP security info.
#   Outputs a clean masterlist of permno-level identifiers with one row
#   per economic entity, ready for price and fundamental data pulls.
#
# INPUT:
#   - WRDS connection object `wrds`  (established in 00_Master.R)
#   - config.R parameters
#
# OUTPUT:
#   - Data/CRSP/Raw/universe_raw.rds      : unfiltered, as-downloaded
#   - Data/CRSP/Processed/universe.rds    : filtered, deduplicated, validated
#     Columns: permno, permco, ticker, issuernm, is_active,
#              listing_date, removal_date, lifetime_years,
#              exchange, siccd, naics,
#              securitytype, securitysubtype, sharetype, shareclass,
#              n_changes
#
# ASSERTIONS (pipeline fails loudly if violated):
#   A) No duplicate permno in output
#   B) No duplicate permco in output (one share class per economic entity)
#   C) All permno are on valid exchanges
#   D) Universe size is plausible (>= 5,000 firms)
#   E) permco present for all rows (join completeness — warning only)
#
# SCHEMA NOTE:
#   crsp_a_stock.*  — full historical CRSP data including delistings (USE THIS)
#   crsp.*          — restricted/partial view in this WRDS subscription.
#                     crsp.msenames returns an incomplete universe — do not use
#                     for msenames. Exception: crsp.stksecurityinfohist only
#                     exists in crsp.* — that table reference is correct.
#
# KEY DESIGN DECISIONS:
#   [1] FIRST valid record, not last.
#       Terminal stksecurityinfohist records for delisted stocks have
#       primaryexch outside {N, A, Q}. Using max(secinfoenddt) caused the
#       exchange filter to silently drop all delisted stocks — root cause of
#       the survivor bias in prior pipeline versions. Fix: filter to valid
#       exchange/type records first, then take min(secinfostartdt).
#
#   [2] lifetime_years filter NOT applied here.
#       Short-lived delisted stocks are disproportionately CSI candidates.
#       Removing them at universe construction directly biases the label
#       distribution. Minimum history requirement enforced in
#       06_Feature_Engineering.R where it affects only the feature window.
#
#   [3] delactiontype / delreasontype NOT included.
#       Only populated on terminal records, which are excluded by the valid
#       exchange filter in Step 3. Cannot be reliably retrieved here.
#
#==============================================================================#

source("config.R")

cat("\n[01_Universe.R] START:", format(Sys.time()), "\n")

#==============================================================================#
# 1. Connect to CRSP tables (lazy — no data pulled yet)
#
#    stksecurityinfohist : security-level SCD table — only exists in crsp.*
#    msenames            : name/permco history — must use crsp_a_stock.*
#==============================================================================#

stk_info_db <- tbl(wrds, I("crsp.stksecurityinfohist"))
msenames_db <- tbl(wrds, I("crsp_a_stock.msenames"))    # crsp.msenames is incomplete

#==============================================================================#
# 2. Compute lifetime statistics per permno (lazy — runs on DB)
#
#    listing_date : earliest secinfostartdt across all records for this permno
#    removal_date : latest   secinfoenddt   across all records for this permno
#    n_changes    : number of SCD record changes (data complexity proxy)
#==============================================================================#

lifetime_stats <- stk_info_db |>
  group_by(permno) |>
  summarise(
    listing_date = min(secinfostartdt, na.rm = TRUE),
    removal_date = max(secinfoenddt,   na.rm = TRUE),
    n_changes    = n()
  ) |>
  ungroup()

#==============================================================================#
# 3. Extract first qualifying security attributes per permno
#
#    RATIONALE: Apply exchange + security type filters INSIDE the lazy query,
#    then take the FIRST (earliest) qualifying record per permno.
#
#    WHY NOT THE LAST RECORD:
#    For delisted stocks the terminal record has primaryexch outside
#    {N, A, Q} because the stock is no longer listed anywhere. Taking
#    max(secinfoenddt) — the prior approach — caused the exchange filter in
#    Step 6 to silently exclude every delisted stock, producing a survivor-
#    biased universe of ~4,500 firms instead of the expected ~10,000+.
#
#    By filtering to valid records first, then taking min(secinfostartdt),
#    we capture the stock's attributes during its active listing period
#    regardless of post-delisting terminal state.
#==============================================================================#

final_attributes <- stk_info_db |>
  filter(
    primaryexch     %in% VALID_EXCHANGES,           # NYSE ("N"), AMEX ("A"), NASDAQ ("Q")
    securitytype    %in% c("EQTY", NA_character_),  # Equity or unclassified legacy
    securitysubtype %in% c("COM",  NA_character_),  # Common stock or unclassified legacy
    sharetype       %in% c("NS",   NA_character_)   # Normal shares or unclassified legacy
  ) |>
  group_by(permno) |>
  filter(secinfostartdt == min(secinfostartdt, na.rm = TRUE)) |>  # First valid record
  ungroup() |>
  select(
    permno,
    ticker,
    issuernm,
    primaryexch,
    securitytype,
    securitysubtype,
    sharetype,
    shareclass,
    siccd,
    naics
    ## delactiontype / delreasontype intentionally excluded — see design note [3]
  ) |>
  collect() |>
  arrange(permno) |>
  distinct(permno, .keep_all = TRUE)   # Break any remaining ties after collect()

cat("[01_Universe.R] Qualifying permno from stksecurityinfohist:",
    nrow(final_attributes), "\n")

#==============================================================================#
# 4. Join permco from msenames
#
#    RATIONALE: permco is CRSP's company-level identifier — all share classes
#    of the same economic entity share one permco. This is the correct key
#    for deduplication; issuernm is a free-text field and unreliable.
#
#    msenames has multiple rows per permno (name/status changes over time).
#    permco is stable across name changes — take the FIRST record to avoid
#    the same terminal-record problem as Step 3.
#
#    SCHEMA: crsp_a_stock.msenames required — crsp.msenames is incomplete
#    in this WRDS subscription (~4,500 records vs ~38,000 in crsp_a_stock).
#==============================================================================#

permco_map <- msenames_db |>
  group_by(permno) |>
  filter(namedt == min(namedt, na.rm = TRUE)) |>    # First record — permco is stable
  ungroup() |>
  select(permno, permco) |>
  collect() |>
  distinct(permno, .keep_all = TRUE)

cat("[01_Universe.R] permco map size:", nrow(permco_map), "\n")

#==============================================================================#
# 5. Assemble raw masterlist in R memory
#
#    inner_join on final_attributes : only permno that passed Step 3 filters
#    left_join  on permco_map       : preserve all qualifying permno even if
#                                     permco is missing (triggers warning E)
#==============================================================================#

universe_raw <- lifetime_stats |>
  collect() |>
  inner_join(final_attributes, by = "permno") |>
  left_join(permco_map,        by = "permno") |>    # Both in R memory
  mutate(
    is_active = if_else(
      removal_date >= max(removal_date, na.rm = TRUE),
      "Yes", "No"
    ),
    lifetime_years = round(
      as.numeric(removal_date - listing_date) / 365.25, 2
    )
  ) |>
  select(
    permno, permco,
    ticker, issuernm, is_active,
    listing_date, removal_date, lifetime_years,
    securitytype, securitysubtype, sharetype, shareclass,
    siccd, naics,
    exchange = primaryexch,
    n_changes
  ) |>
  arrange(permno)

## Save raw — immutable reference, never overwrite
path_raw <- file.path(Data_CRSP_Directory, "Raw", "universe_raw.rds")
saveRDS(universe_raw, PATH_UNIVERSE_RAW)
cat("[01_Universe.R] Raw universe saved:", nrow(universe_raw), "rows\n")

#==============================================================================#
# 6. Filter universe
#
#    Filters A–C are enforced upstream in the Step 3 lazy query and are
#    therefore functionally redundant here. They are retained explicitly as
#    documentation of intent and as a safety net if Step 3 is modified.
#    Filter D is applied here only (removal_date is not in stksecurityinfohist
#    at the record level — it comes from lifetime_stats aggregation).
#
#    A) securitytype    "EQTY" or NA  — excludes FUNDs, ETFs
#       securitysubtype "COM"  or NA  — common stock only
#    B) sharetype       "NS"   or NA  — normal shares only
#       Excludes ADRs ("AD"), preferred ("PS"), warrants ("WS"),
#       rights ("RT"), ETF shares ("ET").
#    C) exchange        VALID_EXCHANGES — NYSE / AMEX / NASDAQ only
#    D) removal_date > START_DATE — excludes stocks fully delisted before
#       the analysis window begins. Not applied in Step 3.
#
#    NOT APPLIED: lifetime_years >= MIN_LIFETIME_YEARS
#    See design note [2] in header. Applied in 06_Feature_Engineering.R.
#==============================================================================#

universe_filtered <- universe_raw |>
  filter(
    ## A) Security type (redundant with Step 3 — safety net)
    securitytype    %in% c("EQTY", NA),
    securitysubtype %in% c("COM",  NA),
    
    ## B) Share type (redundant with Step 3 — safety net)
    sharetype %in% c("NS", NA),
    
    ## C) Exchange (redundant with Step 3 — safety net)
    exchange %in% VALID_EXCHANGES,
    
    ## D) Active within analysis window — NOT redundant, only checked here
    removal_date > START_DATE
  )

cat("[01_Universe.R] After filters:", nrow(universe_filtered), "rows\n")

#==============================================================================#
# 7. Deduplicate to one row per economic entity (permco)
#
#    RATIONALE: Firms with dual/triple share classes (e.g., Alphabet A/B/C,
#    Berkshire A/B) would otherwise appear multiple times in the panel,
#    inflating implosion counts and introducing correlated observations.
#
#    Strategy: within each permco, retain the share class with the longest
#    history (most data for feature engineering). Ties broken by lowest
#    permno (earlier listing — typically the primary share class).
#==============================================================================#

universe <- universe_filtered |>
  arrange(desc(lifetime_years), permno) |>   # arrange BEFORE group_by
  group_by(permco) |>
  slice(1) |>
  ungroup()

cat("[01_Universe.R] After permco deduplication:", nrow(universe), "rows\n")

#==============================================================================#
# 8. Output validation — fail loud if anything looks wrong
#==============================================================================#

## A) No duplicate permno
n_dup_permno <- sum(duplicated(universe$permno))
if (n_dup_permno > 0) {
  stop(sprintf(
    "[01_Universe.R] ASSERTION FAILED: %d duplicate permno in universe.",
    n_dup_permno
  ))
}

## B) No duplicate permco (exclude NA permco from this check)
n_dup_permco <- sum(duplicated(na.omit(universe$permco)))
if (n_dup_permco > 0) {
  stop(sprintf(
    "[01_Universe.R] ASSERTION FAILED: %d duplicate permco in universe.",
    n_dup_permco
  ))
}

## C) Only valid exchanges
invalid_exch <- universe |> filter(!exchange %in% VALID_EXCHANGES)
if (nrow(invalid_exch) > 0) {
  stop(sprintf(
    "[01_Universe.R] ASSERTION FAILED: %d firms on invalid exchanges: %s",
    nrow(invalid_exch),
    paste(unique(invalid_exch$exchange), collapse = ", ")
  ))
}

## D) Plausible universe size — expect 8,000–14,000 before lifetime filter
if (nrow(universe) < 5000) {
  stop(sprintf(
    "[01_Universe.R] ASSERTION FAILED: Universe has only %d firms (expected >= 5,000). Check filters.",
    nrow(universe)
  ))
}

## E) permco present for all rows (left join completeness — warning only)
n_missing_permco <- sum(is.na(universe$permco))
if (n_missing_permco > 0) {
  warning(sprintf(
    "[01_Universe.R] WARNING: %d firms have no permco — msenames join incomplete. Investigate.",
    n_missing_permco
  ))
}

cat("[01_Universe.R] All assertions passed.\n")

#==============================================================================#
# 9. Save processed universe and print diagnostics
#==============================================================================#

path_processed <- file.path(Data_CRSP_Directory, "Processed", "universe.rds")
saveRDS(universe, PATH_UNIVERSE)

cat("[01_Universe.R] Processed universe saved:", nrow(universe), "firms\n")

cat("[01_Universe.R] Exchange breakdown:\n")
print(table(universe$exchange, useNA = "ifany"))

cat("[01_Universe.R] Active vs delisted:\n")
print(table(universe$is_active, useNA = "ifany"))

cat("[01_Universe.R] Lifetime distribution (years):\n")
print(summary(universe$lifetime_years))

cat("[01_Universe.R] DONE:", format(Sys.time()), "\n")

#==============================================================================#
#==============================================================================#
#==============================================================================#
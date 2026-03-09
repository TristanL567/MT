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
              "slider"
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

## Directories.
Data_Directory <- file.path(Directory, "02_Data")
Data_CRSP_Directory <- file.path(Data_Directory, "CRSP")

Functions_Directory <- file.path(Directory, "01_Code/Data/Subfunctions")

## Load all code files in the functions directory.
sourceFunctions(Functions_Directory)

## Date range.
start_date <- ymd("1960-01-01")
end_date <- ymd("2024-12-31")

## Saved objects.

# Data_Stock_Masterlist.rds - Masterlist of all firms after filtering.
# Data_Monthly - Contains all data about the firms in the masterlist.
# Data_Failed_Companies - Contains the tibble with the failed company details.
# Data_y - Final tibble of firms with the target and if they are in a zombie state (target is then NA).

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
#   - Data/CRSP/Raw/universe_raw.rds         : unfiltered, as-downloaded
#   - Data/CRSP/Processed/universe.rds       : filtered, deduplicated, validated
#     Columns: permno, permco, ticker, issuernm, is_active,
#              listing_date, removal_date, lifetime_years,
#              exchange, siccd, naics,
#              securitytype, securitysubtype, sharetype, shareclass,
#              delactiontype, delreasontype, n_changes
#
# ASSERTIONS (pipeline fails loudly if violated):
#   - No duplicate permno in output
#   - No duplicate permco in output (one share class per economic entity)
#   - All permno have lifetime_years >= MIN_LIFETIME_YEARS
#   - All permno are on valid exchanges
#   - Universe size is plausible (>= 3,000 firms)
#
#==============================================================================#

source("config.R")

cat("\n[01_Universe.R] START:", format(Sys.time()), "\n")

#==============================================================================#
# 1. Connect to CRSP tables (lazy — no data pulled yet)
#==============================================================================#

stk_info_db  <- tbl(wrds, I("crsp.stksecurityinfohist"))
msenames_db  <- tbl(wrds, I("crsp.msenames"))   # Needed for permco

#==============================================================================#
# 2. Compute lifetime statistics per permno
#    Still lazy — executed on the DB server
#==============================================================================#

lifetime_stats <- stk_info_db |>
  group_by(permno) |>
  summarise(
    listing_date = min(secinfostartdt, na.rm = TRUE),
    removal_date = max(secinfoenddt,   na.rm = TRUE),
    n_changes    = n()   # Number of security info record changes (complexity proxy)
  ) |>
  ungroup()

#==============================================================================#
# 3. Extract most-recent security attributes per permno
#
#    RATIONALE: stksecurityinfohist is a slowly-changing dimension table.
#    We take the latest record to capture the terminal state of each security
#    (e.g., final exchange, final security type before delisting).
#==============================================================================#

final_attributes <- stk_info_db |>
  group_by(permno) |>
  filter(secinfoenddt == max(secinfoenddt, na.rm = TRUE)) |>
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
    naics,
    delactiontype,
    delreasontype
  ) |>
  collect() |>           # Pull to R first
  arrange(permno) |>     # Deterministic order before dedup
  distinct(permno, .keep_all = TRUE)  # Then deduplicate in R

#==============================================================================#
# 4. Join permco from msenames
#
#    RATIONALE: permco is CRSP's company-level identifier — all share classes
#    of the same economic entity share one permco. This is the correct key for
#    deduplication, not issuernm (which is a free-text field and unreliable).
#
#    msenames can have multiple rows per permno (name changes over time).
#    We take the latest record to get the most recent permco assignment.
#==============================================================================#

permco_map <- msenames_db |>
  group_by(permno) |>
  filter(nameendt == max(nameendt, na.rm = TRUE)) |>
  ungroup() |>
  select(permno, permco) |>
  collect() |>
  distinct(permno, .keep_all = TRUE)

#==============================================================================#
# 5. Assemble raw masterlist and pull to R memory
#==============================================================================#

universe_raw <- lifetime_stats |>
  collect() |>                                     # Pull lifetime_stats to R first
  inner_join(final_attributes, by = "permno") |>   # Both now in R memory
  left_join(permco_map,        by = "permno") |>   # Both now in R memory
  mutate(
    is_active      = if_else(removal_date >= max(removal_date, na.rm = TRUE),
                             "Yes", "No"),
    lifetime_years = round(as.numeric(removal_date - listing_date) / 365.25, 2)
  ) |>
  select(
    permno, permco,
    ticker, issuernm, is_active,
    listing_date, removal_date, lifetime_years,
    delactiontype, delreasontype,
    securitytype, securitysubtype, sharetype, shareclass,
    siccd, naics,
    exchange = primaryexch,
    n_changes
  ) |>
  arrange(permno)

## Save raw (immutable reference — never overwrite this)
path_raw <- file.path(Data_CRSP_Directory, "Raw", "universe_raw.rds")
saveRDS(universe_raw, file = path_raw)
cat("[01_Universe.R] Raw universe saved:", nrow(universe_raw), "rows\n")

#==============================================================================#
# 6. Filter universe
#
#    Filters applied and rationale:
#
#    A) securitytype: exclude "FUND" (mutual funds, ETFs)
#       securitysubtype: whitelist only "COM" (common stock) and NA
#       — subtype is often NA for older records; excluding NA would drop
#         a large fraction of legitimate equities from the 1960s–1990s.
#
#    B) sharetype == "NS": keep only "Normal Shares"
#       — excludes ADRs ("AD"), preferred shares ("PS"), warrants ("WS"),
#         rights ("RT"), and ETF shares ("ET") that pass the securitytype filter.
#
#    C) exchange: NYSE ("N"), AMEX ("A"), NASDAQ ("Q") only
#       — excludes OTC pink sheets and other non-standard venues.
#
#    D) removal_date > START_DATE: exclude securities that were fully
#       delisted before our analysis period begins (dead weight).
#
#    E) lifetime_years >= MIN_LIFETIME_YEARS: require sufficient history
#       for the 5-year rolling feature window (paper: Approach 2).
#==============================================================================#

universe_filtered <- universe_raw |>
  filter(
    ## A) Security type
    !securitytype %in% c("FUND"),
    securitysubtype %in% c("COM", NA),          # Common stock or unclassified legacy
    
    ## B) Share type — normal shares only
    sharetype %in% c("NS", NA),                  # NA retained for pre-classification era
    
    ## C) Exchange
    exchange %in% VALID_EXCHANGES,
    
    ## D) Active within analysis window
    removal_date > START_DATE,
    
    ## E) Minimum lifetime for rolling feature window
    lifetime_years >= MIN_LIFETIME_YEARS
  )

cat("[01_Universe.R] After filters:", nrow(universe_filtered), "rows\n")

#==============================================================================#
# 7. Deduplicate to one row per economic entity (permco)
#
#    RATIONALE: Firms with dual/triple share classes (e.g., Alphabet Class A/B/C,
#    Berkshire A/B) would otherwise appear multiple times in the panel, inflating
#    implosion counts and introducing correlated observations.
#
#    Strategy: within each permco, retain the share class with the longest
#    history (most data for feature engineering). Ties broken by lowest permno
#    (earlier listing — typically the primary share class).
#
#    NOTE: If your research question explicitly requires multi-class analysis
#    (e.g., voting vs. economic rights), comment out this block and handle
#    clustering in the model instead.
#==============================================================================#

universe <- universe_filtered |>
  group_by(permco) |>
  arrange(desc(lifetime_years), permno) |>
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
    "[01_Universe.R] ASSERTION FAILED: %d duplicate permno found in universe.",
    n_dup_permno
  ))
}

## B) No duplicate permco
n_dup_permco <- sum(duplicated(universe$permco))
if (n_dup_permco > 0) {
  stop(sprintf(
    "[01_Universe.R] ASSERTION FAILED: %d duplicate permco found in universe.",
    n_dup_permco
  ))
}

## C) Lifetime years floor respected
n_short <- sum(universe$lifetime_years < MIN_LIFETIME_YEARS, na.rm = TRUE)
if (n_short > 0) {
  stop(sprintf(
    "[01_Universe.R] ASSERTION FAILED: %d firms below MIN_LIFETIME_YEARS threshold.",
    n_short
  ))
}

## D) Only valid exchanges
invalid_exch <- universe |> filter(!exchange %in% VALID_EXCHANGES)
if (nrow(invalid_exch) > 0) {
  stop(sprintf(
    "[01_Universe.R] ASSERTION FAILED: %d firms on invalid exchanges: %s",
    nrow(invalid_exch),
    paste(unique(invalid_exch$exchange), collapse = ", ")
  ))
}

## E) Plausible universe size
if (nrow(universe) < 3000) {
  stop(sprintf(
    "[01_Universe.R] ASSERTION FAILED: Universe has only %d firms — suspiciously small. Check filters.",
    nrow(universe)
  ))
}

## F) permco present for all rows (join succeeded)
n_missing_permco <- sum(is.na(universe$permco))
if (n_missing_permco > 0) {
  warning(sprintf(
    "[01_Universe.R] WARNING: %d firms have no permco — msenames join incomplete. Investigate.",
    n_missing_permco
  ))
}

cat("[01_Universe.R] All assertions passed.\n")

#==============================================================================#
# 9. Save processed universe
#==============================================================================#

path_processed <- file.path(Data_CRSP_Directory, "Processed", "universe.rds")
saveRDS(universe, file = path_processed)

cat("[01_Universe.R] Processed universe saved:", nrow(universe), "firms\n")
cat("[01_Universe.R] Exchange breakdown:\n")
print(table(universe$exchange))
cat("[01_Universe.R] Active vs delisted:\n")
print(table(universe$is_active))
cat("[01_Universe.R] DONE:", format(Sys.time()), "\n")

#==============================================================================#
#==============================================================================#
#==============================================================================#

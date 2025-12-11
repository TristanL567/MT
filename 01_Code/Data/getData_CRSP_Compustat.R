#==============================================================================#
#==== 00 - Description ========================================================#
#==============================================================================#


#==============================================================================#
#==== 1 - Working Directory & Libraries =======================================#
#==============================================================================#

silent=F
.libPaths()

Path <- file.path(here::here("")) ## You need to install the package first incase you do not have it.

#==== 1A - Libraries ==========================================================#

## Needs to enable checking for install & if not then autoinstall.

packages <- c("here", "xts", "dplyr", "tidyr",
              "RPostgres", "tidyverse", "tidyfinance", "scales",
              "RSQLite", "dbplyr"
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
Data_Directory <- file.path(Path, "02_Data")
Data_CRSP_Directory <- file.path(Data_Directory, "CRSP")

Functions_Directory <- file.path(Path, "01_Code/Data/Subfunctions")

## Load all code files in the functions directory.
sourceFunctions(Functions_Directory)

## Date range.
start_date <- ymd("2023-01-01")
end_date <- ymd("2024-12-31")

#==== 1D - Setup WRDS =========================================================#

Path_Environment <- file.path(Path, ".Renviron")
readRenviron(Path_Environment)
Sys.getenv("WRDS_USER")
# set_wrds_credentials()
wrds <- get_wrds_connection()

print(wrds)

##=========================================================================#
## 2. Setup & Universe Selection (Constituents)
##=========================================================================#

# Connect to WRDS (Assuming 'wrds' connection object exists)
msf_db <- tbl(wrds, I("crsp.msf_v2"))
stk_info_db <- tbl(wrds, I("crsp.stksecurityinfohist"))

# 1. Define the end of the dataset to determine "Active" status
# We find the maximum date across the entire database.
max_db_date <- stk_info_db |>
  summarise(max_dt = max(secinfoenddt, na.rm = TRUE)) |>
  pull(max_dt)

# 2. Aggregation: Get Lifetime Stats (Start and End Dates)
# We group by 'permno' to find when the stock was first born and when it died.
lifetime_stats <- stk_info_db |>
  group_by(permno) |>
  summarise(
    listing_date = min(secinfostartdt, na.rm = TRUE),
    removal_date = max(secinfoenddt, na.rm = TRUE),
    n_changes = n() # Proxy for corporate complexity
  ) |>
  ungroup()

# 3. Selection: Get the "Final State" Attributes
# We grab the attributes from the row corresponding to the stock's *last day*.
final_attributes <- stk_info_db |>
  # Filter to keep only the row where the end date matches the stock's max end date
  group_by(permno) |>
  filter(secinfoenddt == max(secinfoenddt, na.rm = TRUE)) |>
  ungroup() |>
  select(
    permno,
    ticker, 
    issuernm, 
    primaryexch,
    # Classification columns you requested
    securitytype, 
    securitysubtype, 
    sharetype, 
    shareclass,
    # Industry Codes
    siccd, 
    naics,
    # The "Implosion" Labels (Crucial for your research)
    delactiontype, 
    delreasontype
  )

# 4. Merge and Finalize
stock_master_list <- lifetime_stats |>
  inner_join(final_attributes, by = "permno") |>
  collect() |> # Execute SQL and pull to R
  mutate(
    # Determine Active Status
    is_active = if_else(removal_date >= max_db_date, "Yes", "No"),
    
    # Calculate Duration
    lifetime_years = round(as.numeric(removal_date - listing_date) / 365.25, 2)
  ) |>
  # Reorder columns for readability
  select(
    permno, ticker, issuernm, is_active, 
    listing_date, removal_date, lifetime_years,
    delactiontype, delreasontype, # <--- Your target variables
    securitytype, securitysubtype, sharetype, shareclass,
    siccd, naics, primaryexch, n_changes
  ) |>
  arrange(permno)

# 5. Preview the Master List
glimpse(stock_master_list)

## Dimensions.

dim(stock_master_list)
unique(stock_master_list$securitytype)
which(duplicated(stock_master_list_filtered$issuernm))

## 6A. Filter for EQUITY (and NA) only.
stock_master_list_filtered <- stock_master_list |>
  filter(
    !securitytype %in% c("FUND"),
    removal_date > "1966-01-01",
    lifetime_years > 4.99,
    primaryexch %in% c("N", "A", "Q") # NYSE, AMEX, NASDAQ
  )

## 6B. filter for any remaining ETFs or mutual funds.
Filter_Output <- filterFunds(Input = stock_master_list_filtered)
stock_master_list_filtered <- Filter_Output[[""]]

##=========================================================================#
## 3. Filter Universe.
##=========================================================================#

# Filter for the "Investable Universe"
# We do this first to avoid pulling data for dead penny stocks we won't trade.
universe_permno <- stock_master_list_filtered |>
  select(permno)

##=========================================================================#
## 4. Stock Prices & Returns (Monthly)
##=========================================================================#

crsp_monthly <- msf_db |>
  filter(mthcaldt >= start_date & mthcaldt <= end_date) |>
  select(permno, mthcaldt, mthret, mthprc, shrout) |>
  inner_join(universe_permno, by = "permno") |>
  filter(mthcaldt >= secinfostartdt & mthcaldt <= secinfoenddt) |>
  collect() |>
  mutate(
    date = ymd(mthcaldt),
    # Handle negative prices (CRSP convention for bid/ask average)
    prc = abs(mthprc),
    # 2a. Compute Log Returns
    # Simple Return: P_t / P_{t-1} - 1
    # Log Return: ln(P_t / P_{t-1}) = ln(1 + Simple Return)
    log_ret = log(1 + mthret),
    mkt_cap = prc * shrout * 1000 # Calculate Market Cap
  ) |>
  select(permno, date, ret = mthret, log_ret, prc, mkt_cap) |>
  arrange(permno, date)

##=========================================================================#
## 5. Balance Sheet with EXACT Timestamps (Quarterly)
##=========================================================================#
# Note: specific timestamps (RDQ) are found in Quarterly (fundq), not Annual (funda).
# This is crucial for distress prediction to capture the "news" moment.

comp_q_db <- tbl(wrds, I("comp.fundq"))
ccm_link_db <- tbl(wrds, I("crsp.ccmxpf_lnkhist")) # The Link Table

# 5a. Filter Link Table
valid_links <- ccm_link_db |>
  filter(linktype %in% c("LC", "LU", "LS"), linkprim %in% c("P", "C")) |>
  select(gvkey, permno = lpermno, linkdt, linkenddt)

# 5b. Fetch Fundamentals
fund_data <- comp_q_db |>
  filter(
    indfmt == "INDL", datafmt == "STD", consol == "C",
    datadate >= start_date # Broad filter first
  ) |>
  select(
    gvkey, datadate, fyearq, fqtr,
    rdq,            # << THE KEY: Report Date of Quarterly Earnings
    atq, ltq,       # Assets, Liabilities (Quarterly)
    cheq,           # Cash & Equivalents
    niq,            # Net Income (Quarterly)
    saleq           # Sales (Quarterly)
  ) |>
  inner_join(valid_links, by = "gvkey") |>
  filter(
    datadate >= linkdt & (is.na(linkenddt) | datadate <= linkenddt)
  ) |>
  collect() |>
  mutate(
    datadate = ymd(datadate),
    rdq = ymd(rdq),
    # If RDQ is missing, we must impute a "Safe Lag" (standard is +3 or +4 months for quarters)
    public_date = case_when(
      !is.na(rdq) ~ rdq,
      TRUE ~ datadate + months(3) # Fallback if no report date exists
    )
  ) |>
  # Filter: Ensure we only keep data that was actually released
  filter(!is.na(public_date)) |>
  select(permno, gvkey, datadate, public_date, atq, ltq, cheq, niq)

# View the result: exact "public_date" tells you when the algorithm can "see" this data.
print(head(fund_data))